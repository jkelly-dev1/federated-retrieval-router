"""The pipeline the README opens with: route, query only the chosen legs, fuse.

The other modules here measure a routing decision. This one executes it:
routing.py chooses backends, fusion.py merges ranked lists, and this is the
callable that searches only the chosen legs and fuses what they return. It is
what the demo runs, so the backends-per-query figure in the headline table is
the cost of an execution and not only an arithmetic property of a set of
decisions.

What `federated_search` adds over `reciprocal_rank_fusion` is the part that
can be wrong: which legs get consulted, and what happens when one of them does
not come back.

The cost number is a measurement. `backends_consulted` counts legs this run
actually searched. It is derived by executing the plan, not by summing it, so
an edit that quietly widens a router's choice shows up here as more work done
as well as a different set on paper.

Partial failure is named, never fused away. `reciprocal_rank_fusion` takes a
mapping and cannot tell a leg that returned nothing from a leg that was never
consulted, or from one that raised on the way: all three arrive as an absent
key, and the merged list looks complete in every case. That is the classic
federated-search silent failure, an answer short by one store's worth of
results with nothing in the output saying so. A leg that raises is therefore
caught, recorded in `failures` with its exception, and `complete` goes False.
The ranked list is still returned, because a partial answer is usually what a
caller wants; what it must not do is arrive looking whole.

The default is sequential. The legs are independent once routing has chosen
them, so they may run in any order or at the same time, and `executor` is the
seam for it: hand in anything with `concurrent.futures`' `submit` and the
chosen legs run together. It defaults to None because the four offline
backends in backends.py are pure-Python and GIL-bound, so threads would buy
nothing a clock could see, and design note 6 of the README says this
repository reports no latency or cost figures; a speedup claim that was never
measured has no place here. The executor is worth using with the adapters in
adapters.py, whose legs are network round trips to pgvector, Elasticsearch and
DuckDB. The seam is exposed and the choice is left to the caller, as `weights`
is in fusion.py for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from router.backends import DEFAULT_K, Federation
from router.fusion import DEFAULT_WINDOW, reciprocal_rank_fusion
from router.models import Backend, FusedHit, RankedHit, RoutingDecision


class _Router(Protocol):
    name: str

    def route(self, query: str, query_id: str = "") -> RoutingDecision: ...


@dataclass(frozen=True)
class LegFailure:
    """One backend that was chosen, consulted, and did not come back.

    `error` keeps the exception itself, not a formatted string, so a
    caller can re-raise it or match on its type. A merge that swallowed this
    into a log line would leave the fused list looking complete, which is the
    failure this class exists to make impossible.
    """

    backend: Backend
    error: BaseException

    def __str__(self) -> str:
        return f"{self.backend.value}: {type(self.error).__name__}: {self.error}"


@dataclass(frozen=True)
class FederatedResult:
    """One routed query: what was decided, what it cost, and what came back."""

    query_id: str
    query: str
    decision: RoutingDecision
    consulted: tuple[Backend, ...]
    per_backend: Mapping[Backend, Sequence[RankedHit]]
    hits: tuple[FusedHit, ...]
    failures: tuple[LegFailure, ...] = field(default_factory=tuple)

    @property
    def backends_consulted(self) -> int:
        """The measured cost of this run: how many legs were actually searched.

        Includes legs that raised, because a leg that was asked and failed
        still cost whatever it cost to ask it.
        """
        return len(self.consulted)

    @property
    def skipped(self) -> frozenset[Backend]:
        """Legs the router declined to consult. The saving, named."""
        return frozenset(Backend) - frozenset(self.consulted)

    @property
    def complete(self) -> bool:
        """False when any consulted leg failed, so a short answer says so."""
        return not self.failures


def federated_search(
    fed: Federation,
    router: _Router,
    query: str,
    query_id: str = "",
    k: int = 10,
    per_leg_k: int = DEFAULT_K,
    window: int = DEFAULT_WINDOW,
    executor: Optional[Any] = None,
) -> FederatedResult:
    """Route the query, search only the chosen legs, and fuse what returns.

    `per_leg_k` is how deep each backend is asked to go and `k` is how long the
    merged list is. They are separate because fusion can only rank what it was
    shown: asking each leg for five and fusing to ten returns at most the union
    of four five-item lists, and a document ranked sixth by the one leg that
    found it is not in the answer no matter how deep the window goes. Keeping
    them one number is a quiet way to lose results.

    `executor` runs the chosen legs concurrently when supplied; the module
    docstring says why that is not the default.
    """
    decision = router.route(query, query_id)
    # Sorted so two runs consult the legs in the same order and the output is
    # reproducible. `chosen` is a frozenset, whose iteration order is not
    # stable across processes, and an unstable order here would make the demo
    # and the gate flap for reasons unrelated to any routing decision.
    consulted = tuple(sorted(decision.chosen, key=lambda b: b.value))

    per_backend: dict[Backend, Sequence[RankedHit]] = {}
    failures: list[LegFailure] = []

    def _search(backend: Backend) -> list[RankedHit]:
        return fed.get(backend).search(query, k=per_leg_k)

    if executor is None:
        for backend in consulted:
            try:
                per_backend[backend] = _search(backend)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failures.append(LegFailure(backend, exc))
    else:
        futures = {b: executor.submit(_search, b) for b in consulted}
        for backend in consulted:
            try:
                per_backend[backend] = futures[backend].result()
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                failures.append(LegFailure(backend, exc))

    fused = reciprocal_rank_fusion(per_backend, k=k, window=window)
    return FederatedResult(
        query_id=query_id,
        query=query,
        decision=decision,
        consulted=consulted,
        per_backend=per_backend,
        hits=tuple(fused),
        failures=tuple(failures),
    )


def measured_fan_out(results: Sequence[FederatedResult]) -> float:
    """Mean backends per query, counted from runs that happened.

    The same figure `RoutingReport.fan_out` reports from decisions alone. Both
    exist because they are derived by different routes from the same
    routing, so where they disagree, something between deciding and executing
    is dropping or adding a leg. The gate asserts they agree.
    """
    if not results:
        return 0.0
    return sum(r.backends_consulted for r in results) / len(results)
