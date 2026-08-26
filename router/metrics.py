"""Scoring the routing decision, which is the thing this project measures.

The trap this file exists to avoid. It is easy to evaluate a federated router
by end-to-end retrieval quality and call it done. That number moves for reasons
that have nothing to do with routing, a better embedder, a luckier corpus, a
different k, and it cannot distinguish a router that chose correctly from one
that fanned out to everything. So routing is scored on its own axis, per
backend, and cost is reported beside correctness rather than folded into it.

Four numbers, and each one answers a different objection.

  CORRECTNESS   did the router choose every backend the query REQUIRED?
                A fan-out router scores 1.0 here by construction. That is not
                a flaw in the metric; it is the reason fan-out is reported
                alongside as a baseline rather than omitted as trivial.

  FAN-OUT       how many backends were queried per question, on average?
                The cost axis. Correctness without it is free to be perfect.

  PRECISION     of the backends chosen, how many were needed?
    (per        Reported per backend because the failures are asymmetric: a
     backend)   router that over-fires the graph leg is a different problem
                from one that over-fires the relational leg, and an average
                hides which.

  RECALL        of the backends required, how many were chosen?
    (per        The one that matters for correctness. A missed REQUIRED
     backend)   backend means the question cannot be answered at all.

And separately, trap accuracy. The labeled set contains queries whose surface
form points at the wrong backend. They are a minority and they are where a
keyword router fails, so folding them into the headline average would let a
good score on the easy majority hide a total failure on the traps.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from router.models import Backend, LabeledQuery, RoutingDecision


@dataclass(frozen=True)
class BackendScore:
    backend: Backend
    chosen: int = 0
    required: int = 0
    correctly_chosen: int = 0

    @property
    def precision(self) -> float:
        """Of the times it was chosen, how often was it needed."""
        return self.correctly_chosen / self.chosen if self.chosen else 1.0

    @property
    def recall(self) -> float:
        """Of the times it was needed, how often was it chosen."""
        return self.correctly_chosen / self.required if self.required else 1.0

    @property
    def over_fires(self) -> int:
        return self.chosen - self.correctly_chosen

    @property
    def misses(self) -> int:
        return self.required - self.correctly_chosen


@dataclass(frozen=True)
class RoutingReport:
    router: str
    total: int
    correct: int
    fan_out: float
    per_backend: dict[Backend, BackendScore]
    trap_total: int = 0
    trap_correct: int = 0
    failures: tuple[tuple[str, frozenset[Backend], frozenset[Backend]], ...] = ()

    @property
    def correctness(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def trap_accuracy(self) -> float:
        return self.trap_correct / self.trap_total if self.trap_total else 1.0

    @property
    def plain_total(self) -> int:
        return self.total - self.trap_total

    @property
    def plain_accuracy(self) -> float:
        plain = self.plain_total
        return (self.correct - self.trap_correct) / plain if plain else 0.0


def score_routing(
    router_name: str,
    queries: Sequence[LabeledQuery],
    decisions: Sequence[RoutingDecision],
) -> RoutingReport:
    """Score a router's decisions against the labeled ground truth."""
    if len(queries) != len(decisions):
        raise ValueError(
            f"{len(queries)} queries but {len(decisions)} decisions; scoring a "
            f"router against a different query set is silently meaningless"
        )

    chosen_n: dict[Backend, int] = defaultdict(int)
    required_n: dict[Backend, int] = defaultdict(int)
    hit_n: dict[Backend, int] = defaultdict(int)
    correct = trap_total = trap_correct = 0
    total_fan_out = 0
    failures: list[tuple[str, frozenset[Backend], frozenset[Backend]]] = []

    for labeled, decision in zip(queries, decisions):
        if labeled.query_id != decision.query_id:
            raise ValueError(
                f"decision {decision.query_id!r} does not match query "
                f"{labeled.query_id!r}; the lists are misaligned"
            )
        total_fan_out += len(decision.chosen)
        ok = decision.is_correct(labeled)
        correct += int(ok)
        if labeled.trap:
            trap_total += 1
            trap_correct += int(ok)
        if not ok:
            failures.append((labeled.query_id, labeled.required, decision.chosen))

        for backend in decision.chosen:
            chosen_n[backend] += 1
            # "Needed" means required, or sufficient-and-actually-useful. A
            # backend in `sufficient` that was chosen is not an over-fire: it
            # would have answered.
            if backend in labeled.acceptable:
                hit_n[backend] += 1
        for backend in labeled.required:
            required_n[backend] += 1

    per_backend = {
        b: BackendScore(
            backend=b,
            chosen=chosen_n.get(b, 0),
            required=required_n.get(b, 0),
            correctly_chosen=min(hit_n.get(b, 0), chosen_n.get(b, 0)),
        )
        for b in Backend
    }

    return RoutingReport(
        router=router_name,
        total=len(queries),
        correct=correct,
        fan_out=total_fan_out / len(queries) if queries else 0.0,
        per_backend=per_backend,
        trap_total=trap_total,
        trap_correct=trap_correct,
        failures=tuple(failures),
    )


# ------------------------------------------------------- retrieval quality


@dataclass(frozen=True)
class RetrievalScore:
    """Document-level quality, for the queries where documents are the answer.

    Not defined for aggregate queries, and the field says so rather than
    reporting a misleading zero. "How many incidents did payments have" is
    answered by a computed number that appears in no document, so
    document-recall for that query is not a low score; it is a category
    error. `applicable` is the count this was averaged over.
    """

    applicable: int
    recall_at_k: float
    queries_with_no_hit: tuple[str, ...] = ()


def score_retrieval(
    queries: Sequence[LabeledQuery],
    retrieved: Mapping[str, Sequence[str]],
    k: int = 5,
) -> RetrievalScore:
    """Recall@k over the queries that have relevant documents at all."""
    applicable = [q for q in queries if q.relevant_docs]
    if not applicable:
        return RetrievalScore(applicable=0, recall_at_k=0.0)
    total = 0.0
    misses: list[str] = []
    for q in applicable:
        got = set(retrieved.get(q.query_id, ())[:k])
        found = len(got & set(q.relevant_docs))
        total += found / len(q.relevant_docs)
        if found == 0:
            misses.append(q.query_id)
    return RetrievalScore(
        applicable=len(applicable),
        recall_at_k=total / len(applicable),
        queries_with_no_hit=tuple(misses),
    )


@dataclass(frozen=True)
class CompetenceCheck:
    """Does the backend a question SHAPE calls for actually win on it?

    The honest bridge between the designed labels and what the stack does. A
    row where `designed` is not in `measured` is not a labeling error: it is a
    backend failing at its own competence, and reporting it as that beats
    quietly relabeling until the two agree.
    """

    query_id: str
    designed: frozenset[Backend]
    measured: frozenset[Backend]
    k: int

    @property
    def designed_wins(self) -> bool:
        return bool(self.designed & self.measured)

    @property
    def nobody_wins(self) -> bool:
        return not self.measured


def validate_competences(
    queries: Sequence[LabeledQuery],
    retrieved_by_backend: Mapping[str, Mapping[Backend, Sequence[str]]],
    k: int = 3,
) -> list[CompetenceCheck]:
    """Compare the designed ground truth against which backend actually wins.

    Only defined for queries with relevant documents; aggregate and pure
    relationship questions have no document answer and are excluded rather than
    scored zero. See RetrievalScore for why that distinction decides the score.
    """
    out: list[CompetenceCheck] = []
    for q in queries:
        if not q.relevant_docs:
            continue
        measured = {
            backend
            for backend, docs in retrieved_by_backend.get(q.query_id, {}).items()
            if set(q.relevant_docs) <= set(list(docs)[:k])
        }
        out.append(
            CompetenceCheck(
                query_id=q.query_id,
                designed=q.required,
                measured=frozenset(measured),
                k=k,
            )
        )
    return out


def weighted_correctness(
    queries: Sequence[LabeledQuery],
    decisions: Sequence[RoutingDecision],
    mix: Mapping[object, float],
) -> float:
    """Correctness re-weighted to a realistic query mix.

    The balanced set overstates the value of federation and this is the
    correction. An evaluation set needs roughly equal numbers of each
    competence to say anything per backend; a production log is dominated by
    semantic lookups. A vector-only baseline scored on the balanced set looks
    far worse than it would behave, and publishing that number alone would be
    selling the project's own thesis.

    Each query's weight is its competences' mix weight, split when a query
    spans several, then renormalized. Queries whose competences carry no
    weight contribute nothing rather than silently defaulting to one.
    """
    total_weight = 0.0
    correct_weight = 0.0
    for labeled, decision in zip(queries, decisions):
        if not labeled.competences:
            continue
        weight = sum(mix.get(c, 0.0) for c in labeled.competences) / len(
            labeled.competences
        )
        total_weight += weight
        if decision.is_correct(labeled):
            correct_weight += weight
    return correct_weight / total_weight if total_weight else 0.0


def confusion(
    queries: Sequence[LabeledQuery], decisions: Sequence[RoutingDecision]
) -> dict[tuple[Backend, Backend], int]:
    """required -> chosen counts, for the routing confusion matrix.

    Read the off-diagonal. A cell at (GRAPH, VECTOR) counts questions that
    needed a traversal and were sent to the embedding store, which is the
    misroute that produces a confident, fluent, wrong answer rather than an
    empty result: the expensive kind.
    """
    matrix: dict[tuple[Backend, Backend], int] = defaultdict(int)
    for labeled, decision in zip(queries, decisions):
        for req in labeled.required:
            for got in decision.chosen:
                matrix[(req, got)] += 1
    return dict(matrix)
