"""The routed execution path: does the plan the router made actually run?

Every test here is about the seam between deciding and executing. A router
and a fusion step can each work well while nothing connects them, and fusing
every leg on every query produces a perfectly reasonable merged list while
silently making the cost figure fiction.

The assertions are therefore mostly negative. That only the chosen legs
returned results is not evidence: an unchosen leg that was searched and whose
hits lost the merge leaves no trace in the output. What has to be asserted is
that the unchosen legs were never asked, which needs a federation that records
its calls.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from router.backends import build_federation
from router.corpus import build_corpus
from router.models import Backend, RoutingDecision
from router.routing import FanOutRouter, HeuristicRouter, VectorOnlyRouter
from router.search import federated_search, measured_fan_out


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def fed(corpus):
    return build_federation(corpus)


class _RecordingFederation:
    """Wraps a real Federation and remembers which legs were asked.

    Delegates `get` so the backends behave exactly as they normally do; the
    only added behavior is the log. `all()` raises instead of delegating,
    because federated_search calling it would mean the router's decision was
    bypassed, and that must be a loud failure, not a subtly wrong cost
    number.
    """

    def __init__(self, inner):
        self.inner = inner
        self.asked: list[Backend] = []

    def get(self, backend: Backend):
        self.asked.append(backend)
        return self.inner.get(backend)

    def all(self):
        raise AssertionError(
            "federated_search consulted every leg instead of the routed ones"
        )


class _StubRouter:
    """A router that returns a fixed set, so a test can name the legs exactly."""

    def __init__(self, chosen, name="stub"):
        self.name = name
        self._chosen = frozenset(chosen)

    def route(self, query: str, query_id: str = "") -> RoutingDecision:
        return RoutingDecision(
            query_id=query_id,
            chosen=self._chosen,
            competences=frozenset(),
            rationale=("stub",),
        )


class _ExplodingBackend:
    """A leg that raises on search. The partial-failure case."""

    def __init__(self, backend: Backend):
        self.backend = backend

    def name(self) -> str:
        return f"exploding({self.backend.value})"

    def search(self, query: str, k: int = 5):
        raise RuntimeError("store unreachable")


class _FederationWithOneBrokenLeg:
    def __init__(self, inner, broken: Backend):
        self.inner = inner
        self.broken = broken

    def get(self, backend: Backend):
        if backend is self.broken:
            return _ExplodingBackend(backend)
        return self.inner.get(backend)


# ---------------------------------------------------- only the chosen legs


def test_only_the_chosen_backends_are_consulted(fed):
    """Only the legs the router chose are searched.

    Asserts on what was asked, not on what came back. The vector-only router
    picks one leg; the other three must never be searched. Checking the fused
    output instead would pass against a `fed.all()` fan-out whose extra legs
    happened to lose.
    """
    recording = _RecordingFederation(fed)
    result = federated_search(recording, VectorOnlyRouter(), "anything at all", "q")

    assert recording.asked == [Backend.VECTOR]
    assert result.consulted == (Backend.VECTOR,)
    assert result.skipped == frozenset(Backend) - {Backend.VECTOR}
    assert result.backends_consulted == 1


def test_a_narrower_route_asks_strictly_fewer_legs_than_fan_out(fed):
    """The cost claim, as an execution rather than as arithmetic."""
    query = "how many incidents did payments have in 2026"

    narrow = _RecordingFederation(fed)
    federated_search(narrow, HeuristicRouter(fed), query, "q")
    wide = _RecordingFederation(fed)
    federated_search(wide, FanOutRouter(), query, "q")

    assert len(wide.asked) == len(Backend)
    assert len(narrow.asked) < len(wide.asked)
    assert set(narrow.asked) <= set(wide.asked)


def test_every_chosen_leg_is_asked_exactly_once(fed):
    """A leg searched twice would inflate nothing visible but cost double."""
    recording = _RecordingFederation(fed)
    federated_search(recording, FanOutRouter(), "which teams own checkout", "q")
    assert sorted(recording.asked, key=lambda b: b.value) == sorted(
        Backend, key=lambda b: b.value
    )
    assert len(recording.asked) == len(set(recording.asked))


def test_the_consulted_order_is_stable(fed):
    """`chosen` is a frozenset. Two runs must still consult in one order."""
    first = _RecordingFederation(fed)
    federated_search(first, FanOutRouter(), "why do retries fail", "q")
    second = _RecordingFederation(fed)
    federated_search(second, FanOutRouter(), "why do retries fail", "q")
    assert first.asked == second.asked


# ------------------------------------------------------------- provenance


def test_the_merged_list_carries_which_leg_produced_each_item(fed):
    """The README's "every item carries which leg produced it"."""
    result = federated_search(
        fed, FanOutRouter(), "why do we keep seeing ERR_TOKEN_9101", "q"
    )
    assert result.hits, "the fan-out router should return something"
    for hit in result.hits:
        assert hit.contributors, f"{hit.doc_id} arrived with no provenance"
        assert set(hit.contributors) <= set(result.consulted)


def test_no_hit_can_be_attributed_to_a_leg_that_was_never_consulted(fed):
    """The mirror of the test above, and the one that catches a leaked leg."""
    result = federated_search(fed, VectorOnlyRouter(), "an ordinary question", "q")
    for hit in result.hits:
        assert set(hit.contributors) == {Backend.VECTOR}


def test_the_decision_travels_with_the_result(fed):
    """The rationale must survive execution, or a misroute cannot be explained."""
    result = federated_search(
        fed, HeuristicRouter(fed), "how many incidents did payments have in 2026", "q-agg"
    )
    assert result.decision.query_id == "q-agg"
    assert result.decision.rationale
    assert frozenset(result.consulted) == result.decision.chosen


# --------------------------------------------------------- partial failure


def test_a_leg_that_raises_is_named_and_the_rest_still_fuse(fed):
    """The silent failure this module exists to prevent.

    RRF takes a mapping and cannot distinguish a leg that returned nothing
    from a leg that was never consulted from a leg that raised. All three are
    an absent key, and the merged list looks complete in every case.
    """
    broken = _FederationWithOneBrokenLeg(fed, Backend.FULLTEXT)
    result = federated_search(
        broken, FanOutRouter(), "why do we keep seeing ERR_TOKEN_9101", "q"
    )

    assert not result.complete, "a failed leg must not report a complete answer"
    assert [f.backend for f in result.failures] == [Backend.FULLTEXT]
    assert isinstance(result.failures[0].error, RuntimeError)
    assert "fulltext" in str(result.failures[0])
    # The surviving legs still produced an answer, and the failed one is absent
    # from the merge rather than represented by an empty list.
    assert result.hits
    assert Backend.FULLTEXT not in result.per_backend
    # It still counts as consulted: asking a store that failed cost what
    # asking it cost.
    assert Backend.FULLTEXT in result.consulted


def test_a_complete_run_reports_complete_and_has_no_failures(fed):
    """The negative half: `complete` must not be True unconditionally."""
    result = federated_search(fed, FanOutRouter(), "which teams own checkout", "q")
    assert result.complete
    assert result.failures == ()


def test_an_empty_leg_is_not_reported_as_a_failure(fed):
    """A store that legitimately found nothing is not a broken store.

    Conflating the two would make `complete` useless: the graph leg returns []
    for every query that links no entity, which is ordinary and correct.
    """
    result = federated_search(
        fed, FanOutRouter(), "which service owns gateway.envelope.strict", "q-trap-3"
    )
    assert result.complete
    assert result.per_backend[Backend.GRAPH] == []


# ------------------------------------------------------------- concurrency


def test_an_executor_produces_identical_results_to_sequential(fed):
    """The concurrency seam must not change the answer, only who waits."""
    query = "why do we keep seeing ERR_TOKEN_9101 after partner rotations"
    sequential = federated_search(fed, FanOutRouter(), query, "q")
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = federated_search(fed, FanOutRouter(), query, "q", executor=pool)

    assert concurrent.consulted == sequential.consulted
    assert [h.doc_id for h in concurrent.hits] == [h.doc_id for h in sequential.hits]
    assert [h.fused_score for h in concurrent.hits] == [
        h.fused_score for h in sequential.hits
    ]


def test_an_executor_still_consults_only_the_chosen_legs(fed):
    """Concurrency must not become an excuse to submit everything."""
    recording = _RecordingFederation(fed)
    with ThreadPoolExecutor(max_workers=4) as pool:
        federated_search(recording, VectorOnlyRouter(), "anything", "q", executor=pool)
    assert recording.asked == [Backend.VECTOR]


def test_a_failing_leg_under_an_executor_is_still_named(fed):
    """The failure path must survive the seam it is least likely to be run on."""
    broken = _FederationWithOneBrokenLeg(fed, Backend.VECTOR)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = federated_search(
            broken, FanOutRouter(), "why do retries fail", "q", executor=pool
        )
    assert not result.complete
    assert [f.backend for f in result.failures] == [Backend.VECTOR]


# --------------------------------------------------------- depth and cost


def test_per_leg_depth_and_merged_length_are_separate_knobs(fed):
    """Fusion can only rank what it was shown.

    Asking each leg for two and fusing to ten cannot return more than the
    union of four two-item lists, so collapsing the two numbers into one is a
    quiet way to lose results.
    """
    query = "why do we keep seeing ERR_TOKEN_9101 after partner rotations"
    shallow = federated_search(fed, FanOutRouter(), query, "q", k=10, per_leg_k=1)
    deep = federated_search(fed, FanOutRouter(), query, "q", k=10, per_leg_k=5)
    assert len(shallow.hits) <= len(Backend)
    assert len(deep.hits) > len(shallow.hits)


def test_measured_fan_out_equals_the_scored_fan_out(corpus, fed):
    """The number the headline table reports, derived the other way.

    `RoutingReport.fan_out` sums the routing decisions; this sums the legs
    execution actually consulted. They must agree, because a disagreement
    means something between deciding and executing is adding or dropping a
    leg, as fusing `fed.all()` while the table reports routed decisions
    would.
    """
    from router.metrics import score_routing
    from router.routing import route_all

    router = HeuristicRouter(fed)
    results = [
        federated_search(fed, router, q.text, q.query_id) for q in corpus.queries
    ]
    report = score_routing("heuristic", corpus.queries, route_all(router, corpus.queries))

    assert measured_fan_out(results) == pytest.approx(report.fan_out)


def test_measured_fan_out_of_an_empty_run_is_zero():
    assert measured_fan_out([]) == 0.0


def test_a_router_that_chose_nothing_returns_an_empty_answer_not_a_crash(fed):
    """A degenerate decision must be an empty answer, not a traceback."""
    result = federated_search(fed, _StubRouter(frozenset()), "anything", "q")
    assert result.consulted == ()
    assert result.hits == ()
    assert result.backends_consulted == 0
    assert result.skipped == frozenset(Backend)
    assert result.complete
