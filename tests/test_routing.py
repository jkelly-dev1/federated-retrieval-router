"""Routing, fusion, and the metrics that score them.

The invariants here are mostly about the MEASUREMENT rather than the router: a
metric that cannot distinguish a correct router from an expensive one, or that
silently scores a misaligned query list, is worse than no metric.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from router.backends import build_federation
from router.corpus import PRODUCTION_MIX, build_corpus
from router.fusion import (
    DEFAULT_WINDOW,
    RANK_CONSTANT,
    reciprocal_rank_fusion,
    unique_contributions,
    window_sweep,
)
from router.metrics import (
    confusion,
    score_routing,
    validate_competences,
    weighted_correctness,
)
from router.models import Backend, Competence, Edge, RankedHit, RoutingDecision
from router.routing import (
    FanOutRouter,
    HeuristicRouter,
    VectorOnlyRouter,
    route_all,
)


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def fed(corpus):
    return build_federation(corpus)


@pytest.fixture(scope="module")
def heuristic(fed):
    return HeuristicRouter(fed)


# --------------------------------------------------------------- the guards


def test_an_aggregate_tell_without_anything_countable_is_suppressed(heuristic):
    """'How many retries does the backoff design recommend' opens with the
    strongest counting phrase in the language and needs no counting."""
    decision = heuristic.route(
        "how many retries does the backoff design recommend", "t"
    )
    assert Backend.RELATIONAL not in decision.chosen
    assert any("SUPPRESSED" in r for r in decision.rationale)


def test_the_aggregate_guard_still_fires_when_something_is_countable(heuristic):
    """Mutation check: the guard must suppress a figure of speech, not the
    competence."""
    decision = heuristic.route("how many incidents did payments have", "t")
    assert Backend.RELATIONAL in decision.chosen


def test_a_relationship_tell_with_no_linked_entity_is_suppressed(heuristic):
    """'Which service owns gateway.envelope.strict' says owns, and the subject
    is a config key with no node. A traversal has nowhere to start."""
    decision = heuristic.route("which service owns gateway.envelope.strict", "t")
    assert Backend.GRAPH not in decision.chosen
    assert any("SUPPRESSED" in r for r in decision.rationale)


def test_the_relationship_guard_still_fires_on_a_real_entity(heuristic):
    decision = heuristic.route("which teams own the services checkout depends on", "t")
    assert Backend.GRAPH in decision.chosen


def test_an_identifier_always_routes_to_fulltext(heuristic):
    for query in ("ERR_UPSTREAM_4423", "what does queue.prefetch.count do"):
        assert Backend.FULLTEXT in heuristic.route(query, "t").chosen


def test_every_decision_carries_a_rationale(heuristic, corpus):
    """A confusion matrix says THAT a router misroutes and never WHY."""
    for decision in route_all(heuristic, corpus.queries):
        assert decision.rationale
        assert all(r.strip() for r in decision.rationale)


# -------------------------------------------------------------- the metrics


def test_choosing_a_superset_is_correct_but_not_free(corpus, fed):
    """Correctness and cost are separate axes.

    Fan-out cannot be wrong. If correctness folded cost in, the metric could
    not say that, and 'route everything everywhere' would look like a result.
    """
    fan = score_routing("fan-out", corpus.queries, route_all(FanOutRouter(), corpus.queries))
    heur = score_routing("heuristic", corpus.queries, route_all(HeuristicRouter(fed), corpus.queries))
    assert fan.correctness == 1.0
    assert fan.fan_out == float(len(Backend))
    assert heur.fan_out < fan.fan_out


def test_a_single_store_fails_the_other_competences(corpus):
    report = score_routing(
        "vector-only", corpus.queries, route_all(VectorOnlyRouter(), corpus.queries)
    )
    assert report.correctness < 0.5
    failed = {qid for qid, _, _ in report.failures}
    assert any(q.startswith("q-exact") for q in failed)
    assert any(q.startswith("q-rel") for q in failed)
    assert any(q.startswith("q-agg") for q in failed)


def test_scoring_a_misaligned_query_list_raises(corpus, fed):
    """Silently scoring a router against a different query set produces a
    confident number about nothing."""
    decisions = route_all(HeuristicRouter(fed), corpus.queries)
    with pytest.raises(ValueError, match="queries but"):
        score_routing("x", corpus.queries[:-1], decisions)
    shuffled = [decisions[-1]] + list(decisions[1:])
    with pytest.raises(ValueError, match="does not match"):
        score_routing("x", corpus.queries, shuffled)


def test_every_scorer_refuses_a_misaligned_query_list(corpus, fed):
    """All three scorers refuse a misaligned query list, not just one.

    `zip` stops at the shorter list and pairs whatever lines up. Both
    published figures (the prod-mix column and the confusion matrix) flow
    through these scorers, and a bare zip would produce confident, plausible
    numbers from lists that do not correspond: rotating the decisions by one
    moves prod-mix to 0.496, and truncating them to ten gives a perfect 1.000.

    A scorer that raises on one input shape and invents a number on the same
    shape one function over is worse than one that never checked: the reader
    who tested the first one has no reason to test the others.
    """
    decisions = route_all(HeuristicRouter(fed), corpus.queries)
    rotated = list(decisions[1:]) + [decisions[0]]
    truncated = list(decisions[:10])

    for scorer in (
        lambda q, d: score_routing("x", q, d),
        lambda q, d: weighted_correctness(q, d, PRODUCTION_MIX),
        lambda q, d: confusion(q, d),
    ):
        with pytest.raises(ValueError, match="queries but"):
            scorer(corpus.queries, truncated)
        with pytest.raises(ValueError, match="does not match"):
            scorer(corpus.queries, rotated)


def test_the_relationship_guard_follows_the_graph_it_is_given(corpus):
    """The design claim in routing.py, pinned.

    `HeuristicRouter` takes a `federation` for exactly one reason, and the
    docstring says so: the relationship guard asks the graph whether a query
    links to a real node, because "the alternative is a hardcoded entity list
    in this file, which would be a second copy of the graph that silently
    drifts from the first."

    A test that builds the router against the standard corpus cannot see that
    drift, because there the graph and a hardcoded list of its node names
    agree by construction. This builds federations whose graph does not match
    the default corpus, in both directions, and asserts the routing follows
    the graph it was handed. A hardcoded list fails both halves.
    """
    # --- a node the default corpus does not have --------------------------
    invented = build_federation(
        replace(corpus, edges=corpus.edges + (Edge("widgetron", "owned_by", "growth"),))
    )
    decision = HeuristicRouter(invented).route("which team owns widgetron", "q")
    assert Backend.GRAPH in decision.chosen, (
        "the guard did not see a node that exists in the graph it was given; "
        "it is consulting something other than the graph")

    # --- and the mirror: a node the default corpus DOES have, removed -----
    without_checkout = build_federation(
        replace(
            corpus,
            edges=tuple(
                e for e in corpus.edges
                if e.source != "checkout" and e.target != "checkout"
            ),
        )
    )
    decision = HeuristicRouter(without_checkout).route(
        "which teams own the services checkout depends on", "q")
    assert Backend.GRAPH not in decision.chosen, (
        "the guard routed to a graph that has no node to start from; a "
        "hardcoded entity list would do exactly this")
    assert any("SUPPRESSED" in line for line in decision.rationale), (
        "the suppression must be stated in the rationale, not merely happen")

    # --- and the control: the same query against the real graph DOES route -
    real = HeuristicRouter(build_federation(corpus)).route(
        "which teams own the services checkout depends on", "q")
    assert Backend.GRAPH in real.chosen, (
        "the query itself stopped routing to the graph, so the assertion above "
        "proves nothing")


def test_the_aligned_case_still_scores(corpus, fed):
    """The negative half: the guard must not reject correct input.

    A check that raised on everything would pass the test above while making
    every scorer useless, which is the usual way an input guard goes wrong.
    """
    decisions = route_all(HeuristicRouter(fed), corpus.queries)
    assert score_routing("x", corpus.queries, decisions).correctness > 0
    assert weighted_correctness(corpus.queries, decisions, PRODUCTION_MIX) > 0
    assert confusion(corpus.queries, decisions)


def test_the_query_mix_changes_the_verdict(corpus):
    """THE HEADLINE. Same router, same corpus, different assumption about
    what people ask."""
    decisions = route_all(VectorOnlyRouter(), corpus.queries)
    balanced = score_routing("v", corpus.queries, decisions).correctness
    weighted = weighted_correctness(corpus.queries, decisions, PRODUCTION_MIX)
    assert weighted > balanced * 1.5, (
        "the balanced set and the production mix now agree, so the evaluation "
        "set has drifted toward the mix and the finding has evaporated"
    )


def test_trap_accuracy_is_reported_separately(corpus, fed):
    """Folding traps into the average lets a good score on the easy majority
    hide a total failure on them."""
    report = score_routing(
        "vector-only", corpus.queries, route_all(VectorOnlyRouter(), corpus.queries)
    )
    assert report.trap_total >= 4
    assert report.trap_accuracy < report.plain_accuracy + 1.0
    assert report.plain_total + report.trap_total == report.total


def test_the_confusion_matrix_records_off_diagonal_misroutes(corpus):
    matrix = confusion(corpus.queries, route_all(VectorOnlyRouter(), corpus.queries))
    assert matrix.get((Backend.GRAPH, Backend.VECTOR), 0) > 0


def test_competence_validation_excludes_queries_with_no_document_answer(corpus, fed):
    """An aggregate has no document answer, so scoring it by document overlap
    is a category error rather than a low score."""
    retrieved = {
        q.query_id: {b.backend: [h.doc_id for h in b.search(q.text, k=3)] for b in fed.all()}
        for q in corpus.queries
    }
    checks = validate_competences(corpus.queries, retrieved, k=3)
    checked = {c.query_id for c in checks}
    assert not any(q.startswith("q-agg") for q in checked)
    assert all(corpus_q.relevant_docs for corpus_q in corpus.queries if corpus_q.query_id in checked)


# ---------------------------------------------------------------- fusion


def _hits(backend: Backend, ids: list[str]) -> list[RankedHit]:
    return [
        RankedHit(doc_id=d, backend=backend, rank=i, score=1.0 / i)
        for i, d in enumerate(ids, start=1)
    ]


def test_fusion_uses_rank_and_ignores_the_native_score():
    """A BM25 score of 14.2 and a cosine of 0.83 are not on the same scale.

    Mutation check: multiplying one leg's scores by a thousand must not move
    the fused order at all.
    """
    a = _hits(Backend.VECTOR, ["x", "y"])
    b = _hits(Backend.FULLTEXT, ["y", "z"])
    before = [h.doc_id for h in reciprocal_rank_fusion({Backend.VECTOR: a, Backend.FULLTEXT: b})]
    inflated = [RankedHit(h.doc_id, h.backend, h.rank, h.score * 1000) for h in a]
    after = [h.doc_id for h in reciprocal_rank_fusion({Backend.VECTOR: inflated, Backend.FULLTEXT: b})]
    assert before == after


def test_agreement_between_backends_outranks_a_single_first_place():
    """The whole argument for fusion, as an assertion."""
    fused = reciprocal_rank_fusion({
        Backend.VECTOR: _hits(Backend.VECTOR, ["agreed", "solo"]),
        Backend.FULLTEXT: _hits(Backend.FULLTEXT, ["agreed", "other"]),
    })
    assert fused[0].doc_id == "agreed"
    assert len(fused[0].contributors) == 2


def test_provenance_survives_fusion():
    fused = reciprocal_rank_fusion({
        Backend.GRAPH: _hits(Backend.GRAPH, ["only-graph"]),
        Backend.VECTOR: _hits(Backend.VECTOR, ["shared"]),
        Backend.FULLTEXT: _hits(Backend.FULLTEXT, ["shared"]),
    })
    by_id = {h.doc_id: h for h in fused}
    assert by_id["only-graph"].unique_to is Backend.GRAPH
    assert by_id["shared"].unique_to is None
    assert unique_contributions(fused) == {Backend.GRAPH: 1}


def test_a_narrow_window_silently_discards_agreement():
    """The constant that decides fusion and fails quietly.

    A document one leg ranks deep cannot be fused no matter how strongly
    another leg agrees, and the merged list looks perfectly reasonable.
    """
    deep = _hits(Backend.FULLTEXT, [f"pad{i}" for i in range(1, 12)] + ["target"])
    shallow = _hits(Backend.VECTOR, ["target"])
    per_backend = {Backend.FULLTEXT: deep, Backend.VECTOR: shallow}

    narrow = reciprocal_rank_fusion(per_backend, window=5)
    wide = reciprocal_rank_fusion(per_backend, window=DEFAULT_WINDOW)
    narrow_contrib = next(h for h in narrow if h.doc_id == "target").contributors
    wide_contrib = next(h for h in wide if h.doc_id == "target").contributors
    assert len(narrow_contrib) == 1, "the deep hit should be invisible at window 5"
    assert len(wide_contrib) == 2, "the default window must see it"


def test_the_window_sweep_reports_where_recall_appears():
    deep = _hits(Backend.FULLTEXT, [f"pad{i}" for i in range(1, 9)] + ["target"])
    sweep = window_sweep({Backend.FULLTEXT: deep}, {"target"}, windows=(1, 5, 10, 20))
    found = dict((w, n) for w, n, _ in sweep)
    assert found[1] == 0 and found[20] == 1


def test_the_default_window_sits_above_the_corpus_plateau(corpus, fed):
    """The test that holds DEFAULT_WINDOW, against the real corpus.

    Over every labeled query with document answers, this finds the deepest
    window any of them needs before fusion stops finding more, and the
    constant must sit at or above it. A sweep of a synthetic list would not
    move whatever value the constant held.
    """
    plateaus = {}
    for q in corpus.queries:
        if not q.relevant_docs:
            continue
        per_backend = {b.backend: b.search(q.text, k=50) for b in fed.all()}
        sweep = window_sweep(per_backend, q.relevant_docs)
        best = max(found for _, found, _ in sweep)
        plateaus[q.query_id] = next(w for w, found, _ in sweep if found == best)

    deepest_id = max(plateaus, key=lambda qid: plateaus[qid])
    deepest = plateaus[deepest_id]
    # Both directions. If every query plateaued at window 1 the assertion
    # below would hold for any constant whatsoever, which is how the gate
    # check on this same constant came to be unable to fail.
    assert deepest > 1, (
        "no labeled query needs a window deeper than 1, so nothing here "
        "measures DEFAULT_WINDOW")
    assert DEFAULT_WINDOW >= deepest, (
        f"{deepest_id} needs window {deepest} but DEFAULT_WINDOW is "
        f"{DEFAULT_WINDOW}; fusion is discarding agreement it was shown")


def test_the_rank_constant_is_the_published_default():
    assert RANK_CONSTANT == 60
