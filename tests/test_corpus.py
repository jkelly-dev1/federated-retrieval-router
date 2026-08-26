"""The evaluation set, which everything else is scored against.

These are the most important tests in the repository, because a corpus can
fail every one of them silently. A router measured against a broken ground
truth produces confident numbers about nothing, and that failure is invisible
from inside the routing metrics.
"""
from __future__ import annotations

import pytest

from router.backends import build_federation
from router.corpus import PRODUCTION_MIX, build_corpus
from router.embeddings import content_tokens
from router.models import Backend, Competence

EVAL_K = 5
MAX_SELECTIVITY = 0.15


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def fed(corpus):
    return build_federation(corpus)


# ------------------------------------------------------------- selectivity


def test_the_corpus_is_large_enough_for_top_k_to_discriminate(corpus):
    """The defect that invalidated the first evaluation.

    At 12 documents a top-5 result set was 42% of the corpus, so every backend
    returned the answer and recall@5 could not tell a leg that ranked it first
    from one that ranked it fifth. Labels agreed with measurement on 1 of 11
    queries and nothing downstream was measuring anything.
    """
    selectivity = EVAL_K / len(corpus.documents)
    assert selectivity <= MAX_SELECTIVITY, (
        f"top-{EVAL_K} is {selectivity:.1%} of the corpus; a candidate pool "
        f"this large cannot discriminate between backends"
    )


def test_the_corpus_has_distractors_not_just_anchors(corpus):
    """Mutation check on the test above: size alone is not enough.

    A corpus padded with documents that share no vocabulary would pass the
    selectivity check and still fail to make retrieval hard. These groups exist
    to compete with the anchors, so their presence is asserted by name.
    """
    ids = {d.doc_id for d in corpus.documents}
    assert any(i.startswith("design-circuit") for i in ids), "near-miss designs"
    assert any(i.startswith("design-cachewarm") for i in ids), "near-miss designs"
    assert sum(1 for i in ids if i.startswith("runbook-err_")) >= 10
    assert sum(1 for d in corpus.documents if d.kind == "incident") >= 30


# ----------------------------------------------------------- label integrity


def test_every_required_backend_actually_returns_something(corpus, fed):
    """A label naming a backend that answers nothing is a broken label.

    Two were broken when this test was written: a fulltext label defeated by a
    tokenizer bug, and a graph label with no forward path from a team node.
    Both looked like hard queries and were not.
    """
    broken = [
        (q.query_id, b.value)
        for q in corpus.queries
        for b in q.required
        if not fed.get(b).search(q.text, k=EVAL_K)
    ]
    assert not broken, f"labels naming a backend that returns nothing: {broken}"


def test_every_relevant_document_exists(corpus):
    for q in corpus.queries:
        for doc_id in q.relevant_docs:
            corpus.by_id(doc_id)  # raises KeyError if absent


def test_query_ids_are_unique(corpus):
    ids = [q.query_id for q in corpus.queries]
    assert len(ids) == len(set(ids))


def test_document_ids_are_unique(corpus):
    ids = [d.doc_id for d in corpus.documents]
    assert len(ids) == len(set(ids))


def test_aggregate_queries_have_no_relevant_documents(corpus):
    """The answer to a count is a number that appears in no document.

    Attaching relevant_docs to an aggregate query would invite scoring the
    relational leg by document overlap, which reports zero for a leg that is
    answering perfectly.
    """
    for q in corpus.queries:
        if q.competences == frozenset({Competence.AGGREGATE}):
            assert not q.relevant_docs, (
                f"{q.query_id} is a pure aggregate and must not carry "
                f"relevant_docs"
            )


# ---------------------------------------------------------------- coverage


def test_every_backend_is_required_by_several_queries(corpus):
    """A backend required by one query is measured by a coin flip."""
    for backend in Backend:
        n = sum(1 for q in corpus.queries if backend in q.required)
        assert n >= 4, f"{backend.value} is required by only {n} queries"


def test_the_set_contains_traps_and_multi_backend_queries(corpus):
    assert sum(1 for q in corpus.queries if q.trap) >= 4
    assert sum(1 for q in corpus.queries if len(q.required) > 1) >= 3


def test_every_trap_explains_itself(corpus):
    """A trap with no stated reason is indistinguishable from a mistake."""
    for q in corpus.queries:
        if q.trap:
            assert len(q.trap) > 40, f"{q.query_id} trap note is too thin"


def test_the_semantic_queries_are_genuine_paraphrases(corpus):
    """Low lexical overlap, or the semantic competence is not being tested.

    If a semantic query shared most of its vocabulary with its target, BM25
    would win for the honest reason and the vector leg would never be
    exercised. Measured jaccard here runs 0.02 to 0.10.
    """
    for q in corpus.queries:
        if Backend.VECTOR not in q.required or not q.relevant_docs:
            continue
        qt = set(content_tokens(q.text))
        for doc_id in q.relevant_docs:
            doc = corpus.by_id(doc_id)
            dt = set(content_tokens(f"{doc.title} {doc.text}"))
            jaccard = len(qt & dt) / len(qt | dt)
            assert jaccard < 0.20, (
                f"{q.query_id} shares {jaccard:.2f} of its tokens with "
                f"{doc_id}; that is a keyword query wearing a paraphrase"
            )


# --------------------------------------------------------------- the graph


def test_every_service_and_team_has_a_graph_node(corpus, fed):
    """A relationship question about an unmodeled entity is unanswerable, and
    the router looks correct while returning nothing."""
    nodes = fed.graph.nodes()
    services = {d.service for d in corpus.documents if d.service}
    teams = {d.team for d in corpus.documents if d.team}
    assert not services - nodes, f"services with no node: {sorted(services - nodes)}"
    assert not teams - nodes, f"teams with no node: {sorted(teams - nodes)}"


def test_every_incident_is_reachable_from_the_graph(corpus, fed):
    """The graph and the relational store must see the same incidents.

    Without this, a question joining the two is answered against whatever
    fraction happens to be modeled, while looking complete.
    """
    incidents = {d.doc_id for d in corpus.documents if d.kind == "incident"}
    touched = {e.source for e in corpus.edges if e.kind == "touched_by"}
    assert not incidents - touched, (
        f"{len(incidents - touched)} incidents have no touched_by edge"
    )


# ------------------------------------------------------------ the mix model


def test_the_production_mix_covers_every_competence_and_sums_to_one(corpus):
    assert set(PRODUCTION_MIX) == set(Competence)
    assert abs(sum(PRODUCTION_MIX.values()) - 1.0) < 1e-9


def test_the_evaluation_set_is_balanced_and_the_mix_is_not(corpus):
    """The two must differ, or the mix finding has nothing to say.

    A balanced set is needed to measure per backend; a production log is not
    balanced. If the evaluation set drifted toward the mix, the project would
    lose its most transferable result without any test going red.
    """
    counts = {c: 0 for c in Competence}
    for q in corpus.queries:
        for c in q.competences:
            counts[c] += 1
    share = {c: n / sum(counts.values()) for c, n in counts.items()}
    assert max(share.values()) < 0.45, "the evaluation set is no longer balanced"
    assert max(PRODUCTION_MIX.values()) >= 0.60, "the mix is no longer skewed"
