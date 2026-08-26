"""The four backends, and the competences they are supposed to have.

The point of these is not that BM25 works. It is that each leg is BEST at its
own thing and demonstrably WORSE at the others, because a federation of four
interchangeable stores has nothing to route.
"""
from __future__ import annotations

import pytest

from router.backends import (
    FulltextBackend,
    GraphBackend,
    RelationalBackend,
    VectorBackend,
    build_federation,
)
from router.corpus import build_corpus
from router.embeddings import (
    DIMENSIONS,
    HashingEmbedder,
    content_tokens,
    cosine,
    tokenize,
)
from router.models import Backend


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


@pytest.fixture(scope="module")
def fed(corpus):
    return build_federation(corpus)


# ------------------------------------------------------------- tokenization


def test_an_identifier_does_not_absorb_trailing_punctuation():
    """The bug that silently broke the exact-term competence.

    `[a-z0-9_.]+` captures a sentence-ending period, so a runbook saying
    "the setting is auth.audience.allowlist." indexed a token no query could
    match, and the fulltext leg returned NOTHING on identifier lookups. It
    failed silently: an empty result set looks like a hard query.
    """
    assert tokenize("auth.audience.allowlist.") == ["auth.audience.allowlist"]
    assert tokenize("see ERR_UPSTREAM_4423.") == ["see", "err_upstream_4423"]
    assert tokenize("raising queue.prefetch.count, note") == [
        "raising", "queue.prefetch.count", "note",
    ]
    doc = set(tokenize("The setting is auth.audience.allowlist. Adding an audience"))
    assert doc & set(tokenize("auth.audience.allowlist"))


def test_dots_inside_identifiers_still_survive():
    """Mutation check: the fix must not split identifiers instead."""
    assert tokenize("gateway.envelope.strict") == ["gateway.envelope.strict"]


# -------------------------------------------------------------- competences


def test_fulltext_beats_vector_on_an_opaque_identifier(fed):
    """The standard argument for hybrid retrieval, measured rather than cited."""
    query = "ERR_UPSTREAM_4423"
    ft = fed.fulltext.search(query, k=1)
    vec = fed.vector.search(query, k=1)
    assert ft and ft[0].doc_id == "runbook-err-101"
    assert not vec or vec[0].doc_id != "runbook-err-101", (
        "the vector leg ranked an opaque identifier first; either the embedder "
        "gained a competence it should not have, or the corpus lost its noise"
    )


def test_only_the_graph_answers_a_pure_traversal(fed):
    """The answer appears in no document, so no text retriever can find it."""
    query = "which teams own the services checkout depends on"
    graph = {h.doc_id for h in fed.graph.search(query, k=5)}
    assert {"storefront", "discovery"} & graph or {"catalog", "payments"} & graph
    for other in (fed.vector, fed.fulltext, fed.relational):
        hits = {h.doc_id for h in other.search(query, k=5)}
        assert not (hits & {"storefront", "discovery", "money"}), (
            f"{other.backend.value} returned a team node it cannot know about"
        )


def test_only_the_relational_leg_computes_an_aggregate(fed):
    """A count is not retrievable. It is computed, and its id says so."""
    hits = fed.relational.search("how many incidents did payments have in 2026", k=3)
    assert hits and hits[0].doc_id.startswith("agg:")
    assert "count" in hits[0].why
    for other in (fed.vector, fed.fulltext, fed.graph):
        assert not any(
            h.doc_id.startswith("agg:") for h in other.search("how many", k=5)
        )


def test_the_graph_returns_nothing_when_nothing_links(fed):
    """A relationship-shaped question about an unmodeled entity.

    Empty is the correct answer and the reason the router's relationship guard
    consults the graph rather than a keyword list.
    """
    assert fed.graph.linked_entities("which service owns gateway.envelope.strict") == frozenset()
    assert fed.graph.search("which service owns gateway.envelope.strict") == []


def test_the_graph_traverses_backwards_from_a_sink_node(fed):
    """`owned_by` runs service -> team, so a question about a TEAM has no
    forward path. Direction is decided by the graph's shape at that node."""
    hits = {h.doc_id for h in fed.graph.search(
        "incidents touching services owned by the money team", k=10)}
    assert "payments" in hits or "billing" in hits


def test_a_blast_radius_question_reverses_the_dependency_edges(fed):
    hits = {h.doc_id for h in fed.graph.search(
        "which services would be affected if identity went down", k=10)}
    assert {"payments", "catalog"} <= hits


def test_traversal_follows_only_the_edge_kinds_it_was_asked_for(fed):
    """The graph leg's scoping rule, pinned in both directions.

    `kinds` is what turns this into a graph of MEANING rather than a graph
    of connectivity. Models.py says it outright: "A graph of untyped edges
    answers \'is there a path\' and not \'is there a path THAT MEANS
    SOMETHING\'." Dropping `edge.kind in kinds` from either traversal left
    the whole suite green, in both copies, independently.

    checkout is reached by `touched_by` from four incident records and by
    `depends_on` from nothing. Ask for dependents and the answer is empty;
    that emptiness is the correct answer and it is what the filter buys.
    """
    graph = fed.graph
    kinds = sorted({e.kind for edges in graph._out.values() for e in edges})
    assert set(kinds) >= {"depends_on", "owned_by", "touched_by"}, kinds

    scoped = graph.dependents_of("checkout", ("depends_on",))
    unscoped = graph.dependents_of("checkout", kinds)
    assert scoped == {}, (
        f"nothing depends_on checkout, got {sorted(scoped)} -- the traversal "
        "is following edge kinds it was not asked for")
    assert len(unscoped) == 4, (
        "the fixture no longer has untyped neighbors to be confused by, so "
        f"this test cannot detect a dropped filter (got {sorted(unscoped)})")
    assert all(n.startswith("incident-") for n in unscoped), sorted(unscoped)


def test_forward_traversal_also_honors_the_edge_kinds(fed):
    """The same guard, on the other copy. It was implemented twice and tested
    zero times; a fix applied to one traversal and not the other would have
    looked identical from the suite."""
    graph = fed.graph
    kinds = sorted({e.kind for edges in graph._out.values() for e in edges})
    scoped = graph.neighbors("identity", ("depends_on",), 3)
    unscoped = graph.neighbors("identity", kinds, 3)
    assert len(unscoped) > len(scoped), (
        "the two traversals agree, so this test cannot tell a scoped walk "
        f"from an unscoped one (scoped {len(scoped)}, all {len(unscoped)})")
    assert set(scoped) <= set(unscoped)


# ----------------------------------------------------------------- ranking


def test_ranks_are_dense_and_one_based(fed):
    for backend in fed.all():
        hits = backend.search("incident payments settlement", k=5)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_ties_break_deterministically(corpus):
    """An unstable ordering would make the gate flap for reasons unrelated to
    any routing decision."""
    a = FulltextBackend(corpus.documents).search("incident", k=10)
    b = FulltextBackend(corpus.documents).search("incident", k=10)
    assert [h.doc_id for h in a] == [h.doc_id for h in b]


def test_zero_scoring_documents_are_not_returned(fed):
    """A result set padded with non-matches makes recall@k meaningless."""
    for hit in fed.fulltext.search("ERR_UPSTREAM_4423", k=20):
        assert hit.score > 0.0


# -------------------------------------------------------------- embeddings


def test_the_embedder_is_deterministic_across_instances():
    assert HashingEmbedder().embed("payments") == HashingEmbedder().embed("payments")


def test_an_empty_query_embeds_to_a_zero_vector_and_matches_nothing():
    emb = HashingEmbedder()
    zero = emb.embed("the and of to")  # all stopwords
    assert cosine(zero, emb.embed("payments")) == 0.0


def test_cosine_raises_on_a_width_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_the_dimension_is_wide_enough_to_avoid_manufactured_similarity(corpus):
    """DIMENSIONS is measured, not chosen, and the statistic matters.

    Manufactured similarity is a floor, not a spread. Comparing the top hit
    against the median does not measure it: at 32 dimensions that gap is
    WIDER, because collisions make every score noisier in both directions. The quantity that actually degrades is how alike UNRELATED
    documents look, which rises as the width falls and puts a floor under the
    vector leg's false-positive rate.

    Measured background similarity between unrelated document pairs:

        dims    mean |cos|     p95
          16        0.2218  0.5902
          32        0.1578  0.3831
          64        0.1202  0.2887
         128        0.0949  0.2169
         256        0.0772  0.1852
         512        0.0616  0.1435

    Monotone in width, and 256 is where the mean drops below 0.08 without
    paying for a width the corpus cannot use.
    """
    import statistics

    docs = corpus.documents[:40]

    def background(width: int) -> float:
        emb = HashingEmbedder(dimensions=width)
        vectors = [emb.embed(f"{d.title} {d.text}") for d in docs]
        return statistics.mean(
            abs(cosine(vectors[i], vectors[j]))
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        )

    narrow, shipped = background(32), background(DIMENSIONS)
    assert shipped < narrow, (
        f"background similarity at {DIMENSIONS} dims ({shipped:.4f}) is not "
        f"below the 32-dim floor ({narrow:.4f}); the collision argument fails"
    )
    assert shipped < 0.10, (
        f"unrelated documents average {shipped:.4f} cosine, which puts a floor "
        f"under every retrieval score the vector leg produces"
    )
