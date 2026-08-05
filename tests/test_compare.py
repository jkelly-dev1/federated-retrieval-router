"""The store A/B, tested with two fake stores and no store.

THE FAILURE THIS FILE IS BUILT AROUND. The interesting outcome of a store
comparison is not "the real one is better". It is the case where the SCORE is
identical and the RETRIEVED DOCUMENTS are not: a project that reported only the
score would publish "the stand-in cost us nothing" while its two stores agreed
on almost nothing. That case cannot be produced on demand from a real
Elasticsearch, so it is constructed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from router.compare import (
    WIDTH,
    compare_backend,
    render_comparison,
    render_detail,
    render_headline,
)
from router.models import Backend, Competence, LabeledQuery, RankedHit


@dataclass
class _Store:
    """A store whose answers are scripted per query."""

    backend: Backend
    answers: dict
    label: str = "fake"

    def name(self) -> str:
        return self.label

    def search(self, query: str, k: int = 5):
        return [
            RankedHit(doc_id=doc, backend=self.backend, rank=i, score=1.0 / i)
            for i, doc in enumerate(self.answers.get(query, ()), start=1)
        ][:k]


def _queries():
    return (
        LabeledQuery(
            query_id="q1",
            text="paraphrase one",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"a"}),
        ),
        LabeledQuery(
            query_id="q2",
            text="paraphrase two",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"b"}),
        ),
        LabeledQuery(
            query_id="q-agg",
            text="how many incidents",
            competences=frozenset({Competence.AGGREGATE}),
            required=frozenset({Backend.RELATIONAL}),
        ),
    )


def test_a_zero_net_delta_with_total_disagreement_is_not_no_difference():
    """THE HEADLINE FAILURE MODE, CONSTRUCTED.

    The real store answers q2 and loses q1; the toy store does the reverse. Net
    zero. They also return completely different documents on both queries. A
    report that printed only the net would say the stand-in cost nothing.
    """
    queries = _queries()
    toy = _Store(Backend.VECTOR, {"paraphrase one": ["a"], "paraphrase two": ["z"]}, "toy")
    real = _Store(Backend.VECTOR, {"paraphrase one": ["y"], "paraphrase two": ["b"]}, "real")

    result = compare_backend(queries, toy, real)
    assert result.artifact == 0
    assert result.toy_answers == result.real_answers == 1
    assert result.same_set == 0
    assert result.gained == ("q2",) and result.lost == ("q1",)

    headline = "\n".join(render_headline([result]))
    assert "'No difference in" in headline
    assert "not 'no difference'" in headline


def test_aggregate_queries_are_excluded_from_recall_but_not_from_agreement():
    """Their answer appears in no document, so document recall for them is a
    category error -- but the two stores still either agree on the rows they
    compute or they do not, and that is the only comparison this leg has."""
    queries = _queries()
    toy = _Store(Backend.RELATIONAL, {"how many incidents": ["agg:count:all"]})
    real = _Store(Backend.RELATIONAL, {"how many incidents": ["agg:count:all"]})
    result = compare_backend(queries, toy, real)

    # q1 and q2 have document answers but are the VECTOR leg's job, and q-agg
    # is this leg's job but has no document answer -- so nothing is
    # document-scored, and the agreement column is the entire comparison.
    assert result.scored == 0
    assert result.compared == 1        # only q-agg produced anything
    assert result.same_set == 1


def test_two_stores_that_both_return_nothing_have_not_agreed():
    """THE DEFECT THE FIRST REAL RUN OF THIS MODULE EXPOSED.

    The relational leg was compared on the eleven document-bearing queries --
    the ones it is not for -- where both implementations correctly return
    nothing. Eleven empty-vs-empty pairs were counted as eleven agreements, and
    the headline announced that the stand-in cost nothing. The comparison had
    measured zero queries and reported a perfect score.
    """
    queries = _queries()
    silent = _Store(Backend.RELATIONAL, {})
    result = compare_backend(queries, silent, _Store(Backend.RELATIONAL, {}))

    assert result.vacuous
    assert result.compared == 0
    assert result.same_set == 0, "silence is not agreement"
    assert result.toy_recall is None and result.real_recall is None

    table = "\n".join(render_comparison([result]))
    assert "NOTHING" in table and "0.000" not in table
    headline = "\n".join(render_headline([result]))
    assert "NOTHING WAS MEASURED" in headline
    assert "was not costing anything" not in headline


def test_a_leg_with_no_document_ground_truth_reports_n_a_not_zero():
    """A recall of 0.000 reads as 'the store found nothing'. For a leg whose
    answers are computed values matching no document, it means 'this number is
    not defined here', and the difference changes what a reader concludes."""
    queries = (_queries()[2],)  # the aggregate query alone
    toy = _Store(Backend.RELATIONAL, {"how many incidents": ["agg:count:all"]})
    real = _Store(Backend.RELATIONAL, {"how many incidents": ["agg:count:2026"]})
    result = compare_backend(queries, toy, real)

    assert result.scored == 0 and result.compared == 1
    assert result.toy_recall is None
    table = "\n".join(render_comparison([result]))
    assert "n/a" in table
    headline = "\n".join(render_headline([result]))
    assert "structurally zero and means nothing" in headline
    assert result.same_set == 0  # they computed different rows


def test_comparing_two_different_legs_is_refused():
    """A vector store against a fulltext store measures two jobs, not two
    implementations of one."""
    with pytest.raises(ValueError, match="two different jobs"):
        compare_backend(
            _queries(), _Store(Backend.VECTOR, {}), _Store(Backend.FULLTEXT, {})
        )


def test_the_four_verdicts_are_all_reachable():
    queries = _queries()
    toy = _Store(Backend.VECTOR, {"paraphrase one": ["a"], "paraphrase two": []})
    real = _Store(Backend.VECTOR, {"paraphrase one": ["a"], "paraphrase two": ["b"]})
    result = compare_backend(queries, toy, real)
    assert [d.verdict for d in result.document_deltas] == ["both", "real only"]

    flipped = compare_backend(queries, real, toy)
    assert [d.verdict for d in flipped.document_deltas] == ["both", "toy only"]

    neither = compare_backend(
        queries,
        _Store(Backend.VECTOR, {"paraphrase one": ["x"]}),
        _Store(Backend.VECTOR, {"paraphrase one": ["y"]}),
    )
    assert neither.document_deltas[0].verdict == "neither"


def test_recall_gives_partial_credit_where_answers_does_not():
    """A leg can improve recall without answering one more query outright. Both
    numbers are reported because either alone tells half the story."""
    queries = (
        LabeledQuery(
            query_id="q1",
            text="two relevant",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"a", "b"}),
        ),
    )
    toy = _Store(Backend.VECTOR, {"two relevant": ["a", "z"]})
    real = _Store(Backend.VECTOR, {"two relevant": ["a", "b"]})
    result = compare_backend(queries, toy, real)

    assert result.toy_answers == 0 and result.real_answers == 1
    assert result.toy_recall == pytest.approx(0.5)
    assert result.real_recall == pytest.approx(1.0)


def test_order_and_set_agreement_are_separate_columns():
    """Two stores can return the same documents in a different order. That is a
    weaker disagreement than returning different documents and is reported as
    its own column rather than folded in."""
    queries = _queries()[:1]
    toy = _Store(Backend.VECTOR, {"paraphrase one": ["a", "c"]})
    real = _Store(Backend.VECTOR, {"paraphrase one": ["c", "a"]})
    result = compare_backend(queries, toy, real)
    assert result.same_set == 1 and result.same_order == 0


def test_the_detail_block_does_not_truncate_silently():
    queries = tuple(
        LabeledQuery(
            query_id=f"q{i}",
            text=f"query {i}",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({f"doc{i}"}),
        )
        for i in range(8)
    )
    toy = _Store(Backend.VECTOR, {f"query {i}": [f"doc{i}"] for i in range(8)})
    real = _Store(Backend.VECTOR, {f"query {i}": [f"other{i}"] for i in range(8)})
    block = "\n".join(render_detail(compare_backend(queries, toy, real)))
    assert "and 3 more, not truncated silently" in block


def test_every_rendered_line_fits_the_capture_width():
    queries = _queries()
    toy = _Store(
        Backend.VECTOR,
        {"paraphrase one": ["a"], "paraphrase two": ["z"]},
        "vector(hashing-256d-router-v1)",
    )
    real = _Store(
        Backend.VECTOR,
        {"paraphrase one": ["y"], "paraphrase two": ["b"]},
        "pgvector(openai-text-embedding-3-small-1536d, exact scan)",
    )
    result = compare_backend(queries, toy, real)
    lines = (
        render_comparison([result]) + render_detail(result) + render_headline([result])
    )
    wide = [line for line in lines if len(line) > WIDTH]
    assert not wide, f"{len(wide)} lines over {WIDTH} columns: {wide[:2]}"


def test_nothing_compared_claims_nothing():
    assert "nothing is claimed" in "\n".join(render_headline([]))
