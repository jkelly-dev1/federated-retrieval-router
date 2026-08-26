"""Comparing one retrieval leg against a real store implementing the same job.

The question this answers: how much of what this project measured is a property
of retrieval, and how much is a property of the stand-ins it measured retrieval
with? Every offline number in this repository comes from a hand-rolled BM25, a
hash embedder and a Python loop. Each is a reasonable stand-in. "Reasonable
stand-in" is precisely the kind of claim the rest of the project declines to
accept without measuring, so this module measures it.

What is held fixed, and why it matters more than what is compared. The corpus,
the 19 labeled queries, the ground truth and k are identical on both sides. The
ONLY difference is which implementation answered. A comparison that also
changed the embedding model, or the k, or the analyzer, would produce a number
that cannot be attributed to anything, which is how "we moved to a real vector
store and quality improved" gets published without evidence.

Three numbers, because one would hide the interesting case.

  ANSWERS      on how many queries does this leg alone retrieve every relevant
               document within top-k? The strict version, and the one the
               project's competence tables are built on.
  RECALL@K     the partial-credit version. A leg can improve its recall while
               answering no additional query outright, and the reverse.
  AGREEMENT    on how many queries did the two implementations return the SAME
               documents? This is the number people never report, the one that
               says whether a delta of zero means "the store does not matter"
               or "the stores disagree constantly and it happens to come out
               even".

A leg can score identically and agree on nothing. That combination is a real
finding and this module is shaped so it cannot be rounded away into "no
difference".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from router.models import Backend, LabeledQuery

WIDTH = 78


@dataclass(frozen=True)
class QueryDelta:
    """One query, both implementations, what each retrieved."""

    query_id: str
    toy: tuple[str, ...]
    real: tuple[str, ...]
    relevant: frozenset[str]
    for_this_leg: bool = True

    @property
    def document_scored(self) -> bool:
        """Is this query one this leg can be right or wrong about, with a
        document answer to be right or wrong with?

        Two conditions, and the second one was missing at first. A query needs
        a document ground truth, aggregate answers appear in no document, and
        it has to be a query this leg is actually required or sufficient for.
        Scoring the relational leg on a paraphrase question measures it on
        somebody else's job and reports the result as its recall.
        """
        return bool(self.relevant) and self.for_this_leg

    @property
    def both_empty(self) -> bool:
        """Neither implementation returned anything.

        Agreement on nothing is not agreement. Counting these as matches is
        how a comparison that measured nothing reports perfect consistency,
        and the relational leg is where that bites: its answers are computed
        and match no document.
        """
        return not self.toy and not self.real

    @property
    def same_order(self) -> bool:
        return self.toy == self.real

    @property
    def same_set(self) -> bool:
        return set(self.toy) == set(self.real)

    @property
    def toy_answers(self) -> bool:
        return bool(self.relevant) and self.relevant <= set(self.toy)

    @property
    def real_answers(self) -> bool:
        return bool(self.relevant) and self.relevant <= set(self.real)

    @property
    def verdict(self) -> str:
        """Which side answered, in the four states that can happen."""
        if self.toy_answers and self.real_answers:
            return "both"
        if self.real_answers:
            return "real only"
        if self.toy_answers:
            return "toy only"
        return "neither"


def _recall(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved)) / len(relevant)


@dataclass(frozen=True)
class StoreComparison:
    """One leg, two implementations, over the queries that have a document answer."""

    backend: Backend
    toy_name: str
    real_name: str
    deltas: tuple[QueryDelta, ...]

    @property
    def document_deltas(self) -> tuple[QueryDelta, ...]:
        """Queries with a document answer; the only ones recall means
        anything for. Aggregate queries are excluded because their answer
        appears in no document, the same rule metrics.py applies."""
        return tuple(d for d in self.deltas if d.document_scored)

    @property
    def active_deltas(self) -> tuple[QueryDelta, ...]:
        """Queries where at least one implementation returned something.

        The column that must not count silence. Two stores that both return
        nothing have not agreed; they have both declined, usually because the
        question was never theirs. Including those pairs turns an empty
        comparison into a perfect score.
        """
        return tuple(d for d in self.deltas if not d.both_empty)

    @property
    def scored(self) -> int:
        return len(self.document_deltas)

    @property
    def compared(self) -> int:
        return len(self.active_deltas)

    @property
    def vacuous(self) -> bool:
        """Nothing was measured: neither store ever returned anything.

        A comparison in this state supports no conclusion at all, and the
        rendering says so instead of reporting agreement.
        """
        return self.compared == 0

    @property
    def toy_answers(self) -> int:
        return sum(1 for d in self.document_deltas if d.toy_answers)

    @property
    def real_answers(self) -> int:
        return sum(1 for d in self.document_deltas if d.real_answers)

    @property
    def toy_recall(self) -> Optional[float]:
        """None, not 0.0, when there is nothing to score. A zero here reads as
        'the store found nothing' when it means 'this leg does not answer with
        documents'."""
        docs = self.document_deltas
        if not docs:
            return None
        return sum(_recall(d.toy, d.relevant) for d in docs) / len(docs)

    @property
    def real_recall(self) -> Optional[float]:
        docs = self.document_deltas
        if not docs:
            return None
        return sum(_recall(d.real, d.relevant) for d in docs) / len(docs)

    @property
    def same_order(self) -> int:
        return sum(1 for d in self.active_deltas if d.same_order)

    @property
    def same_set(self) -> int:
        return sum(1 for d in self.active_deltas if d.same_set)

    @property
    def gained(self) -> tuple[str, ...]:
        return tuple(d.query_id for d in self.document_deltas if d.verdict == "real only")

    @property
    def lost(self) -> tuple[str, ...]:
        return tuple(d.query_id for d in self.document_deltas if d.verdict == "toy only")

    @property
    def artifact(self) -> int:
        """Net queries whose answer was an artifact of the implementation.

        Signed on purpose. A real store that answers two more and two fewer
        nets to zero and has changed the answer on four queries, which the
        agreement column is there to expose. Reporting only this number is the
        mistake the module docstring is about.
        """
        return len(self.gained) - len(self.lost)


def compare_backend(
    queries: Sequence[LabeledQuery],
    toy,
    real,
    k: int = 5,
) -> StoreComparison:
    """Run the same queries through both implementations of one leg."""
    if toy.backend is not real.backend:
        raise ValueError(
            f"comparing {toy.backend.value} against {real.backend.value} would "
            f"measure two different jobs, not two implementations of one"
        )
    # EVERY query is run through both, not only the document-bearing ones.
    # Filtering to `q.relevant_docs` here compares the relational leg, whose
    # answers are computed and match no document, exclusively on the queries
    # it is not for: it agrees with itself on eleven empty results and reports
    # perfect consistency. Which queries a given number is defined over is
    # decided in StoreComparison, per number.
    deltas = [
        QueryDelta(
            query_id=q.query_id,
            toy=tuple(h.doc_id for h in toy.search(q.text, k=k)),
            real=tuple(h.doc_id for h in real.search(q.text, k=k)),
            relevant=q.relevant_docs,
            for_this_leg=toy.backend in q.acceptable,
        )
        for q in queries
    ]
    return StoreComparison(
        backend=toy.backend,
        toy_name=toy.name(),
        real_name=real.name(),
        deltas=tuple(deltas),
    )


# ------------------------------------------------------------------ rendering


def render_comparison(comparisons: Sequence[StoreComparison]) -> list[str]:
    """The table, built here so its width is checked without a container."""
    lines = [
        "  Same corpus, same 19 labeled queries, same ground truth, same k.",
        "  The only thing that differs between the two columns is which",
        "  implementation answered.",
        "",
        "  ANSWERS AND RECALL ARE OVER THE QUERIES WITH A DOCUMENT ANSWER;",
        "  n/a means this leg answers with computed values that match no",
        "  document, so document overlap is a category error rather than a",
        "  zero. AGREEMENT IS OVER THE QUERIES WHERE EITHER STORE RETURNED",
        "  ANYTHING -- two stores that both returned nothing have not agreed.",
        "",
        "  "
        + "leg".ljust(11)
        + "answers".rjust(14)
        + "recall@k".rjust(18)
        + "same set".rjust(11)
        + "same order".rjust(12),
    ]
    for c in comparisons:
        if c.scored:
            answers = f"{c.toy_answers} -> {c.real_answers} /{c.scored}"
            recall = f"{c.toy_recall:.3f} -> {c.real_recall:.3f}"
        else:
            answers = "n/a"
            recall = "n/a (computed)"
        agreement = (
            (f"{c.same_set}/{c.compared}", f"{c.same_order}/{c.compared}")
            if c.compared
            else ("NOTHING", "NOTHING")
        )
        lines.append(
            "  "
            + c.backend.value.ljust(11)
            + answers.rjust(14)
            + recall.rjust(18)
            + agreement[0].rjust(11)
            + agreement[1].rjust(12)
        )
    return lines


def render_detail(comparison: StoreComparison) -> list[str]:
    """Which queries changed hands, named rather than counted."""
    lines = [
        "-" * WIDTH,
        f"  {comparison.backend.value}: {comparison.toy_name}",
        f"  {' ' * len(comparison.backend.value)}  vs {comparison.real_name}",
        "-" * WIDTH,
    ]
    if comparison.vacuous:
        lines.append("    NEITHER STORE RETURNED ANYTHING ON ANY QUERY.")
        lines.append("    Nothing was compared, so nothing is claimed about this leg.")
        return lines
    if comparison.gained:
        lines.append(f"    only the real store answers: {', '.join(comparison.gained)}")
    if comparison.lost:
        lines.append(f"    only the toy store answers:  {', '.join(comparison.lost)}")
    if comparison.scored and not comparison.gained and not comparison.lost:
        lines.append("    no query changed hands")
    if not comparison.scored:
        lines.append(
            "    no query here has a document answer, so 'changed hands' is"
        )
        lines.append(
            "    undefined; the agreement column below is the whole comparison"
        )
    disagreed = [d for d in comparison.active_deltas if not d.same_set]
    lines.append(
        f"    returned a different set on {len(disagreed)} of "
        f"{comparison.compared} queries where either store answered"
    )
    for delta in disagreed[:5]:
        lines.append(f"      {delta.query_id}  ({delta.verdict})")
        lines.append(f"        toy   {', '.join(delta.toy) or 'NOTHING'}")
        lines.append(f"        real  {', '.join(delta.real) or 'NOTHING'}")
    if len(disagreed) > 5:
        lines.append(f"      ... and {len(disagreed) - 5} more, not truncated silently")
    return lines


def render_headline(comparisons: Sequence[StoreComparison]) -> list[str]:
    """The sentence the whole A/B exists to make available."""
    if not comparisons:
        return ["  Nothing was compared, so nothing is claimed."]
    if all(c.vacuous for c in comparisons):
        return [
            "  NOTHING WAS MEASURED. On every query, neither implementation",
            "  returned anything, so there is no agreement to report and no",
            "  conclusion to draw. This is printed instead of a table of",
            "  perfect scores, which is what an empty comparison looks like",
            "  if silence is counted as agreement.",
        ]
    net = sum(c.artifact for c in comparisons)
    changed = sum(c.compared - c.same_set for c in comparisons)
    total = sum(c.compared for c in comparisons)
    document_slots = sum(c.scored for c in comparisons)
    lines = [
        f"  ACROSS {len(comparisons)} LEG(S):",
        f"    queries where either store answered      {total}",
        f"    of those, with a document ground truth   {document_slots}",
        f"    net queries whose answer changed hands   {net:+d}",
        f"    queries where the two stores disagreed   {changed}/{total}",
        "",
    ]
    if not document_slots:
        lines.extend([
            "  NO LEG HERE HAS A DOCUMENT GROUND TRUTH, so 'changed hands' is",
            "  structurally zero and means nothing. The one number that does",
            "  mean something is the disagreement count above: it is over the",
            "  queries these stores actually answer, and it is the whole",
            "  finding for a leg whose answers are computed rather than",
            "  retrieved.",
        ])
        return lines
    if net == 0 and changed:
        lines.extend([
            "  READ THOSE TWO LINES TOGETHER. The net is zero and the stores",
            "  still disagree on the documents they return. 'No difference in",
            "  the score' is not 'no difference', and a project that reported",
            "  only the first number would have published the wrong sentence.",
        ])
    elif net == 0:
        lines.extend([
            "  The stores agree on the answer AND on the documents. For this",
            "  corpus at this size, the stand-in was not costing anything --",
            "  which is a measurement, not an assumption, for the first time.",
        ])
    else:
        lines.extend([
            "  THAT NET IS THE PART OF THIS PROJECT'S RETRIEVAL NUMBERS THAT",
            "  BELONGED TO THE INSTRUMENT RATHER THAN TO RETRIEVAL. It is",
            "  reported signed and per leg because an average over legs would",
            "  let one store's gain hide another's loss.",
        ])
    return lines
