"""Reciprocal rank fusion, and the constant that quietly decides its results.

RRF scores a document as the sum over result lists of 1 / (K + rank). It is the
default for hybrid retrieval for a good reason and it has a specific,
measurable weakness, and this file is written so both are visible.

Why rank and not score. A BM25 score of 14.2 and a cosine similarity of 0.83
are not on the same scale, are not comparable across queries, and BM25's range
moves as the corpus grows. Min-max normalizing them is possible and is least
stable exactly when it matters, when one leg returns two results and another
returns fifty. Rank has no units, so it composes.

What RRF throws away, stated plainly: magnitude. A screaming lexical match and
a mediocre one contribute identically if they are adjacent in rank. That is the
price of the scale problem going away, and on the exact-term queries in this
corpus it is visible. The fulltext leg's overwhelming confidence in a runbook
becomes one rank slot.

The window is the constant that matters and it is measured, not chosen. Fusion
can only see as deep into each list as `window` allows. A document a backend
ranks 12th cannot be fused at all at window 10, no matter how strongly a second
backend agrees with it, and the failure is silent, because the merged list
looks perfectly reasonable. `window_sweep` re-derives the value on this corpus
and tests/test_routing.py::test_the_window_sweep_reports_where_recall_appears
asserts the chosen default sits above the point where recall stops improving.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from router.models import Backend, FusedHit, RankedHit

# The RRF rank constant. 60 is the value from the original Cormack et al.
# formulation and the Elasticsearch default; it is a smoothing term that damps
# the difference between rank 1 and rank 2 relative to rank 9 and rank 10. It
# is NOT tuned here, because tuning it on eighteen queries would fit the
# evaluation rather than improve the retriever.
RANK_CONSTANT = 60

# How deep into each backend's list fusion looks. Re-derived by `window_sweep`
# and asserted in tests. Small windows are the classic silent failure: a
# document one leg ranks deep and another ranks first never gets the benefit.
DEFAULT_WINDOW = 20


def reciprocal_rank_fusion(
    per_backend: Mapping[Backend, Sequence[RankedHit]],
    k: int = 10,
    window: int = DEFAULT_WINDOW,
    rank_constant: int = RANK_CONSTANT,
    weights: Mapping[Backend, float] | None = None,
) -> list[FusedHit]:
    """Merge several ranked lists into one, carrying provenance.

    `weights` supports weighted RRF, the usual compromise for RRF discarding
    magnitude. It defaults to uniform, and it is exposed rather than used:
    picking weights is a decision that must be made against a judgment set, and
    this project's judgment set is small enough that tuning them would be
    fitting noise. It exists so the seam is visible.
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    contributors: dict[str, list[Backend]] = defaultdict(list)
    ranks: dict[str, dict[Backend, int]] = defaultdict(dict)

    for backend, hits in per_backend.items():
        weight = weights.get(backend, 1.0)
        for hit in hits[:window]:
            scores[hit.doc_id] += weight / (rank_constant + hit.rank)
            if backend not in contributors[hit.doc_id]:
                contributors[hit.doc_id].append(backend)
            ranks[hit.doc_id][backend] = hit.rank

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedHit(
            doc_id=doc_id,
            fused_score=score,
            rank=i,
            contributors=tuple(sorted(contributors[doc_id], key=lambda b: b.value)),
            per_backend_rank=dict(ranks[doc_id]),
        )
        for i, (doc_id, score) in enumerate(ordered[:k], start=1)
    ]


def window_sweep(
    per_backend: Mapping[Backend, Sequence[RankedHit]],
    relevant: Iterable[str],
    windows: Sequence[int] = (1, 2, 3, 5, 10, 20, 50),
    k: int = 10,
) -> list[tuple[int, int, int]]:
    """(window, relevant found, total fused) for each window.

    The measurement behind DEFAULT_WINDOW. Where the middle column stops
    rising, the window is deep enough; below that point fusion is discarding
    agreement it was never shown.
    """
    relevant = set(relevant)
    out = []
    for w in windows:
        fused = reciprocal_rank_fusion(per_backend, k=k, window=w)
        found = sum(1 for h in fused if h.doc_id in relevant)
        out.append((w, found, len(fused)))
    return out


def unique_contributions(fused: Sequence[FusedHit]) -> dict[Backend, int]:
    """How many fused results each backend found ALONE.

    The number that decides whether a leg earned its cost. A backend whose
    every hit was also found by another backend contributed ranking signal and
    no reach, and on a small corpus that is the common case, which is
    precisely the claim this project exists to test rather than assume.
    """
    counts: dict[Backend, int] = defaultdict(int)
    for hit in fused:
        sole = hit.unique_to
        if sole is not None:
            counts[sole] += 1
    return dict(counts)
