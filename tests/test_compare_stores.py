"""scripts/compare_stores.py, and the three refusals it documents.

No test imported this module. All three refusals (the no-store refusal, the
exit code that goes with it, and the "NOT COMPARED, and therefore not claimed"
block) could each be deleted with the suite green. Deleting the third, with
only DuckDB running, produced a comparison table that went straight from "legs
compared 1" into the results with no mention of the two legs that never ran.
The module's own docstring calls that out by name:

    "a partial comparison labeled as complete is the failure this whole
     repository is about"

The IDENTICAL refusal in scripts/real_run.py is tested
(test_the_capture_refuses_rather_than_faking_one). Implemented twice, covered
once, which is the shape that keeps recurring in this portfolio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_stores import build_pairs, render          # noqa: E402
from router.corpus import build_corpus                  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


def _capture(corpus, pairs, skipped):
    out, err = [], []
    code = render(corpus, pairs, skipped, echo=out.append, warn=err.append)
    return code, "\n".join(out), "\n".join(err)


def test_no_available_store_refuses_rather_than_printing_a_table(corpus):
    code, out, err = _capture(corpus, [], [("vector / pgvector", "nothing listening")])
    assert code == 2, "a comparison with nothing to compare did not fail"
    assert "STORE COMPARISON" not in out
    assert out.strip() == "", f"it printed a table anyway:\n{out[:300]}"
    assert "nothing to compare" in err
    assert "vector / pgvector" in err, "it refused without saying which leg was missing"


def test_a_partial_comparison_names_every_leg_it_did_not_run(corpus):
    """The one that matters. Some stores up, some down: the ones that did not
    run must appear in the OUTPUT, every time, not only in a log."""
    pairs, skipped = build_pairs(corpus)
    if not pairs:
        pytest.skip("no store available locally, so there is no partial case to check")
    assert skipped, (
        "every store is up, so this environment cannot exercise a partial "
        "comparison; run it with some services stopped")

    code, out, _ = _capture(corpus, pairs, skipped)
    assert code == 0
    assert "NOT COMPARED, and therefore not claimed:" in out
    for leg, why in skipped:
        assert leg in out, f"{leg} was skipped and the output does not say so"
        assert why in out, f"{leg} was skipped without saying why"


def test_the_skipped_block_is_absent_only_when_nothing_was_skipped(corpus):
    pairs, _ = build_pairs(corpus)
    if not pairs:
        pytest.skip("no store available locally")
    code, out, _ = _capture(corpus, pairs, [])
    assert code == 0
    assert "NOT COMPARED" not in out


def test_the_number_of_legs_compared_matches_the_pairs_it_was_given(corpus):
    """A count that does not follow its input is how a partial run reads as a
    whole one."""
    pairs, skipped = build_pairs(corpus)
    if not pairs:
        pytest.skip("no store available locally")
    _, out, _ = _capture(corpus, pairs, skipped)
    assert f"legs compared                          {len(pairs)}" in out
