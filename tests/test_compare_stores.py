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

from compare_stores import build_pairs, endpoint, render  # noqa: E402
from router.backends import RelationalBackend            # noqa: E402
from router.corpus import build_corpus                   # noqa: E402


def _driverless_pair(corpus):
    """A comparison pair that needs no store, no driver and no container.

    `render` and `compare_backend` take anything with the backend protocol, so
    a leg compared against ITSELF exercises every line of the rendering path
    while being available in every environment. The comparison is trivially
    identical, which is fine: these tests are about what the OUTPUT says, not
    about what the stores found.

    A test that built real pairs would skip on the install the README
    documents (`pip install -r requirements.txt`, no optional drivers), and
    deleting the "NOT COMPARED" block from compare_stores.py would then leave
    the whole suite green. A refusal covered only when an optional dependency
    happens to be present is not covered.
    """
    leg = RelationalBackend(corpus.documents)
    return ("relational", leg, leg)


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


def test_a_partial_comparison_names_every_leg_it_did_not_run_without_any_driver(
    corpus,
):
    """The one that matters, and it runs everywhere.

    Some legs compared, some not: the ones that did not run must appear in the
    output every time, not only in a log. Driven with a self-comparison pair
    so it needs no container and no optional driver: deleting the
    `if skipped:` block in compare_stores.py must be red on
    `pip install -r requirements.txt`, which is the install the README
    documents and the one CI's test-and-gate job uses.
    """
    skipped = [
        ("vector / pgvector", "nothing listening on 127.0.0.1:55432"),
        ("fulltext / elasticsearch", "nothing listening on 127.0.0.1:59200"),
    ]
    code, out, _ = _capture(corpus, [_driverless_pair(corpus)], skipped)

    assert code == 0
    assert "STORE COMPARISON" in out, "the table should have been printed"
    assert "NOT COMPARED" in out, (
        "a partial comparison printed a table with no mention of the legs that "
        "never ran; that is the failure this whole repository is about")
    for leg, _why in skipped:
        assert leg in out, f"{leg} was skipped and the output never names it"


def test_the_not_compared_block_is_absent_when_nothing_was_skipped(corpus):
    """The negative half: the block must be conditional, not unconditional.

    A `render` that printed "NOT COMPARED" always would satisfy the test above
    perfectly while telling every reader of a complete comparison that it was
    partial.
    """
    code, out, _ = _capture(corpus, [_driverless_pair(corpus)], [])
    assert code == 0
    assert "STORE COMPARISON" in out
    assert "NOT COMPARED" not in out


def test_the_reachability_probe_follows_the_configured_address(monkeypatch):
    """The probe must test the address the driver will connect to.

    A probe that checked 127.0.0.1 on the default port whatever FRR_PG_DSN and
    FRR_ES_URL said would skip a remote store with a message naming an address
    the operator never configured: honest that it skipped the leg and wrong
    about why, which is the harder of the two to notice.
    """
    assert endpoint("postgresql://u:p@db.internal:6543/frr", 5432) == (
        "db.internal", 6543)
    assert endpoint("http://es.internal:9201", 9200) == ("es.internal", 9201)
    # No port given: fall back to the driver's default rather than to the
    # local compose port, which is what the driver itself would do.
    assert endpoint("http://es.internal", 9200) == ("es.internal", 9200)
    assert endpoint("postgresql://u:p@remote/frr", 5432) == ("remote", 5432)


def test_a_partial_comparison_against_real_stores_when_any_are_available(corpus):
    """The same property against whatever is actually running locally.

    Kept alongside the driverless test rather than replaced by it: this one
    exercises build_pairs, which the other cannot.
    """
    pairs, skipped = build_pairs(corpus)
    if not pairs:
        pytest.skip("no store available locally, so there is no partial case to check")
    if not skipped:
        pytest.skip("every store is up, so there is no partial case here")

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
