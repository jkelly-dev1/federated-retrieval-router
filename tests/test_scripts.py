"""The commands the README documents, run the way the README documents them.

Python puts the SCRIPT's directory on `sys.path`, not the working directory, so
`python scripts/run_demo.py` cannot import the package beside `scripts/`. An
in-process suite never catches that, because pytest supplies its own path.
These run as SUBPROCESSES with a clean environment, which is the only way to
reproduce what a reviewer actually types.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _run(args, cwd=ROOT):
    return subprocess.run(
        args, cwd=cwd, env=_clean_env(), capture_output=True, text=True, timeout=180
    )


def test_the_documented_demo_command_runs():
    result = _run([sys.executable, "scripts/run_demo.py"])
    assert result.returncode == 0, result.stderr
    assert "federated-retrieval-router demo" in result.stdout


def test_the_demo_runs_from_any_working_directory(tmp_path):
    """Mutation check for the sys.path bootstrap."""
    result = _run([sys.executable, str(ROOT / "scripts" / "run_demo.py")], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_the_gate_is_runnable_as_a_module():
    result = _run([sys.executable, "-m", "router.gate"])
    assert result.returncode == 0, result.stderr
    assert "GATE PASSED" in result.stdout


def test_the_demo_is_byte_identical_across_runs():
    a = _run([sys.executable, "scripts/run_demo.py"])
    b = _run([sys.executable, "scripts/run_demo.py"])
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_the_demo_prints_the_feature_that_fired_for_every_misroute():
    """The README says misroutes are printed in the demo "with the feature
    that fired", so the demo's output must carry `decision.rationale` for
    each one. Without this test, removing the two-line loop that prints it
    would leave the whole suite green.

    The expected text is derived from the router instead of typed in here, so
    the test cannot drift into asserting a sentence the router stopped
    producing. A hardcoded "-> 9 prose token(s)" would keep passing after the
    heuristic changed and would then be pinning a string, not the behavior.
    """
    from router.backends import build_federation
    from router.corpus import build_corpus
    from router.metrics import score_routing
    from router.routing import HeuristicRouter, route_all

    corpus = build_corpus()
    fed = build_federation(corpus)
    router = HeuristicRouter(fed)
    decisions = route_all(router, corpus.queries)
    report = score_routing("heuristic", corpus.queries, decisions)
    assert report.failures, (
        "the heuristic router stopped misrouting anything, so this test can no "
        "longer check that a misroute is explained")

    result = _run([sys.executable, "scripts/run_demo.py"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    by_id = {d.query_id: d for d in decisions}

    for query_id, _required, _chosen in report.failures:
        assert query_id in out, f"{query_id} misroutes and the demo never names it"
        rationale = by_id[query_id].rationale
        assert rationale, f"{query_id} has no rationale to print"
        # EVERY reason the router gave must reach the output, not just one.
        # Printing only rationale[0] would satisfy a laxer assertion while
        # hiding the guard that actually suppressed a leg.
        for line in rationale:
            assert line in out, (
                f"{query_id} misroutes because {line!r} and the demo does not "
                f"print it; README:41-43 claims the feature that fired is shown")


def test_the_demo_output_fits_eighty_columns():
    """A capture that wraps in a terminal or on GitHub is unreadable, and
    SAMPLE_RUN.md is pasted verbatim."""
    result = _run([sys.executable, "scripts/run_demo.py"])
    wide = [l for l in result.stdout.splitlines() if len(l) > 80]
    assert not wide, f"{len(wide)} lines over 80 columns: {wide[:2]}"


def test_the_documented_test_counts_add_up_to_the_suite():
    """The README's and SAMPLE_RUN.md's suite counts cover every test.

    Which tests pass and which skip depends on the drivers installed, but
    passed plus skipped is the size of the suite in every environment. Adding
    a test without updating the prose breaks that sum.
    """
    import re

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    total = int(re.search(r"(\d+) tests? collected", collected.stdout).group(1))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    passed, skipped = map(int, re.search(
        r"pytest -q\s+# (\d+) passed, (\d+) skipped", readme).groups())
    assert passed + skipped == total, (
        f"README says {passed} passed + {skipped} skipped, the suite has {total}")

    sample = (ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8")
    counts = re.findall(r"(\d+) passed, (\d+) skipped", sample)
    assert counts, "SAMPLE_RUN.md states no suite count"
    for p, s in counts:
        assert int(p) + int(s) == total, (
            f"SAMPLE_RUN.md says {p} passed + {s} skipped, the suite has {total}")
