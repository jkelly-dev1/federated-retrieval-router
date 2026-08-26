"""The gate's VERDICT, as opposed to the gate's checks.

Why this file exists. The gate had one test: run it as a module, assert exit 0
and "GATE PASSED" in the output. Every individual check was carefully built and
none of that mattered, because a gate that always passes satisfies that test
perfectly. Demonstrated by tightening one floor so a real check failed and then
replacing a single line in main():

    failed = [r for r in results if not r.passed]   ->   failed = []

    [FAIL] the heuristic router routes correctly   0.947 (floor 0.99)
    GATE PASSED (9 checks)                         exit 0
    127 passed, 7 skipped

The gate printed FAIL against a check and reported PASSED on the next line, and
the suite was green. Checks were never the weak part. The line that turns
them into an exit code was, and no test could reach it.

Compounding it: nothing else in the suite asserted the heuristic router's
correctness on the real corpus either, so the gate was the sole guard for the
project's headline number, and the gate's verdict was the unpinned line.
Both halves are covered here.
"""
from __future__ import annotations

import pytest

from router.corpus import PRODUCTION_MIX, build_corpus
from router.gate import (
    EVAL_K,
    MIN_ROUTING_CORRECTNESS,
    GateResult,
    render,
    run_checks,
)
from router.backends import build_federation
from router.metrics import score_routing
from router.routing import HeuristicRouter, VectorOnlyRouter, route_all


def _render(results):
    lines: list[str] = []
    code = render(results, echo=lines.append)
    return code, "\n".join(lines)


def test_all_passing_gives_exit_zero_and_says_passed():
    code, out = _render([GateResult("a", True, "ok"), GateResult("b", True, "ok")])
    assert code == 0
    assert "GATE PASSED (2 checks)" in out
    assert "GATE FAILED" not in out


def test_one_failing_check_fails_the_gate():
    """The test the gate did not have. A single failure must produce exit 1."""
    code, out = _render([GateResult("a", True, "ok"), GateResult("b", False, "bad")])
    assert code == 1, "a failing check did not fail the gate"
    assert "GATE FAILED (1 of 2 checks)" in out
    assert "GATE PASSED" not in out


def test_every_failing_check_is_counted_and_printed():
    results = [GateResult(f"c{i}", i % 2 == 0, "d") for i in range(6)]
    code, out = _render(results)
    assert code == 1
    assert "GATE FAILED (3 of 6 checks)" in out
    assert out.count("[FAIL]") == 3
    assert out.count("[PASS]") == 3


def test_a_gate_of_all_failures_cannot_report_passed():
    code, out = _render([GateResult("a", False, "x"), GateResult("b", False, "y")])
    assert code == 1
    assert "GATE PASSED" not in out


@pytest.mark.parametrize("passed", [True, False])
def test_the_verdict_follows_the_checks_and_not_their_number(passed):
    code, _ = _render([GateResult("only", passed, "d")])
    assert code == (0 if passed else 1)


def test_the_real_checks_all_pass_today():
    """Separate from the verdict logic above: the shipped stack is green, and
    the count is pinned so a check cannot quietly stop being run."""
    results = run_checks()
    assert len(results) == 9, f"expected 9 checks, got {len(results)}"
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"gate checks failing: {failed}"


def test_the_headline_routing_correctness_is_pinned_outside_the_gate():
    """The 0.947 the README leads with was checked ONLY by the gate. If the
    gate's verdict is ever neutralized again, this still fails."""
    corpus = build_corpus()
    fed = build_federation(corpus)
    queries = corpus.queries
    heur = score_routing("heuristic", queries, route_all(HeuristicRouter(fed), queries))
    assert heur.correctness == pytest.approx(0.947, abs=0.001), (
        f"heuristic routing correctness moved to {heur.correctness:.3f}")
    assert heur.correctness >= MIN_ROUTING_CORRECTNESS


def test_the_vector_only_baseline_still_loses_outside_the_gate():
    corpus = build_corpus()
    queries = corpus.queries
    vec = score_routing("vector-only", queries, route_all(VectorOnlyRouter(), queries))
    assert vec.correctness < 0.60, (
        f"the single-store baseline stopped losing ({vec.correctness:.3f}); "
        "the corpus has stopped exercising four competences")
