"""The CI gate. It protects the MEASUREMENT, not a product.

It fails in both directions. Failing when routing regresses is obvious.
Failing when EVERYTHING PASSES is the part that matters: if the vector-only
baseline stops losing, or the traps stop trapping, or the corpus stops being
big enough to discriminate, then the evaluation has stopped measuring and
every subsequent green run is worthless.

Two checks exist because of defects this project already had, and they are the
most valuable ones here:

  The selectivity check. The first corpus was 12 documents, so a top-5 result
  set held 42% of it and every backend "found" every answer. The labels agreed
  with measurement on 1 of 11 queries and the whole evaluation was noise. It
  now fails if k/N rises above a threshold, because a corpus that stops
  discriminating does so silently.

  The competence-gap check. On the offline embedder the fulltext leg beats the
  vector leg on genuine paraphrases. That is a known, documented limit of the
  mock, and it is asserted rather than merely described, if it ever quietly
  disappears, either the embedder changed or the queries stopped being
  paraphrases, and both are things a maintainer must be told about.

The gate always runs on the deterministic offline stack. A live embedding model
moves the numbers between runs, so real-model behavior belongs in SAMPLE_RUN.md
and never in an exit code.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from router.backends import build_federation
from router.corpus import PRODUCTION_MIX, build_corpus
from router.fusion import DEFAULT_WINDOW, reciprocal_rank_fusion, window_sweep
from router.metrics import (
    score_routing,
    validate_competences,
    weighted_correctness,
)
from router.models import Backend
from router.routing import (
    FanOutRouter,
    HeuristicRouter,
    VectorOnlyRouter,
    route_all,
)

# The heuristic router must stay at or above this on the balanced set. Set
# below the measured value so ordinary drift does not flap the gate, and high
# enough that a real regression trips it.
MIN_ROUTING_CORRECTNESS = 0.90
# The vector-only baseline must stay BELOW this. If a single store starts
# answering everything, the corpus has stopped exercising four competences and
# the project has no subject.
MAX_VECTOR_ONLY_CORRECTNESS = 0.60
# Top-k as a fraction of the corpus. Above this, retrieval stops discriminating
# and every label agrees with every backend. See the module docstring.
MAX_SELECTIVITY = 0.15
EVAL_K = 5


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def run_checks() -> list[GateResult]:
    results: list[GateResult] = []
    corpus = build_corpus()
    fed = build_federation(corpus)
    queries = corpus.queries

    # 1. The corpus must be able to discriminate at all.
    selectivity = EVAL_K / len(corpus.documents)
    results.append(
        GateResult(
            "the corpus is selective enough to measure anything",
            selectivity <= MAX_SELECTIVITY,
            f"top-{EVAL_K} is {selectivity:.1%} of {len(corpus.documents)} docs "
            f"(ceiling {MAX_SELECTIVITY:.0%})",
        )
    )

    # 2. Every required backend must actually return something. A label naming
    #    a backend that answers nothing is a broken label, not a hard query.
    broken = [
        f"{q.query_id}:{b.value}"
        for q in queries
        for b in q.required
        if not fed.get(b).search(q.text, k=EVAL_K)
    ]
    results.append(
        GateResult(
            "every required backend returns results for its queries",
            not broken,
            "all labels answerable" if not broken else f"broken: {broken}",
        )
    )

    # 3. Routing correctness, and the two baselines that give it meaning.
    heur = score_routing("heuristic", queries, route_all(HeuristicRouter(fed), queries))
    fan = score_routing("fan-out", queries, route_all(FanOutRouter(), queries))
    vec = score_routing("vector-only", queries, route_all(VectorOnlyRouter(), queries))

    results.append(
        GateResult(
            "the heuristic router routes correctly",
            heur.correctness >= MIN_ROUTING_CORRECTNESS,
            f"{heur.correctness:.3f} (floor {MIN_ROUTING_CORRECTNESS})",
        )
    )
    results.append(
        GateResult(
            "a single store still fails the other three competences",
            vec.correctness <= MAX_VECTOR_ONLY_CORRECTNESS,
            f"vector-only {vec.correctness:.3f} (ceiling {MAX_VECTOR_ONLY_CORRECTNESS})",
        )
    )
    results.append(
        GateResult(
            "correctness is not free: fan-out is perfect and costs the most",
            fan.correctness == 1.0 and fan.fan_out > heur.fan_out,
            f"fan-out {fan.correctness:.3f} at {fan.fan_out:.2f} backends/query "
            f"vs heuristic {heur.correctness:.3f} at {heur.fan_out:.2f}",
        )
    )
    results.append(
        GateResult(
            "the traps still trap a single-store baseline",
            vec.trap_correct < vec.trap_total and heur.trap_correct == heur.trap_total,
            f"vector-only {vec.trap_correct}/{vec.trap_total}, "
            f"heuristic {heur.trap_correct}/{heur.trap_total}",
        )
    )

    # 4. The query mix must still change the answer. If it stops mattering the
    #    balanced set has drifted toward the production mix and the project has
    #    lost its most transferable finding.
    vec_prod = weighted_correctness(
        queries, route_all(VectorOnlyRouter(), queries), PRODUCTION_MIX
    )
    swing = vec_prod / vec.correctness if vec.correctness else 0.0
    results.append(
        GateResult(
            "the query mix still changes the verdict",
            swing >= 1.5,
            f"vector-only {vec.correctness:.3f} balanced vs {vec_prod:.3f} "
            f"prod-mix, a {swing:.1f}x swing",
        )
    )

    # 5. The known competence gap must still be present and still be a gap.
    retrieved = {
        q.query_id: {
            b.backend: [h.doc_id for h in b.search(q.text, k=3)] for b in fed.all()
        }
        for q in queries
    }
    checks = validate_competences(queries, retrieved, k=3)
    wins = sum(1 for c in checks if c.designed_wins)
    results.append(
        GateResult(
            "the offline embedder still loses paraphrases to BM25",
            0 < wins < len(checks),
            f"designed backend wins {wins}/{len(checks)}; a clean sweep would "
            f"mean the mock got semantic or the queries stopped being paraphrases",
        )
    )

    # 6. The fusion window must sit at or above where recall stops improving.
    per_backend = {
        b.backend: b.search("ERR_UPSTREAM_4423 settlement envelope", k=50)
        for b in fed.all()
    }
    sweep = window_sweep(per_backend, {"runbook-err-101"})
    plateau = next((w for w, found, _ in sweep if found > 0), None)
    results.append(
        GateResult(
            "the fusion window is deep enough to fuse what the legs return",
            plateau is not None and DEFAULT_WINDOW >= plateau,
            f"recall appears at window {plateau}, default is {DEFAULT_WINDOW}",
        )
    )

    return results


def render(results: list[GateResult], echo=print) -> int:
    """Print the verdict and return the exit code that goes with it.

    SEPARATED FROM main() so the verdict itself can be tested. The gate's only
    test drove the real checks and asserted exit 0 and "GATE PASSED" in the
    output, which a gate that always passes satisfies perfectly. Replacing the
    one line that collects failures with an empty list printed [FAIL] against
    a real check and then GATE PASSED on the next line, exit 0, with the whole
    suite green. The checks were never the weak part; the line that turns them
    into an exit code was, and nothing could reach it.
    """
    width = max(len(r.name) for r in results)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        echo(f"  [{flag}] {r.name:{width}s}  {r.detail}")
    failed = [r for r in results if not r.passed]
    echo("")
    if failed:
        echo(f"GATE FAILED ({len(failed)} of {len(results)} checks)")
        return 1
    echo(f"GATE PASSED ({len(results)} checks)")
    echo("  routing is still measurable, a single store still loses, and the")
    echo("  corpus can still tell one competence from another.")
    return 0


def main() -> int:
    print("=" * 78)
    print("federated-retrieval-router gate: deterministic mock, offline")
    print("=" * 78)
    return render(run_checks())


if __name__ == "__main__":
    sys.exit(main())
