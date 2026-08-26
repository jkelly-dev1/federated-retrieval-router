#!/usr/bin/env python3
"""End-to-end demonstration, offline and deterministic.

Two runs produce byte-identical output. Everything here runs against a hash
embedder and an in-memory corpus; no key is needed and no network call is made.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Python puts the SCRIPT's directory on sys.path, not the working directory, so
# `python scripts/run_demo.py` cannot import the package beside `scripts/`
# without help. The README documents exactly that command.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.backends import build_federation  # noqa: E402
from router.corpus import PRODUCTION_MIX, build_corpus  # noqa: E402
from router.embeddings import content_tokens  # noqa: E402
from router.fusion import (  # noqa: E402
    DEFAULT_WINDOW,
    reciprocal_rank_fusion,
    unique_contributions,
    window_sweep,
)
from router.metrics import (  # noqa: E402
    score_routing,
    validate_competences,
    weighted_correctness,
)
from router.models import Backend  # noqa: E402
from router.routing import (  # noqa: E402
    FanOutRouter,
    HeuristicRouter,
    VectorOnlyRouter,
    route_all,
)

RULE = "=" * 78
THIN = "-" * 78


def header(corpus, fed) -> None:
    print(RULE)
    print("federated-retrieval-router demo")
    print(RULE)
    docs = len(corpus.documents)
    print(f"  documents                              {docs}")
    print(f"  graph edges                            {len(corpus.edges)}")
    print(f"  labeled queries                        {len(corpus.queries)}")
    print(f"  of those, traps                        "
          f"{sum(1 for q in corpus.queries if q.trap)}")
    print(f"  of those, genuinely multi-backend      "
          f"{sum(1 for q in corpus.queries if len(q.required) > 1)}")
    print(f"  top-5 as a fraction of the corpus      {5 / docs:.1%}")
    print()
    print("  THE LAST LINE IS A PRECONDITION, NOT TRIVIA. At 12 documents a")
    print("  top-5 result set is 42% of everything, every backend \"finds\"")
    print("  every answer, and the labels agree with measurement on 1 query")
    print("  out of 11. A retrieval evaluation whose candidate pool is half")
    print("  the corpus measures nothing, and it fails silently.")
    print()
    print("  The corpus is synthetic. Services, teams and incidents are")
    print("  fictional and no real operational data is present.")
    print()


def section_backends(corpus, fed) -> None:
    print(RULE)
    print("1. Four backends, four competences")
    print(RULE)
    for b in fed.all():
        print(f"  {b.backend.value:11s} {b.name()}")
    print()
    probes = (
        ("EXACT_TERM  ", "ERR_UPSTREAM_4423"),
        ("SEMANTIC    ", "how should a caller behave when a dependency starts failing"),
        ("RELATIONSHIP", "which teams own the services checkout depends on"),
        ("AGGREGATE   ", "how many incidents did payments have in 2026"),
    )
    for label, query in probes:
        print(THIN)
        print(f"  {label}  {query}")
        print(THIN)
        for b in fed.all():
            hits = b.search(query, k=2)
            got = ", ".join(f"{h.doc_id}" for h in hits) or "(nothing)"
            print(f"    {b.backend.value:11s} {got}")
        print()
    print("  READ THE AGGREGATE ROW TWICE. The relational leg answers with an")
    print("  `agg:` row, which is a NUMBER THAT APPEARS IN NO DOCUMENT. The")
    print("  other three return documents, confidently, and none of them")
    print("  contains the count. An evaluation scoring that leg by document")
    print("  overlap would report zero for the only backend that answered.")
    print()


def section_routing(corpus, fed) -> None:
    print(RULE)
    print("2. Routing, and what correctness costs")
    print(RULE)
    print(f"  {'router':14s} {'balanced':>9s} {'prod-mix':>9s} {'fan-out':>8s} {'traps':>7s}")
    reports = {}
    for r in (HeuristicRouter(fed), FanOutRouter(), VectorOnlyRouter()):
        decisions = route_all(r, corpus.queries)
        rep = score_routing(r.name, corpus.queries, decisions)
        weighted = weighted_correctness(corpus.queries, decisions, PRODUCTION_MIX)
        reports[r.name] = rep
        print(f"  {rep.router:14s} {rep.correctness:>9.3f} {weighted:>9.3f} "
              f"{rep.fan_out:>8.2f} {rep.trap_correct:>4d}/{rep.trap_total}")
    print()
    print("  FAN-OUT SCORES A PERFECT 1.000 BY CONSTRUCTION and queries every")
    print("  store every time. It is in the table because a correctness number")
    print("  reported without its cost beside it is not a result.")
    print()

    heur = reports["heuristic"]
    print(THIN)
    print("  Per backend, heuristic router")
    print(THIN)
    print(f"  {'backend':12s} {'chosen':>7s} {'needed':>7s} {'prec':>6s} "
          f"{'recall':>7s} {'over':>5s} {'miss':>5s}")
    for b in Backend:
        s = heur.per_backend[b]
        print(f"  {b.value:12s} {s.chosen:>7d} {s.required:>7d} {s.precision:>6.2f} "
              f"{s.recall:>7.2f} {s.over_fires:>5d} {s.misses:>5d}")
    print()
    if heur.failures:
        print("  Where it still misroutes:")
        for qid, req, got in heur.failures:
            q = next(x for x in corpus.queries if x.query_id == qid)
            print(f"    {qid}: needed {sorted(b.value for b in req)}, "
                  f"chose {sorted(b.value for b in got)}")
            text = q.text if len(q.text) <= 66 else q.text[:63] + "..."
            print(f"      {text}")
        print()


def section_mix(corpus, fed) -> None:
    print(RULE)
    print("3. The query mix decides the verdict")
    print(RULE)
    decisions = route_all(VectorOnlyRouter(), corpus.queries)
    rep = score_routing("vector-only", corpus.queries, decisions)
    weighted = weighted_correctness(corpus.queries, decisions, PRODUCTION_MIX)
    print(f"  vector-only on the balanced evaluation set    {rep.correctness:.3f}")
    print(f"  vector-only weighted to a production mix      {weighted:.3f}")
    print(f"  ratio                                         "
          f"{weighted / rep.correctness:.1f}x")
    print()
    print("  assumed production mix:")
    for comp, w in sorted(PRODUCTION_MIX.items(), key=lambda kv: -kv[1]):
        print(f"    {comp.value:14s} {w:.0%}")
    print()
    print("  SAME ROUTER, SAME CORPUS, SAME CODE. The only thing that changed")
    print("  is an assumption about what people ask. A balanced evaluation set")
    print("  is necessary to say anything per backend and it OVERSTATES the")
    print("  case for federation by this ratio, because real query logs are")
    print("  dominated by ordinary semantic lookups.")
    print()
    print("  The weights above are an assumption and not a measurement. They")
    print("  are printed so a reader can substitute their own, and both")
    print("  scorings are reported rather than whichever is more flattering.")
    print()


def section_competence(corpus, fed) -> None:
    print(RULE)
    print("4. Does the backend a question calls for actually win?")
    print(RULE)
    retrieved = {
        q.query_id: {
            b.backend: [h.doc_id for h in b.search(q.text, k=3)] for b in fed.all()
        }
        for q in corpus.queries
    }
    checks = validate_competences(corpus.queries, retrieved, k=3)
    wins = sum(1 for c in checks if c.designed_wins)
    print(f"  queries with a document answer         {len(checks)}")
    print(f"  designed backend actually wins         {wins}")
    print(f"  no backend retrieves the answer        "
          f"{sum(1 for c in checks if c.nobody_wins)}")
    print()
    for c in checks:
        if c.designed_wins:
            continue
        designed = sorted(b.value for b in c.designed)
        measured = sorted(b.value for b in c.measured) or ["NOBODY"]
        print(f"    {c.query_id:12s} designed={designed} measured={measured}")
    print()
    print("  THIS IS A LIMIT OF THE OFFLINE INSTRUMENT AND IT IS REPORTED, NOT")
    print("  HIDDEN. The embedder is a bag-of-tokens hash. On a paraphrase that")
    print("  shares even one RARE term with its target -- 'dependency',")
    print("  'backoff', 'twice' -- BM25's idf carries more signal than the whole")
    print("  vector carries cosine, so the fulltext leg wins queries the vector")
    print("  leg is supposed to own. Measured overlap on those queries is 0.02")
    print("  to 0.10, so they are genuine paraphrases and the win is real.")
    print()
    print("  `required` is therefore a DESIGNED ground truth about question")
    print("  shape, not a claim about which store wins here. Scoring routing")
    print("  against measured winners would define the router's job as")
    print("  predicting the mock, and the project would score well by learning")
    print("  an artifact. Closing this gap is what the real-model run is for.")
    print()


def section_fusion(corpus, fed) -> None:
    print(RULE)
    print("5. Fusion, and the window that silently decides it")
    print(RULE)
    query = "why do we keep seeing ERR_TOKEN_9101 after partner rotations"
    per_backend = {b.backend: b.search(query, k=50) for b in fed.all()}
    print(f"  query: {query}")
    print()
    for backend, hits in per_backend.items():
        top = ", ".join(f"{h.doc_id}#{h.rank}" for h in hits[:2]) or "(nothing)"
        print(f"    {backend.value:11s} {top}")
    print()
    fused = reciprocal_rank_fusion(per_backend, k=5)
    print(f"  {'rank':>4s}  {'document':24s} {'score':>7s}  contributors")
    for hit in fused:
        who = ",".join(b.value[:4] for b in hit.contributors)
        print(f"  {hit.rank:>4d}  {hit.doc_id:24s} {hit.fused_score:>7.4f}  {who}")
    print()
    unique = unique_contributions(fused)
    print("  found by exactly one backend:",
          {b.value: n for b, n in sorted(unique.items(), key=lambda kv: kv[0].value)}
          or "none")
    print()
    print(THIN)
    print("  Window sweep: how deep fusion has to look")
    print(THIN)
    print(f"  {'window':>7s} {'relevant found':>15s} {'fused size':>11s}")
    for w, found, size in window_sweep(per_backend, {"runbook-err-102"}):
        print(f"  {w:>7d} {found:>15d} {size:>11d}")
    print()
    print(f"  DEFAULT_WINDOW is {DEFAULT_WINDOW}. A document one leg ranks deep")
    print("  and another ranks first cannot be fused at all if the window ends")
    print("  above it, and the merged list looks perfectly reasonable while it")
    print("  happens. That is why the window is measured rather than chosen.")
    print()


def main() -> int:
    corpus = build_corpus()
    fed = build_federation(corpus)
    header(corpus, fed)
    section_backends(corpus, fed)
    section_routing(corpus, fed)
    section_mix(corpus, fed)
    section_competence(corpus, fed)
    section_fusion(corpus, fed)
    print(RULE)
    print("What this demo does NOT claim")
    print(RULE)
    print("  - The embedder is a hash model, not a semantic one. Every vector")
    print("    number here is a FLOOR and section 4 measures the shortfall.")
    print("  - 85 synthetic documents. Selectivity is adequate for this query")
    print("    set and says nothing about behavior at corpus scale.")
    print("  - The production mix in section 3 is an assumption, not a")
    print("    measurement from any real query log.")
    print("  - Entity linking for the graph leg is substring matching. Routing")
    print("    is what is measured here; linking is a separate problem.")
    print("  - No latency or cost figures. Fan-out is reported as backends per")
    print("    query, which is the part that does not depend on hardware.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
