#!/usr/bin/env python3
"""The paid capture: three questions the offline stack cannot answer.

    ENV_FILE=~/.secrets/ai.env python scripts/real_run.py
    STABILITY_RUNS=5 ENV_FILE=~/.secrets/ai.env python scripts/real_run.py

Everything else in this repository runs on a hash embedder and a keyword
router, offline, and the gate never consults a provider. This script is the
exception and it exists for exactly three measurements:

  1. DOES A REAL EMBEDDER RECLAIM ITS OWN QUERIES? Offline, the designed
     backend wins on only 7 of 11 document-bearing queries, because a
     bag-of-tokens model loses a genuine paraphrase to one rare shared term.
     A semantic model should take those back. The corpus is embedded TWICE,
     through two cold caches, because a capture that cannot say whether its
     own numbers reproduce is asserting reproducibility rather than measuring
     it.

  2. DOES A MODEL ROUTE BETTER THAN TWO HAND-WRITTEN GUARDS? Scored on the same
     two axes as every other router here -- correctness AND fan-out -- because
     a model that selects everything has not beaten anything.

  3. AND DOES IT ANSWER THE SAME WAY TWICE? Every router is re-run
     STABILITY_RUNS times, the deterministic ones included, and correctness
     and fan-out are reported as ranges. A capture that runs once publishes a
     point estimate, and in a repository whose whole argument is that a number
     without its spread is not a result, that will not do. See
     router/stability.py.

It refuses to run when nothing real is configured, rather than producing a file
that looks like a paid capture and is not.

Model replies are quoted exactly as returned. Everything else in this
repository is plain ASCII by house rule; a sampled model's own words are the
one exception, because a capture of what a model said is evidence.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.backends import build_federation  # noqa: E402
from router.corpus import PRODUCTION_MIX, build_corpus  # noqa: E402
from router.embeddings import HashingEmbedder  # noqa: E402
from router.metrics import validate_competences  # noqa: E402
from router.providers import CachingEmbedder, build_providers  # noqa: E402
from router.routing import (  # noqa: E402
    FanOutRouter,
    HeuristicRouter,
    VectorOnlyRouter,
    route_all,
)
from router.stability import (  # noqa: E402
    dump_runs,
    ranking_survives,
    render_misroutes,
    render_per_query_table,
    render_summary_table,
    render_unstable_detail,
    render_verdict,
    summarize_runs,
)

RULE = "=" * 78
THIN = "-" * 78
EVAL_K = 3
DEFAULT_RUNS = 5


def stability_runs() -> int:
    """How many independent routing passes. Five by default, and the reason
    five is in router/stability.py: it detects instability and does not rank
    two close models."""
    raw = os.environ.get("STABILITY_RUNS", str(DEFAULT_RUNS))
    try:
        runs = int(raw)
    except ValueError:
        raise SystemExit(f"STABILITY_RUNS={raw!r} is not an integer")
    if runs < 1:
        raise SystemExit("STABILITY_RUNS must be at least 1")
    return runs


def progress(message: str) -> None:
    """Progress goes to stderr so the captured stdout stays paste-ready."""
    print(message, file=sys.stderr, flush=True)


def write_dump(corpus, passes) -> str:
    """Persist every routing decision beside the capture, gitignored.

    The tables in section 2 are a RENDERING of these decisions. Keeping them
    means a later session can re-render the capture with different wording for
    nothing, instead of paying for the routing calls again to change a
    sentence. `audit/` is excluded in .gitignore, so this never ships.
    """
    root = Path(__file__).resolve().parent.parent
    out = root / "audit" / "routing_decisions.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(dump_runs(corpus.queries, passes), indent=1),
        encoding="utf-8",
    )
    return str(out.relative_to(root))


def header(providers, corpus, runs) -> None:
    print(RULE)
    print("federated-retrieval-router REAL MODEL RUN")
    print(RULE)
    print(f"  embedding provider                     {providers.embedding_provider}")
    print(f"  embedder                               {providers.embedder.name()}")
    print(f"  llm routers                            "
          f"{[r.name for r in providers.llm_routers] or 'none'}")
    print(f"  documents / queries                    "
          f"{len(corpus.documents)} / {len(corpus.queries)}")
    print(f"  routing passes per router              {runs}")
    print()


def measure_competences(corpus, embedder):
    """Which backend actually wins each query, under one embedder."""
    fed = build_federation(corpus, embedder)
    retrieved = {
        q.query_id: {
            b.backend: [h.doc_id for h in b.search(q.text, k=EVAL_K)]
            for b in fed.all()
        }
        for q in corpus.queries
    }
    return validate_competences(corpus.queries, retrieved, k=EVAL_K)


def section_competence(corpus, providers) -> dict:
    """The question the mock is structurally unable to answer."""
    print(RULE)
    print("1. Does a real embedder reclaim the queries it is supposed to own?")
    print(RULE)

    results = {}
    for label, embedder in (
        ("mock", CachingEmbedder(HashingEmbedder())),
        ("real", providers.embedder),
    ):
        progress(f"  embedding pass: {label}")
        results[label] = measure_competences(corpus, embedder)

    total = len(results["mock"])
    print(f"  {'embedder':8s} {'dims':>6s} {'designed backend wins':>22s} "
          f"{'nobody retrieves it':>21s}")
    for label in ("mock", "real"):
        emb = CachingEmbedder(HashingEmbedder()) if label == "mock" else providers.embedder
        wins = sum(1 for c in results[label] if c.designed_wins)
        none = sum(1 for c in results[label] if c.nobody_wins)
        print(f"  {label:8s} {emb.dimensions:>6d} {f'{wins}/{total}':>22s} "
              f"{f'{none}/{total}':>21s}")
    print()

    print(f"  {'query':12s} {'mock':>26s} {'real':>26s}")
    for mock, real in zip(results["mock"], results["real"]):
        m = ",".join(sorted(b.value for b in mock.measured)) or "NOBODY"
        r = ",".join(sorted(b.value for b in real.measured)) or "NOBODY"
        flag = " *" if m != r else "  "
        print(f"  {mock.query_id:12s} {m:>26s} {r:>26s}{flag}")
    print()
    print("  Rows marked * are where the embedder changed the answer. The")
    print("  designed backend for every q-sem row is `vector`; offline the")
    print("  fulltext leg takes them, because one rare shared term carries more")
    print("  BM25 signal than the whole bag-of-tokens vector carries cosine.")
    print()
    return results


def section_repeat(corpus, providers, first) -> None:
    """One repeat of the embedding pass, through a second cold cache.

    The whole point of the memo in CachingEmbedder is that a text is embedded
    once, so asking the SAME cached embedder again would measure a dict. This
    builds a second embedder over the same model and pays for the corpus a
    second time -- the cheapest section of the run, and the only one that can
    say whether the numbers above reproduce at all.
    """
    print(RULE)
    print("1b. Do those numbers reproduce? One repeat, cold cache")
    print(RULE)
    progress("  embedding pass: repeat")
    second = providers.new_embedder()
    repeat = measure_competences(corpus, second)

    same_table = [(c.query_id, c.measured) for c in first["real"]] == [
        (c.query_id, c.measured) for c in repeat
    ]
    before = getattr(providers.embedder, "cache", {})
    after = getattr(second, "cache", {})
    shared = sorted(set(before) & set(after))
    identical = sum(1 for text in shared if before[text] == after[text])
    delta = 0.0
    for text in shared:
        for a, b in zip(before[text], after[text]):
            delta = max(delta, abs(a - b))

    print(f"  texts embedded in both passes          {len(shared)}")
    print(f"  bit-identical vectors                  {identical}/{len(shared)}")
    print(f"  largest component difference           {delta:.3e}")
    print(f"  the table in section 1 is              "
          f"{'unchanged' if same_table else 'DIFFERENT'}")
    print()
    print("  This is the reproducibility claim, measured. The routing section")
    print("  below is where a live model is NOT reproducible, and the two are")
    print("  reported separately because they fail differently.")
    print()
    return second


def section_routing(corpus, providers, runs) -> None:
    print(RULE)
    print(f"2. Does a model route better than two hand-written guards? (n={runs})")
    print(RULE)
    fed = build_federation(corpus)
    routers = [
        HeuristicRouter(fed),
        FanOutRouter(),
        VectorOnlyRouter(),
        *providers.llm_routers,
    ]
    # Runs on the OUTSIDE: every router sees each pass at roughly the same
    # moment, so a vendor's mid-capture deployment lands on all of them rather
    # than on whichever one was still running.
    passes: dict[str, list] = {r.name: [] for r in routers}
    for index in range(runs):
        for router in routers:
            progress(f"  routing pass {index + 1}/{runs}: {router.name}")
            passes[router.name].append(route_all(router, corpus.queries))

    stats = [
        summarize_runs(r.name, corpus.queries, passes[r.name], PRODUCTION_MIX)
        for r in routers
    ]
    by_name = {s.router: s for s in stats}
    dump_path = write_dump(corpus, passes)

    for line in render_summary_table(stats):
        print(line)
    print()
    for line in render_per_query_table(stats):
        print(line)
    print()

    for stat in stats:
        if stat.deterministic and not any(
            stat.router == r.name for r in providers.llm_routers
        ):
            continue
        for line in render_unstable_detail(stat, corpus.queries):
            print(line)
        print()

    for router in providers.llm_routers:
        for line in render_misroutes(by_name[router.name], corpus.queries):
            print(line)
        print()

    baseline = by_name["heuristic"]
    verdicts = [
        ranking_survives(by_name[r.name], baseline) for r in providers.llm_routers
    ]
    if verdicts:
        for line in render_verdict(verdicts):
            print(line)
        print()

    print("  READ FAN-OUT BESIDE CORRECTNESS. A model that selects every")
    print("  backend on every question scores a perfect 1.000 and has beaten")
    print("  nothing; it has reinvented the fan-out baseline at a per-query")
    print("  cost in latency and tokens that the keyword router does not pay.")
    print()
    print("  Every decision above was written to")
    print(f"  {dump_path}, so these tables can be re-rendered")
    print("  offline. Changing the wording of a capture is not a reason to buy")
    print("  the routing calls a second time.")
    print()


def section_cost(providers, extra_embedders=()) -> None:
    """Every embedder that billed, including the repeat pass's own.

    The repeat exists to pay for a second copy of the corpus on purpose, so
    leaving it out of the cost section would understate the run by exactly the
    thing the section above is measuring.
    """
    print(RULE)
    print("3. What the run consumed, from the vendors' own usage fields")
    print(RULE)
    embedders = [providers.embedder, *extra_embedders]
    usages = [
        getattr(getattr(e, "inner", None), "usage", None) for e in embedders
    ]
    billed = [u for u in usages if u is not None]
    if billed:
        print(f"  embedding calls billed                 "
              f"{sum(u.calls for u in billed)}")
        print(f"  embedding input tokens                 "
              f"{sum(u.input_tokens for u in billed)}")
        print(f"  cache hits avoided                     "
              f"{sum(e.hits for e in embedders)}")
        print(f"  distinct texts embedded                "
              f"{sum(e.misses for e in embedders)}")
        print(f"  of which the repeat pass                "
              f"{sum(u.calls for u in usages[1:] if u is not None)}")
    else:
        print("  embeddings ran on the mock; nothing was billed")
    for router in providers.llm_routers:
        print(f"  {router.name} calls / in / out          "
              f"{router.usage.calls} / {router.usage.input_tokens} / "
              f"{router.usage.output_tokens}")
    print()


def main() -> int:
    corpus = build_corpus()
    runs = stability_runs()
    providers = build_providers()
    if providers.embedding_provider == "mock" and not providers.llm_routers:
        print(
            "Nothing real is configured, so there is nothing to capture and\n"
            "this script will not produce a file that looks like there was.\n"
            "Point ENV_FILE at a file holding OPENAI_API_KEY and/or\n"
            "ANTHROPIC_API_KEY:\n"
            "    ENV_FILE=~/.secrets/ai.env python scripts/real_run.py",
            file=sys.stderr,
        )
        return 2

    header(providers, corpus, runs)
    repeat_embedders = []
    if providers.embedding_provider == "openai":
        first = section_competence(corpus, providers)
        repeat_embedders.append(section_repeat(corpus, providers, first))
    else:
        print(RULE)
        print("1. Embedding section skipped")
        print(RULE)
        print("  No OpenAI key, so there is no real embedder to compare and the")
        print("  mock's numbers are already in SAMPLE_RUN.md.")
        print()
    section_routing(corpus, providers, runs)
    section_cost(providers, repeat_embedders)

    print(RULE)
    print("What this capture does NOT claim")
    print(RULE)
    print("  - One embedding model, one corpus, 19 queries. Another model's")
    print("    geometry is another measurement.")
    print(f"  - n={runs} DETECTS INSTABILITY AND DOES NOT RANK TWO CLOSE MODELS.")
    print("    Correctness over 19 queries moves in steps of 0.053 and five")
    print("    draws is not a sample to compute a p-value from. None is")
    print("    computed. Overlapping ranges are reported as indistinguishable.")
    print("  - A live model moves between runs. These are captures, not a")
    print("    baseline, and the gate reads none of them.")
    print("  - The routing prompt describes the backends by competence and")
    print("    never names the corpus, its entities or the heuristic's tells.")
    print("    A different prompt is a different result.")
    print("  - No latency figures. Fan-out is the cost axis that does not")
    print("    depend on hardware or on who else is using the API.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
