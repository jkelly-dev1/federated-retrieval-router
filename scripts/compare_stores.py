#!/usr/bin/env python3
"""How much of this project's retrieval numbers belonged to the stand-ins.

    docker compose up -d
    pip install -r requirements-integration.txt
    python scripts/compare_stores.py

Every offline number in this repository comes from a hand-rolled BM25, a hash
embedder and a Python loop over incident rows. This script runs the SAME 19
labeled queries, the SAME corpus and the SAME ground truth through real
stores, pgvector, Elasticsearch, DuckDB, and reports the difference.

It costs nothing and calls no vendor. Unlike scripts/real_run.py there is no
key, no API and no bill: the stores are local containers or an in-process
engine. What it needs is infrastructure, and it says exactly which piece is
missing rather than quietly comparing fewer legs than it claims.

What it refuses to do. With nothing available it exits 2 instead of printing an
empty table that reads like a finding. With SOME stores available it compares
those and names the ones it skipped, in the output, every time. A partial
comparison labeled as complete is the failure this whole repository is about.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.adapters import (  # noqa: E402
    DuckDBBackend,
    ElasticsearchBackend,
    PgVectorBackend,
)
from router.backends import (  # noqa: E402
    FulltextBackend,
    RelationalBackend,
    VectorBackend,
)
from router.compare import (  # noqa: E402
    compare_backend,
    render_comparison,
    render_detail,
    render_headline,
)
from router.corpus import build_corpus  # noqa: E402
from router.embeddings import HashingEmbedder  # noqa: E402

RULE = "=" * 78
EVAL_K = 5

PG_DSN = os.environ.get(
    "FRR_PG_DSN", "postgresql://postgres:frr-local-only@127.0.0.1:55432/frr"
)
ES_URL = os.environ.get("FRR_ES_URL", "http://127.0.0.1:59200")


def listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_pairs(corpus):
    """Every leg that has both implementations available right now.

    Returns (pairs, skipped). `skipped` is printed rather than swallowed: a
    reader has to be able to tell a leg that agreed from a leg that never ran.
    """
    pairs = []
    skipped = []
    embedder = HashingEmbedder()

    try:
        duck = DuckDBBackend(corpus.documents)
        duck.load()
        pairs.append(("relational", RelationalBackend(corpus.documents), duck))
    except RuntimeError as exc:
        skipped.append(("relational / duckdb", str(exc).splitlines()[0]))

    if listening("127.0.0.1", 55432):
        try:
            pg = PgVectorBackend(corpus.documents, embedder, dsn=PG_DSN)
            pg.load()
            pairs.append(
                ("vector", VectorBackend(corpus.documents, embedder), pg)
            )
        except Exception as exc:  # driver missing, auth, extension absent
            skipped.append(("vector / pgvector", str(exc).splitlines()[0]))
    else:
        skipped.append(("vector / pgvector", "nothing listening on 127.0.0.1:55432"))

    if listening("127.0.0.1", 59200):
        try:
            es = ElasticsearchBackend(corpus.documents, url=ES_URL, analyzer="matching")
            es.load()
            pairs.append(("fulltext", FulltextBackend(corpus.documents), es))
        except Exception as exc:
            skipped.append(("fulltext / elasticsearch", str(exc).splitlines()[0]))
    else:
        skipped.append(
            ("fulltext / elasticsearch", "nothing listening on 127.0.0.1:59200")
        )

    return pairs, skipped


def render(corpus, pairs, skipped, echo=print, warn=None) -> int:
    """Print the comparison, or refuse to, and return the exit code.

    SEPARATED FROM main() so the refusals can be tested. No test imported this
    module at all, and all three of its documented refusals (the no-store
    refusal, its exit code, and the "NOT COMPARED, and therefore not claimed"
    block) could each be deleted with the suite green. With only DuckDB up,
    deleting the third produced a comparison table with no mention of the two
    legs that never ran, which is precisely what this module's own docstring
    calls "a partial comparison labeled as complete".

    The identical refusal in scripts/real_run.py IS tested. Implemented twice,
    covered once.
    """
    if warn is None:
        def warn(msg):
            echo(msg, file=sys.stderr)

    if not pairs:
        warn(
            "No real store is available, so there is nothing to compare and\n"
            "this script will not print a table that looks like a finding.\n"
            "    docker compose up -d\n"
            "    pip install -r requirements-integration.txt"
        )
        for leg, why in skipped:
            warn(f"    {leg}: {why}")
        return 2

    echo(RULE)
    echo("federated-retrieval-router STORE COMPARISON")
    echo(RULE)
    echo(f"  documents / queries                    "
          f"{len(corpus.documents)} / {len(corpus.queries)}")
    echo(f"  legs compared                          {len(pairs)}")
    echo(f"  top-k                                  {EVAL_K}")
    echo("")
    if skipped:
        echo("  NOT COMPARED, and therefore not claimed:")
        for leg, why in skipped:
            echo(f"    {leg} -- {why}")
        echo("")

    comparisons = [
        compare_backend(corpus.queries, toy, real, k=EVAL_K) for _, toy, real in pairs
    ]

    echo(RULE)
    echo("1. The same queries through two implementations of each leg")
    echo(RULE)
    for line in render_comparison(comparisons):
        echo(line)
    echo("")

    for comparison in comparisons:
        for line in render_detail(comparison):
            echo(line)
        echo("")

    echo(RULE)
    echo("2. What belonged to the instrument")
    echo(RULE)
    for line in render_headline(comparisons):
        echo(line)
    echo("")

    echo(RULE)
    echo("What this comparison does NOT claim")
    echo(RULE)
    echo("  - One corpus of 85 synthetic documents and 19 queries. A real")
    echo("    corpus is a different measurement, and a larger one may be a")
    echo("    different result.")
    echo("  - The vector leg is compared with the SAME hash embedder on both")
    echo("    sides on purpose, so the delta is storage and ranking alone. It")
    echo("    is not a claim about a real embedding model; that is what")
    echo("    scripts/real_run.py measures.")
    echo("  - No approximate index is built. pgvector runs an exact scan, so")
    echo("    ANN recall loss is a separate number this does not report.")
    echo("  - No latency figures. Two stores on one laptop, one of them")
    echo("    in-process, would measure the laptop.")
    echo(RULE)
    return 0


def main() -> int:
    corpus = build_corpus()
    pairs, skipped = build_pairs(corpus)
    return render(corpus, pairs, skipped)


if __name__ == "__main__":
    sys.exit(main())
