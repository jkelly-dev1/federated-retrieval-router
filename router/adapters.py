"""Real stores behind the same protocol the hand-rolled ones implement.

Why this file exists, and what it is for. Every other backend in this
repository is written out in Python so the comparison between competences is
legible. That is good for reading and useless for one specific question:

    How much of what this project measured is an artifact of the instrument?

The hand-rolled BM25 has textbook constants and a tokenizer this repository
wrote. Its hash embedder is a bag of tokens. The relational leg computes its
aggregates in a Python loop. Each of those is a plausible stand-in for a real
store, and "plausible stand-in" is exactly the claim the rest of the project
refuses to accept without measuring. So these adapters put pgvector,
Elasticsearch and DuckDB behind the same `RetrievalBackend` protocol, run the
same 19 labeled queries through them, and report the delta.

The offline path remains the default and stays dependency-free. `pip install
-r requirements.txt && pytest` installs nothing from this file's imports and
never touches a network. Every driver import happens inside a constructor,
behind an injected-client escape hatch, and a missing driver produces a
sentence naming the package rather than a traceback: the same shape as
router/providers.py, and for the same reason. A sibling repository shipped
provider tests that constructed live clients, passed on the maintainer's
machine, and failed on all three Python versions in CI.

What an a/b between two stores can and cannot separate. Swapping the
hand-rolled BM25 for Elasticsearch changes the scoring implementation AND the
tokenizer at once, and a delta that mixes the two says nothing about either.
`ElasticsearchBackend` therefore takes an `analyzer` argument with two
settings. One mirroring this repository's own tokenizer, one Elasticsearch's
default English analysis, so the tokenizer's contribution can be measured
instead of assumed. The scoring constants are already identical: BM25 defaults
to k1=1.2, b=0.75 in both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from router.backends import DEFAULT_K, _rank
# _TOKEN is private and imported anyway, on purpose: the mapping must not hold
# a COPY of the pattern that can drift from the one the in-memory leg uses. A
# copied string is what a test can only check for equality after the fact; a
# shared object cannot diverge at all.
from router.embeddings import STOPWORDS, _TOKEN, Embedder, content_tokens
from router.models import Backend, Document, RankedHit

# One table/index name per store. Constants rather than parameters: these
# adapters own their schema and a caller pointing them at an existing
# production index would be measuring that index, not this corpus.
PG_TABLE = "frr_documents"
ES_INDEX = "frr-documents"
DUCKDB_TABLE = "frr_incidents"


def _missing(package: str, extra: str) -> RuntimeError:
    return RuntimeError(
        f"This adapter needs the '{package}' package, which is NOT in "
        f"requirements.txt -- the offline suite installs nothing that can "
        f"reach a network.\n"
        f"    pip install -r requirements-integration.txt   # or: pip install {extra}"
    )


# ------------------------------------------------------------------ pgvector


@dataclass
class PgVectorBackend:
    """Dense retrieval in Postgres, replacing the in-memory cosine loop.

    The most likely place the hand-rolled version is wrong, and not for the
    reason people expect. The toy leg scores every document with an exact
    cosine and sorts; pgvector with an approximate index does not promise the
    same top-k. So `probes`/index choice is part of the measurement rather than
    a deployment detail, and this adapter defaults to an EXACT scan (no ANN
    index) precisely so the first A/B isolates storage from approximation. Add
    the index and re-run to measure recall loss as a separate number.

    `embedder` is injected and is whatever the caller is comparing against:
    the same hash embedder for a pure storage comparison, a real one for a
    combined comparison. Mixing those two in one run is how a delta stops
    meaning anything.
    """

    documents: tuple[Document, ...]
    embedder: Embedder
    connection: Optional[Any] = None
    dsn: Optional[str] = None
    table: str = PG_TABLE
    backend: Backend = Backend.VECTOR
    _indexed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.connection is not None:
            return
        # Configuration is checked before the driver, deliberately. "You did
        # not say where to connect" is true whether or not psycopg is
        # installed, and installing psycopg would not fix it. Reporting the
        # import failure first sends the reader to the wrong problem, and it
        # also makes the error depend on what happens to be on the machine.
        if not self.dsn:
            raise ValueError("PgVectorBackend needs a dsn or an injected connection")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing("psycopg", "'psycopg[binary]' pgvector") from exc
        self.connection = psycopg.connect(self.dsn)

    def name(self) -> str:
        return f"pgvector({self.embedder.name()}, exact scan)"

    # -- schema and load ---------------------------------------------------

    def schema_statements(self) -> list[str]:
        """The DDL, as strings, so a test can read it without a server."""
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"DROP TABLE IF EXISTS {self.table}",
            f"CREATE TABLE {self.table} ("
            "doc_id TEXT PRIMARY KEY, "
            "title TEXT NOT NULL, "
            "body TEXT NOT NULL, "
            f"embedding vector({self.embedder.dimensions}) NOT NULL)",
        ]

    def load(self) -> int:
        """Create the table and insert one row per document. Returns the count.

        Embedding happens HERE, on the client, exactly as the in-memory leg
        does it; the store is what changes in the A/B, not the vectors.
        """
        with self.connection.cursor() as cur:
            for statement in self.schema_statements():
                cur.execute(statement)
            for doc in self.documents:
                vector = self.embedder.embed(f"{doc.title} {doc.text}")
                cur.execute(
                    f"INSERT INTO {self.table} (doc_id, title, body, embedding) "
                    "VALUES (%s, %s, %s, %s)",
                    (doc.doc_id, doc.title, doc.text, _pg_vector(vector)),
                )
        self.connection.commit()
        self._indexed = True
        return len(self.documents)

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        """Cosine similarity, computed by the database.

        `<=>` is pgvector's cosine DISTANCE, so similarity is 1 - distance.
        Getting that backwards produces a ranking that looks plausible and is
        exactly inverted, which no unit test of the SQL string would catch:
        tests/test_adapters.py pins the arithmetic on a fake cursor.
        """
        vector = _pg_vector(self.embedder.embed(query))
        with self.connection.cursor() as cur:
            cur.execute(
                f"SELECT doc_id, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, k),
            )
            rows = cur.fetchall()
        return _rank(
            [(doc_id, float(score), "cosine (pgvector)") for doc_id, score in rows],
            self.backend,
            k,
        )


def _pg_vector(values: Sequence[float]) -> str:
    """pgvector's text input format. Kept separate so a test can read it."""
    return "[" + ",".join(f"{v:.9g}" for v in values) + "]"


# ------------------------------------------------------------- elasticsearch


# The analyzer that mirrors router/embeddings.py: lowercase, this repository's
# own token pattern, and its short stop list. It exists so the A/B can separate
# "a real BM25 scores differently" from "a real analyzer tokenizes
# differently": two changes that arrive together when you simply swap the
# store.
MATCHING_ANALYZER = {
    "settings": {
        "analysis": {
            "filter": {
                "frr_stops": {
                    "type": "stop",
                    "stopwords": sorted(STOPWORDS),
                }
            },
            "tokenizer": {
                "frr_pattern": {
                    "type": "pattern",
                    # Mirrors _TOKEN in router/embeddings.py, including the
                    # rule that a dot may sit INSIDE an identifier and never at
                    # the end. That was the defect that made
                    # `queue.prefetch.count` unfindable when it appeared at the
                    # end of a sentence.
                    "pattern": _TOKEN.pattern,
                    "group": 0,
                    # CASE_INSENSITIVE is required and the pattern is wrong
                    # without it. Router/embeddings.py lowercases and THEN
                    # matches: `_TOKEN.findall(text.lower())`. Elasticsearch
                    # runs the tokenizer BEFORE the lowercase filter, so a
                    # lowercase-only character class treats every capital as a
                    # DELIMITER: "Checkout Service" tokenized to ["heckout",
                    # "ervice"] and "ERR_UPSTREAM_4423" to ["_", "_4423"]. Same
                    # regex source, opposite pipeline order, silently different
                    # tokens.
                    "flags": "CASE_INSENSITIVE",
                }
            },
            "analyzer": {
                "frr_matching": {
                    "type": "custom",
                    "tokenizer": "frr_pattern",
                    "filter": ["lowercase", "frr_stops"],
                }
            },
        }
    }
}


@dataclass
class ElasticsearchBackend:
    """Real BM25, against a hand-rolled BM25 that already had one defect.

    The hand-rolled leg's tokenizer absorbed a sentence-ending period and made
    identifier lookups silently return nothing (bug log, defect 2). That is the
    class of mistake a real analyzer does not make, which is the honest reason
    to measure against one.

    `analyzer` selects what is being compared:
      "matching"  mirrors this repository's tokenizer, so the delta is scoring
                  implementation alone.
      "english"   Elasticsearch's own English analysis (stemming, its own
                  stop list), so the delta is scoring AND tokenization.
    Running both is the only way to attribute the difference to either.
    """

    documents: tuple[Document, ...]
    client: Optional[Any] = None
    url: Optional[str] = None
    index: str = ES_INDEX
    analyzer: str = "matching"
    backend: Backend = Backend.FULLTEXT

    def __post_init__(self) -> None:
        if self.analyzer not in {"matching", "english"}:
            raise ValueError(
                f"analyzer must be 'matching' or 'english', not {self.analyzer!r}; "
                f"the two answer different questions and the capture says which"
            )
        if self.client is not None:
            return
        if not self.url:  # configuration before driver: see PgVectorBackend
            raise ValueError("ElasticsearchBackend needs a url or an injected client")
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing("elasticsearch", "elasticsearch") from exc
        self.client = Elasticsearch(self.url)

    def name(self) -> str:
        return f"elasticsearch(bm25, {self.analyzer} analyzer)"

    def index_body(self) -> dict:
        """The create-index request, readable without a server."""
        text_field = (
            {"type": "text", "analyzer": "frr_matching"}
            if self.analyzer == "matching"
            else {"type": "text", "analyzer": "english"}
        )
        body: dict = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": text_field,
                    "body": text_field,
                }
            }
        }
        if self.analyzer == "matching":
            body.update(MATCHING_ANALYZER)
        return body

    def load(self) -> int:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        self.client.indices.create(index=self.index, body=self.index_body())
        for doc in self.documents:
            self.client.index(
                index=self.index,
                id=doc.doc_id,
                document={"doc_id": doc.doc_id, "title": doc.title, "body": doc.text},
            )
        self.client.indices.refresh(index=self.index)
        return len(self.documents)

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        """One multi_match over title and body.

        Title and body are searched together because the in-memory leg indexes
        `f"{title} {text}"` as one string. Boosting the title here would be a
        second change riding along with the store swap.
        """
        response = self.client.search(
            index=self.index,
            query={"multi_match": {"query": query, "fields": ["title", "body"]}},
            size=k,
        )
        hits = response["hits"]["hits"]
        return _rank(
            [
                (hit["_id"], float(hit["_score"]), "bm25 (elasticsearch)")
                for hit in hits
            ],
            self.backend,
            k,
        )


# -------------------------------------------------------------------- duckdb


@dataclass
class DuckDBBackend:
    """A real SQL engine for the leg whose answers are computed, not retrieved.

    The expectation this is built to test, stated before the run: swapping a
    Python loop for a SQL engine should change NOTHING about the answers, and
    if that holds it is a result rather than an anticlimax. Relational work
    was never where the fudge was. The fudge is in the phrase-matching that
    decides WHICH aggregate to compute, and that layer is unchanged here on
    purpose. Moving it too would confound the store swap with a parser swap.

    So this adapter reuses the same surface-form sniffing as the in-memory leg
    and differs only in who executes the arithmetic.
    """

    documents: tuple[Document, ...]
    connection: Optional[Any] = None
    table: str = DUCKDB_TABLE
    backend: Backend = Backend.RELATIONAL

    def __post_init__(self) -> None:
        if self.connection is not None:
            return
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise _missing("duckdb", "duckdb") from exc
        self.connection = duckdb.connect(":memory:")

    def name(self) -> str:
        return f"duckdb({len(self.rows())} rows, in-process)"

    def rows(self) -> tuple[Document, ...]:
        return tuple(d for d in self.documents if d.kind == "incident")

    def load(self) -> int:
        self.connection.execute(f"DROP TABLE IF EXISTS {self.table}")
        self.connection.execute(
            f"CREATE TABLE {self.table} ("
            "doc_id VARCHAR, service VARCHAR, quarter VARCHAR, "
            "minutes_to_resolve INTEGER)"
        )
        rows = self.rows()
        for row in rows:
            self.connection.execute(
                f"INSERT INTO {self.table} VALUES (?, ?, ?, ?)",
                [row.doc_id, row.service, row.quarter, row.minutes_to_resolve or 0],
            )
        return len(rows)

    # -- the same sniffing the in-memory leg does, deliberately unchanged ---

    def _scope(self, query: str) -> tuple[str, list, Optional[str], Optional[str]]:
        lowered = query.lower()
        year = "2026" if "2026" in lowered else None
        quarters = re.findall(r"20\d\d\s*q[1-4]", lowered.replace(" ", ""))
        services = {d.service for d in self.rows() if d.service}
        service = next((s for s in sorted(services) if s in lowered), None)

        where = ["1=1"]
        params: list = []
        if service:
            where.append("service = ?")
            params.append(service)
        if quarters:
            placeholders = ",".join("?" for _ in quarters)
            where.append(f"lower(quarter) IN ({placeholders})")
            params.extend(quarters)
        elif year:
            where.append("quarter LIKE ?")
            params.append(f"{year}%")
        return " AND ".join(where), params, service, quarters[0] if quarters else year

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        lowered = query.lower()
        where, params, service, window = self._scope(query)
        results: list[tuple[str, float, str]] = []

        if any(w in lowered for w in ("average", "mean", "avg")):
            row = self.connection.execute(
                f"SELECT avg(minutes_to_resolve), count(*) FROM {self.table} "
                f"WHERE {where}",
                params,
            ).fetchone()
            if row and row[1]:
                results.append((
                    f"agg:mean_minutes:{service or 'all'}:{window or 'all'}",
                    1.0,
                    f"mean minutes_to_resolve = {row[0]:.1f} over {row[1]} rows",
                ))
        if any(w in lowered for w in ("longest", "max", "worst", "slowest")):
            row = self.connection.execute(
                f"SELECT doc_id, minutes_to_resolve FROM {self.table} "
                f"WHERE {where} ORDER BY minutes_to_resolve DESC, doc_id LIMIT 1",
                params,
            ).fetchone()
            if row:
                results.append((
                    f"agg:max_minutes:{row[0]}", 1.0,
                    f"max minutes_to_resolve = {row[1]} ({row[0]})",
                ))
        if "per quarter" in lowered or "by quarter" in lowered:
            for quarter, count in self.connection.execute(
                f"SELECT quarter, count(*) FROM {self.table} WHERE {where} "
                f"GROUP BY quarter ORDER BY quarter",
                params,
            ).fetchall():
                results.append(
                    (f"agg:count:{quarter or 'unknown'}", 1.0,
                     f"{quarter} = {count} incidents")
                )
        if not results and any(
            w in lowered for w in ("how many", "count", "number of", "total")
        ):
            row = self.connection.execute(
                f"SELECT count(*) FROM {self.table} WHERE {where}", params
            ).fetchone()
            results.append((
                f"agg:count:{service or 'all'}:{window or 'all'}", 1.0,
                f"count = {row[0]} incidents",
            ))
        return _rank(results, self.backend, k)
