"""The real stores, when they happen to be running. Skipped when they are not.

THE RULE THIS FILE OBEYS: `pip install -r requirements.txt && pytest` must stay
green on a machine with no Docker, no drivers and no network. So every test
here skips -- loudly, with a reason naming what was missing -- rather than
failing. A red suite that means "you did not start a container" trains people
to ignore red suites.

WHAT THESE TESTS ASSERT AND WHAT THEY DELIBERATELY DO NOT. They assert that
each adapter can create its schema, load the corpus, and return ranked hits
whose shape matches the in-memory leg. They do NOT assert that a real store
ranks the same documents as the hand-rolled one: measuring that difference is
the entire point of scripts/compare_stores.py, and pinning it in a test would
turn a measurement into a threshold that fails whenever a store version moves.

    docker compose up -d
    pip install -r requirements-integration.txt
    pytest tests/test_integration.py -v
"""
from __future__ import annotations

import os
import socket

import pytest

from router.adapters import DuckDBBackend, ElasticsearchBackend, PgVectorBackend
from router.backends import FulltextBackend, RelationalBackend, VectorBackend
from router.corpus import build_corpus
from router.embeddings import HashingEmbedder
from router.models import Backend

# Ports match docker-compose.yml, which binds both services to localhost on
# non-default ports so a run cannot accidentally hit a real local database.
PG_DSN = os.environ.get(
    "FRR_PG_DSN", "postgresql://postgres:frr-local-only@127.0.0.1:55432/frr"
)
ES_URL = os.environ.get("FRR_ES_URL", "http://127.0.0.1:59200")


def _listening(host: str, port: int, timeout: float = 0.4) -> bool:
    """Cheap reachability probe. A connect attempt with a short timeout beats
    importing a driver and waiting out its own retry policy."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _require(module: str):
    return pytest.importorskip(
        module, reason=f"{module} is optional; pip install -r requirements-integration.txt"
    )


def _require_service(name: str, host: str, port: int) -> None:
    if not _listening(host, port):
        pytest.skip(f"{name} is not listening on {host}:{port}; docker compose up -d")


@pytest.fixture(scope="module")
def corpus():
    return build_corpus()


# -------------------------------------------------------------------- duckdb


@pytest.fixture()
def duckdb_backend(corpus):
    _require("duckdb")
    backend = DuckDBBackend(corpus.documents)
    backend.load()
    return backend


def test_duckdb_counts_what_the_python_loop_counts(duckdb_backend, corpus):
    """THE EXPECTATION STATED BEFORE THE RUN: a real SQL engine should change
    NOTHING here, because the relational leg was never where the fudge was.

    If this ever diverges, the interesting question is which one is right --
    so the assertion is equality of the reported figure, not a tolerance.
    """
    query = "how many incidents did payments have in 2026"
    memory = RelationalBackend(corpus.documents).search(query)
    duck = duckdb_backend.search(query)

    assert [h.doc_id for h in duck] == [h.doc_id for h in memory]
    assert [h.why for h in duck] == [h.why for h in memory]
    assert duck[0].backend is Backend.RELATIONAL


@pytest.mark.parametrize(
    "query",
    [
        "how many incidents did payments have in 2026",
        "average time to resolve incidents in 2026Q3",
        "what is the longest an incident took to resolve",
        "count incidents per quarter",
    ],
)
def test_duckdb_and_the_python_loop_agree_on_every_aggregate_query(
    duckdb_backend, corpus, query
):
    memory = RelationalBackend(corpus.documents).search(query)
    duck = duckdb_backend.search(query)
    assert [(h.doc_id, h.why) for h in duck] == [(h.doc_id, h.why) for h in memory]


def test_duckdb_loaded_only_the_incident_rows(duckdb_backend, corpus):
    total = duckdb_backend.connection.execute(
        f"SELECT count(*) FROM {duckdb_backend.table}"
    ).fetchone()[0]
    assert total == len([d for d in corpus.documents if d.kind == "incident"])
    assert total < len(corpus.documents)


# ------------------------------------------------------------------ pgvector


@pytest.fixture(scope="module")
def pg_backend(corpus):
    _require("psycopg")
    _require_service("pgvector", "127.0.0.1", 55432)
    backend = PgVectorBackend(corpus.documents, HashingEmbedder(), dsn=PG_DSN)
    backend.load()
    return backend


def test_pgvector_returns_ranked_hits_in_the_same_shape(pg_backend):
    hits = pg_backend.search("how should a caller behave when a dependency fails", k=5)
    assert hits, "an exact scan over 85 rows should return something"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.backend is Backend.VECTOR for h in hits)
    assert all(h.score > 0 for h in hits)


def test_pgvector_agrees_with_the_in_memory_cosine_on_the_top_hit(pg_backend, corpus):
    """SAME EMBEDDER, SAME MATH, DIFFERENT EXECUTOR -- so the top hit should
    match. This is a correctness check on the adapter, NOT the A/B: it uses the
    hash embedder on both sides precisely so the only variable is the store.

    Ties are the one honest exception. Two documents with identical cosine can
    order either way, so the assertion is on the top hit rather than the list.
    """
    query = "how should a caller behave when a dependency fails"
    memory = VectorBackend(corpus.documents, HashingEmbedder()).search(query, k=5)
    remote = pg_backend.search(query, k=5)
    assert remote[0].doc_id == memory[0].doc_id
    assert remote[0].score == pytest.approx(memory[0].score, abs=1e-6)


# ------------------------------------------------------------- elasticsearch


@pytest.fixture(scope="module")
def es_backend(corpus):
    _require("elasticsearch")
    _require_service("elasticsearch", "127.0.0.1", 59200)
    backend = ElasticsearchBackend(corpus.documents, url=ES_URL, analyzer="matching")
    backend.load()
    return backend


def test_the_matching_analyzer_produces_the_same_tokens_as_this_repository(
    es_backend, corpus
):
    """THE TEST THAT WAS MISSING, and the reason the fulltext A/B had to be
    thrown away and re-run once (bug log, defect 19).

    Two offline tests already claimed this analyzer mirrored
    router/embeddings.py. One asserted the mapping carried the same regex
    SOURCE as `_TOKEN`; the other asserted both legs return the same top hit
    for an identifier. Both passed against an analyzer that was emitting
    ["heckout", "ervice"] for "Checkout Service" -- because Elasticsearch runs
    the tokenizer BEFORE the lowercase filter while this repository lowercases
    first, and because a query mangled the same way as the documents still
    finds them.

    So this asserts the only thing that settles it: run real corpus text and
    real query text through the actual _analyze endpoint and require the token
    LIST to equal content_tokens() exactly. Nothing short of the running
    service can check this, which is why it lives here rather than offline.
    """
    from router.embeddings import content_tokens

    samples = [d.title for d in corpus.documents]
    samples += [d.text for d in corpus.documents]
    samples += [q.text for q in corpus.queries]

    mismatches = []
    for text in samples:
        response = es_backend.client.indices.analyze(
            index=es_backend.index, analyzer="frr_matching", text=text
        )
        got = [t["token"] for t in response["tokens"]]
        want = content_tokens(text)
        if got != want:
            mismatches.append((text, want, got))

    assert not mismatches, (
        f"{len(mismatches)} of {len(samples)} strings tokenize differently in "
        f"Elasticsearch than in router/embeddings.py, so the 'matching' "
        f"analyzer is not mirroring anything and the fulltext delta would be "
        f"measuring the mapping. First: {mismatches[0]}"
    )


def test_uppercase_survives_the_matching_analyzer(es_backend):
    """The specific shape of defect 19, pinned on its own so a regression names
    itself instead of surfacing as a bulk count.

    A lowercase-only character class in the tokenizer makes every capital a
    DELIMITER rather than a character to be folded.
    """
    from router.embeddings import content_tokens

    for text in ("ERR_UPSTREAM_4423", "Checkout Service Retries", "Queue.Prefetch"):
        response = es_backend.client.indices.analyze(
            index=es_backend.index, analyzer="frr_matching", text=text
        )
        assert [t["token"] for t in response["tokens"]] == content_tokens(text), text


def test_elasticsearch_finds_the_identifier_the_hand_rolled_index_finds(
    es_backend, corpus
):
    """An opaque identifier is what an inverted index is for. Both legs should
    surface the same runbook; if the real one does not, the analyzer is not
    doing what the mapping says."""
    query = "ERR_UPSTREAM_4423"
    memory = FulltextBackend(corpus.documents).search(query, k=5)
    remote = es_backend.search(query, k=5)
    assert memory and remote
    assert remote[0].doc_id == memory[0].doc_id
    assert all(h.backend is Backend.FULLTEXT for h in remote)


def test_elasticsearch_finds_a_dotted_config_key(es_backend):
    """The tokenizer defect the hand-rolled index once had (bug log, defect 2)
    was that a trailing period made a dotted key unfindable. The mirrored
    analyzer must not reintroduce it."""
    hits = es_backend.search("queue.prefetch.count", k=5)
    assert hits, "a dotted identifier must survive analysis"


def test_the_two_analyzers_are_two_different_measurements(corpus):
    """Both indexes are built and both are queried, because the A/B's whole
    claim is that swapping a store changes the tokenizer as well as the
    scoring. If these two ever returned identical rankings on every query, the
    'matching' analyzer would not be mirroring anything."""
    _require("elasticsearch")
    _require_service("elasticsearch", "127.0.0.1", 59200)
    english = ElasticsearchBackend(
        corpus.documents, url=ES_URL, index="frr-documents-english", analyzer="english"
    )
    english.load()
    matching = ElasticsearchBackend(corpus.documents, url=ES_URL, analyzer="matching")
    matching.load()

    queries = [q.text for q in corpus.queries]
    differ = sum(
        1
        for q in queries
        if [h.doc_id for h in english.search(q, k=5)]
        != [h.doc_id for h in matching.search(q, k=5)]
    )
    assert differ > 0, (
        "two analyzers that agree on all 19 queries would mean the analyzer "
        "setting is not reaching the index"
    )
