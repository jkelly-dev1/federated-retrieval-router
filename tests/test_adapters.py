"""The real-store adapters, tested with no store running and no driver installed.

WHAT THESE TESTS ARE FOR. An adapter is mostly a request shape and a response
mapping, and both can be wrong in ways that produce plausible output: a cosine
distance read as a similarity ranks the corpus exactly backwards, and an
Elasticsearch index created without its analyzer silently falls back to the
default one, making an A/B that claims to isolate scoring measure tokenization
instead. Neither needs a server to catch, and neither is caught by an
integration test that only asserts "some documents came back".

The integration tests that DO need servers live in tests/test_integration.py
and skip themselves when nothing is listening.

Every driver import here is absent on purpose: the fixtures inject fakes, so
this file behaves identically on a laptop with psycopg installed and in CI
where it is not.
"""
from __future__ import annotations

import builtins

import pytest

from router.adapters import (
    DUCKDB_TABLE,
    ES_INDEX,
    PG_TABLE,
    DuckDBBackend,
    ElasticsearchBackend,
    PgVectorBackend,
    _pg_vector,
)
from router.corpus import build_corpus
from router.embeddings import HashingEmbedder
from router.models import Backend, Document


def _docs() -> tuple[Document, ...]:
    return (
        Document(doc_id="d1", title="Retry design", text="callers retry twice",
                 kind="design"),
        Document(doc_id="d2", title="Token error", text="ERR_TOKEN_9101 after rotation",
                 kind="runbook"),
    )


# ------------------------------------------------------- the optional driver


@pytest.mark.parametrize(
    "factory, module, package",
    [
        (lambda: PgVectorBackend(_docs(), HashingEmbedder(), dsn="postgresql:///x"),
         "psycopg", "psycopg"),
        (lambda: ElasticsearchBackend(_docs(), url="http://localhost:9200"),
         "elasticsearch", "elasticsearch"),
        (lambda: DuckDBBackend(_docs()), "duckdb", "duckdb"),
    ],
)
def test_a_missing_driver_is_a_sentence_not_a_traceback(
    factory, module, package, monkeypatch
):
    """The failure that broke CI on a sibling repository, pinned per adapter.

    None of these packages is in requirements.txt, so the offline suite must
    behave identically whether or not they happen to be installed. Hiding the
    import makes this test say the same thing on both machines.
    """
    real_import = builtins.__import__

    def hidden(name, *args, **kwargs):
        if name == module or name.startswith(f"{module}."):
            raise ImportError(f"No module named {module!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", hidden)
    with pytest.raises(RuntimeError, match=f"pip install.*{package}"):
        factory()


def test_an_adapter_with_no_client_and_no_address_says_which_is_missing():
    """A constructor that quietly connected to a default localhost would make
    an empty result look like a measurement."""
    with pytest.raises(ValueError, match="dsn or an injected connection"):
        PgVectorBackend(_docs(), HashingEmbedder(), connection=None, dsn=None)


# ------------------------------------------------------------------ pgvector


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, rows=()):
        self.cur = _FakeCursor(list(rows))
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


def test_pgvector_similarity_is_one_minus_distance_not_the_distance():
    """THE INVERSION THAT WOULD LOOK FINE.

    `<=>` is cosine DISTANCE. Ranking by it directly returns the LEAST similar
    documents, in a plausible-looking order, with plausible-looking scores. The
    SQL string alone cannot show which way round it is; the arithmetic can.
    """
    connection = _FakeConnection(rows=[("d1", 0.9), ("d2", 0.2)])
    backend = PgVectorBackend(_docs(), HashingEmbedder(), connection=connection)

    hits = backend.search("retry", k=5)
    assert [h.doc_id for h in hits] == ["d1", "d2"]
    assert hits[0].score == pytest.approx(0.9)
    assert hits[0].rank == 1 and hits[0].backend is Backend.VECTOR

    sql, params = connection.cur.executed[-1]
    assert "1 - (embedding <=> %s::vector)" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert params[-1] == 5


def test_pgvector_drops_non_positive_scores_exactly_as_the_toy_leg_does():
    """`_rank` filters score <= 0 in the in-memory backend. If the adapter
    kept them, the A/B would compare two different definitions of a hit."""
    connection = _FakeConnection(rows=[("d1", 0.4), ("d2", 0.0), ("d3", -0.3)])
    backend = PgVectorBackend(_docs(), HashingEmbedder(), connection=connection)
    assert [h.doc_id for h in backend.search("retry")] == ["d1"]


def test_the_pgvector_schema_matches_the_embedder_width():
    """A vector column declared at the wrong width fails at INSERT time with a
    message about the column, not about the embedder that caused it."""
    embedder = HashingEmbedder(dimensions=64)
    backend = PgVectorBackend(_docs(), embedder, connection=_FakeConnection())
    ddl = "\n".join(backend.schema_statements())
    assert "vector(64)" in ddl
    assert PG_TABLE in ddl
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl


def test_loading_embeds_on_the_client_so_only_the_store_changes():
    connection = _FakeConnection()
    backend = PgVectorBackend(_docs(), HashingEmbedder(), connection=connection)
    assert backend.load() == 2
    inserts = [e for e in connection.cur.executed if e[0].startswith("INSERT")]
    assert len(inserts) == 2
    assert connection.commits == 1
    # The vector goes over as pgvector's text format, not a Python list repr.
    assert inserts[0][1][3].startswith("[") and "," in inserts[0][1][3]


def test_the_vector_literal_is_pgvector_text_format():
    assert _pg_vector([0.5, -0.25]) == "[0.5,-0.25]"
    assert _pg_vector([1e-12]).startswith("[1e-12")


# ------------------------------------------------------------- elasticsearch


class _FakeIndices:
    def __init__(self):
        self.created: list[tuple[str, dict]] = []
        self.exists_result = False
        self.deleted: list[str] = []
        self.refreshed: list[str] = []

    def exists(self, index):
        return self.exists_result

    def create(self, index, body):
        self.created.append((index, body))

    def delete(self, index):
        self.deleted.append(index)

    def refresh(self, index):
        self.refreshed.append(index)


class _FakeES:
    def __init__(self, hits=()):
        self.indices = _FakeIndices()
        self.documents: list[dict] = []
        self.searches: list[dict] = []
        self._hits = list(hits)

    def index(self, index, id, document):
        self.documents.append({"index": index, "id": id, **document})

    def search(self, index, query, size):
        self.searches.append({"index": index, "query": query, "size": size})
        return {"hits": {"hits": self._hits}}


def test_the_matching_analyzer_is_actually_attached_to_the_fields():
    """AN INDEX CREATED WITHOUT ITS ANALYZER SILENTLY USES THE DEFAULT ONE.

    That failure has no error and no empty result -- it just quietly turns an
    A/B that claims to isolate BM25 scoring into one that also swapped the
    tokenizer. The mapping has to name the analyzer on the fields, not merely
    define it in settings.
    """
    backend = ElasticsearchBackend(_docs(), client=_FakeES(), analyzer="matching")
    body = backend.index_body()
    assert body["mappings"]["properties"]["body"]["analyzer"] == "frr_matching"
    assert body["mappings"]["properties"]["title"]["analyzer"] == "frr_matching"
    analysis = body["settings"]["analysis"]
    assert analysis["analyzer"]["frr_matching"]["tokenizer"] == "frr_pattern"
    assert "lowercase" in analysis["analyzer"]["frr_matching"]["filter"]


def test_the_matching_analyzer_mirrors_this_repositorys_own_tokenizer():
    """The point of the 'matching' setting is that the ONLY difference from the
    in-memory leg is the scoring implementation. A different pattern or a
    different stop list would quietly reintroduce the confound.

    THIS TEST IS NOT SUFFICIENT AND SAYS SO. It passed for a full A/B run
    against an analyzer that tokenized "Checkout Service" as
    ["heckout", "ervice"], because an identical regex SOURCE is not identical
    tokenization when the two engines apply it at different points in the
    pipeline. What settles it is running text through the real _analyze
    endpoint and comparing token lists -- see
    tests/test_integration.py::test_the_matching_analyzer_produces_the_same_tokens_as_this_repository.
    """
    from router.embeddings import STOPWORDS, _TOKEN

    analysis = ElasticsearchBackend(
        _docs(), client=_FakeES(), analyzer="matching"
    ).index_body()["settings"]["analysis"]
    assert analysis["tokenizer"]["frr_pattern"]["pattern"] == _TOKEN.pattern
    assert set(analysis["filter"]["frr_stops"]["stopwords"]) == set(STOPWORDS)


def test_the_pattern_tokenizer_is_case_insensitive():
    """WITHOUT THIS FLAG THE MIRROR IS BACKWARDS (bug log, defect 19).

    router/embeddings.py lowercases and then matches. Elasticsearch tokenizes
    and then lowercases. A lowercase-only character class therefore treats
    every capital letter as a delimiter rather than folding it, and the two
    legs silently index different tokens while every offline assertion about
    the mapping still passes.
    """
    analysis = ElasticsearchBackend(
        _docs(), client=_FakeES(), analyzer="matching"
    ).index_body()["settings"]["analysis"]
    assert analysis["tokenizer"]["frr_pattern"]["flags"] == "CASE_INSENSITIVE"


def test_the_english_analyzer_does_not_smuggle_the_custom_settings():
    body = ElasticsearchBackend(
        _docs(), client=_FakeES(), analyzer="english"
    ).index_body()
    assert body["mappings"]["properties"]["body"]["analyzer"] == "english"
    assert "settings" not in body


def test_an_unknown_analyzer_is_refused_rather_than_defaulted():
    """Two analyzers answer two different questions. Silently picking one would
    make the capture's own label wrong."""
    with pytest.raises(ValueError, match="'matching' or 'english'"):
        ElasticsearchBackend(_docs(), client=_FakeES(), analyzer="standard")


def test_elasticsearch_search_maps_id_and_score_to_a_ranked_hit():
    client = _FakeES(hits=[
        {"_id": "d2", "_score": 4.5},
        {"_id": "d1", "_score": 1.25},
    ])
    backend = ElasticsearchBackend(_docs(), client=client)
    hits = backend.search("ERR_TOKEN_9101", k=3)

    assert [(h.doc_id, h.rank) for h in hits] == [("d2", 1), ("d1", 2)]
    assert hits[0].score == pytest.approx(4.5)
    assert hits[0].backend is Backend.FULLTEXT
    assert client.searches[0]["size"] == 3
    assert client.searches[0]["query"]["multi_match"]["fields"] == ["title", "body"]


def test_loading_elasticsearch_refreshes_so_the_first_search_can_see_anything():
    """Elasticsearch is near-real-time. Without an explicit refresh the first
    query after a load returns nothing, and an A/B would record a real store
    scoring zero on every query."""
    client = _FakeES()
    backend = ElasticsearchBackend(_docs(), client=client)
    assert backend.load() == 2
    assert client.indices.refreshed == [ES_INDEX]
    assert len(client.documents) == 2


# -------------------------------------------------------------------- duckdb


class _FakeDuck:
    def __init__(self, results=None):
        self.results = results or {}
        self.executed: list[tuple[str, list]] = []
        self._last = None

    def execute(self, sql, params=None):
        self.executed.append((sql, list(params or [])))
        self._last = sql
        return self

    def fetchone(self):
        for key, value in self.results.items():
            if key in (self._last or ""):
                return value
        return None

    def fetchall(self):
        for key, value in self.results.items():
            if key in (self._last or ""):
                return value
        return []


def test_duckdb_scopes_with_bound_parameters_not_string_interpolation():
    """The service name and quarter come out of a user query. Interpolating
    them into SQL would be an injection seam in a demo about trustworthy
    systems, and it would also break on any name containing a quote."""
    corpus = build_corpus()
    backend = DuckDBBackend(corpus.documents, connection=_FakeDuck(
        {"count(*)": (3,)}
    ))
    hits = backend.search("how many incidents did payments have in 2026")

    sql, params = backend.connection.executed[-1]
    assert "?" in sql and "payments" not in sql
    assert "payments" in params
    assert hits and hits[0].doc_id.startswith("agg:count:payments")
    assert "count = 3 incidents" in hits[0].why


def test_duckdb_reports_the_same_aggregate_shapes_as_the_in_memory_leg():
    """The A/B is about WHO computes the aggregate, so the result ids have to
    match. Different ids would make every comparison a mismatch by
    construction."""
    corpus = build_corpus()
    duck = DuckDBBackend(corpus.documents, connection=_FakeDuck({
        "avg(minutes_to_resolve)": (42.0, 5),
        "ORDER BY minutes_to_resolve DESC": ("incident-2026q1-che-00", 180),
    }))
    mean = duck.search("average time to resolve incidents in 2026Q3")
    worst = duck.search("what is the longest an incident took to resolve")

    assert mean[0].doc_id.startswith("agg:mean_minutes:")
    assert worst[0].doc_id == "agg:max_minutes:incident-2026q1-che-00"
    assert DUCKDB_TABLE in duck.connection.executed[0][0]


def test_duckdb_only_loads_incident_rows():
    """The relational leg's universe is incident rows. Loading design documents
    into it would change what an aggregate counts."""
    corpus = build_corpus()
    duck = DuckDBBackend(corpus.documents, connection=_FakeDuck())
    assert duck.load() == len(duck.rows())
    assert all(d.kind == "incident" for d in duck.rows())
    assert len(duck.rows()) < len(corpus.documents)
