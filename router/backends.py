"""Four backends over one corpus, each genuinely best at something.

They share an interface and nothing else. The point of the project is that
their competences do not overlap as much as a single vector store makes it
tempting to assume, so each one here is implemented well enough to be good at
its own job and is deliberately NOT patched to be adequate at the others.

WHY EACH ONE IS WRITTEN OUT RATHER THAN IMPORTED. BM25, cosine ranking, a graph
traversal and a group-by are all short, and writing them out makes the
comparison legible: a reader can see exactly why the fulltext leg finds an
error code that the vector leg loses, instead of taking it on faith from two
library calls. It also keeps the repository dependency-free.

THE RELATIONAL LEG IS THE ONE PEOPLE FUDGE. Its answers are NUMBERS THAT APPEAR
IN NO DOCUMENT -- a count, a mean, a maximum -- so it cannot be evaluated by
document overlap at all. It returns synthetic result rows with `agg:` ids, and
any evaluation that scored it by retrieved-document recall would score it zero
while it was answering perfectly. That trap is why `Competence.AGGREGATE`
exists as a separate axis.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

from router.embeddings import Embedder, content_tokens, cosine, default_embedder
from router.models import Backend, Document, Edge, RankedHit

DEFAULT_K = 5


class RetrievalBackend(Protocol):
    backend: Backend

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]: ...

    def name(self) -> str: ...


def _rank(scored: list[tuple[str, float, str]], backend: Backend, k: int) -> list[RankedHit]:
    """Sort, truncate, and assign 1-based ranks.

    Ties break on doc_id so two runs order identically. An unstable ordering
    would make the gate flap for reasons unrelated to any routing decision,
    which is the same failure the sibling repos guard against.
    """
    scored = [s for s in scored if s[1] > 0.0]
    scored.sort(key=lambda s: (-s[1], s[0]))
    return [
        RankedHit(doc_id=doc_id, backend=backend, rank=i, score=score, why=why)
        for i, (doc_id, score, why) in enumerate(scored[:k], start=1)
    ]


# ------------------------------------------------------------------ vector


@dataclass
class VectorBackend:
    """Dense retrieval. Best at paraphrase, worst at opaque identifiers.

    An error code embeds to a point determined by a hash of a token that
    appears nowhere else, so its nearest neighbors are noise. That is not a
    defect in this implementation -- real embedding models are also poor at
    exact identifiers, which is the standard argument for hybrid retrieval.
    """

    documents: tuple[Document, ...]
    embedder: Embedder = field(default_factory=default_embedder)
    backend: Backend = Backend.VECTOR
    _vectors: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for doc in self.documents:
            self._vectors[doc.doc_id] = self.embedder.embed(
                f"{doc.title} {doc.text}"
            )

    def name(self) -> str:
        return f"vector({self.embedder.name()})"

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        qv = self.embedder.embed(query)
        scored = [
            (doc_id, cosine(qv, vec), "cosine")
            for doc_id, vec in self._vectors.items()
        ]
        return _rank(scored, self.backend, k)


# ---------------------------------------------------------------- fulltext


@dataclass
class FulltextBackend:
    """BM25 over an inverted index. Best at rare exact terms.

    k1 and b are the textbook defaults and are NOT tuned here. Tuning them on
    an eighteen-query set would fit the evaluation rather than improve the
    retriever, and the resulting numbers would be about the tuning.
    """

    documents: tuple[Document, ...]
    k1: float = 1.2
    b: float = 0.75
    backend: Backend = Backend.FULLTEXT
    _index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)
    _tf: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    _len: dict[str, int] = field(default_factory=dict, repr=False)
    _avglen: float = 0.0

    def __post_init__(self) -> None:
        for doc in self.documents:
            tokens = content_tokens(f"{doc.title} {doc.text}")
            counts: dict[str, int] = defaultdict(int)
            for tok in tokens:
                counts[tok] += 1
                self._index[tok].add(doc.doc_id)
            self._tf[doc.doc_id] = dict(counts)
            self._len[doc.doc_id] = len(tokens)
        self._avglen = (
            sum(self._len.values()) / len(self._len) if self._len else 0.0
        )

    def name(self) -> str:
        return f"fulltext(bm25 k1={self.k1} b={self.b})"

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = len(self._index.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        terms = content_tokens(query)
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)
        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id in self._index.get(term, ()):
                tf = self._tf[doc_id].get(term, 0)
                dl = self._len[doc_id]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (dl / self._avglen if self._avglen else 1)
                )
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
                matched[doc_id].append(term)
        scored = [
            (doc_id, score, "matched " + ",".join(sorted(set(matched[doc_id]))[:3]))
            for doc_id, score in scores.items()
        ]
        return _rank(scored, self.backend, k)


# ------------------------------------------------------------------- graph


@dataclass
class GraphBackend:
    """Typed-edge traversal. The only leg whose answers live in no document.

    `search` accepts natural language and extracts the entities it recognizes,
    which is a deliberate simplification: entity linking is a whole problem and
    this project measures ROUTING, not linking. The simplification is stated
    here rather than hidden, and `linked_entities` is exposed so a caller can
    see what the traversal actually started from -- an empty start set is the
    failure mode behind the `q-trap-3` config-key trap.
    """

    documents: tuple[Document, ...]
    edges: tuple[Edge, ...]
    backend: Backend = Backend.GRAPH
    max_hops: int = 3
    _out: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list), repr=False)
    _in: dict[str, list[Edge]] = field(default_factory=lambda: defaultdict(list), repr=False)

    def __post_init__(self) -> None:
        for edge in self.edges:
            self._out[edge.source].append(edge)
            self._in[edge.target].append(edge)

    def name(self) -> str:
        return f"graph({len(self.edges)} edges, max {self.max_hops} hops)"

    def nodes(self) -> frozenset[str]:
        return frozenset(self._out) | frozenset(self._in)

    def linked_entities(self, query: str) -> frozenset[str]:
        """Entities from the query that exist as graph nodes.

        EMPTY IS A MEANINGFUL RESULT and the reason it is a public method. A
        relationship-shaped question about something with no node -- a config
        key, say -- traverses nothing and returns nothing, and a router that
        sent it here has misrouted in a way worth naming.
        """
        tokens = set(content_tokens(query))
        return frozenset(n for n in self.nodes() if n.lower() in tokens)

    def neighbors(self, start: str, kinds: Iterable[str], hops: int) -> dict[str, int]:
        """Breadth-first traversal returning node -> hop distance."""
        kinds = set(kinds)
        seen: dict[str, int] = {}
        frontier = [start]
        for depth in range(1, hops + 1):
            nxt: list[str] = []
            for node in frontier:
                for edge in self._out.get(node, ()):
                    if edge.kind in kinds and edge.target not in seen:
                        seen[edge.target] = depth
                        nxt.append(edge.target)
            frontier = nxt
            if not frontier:
                break
        return seen

    def dependents_of(self, target: str, kinds: Iterable[str] = ("depends_on",)) -> dict[str, int]:
        """Reverse traversal: which nodes point AT `target`.

        `kinds` is a parameter and not a hardcoded "depends_on" because the
        edges are directional and half the useful questions run against the
        arrow. `owned_by` points service -> team, so "which services does the
        money team own" has NO forward path from a team node at all: an
        ownership question asked about the owner traverses nothing and returns
        nothing, while the router that sent it here looks correct.
        """
        kinds = set(kinds)
        seen: dict[str, int] = {}
        frontier = [target]
        for depth in range(1, self.max_hops + 1):
            nxt: list[str] = []
            for node in frontier:
                for edge in self._in.get(node, ()):
                    if edge.kind in kinds and edge.source not in seen:
                        seen[edge.source] = depth
                        nxt.append(edge.source)
            frontier = nxt
            if not frontier:
                break
        return seen

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        starts = self.linked_entities(query)
        if not starts:
            return []
        lowered = query.lower()
        blast_radius = any(
            w in lowered
            for w in ("affected", "impacted", "depends on it", "went down",
                      "downstream of", "breaks if", "fails")
        )
        scored: list[tuple[str, float, str]] = []
        for start in sorted(starts):
            # A node with no outgoing edges of any traversable kind is a SINK --
            # a team, in this graph. Every question about it is necessarily a
            # reverse question, so direction is decided by the graph's shape at
            # that node rather than by hoping the phrasing gives it away.
            sink = not self._out.get(start)
            if blast_radius:
                reached = self.dependents_of(start, ("depends_on",))
                direction = "reverse"
            elif sink:
                reached = self.dependents_of(start, ("owned_by", "touched_by", "depends_on"))
                direction = "reverse (sink node)"
            else:
                reached = self.neighbors(
                    start, ("depends_on", "owned_by", "touched_by"), self.max_hops
                )
                direction = "forward"
            for node, depth in reached.items():
                # Closer hops rank higher. A two-hop relationship is real and
                # weaker evidence than a one-hop one, and flattening that is
                # how a graph leg starts returning the whole component.
                scored.append((node, 1.0 / depth, f"{direction} {depth} hop(s) from {start}"))
        best: dict[str, tuple[float, str]] = {}
        for node, score, why in scored:
            if node not in best or score > best[node][0]:
                best[node] = (score, why)
        return _rank([(n, s, w) for n, (s, w) in best.items()], self.backend, k)


# -------------------------------------------------------------- relational


@dataclass
class RelationalBackend:
    """Aggregates over incident rows. Answers are computed, never retrieved.

    Result ids are prefixed `agg:` and correspond to NO DOCUMENT, which is the
    honest shape: "payments had 3 incidents in 2026" is not in the corpus and
    cannot be. Any evaluation scoring this leg by document overlap reports zero
    for a leg that is answering correctly, which is precisely the mistake the
    labeled set is designed to expose.
    """

    documents: tuple[Document, ...]
    backend: Backend = Backend.RELATIONAL

    def name(self) -> str:
        return f"relational({len(self.rows())} rows)"

    def rows(self) -> tuple[Document, ...]:
        return tuple(d for d in self.documents if d.kind == "incident")

    def search(self, query: str, k: int = DEFAULT_K) -> list[RankedHit]:
        lowered = query.lower()
        rows = self.rows()
        year = "2026" if "2026" in lowered else None
        quarters = re.findall(r"20\d\d\s*q[1-4]", lowered.replace(" ", ""))
        service = next(
            (s for s in {d.service for d in rows if d.service} if s in lowered),
            None,
        )
        scoped = [
            r for r in rows
            if (service is None or r.service == service)
            and (not quarters or (r.quarter or "").lower() in quarters)
            and (year is None or (r.quarter or "").startswith(year))
        ]
        results: list[tuple[str, float, str]] = []

        if any(w in lowered for w in ("average", "mean", "avg")):
            if scoped:
                mins = [r.minutes_to_resolve or 0 for r in scoped]
                mean = sum(mins) / len(mins)
                results.append((
                    f"agg:mean_minutes:{service or 'all'}:{quarters[0] if quarters else year or 'all'}",
                    1.0, f"mean minutes_to_resolve = {mean:.1f} over {len(scoped)} rows",
                ))
        if any(w in lowered for w in ("longest", "max", "worst", "slowest")):
            if scoped:
                worst = max(scoped, key=lambda r: r.minutes_to_resolve or 0)
                results.append((
                    f"agg:max_minutes:{worst.doc_id}", 1.0,
                    f"max minutes_to_resolve = {worst.minutes_to_resolve} ({worst.doc_id})",
                ))
        if "per quarter" in lowered or "by quarter" in lowered:
            counts: dict[str, int] = defaultdict(int)
            for r in scoped:
                counts[r.quarter or "unknown"] += 1
            for q, n in sorted(counts.items()):
                results.append((f"agg:count:{q}", 1.0, f"{q} = {n} incidents"))
        if not results and any(
            w in lowered for w in ("how many", "count", "number of", "total")
        ):
            results.append((
                f"agg:count:{service or 'all'}:{quarters[0] if quarters else year or 'all'}",
                1.0, f"count = {len(scoped)} incidents",
            ))
        return _rank(results, self.backend, k)


# ------------------------------------------------------------------ bundle


@dataclass(frozen=True)
class Federation:
    """The four backends, addressable by name."""

    vector: VectorBackend
    fulltext: FulltextBackend
    graph: GraphBackend
    relational: RelationalBackend

    def get(self, backend: Backend) -> RetrievalBackend:
        return {
            Backend.VECTOR: self.vector,
            Backend.FULLTEXT: self.fulltext,
            Backend.GRAPH: self.graph,
            Backend.RELATIONAL: self.relational,
        }[backend]

    def all(self) -> tuple[RetrievalBackend, ...]:
        return (self.vector, self.fulltext, self.graph, self.relational)


def build_federation(corpus, embedder: Optional[Embedder] = None) -> Federation:
    return Federation(
        vector=VectorBackend(corpus.documents, embedder or default_embedder()),
        fulltext=FulltextBackend(corpus.documents),
        graph=GraphBackend(corpus.documents, corpus.edges),
        relational=RelationalBackend(corpus.documents),
    )
