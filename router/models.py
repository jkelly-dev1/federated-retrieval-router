"""Domain types for the federated retrieval router.

Three distinctions in this file carry the project, and getting the vocabulary
wrong here is how the measurement quietly stops meaning anything.

1. BACKEND vs COMPETENCE. A backend is a store with a retrieval mechanism. A
   competence is the QUESTION SHAPE that mechanism is actually good at. Four
   backends do not mean four competences: two stores can both answer a lookup
   and only one can answer a multi-hop. The labeled query set is annotated with
   competences, not with store names, so "the router picked Elasticsearch" is
   never mistaken for "the router was right".

2. CAN-ANSWER vs SHOULD-ROUTE. Several backends can often produce SOMETHING for
   a query. The ground truth in this project is the set of backends that can
   produce the CORRECT answer, which is usually smaller and occasionally
   surprising. Scoring a router against "returned any result" is how a router
   that fans out to everything scores 100% and costs four times as much.

3. RANK vs SCORE. Every backend returns a score, and none of them are on the
   same scale: BM25 is unbounded and corpus-dependent, cosine is [-1, 1], a
   graph hop count is an integer, a SQL aggregate has no similarity at all.
   `RankedHit` therefore carries BOTH, and fusion uses the rank. See fusion.py
   for why rank fusion is the default and where it loses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Backend(str, Enum):
    """A store plus the retrieval mechanism over it.

    Deliberately four, and deliberately these four: they are the four shapes a
    real platform actually has, and each one is genuinely BEST at something the
    others are bad at. A fifth store that duplicated an existing competence
    would add plumbing and no measurement.
    """

    VECTOR = "vector"          # dense embeddings, semantic similarity
    GRAPH = "graph"            # typed edges, relationship traversal
    FULLTEXT = "fulltext"      # BM25 over an inverted index, exact terms
    RELATIONAL = "relational"  # rows and aggregates, counts and joins


class Competence(str, Enum):
    """The question shape a backend is good at.

    THE LABELED SET IS ANNOTATED WITH THESE, NOT WITH BACKEND NAMES. A router
    that learns "questions with quotes go to fulltext" has learned a rule about
    the corpus; a router that learns "exact-term questions go to whatever
    indexes exact terms" has learned something portable. Keeping the two apart
    also makes it possible to say WHY a misroute happened.
    """

    SEMANTIC = "semantic"          # paraphrase, concept, "how do we handle X"
    RELATIONSHIP = "relationship"  # who depends on what, multi-hop
    EXACT_TERM = "exact_term"      # error codes, config keys, quoted strings
    AGGREGATE = "aggregate"        # counts, averages, group-by, time windows


@dataclass(frozen=True)
class Document:
    """A retrievable unit. Every backend indexes the same corpus differently.

    ONE CORPUS, FOUR INDEXES, ON PURPOSE. A federated router evaluated against
    four DIFFERENT corpora measures nothing but which corpus was easiest. Every
    backend here sees the same documents and differs only in what it can do
    with them.
    """

    doc_id: str
    title: str
    text: str
    kind: str                       # "design" | "incident" | "runbook" | "row"
    service: Optional[str] = None
    team: Optional[str] = None
    quarter: Optional[str] = None   # "2026Q1", for the relational leg
    minutes_to_resolve: Optional[int] = None


@dataclass(frozen=True)
class Edge:
    """One typed relationship. The graph leg's entire substrate.

    `kind` is a verb because the traversal reads better as a sentence:
    (checkout) --depends_on--> (payments). A graph of untyped edges answers
    "is there a path" and not "is there a path THAT MEANS SOMETHING", and the
    difference is most of why the graph leg earns its cost when it does.
    """

    source: str
    kind: str        # "depends_on" | "owned_by" | "touched_by"
    target: str


@dataclass(frozen=True)
class RankedHit:
    """One result from one backend, carrying where it came from.

    PROVENANCE IS NOT DECORATION. The whole claim of a federated router is that
    a merged list is better than any single list, and that claim is unverifiable
    if the merged list cannot say which leg produced each item. It is also what
    makes the per-backend contribution measurable after fusion rather than
    asserted.

    `score` is the backend's native score and is NOT comparable across
    backends. `rank` is 1-based within that backend's own list and is the only
    quantity fusion may use.
    """

    doc_id: str
    backend: Backend
    rank: int
    score: float
    why: str = ""    # one phrase: the term that matched, the edge traversed


@dataclass(frozen=True)
class FusedHit:
    """One row of the merged answer.

    `contributors` is the set of backends that returned this document at all,
    and it is the field that makes "the graph leg earned its cost" a
    measurable statement rather than a claim: a document only the graph leg
    found has exactly one contributor.
    """

    doc_id: str
    fused_score: float
    rank: int
    contributors: tuple[Backend, ...]
    per_backend_rank: dict[Backend, int] = field(default_factory=dict)

    @property
    def unique_to(self) -> Optional[Backend]:
        """The backend that found this alone, or None if several did."""
        return self.contributors[0] if len(self.contributors) == 1 else None


@dataclass(frozen=True)
class LabeledQuery:
    """A query, the backends that can actually answer it, and why.

    `required` IS THE GROUND TRUTH AND IT IS A SET, not a single backend. Some
    questions genuinely need two legs -- "which services did the team that owns
    checkout touch during the March incident" needs the graph for ownership and
    the relational leg for the window -- and a router scored against a single
    correct answer would be marked wrong for getting that right.

    `sufficient` is the weaker relation: backends that would also produce the
    correct answer on their own. A router choosing one of those is not wrong,
    it is just not cheapest, and the metrics report the two separately.

    `trap` marks queries that LOOK like one competence and are another. They
    exist because a router with no traps in its evaluation set is measured
    entirely on the easy half of the problem.

    `required` IS A DESIGNED GROUND TRUTH, NOT AN EMPIRICAL ONE, and the
    distinction is load-bearing. It records
    WHICH BACKEND THE QUESTION SHAPE CALLS FOR -- a paraphrase needs semantic
    retrieval, an identifier needs an inverted index, a traversal needs a graph.
    It does NOT record which backend happens to win on this corpus with this
    embedder, and on the offline stack those two answers differ.

    They differ for a specific, measurable reason. The offline embedder is a
    bag-of-tokens hash, so on a paraphrase sharing even one RARE term with its
    target -- "dependency", "backoff", "twice" -- BM25's idf carries more signal
    than the whole vector carries cosine, and the fulltext leg wins a query the
    vector leg is supposed to own. Measured jaccard overlap on those queries is
    0.02 to 0.10, so they are genuine paraphrases and the win is real.

    KEEPING THE TWO APART IS WHAT MAKES THE GAP REPORTABLE. Scoring routing
    against measured winners would define the router's job as "predict what the
    mock does", and the project would score well by learning an artifact.
    Scoring against the designed shape keeps routing honest, and
    `metrics.validate_competences` reports the gap between design and
    measurement as a finding in its own right -- one the real-model capture is
    built to close or confirm.
    """

    query_id: str
    text: str
    competences: frozenset[Competence]
    required: frozenset[Backend]
    sufficient: frozenset[Backend] = frozenset()
    relevant_docs: frozenset[str] = frozenset()
    trap: str = ""

    @property
    def acceptable(self) -> frozenset[Backend]:
        """Any backend set that answers correctly must cover `required`."""
        return self.required | self.sufficient


@dataclass
class RoutingDecision:
    """Which backends the router chose, and what it thought it was doing.

    `rationale` is recorded per decision because a routing confusion matrix
    tells you THAT the router misroutes and never WHY. With the rationale
    attached, a misroute is traceable to the feature that fired.
    """

    query_id: str
    chosen: frozenset[Backend]
    competences: frozenset[Competence]
    rationale: tuple[str, ...] = ()

    def is_correct(self, labeled: LabeledQuery) -> bool:
        """Correct means: every REQUIRED backend was chosen.

        Deliberately not "chosen == required". Choosing a superset still
        answers the question; it just costs more, and cost is reported
        separately as fan-out rather than folded into a correctness number.
        Conflating the two produces a metric that cannot distinguish a router
        that is wrong from a router that is merely expensive.
        """
        return labeled.required <= self.chosen
