"""Deciding which backends a question needs.

This file is the project. The four backends are ordinary; the merge is twelve
lines; what is worth measuring is whether a router can tell, before spending
anything, which stores a question actually requires.

Two routers are implemented and both are meant to be read.

`FanOutRouter` sends every query to every backend. It is the baseline that
cannot be wrong, and including it is not a formality: a fan-out router scores
100% on routing correctness by construction, so any correctness number reported
without its cost beside it is meaningless. It is here to make that visible.

`HeuristicRouter` decides from surface features. It is deliberately NOT a
model: it is deterministic, it runs offline, it can be reasoned about when it
fails, and the gate can depend on it. A model-based router is measured against
it in the paid capture, which is the honest place for a nondeterministic
component.

The two guards below are the engineering, and whether they generalize is the
measurement. Both were written against a named trap in the labeled set:

  An aggregate tell is not an aggregate. "How many retries does the backoff
  design recommend" opens with the strongest counting phrase in the language
  and needs no counting; the answer is a sentence in a design document. So an
  aggregate tell only routes to the relational leg when it CO-OCCURS with
  something that store can actually count.

  A relationship tell is not a relationship. "Which service owns
  gateway.envelope.strict" says "owns", and the subject is a config key with no
  node in the graph. So a relationship tell only routes to the graph when the
  query links to at least one real entity, which is a question the graph itself
  answers rather than a keyword list.

Both guards are stated as rules a reader can disagree with, and both are
measured per-query rather than asserted. Where they fire wrongly, the
confusion matrix in metrics.py shows it and the rationale says which feature
was responsible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from router.backends import Federation
from router.embeddings import content_tokens
from router.models import Backend, Competence, LabeledQuery, RoutingDecision

# Surface tells. Kept as data rather than buried in conditionals so a reader
# can see the whole rule set at once and argue with it.
AGGREGATE_TELLS = (
    "how many", "count", "number of", "total", "average", "mean", "avg",
    "per quarter", "by quarter", "longest", "shortest", "max", "min",
    "worst", "slowest", "sum",
)
RELATIONSHIP_TELLS = (
    "depend", "depends", "depended", "owns", "owned by", "who owns",
    "which team", "which teams", "affected if", "impacted if", "downstream",
    "upstream", "transitively", "connected to", "went down",
)
# Nouns the relational store can actually aggregate over. THE GUARD: an
# aggregate tell with none of these present is a figure of speech.
COUNTABLE = ("incident", "incidents", "outage", "outages", "quarter", "quarters")

# An identifier: SCREAMING_SNAKE or dotted.lower.path. These are the tokens an
# embedding cannot place and an inverted index finds instantly.
IDENTIFIER = re.compile(r"^(?:[A-Z][A-Z0-9_]{3,}|[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,})$")


class Router(Protocol):
    name: str

    def route(self, query: str, query_id: str = "") -> RoutingDecision: ...


@dataclass
class FanOutRouter:
    """Send everything everywhere. Cannot misroute; costs the most.

    The control condition. Its correctness is 1.0 by construction, which is
    exactly why correctness alone is not a metric.
    """

    name: str = "fan-out"

    def route(self, query: str, query_id: str = "") -> RoutingDecision:
        return RoutingDecision(
            query_id=query_id,
            chosen=frozenset(Backend),
            competences=frozenset(Competence),
            rationale=("fan-out: every backend, always",),
        )


@dataclass
class VectorOnlyRouter:
    """The other baseline, and the one most systems actually ship.

    "Just put it all in a vector store" is the default architecture in most
    RAG deployments. Measuring it here is the point: the project's likely
    headline is that this is fine for most queries, and that claim is only
    worth making with the failing minority named.
    """

    name: str = "vector-only"

    def route(self, query: str, query_id: str = "") -> RoutingDecision:
        return RoutingDecision(
            query_id=query_id,
            chosen=frozenset({Backend.VECTOR}),
            competences=frozenset({Competence.SEMANTIC}),
            rationale=("vector-only: the single-store default",),
        )


@dataclass
class HeuristicRouter:
    """Feature-based routing with the two guards, and a rationale per decision.

    `federation` is needed for one reason only: the relationship guard asks the
    GRAPH whether the query links to a real node. That is a deliberate
    coupling. The alternative is a hardcoded entity list in this file, which
    would be a second copy of the graph that silently drifts from the first.
    """

    federation: Optional[Federation] = None
    name: str = "heuristic"

    def route(self, query: str, query_id: str = "") -> RoutingDecision:
        lowered = query.lower()
        tokens = content_tokens(query)
        raw_tokens = query.split()
        chosen: set[Backend] = set()
        competences: set[Competence] = set()
        why: list[str] = []

        # --- exact term -------------------------------------------------
        identifiers = [t for t in raw_tokens if IDENTIFIER.match(t.strip(".,?"))]
        if identifiers:
            chosen.add(Backend.FULLTEXT)
            competences.add(Competence.EXACT_TERM)
            why.append(f"identifier token(s) {identifiers[:2]} -> fulltext")

        # --- aggregate, behind the countable-noun guard -----------------
        agg_tell = next((t for t in AGGREGATE_TELLS if t in lowered), None)
        if agg_tell:
            countable = next((c for c in COUNTABLE if c in lowered), None)
            if countable:
                chosen.add(Backend.RELATIONAL)
                competences.add(Competence.AGGREGATE)
                why.append(f"aggregate tell '{agg_tell}' + countable '{countable}'")
            else:
                why.append(
                    f"aggregate tell '{agg_tell}' SUPPRESSED: nothing countable "
                    f"in the query, so this is a figure of speech"
                )

        # --- relationship, behind the linked-entity guard ---------------
        rel_tell = next((t for t in RELATIONSHIP_TELLS if t in lowered), None)
        if rel_tell:
            linked = (
                self.federation.graph.linked_entities(query)
                if self.federation is not None
                else frozenset()
            )
            if linked:
                chosen.add(Backend.GRAPH)
                competences.add(Competence.RELATIONSHIP)
                why.append(
                    f"relationship tell '{rel_tell}' + linked {sorted(linked)}"
                )
            else:
                why.append(
                    f"relationship tell '{rel_tell}' SUPPRESSED: no query token "
                    f"is a graph node, so a traversal has nowhere to start"
                )

        # --- semantic -----------------------------------------------------
        # Not a keyword rule. Semantic is what a question IS when it is asking
        # about meaning, so it fires when the query carries ordinary content
        # words beyond the identifiers and tells that already routed it.
        structural = set()
        for phrase in (agg_tell, rel_tell):
            if phrase:
                structural.update(phrase.split())
        prose = [
            t for t in tokens
            if t not in structural and not IDENTIFIER.match(t)
        ]
        if len(prose) >= 3 or not chosen:
            chosen.add(Backend.VECTOR)
            competences.add(Competence.SEMANTIC)
            why.append(
                f"{len(prose)} prose token(s) -> vector"
                if len(prose) >= 3
                else "nothing else fired -> vector as the fallback"
            )

        return RoutingDecision(
            query_id=query_id,
            chosen=frozenset(chosen),
            competences=frozenset(competences),
            rationale=tuple(why),
        )


def route_all(router: Router, queries: tuple[LabeledQuery, ...]) -> list[RoutingDecision]:
    return [router.route(q.text, q.query_id) for q in queries]
