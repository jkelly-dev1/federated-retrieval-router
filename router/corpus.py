"""A synthetic corpus built so that four backends have four real competences.

EVERYTHING HERE IS INVENTED. The services, teams, incidents and people are
fictional, every host is under an .example TLD reserved by RFC 2606, and no
real operational data is present. The failure modes this project measures are
structural, and a structural result reproduces perfectly on synthetic data.

THE CORPUS IS BUILT BACKWARDS FROM THE MEASUREMENT, which is the only way it
can support one. Each of the four competences needs questions that ONE backend
answers correctly and the others answer plausibly and wrongly:

  SEMANTIC      prose that shares meaning without sharing words, so a
                paraphrase finds it and a term match does not
  RELATIONSHIP  ownership and dependency edges two or three hops apart, so the
                answer exists only as a traversal and appears in no document
  EXACT_TERM    error codes, config keys and quoted strings that are
                lexically distinctive and semantically meaningless, so an
                embedding of them lands nowhere useful
  AGGREGATE     counts, averages and time windows, where the answer is a
                NUMBER THAT APPEARS IN NO DOCUMENT and has to be computed

THE LAST ONE IS THE SHARPEST AND THE MOST OFTEN FUDGED. "How many incidents did
payments have in Q1" cannot be retrieved. It can only be computed. A vector
store will happily return the three most incident-shaped documents and a naive
evaluation will call that a hit.

AND THE CORPUS CONTAINS TRAPS ON PURPOSE. A query set with no traps measures a
router on the easy half of the problem, so several questions here look like one
competence and are another -- an error code that is actually a semantic
question, a "how many" that is actually a lookup. They are marked, and the
metrics report trap accuracy separately.
"""
from __future__ import annotations

from dataclasses import dataclass

from router.models import Backend, Competence, Document, Edge, LabeledQuery

# ---------------------------------------------------------------- documents


def documents() -> tuple[Document, ...]:
    return (
        # ---- SEMANTIC: prose that shares meaning, not words. -------------
        Document(
            doc_id="design-retry-001",
            title="Backoff strategy for downstream calls",
            kind="design",
            service="checkout",
            team="storefront",
            text=(
                "When a dependency starts refusing work, hammering it is the "
                "fastest way to keep it down. Callers wait progressively "
                "longer between attempts, with a random offset so a thousand "
                "clients do not wake up together, and give up entirely once "
                "the wait would exceed the caller's own deadline."
            ),
        ),
        Document(
            doc_id="design-cache-002",
            title="Read-through caching for the catalog",
            kind="design",
            service="catalog",
            team="discovery",
            text=(
                "Reads consult the fast store first and fall through to the "
                "system of record on a miss, populating the fast store on the "
                "way back. The hard part is not the lookup, it is deciding "
                "when an entry has grown too old to trust and who is allowed "
                "to declare that."
            ),
        ),
        Document(
            doc_id="design-idem-003",
            title="Making payment submission safe to repeat",
            kind="design",
            service="payments",
            team="money",
            text=(
                "A client that times out cannot know whether the work "
                "happened. The submission therefore carries a caller-generated "
                "token, and a second arrival bearing a token already seen "
                "returns the first outcome rather than doing the work twice. "
                "The token store is the real design problem."
            ),
        ),
        # ---- EXACT_TERM: lexically distinctive, semantically empty. -------
        Document(
            doc_id="runbook-err-101",
            title="Runbook: ERR_UPSTREAM_4423",
            kind="runbook",
            service="payments",
            team="money",
            text=(
                "ERR_UPSTREAM_4423 is emitted when the settlement gateway "
                "returns a malformed envelope. Check gateway.envelope.strict "
                "in the service config. If it is true, the gateway partner is "
                "sending a legacy format and the flag must be coordinated "
                "before flipping."
            ),
        ),
        Document(
            doc_id="runbook-err-102",
            title="Runbook: ERR_TOKEN_9101",
            kind="runbook",
            service="identity",
            team="platform",
            text=(
                "ERR_TOKEN_9101 means the presented assertion failed audience "
                "validation. The setting is auth.audience.allowlist. Adding an "
                "audience requires a change request because the allowlist is "
                "also read by the offline batch verifier."
            ),
        ),
        Document(
            doc_id="runbook-cfg-103",
            title="Config reference: queue.prefetch.count",
            kind="runbook",
            service="orders",
            team="fulfillment",
            text=(
                "queue.prefetch.count controls how many messages a consumer "
                "holds unacknowledged. Raising it improves throughput and "
                "increases redelivery on a crash. The default of 32 was chosen "
                "before the consumers were made idempotent and has not been "
                "revisited."
            ),
        ),
        # ---- Incidents: rows for the relational leg AND prose. ------------
        Document(
            doc_id="incident-2026-014",
            title="Checkout latency spike during the March promotion",
            kind="incident",
            service="checkout",
            team="storefront",
            quarter="2026Q1",
            minutes_to_resolve=94,
            text=(
                "Response times tripled for forty minutes under promotional "
                "load. The immediate cause was connection pool exhaustion "
                "against the catalog service; the contributing cause was a "
                "retry policy with no jitter amplifying the initial slowdown."
            ),
        ),
        Document(
            doc_id="incident-2026-015",
            title="Duplicate settlement submissions",
            kind="incident",
            service="payments",
            team="money",
            quarter="2026Q1",
            minutes_to_resolve=210,
            text=(
                "A client retried after a gateway timeout and the submission "
                "was processed twice. Idempotency tokens were specified in the "
                "design and not enforced at the boundary."
            ),
        ),
        Document(
            doc_id="incident-2026-021",
            title="Catalog staleness after a bulk import",
            kind="incident",
            service="catalog",
            team="discovery",
            quarter="2026Q2",
            minutes_to_resolve=45,
            text=(
                "A bulk import bypassed the invalidation path, so the fast "
                "store served superseded prices for just under an hour."
            ),
        ),
        Document(
            doc_id="incident-2026-022",
            title="Identity assertion rejections after a partner rotation",
            kind="incident",
            service="identity",
            team="platform",
            quarter="2026Q2",
            minutes_to_resolve=17,
            text=(
                "A partner rotated a signing key without notice and assertions "
                "began failing audience validation until the allowlist was "
                "updated."
            ),
        ),
        Document(
            doc_id="incident-2026-030",
            title="Order consumer redelivery storm",
            kind="incident",
            service="orders",
            team="fulfillment",
            quarter="2026Q3",
            minutes_to_resolve=133,
            text=(
                "A consumer crash with a large prefetch window caused mass "
                "redelivery. Downstream effects were contained because the "
                "handlers were idempotent."
            ),
        ),
        Document(
            doc_id="incident-2026-031",
            title="Settlement gateway envelope rejections",
            kind="incident",
            service="payments",
            team="money",
            quarter="2026Q3",
            minutes_to_resolve=62,
            text=(
                "The settlement partner began sending a legacy envelope format "
                "and submissions were rejected until the strict flag was "
                "relaxed under a change request."
            ),
        ),
    )


# ------------------------------------------------------- distractors
#
# THE ANCHORS ABOVE CANNOT CARRY A MEASUREMENT ON THEIR OWN, AND THE FIRST
# VERSION OF THIS FILE TRIED TO. Twelve documents means a top-5 result set
# holds 42% of the corpus, so every backend "finds" the answer and recall@5
# cannot tell a leg that ranks it first from one that ranks it fifth. Measured
# against the labels, agreement was 1 of 11.
#
# The documents below are the fix. They are not padding: each group exists to
# make a specific competence hard to fake.
#
#   NEAR-MISS DESIGNS   adjacent topics on other services -- circuit breaking
#                       next to retry, cache warming next to invalidation. They
#                       share the anchor's vocabulary and answer a DIFFERENT
#                       question, so a semantic query has real competition
#                       without the ground truth becoming ambiguous.
#   RUNBOOK DISTRACTORS more identifier-shaped tokens than any query asks for,
#                       which is what gives BM25's idf something to discriminate
#                       with and gives the vector leg realistic confusable noise.
#   INCIDENT ROWS       enough rows across services and quarters that a count or
#                       a mean is a real computation rather than a rounding of
#                       three.

_NEAR_MISS = (
    ("circuit", "orders", "fulfillment", "Circuit breaking for the settlement call",
     "Rather than continuing to attempt a call that keeps failing, the caller "
     "trips a breaker after a threshold and fails fast until a probe succeeds. "
     "This is not the same as waiting longer between attempts and the two are "
     "often confused in review."),
    ("circuit", "notifications", "growth", "Shedding load when the provider degrades",
     "Under provider degradation the sender drops low-priority traffic first "
     "and keeps transactional messages flowing, which is a capacity decision "
     "rather than a timing one."),
    ("cachewarm", "search", "discovery", "Warming the query cache after deploy",
     "A cold cache after a rollout produces a latency cliff, so the top queries "
     "from the previous day are replayed before traffic is admitted. Warming is "
     "about what is present, not about when an entry stops being trustworthy."),
    ("cachewarm", "inventory", "fulfillment", "Negative caching for stock lookups",
     "Absent items are cached as absent for a short window so a hot missing key "
     "does not become a thundering herd against the system of record."),
    ("idem", "billing", "money", "Deduplicating invoice generation",
     "Invoice runs are keyed on a period identifier so a re-run produces no "
     "second document. The dedup key is derived rather than caller-supplied, "
     "which is the difference from a submission token."),
    ("dedupe", "orders", "fulfillment", "Detecting duplicate order submissions",
     "Duplicate detection here is a fuzzy match over recent orders from the "
     "same account rather than an exact token, and it trades false positives "
     "for coverage."),
)

_DESIGN_TOPICS = (
    ("capacity", "Capacity planning for {svc}",
     "Headroom is sized against the ninety-fifth percentile of the last four "
     "weeks rather than the mean, because the mean hides the shape that "
     "actually causes saturation."),
    ("deploy", "Progressive rollout for {svc}",
     "Changes reach one percent of traffic, then ten, then everything, with an "
     "automatic halt on error-rate regression measured against the previous "
     "window."),
    ("schema", "Schema evolution in {svc}",
     "Columns are added nullable and backfilled separately so a deploy never "
     "waits on a migration. Removals go through a deprecation window during "
     "which readers must tolerate both shapes."),
    ("flags", "Feature flag hygiene in {svc}",
     "Every flag carries an owner and an expiry. A flag past expiry fails the "
     "build, because the cost of a flag is not the branch, it is the untested "
     "combination of branches."),
    ("ratelimit", "Rate limiting policy for {svc}",
     "Limits are per-principal rather than per-connection, since a single "
     "misbehaving client behind a pool would otherwise consume the budget of "
     "everyone sharing it."),
    ("health", "Health checks in {svc}",
     "A liveness probe answers whether the process should be restarted and a "
     "readiness probe answers whether it should receive traffic. Conflating "
     "them produces restart loops under dependency failure."),
    ("logs", "Log retention for {svc}",
     "Structured events are kept hot for fourteen days and cold for a year. "
     "The retention decision is driven by the investigation window, not by "
     "storage cost, which is smaller than people assume."),
    ("access", "Access review cadence for {svc}",
     "Standing grants are re-attested quarterly and anything unattested is "
     "revoked automatically rather than escalated, because an escalation path "
     "with no deadline is a permanent grant."),
    ("oncall", "On-call rotation for {svc}",
     "A week of primary is followed by two weeks off rotation. Handover "
     "requires an explicit written state of anything unresolved."),
    ("dashboard", "Service dashboard conventions for {svc}",
     "The top row answers whether the service is healthy for users; everything "
     "below it is for diagnosis. Ordering panels by implementation detail is "
     "the most common way a dashboard becomes unreadable during an incident."),
)

_SERVICES = (
    ("checkout", "storefront"), ("catalog", "discovery"), ("payments", "money"),
    ("identity", "platform"), ("orders", "fulfillment"), ("search", "discovery"),
    ("notifications", "growth"), ("billing", "money"), ("inventory", "fulfillment"),
    ("shipping", "fulfillment"),
)

_RUNBOOK_CODES = (
    ("ERR_QUEUE_2201", "orders", "fulfillment", "queue.visibility.timeout",
     "the consumer failed to extend a visibility window before it lapsed"),
    ("ERR_STOCK_3310", "inventory", "fulfillment", "stock.reservation.ttl",
     "a reservation expired between the cart step and the order step"),
    ("ERR_SHIP_5504", "shipping", "fulfillment", "carrier.rate.cachesec",
     "a cached carrier rate was quoted after the carrier withdrew it"),
    ("ERR_SEARCH_6612", "search", "discovery", "index.refresh.interval",
     "a document was queried before the refresh interval had elapsed"),
    ("ERR_NOTIFY_7708", "notifications", "growth", "sender.retry.ceiling",
     "the provider returned a soft bounce more times than the ceiling allows"),
    ("ERR_BILL_8814", "billing", "money", "invoice.period.lockday",
     "an adjustment arrived after the period was locked"),
    ("ERR_CAT_1102", "catalog", "discovery", "catalog.price.precision",
     "a price arrived with more precision than the schema permits"),
    ("ERR_ID_9920", "identity", "platform", "session.idle.maxminutes",
     "a session was presented after the idle ceiling"),
    ("ERR_PAY_4431", "payments", "money", "settlement.batch.size",
     "a settlement batch exceeded the partner's declared maximum"),
    ("ERR_CHK_2288", "checkout", "storefront", "cart.merge.strategy",
     "two carts for the same principal could not be merged automatically"),
    ("ERR_SEARCH_6613", "search", "discovery", "query.expansion.depth",
     "synonym expansion produced more clauses than the parser accepts"),
    ("ERR_INV_3311", "inventory", "fulfillment", "stock.count.tolerance",
     "a cycle count differed from the ledger by more than tolerance"),
    ("ERR_NOTIFY_7709", "notifications", "growth", "template.locale.fallback",
     "no template existed for the locale and no fallback was configured"),
    ("ERR_SHIP_5505", "shipping", "fulfillment", "label.void.window",
     "a label was voided outside the carrier's permitted window"),
    ("ERR_BILL_8815", "billing", "money", "tax.jurisdiction.resolver",
     "an address resolved to two overlapping tax jurisdictions"),
)


def _near_miss_designs() -> tuple[Document, ...]:
    return tuple(
        Document(
            doc_id=f"design-{tag}-{svc[:3]}",
            title=title,
            kind="design",
            service=svc,
            team=team,
            text=text,
        )
        for tag, svc, team, title, text in _NEAR_MISS
    )


def _general_designs() -> tuple[Document, ...]:
    """One design doc per (topic, service) pair, walked deterministically.

    The offset keeps the pairing from lining every topic up with the same
    service, which would let a query about one service accidentally select a
    whole topic band.
    """
    out = []
    for i, (tag, title, text) in enumerate(_DESIGN_TOPICS):
        for j in range(2):
            svc, team = _SERVICES[(i * 3 + j * 7) % len(_SERVICES)]
            out.append(
                Document(
                    doc_id=f"design-{tag}-{svc}",
                    title=title.format(svc=svc),
                    kind="design",
                    service=svc,
                    team=team,
                    text=text,
                )
            )
    # The walk can repeat a (topic, service) pair; ids are unique by
    # construction so dedup on the id rather than trusting the arithmetic.
    seen: dict[str, Document] = {}
    for doc in out:
        seen.setdefault(doc.doc_id, doc)
    return tuple(seen.values())


def _runbook_distractors() -> tuple[Document, ...]:
    return tuple(
        Document(
            doc_id=f"runbook-{code.lower()}",
            title=f"Runbook: {code}",
            kind="runbook",
            service=svc,
            team=team,
            text=(
                f"{code} is raised when {cause}. The relevant setting is "
                f"{key}. Changing it requires the owning team to confirm the "
                f"downstream effect first."
            ),
        )
        for code, svc, team, key, cause in _RUNBOOK_CODES
    )


# Incident rows: (service, quarter, minutes, one-line cause). Written out
# rather than randomized so the aggregates are stable and checkable by hand.
_INCIDENT_ROWS = (
    ("checkout", "2026Q1", 38, "a slow dependency held connections open"),
    ("checkout", "2026Q2", 71, "a rollout doubled payload size unnoticed"),
    ("checkout", "2026Q4", 22, "a flag flip changed cart merge behavior"),
    ("catalog", "2026Q1", 55, "a reindex ran during peak"),
    ("catalog", "2026Q3", 88, "a price feed delivered stale rows"),
    ("catalog", "2026Q4", 30, "an import job double-counted a supplier"),
    ("payments", "2026Q2", 145, "a partner certificate expired"),
    ("payments", "2026Q4", 96, "a batch exceeded the declared maximum"),
    ("identity", "2026Q1", 27, "a token cache was invalidated wholesale"),
    ("identity", "2026Q3", 64, "a clock skew broke assertion validation"),
    ("identity", "2026Q4", 41, "a session ceiling was lowered without notice"),
    ("orders", "2026Q1", 112, "a consumer lag alarm was routed to nobody"),
    ("orders", "2026Q2", 49, "a reservation expiry raced the order step"),
    ("orders", "2026Q4", 77, "a retry storm followed a partial outage"),
    ("search", "2026Q1", 33, "an index refresh fell behind ingestion"),
    ("search", "2026Q2", 58, "a synonym change widened queries unexpectedly"),
    ("search", "2026Q3", 19, "a shard rebalance saturated a node"),
    ("search", "2026Q4", 44, "a query expansion depth change hit the parser"),
    ("notifications", "2026Q1", 26, "a provider soft-bounced a whole domain"),
    ("notifications", "2026Q2", 83, "a template locale had no fallback"),
    ("notifications", "2026Q3", 37, "a send window overlapped a maintenance"),
    ("billing", "2026Q1", 104, "an adjustment arrived after a period lock"),
    ("billing", "2026Q2", 61, "a tax resolver returned two jurisdictions"),
    ("billing", "2026Q3", 150, "an invoice run repeated after a restart"),
    ("billing", "2026Q4", 35, "a currency rounding rule changed midcycle"),
    ("inventory", "2026Q1", 47, "a cycle count exceeded tolerance"),
    ("inventory", "2026Q2", 29, "a negative cache masked a real restock"),
    ("inventory", "2026Q3", 92, "a ledger and a warehouse feed disagreed"),
    ("shipping", "2026Q1", 68, "a carrier withdrew a rate mid-quote"),
    ("shipping", "2026Q2", 53, "a label void window was missed"),
    ("shipping", "2026Q3", 121, "a carrier API deprecated a field silently"),
    ("shipping", "2026Q4", 40, "an address normalizer rejected valid input"),
)


def _incident_rows() -> tuple[Document, ...]:
    return tuple(
        Document(
            doc_id=f"incident-{quarter.lower()}-{svc[:3]}-{i:02d}",
            title=f"{svc} incident in {quarter}",
            kind="incident",
            service=svc,
            team=dict(_SERVICES)[svc],
            quarter=quarter,
            minutes_to_resolve=mins,
            text=f"An operational event in {svc}: {cause}.",
        )
        for i, (svc, quarter, mins, cause) in enumerate(_INCIDENT_ROWS)
    )


def all_documents() -> tuple[Document, ...]:
    """The anchors plus every distractor group, deduplicated by id."""
    seen: dict[str, Document] = {}
    for group in (
        documents(),
        _near_miss_designs(),
        _general_designs(),
        _runbook_distractors(),
        _incident_rows(),
    ):
        for doc in group:
            if doc.doc_id in seen:
                raise ValueError(f"duplicate doc_id {doc.doc_id!r}")
            seen[doc.doc_id] = doc
    return tuple(seen.values())


# -------------------------------------------------------------------- graph
#
# THE ANSWER TO A RELATIONSHIP QUESTION APPEARS IN NO DOCUMENT. That is the
# whole point of the graph leg and the reason these edges are data rather than
# prose. "Which team owns a service that checkout depends on" requires two hops
# and is stated nowhere.


def edges() -> tuple[Edge, ...]:
    return (
        Edge("checkout", "depends_on", "catalog"),
        Edge("checkout", "depends_on", "payments"),
        Edge("payments", "depends_on", "identity"),
        Edge("orders", "depends_on", "payments"),
        Edge("orders", "depends_on", "identity"),
        Edge("catalog", "depends_on", "identity"),
        Edge("checkout", "owned_by", "storefront"),
        Edge("catalog", "owned_by", "discovery"),
        Edge("payments", "owned_by", "money"),
        Edge("identity", "owned_by", "platform"),
        Edge("orders", "owned_by", "fulfillment"),
        Edge("incident-2026-014", "touched_by", "checkout"),
        Edge("incident-2026-014", "touched_by", "catalog"),
        Edge("incident-2026-015", "touched_by", "payments"),
        Edge("incident-2026-021", "touched_by", "catalog"),
        Edge("incident-2026-022", "touched_by", "identity"),
        Edge("incident-2026-030", "touched_by", "orders"),
        Edge("incident-2026-031", "touched_by", "payments"),
        # The services added with the distractor corpus. A graph that covered
        # only the five anchor services would make every relationship question
        # about the other five silently unanswerable, and the router would look
        # correct while returning nothing.
        Edge("search", "depends_on", "catalog"),
        Edge("search", "depends_on", "identity"),
        Edge("notifications", "depends_on", "identity"),
        Edge("billing", "depends_on", "payments"),
        Edge("billing", "depends_on", "identity"),
        Edge("inventory", "depends_on", "orders"),
        Edge("shipping", "depends_on", "orders"),
        Edge("shipping", "depends_on", "inventory"),
        Edge("search", "owned_by", "discovery"),
        Edge("notifications", "owned_by", "growth"),
        Edge("billing", "owned_by", "money"),
        Edge("inventory", "owned_by", "fulfillment"),
        Edge("shipping", "owned_by", "fulfillment"),
    ) + _incident_edges()


def _incident_edges() -> tuple[Edge, ...]:
    """One `touched_by` edge per generated incident row.

    Without these the graph knows about six incidents and the relational store
    knows about thirty-eight, so any question joining the two -- "incidents
    touching services owned by X" -- would be answered against a tenth of the
    data while looking complete.
    """
    return tuple(
        Edge(doc.doc_id, "touched_by", doc.service)
        for doc in _incident_rows()
        if doc.service
    )


# ----------------------------------------------------------- labeled queries
#
# `required` is the ground truth: the backends without which the answer is not
# obtainable. `sufficient` is the weaker set: backends that would also get
# there alone. A router picking a sufficient backend is not wrong, only more
# expensive, and the metrics keep those apart.


def labeled_queries() -> tuple[LabeledQuery, ...]:
    return (
        # ---- SEMANTIC ----------------------------------------------------
        LabeledQuery(
            query_id="q-sem-1",
            text="how should a caller behave when a dependency starts failing",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"design-retry-001"}),
        ),
        LabeledQuery(
            query_id="q-sem-2",
            text="what stops the same payment being taken twice",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"design-idem-003", "incident-2026-015"}),
        ),
        LabeledQuery(
            query_id="q-sem-3",
            text="how do we decide that cached data is too old to use",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"design-cache-002"}),
        ),
        # ---- EXACT_TERM --------------------------------------------------
        LabeledQuery(
            query_id="q-exact-1",
            text="ERR_UPSTREAM_4423",
            competences=frozenset({Competence.EXACT_TERM}),
            required=frozenset({Backend.FULLTEXT}),
            relevant_docs=frozenset({"runbook-err-101"}),
        ),
        LabeledQuery(
            query_id="q-exact-2",
            text="what does queue.prefetch.count do",
            competences=frozenset({Competence.EXACT_TERM}),
            required=frozenset({Backend.FULLTEXT}),
            relevant_docs=frozenset({"runbook-cfg-103"}),
        ),
        LabeledQuery(
            query_id="q-exact-3",
            text="auth.audience.allowlist",
            competences=frozenset({Competence.EXACT_TERM}),
            required=frozenset({Backend.FULLTEXT}),
            relevant_docs=frozenset({"runbook-err-102"}),
        ),
        # ---- RELATIONSHIP: the answer is in no document. -----------------
        LabeledQuery(
            query_id="q-rel-1",
            text="which teams own the services checkout depends on",
            competences=frozenset({Competence.RELATIONSHIP}),
            required=frozenset({Backend.GRAPH}),
            relevant_docs=frozenset(),
        ),
        LabeledQuery(
            query_id="q-rel-2",
            text="what does payments depend on transitively",
            competences=frozenset({Competence.RELATIONSHIP}),
            required=frozenset({Backend.GRAPH}),
            relevant_docs=frozenset(),
        ),
        LabeledQuery(
            query_id="q-rel-3",
            text="which services would be affected if identity went down",
            competences=frozenset({Competence.RELATIONSHIP}),
            required=frozenset({Backend.GRAPH}),
            relevant_docs=frozenset(),
        ),
        # ---- AGGREGATE: the answer is a number in no document. -----------
        LabeledQuery(
            query_id="q-agg-1",
            text="how many incidents did payments have in 2026",
            competences=frozenset({Competence.AGGREGATE}),
            required=frozenset({Backend.RELATIONAL}),
            relevant_docs=frozenset(),
        ),
        LabeledQuery(
            query_id="q-agg-2",
            text="average time to resolve incidents in 2026Q3",
            competences=frozenset({Competence.AGGREGATE}),
            required=frozenset({Backend.RELATIONAL}),
            relevant_docs=frozenset(),
        ),
        LabeledQuery(
            query_id="q-agg-3",
            text="count incidents per quarter",
            competences=frozenset({Competence.AGGREGATE}),
            required=frozenset({Backend.RELATIONAL}),
            relevant_docs=frozenset(),
        ),
        # ---- GENUINELY MULTI-BACKEND. Two legs, both required. -----------
        LabeledQuery(
            query_id="q-multi-1",
            text=(
                "how many incidents touched services owned by the money team "
                "in 2026Q3"
            ),
            competences=frozenset({Competence.RELATIONSHIP, Competence.AGGREGATE}),
            required=frozenset({Backend.GRAPH, Backend.RELATIONAL}),
            relevant_docs=frozenset(),
        ),
        LabeledQuery(
            query_id="q-multi-2",
            text=(
                "what went wrong with retries in the incident that hit "
                "checkout and catalog together"
            ),
            competences=frozenset({Competence.RELATIONSHIP, Competence.SEMANTIC}),
            required=frozenset({Backend.GRAPH, Backend.VECTOR}),
            relevant_docs=frozenset({"incident-2026-014", "design-retry-001"}),
        ),
        # ---- TRAPS. Look like one competence, are another. ---------------
        LabeledQuery(
            query_id="q-trap-1",
            text="why do we keep seeing ERR_TOKEN_9101 after partner rotations",
            competences=frozenset({Competence.SEMANTIC, Competence.EXACT_TERM}),
            required=frozenset({Backend.FULLTEXT, Backend.VECTOR}),
            relevant_docs=frozenset({"runbook-err-102", "incident-2026-022"}),
            trap=(
                "carries an error code, so it looks purely lexical, but the "
                "question is causal and the answer is split across the runbook "
                "and the incident narrative"
            ),
        ),
        LabeledQuery(
            query_id="q-trap-2",
            text="how many retries does the backoff design recommend",
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"design-retry-001"}),
            trap=(
                "opens with 'how many', which is the aggregate tell, but no "
                "counting is involved -- the answer is a sentence in a design "
                "document and the relational leg has nothing to offer"
            ),
        ),
        LabeledQuery(
            query_id="q-trap-3",
            text="which service owns gateway.envelope.strict",
            competences=frozenset({Competence.EXACT_TERM}),
            required=frozenset({Backend.FULLTEXT}),
            sufficient=frozenset(),
            relevant_docs=frozenset({"runbook-err-101"}),
            trap=(
                "says 'owns', which is the ownership-edge tell, but the "
                "subject is a config key rather than a service, and the graph "
                "has no node for it -- a relationship router walks into an "
                "empty traversal and returns nothing"
            ),
        ),
        LabeledQuery(
            query_id="q-trap-5",
            text=(
                "how do checkout and catalog differ in the way they handle "
                "stale data"
            ),
            competences=frozenset({Competence.SEMANTIC}),
            required=frozenset({Backend.VECTOR}),
            relevant_docs=frozenset({"design-cache-002", "incident-2026-021"}),
            trap=(
                "names TWO services that are both graph nodes, which is the "
                "strongest available relationship signal short of a keyword, "
                "and asks a purely semantic question about design prose. It "
                "exists to catch a router that treats co-occurring entities as "
                "a traversal cue -- written as the counter-example to a rule "
                "that scored perfectly without it"
            ),
        ),
        LabeledQuery(
            query_id="q-trap-4",
            text="what is the longest an incident took to resolve",
            competences=frozenset({Competence.AGGREGATE}),
            required=frozenset({Backend.RELATIONAL}),
            relevant_docs=frozenset(),
            trap=(
                "phrased as a superlative rather than a count, so keyword "
                "aggregate tells like 'how many' and 'average' do not fire"
            ),
        ),
    )


# A REAL QUERY LOG IS NOT BALANCED, AND THE EVALUATION SET DELIBERATELY IS.
#
# The labeled set above holds roughly equal numbers of each competence, because
# a diagnostic set has to exercise every backend enough to say anything about
# it. A PRODUCTION query log looks nothing like that: it is dominated by
# ordinary semantic lookups, with exact-term next, and relationship and
# aggregate questions a small tail.
#
# THAT DIFFERENCE IS NOT A DETAIL, IT DECIDES THE HEADLINE. Scored on the
# balanced set, a vector-only baseline looks catastrophic. Weighted by a
# realistic mix, it looks mostly fine -- and the honest claim this project can
# make is the second one, with the failing tail named. Reporting only the
# balanced number would sell federation far harder than the evidence supports.
#
# These weights are an ASSUMPTION, not a measurement. They are stated here so a
# reader can substitute their own, and the demo reports both scorings side by
# side rather than picking one.
PRODUCTION_MIX: dict[Competence, float] = {
    Competence.SEMANTIC: 0.70,
    Competence.EXACT_TERM: 0.20,
    Competence.RELATIONSHIP: 0.06,
    Competence.AGGREGATE: 0.04,
}


@dataclass(frozen=True)
class Corpus:
    documents: tuple[Document, ...]
    edges: tuple[Edge, ...]
    queries: tuple[LabeledQuery, ...]

    def by_id(self, doc_id: str) -> Document:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)


def build_corpus() -> Corpus:
    return Corpus(
        documents=all_documents(), edges=edges(), queries=labeled_queries()
    )
