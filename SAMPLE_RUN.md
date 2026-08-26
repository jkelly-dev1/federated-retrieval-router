# Sample run

Verbatim captures of `scripts/run_demo.py`, the gate and the test suite, so a
reviewer without an API key can see exactly what this project measures and what
it refuses to claim. Nothing here is retyped or cleaned up. Offline captures
2026-08-02; the paid capture 2026-08-03.

Every number in the first three sections comes from the deterministic offline
stack: a 256-dimension hash embedder and an 85-document in-memory corpus. That
makes them evidence about the harness and about the routing logic, and evidence
about nothing else. Section 4 of the demo measures the offline instrument's own
shortfall, which is otherwise only implied.

[The paid capture](#the-paid-capture) is the one section with a real embedding
model and two real routers in it. It is a capture, not a baseline: the gate
does not read it, no test asserts on it, and re-running it will produce
different numbers. It is here because three of this project's questions cannot
be answered offline at all, and one of them is how much a re-run moves, so
every router is run five times and the corpus is embedded twice.

What is and is not deterministic. Two runs of `scripts/run_demo.py` produce
BYTE-IDENTICAL output, which `test_the_demo_is_byte_identical_across_runs`
enforces. There is no timestamp, no randomness and no network anywhere in the
offline path.

Contents

- [The demonstration](#the-demonstration)
- [The gate](#the-gate)
- [The test suite](#the-test-suite)
- [The paid capture](#the-paid-capture)
- [The same capture, taken twice](#the-same-capture-taken-twice)
- [The store comparison](#the-store-comparison)
- [What the captures show](#what-the-captures-show)

## The demonstration

```
python scripts/run_demo.py
```

```
==============================================================================
federated-retrieval-router demo
==============================================================================
  documents                              85
  graph edges                            63
  labeled queries                        19
  of those, traps                        5
  of those, genuinely multi-backend      3
  top-5 as a fraction of the corpus      5.9%

  THE LAST LINE IS A PRECONDITION, NOT TRIVIA. At 12 documents a
  top-5 result set is 42% of everything, every backend "finds"
  every answer, and the labels agree with measurement on 1 query
  out of 11. A retrieval evaluation whose candidate pool is half
  the corpus measures nothing, and it fails silently.

  The corpus is synthetic. Services, teams and incidents are
  fictional and no real operational data is present.

==============================================================================
1. Four backends, four competences
==============================================================================
  vector      vector(hashing-256d-router-v1)
  fulltext    fulltext(bm25 k1=1.2 b=0.75)
  graph       graph(63 edges, max 3 hops)
  relational  relational(38 rows)

------------------------------------------------------------------------------
  EXACT_TERM    ERR_UPSTREAM_4423
------------------------------------------------------------------------------
    vector      design-idem-003, runbook-err-101
    fulltext    runbook-err-101
    graph       (nothing)
    relational  (nothing)

------------------------------------------------------------------------------
  SEMANTIC      how should a caller behave when a dependency starts failing
------------------------------------------------------------------------------
    vector      design-health-search, design-health-payments
    fulltext    design-retry-001, design-health-payments
    graph       (nothing)
    relational  (nothing)

------------------------------------------------------------------------------
  RELATIONSHIP  which teams own the services checkout depends on
------------------------------------------------------------------------------
    vector      incident-2026q1-che-00, incident-2026q2-che-01
    fulltext    incident-2026q1-che-00, incident-2026q2-che-01
    graph       catalog, payments
    relational  (nothing)

------------------------------------------------------------------------------
  AGGREGATE     how many incidents did payments have in 2026
------------------------------------------------------------------------------
    vector      incident-2026q2-pay-06, runbook-cfg-103
    fulltext    runbook-cfg-103, incident-2026q2-pay-06
    graph       identity, money
    relational  agg:count:payments:2026

  READ THE AGGREGATE ROW TWICE. The relational leg answers with an
  `agg:` row, which is a NUMBER THAT APPEARS IN NO DOCUMENT. The
  other three return documents, confidently, and none of them
  contains the count. An evaluation scoring that leg by document
  overlap would report zero for the only backend that answered.

==============================================================================
2. Routing, and what correctness costs
==============================================================================
  router          balanced  prod-mix  fan-out   traps
  heuristic          0.947     0.931     1.53    5/5
  fan-out            1.000     1.000     4.00    5/5
  vector-only        0.263     0.634     1.00    2/5

  FAN-OUT SCORES A PERFECT 1.000 BY CONSTRUCTION and queries every
  store every time. It is in the table because a correctness number
  reported without its cost beside it is not a result.

------------------------------------------------------------------------------
  Per backend, heuristic router
------------------------------------------------------------------------------
  backend       chosen  needed   prec  recall  over  miss
  vector            15       7   0.47    1.00     8     0
  graph              4       5   1.00    0.80     0     1
  fulltext           5       5   1.00    1.00     0     0
  relational         5       5   1.00    1.00     0     0

  Where it still misroutes:
    q-multi-2: needed ['graph', 'vector'], chose ['vector']
      what went wrong with retries in the incident that hit checkout ...

==============================================================================
3. The query mix decides the verdict
==============================================================================
  vector-only on the balanced evaluation set    0.263
  vector-only weighted to a production mix      0.634
  ratio                                         2.4x

  assumed production mix:
    semantic       70%
    exact_term     20%
    relationship   6%
    aggregate      4%

  SAME ROUTER, SAME CORPUS, SAME CODE. The only thing that changed
  is an assumption about what people ask. A balanced evaluation set
  is necessary to say anything per backend and it OVERSTATES the
  case for federation by this ratio, because real query logs are
  dominated by ordinary semantic lookups.

  The weights above are an assumption and not a measurement. They
  are printed so a reader can substitute their own, and both
  scorings are reported rather than whichever is more flattering.

==============================================================================
4. Does the backend a question calls for actually win?
==============================================================================
  queries with a document answer         11
  designed backend actually wins         7
  no backend retrieves the answer        2

    q-sem-1      designed=['vector'] measured=['fulltext']
    q-sem-2      designed=['vector'] measured=['fulltext']
    q-multi-2    designed=['graph', 'vector'] measured=['NOBODY']
    q-trap-5     designed=['vector'] measured=['NOBODY']

  THIS IS A LIMIT OF THE OFFLINE INSTRUMENT AND IT IS REPORTED, NOT
  HIDDEN. The embedder is a bag-of-tokens hash. On a paraphrase that
  shares even one RARE term with its target -- 'dependency',
  'backoff', 'twice' -- BM25's idf carries more signal than the whole
  vector carries cosine, so the fulltext leg wins queries the vector
  leg is supposed to own. Measured overlap on those queries is 0.02
  to 0.10, so they are genuine paraphrases and the win is real.

  `required` is therefore a DESIGNED ground truth about question
  shape, not a claim about which store wins here. Scoring routing
  against measured winners would define the router's job as
  predicting the mock, and the project would score well by learning
  an artifact. Closing this gap is what the real-model run is for.

==============================================================================
5. Fusion, and the window that silently decides it
==============================================================================
  query: why do we keep seeing ERR_TOKEN_9101 after partner rotations

    vector      incident-2026-022#1, design-circuit-ord#2
    fulltext    incident-2026-022#1, runbook-err-102#2
    graph       (nothing)
    relational  (nothing)

  rank  document                   score  contributors
     1  incident-2026-022         0.0328  full,vect
     2  runbook-err-102           0.0315  full,vect
     3  incident-2026-031         0.0310  full,vect
     4  design-cachewarm-sea      0.0304  full,vect
     5  design-retry-001          0.0303  full,vect

  found by exactly one backend: none

------------------------------------------------------------------------------
  Window sweep: how deep fusion has to look
------------------------------------------------------------------------------
   window  relevant found  fused size
        1               0           1
        2               1           3
        3               1           5
        5               1           7
       10               1          10
       20               1          10
       50               1          10

  DEFAULT_WINDOW is 20. A document one leg ranks deep
  and another ranks first cannot be fused at all if the window ends
  above it, and the merged list looks perfectly reasonable while it
  happens. That is why the window is measured rather than chosen.

==============================================================================
What this demo does NOT claim
==============================================================================
  - The embedder is a hash model, not a semantic one. Every vector
    number here is a FLOOR and section 4 measures the shortfall.
  - 85 synthetic documents. Selectivity is adequate for this query
    set and says nothing about behavior at corpus scale.
  - The production mix in section 3 is an assumption, not a
    measurement from any real query log.
  - Entity linking for the graph leg is substring matching. Routing
    is what is measured here; linking is a separate problem.
  - No latency or cost figures. Fan-out is reported as backends per
    query, which is the part that does not depend on hardware.
==============================================================================
```

## The gate

The gate protects the MEASUREMENT rather than a product, and fails in both
directions. Two of its checks exist because of defects this project already
had: the corpus was once too small to discriminate at all, and the offline
embedder loses paraphrases to BM25 in a way that must not silently disappear.

```
python -m router.gate ; echo "exit=$?"
```

```
==============================================================================
federated-retrieval-router gate: deterministic mock, offline
==============================================================================
  [PASS] the corpus is selective enough to measure anything              top-5 is 5.9% of 85 docs (ceiling 15%)
  [PASS] every required backend returns results for its queries          all labels answerable
  [PASS] the heuristic router routes correctly                           0.947 (floor 0.9)
  [PASS] a single store still fails the other three competences          vector-only 0.263 (ceiling 0.6)
  [PASS] correctness is not free: fan-out is perfect and costs the most  fan-out 1.000 at 4.00 backends/query vs heuristic 0.947 at 1.53
  [PASS] the traps still trap a single-store baseline                    vector-only 2/5, heuristic 5/5
  [PASS] the query mix still changes the verdict                         vector-only 0.263 balanced vs 0.634 prod-mix, a 2.4x swing
  [PASS] the offline embedder still loses paraphrases to BM25            designed backend wins 7/11; a clean sweep would mean the mock got semantic or the queries stopped being paraphrases
  [PASS] the fusion window is deep enough to fuse what the legs return   recall appears at window 1, default is 20

GATE PASSED (9 checks)
  routing is still measurable, a single store still loses, and the
  corpus can still tell one competence from another.
exit=0
```

## The test suite

```
pytest -q
```

```
...........................................................sssssssssss.. [ 54%]
...........................................................              [100%]
136 passed, 16 skipped in 0.84s
```

That capture is from a fresh virtual environment holding nothing but `pytest`.
The `pip install -r requirements.txt && pytest` path a reader would take. The
suite installs no optional dependency and makes no network call.

Its 16 skips are the design, not a gap. Thirteen are the real-store integration
tests in `tests/test_integration.py` and three are the partial- comparison tests
in `tests/test_compare_stores.py`. All of them skip with a reason naming what
was missing rather than failing because a container is not running. A red suite
that means "you did not start Docker" teaches people to ignore red suites.

In the development environment, where `duckdb` is installed, the same suite
reports 145 passed, 7 skipped: the six DuckDB integration tests and the three
partial-comparison tests run instead of skipping, and the seven that need
pgvector or Elasticsearch containers still skip. Same tests, different
environment, and the difference is visible in the count rather than hidden.

The suite is run in both environments before publication, and that is not
ceremony. A sibling repository shipped provider tests that constructed live SDK
clients, passed on the maintainer's machine and failed on all three Python
versions in CI. `tests/test_providers.py` and `tests/test_adapters.py` now pin
the behavior that differs: every optional driver is hidden from the import
system in a test, so the result is identical whether or not it happens to be
installed.

## The paid capture

```
STABILITY_RUNS=5 ENV_FILE=~/.secrets/ai.env python scripts/real_run.py
```

Run on 2026-08-03 against `text-embedding-3-small` at 1536 dimensions,
`gpt-5.6-terra` and `claude-opus-5`. Five independent routing passes per
router, and the corpus embedded twice through two cold caches. Model replies
are quoted exactly as returned.

This capture was taken twice. The whole thing was re-run about forty minutes
after the first one, and the second sample is the one printed below. [What
changed between them](#the-same-capture-taken-twice) is the sharpest caveat in
this file, so it has its own section.

```
==============================================================================
federated-retrieval-router REAL MODEL RUN
==============================================================================
  embedding provider                     openai
  embedder                               cached(openai-text-embedding-3-small-1536d)
  llm routers                            ['llm-openai', 'llm-anthropic']
  documents / queries                    85 / 19
  routing passes per router              5

==============================================================================
1. Does a real embedder reclaim the queries it is supposed to own?
==============================================================================
  embedder   dims  designed backend wins   nobody retrieves it
  mock        256                   7/11                  2/11
  real       1536                   9/11                  1/11

  query                              mock                       real
  q-sem-1                        fulltext            fulltext,vector *
  q-sem-2                        fulltext                   fulltext  
  q-sem-3                 fulltext,vector            fulltext,vector  
  q-exact-1               fulltext,vector            fulltext,vector  
  q-exact-2               fulltext,vector            fulltext,vector  
  q-exact-3               fulltext,vector            fulltext,vector  
  q-multi-2                        NOBODY                     NOBODY  
  q-trap-1                       fulltext            fulltext,vector *
  q-trap-2                fulltext,vector            fulltext,vector  
  q-trap-3                       fulltext            fulltext,vector *
  q-trap-5                         NOBODY                     vector *

  Rows marked * are where the embedder changed the answer. The
  designed backend for every q-sem row is `vector`; offline the
  fulltext leg takes them, because one rare shared term carries more
  BM25 signal than the whole bag-of-tokens vector carries cosine.

==============================================================================
1b. Do those numbers reproduce? One repeat, cold cache
==============================================================================
  texts embedded in both passes          104
  bit-identical vectors                  76/104
  largest component difference           3.357e-03
  the table in section 1 is              unchanged

  This is the reproducibility claim, measured. The routing section
  below is where a live model is NOT reproducible, and the two are
  reported separately because they fail differently.

==============================================================================
2. Does a model route better than two hand-written guards? (n=5)
==============================================================================
  Every cell is min/median/max across the 5 runs. The three
  deterministic routers are re-run too: they are the control, and a
  spread on any of their rows would mean the instrument moved.

  router                  balanced          prod-mix         fan-out  traps
  heuristic      0.947/0.947/0.947 0.931/0.931/0.931  1.53/1.53/1.53    5/5
  fan-out        1.000/1.000/1.000 1.000/1.000/1.000  4.00/4.00/4.00    5/5
  vector-only    0.263/0.263/0.263 0.634/0.634/0.634  1.00/1.00/1.00    2/5
  llm-openai     0.895/0.895/0.895 0.850/0.882/0.882  1.05/1.11/1.11  3-4/5
  llm-anthropic  1.000/1.000/1.000 1.000/1.000/1.000  1.21/1.21/1.21    5/5

  PER-QUERY STABILITY. How many of the 19 queries got the same set of
  backends in every run. A router that flips between one backend and
  three on the same question is an operational problem whatever its
  mean correctness says.

  router          same set every run  least stable query
  heuristic                    19/19  --
  fan-out                      19/19  --
  vector-only                  19/19  --
  llm-openai                   17/19  q-multi-2: 2 sets, fan-out 1 to 2
  llm-anthropic                19/19  --

------------------------------------------------------------------------------
  llm-openai: the queries it did not answer the same way twice
------------------------------------------------------------------------------
    q-multi-2  2 sets, fan-out 1 to 2
      q: what went wrong with retries in the incident that hit checkout an...
        4x  graph,vector
        1x  vector
    q-trap-3  2 sets
      q: which service owns gateway.envelope.strict
        4x  graph
        1x  fulltext

------------------------------------------------------------------------------
  llm-anthropic: the queries it did not answer the same way twice
------------------------------------------------------------------------------
    every query identical across all 5 runs

------------------------------------------------------------------------------
  llm-openai: where it disagreed with the labels
------------------------------------------------------------------------------
    q-multi-2: wrong in 1 of 5 runs, needed ['graph', 'vector']
      q: what went wrong with retries in the incident that hit checkout an...
      chose vector
      llm-openai: The question seeks a prose explanation of retry behavior ...
    q-trap-1: wrong in 5 of 5 runs, needed ['fulltext', 'vector']
      q: why do we keep seeing ERR_TOKEN_9101 after partner rotations
      chose fulltext
      llm-openai: The exact error code ERR_TOKEN_9101 is the key lookup ter...
    q-trap-3: wrong in 4 of 5 runs, needed ['fulltext']
      q: which service owns gateway.envelope.strict
      chose graph
      llm-openai: This is an ownership relationship lookup for a specific c...

------------------------------------------------------------------------------
  llm-anthropic: where it disagreed with the labels
------------------------------------------------------------------------------
    no misroutes in 5 runs

  DOES THE PUBLISHED RANKING SURVIVE?

    llm-openai vs heuristic
      DOES NOT HOLD at n=5. correctness: NEVER exceeded it -- behind in 5
      (0.895/0.895/0.895 against 0.947/0.947/0.947). No axis straddles: this
      is a measured outcome rather than a sample-size limit.

    llm-anthropic vs heuristic
      HOLDS at n=5. Its worst run scores 1.000 against 0.947 and its most
      expensive run fans out to 1.21 against 1.53, so the claim is true of
      every run and not only of the mean.

  n=5 DETECTS INSTABILITY AND DOES NOT RANK TWO ROUTERS THAT STRADDLE EACH
  OTHER. Correctness over 19 queries moves in steps of 0.053, five draws is
  not a sample anyone should compute a p-value from, and none is computed. A
  router that NEVER exceeded another is not the same finding as one whose runs
  fall on both sides of it, and the two are printed differently above.

  READ FAN-OUT BESIDE CORRECTNESS. A model that selects every
  backend on every question scores a perfect 1.000 and has beaten
  nothing; it has reinvented the fan-out baseline at a per-query
  cost in latency and tokens that the keyword router does not pay.

  Every decision above was written to
  audit/routing_decisions.json, so these tables can be re-rendered
  offline. Changing the wording of a capture is not a reason to buy
  the routing calls a second time.

==============================================================================
3. What the run consumed, from the vendors' own usage fields
==============================================================================
  embedding calls billed                 208
  embedding input tokens                 6314
  cache hits avoided                     0
  distinct texts embedded                208
  of which the repeat pass                104
  llm-openai calls / in / out          95 / 18845 / 5014
  llm-anthropic calls / in / out          95 / 28470 / 9249

==============================================================================
What this capture does NOT claim
==============================================================================
  - One embedding model, one corpus, 19 queries. Another model's
    geometry is another measurement.
  - n=5 DETECTS INSTABILITY AND DOES NOT RANK TWO CLOSE MODELS.
    Correctness over 19 queries moves in steps of 0.053 and five
    draws is not a sample to compute a p-value from. None is
    computed. Overlapping ranges are reported as indistinguishable.
  - A live model moves between runs. These are captures, not a
    baseline, and the gate reads none of them.
  - The routing prompt describes the backends by competence and
    never names the corpus, its entities or the heuristic's tells.
    A different prompt is a different result.
  - No latency figures. Fan-out is the cost axis that does not
    depend on hardware or on who else is using the API.
==============================================================================
```

## The same capture, taken twice

The re-run above was started for a cosmetic reason, the verdict block's wording,
and it accidentally became the most useful measurement in the file: a second,
independent n=5 sample of the same thing. The two disagree, and the disagreement
is not in the direction anyone would have guessed.

```
                                   first sample      second sample
llm-anthropic  balanced       1.000/1.000/1.000  1.000/1.000/1.000
               fan-out           1.16/1.21/1.21     1.21/1.21/1.21
               same set every run         17/19              19/19
               misroutes in 5 runs            0                  0

llm-openai     balanced       0.895/0.895/0.947  0.895/0.895/0.895
               fan-out           1.11/1.11/1.11     1.05/1.11/1.11
               same set every run         18/19              17/19
               least stable query      q-trap-3         q-multi-2

embedding      bit-identical vectors     74/104             76/104
               largest delta          1.602e-03          3.357e-03
```

What reproduced. `claude-opus-5` scored `1.000` in all ten runs across both
samples, with zero misroutes in 190 routing decisions, and the survival verdict
came out `HOLDS` both times. `gpt-5.6-terra` never once exceeded the heuristic
on correctness in either sample. The embedder returned non-identical vectors for
byte-identical input in both, at about the same rate.

What did not. The per-query stability numbers moved, and one of them moved to a
different query entirely. `claude-opus-5` was unstable on 2 of 19 queries in the
first sample and on none at all in the second. `gpt-5.6-terra`'s least stable
query was `q-trap-3` the first time and `q-multi-2` the second.

So the first sample supported a story that the second one does not. After the
first run it was true, and tempting, to write that both models were least
stable on `q-trap-3`; the config-key query the linked-entity guard was written
for. It is a good story. It did not survive the next five runs: in the second
sample `claude-opus-5` was stable on everything, and `gpt-5.6-terra` flipped on
a different query. Five runs are enough to show that a router is not
deterministic. They are not enough to say which question it will be unreliable
about. That distinction is now stated in the README rather than learned by a
reader who runs it again.

The claims that survive both samples are the ones about the ROUTER: opus-5
holding 1.000 at low fan-out, gpt-5.6-terra never getting ahead of two `if`
statements. The claims that did not survive were the ones about a specific
QUERY. That is the same lesson as the query-mix result one level down: the
aggregate is more portable than the instance.

## The store comparison

```
docker compose up -d
pip install -r requirements-integration.txt
python scripts/compare_stores.py
```

Free: no key, no vendor, no bill. What it needs is infrastructure, and it names
the pieces it did not have rather than quietly comparing fewer legs than it
claims. Run on 2026-08-04 against pgvector/pgvector:pg16 and Elasticsearch
8.15.3 in containers, with DuckDB in-process. All three legs, no skips.

```
==============================================================================
federated-retrieval-router STORE COMPARISON
==============================================================================
  documents / queries                    85 / 19
  legs compared                          3
  top-k                                  5

==============================================================================
1. The same queries through two implementations of each leg
==============================================================================
  Same corpus, same 19 labeled queries, same ground truth, same k.
  The only thing that differs between the two columns is which
  implementation answered.

  ANSWERS AND RECALL ARE OVER THE QUERIES WITH A DOCUMENT ANSWER;
  n/a means this leg answers with computed values that match no
  document, so document overlap is a category error rather than a
  zero. AGREEMENT IS OVER THE QUERIES WHERE EITHER STORE RETURNED
  ANYTHING: two stores that both returned nothing have not agreed.

  leg               answers          recall@k   same set  same order
  relational            n/a    n/a (computed)        7/7         7/7
  vector          4 -> 4 /7    0.571 -> 0.571      16/19       15/19
  fulltext        5 -> 5 /5    1.000 -> 1.000      10/19        5/19

------------------------------------------------------------------------------
  relational: relational(38 rows)
              vs duckdb(38 rows, in-process)
------------------------------------------------------------------------------
    no query here has a document answer, so 'changed hands' is
    undefined; the agreement column below is the whole comparison
    returned a different set on 0 of 7 queries where either store answered

------------------------------------------------------------------------------
  vector: vector(hashing-256d-router-v1)
          vs pgvector(hashing-256d-router-v1, exact scan)
------------------------------------------------------------------------------
    no query changed hands
    returned a different set on 3 of 19 queries where either store answered
      q-agg-2  (neither)
        toy   incident-2026q3-ide-09, incident-2026q1-inv-25, incident-2026q1-che-00, incident-2026q1-cat-03, incident-2026q1-ide-08
        real  incident-2026q3-ide-09, incident-2026q1-inv-25, incident-2026q1-che-00, incident-2026q1-ide-08, incident-2026q3-inv-27
      q-multi-1  (neither)
        toy   incident-2026q3-cat-04, runbook-cfg-103, incident-2026q1-inv-25, incident-2026q1-che-00, incident-2026q1-cat-03
        real  incident-2026q3-cat-04, runbook-cfg-103, incident-2026q1-inv-25, incident-2026q1-che-00, incident-2026q1-ide-08
      q-trap-4  (neither)
        toy   incident-2026q3-inv-27, incident-2026q2-ord-12, incident-2026q3-ide-09, incident-2026q3-shi-30, incident-2026q2-not-19
        real  incident-2026q3-inv-27, incident-2026q2-ord-12, incident-2026q3-ide-09, incident-2026q3-shi-30, incident-2026q2-pay-06

------------------------------------------------------------------------------
  fulltext: fulltext(bm25 k1=1.2 b=0.75)
            vs elasticsearch(bm25, matching analyzer)
------------------------------------------------------------------------------
    no query changed hands
    returned a different set on 9 of 19 queries where either store answered
      q-sem-1  (both)
        toy   design-retry-001, design-health-payments, design-health-search, design-circuit-ord, incident-2026q1-che-00
        real  design-retry-001, design-health-payments, design-health-search, design-circuit-ord, incident-2026-022
      q-rel-1  (neither)
        toy   incident-2026q1-che-00, incident-2026q2-che-01, incident-2026q4-che-02, design-retry-001, design-deploy-checkout
        real  incident-2026q1-che-00, incident-2026q2-che-01, incident-2026q4-che-02, design-retry-001, design-capacity-checkout
      q-rel-3  (neither)
        toy   design-retry-001, design-schema-identity, incident-2026q1-ide-08, incident-2026q3-ide-09, incident-2026q4-ide-10
        real  design-retry-001, incident-2026q1-ide-08, incident-2026q3-ide-09, incident-2026q4-ide-10, runbook-err-101
      q-agg-2  (neither)
        toy   incident-2026q3-inv-27, incident-2026q3-not-20, incident-2026q3-sea-16, incident-2026q3-bil-23, incident-2026q3-cat-04
        real  incident-2026q3-bil-23, incident-2026q3-cat-04, incident-2026q3-ide-09, incident-2026q3-not-20, incident-2026q3-sea-16
      q-multi-1  (neither)
        toy   runbook-cfg-103, incident-2026q3-inv-27, incident-2026q3-not-20, incident-2026q3-sea-16, incident-2026q3-bil-23
        real  runbook-cfg-103, incident-2026q3-cat-04, incident-2026q3-ide-09, incident-2026q3-not-20, incident-2026q3-sea-16
      ... and 4 more, not truncated silently

==============================================================================
2. What belonged to the instrument
==============================================================================
  ACROSS 3 LEG(S):
    queries where either store answered      45
    of those, with a document ground truth   12
    net queries whose answer changed hands   +0
    queries where the two stores disagreed   12/45

  READ THOSE TWO LINES TOGETHER. The net is zero and the stores
  still disagree on the documents they return. 'No difference in
  the score' is not 'no difference', and a project that reported
  only the first number would have published the wrong sentence.

==============================================================================
What this comparison does NOT claim
==============================================================================
  - One corpus of 85 synthetic documents and 19 queries. A real
    corpus is a different measurement, and a larger one may be a
    different result.
  - The vector leg is compared with the SAME hash embedder on both
    sides on purpose, so the delta is storage and ranking alone. It
    is not a claim about a real embedding model; that is what
    scripts/real_run.py measures.
  - No approximate index is built. pgvector runs an exact scan, so
    ANN recall loss is a separate number this does not report.
  - No latency figures. Two stores on one laptop, one of them
    in-process, would measure the laptop.
==============================================================================
```

Not one query changed hands, and the stores still disagree on a quarter of
what they return. Every leg answers exactly the same labeled queries at
exactly the same recall as the hand-rolled stand-in it replaced, net `+0`, while
returning a different top-5 on 12 of the 45 queries where either side answered.
Elasticsearch and the hand-rolled BM25 agree on the ordering of only 5 of 19.

Either number alone produces a false sentence. "We moved to real stores and the
numbers did not move" suggests the implementations are interchangeable, and they
are not. "The stores disagree on most queries" suggests the stand-ins were
costing something, and on this corpus they were not. The claim that survives
both is narrower: at this corpus size and this k, the stand-ins cost this
project no measured retrieval quality, and they are not the same function.

That result was predicted before the adapter existed. DuckDB and the
in-memory Python loop return the same rows with the same computed values on all
7 queries where that leg produces anything. The relational leg was never where
the fudge was. The fudge is the phrase matching that picks WHICH aggregate to
compute, and that layer was left unchanged so a store swap could not be
confounded with a parser swap.

Do not skip the `n/a`. That leg answers with computed values matching no
document, so document recall for it is a category error rather than a zero: the
same rule `metrics.py` states. A comparison that ignores that rule scores the
leg on the eleven queries it is not for, counts eleven empty-vs-empty results as
eleven agreements, and prints "the stand-in was not costing anything" having
measured nothing. Agreement is counted only where at least one store answered.

A void fulltext leg is not something this summary would tell you. The analyzer
that claims to mirror `router/embeddings.py` was tokenizing `Checkout Service`
as `["heckout", "ervice"]`, because this repository lowercases before matching
`[a-z0-9_]+` and Elasticsearch tokenizes before lowercasing, making every
capital a delimiter. That run reported the same net `+0`, the same `5 -> 5 /5`,
and the same recall `1.000 -> 1.000` as the corrected one above; only the
agreement columns moved, `8/19` to `10/19` and `4/19` to `5/19`. A defect that
scrambled the entire index left the headline untouched. That is the measured
argument for printing the agreement columns beside it.

## What the captures show

The query mix decides the verdict, and that is the transferable result.
Vector-only scores `0.263` on the balanced evaluation set and `0.634` weighted
to a production mix: a 2.4x swing with no change to the router, the corpus or
the code. A balanced set is necessary to say anything per backend and it
overstates the case for federation by exactly that factor. Both numbers are
reported, and the mix weights are printed as the assumption they are.

Correctness without cost is not a result. The fan-out router scores a perfect
`1.000` because it cannot be wrong, at 4.00 backends per query. The heuristic
router reaches `0.947` at 1.53. Fan-out is in the table to make the first
number legible.

A single store fails three competences out of four. Vector-only misroutes every
exact-term, relationship and aggregate query, and gets 2 of 5 traps. That is the
case for federation, stated at its strongest, and section 3 is why it should not
be read at face value.

The two guards hold. An aggregate tell with nothing countable is suppressed
("how many retries does the backoff design recommend"), and a relationship tell
with no linked entity is suppressed ("which service owns
gateway.envelope.strict"). Both are rules a reader can disagree with rather than
a keyword list, and both are measured per query.

The offline instrument reports its own limit. On 4 of 11 document-bearing
queries the designed backend does not win, because a bag-of-tokens embedder
loses a genuine paraphrase to one rare shared term. Measured overlap on those
queries is 0.02 to 0.10. That gap is the reason a real-model capture is worth
paying for.

A real embedder narrows the designed/measured gap and does not close it.
Swapping the 256-dimension hash for `text-embedding-3-small` at 1536 moves the
designed backend from winning 7 of 11 document-bearing queries to 9, and the
count that nobody retrieves from 2 to 1. Two specific queries came back:
`q-sem-1`, where the vector leg finally joins the fulltext leg it was losing to,
and `q-trap-5`, which no backend could retrieve at all offline. The residual is
what to read. `q-sem-2` still goes to fulltext with a real semantic model in
place: a genuine paraphrase, measured overlap 0.02 to 0.10, and BM25's idf on
one rare shared term still beats cosine. The convenient reading was that the
mock was the whole problem. It was most of it.

The vectors do not reproduce and the numbers do, and both halves are the
finding. The corpus was embedded twice through two cold caches. 76 of 104 texts
returned bit-identical vectors; the other 28 came back different for
byte-identical input, with a largest component difference of `3.357e-03`. That
competence table is unchanged. One sample said 74 of 104 at
`1.602e-03`; same story, different digits, which is itself the point. So a
pipeline that hashes embeddings to detect drift, caches them by content hash
across processes, or diffs two index builds sees churn that means nothing, while
the retrieval decisions those vectors drive are stable at this magnitude.
Neither half of that sentence was knowable without paying for the second pass,
and the second pass cost one hundredth of a cent.

A model router is not one thing, and the difference is larger than the
result. Given the same prompt, the same 19 queries and no corpus knowledge,
`claude-opus-5` scored `1.000` in all ten runs across both samples, at 1.05 to
1.21 backends per query, with zero misroutes in 190 routing decisions.
`gpt-5.6-terra` never once exceeded the heuristic on correctness in either
sample, it tied at `0.947` in two runs of the first and reached `0.895` at best
in the second, while beating it on cost every time. One model beat two
hand-written `if` statements on both axes in every run; the other did not beat
them on correctness in any. Anyone reporting "an LLM router scores X" without
naming the model has reported the model, not the architecture.

Five runs separate two defects that one run cannot. In the second sample
`gpt-5.6-terra` misroutes `q-trap-1` in 5 of 5 runs, `q-trap-3` in 4 of 5, and
`q-multi-2` in 1 of 5. The first is a settled wrong opinion; the last is a
router that is unreliable about a question it usually answers correctly. They
need different fixes, and at n=1 they look identical.

The guard's own trap is where the model reasons its way to the wrong answer. The
linked-entity guard exists because "which service owns
gateway.envelope.strict" says "owns" and its subject is a config key with no
node in the graph. `gpt-5.6-terra` routed it to the graph in 4 of 5 runs,
reasoning: *"This is an ownership relationship lookup for a specific
c[onfiguration key]"*: the exact inference the guard suppresses, arrived at
independently, stated more articulately than the guard states it, and wrong for
the same reason. A hand-written rule that survives contact with a frontier
model, twice, is worth more than one that was never tested against anything.

Routing stability is the number this section exists for, and it is itself not
stable. The heuristic, fan-out and vector-only routers answered 19 of 19
queries identically in all five runs of both samples; they are the control, and
a spread on any of their rows would mean the instrument moved rather than the
router. Against that floor, `llm-openai` held 18 of 19 in the first sample and
17 of 19 in the second; `llm-anthropic` held 17 of 19 and then 19 of 19. A
router that answers the same question two ways costs two different amounts and
returns two different merged lists, and neither run is wrong, but WHICH question
it will be unreliable about is not something five runs can tell you. See [the
same capture, taken twice](#the-same-capture-taken-twice).

What n=5 does not do. It detects instability; it does not rank two routers whose
runs fall on both sides of each other, and it does not identify which query a
model will be unreliable about. Correctness over 19 queries moves in steps of
`0.053`, five draws is not a sample to compute a p-value from, and none is
computed. Where an axis straddles, the capture prints that instead of ordering
the two; where a router never got ahead, it prints that instead; they are
different findings and only the first is a limit of the sample.

None of this is in the gate. No test asserts on any number above, the exit code
does not depend on a provider, and a re-run will move them. A capture that fed
back into a pass/fail threshold would make the build depend on a vendor's
weekend deployment.

The corpus size is a precondition, not trivia. Top-5 is 5.9% of 85 documents. At
12 documents top-5 would be 42% of everything, every backend would "find" every
answer, and the labels would agree with measurement on 1 query out of 11. The
gate fails if selectivity regresses.
