# federated-retrieval-router

[![CI](https://github.com/jkelly-dev1/federated-retrieval-router/actions/workflows/ci.yml/badge.svg)](https://github.com/jkelly-dev1/federated-retrieval-router/actions/workflows/ci.yml)

One query, four stores, one ranked answer -- built as a **personal learning
project** to measure the part that usually goes unmeasured: not whether
federated retrieval works, but whether the **routing decision** was right, and
what it costs to be right.

A router decides which backends a question needs -- vector for paraphrase,
graph for relationships, full-text for exact terms, relational for aggregates
-- runs them concurrently, merges with reciprocal rank fusion, and returns one
ranked list in which every item carries which leg produced it.

**The headline is not that federation wins. It is that how much it wins by is
mostly an assumption about your traffic.**

```
router         balanced  prod-mix  fan-out   traps
heuristic         0.947     0.931     1.53    5/5
fan-out           1.000     1.000     4.00    5/5
vector-only       0.263     0.634     1.00    2/5
```

Same router, same corpus, same code. The only thing that changes between the
two correctness columns is an assumption about what people ask. Scored on a
balanced evaluation set -- which you need in order to say anything per backend
-- a single vector store looks catastrophic at `0.263`. Weighted to a realistic
production mix, it looks mostly adequate at `0.634`. **A 2.4x swing from the
query mix alone.** Publishing only the balanced number would sell federation
about twice as hard as the evidence supports, so both are reported and the mix
weights are printed as the assumption they are.

The rule this repo follows: no claim without a test. The table below maps each
claim to the test that enforces it.

## The second result: correctness is not the metric

A fan-out router that queries every store on every question **cannot be
wrong**. It scores a perfect `1.000` by construction, at 4.00 backends per
query. That is why it is in the table rather than dismissed as trivial: any
routing correctness number reported without its cost beside it is not a result.

The heuristic router gets to `0.947` at **1.53 backends per query** -- 62% less
work than fan-out for 5% less correctness. Where it still misroutes is printed
in the demo, with the feature that fired.

## The third result: which model, not whether a model

One paid capture, `scripts/real_run.py`, asks whether a model routes better
than two hand-written `if` statements -- **five times**, because a repository
whose whole argument is that one number is not a result cannot make its own
headline claim from a single sample. Then the whole capture was taken **a
second time**, which turned out to matter more than the first repetition did.
Same prompt, same 19 queries, no corpus knowledge given to either model. Every
cell is min/median/max across the five runs of the second sample:

```
router                  balanced          prod-mix         fan-out  traps
heuristic      0.947/0.947/0.947 0.931/0.931/0.931  1.53/1.53/1.53    5/5
llm-openai     0.895/0.895/0.895 0.850/0.882/0.882  1.05/1.11/1.11  3-4/5
llm-anthropic  1.000/1.000/1.000 1.000/1.000/1.000  1.21/1.21/1.21    5/5
```

`claude-opus-5` beat the heuristic on **both** axes in **all ten runs across
both samples** -- 1.000 every time at no more than 1.21 backends per query
against 0.947 at 1.53, with zero misroutes in 190 routing decisions. The claim
this section used to make from one sample was given ten chances to fail and did
not take any of them.

`gpt-5.6-terra` **never exceeded** the heuristic on correctness in either
sample: it tied at 0.947 in two runs of the first and never got above 0.895 in
the second. It beat the heuristic on cost every time. The spread between two
frontier models is wider than the gap either one has with the hand-written
baseline, so **"an LLM router scores X" reports the model, not the
architecture.**

**What five runs buy that one cannot: routing stability.** How many of the 19
queries got the *same set of backends* in every run -- an operational property
almost nobody reports, because a router that answers a question two different
ways costs two different amounts and returns two different merged lists, and
neither run is wrong:

```
same set every run       first sample  second sample
heuristic                       19/19          19/19
llm-openai                      18/19          17/19
llm-anthropic                   17/19          19/19
```

**And what five runs do not buy, which the second sample is how I found out.**
After the first capture it was true, and tempting, to write that both models
were least stable on `q-trap-3` -- the config-key query the linked-entity guard
exists for. Five more runs falsified it: `claude-opus-5` was stable on all 19,
and `gpt-5.6-terra`'s flip moved to `q-multi-2`. **Five runs are enough to show
that a router is not deterministic. They are not enough to say which question
it will be unreliable about.** The claims that survived both samples are the
ones about the router; the claim that died was the one about a specific query.

What did reproduce about that trap is the reasoning. The guard exists because
*"which service owns gateway.envelope.strict"* says "owns" and its subject is a
config key with no node in the graph. `gpt-5.6-terra` routed it to the graph in
3 of 5 runs in the first sample and 4 of 5 in the second, reasoning *"This is
an ownership lookup for a specific configuration key."* -- the exact inference
the guard suppresses, stated more articulately than the guard states it, and
wrong for the same reason.

Reading the misroute counts together is the point. `gpt-5.6-terra` misroutes
q-trap-1 in **5 of 5** runs and q-multi-2 in **1 of 5** -- a settled wrong
opinion and an unreliable answer are different defects needing different fixes,
and n=1 cannot tell them apart.

**What n=5 does not buy, stated here rather than discovered later:** it detects
instability, it does not rank two routers whose runs land on both sides of each
other, and it does not identify which query will move. Correctness over 19
queries moves in steps of 1/19 = 0.053, five draws is not a sample to compute a
p-value from, and none is computed. The capture distinguishes a router that
STRADDLES another (unorderable at this sample size) from one that NEVER
exceeded it (a result, not a sample-size limit).

Full capture in [SAMPLE_RUN.md](SAMPLE_RUN.md#the-paid-capture), and the
two-sample comparison in [the same capture, taken
twice](SAMPLE_RUN.md#the-same-capture-taken-twice). **No test asserts on any of
it and the gate does not read it**, because a build that depends on a vendor's
weekend deployment is not a gate. The summarizer behind those tables is
`router/stability.py`, and it *is* tested -- offline, with scripted routers,
including the case that matters most: a deterministic router must report zero
variance, or the instrument is measuring its own noise.

## The fourth result: the vectors do not reproduce, and the numbers do

The same capture embeds the 85-document corpus **twice**, through two cold
caches, because a capture that cannot say whether its own numbers reproduce is
asserting reproducibility rather than measuring it:

```
                              first sample  second sample
texts embedded in both passes          104            104
bit-identical vectors               74/104         76/104
largest component difference     1.602e-03      3.357e-03
the table in section 1 is        unchanged      unchanged
```

**Around 28% of the vectors came back different for byte-identical input**, in
both samples, and every number derived from them is unchanged in both. Both
halves matter. A pipeline that hashes embeddings to detect drift, caches them
by content hash across processes, or diffs two index builds will see churn that
means nothing -- while the retrieval decisions those vectors drive are stable at
this magnitude. The finding is the gap between the two, and it cost one extra
pass over the corpus to measure instead of assume.

## What the four backends are actually for

| backend | competence | why the others lose |
|---|---|---|
| `vector` | paraphrase, concepts | an opaque identifier embeds to noise |
| `fulltext` | error codes, config keys | BM25 has no opinion about meaning |
| `graph` | ownership, dependency, multi-hop | the answer is in **no document** |
| `relational` | counts, means, time windows | the answer is a **number that appears in no document** |

**The relational leg is the one evaluations fudge.** "How many incidents did
payments have in 2026" cannot be retrieved -- only computed. It returns `agg:`
rows corresponding to no document, and an evaluation scoring it by document
overlap would report zero for the only backend that answered. That is why
`Competence.AGGREGATE` is a separate axis and why aggregate queries carry no
`relevant_docs`.

## The measurement, and the trap in it

`required` is a **designed** ground truth about question *shape*, not a claim
about which backend happens to win. Those differ here, and the gap is reported
rather than relabeled away:

```
COMPETENCE VALIDATION (does the designated backend actually win, at k=3?)
  queries with a document answer         11
  designed backend actually wins          7
  no backend retrieves the answer         2

    q-sem-1   designed=['vector']           measured=['fulltext']
    q-sem-2   designed=['vector']           measured=['fulltext']
    q-multi-2 designed=['graph','vector']   measured=NOBODY
    q-trap-5  designed=['vector']           measured=NOBODY
```

The offline embedder is a bag-of-tokens hash. On a paraphrase sharing even one
**rare** term with its target -- `dependency`, `backoff`, `twice` -- BM25's idf
carries more signal than the whole vector carries cosine, so the fulltext leg
wins queries the vector leg is supposed to own. Measured jaccard overlap on
those queries is **0.02 to 0.10**, so they are genuine paraphrases and the win
is real.

Scoring routing against *measured* winners would define the router's job as
"predict what the mock does", and the project would score well by learning an
artifact.

**Closing this gap is exactly what a real embedding model is for, and it only
half closes it.** With `text-embedding-3-small` at 1536 dimensions the designed
backend wins 9 of 11 rather than 7, and the count nobody retrieves falls from 2
to 1. But `q-sem-2` still goes to fulltext with a real semantic model in place:
a genuine paraphrase, overlap 0.02, and BM25's idf on one rare shared term
still carries more signal than cosine. The convenient reading was that the mock
was the whole problem. It was most of it.

## Two guards, and why they are the engineering

Both were written against a named trap, and both are rules you can disagree
with rather than a keyword list:

**An aggregate tell is not an aggregate.** *"How many retries does the backoff
design recommend"* opens with the strongest counting phrase in the language and
needs no counting -- the answer is a sentence in a design document. So an
aggregate tell only routes to the relational leg when it co-occurs with
something that store can actually count.

**A relationship tell is not a relationship.** *"Which service owns
gateway.envelope.strict"* says "owns", and the subject is a config key with no
node in the graph. So a relationship tell only routes to the graph when the
query links to a real entity -- a question the graph itself answers, not a
keyword list.

Both hold across all five traps. `vector-only` gets 2 of 5.

## Fusion, and the constant that decides it quietly

Rank fusion, not score fusion: a BM25 score of 14.2 and a cosine of 0.83 are
not on the same scale and BM25's range moves as the corpus grows. RRF throws
away magnitude in exchange, which is a real cost and is stated in `fusion.py`.

`DEFAULT_WINDOW` is the constant worth knowing about. Fusion only sees as deep
into each list as the window allows, so **a document one leg ranks 12th cannot
be fused at window 10 no matter how strongly another leg agrees** -- and the
merged list looks perfectly reasonable while it happens. It is measured by
`window_sweep` and asserted in tests rather than chosen.

## Claims backed by tests

| claim | test |
|---|---|
| The corpus is selective enough for top-k to discriminate | `test_the_corpus_is_large_enough_for_top_k_to_discriminate` |
| The corpus has distractors, not just anchors | `test_the_corpus_has_distractors_not_just_anchors` |
| Every required backend actually returns something | `test_every_required_backend_actually_returns_something` |
| Aggregate queries carry no relevant documents | `test_aggregate_queries_have_no_relevant_documents` |
| Every backend is required by several queries | `test_every_backend_is_required_by_several_queries` |
| The semantic queries are genuine paraphrases | `test_the_semantic_queries_are_genuine_paraphrases` |
| Every service and team has a graph node | `test_every_service_and_team_has_a_graph_node` |
| Every incident is reachable from the graph | `test_every_incident_is_reachable_from_the_graph` |
| The evaluation set is balanced and the mix is not | `test_the_evaluation_set_is_balanced_and_the_mix_is_not` |
| An identifier does not absorb trailing punctuation | `test_an_identifier_does_not_absorb_trailing_punctuation` |
| Fulltext beats vector on an opaque identifier | `test_fulltext_beats_vector_on_an_opaque_identifier` |
| Only the graph answers a pure traversal | `test_only_the_graph_answers_a_pure_traversal` |
| Only the relational leg computes an aggregate | `test_only_the_relational_leg_computes_an_aggregate` |
| The graph returns nothing when nothing links | `test_the_graph_returns_nothing_when_nothing_links` |
| The graph traverses backwards from a sink node | `test_the_graph_traverses_backwards_from_a_sink_node` |
| DIMENSIONS is wide enough to avoid manufactured similarity | `test_the_dimension_is_wide_enough_to_avoid_manufactured_similarity` |
| An aggregate tell with nothing countable is suppressed | `test_an_aggregate_tell_without_anything_countable_is_suppressed` |
| A relationship tell with no linked entity is suppressed | `test_a_relationship_tell_with_no_linked_entity_is_suppressed` |
| Every routing decision carries a rationale | `test_every_decision_carries_a_rationale` |
| Choosing a superset is correct but not free | `test_choosing_a_superset_is_correct_but_not_free` |
| A single store fails the other competences | `test_a_single_store_fails_the_other_competences` |
| Scoring a misaligned query list raises | `test_scoring_a_misaligned_query_list_raises` |
| The query mix changes the verdict | `test_the_query_mix_changes_the_verdict` |
| Trap accuracy is reported separately | `test_trap_accuracy_is_reported_separately` |
| Competence validation excludes aggregate queries | `test_competence_validation_excludes_queries_with_no_document_answer` |
| Fusion uses rank and ignores the native score | `test_fusion_uses_rank_and_ignores_the_native_score` |
| Agreement between backends outranks a single first place | `test_agreement_between_backends_outranks_a_single_first_place` |
| Provenance survives fusion | `test_provenance_survives_fusion` |
| A narrow window silently discards agreement | `test_a_narrow_window_silently_discards_agreement` |
| The documented demo command runs | `test_the_documented_demo_command_runs` |
| The demo runs from any working directory | `test_the_demo_runs_from_any_working_directory` |
| The demo is byte-identical across runs | `test_the_demo_is_byte_identical_across_runs` |
| The demo output fits eighty columns | `test_the_demo_output_fits_eighty_columns` |
| A missing optional SDK is a sentence, not a traceback | `test_a_missing_sdk_is_a_sentence_not_a_traceback` |
| A key without the SDK gets an actionable message | `test_a_key_without_the_sdk_gets_the_actionable_message` |
| The offline suite needs no key | `test_the_offline_suite_needs_no_key` |
| An explicit variable beats the env file | `test_an_explicit_variable_beats_the_env_file` |
| The capture refuses rather than faking one | `test_the_capture_refuses_rather_than_faking_one` |
| An unparseable model reply is an empty choice, not a fan-out | `test_an_unparseable_reply_is_an_empty_choice_not_a_fan_out` |
| An invented backend name is a parse failure | `test_an_invented_backend_name_is_a_parse_failure` |
| The routing prompt does not hand the model the heuristic | `test_the_prompt_does_not_hand_the_model_the_heuristic` |
| A dimension mismatch is caught at the seam | `test_a_dimension_mismatch_is_caught_at_the_seam` |
| The embedding memo keeps the capture comparable | `test_the_memo_is_what_keeps_the_capture_comparable` |
| A deterministic router reports zero variance | `test_a_deterministic_router_reports_zero_variance` |
| Every run is scored before anything is aggregated | `test_every_run_is_scored_before_anything_is_aggregated` |
| A router that flips is caught per query | `test_a_router_that_flips_is_caught_per_query` |
| A misroute carries how many runs it happened in | `test_a_misroute_carries_how_many_runs_it_happened_in` |
| An unparseable reply is a distinct answer, not a gap | `test_an_unparseable_reply_is_a_distinct_answer_not_a_gap` |
| The published claim survives only if every run beats the baseline | `test_the_published_claim_survives_only_when_every_run_beats_the_baseline` |
| Never exceeding is a result, not a sample-size limit | `test_never_exceeding_the_baseline_is_a_result_not_a_sample_size_limit` |
| Runs on both sides of the baseline is the unorderable case | `test_a_challenger_on_both_sides_of_the_baseline_is_the_unorderable_case` |
| Never losing is not the same finding as straddling | `test_never_losing_is_not_the_same_finding_as_straddling` |
| The run tally is withheld when the baseline itself moved | `test_the_run_tally_is_withheld_when_the_baseline_itself_moved` |
| A capture can be re-rendered without paying again | `test_a_capture_can_be_re_rendered_without_paying_for_it_again` |
| Re-rendering against a different query set is refused | `test_re_rendering_against_a_different_query_set_is_refused` |
| A missing store driver is a sentence, not a traceback | `test_a_missing_driver_is_a_sentence_not_a_traceback` |
| pgvector similarity is one minus distance | `test_pgvector_similarity_is_one_minus_distance_not_the_distance` |
| The Elasticsearch analyzer reaches the mapped fields | `test_the_matching_analyzer_is_actually_attached_to_the_fields` |
| The mirrored analyzer matches this repo's own tokenizer | `test_the_matching_analyzer_mirrors_this_repositorys_own_tokenizer` |
| ...and the mapping alone does not prove it | `test_the_matching_analyzer_produces_the_same_tokens_as_this_repository` |
| A lowercase-only pattern would eat every capital | `test_the_pattern_tokenizer_is_case_insensitive` |
| An unknown analyzer is refused rather than defaulted | `test_an_unknown_analyzer_is_refused_rather_than_defaulted` |
| DuckDB scopes with bound parameters, not interpolation | `test_duckdb_scopes_with_bound_parameters_not_string_interpolation` |
| A zero net delta with total disagreement is not "no difference" | `test_a_zero_net_delta_with_total_disagreement_is_not_no_difference` |
| Comparing two different legs is refused | `test_comparing_two_different_legs_is_refused` |
| The store-comparison detail does not truncate silently | `test_the_detail_block_does_not_truncate_silently` |
| Two stores that both return nothing have not agreed | `test_two_stores_that_both_return_nothing_have_not_agreed` |
| A leg with no document ground truth reports n/a, not zero | `test_a_leg_with_no_document_ground_truth_reports_n_a_not_zero` |
| DuckDB counts what the Python loop counts | `test_duckdb_counts_what_the_python_loop_counts` |
| DuckDB and the loop agree on every aggregate query | `test_duckdb_and_the_python_loop_agree_on_every_aggregate_query` |
| Cheaper without being more correct does not survive | `test_a_cheaper_router_that_is_not_more_correct_does_not_survive` |
| A partial run is rejected rather than scored | `test_a_partial_run_is_rejected_rather_than_scored` |
| Every rendered stability table fits the capture width | `test_every_rendered_table_fits_the_capture_width` |
| A bad run count is rejected before anything is spent | `test_a_bad_run_count_is_rejected_before_anything_is_spent` |

## Quickstart

```
pip install -r requirements.txt
pytest -q                          # 120 offline, 11 skipped
python -m router.gate              # the CI gate
python scripts/run_demo.py         # the full demonstration
```

Everything above runs offline against a deterministic hash embedder and an
in-memory corpus. No key is needed and no network call is made. There are no
runtime dependencies at all: BM25, cosine similarity, the graph traversal and the rank
fusion are written out in this repository so the comparison between backends is
legible rather than delegated to four library calls.

The one paid path is opt-in and refuses to fake itself:

```
pip install openai anthropic                        # commented out of requirements.txt
ENV_FILE=~/.secrets/ai.env python scripts/real_run.py
STABILITY_RUNS=5 ENV_FILE=~/.secrets/ai.env python scripts/real_run.py
```

`STABILITY_RUNS` is how many independent routing passes each router gets; it
defaults to 5, and the capture in SAMPLE_RUN.md was taken at that default.

With no key configured it prints why it is not running and exits `2`, rather
than emitting something that looks like a capture. Its output is already
recorded in [SAMPLE_RUN.md](SAMPLE_RUN.md#the-paid-capture), so there is no
reason to spend anything to read this repository.

## How a query flows

```
question
    |
    v
route ....... which backends does this question SHAPE need?
    |          two guards: countable-noun, linked-entity
    v
fan out ..... only to the chosen legs, concurrently
    |
    v
fuse ........ reciprocal rank fusion, window-bounded
    |
    v
answer ...... one ranked list, every item carrying its provenance
```

## Design notes and honest limits

1. **The embedder is a hash model, not a semantic one.** Every vector number
   here is a **floor**, and the competence-validation section measures the
   shortfall rather than hiding it. The direction matters: an instrument that
   understated the *graph* leg would flatter this project's thesis and should
   be treated as suspect. This one understates *vector*, which works against
   the thesis.

2. **85 synthetic documents.** Enough that top-5 is 5.9% of the corpus and
   retrieval discriminates. It says nothing about behavior at corpus scale, and
   nothing here has been run against a real vector store, search cluster, graph
   database or warehouse.

3. **The production mix is an assumption, not a measurement.** It comes from no
   real query log. It is printed in the demo so a reader can substitute their
   own, and both scorings are always reported.

4. **`required` is designed, not measured.** See the section above. Offline
   the two disagree on 4 of 11 document-bearing queries; with a real embedding
   model, on 2 of 11. The gap is reported, not relabeled away.

5. **Entity linking is substring matching.** Routing is what is measured;
   linking is a separate problem and a real one. `linked_entities` is public so
   a caller can see what a traversal actually started from.

6. **No latency or cost figures.** Fan-out is reported as backends per query,
   which is the part that does not depend on hardware. The claim that an
   encrypted or federated index cannot use an ANN structure is reasoning, not
   a measurement, and is labeled as such.

7. **The heuristic router is not a model.** It is deterministic so the gate can
   depend on it. The model comparison lives in a paid capture rather than an
   exit code, and it was run once per model -- a single sample of something
   that is not deterministic, which is why the capture is reported as a
   capture and nothing in the repository asserts on it.

8. **19 labeled queries.** Small. Wide enough that every backend is required by
   at least four, narrow enough that per-query results are individually
   inspectable, and too small to tune anything against without fitting it.

## TODO / not yet wired (honest scope)

Everything below is a limit of what has been measured, not a feature backlog.
The three-leg store A/B **has** run -- that section is next, and these are the
questions it does not answer.

**Sample size.** Two paid captures, one embedding model, five runs per router in
each. Enough to detect instability and to falsify a claim about every run; not
enough to rank two routers whose runs land on both sides of each other, not
enough to say WHICH query a model will be unreliable about, and no p-value is
computed.

**One corpus, one k.** The A/B is 85 synthetic documents, 19 queries and k=5.
A different corpus is a different measurement and a larger one may be a
different result. Nothing here scales that claim.

**No ANN index.** pgvector runs an exact scan on purpose, so the comparison
isolates storage from approximation. Recall loss from an HNSW or IVFFlat index
is a separate number, and it is not measured rather than assumed small.

**No latency.** Two stores on one machine, one of them in-process, would measure
the machine.

**The `english` analyzer delta is built but not published.** Both Elasticsearch
indexes are created and a test asserts they rank differently, but the published
comparison uses the mirrored analyzer only. Reporting the `english` delta means
deciding what a number means when the tokenizer and the scorer both move, and
that is a second measurement rather than a spare column.

**Which aggregate gets chosen is still unmeasured.** DuckDB replaced the
executor of the relational leg, not the phrase matching in front of it that
decides which aggregate to compute. That layer was deliberately left alone so a
store swap could not be confounded with a parser swap, and it is where the leg
is most likely to be wrong. Related and also unmeasured: that phrase matching
fires on 7 queries while the router's guards route only 5 of them to this leg,
and nothing in the A/B looks at that gap.

**Neo4j is not adapted, and this is a decision rather than a pending item.** The
hand-rolled traversal is the simplest of the four legs, so a real graph store
would add the least, and it is the heaviest container to stand up. The other
three were adapted because each had a plausible way to be wrong that a real
store would expose; this one does not.

**Weighted RRF is exposed and unused.** Picking weights needs a judgment set
larger than this one, and tuning them against 19 queries would be fitting.

**Multi-hop answers are returned as nodes, not as explanations of the path.**

## Real stores: all three legs measured

Every offline number above comes from a hand-rolled BM25, a hash embedder and a
Python loop. Each is a reasonable stand-in, and "reasonable stand-in" is exactly
the sort of claim the rest of this project declines to accept without measuring.
So the same 19 queries, the same corpus and the same ground truth run through
real stores:

```
pip install duckdb                            # relational leg, no container
docker compose up -d                          # pgvector + elasticsearch
pip install -r requirements-integration.txt
pytest tests/test_integration.py -v -rs       # skips become passes
python scripts/compare_stores.py
```

```
leg               answers          recall@k   same set  same order
relational            n/a    n/a (computed)        7/7         7/7
vector          4 -> 4 /7    0.571 -> 0.571      16/19       15/19
fulltext        5 -> 5 /5    1.000 -> 1.000      10/19        5/19
```

**Not one query changed hands, and the stores still disagree on a quarter of
what they return.** Every leg answers exactly the same labeled queries at
exactly the same recall as its hand-rolled stand-in -- net `+0` -- while
returning a different top-5 on 12 of the 45 queries where either side answered.
Elasticsearch and the hand-rolled BM25 agree on the *ordering* of only 5 of 19.

Those two facts have to be read together, and reporting either one alone
produces a false sentence. "We moved to real stores and the numbers did not
move" is true and suggests the implementations are interchangeable, which they
demonstrably are not. "The stores disagree on most queries" is true and suggests
the stand-ins were costing something, which on this corpus they were not. The
honest version is narrower than both: **at this corpus size and this k, the
hand-rolled stand-ins were not costing this project any measured retrieval
quality, and they are not the same function.** `router/compare.py` is shaped so
that combination cannot be rounded into "no difference".

Read the `n/a` on the relational row rather than skipping it. That leg's answers
are computed values matching no document, so document recall for it is a
category error rather than a zero -- the same rule `metrics.py` applies, and one
this comparison got wrong on its first run (see below).

The relational result was the one **predicted in advance**: DuckDB and the
Python loop return the same rows with the same computed values on all 7 queries
where the leg produces anything, 0 disagreements. The relational leg was never
where the fudge was. The fudge is the phrase matching that decides **which**
aggregate to compute, that layer was deliberately left unchanged so a store swap
could not be confounded with a parser swap, and whether the right aggregate gets
chosen remains unmeasured.

The apparatus and the discipline around it:

| decision | why it is a measurement rather than a preference |
|---|---|
| `analyzer="matching"` vs `"english"` | swapping the store changes the scorer **and** the tokenizer at once; the mirrored analyzer reproduces this repo's own pattern and stop list so the two can be told apart |
| pgvector runs an **exact scan** | no ANN index, so the first A/B isolates storage from approximation; recall loss from an index is a separate number and is named as not-measured |
| same hash embedder on both sides | the delta is storage and ranking alone -- a real embedding model is what `scripts/real_run.py` measures, and mixing the two would make either number unattributable |
| DuckDB keeps the existing phrase matching | the relational leg was never where the fudge was; moving the parser too would confound a store swap with a parser swap |

`scripts/compare_stores.py` refuses to print a table when no store is
available, and when only some are up it names the ones it skipped in the output
every time. A partial comparison labeled as complete is the failure this whole
repository is about.

**A mirrored analyzer that is not mirroring voids the fulltext measurement,
and the summary line does not notice.** `router/embeddings.py`
lowercases and *then* matches `[a-z0-9_]+`; Elasticsearch runs the tokenizer
*before* the lowercase filter, so a lowercase-only character class treated every
capital as a **delimiter** -- `Checkout Service` was indexed as
`["heckout", "ervice"]` and `ERR_UPSTREAM_4423` as `["_", "_4423"]`. Two offline
tests asserted the analyzer was faithful and both passed: one compared the regex
*source string* to `_TOKEN.pattern` (identical source, different pipeline
position), the other checked that both legs return the same top hit for an
identifier (they did -- the query was mangled the same way as the documents).
The fix is `"flags": "CASE_INSENSITIVE"`, and the test that settles it runs real
corpus text through the live `_analyze` endpoint and requires the token list to
equal `content_tokens()` on all 189 strings.

**What is worth more than the fix: the headline was identical before and after.**
The broken run also reported net `+0`, also reported `5 -> 5 /5` answers, also
reported recall `1.000 -> 1.000`. Only the agreement columns moved (same set
`8/19` to `10/19`, same order `4/19` to `5/19`, total disagreement `14/45` to
`12/45`). A defect that scrambled tokenization across the entire index left the
top-line number untouched -- which is a direct measurement of how insensitive
that number is, and the reason the agreement columns exist next to it.

**The comparison harness failed that standard on its first run too.** It scored the
relational leg on the 11 document-bearing queries -- the ones it is not for --
where both implementations correctly return nothing, counted 11 empty-vs-empty
pairs as 11 agreements, and printed *"the stand-in was not costing anything"*
from a comparison that had measured zero queries. Agreement is now computed
only over queries where at least one store returned something, recall only over
queries this leg is required for, and a comparison with nothing in it prints
`NOTHING WAS MEASURED`. It was caught by running the thing, which is the
argument for running it.

## Scope

One of sixteen small projects on the theme of AI systems you can trust and prove:

- [prompt-injection-benchmark](https://github.com/jkelly-dev1/prompt-injection-benchmark)
- [ai-data-boundary-proxy](https://github.com/jkelly-dev1/ai-data-boundary-proxy)
- [llm-eval-gate](https://github.com/jkelly-dev1/llm-eval-gate)
- [least-privilege-agent](https://github.com/jkelly-dev1/least-privilege-agent)
- [citation-abstention-rag](https://github.com/jkelly-dev1/citation-abstention-rag)
- [agentic-review-gate](https://github.com/jkelly-dev1/agentic-review-gate)
- [typed-agent-service](https://github.com/jkelly-dev1/typed-agent-service)
- [temporal-multi-agent](https://github.com/jkelly-dev1/temporal-multi-agent)
- [hardened-mcp-server](https://github.com/jkelly-dev1/hardened-mcp-server)
- [vlm-extraction-integrity](https://github.com/jkelly-dev1/vlm-extraction-integrity)
- [llm-observability-stack](https://github.com/jkelly-dev1/llm-observability-stack)
- [ai-compliance-checker](https://github.com/jkelly-dev1/ai-compliance-checker)
- [airgapped-ai-bundle](https://github.com/jkelly-dev1/airgapped-ai-bundle)
- [agent-sandbox-escape](https://github.com/jkelly-dev1/agent-sandbox-escape)
- [parser-eval](https://github.com/jkelly-dev1/parser-eval)

The corpus is synthetic. Services, teams and incidents are fictional, every
host uses an `.example` TLD reserved by RFC 2606, and no real operational data
is in this repository.

## License

MIT
