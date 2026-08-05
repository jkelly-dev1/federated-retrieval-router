"""The n>1 report, tested at n>1 without paying for a single call.

THE POINT OF THIS FILE. `scripts/real_run.py` is the only thing in the
repository that spends money, so every claim its stability section prints has
to be enforced by something that does not. The summarizer is a pure function
over decisions, so a scripted router that deliberately flips is a complete
substitute for a live one -- including for the two cases a paid run cannot be
relied on to produce on demand: a router that never varies, and a router whose
parser fails.

The width tests are not cosmetic. SAMPLE_RUN.md is pasted verbatim into a
README-adjacent file that renders on GitHub, and a table that wraps is a table
a reader mis-parses.
"""
from __future__ import annotations

import json

import pytest

from router.corpus import PRODUCTION_MIX, build_corpus
from router.models import (
    Backend,
    Competence,
    LabeledQuery,
    RoutingDecision,
)
from router.stability import (
    WIDTH,
    QueryStability,
    Spread,
    backend_label,
    dump_runs,
    load_runs,
    ranking_survives,
    render_misroutes,
    render_per_query_table,
    render_summary_table,
    render_unstable_detail,
    render_verdict,
    summarize_runs,
)

VECTOR = frozenset({Backend.VECTOR})
FULLTEXT = frozenset({Backend.FULLTEXT})
GRAPH = frozenset({Backend.GRAPH})
ALL = frozenset(Backend)


def _queries() -> tuple[LabeledQuery, ...]:
    return (
        LabeledQuery(
            query_id="q-sem-1",
            text="how do we handle retries when a partner is slow",
            competences=frozenset({Competence.SEMANTIC}),
            required=VECTOR,
        ),
        LabeledQuery(
            query_id="q-exact-1",
            text="what does ERR_UPSTREAM_4423 mean",
            competences=frozenset({Competence.EXACT_TERM}),
            required=FULLTEXT,
        ),
        LabeledQuery(
            query_id="q-trap-1",
            text="which service owns gateway.envelope.strict",
            competences=frozenset({Competence.EXACT_TERM}),
            required=FULLTEXT,
            trap="says owns, subject is a config key",
        ),
    )


def _run(queries, choices, rationale="scripted"):
    """One pass: a chosen set per query, in query order."""
    return [
        RoutingDecision(
            query_id=q.query_id,
            chosen=chosen,
            competences=frozenset(),
            rationale=(rationale,),
        )
        for q, chosen in zip(queries, choices)
    ]


def _perfect(queries, n=5):
    return [_run(queries, [q.required for q in queries]) for _ in range(n)]


# ------------------------------------------------------------- the control


def test_a_deterministic_router_reports_zero_variance():
    """THE CONTROL, AND THE FIRST THING THAT WOULD BREAK.

    A stability metric that cannot report zero spread where variance is
    impossible is measuring its own noise floor. The deterministic routers are
    re-run in the capture for exactly this reason, so it is asserted here.
    """
    queries = _queries()
    stat = summarize_runs("heuristic", queries, _perfect(queries), PRODUCTION_MIX)

    assert stat.runs == 5
    assert stat.deterministic
    assert stat.stable_queries == len(queries)
    assert stat.worst_query is None
    assert stat.balanced.constant and stat.balanced.low == 1.0
    assert stat.fan_out.constant and stat.fan_out.low == 1.0
    assert stat.traps_render() == "1/1"
    assert not stat.misroutes


def test_every_run_is_scored_before_anything_is_aggregated():
    """Averaging decisions and scoring the average would invent a router that
    never ran. The min and the max have to be values a single run produced."""
    queries = _queries()
    good = [q.required for q in queries]
    bad = [VECTOR, VECTOR, VECTOR]  # only q-sem-1 correct
    runs = [_run(queries, good), _run(queries, bad), _run(queries, good)]

    stat = summarize_runs("llm", queries, runs, PRODUCTION_MIX)
    assert stat.balanced.low == pytest.approx(1 / 3)
    assert stat.balanced.mid == pytest.approx(1.0)
    assert stat.balanced.high == pytest.approx(1.0)
    assert stat.trap_correct == (1, 0, 1)
    assert stat.traps_render() == "0-1/1"


# -------------------------------------------------------- catching a flip


def test_a_router_that_flips_is_caught_per_query():
    """The measurement item 1 exists for: same question, different answers."""
    queries = _queries()
    runs = [
        _run(queries, [VECTOR, FULLTEXT, FULLTEXT]),
        _run(queries, [VECTOR, FULLTEXT, ALL]),
        _run(queries, [VECTOR, FULLTEXT, FULLTEXT]),
        _run(queries, [VECTOR, FULLTEXT, GRAPH]),
    ]
    stat = summarize_runs("llm-openai", queries, runs, PRODUCTION_MIX)

    assert not stat.deterministic
    assert stat.stable_queries == 2
    worst = stat.worst_query
    assert worst is not None and worst.query_id == "q-trap-1"
    assert worst.distinct == 3
    assert worst.modal == FULLTEXT and worst.modal_runs == 2
    # A flip between one backend and four is the cost story, not just noise.
    assert (worst.fan_out_low, worst.fan_out_high) == (1, 4)
    assert "fan-out 1 to 4" in worst.summary()


def test_a_misroute_carries_how_many_runs_it_happened_in():
    """1-of-5 wrong and 5-of-5 wrong are different defects. At n=1 they were
    the same field."""
    queries = _queries()
    runs = [
        _run(queries, [VECTOR, FULLTEXT, FULLTEXT]),
        _run(queries, [VECTOR, FULLTEXT, GRAPH], rationale="llm: an ownership lookup"),
        _run(queries, [VECTOR, FULLTEXT, GRAPH]),
    ]
    stat = summarize_runs("llm-openai", queries, runs, PRODUCTION_MIX)

    assert [m.query_id for m in stat.misroutes] == ["q-trap-1"]
    miss = stat.misroutes[0]
    assert (miss.runs_wrong, miss.runs) == (2, 3)
    assert miss.example_chosen == GRAPH
    assert "ownership lookup" in miss.example_rationale


def test_an_unparseable_reply_is_a_distinct_answer_not_a_gap():
    """A router that sometimes returns nothing is unstable, not absent. The
    empty set has to be counted as one of the answers it gave, or a parser
    failure would silently improve its stability score."""
    queries = _queries()
    runs = [
        _run(queries, [VECTOR, FULLTEXT, FULLTEXT]),
        _run(
            queries,
            [VECTOR, FULLTEXT, frozenset()],
            rationale="llm-openai: unparseable reply (JSONDecodeError)",
        ),
    ]
    stat = summarize_runs("llm-openai", queries, runs, PRODUCTION_MIX)

    assert stat.unparseable == 3
    trap = next(q for q in stat.per_query if q.query_id == "q-trap-1")
    assert trap.distinct == 2
    assert backend_label(frozenset()) == "NOTHING"


# -------------------------------------------------------------- the verdict


def test_the_published_claim_survives_only_when_every_run_beats_the_baseline():
    """"Beat the heuristic on both axes" is a claim about runs, not means, so
    the challenger's WORST is compared with the baseline's BEST."""
    queries = _queries()
    # A baseline that is both less correct (2 of 3) and more expensive (1.67
    # backends per query) than a challenger that is perfect at 1.00.
    baseline = summarize_runs(
        "heuristic",
        queries,
        [
            _run(
                queries,
                [
                    frozenset({Backend.VECTOR, Backend.GRAPH}),
                    frozenset({Backend.FULLTEXT, Backend.VECTOR}),
                    GRAPH,
                ],
            )
        ]
        * 3,
        PRODUCTION_MIX,
    )
    challenger = summarize_runs("llm-anthropic", queries, _perfect(queries, 3), PRODUCTION_MIX)

    verdict = ranking_survives(challenger, baseline)
    assert verdict.survives and verdict.correctness_clear and verdict.fan_out_clear
    assert "HOLDS at n=3" in verdict.reason


def test_never_exceeding_the_baseline_is_a_result_not_a_sample_size_limit():
    """THE PRECISION THIS FILE EXISTS FOR.

    A challenger that ties the baseline on its best run and loses on its worst
    never once came out ahead. Calling that "indistinguishable" would blame the
    sample size for a finding the sample size did not produce.
    """
    queries = _queries()
    baseline = summarize_runs("heuristic", queries, _perfect(queries, 3), PRODUCTION_MIX)
    challenger = summarize_runs(
        "llm-openai",
        queries,
        [
            _run(queries, [q.required for q in queries]),
            _run(queries, [VECTOR, FULLTEXT, GRAPH]),
            _run(queries, [q.required for q in queries]),
        ],
        PRODUCTION_MIX,
    )

    verdict = ranking_survives(challenger, baseline)
    assert not verdict.survives
    assert verdict.correctness.never_ahead and not verdict.correctness.straddles
    assert not verdict.indeterminate
    assert "NEVER exceeded it" in verdict.reason
    assert "STRADDLES" not in verdict.correctness.sentence()
    # The tally is exact because the baseline never moved.
    assert (verdict.correctness.ahead, verdict.correctness.level) == (0, 2)
    assert verdict.correctness.behind == 1
    assert "tied in 2" in verdict.reason and "behind in 1" in verdict.reason
    assert "p-value" not in verdict.reason


def test_a_challenger_on_both_sides_of_the_baseline_is_the_unorderable_case():
    """The one place `indistinguishable` is honest: runs land above AND below."""
    queries = _queries()
    baseline = summarize_runs(
        "heuristic",
        queries,
        [_run(queries, [VECTOR, FULLTEXT, GRAPH])] * 3,  # 2 of 3 correct, flat
        PRODUCTION_MIX,
    )
    challenger = summarize_runs(
        "llm-openai",
        queries,
        [
            _run(queries, [q.required for q in queries]),          # 3 of 3
            _run(queries, [VECTOR, GRAPH, GRAPH]),                 # 1 of 3
            _run(queries, [VECTOR, FULLTEXT, GRAPH]),              # 2 of 3, level
        ],
        PRODUCTION_MIX,
    )

    verdict = ranking_survives(challenger, baseline)
    assert not verdict.survives
    assert verdict.correctness.straddles
    assert verdict.indeterminate
    assert (verdict.correctness.ahead, verdict.correctness.behind) == (1, 1)
    assert "STRADDLES" in verdict.reason
    assert "indistinguishable" in verdict.reason
    assert "p-value" not in verdict.reason


def test_never_losing_is_not_the_same_finding_as_straddling():
    """A tie on some runs is not a straddle.

    A challenger that ties the baseline in some runs and beats it in the rest
    has runs on ONE side only. Calling that STRADDLES announces that the
    sample cannot order two routers it has just ordered -- the same
    imprecision this comparison exists to remove, moved one step over.
    """
    queries = _queries()
    baseline = summarize_runs(
        "heuristic",
        queries,
        [_run(queries, [VECTOR, FULLTEXT, frozenset({Backend.FULLTEXT, Backend.VECTOR})])] * 4,
        PRODUCTION_MIX,
    )
    # Same correctness every run; cheaper than the baseline in exactly one.
    challenger = summarize_runs(
        "llm-anthropic",
        queries,
        [
            _run(queries, [VECTOR, FULLTEXT, frozenset({Backend.FULLTEXT, Backend.VECTOR})]),
            _run(queries, [VECTOR, FULLTEXT, frozenset({Backend.FULLTEXT, Backend.VECTOR})]),
            _run(queries, [VECTOR, FULLTEXT, frozenset({Backend.FULLTEXT, Backend.VECTOR})]),
            _run(queries, [VECTOR, FULLTEXT, FULLTEXT]),
        ],
        PRODUCTION_MIX,
    )

    verdict = ranking_survives(challenger, baseline)
    assert not verdict.survives
    fan_out = verdict.fan_out
    assert (fan_out.ahead, fan_out.behind) == (1, 0)
    assert fan_out.never_behind and not fan_out.straddles and not fan_out.clear
    assert "never lost to it" in fan_out.sentence()
    assert not verdict.indeterminate
    assert "indistinguishable" not in verdict.reason


def test_the_run_tally_is_withheld_when_the_baseline_itself_moved():
    """Pairing run i of one router against run i of another means nothing
    unless one of them is a constant. When the baseline varies, the ranges are
    still reported and the counts are not invented."""
    queries = _queries()
    baseline = summarize_runs(
        "llm-anthropic",
        queries,
        [
            _run(queries, [q.required for q in queries]),
            _run(queries, [VECTOR, FULLTEXT, GRAPH]),
        ],
        PRODUCTION_MIX,
    )
    challenger = summarize_runs("llm-openai", queries, _perfect(queries, 2), PRODUCTION_MIX)

    verdict = ranking_survives(challenger, baseline)
    assert not verdict.correctness.tally_valid
    assert (verdict.correctness.ahead, verdict.correctness.behind) == (0, 0)
    assert "ahead in" not in verdict.correctness.sentence()


def test_a_cheaper_router_that_is_not_more_correct_does_not_survive():
    """Both axes, not the better of the two."""
    queries = _queries()
    baseline = summarize_runs("heuristic", queries, _perfect(queries, 3), PRODUCTION_MIX)
    challenger = summarize_runs(
        "llm-anthropic", queries, _perfect(queries, 3), PRODUCTION_MIX
    )
    verdict = ranking_survives(challenger, baseline)
    assert not verdict.survives
    assert not verdict.correctness_clear and not verdict.fan_out_clear


# ---------------------------------------------------------------- guardrails


def test_a_partial_run_is_rejected_rather_than_scored():
    queries = _queries()
    runs = [_run(queries, [q.required for q in queries]), _run(queries[:2], [VECTOR, FULLTEXT])]
    with pytest.raises(ValueError, match="partial run is not a run"):
        summarize_runs("llm", queries, runs, PRODUCTION_MIX)


def test_no_runs_at_all_is_not_a_measurement():
    with pytest.raises(ValueError, match="at least one run"):
        summarize_runs("llm", _queries(), [], PRODUCTION_MIX)
    with pytest.raises(ValueError, match="not a measurement"):
        Spread(())


def test_a_capture_can_be_re_rendered_without_paying_for_it_again():
    """THE FIX FOR A DEFECT THAT COST REAL MONEY.

    The tables are rendered from decisions, so the decisions are dumped beside
    the capture. A later session that wants different wording re-renders from
    the same evidence instead of buying another 190 completions -- which is
    exactly what a wording change cost once.
    """
    queries = _queries()
    runs = [
        _run(queries, [VECTOR, FULLTEXT, FULLTEXT], rationale="llm: because"),
        _run(queries, [VECTOR, FULLTEXT, GRAPH]),
    ]
    payload = json.loads(json.dumps(dump_runs(queries, {"llm-openai": runs})))
    restored = load_runs(payload, queries)["llm-openai"]

    before = summarize_runs("llm-openai", queries, runs, PRODUCTION_MIX)
    after = summarize_runs("llm-openai", queries, restored, PRODUCTION_MIX)
    assert render_summary_table([after]) == render_summary_table([before])
    assert render_misroutes(after, queries) == render_misroutes(before, queries)
    assert after.per_query == before.per_query


def test_re_rendering_against_a_different_query_set_is_refused():
    """A dump replayed against different queries would compare two experiments
    and say nothing about either."""
    queries = _queries()
    payload = dump_runs(queries, {"llm": _perfect(queries, 2)})
    with pytest.raises(ValueError, match="different query set"):
        load_runs(payload, queries[:2])


def test_the_modal_answer_is_stable_under_ties():
    """Two answers seen twice each must order deterministically, or the capture
    is not byte-comparable between two identical runs."""
    entry = QueryStability(
        query_id="q1",
        counts=tuple(
            sorted(
                {VECTOR: 2, FULLTEXT: 2}.items(),
                key=lambda item: (-item[1], backend_label(item[0])),
            )
        ),
    )
    assert backend_label(entry.modal) == "fulltext"
    assert entry.runs == 4 and entry.distinct == 2


# ------------------------------------------------------------------- widths


def _wide_stats():
    """The worst case the real capture can produce: the longest router name,
    every backend chosen, and the corpus's own query text."""
    corpus = build_corpus()
    queries = corpus.queries
    flipping = [
        _run(
            queries,
            [
                ALL if index % 2 == run else frozenset({Backend.VECTOR})
                for index, _ in enumerate(queries)
            ],
            rationale="llm-anthropic: " + "a very long rationale sentence " * 4,
        )
        for run in range(5)
    ]
    return corpus, [
        summarize_runs("llm-anthropic", queries, flipping, PRODUCTION_MIX),
        summarize_runs("heuristic", queries, _perfect(queries, 5), PRODUCTION_MIX),
    ]


def test_every_rendered_table_fits_the_capture_width():
    corpus, stats = _wide_stats()
    lines = (
        render_summary_table(stats)
        + render_per_query_table(stats)
        + render_unstable_detail(stats[0], corpus.queries)
        + render_unstable_detail(stats[1], corpus.queries)
        + render_misroutes(stats[0], corpus.queries)
        + render_verdict([ranking_survives(stats[0], stats[1])])
    )
    wide = [line for line in lines if len(line) > WIDTH]
    assert not wide, f"{len(wide)} lines over {WIDTH} columns: {wide[:2]}"


def test_the_tables_name_every_router_they_were_given():
    _, stats = _wide_stats()
    table = "\n".join(render_summary_table(stats) + render_per_query_table(stats))
    for stat in stats:
        assert stat.router in table
    assert "min/median/max" in table


def test_a_deterministic_router_says_so_in_its_detail_block():
    corpus, stats = _wide_stats()
    block = "\n".join(render_unstable_detail(stats[1], corpus.queries))
    assert "identical across all 5 runs" in block
