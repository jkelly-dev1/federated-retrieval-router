"""Turning N routing passes into a stability report, and nothing else.

Why this file exists. A paid capture that runs once and reports that a model
"beat the heuristic on both axes" is a point estimate from n=1, in a repository
whose entire argument is that a number reported without its cost, or without
its spread, is not a result. This module exists so the capture cannot say that,
and it deliberately measures something more useful than a mean.

Routing stability is an operational property and almost nobody reports it. A
router that answers a question with {vector} on one call and
{vector, graph, fulltext} on the next has a production problem no average
correctness can show: the same question costs three times as much, returns a
different merged list, and neither run is wrong. So the primary number here is
per-query distinct decision sets, and correctness arrives as a range rather
than as a point estimate.

WHAT n=5 buys and what it does not:

  It detects instability. One query answered two different ways in five draws
  falsifies "this router chooses X for this question", and one counterexample
  is enough for that.

  It does not rank two close models. Correctness over 19 queries moves in steps
  of 1/19 = 0.053, so five draws cannot separate routers whose ranges overlap.
  When they overlap this module says INDISTINGUISHABLE and refuses to order
  them. There are no p-values here and none are implied.

The deterministic routers are re-run too, and that is the control. A stability
metric that cannot report zero variance where variance is impossible is
measuring its own noise floor rather than the router's. The heuristic, fan-out
and vector-only rows exist in the stability table for exactly that reason, and
if one of them ever shows a spread the instrument is broken, not the router.

Rendering lives here, not in the script. The capture in SAMPLE_RUN.md is pasted
verbatim and has to fit 78 columns, and a table built inside a paid script is a
table that can only be checked by paying. Every table below is produced by a
pure function over synthetic data in the offline test suite.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Optional, Sequence

from router.metrics import score_routing, weighted_correctness
from router.models import Backend, LabeledQuery, RoutingDecision

WIDTH = 78
THIN = "-" * WIDTH


def backend_label(backends: Iterable[Backend]) -> str:
    """The one place a chosen set becomes text, so tables cannot disagree."""
    return ",".join(sorted(b.value for b in backends)) or "NOTHING"


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


@dataclass(frozen=True)
class Spread:
    """min / median / max of one metric across runs.

    Deliberately not a mean and a standard deviation. Correctness over 19
    queries is a step function with 20 possible values, and a standard
    deviation over five draws of it invites exactly the inference this file
    refuses to support. The min is the number that matters operationally: it is
    the run a user would have gotten on their worst day.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("a spread of no runs is not a measurement")

    @property
    def low(self) -> float:
        return min(self.values)

    @property
    def mid(self) -> float:
        return float(median(self.values))

    @property
    def high(self) -> float:
        return max(self.values)

    @property
    def constant(self) -> bool:
        return self.high == self.low

    def render(self, places: int = 3) -> str:
        return (
            f"{self.low:.{places}f}/{self.mid:.{places}f}/{self.high:.{places}f}"
        )


@dataclass(frozen=True)
class QueryStability:
    """How many different ways one router answered one question.

    `counts` is every distinct chosen set with the number of runs that produced
    it, most frequent first. An UNPARSEABLE reply lands here as the empty set
    and is counted as a distinct answer, which is the honest treatment: a
    router that sometimes returns nothing is unstable, not absent.
    """

    query_id: str
    counts: tuple[tuple[frozenset[Backend], int], ...]

    @property
    def runs(self) -> int:
        return sum(n for _, n in self.counts)

    @property
    def distinct(self) -> int:
        return len(self.counts)

    @property
    def stable(self) -> bool:
        return self.distinct == 1

    @property
    def modal(self) -> frozenset[Backend]:
        return self.counts[0][0]

    @property
    def modal_runs(self) -> int:
        return self.counts[0][1]

    @property
    def fan_out_low(self) -> int:
        return min(len(s) for s, _ in self.counts)

    @property
    def fan_out_high(self) -> int:
        return max(len(s) for s, _ in self.counts)

    def summary(self) -> str:
        """One phrase for the table: how unstable, and how expensively."""
        if self.stable:
            return "same set every run"
        cost = (
            f", fan-out {self.fan_out_low} to {self.fan_out_high}"
            if self.fan_out_low != self.fan_out_high
            else ""
        )
        return f"{self.distinct} sets{cost}"


@dataclass(frozen=True)
class Misroute:
    """A query the router got wrong, and in how many of the runs.

    `runs_wrong` is the field that did not exist at n=1. A misroute in 1 of 5
    runs and a misroute in 5 of 5 are different defects: the first is a router
    that is unreliable about a question it can answer, the second is a router
    that has a settled wrong opinion.
    """

    query_id: str
    required: frozenset[Backend]
    runs_wrong: int
    runs: int
    example_chosen: frozenset[Backend]
    example_rationale: str


@dataclass(frozen=True)
class RouterStability:
    """Everything N runs of one router say about it."""

    router: str
    runs: int
    balanced: Spread
    prod_mix: Spread
    fan_out: Spread
    trap_correct: tuple[int, ...]
    trap_total: int
    per_query: tuple[QueryStability, ...]
    misroutes: tuple[Misroute, ...] = ()
    unparseable: int = 0

    @property
    def stable_queries(self) -> int:
        return sum(1 for q in self.per_query if q.stable)

    @property
    def unstable(self) -> tuple[QueryStability, ...]:
        return tuple(q for q in self.per_query if not q.stable)

    @property
    def deterministic(self) -> bool:
        """Every query answered identically in every run."""
        return not self.unstable

    @property
    def worst_query(self) -> Optional[QueryStability]:
        """The least stable query, or None if the router never varied."""
        unstable = self.unstable
        if not unstable:
            return None
        return max(unstable, key=lambda q: (q.distinct, q.fan_out_high))

    def traps_render(self) -> str:
        low, high = min(self.trap_correct), max(self.trap_correct)
        head = f"{low}" if low == high else f"{low}-{high}"
        return f"{head}/{self.trap_total}"


def summarize_runs(
    router_name: str,
    queries: Sequence[LabeledQuery],
    runs: Sequence[Sequence[RoutingDecision]],
    mix: Mapping[object, float],
) -> RouterStability:
    """Score N independent passes of one router over the same query set.

    Every run is scored on its own before anything is aggregated. Averaging
    decisions first and scoring the average would produce a router that never
    existed and a correctness number no single run achieved.
    """
    if not runs:
        raise ValueError("a stability summary needs at least one run")
    for index, run in enumerate(runs):
        if len(run) != len(queries):
            raise ValueError(
                f"run {index} has {len(run)} decisions for {len(queries)} "
                f"queries; a partial run is not a run"
            )

    reports = [score_routing(router_name, queries, run) for run in runs]
    weighted = [weighted_correctness(queries, run, mix) for run in runs]

    per_query: list[QueryStability] = []
    misroutes: list[Misroute] = []
    for index, labeled in enumerate(queries):
        counter: Counter = Counter()
        wrong = 0
        example: Optional[tuple[frozenset[Backend], str]] = None
        for run in runs:
            decision = run[index]
            counter[decision.chosen] += 1
            if not decision.is_correct(labeled):
                wrong += 1
                if example is None:
                    rationale = decision.rationale[0] if decision.rationale else ""
                    example = (decision.chosen, rationale)
        per_query.append(
            QueryStability(
                query_id=labeled.query_id,
                counts=tuple(
                    sorted(
                        counter.items(),
                        key=lambda item: (-item[1], backend_label(item[0])),
                    )
                ),
            )
        )
        if wrong and example is not None:
            misroutes.append(
                Misroute(
                    query_id=labeled.query_id,
                    required=labeled.required,
                    runs_wrong=wrong,
                    runs=len(runs),
                    example_chosen=example[0],
                    example_rationale=example[1],
                )
            )

    return RouterStability(
        router=router_name,
        runs=len(runs),
        balanced=Spread(tuple(r.correctness for r in reports)),
        prod_mix=Spread(tuple(weighted)),
        fan_out=Spread(tuple(r.fan_out for r in reports)),
        trap_correct=tuple(r.trap_correct for r in reports),
        trap_total=reports[0].trap_total,
        per_query=tuple(per_query),
        misroutes=tuple(misroutes),
        unparseable=sum(
            1
            for run in runs
            for d in run
            if any("unparseable" in line for line in d.rationale)
        ),
    )


# --------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class AxisComparison:
    """One axis of one challenger against one baseline, in three states.

    The distinction this type exists for. "The ranges overlap" is true of two
    very different situations, and calling both of them INDISTINGUISHABLE is
    the imprecision this file is supposed to be immune to:

      NEVER AHEAD   the challenger's best run does not beat the baseline's
                    worst. It tied, or it lost, and it never once came out on
                    top. That is a RESULT, not a limit of the sample size.
      STRADDLES     the challenger is ahead in some runs and behind in others.
                    THAT is where n=5 has nothing to say, and the only place
                    the word "indistinguishable" is honest.
      CLEAR         the challenger's worst run beats the baseline's best.

    `ahead` / `level` / `behind` count runs rather than describing ranges, and
    they are only computed when the baseline never moved. Comparing run i of
    one router against run i of another means nothing unless one of them is a
    constant. `tally_valid` says which it was.
    """

    axis: str
    better_is_higher: bool
    challenger: Spread
    baseline: Spread
    ahead: int
    level: int
    behind: int
    tally_valid: bool
    places: int = 3

    @property
    def clear(self) -> bool:
        """The challenger's worst run beats the baseline's best.

        Deliberately range-based even when the tally is available: this is the
        survival condition the README's sentence asserts, and it must not
        quietly become "ahead on average".
        """
        if self.better_is_higher:
            return self.challenger.low > self.baseline.high
        return self.challenger.high < self.baseline.low

    @property
    def never_ahead(self) -> bool:
        """It never once came out on top; it tied, or it lost."""
        if self.tally_valid:
            return self.ahead == 0
        if self.better_is_higher:
            return self.challenger.high <= self.baseline.low
        return self.challenger.low >= self.baseline.high

    @property
    def never_behind(self) -> bool:
        """It never once came out underneath; it tied, or it won.

        DISTINCT FROM `clear`, and the distinction decides what the run
        measures. A challenger that ties the baseline in four runs and beats
        it in one has not beaten it in every run, so the survival claim
        fails, but calling that STRADDLES would say the sample cannot order
        them, when in fact it ordered them and found no run on the losing
        side.
        """
        if self.tally_valid:
            return self.behind == 0
        if self.better_is_higher:
            return self.challenger.low >= self.baseline.high
        return self.challenger.high <= self.baseline.low

    @property
    def straddles(self) -> bool:
        """Runs on both sides. The only state n=5 genuinely cannot order."""
        return not self.never_ahead and not self.never_behind

    def _tally(self) -> str:
        if not self.tally_valid:
            return ""
        parts = []
        total = self.ahead + self.level + self.behind
        if self.ahead:
            parts.append(f"ahead in {self.ahead} of {total}")
        if self.level:
            parts.append(f"tied in {self.level}")
        if self.behind:
            parts.append(f"behind in {self.behind}")
        return " -- " + ", ".join(parts) if parts else ""

    def sentence(self) -> str:
        ranges = (
            f"{self.challenger.render(self.places)} against "
            f"{self.baseline.render(self.places)}"
        )
        if self.clear:
            head = "beat it in every run"
        elif self.never_ahead:
            head = "NEVER exceeded it"
        elif self.never_behind:
            head = "never lost to it, and did not beat it every time"
        else:
            head = "STRADDLES it"
        return f"{self.axis}: {head}{self._tally()} ({ranges})."


@dataclass(frozen=True)
class SurvivalVerdict:
    """Does a published claim survive being re-run?

    The claim under test is "beat the heuristic on both axes at once", so both
    axes have to survive the challenger's WORST run against the baseline's
    BEST. Anything weaker is a claim about an average, and an average is what
    n=1 was already pretending to report.
    """

    challenger: str
    baseline: str
    survives: bool
    correctness: AxisComparison
    fan_out: AxisComparison
    reason: str

    @property
    def correctness_clear(self) -> bool:
        return self.correctness.clear

    @property
    def fan_out_clear(self) -> bool:
        return self.fan_out.clear

    @property
    def indeterminate(self) -> bool:
        """Is any axis genuinely unorderable at this sample size?"""
        return self.correctness.straddles or self.fan_out.straddles


def _compare_axis(
    axis: str,
    better_is_higher: bool,
    challenger: Spread,
    baseline: Spread,
    places: int = 3,
) -> AxisComparison:
    valid = baseline.constant
    reference = baseline.low
    ahead = level = behind = 0
    if valid:
        for value in challenger.values:
            if value == reference:
                level += 1
            elif (value > reference) == better_is_higher:
                ahead += 1
            else:
                behind += 1
    return AxisComparison(
        axis=axis,
        better_is_higher=better_is_higher,
        challenger=challenger,
        baseline=baseline,
        ahead=ahead,
        level=level,
        behind=behind,
        tally_valid=valid,
        places=places,
    )


def ranking_survives(
    challenger: RouterStability, baseline: RouterStability
) -> SurvivalVerdict:
    """Compare a challenger's worst run against a baseline's best, per axis."""
    correctness = _compare_axis(
        "correctness", True, challenger.balanced, baseline.balanced, 3
    )
    fan_out = _compare_axis("fan-out", False, challenger.fan_out, baseline.fan_out, 2)
    survives = correctness.clear and fan_out.clear
    n = challenger.runs

    if survives:
        reason = (
            f"HOLDS at n={n}. Its worst run scores "
            f"{challenger.balanced.low:.3f} against {baseline.balanced.high:.3f} "
            f"and its most expensive run fans out to "
            f"{challenger.fan_out.high:.2f} against {baseline.fan_out.low:.2f}, "
            f"so the claim is true of every run and not only of the mean."
        )
    else:
        reason = (
            f"DOES NOT HOLD at n={n}. "
            + " ".join(
                axis.sentence()
                for axis in (correctness, fan_out)
                if not axis.clear
            )
            + (
                " Where an axis STRADDLES, n=5 has nothing to say about the "
                "order and the two are reported as indistinguishable on it. "
                "NEVER exceeded is a different statement: it is a result, not "
                "a limit of the sample."
                if correctness.straddles or fan_out.straddles
                else " No axis straddles: this is a measured outcome rather "
                "than a sample-size limit."
            )
        )
    return SurvivalVerdict(
        challenger=challenger.router,
        baseline=baseline.router,
        survives=survives,
        correctness=correctness,
        fan_out=fan_out,
        reason=reason,
    )


# ---------------------------------------------------- replay without paying


def dump_runs(
    queries: Sequence[LabeledQuery],
    passes: Mapping[str, Sequence[Sequence[RoutingDecision]]],
) -> dict:
    """Every routing decision the capture was built from, as plain JSON.

    This exists because a wording change once cost a paid re-run. The tables
    above are rendered from decisions, so a reader who wants a different
    sentence should be able to re-render from the same evidence instead of
    buying 190 completions to improve an adjective. Written beside the capture
    and never committed. See .gitignore.
    """
    return {
        "queries": [q.query_id for q in queries],
        "routers": {
            name: [
                [sorted(b.value for b in d.chosen) for d in run] for run in runs
            ]
            for name, runs in passes.items()
        },
        "rationales": {
            name: [[list(d.rationale) for d in run] for run in runs]
            for name, runs in passes.items()
        },
    }


def load_runs(
    payload: Mapping, queries: Sequence[LabeledQuery]
) -> dict[str, list[list[RoutingDecision]]]:
    """Rebuild decisions from a dump, so a capture can be re-rendered offline."""
    ids = list(payload["queries"])
    if [q.query_id for q in queries] != ids:
        raise ValueError(
            "the dump was taken against a different query set; re-rendering it "
            "against this one would silently compare two experiments"
        )
    out: dict[str, list[list[RoutingDecision]]] = {}
    for name, runs in payload["routers"].items():
        rationales = payload.get("rationales", {}).get(name)
        out[name] = [
            [
                RoutingDecision(
                    query_id=ids[index],
                    chosen=frozenset(Backend(value) for value in chosen),
                    competences=frozenset(),
                    rationale=tuple(
                        rationales[run_index][index] if rationales else ()
                    ),
                )
                for index, chosen in enumerate(run)
            ]
            for run_index, run in enumerate(runs)
        ]
    return out


# ------------------------------------------------------------------ rendering


def _wrap(text: str, indent: str) -> list[str]:
    """Wrap to WIDTH without importing a formatter, so output is predictable."""
    lines: list[str] = []
    current = indent
    for word in text.split():
        if current == indent:
            current = f"{indent}{word}"
        elif len(current) + 1 + len(word) > WIDTH:
            lines.append(current)
            current = f"{indent}{word}"
        else:
            current = f"{current} {word}"
    if current.strip():
        lines.append(current)
    return lines


def render_summary_table(stats: Sequence[RouterStability]) -> list[str]:
    """Correctness, cost and traps as ranges instead of points."""
    runs = stats[0].runs if stats else 0
    lines = [
        f"  Every cell is min/median/max across the {runs} runs. The three",
        "  deterministic routers are re-run too: they are the control, and a",
        "  spread on any of their rows would mean the instrument moved.",
        "",
        "  "
        + "router".ljust(14)
        + "balanced".rjust(18)
        + "prod-mix".rjust(18)
        + "fan-out".rjust(16)
        + "traps".rjust(7),
    ]
    for stat in stats:
        lines.append(
            "  "
            + stat.router.ljust(14)
            + stat.balanced.render().rjust(18)
            + stat.prod_mix.render().rjust(18)
            + stat.fan_out.render(2).rjust(16)
            + stat.traps_render().rjust(7)
        )
    return lines


def render_per_query_table(stats: Sequence[RouterStability]) -> list[str]:
    """The number this section exists for: did the answer change at all."""
    total = len(stats[0].per_query) if stats else 0
    lines = [
        "  PER-QUERY STABILITY. How many of the "
        + f"{total} queries got the same set of",
        "  backends in every run. A router that flips between one backend and",
        "  three on the same question is an operational problem whatever its",
        "  mean correctness says.",
        "",
        "  "
        + "router".ljust(14)
        + "same set every run".rjust(20)
        + "  "
        + "least stable query",
    ]
    for stat in stats:
        worst = stat.worst_query
        detail = "--" if worst is None else f"{worst.query_id}: {worst.summary()}"
        lines.append(
            "  "
            + stat.router.ljust(14)
            + f"{stat.stable_queries}/{len(stat.per_query)}".rjust(20)
            + "  "
            + _clip(detail, WIDTH - 38)
        )
    return lines


def render_unstable_detail(
    stat: RouterStability, queries: Sequence[LabeledQuery]
) -> list[str]:
    """Every distinct answer, with its run count, for the queries that moved."""
    text_by_id = {q.query_id: q.text for q in queries}
    lines = [
        THIN,
        f"  {stat.router}: the queries it did not answer the same way twice",
        THIN,
    ]
    if stat.deterministic:
        lines.append(f"    every query identical across all {stat.runs} runs")
        return lines
    for entry in stat.unstable:
        lines.append(f"    {entry.query_id}  {entry.summary()}")
        lines.append(
            "      q: " + _clip(text_by_id.get(entry.query_id, ""), WIDTH - 10)
        )
        for chosen, count in entry.counts:
            lines.append(f"        {count}x  {backend_label(chosen)}")
    return lines


def render_misroutes(
    stat: RouterStability, queries: Sequence[LabeledQuery]
) -> list[str]:
    """Where it disagreed with the labels, and in how many of the runs."""
    text_by_id = {q.query_id: q.text for q in queries}
    lines = [
        THIN,
        f"  {stat.router}: where it disagreed with the labels",
        THIN,
    ]
    if stat.unparseable:
        lines.append(f"    unparseable replies: {stat.unparseable}")
    if not stat.misroutes:
        lines.append(f"    no misroutes in {stat.runs} runs")
        return lines
    for miss in stat.misroutes:
        lines.append(
            f"    {miss.query_id}: wrong in {miss.runs_wrong} of {miss.runs} "
            f"runs, needed {sorted(b.value for b in miss.required)}"
        )
        lines.append(
            "      q: " + _clip(text_by_id.get(miss.query_id, ""), WIDTH - 10)
        )
        lines.append(
            "      chose " + _clip(backend_label(miss.example_chosen), 40)
        )
        if miss.example_rationale:
            lines.append("      " + _clip(miss.example_rationale, WIDTH - 6))
    return lines


def render_verdict(verdicts: Sequence[SurvivalVerdict]) -> list[str]:
    """The falsifiable part: does the number already in the README survive."""
    lines = ["  DOES THE PUBLISHED RANKING SURVIVE?"]
    for verdict in verdicts:
        lines.append("")
        lines.append(f"    {verdict.challenger} vs {verdict.baseline}")
        lines.extend(_wrap(verdict.reason, "      "))
    lines.append("")
    lines.extend(
        _wrap(
            "n=5 DETECTS INSTABILITY AND DOES NOT RANK TWO ROUTERS THAT "
            "STRADDLE EACH OTHER. Correctness over 19 queries moves in steps "
            "of 0.053, five draws is not a sample anyone should compute a "
            "p-value from, and none is computed. A router that NEVER exceeded "
            "another is not the same finding as one whose runs fall on both "
            "sides of it, and the two are printed differently above.",
            "  ",
        )
    )
    return lines
