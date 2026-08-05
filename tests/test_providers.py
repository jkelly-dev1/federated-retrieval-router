"""The paid seam, tested without paying and without the SDKs installed.

THIS FILE EXISTS BECAUSE OF A REAL FAILURE IN A SIBLING REPOSITORY. Its
provider tests constructed live SDK clients, which passed on a machine where
`openai` and `anthropic` happened to be installed and failed on all three
Python versions in CI, where they deliberately are not. The lesson is in the
first test below: the interesting assertion is not that the provider works with
the SDK present, it is what happens when it is absent.

So every test here runs with no key, no network and no optional dependency.
"""
from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from router.embeddings import HashingEmbedder
from router.models import Backend
from router.providers import (
    ROUTER_SYSTEM,
    CachingEmbedder,
    LLMRouter,
    OpenAIEmbedder,
    _load_env_file,
    build_providers,
)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------- the optional dependency


def test_a_missing_sdk_is_a_sentence_not_a_traceback(monkeypatch):
    """The failure that broke CI on a sibling repository, pinned here.

    `openai` is commented out of requirements.txt on purpose, so the offline
    suite must behave identically whether or not it happens to be installed.
    Hiding the import makes this test say the same thing on both machines.
    """
    real_import = builtins.__import__

    def no_openai(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_openai)
    with pytest.raises(RuntimeError, match="pip install openai"):
        OpenAIEmbedder("sk-not-a-real-key")


def test_the_offline_suite_needs_no_key(monkeypatch):
    """A missing key is a configuration state, not an error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ENV_FILE", raising=False)
    providers = build_providers()
    assert providers.embedding_provider == "mock"
    assert providers.llm_routers == ()


def test_an_explicit_variable_beats_the_env_file(monkeypatch, tmp_path):
    """`setdefault`, not assignment.

    A stale key in a file the user forgot about must not silently override the
    one they just typed on the command line.

    This exercises `_load_env_file` rather than `build_providers`, because
    building providers with a key present constructs a live SDK client -- and
    a test that needs the optional dependency in order to make its point is
    the exact test that passes locally and fails in CI.
    """
    import os

    env_file = tmp_path / "keys.env"
    env_file.write_text(
        "OPENAI_API_KEY=from-the-file\n"
        "# comment\n"
        "\n"
        "ANTHROPIC_API_KEY=only-in-the-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-command-line")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _load_env_file()
    assert os.environ["OPENAI_API_KEY"] == "from-the-command-line"
    assert os.environ["ANTHROPIC_API_KEY"] == "only-in-the-file"


def test_a_missing_env_file_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent.env"))
    with pytest.raises(SystemExit, match="does not exist"):
        _load_env_file()


def test_a_key_without_the_sdk_gets_the_actionable_message(monkeypatch):
    """The user story behind the CI failure: a key in the environment, the
    optional package not installed. That must name the missing package rather
    than surfacing a bare ImportError from three frames down."""
    real_import = builtins.__import__

    def no_openai(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("ENV_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setattr(builtins, "__import__", no_openai)
    with pytest.raises(RuntimeError, match="pip install openai"):
        build_providers()


def test_the_capture_refuses_rather_than_faking_one(tmp_path):
    """A script that prints a plausible capture with nothing behind it is
    worse than a script that exits."""
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ENV_FILE"}
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "real_run.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=180,
    )
    assert result.returncode == 2
    assert "will not produce a file that looks like there was" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "value, message",
    [("abc", "not an integer"), ("0", "at least 1"), ("-3", "at least 1")],
)
def test_a_bad_run_count_is_rejected_before_anything_is_spent(value, message, tmp_path):
    """STABILITY_RUNS is documented in the README, so it is a claim.

    It is also validated BEFORE providers are built, which is the part worth
    pinning: a typo in the run count must not be discovered after the first
    ninety paid calls have already gone out.
    """
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ENV_FILE"}
    }
    env["STABILITY_RUNS"] = value
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "real_run.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=180,
    )
    assert result.returncode == 1
    assert message in result.stderr
    assert not result.stdout


# ------------------------------------------------------------- the embedder


class _FakeEmbeddings:
    def __init__(self, vector, prompt_tokens=7):
        self.vector = vector
        self.prompt_tokens = prompt_tokens
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        data = [type("D", (), {"embedding": list(self.vector)})()]
        usage = type("U", (), {"prompt_tokens": self.prompt_tokens})()
        return type("R", (), {"data": data, "usage": usage})()


class _FakeOpenAI:
    def __init__(self, vector, prompt_tokens=7):
        self.embeddings = _FakeEmbeddings(vector, prompt_tokens)


def test_an_empty_text_costs_nothing(monkeypatch):
    """The API rejects empty input, and cosine already returns 0.0 for a zero
    vector, so an empty document simply never matches -- as with the mock."""
    fake = _FakeOpenAI([0.1] * 4)
    embedder = OpenAIEmbedder("k", dimensions=4, client=fake)
    assert embedder.embed("   ") == tuple([0.0] * 4)
    assert fake.embeddings.calls == 0
    assert embedder.usage.calls == 0


def test_a_dimension_mismatch_is_caught_at_the_seam():
    """A vector of the wrong width would otherwise become silent nonsense two
    layers down, inside a cosine that happily consumes it."""
    embedder = OpenAIEmbedder("k", dimensions=1536, client=_FakeOpenAI([0.1] * 4))
    with pytest.raises(ValueError, match="expected 1536"):
        embedder.embed("anything")


def test_the_memo_is_what_keeps_the_capture_comparable():
    """Two sections embedding the same 85 documents must get the same vectors,
    or a vendor's own nondeterminism moves numbers that are supposed to be
    comparing embedders -- and the second copy is billed."""
    inner = OpenAIEmbedder("k", dimensions=4, client=_FakeOpenAI([0.1] * 4))
    cached = CachingEmbedder(inner)
    first = cached.embed("checkout retries twice")
    second = cached.embed("checkout retries twice")
    assert first is second
    assert (cached.hits, cached.misses) == (1, 1)
    assert inner.usage.calls == 1


def test_the_cached_embedder_reports_the_real_width():
    cached = CachingEmbedder(HashingEmbedder())
    assert cached.dimensions == HashingEmbedder().dimensions
    assert "cached(" in cached.name()


# --------------------------------------------------------------- the router


class _FakeCompletions:
    def __init__(self, body):
        self.body = body

    def create(self, **kwargs):
        message = type("M", (), {"content": self.body})()
        choice = type("C", (), {"message": message})()
        usage = type("U", (), {"prompt_tokens": 40, "completion_tokens": 12})()
        return type("R", (), {"choices": [choice], "usage": usage})()


class _FakeOpenAIChat:
    def __init__(self, body):
        self.chat = type("X", (), {"completions": _FakeCompletions(body)})()


def _router(body):
    return LLMRouter("k", provider="openai", client=_FakeOpenAIChat(body))


def test_an_unparseable_reply_is_an_empty_choice_not_a_fan_out():
    """MARKING YOUR OWN HOMEWORK, PREVENTED.

    A router that quietly falls back to every backend when its parser fails
    would score its own failures as correct -- fan-out cannot be wrong. The
    failure has to be visible in the numbers, so it chooses nothing and says
    so.
    """
    decision = _router("I think you should probably use the graph.").route("q", "q1")
    assert decision.chosen == frozenset()
    assert any("unparseable" in r for r in decision.rationale)


def test_a_fenced_reply_is_still_parsed():
    decision = _router(
        '```json\n{"backends": ["fulltext"], "why": "an error code"}\n```'
    ).route("ERR_UPSTREAM_4423", "q1")
    assert decision.chosen == frozenset({Backend.FULLTEXT})
    assert not decision.rationale[0].endswith("unparseable")


def test_an_invented_backend_name_is_a_parse_failure():
    """A model naming a store this federation does not have must not be
    rounded to the nearest real one."""
    decision = _router('{"backends": ["elasticsearch"], "why": "x"}').route("q", "q1")
    assert decision.chosen == frozenset()
    assert decision.query_id == "q1"


def test_every_llm_decision_carries_a_rationale():
    decision = _router('{"backends": ["vector"], "why": "paraphrase"}').route("q", "q1")
    assert decision.rationale and all(r.strip() for r in decision.rationale)
    assert "llm-openai" in decision.rationale[0]


def test_the_prompt_does_not_hand_the_model_the_heuristic(monkeypatch):
    """THE FAIRNESS INVARIANT OF THE WHOLE COMPARISON.

    A model told "config keys look like a.b.c" or handed the corpus entity
    names would be reciting the guard rather than being compared with it. The
    prompt describes the four backends by competence and nothing else.
    """
    from router.corpus import build_corpus

    lowered = ROUTER_SYSTEM.lower()
    for tell in ("a.b.c", "how many", "which service owns", "countable", "suppress"):
        assert tell not in lowered, f"the prompt leaks the heuristic's tell: {tell}"
    corpus = build_corpus()
    entities = {e.source for e in corpus.edges} | {e.target for e in corpus.edges}
    assert len(entities) > 10
    for entity in entities:
        assert entity.lower() not in lowered, f"the prompt names an entity: {entity}"
