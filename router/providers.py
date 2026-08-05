"""Real providers for the paid capture, and nothing else imports this.

The offline stack has no configuration and no dependencies. This module is the
one seam where a key and a network appear, and it exists to answer two
questions the deterministic stack structurally cannot:

  DOES A REAL EMBEDDER RECLAIM ITS OWN QUERIES? The mock is a bag-of-tokens
  hash, so on a paraphrase sharing one rare term with its target BM25 wins.
  Four of eleven document-bearing queries go to the wrong leg because of it.
  A semantic model should take them back, and if it does not that is a more
  interesting result than if it does.

  DOES AN LLM ROUTER BEAT A KEYWORD ONE? The heuristic router reaches 0.947 on
  two hand-written guards. A model reading the same question has no keyword
  list and no corpus knowledge, and the comparison is fair only if it is scored
  on the same two axes -- correctness AND fan-out. A model that routes
  perfectly by selecting everything has not beaten anything.

THE HOUSE RULE APPLIES HERE TOO: a provider NAME without its matching KEY falls
back to the offline component rather than raising. A missing key is a
configuration state, not an error, and treating it as an error is how a suite
ends up green only on the maintainer's machine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from router.embeddings import Embedder, HashingEmbedder
from router.models import Backend, Competence, RoutingDecision

OPENAI_EMBED_DIMENSIONS = 1536


def _load_env_file() -> None:
    """Load keys from a private file outside the repo, if ENV_FILE points at one.

    setdefault, not assignment: an explicit variable on the command line must
    win over a stale value in the file. The file's PATH appears in commands and
    transcripts; its CONTENTS never do.
    """
    path = os.environ.get("ENV_FILE")
    if not path:
        return
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise SystemExit(f"ENV_FILE={expanded} does not exist")
    with open(expanded, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


@dataclass
class CachingEmbedder:
    """Memoize by exact text.

    The corpus is embedded once and every query once. Without the memo the
    competence sweep would pay for the same 85 document vectors on every
    section, and a vendor's own nondeterminism would move numbers between
    sections that are supposed to be comparing embedders.
    """

    inner: Embedder
    cache: dict[str, tuple[float, ...]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def name(self) -> str:
        return f"cached({self.inner.name()})"

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self.cache.get(text)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed(text)
        self.cache[text] = vector
        return vector


class OpenAIEmbedder:
    """text-embedding-3-* behind the `Embedder` protocol.

    An empty text returns a zero vector without calling the API, which rejects
    empty input. `cosine` already returns 0.0 for a zero vector, so an empty
    document simply never matches -- the same behavior the mock has.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = OPENAI_EMBED_DIMENSIONS,
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.usage = Usage()
        if client is not None:
            self._client = client
            return
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The paid capture needs the 'openai' package, which is "
                "commented out of requirements.txt so the offline suite "
                "installs nothing that can reach a network.\n"
                "    pip install openai"
            ) from exc
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def name(self) -> str:
        return f"openai-{self.model}-{self.dimensions}d"

    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            return tuple([0.0] * self.dimensions)
        response = self._client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimensions
        )
        usage = getattr(response, "usage", None)
        self.usage.add(input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0))
        vector = tuple(float(v) for v in response.data[0].embedding)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"{self.model} returned {len(vector)} dimensions, "
                f"expected {self.dimensions}"
            )
        return vector


# The routing prompt. Deliberately describes the four backends by COMPETENCE
# and never mentions the corpus, the entity names or the heuristic's tells --
# a model told "config keys look like a.b.c" would be reciting the guard rather
# than being compared with it.
ROUTER_SYSTEM = """\
You route a question to the retrieval backends that can answer it.

  vector      dense embeddings. Paraphrase and concept questions, where the
              answer is prose that means the same thing in different words.
  fulltext    an inverted index. Exact identifiers, error codes, config keys,
              literal strings.
  graph       typed relationship edges between services, teams and incidents.
              Ownership, dependency and multi-hop questions whose answer is not
              written down in any single document.
  relational  rows with aggregates. Counts, averages, maxima and time windows,
              where the answer is a number that has to be computed.

Choose the SMALLEST set of backends that can answer the question correctly.
Choosing everything is always safe and always wrong: it is scored as a cost.

Reply with JSON only: {"backends": ["..."], "why": "one short sentence"}\
"""


@dataclass
class LLMRouter:
    """A model deciding the same question the heuristic router decides.

    Scored on the same two axes. A model that selects all four backends every
    time is not a better router, it is a fan-out router with a bill, and the
    fan-out column is what makes that visible.
    """

    api_key: str
    provider: str = "openai"
    model: str = "gpt-5.6-terra"
    max_tokens: int = 512
    client: Optional[object] = None
    name: str = "llm"
    usage: Usage = field(default_factory=Usage)
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = f"llm-{self.provider}"
        if self.client is not None:
            return
        if self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        else:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)

    def _ask(self, query: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": query},
                ],
            )
            usage = getattr(response, "usage", None)
            self.usage.add(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )
            return response.choices[0].message.content or ""

        # No temperature, top_p or top_k: removed on claude-opus-5, 400 if sent.
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=ROUTER_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        usage = getattr(message, "usage", None)
        self.usage.add(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        if getattr(message, "stop_reason", None) == "refusal":
            return ""
        return "".join(
            b.text for b in message.content if getattr(b, "type", "") == "text"
        )

    def route(self, query: str, query_id: str = "") -> RoutingDecision:
        """Ask the model, and treat an unparseable reply as an EMPTY choice.

        Not as a fallback to everything. A router that silently fans out when
        its parser fails would score its own failures as correct, which is the
        measurement equivalent of marking your own homework. Parse failures are
        recorded in `failures` and reported in the capture.
        """
        raw = self._ask(query)
        chosen: set[Backend] = set()
        why = ""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            parsed = json.loads(text)
            why = str(parsed.get("why", ""))[:80]
            for name in parsed.get("backends", []):
                chosen.add(Backend(str(name).strip().lower()))
        except (ValueError, KeyError, TypeError) as exc:
            self.failures.append(f"{query_id}: {type(exc).__name__}")
            why = f"unparseable reply ({type(exc).__name__})"

        return RoutingDecision(
            query_id=query_id,
            chosen=frozenset(chosen),
            competences=frozenset(),
            rationale=(f"{self.name}: {why}",),
        )


# ------------------------------------------------------------------ factories


@dataclass(frozen=True)
class Providers:
    embedder: Embedder
    embedding_provider: str
    llm_routers: tuple[LLMRouter, ...]
    new_embedder: Callable[[], Embedder]
    """A SECOND embedder over the same model, with a COLD cache.

    The repeat pass in the capture asks whether the vendor returns the same
    vectors for the same texts. Re-using `embedder` would answer a different
    and worthless question -- whether a dict returns what was put into it --
    so the repeat has to pay for its own calls. It is one extra pass over the
    corpus and it is the cheapest section of the run.
    """


def build_providers(dimensions: int = OPENAI_EMBED_DIMENSIONS) -> Providers:
    """Whatever the environment actually supports, named honestly."""
    _load_env_file()
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    def new_embedder() -> Embedder:
        if openai_key:
            return CachingEmbedder(OpenAIEmbedder(openai_key, dimensions=dimensions))
        return CachingEmbedder(HashingEmbedder())

    embedder = new_embedder()
    embedding_provider = "openai" if openai_key else "mock"

    routers: list[LLMRouter] = []
    if openai_key:
        routers.append(
            LLMRouter(
                openai_key,
                provider="openai",
                model=os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"),
            )
        )
    if anthropic_key:
        routers.append(
            LLMRouter(
                anthropic_key,
                provider="anthropic",
                model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
            )
        )
    return Providers(
        embedder=embedder,
        embedding_provider=embedding_provider,
        llm_routers=tuple(routers),
        new_embedder=new_embedder,
    )
