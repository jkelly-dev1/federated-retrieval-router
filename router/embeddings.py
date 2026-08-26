"""A deterministic embedder, and an honest account of what it cannot show.

The default embedder is a hash-based bag-of-tokens model so the whole suite is
reproducible offline with no key and no network. It places texts near each
other when they SHARE TOKENS. A real embedding model places them near each
other when they MEAN the same thing, and that difference is the entire
competence the vector leg is supposed to have.

So the mock understates the vector leg, and the project says so rather than
quietly benefiting. A paraphrase query that shares no vocabulary with its
target document is exactly what a semantic model finds and this one cannot.
Every offline number about semantic retrieval is therefore a FLOOR, not an
estimate, and the real-model capture exists to measure the gap.

That direction matters. An instrument that understated the GRAPH leg would
flatter this project's likely headline, "vector-only is usually fine", and
would have to be treated as suspect. This one understates the vector leg, which
works against that headline, so a vector-only result that still looks good
offline is a conservative result rather than a convenient one.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Protocol, Sequence

# A width of 256 is carried over from the sibling ai-data-boundary-proxy, where
# a dimension sweep showed 64 manufactures similarity through hash collisions
# and 256 is the smallest width where unrelated texts stop colliding on this
# corpus size.
# tests/test_backends.py::test_the_dimension_is_wide_enough_to_avoid
# _manufactured_similarity re-derives it rather than trusting it.
DIMENSIONS = 256

# Dots are allowed INSIDE an identifier and never at the end. The obvious
# pattern, [a-z0-9_.]+, absorbs a sentence-ending period, so a runbook saying
# "the setting is auth.audience.allowlist." indexes a token that a query for
# `auth.audience.allowlist` can never match. That failure is silent. The
# fulltext leg simply returns nothing on precisely the identifier lookups it is
# supposed to be best at, and it is why tests/test_backends.py::test_an
# _identifier_does_not_absorb_trailing_punctuation pins punctuation handling
# rather than assuming it.
_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")

# Words that carry no retrieval signal. Kept SHORT on purpose: an aggressive
# stop list is a silent tuning knob, and half the point of the trap queries is
# that words like "how many" and "owns" are signal rather than noise.
STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
     "of", "to", "in", "on", "for", "and", "or", "that", "this", "it",
     "we", "our", "us", "be", "been", "with", "as", "at", "by", "from"}
)


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> tuple[float, ...]: ...

    def name(self) -> str: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping dots and underscores.

    `queue.prefetch.count` and `ERR_UPSTREAM_4423` must survive tokenization as
    single tokens or the exact-term competence disappears into fragments and
    the fulltext leg stops being able to demonstrate anything.
    """
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


class HashingEmbedder:
    """Signed-hash bag of tokens with L2 normalization.

    The sign matters: without it every token adds positively, long texts drift
    toward a common direction, and cosine similarity becomes a proxy for length
    rather than for content.
    """

    def __init__(self, dimensions: int = DIMENSIONS, seed: str = "router-v1"):
        self.dimensions = dimensions
        self.seed = seed

    def name(self) -> str:
        return f"hashing-{self.dimensions}d-{self.seed}"

    def _slot(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.seed}|{token}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dimensions
        for token in content_tokens(text):
            idx, sign = self._slot(token)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Returns 0.0 for a zero vector rather than raising.

    An all-stopword query is a real thing a user types, and it should retrieve
    nothing rather than crash a sweep.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def default_embedder() -> Embedder:
    return HashingEmbedder()
