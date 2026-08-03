"""Embeddings, and why they are the artifact erasure forgets.

An embedding is DERIVED data. It is computed from text, it is stored separately
from that text, and it survives the deletion of its source. That is the whole
premise of this project: a subject deleted from the source database is still
present, in a lossy but recoverable form, inside every vector computed from a
chunk that mentioned them.

The default embedder is a deterministic hash-based model so the entire suite is
reproducible offline. It is NOT a semantic model, and the README says so: it
places texts that share tokens near each other, which is enough to demonstrate
retrieval, neighbor probes and erasure leaks, and is not enough to claim
anything about real embedding quality. The real-provider seam exists for the
paid run.

One property is deliberate and matters: the mock embedder is TOKEN-ADDITIVE, so
a chunk mentioning a subject sits measurably closer to that subject's
identifiers than an unrelated chunk. Without that property the neighbor probe
in verify.py could not detect anything, and the erasure demonstration would be
a tautology.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Protocol, Sequence

# A signed hashing embedder collides tokens into shared dimensions, and at low
# dimensionality those collisions manufacture similarity between unrelated
# texts. That is not a harmless approximation: it puts a floor under the
# erasure verifier's false-positive rate, so a fully-erased subject can still
# look reachable.
#
# Measured on the shipped corpus (tests/test_embeddings.py re-runs this sweep):
#
#     dims   min_with   max_without   separated   false positives after FULL
#       64      0.061         0.100          no                            1
#      128      0.171         0.150         yes                            1
#      256      0.120         0.071         yes                            0
#      512      0.136         0.071         yes                            0
#
# A width of 256 is the smallest size where the two populations separate AND
# a complete erasure verifies clean. Real embedding models run 384-3072, so
# this is also the first value in the sweep that is not unrealistically
# small.
DIMENSIONS = 256

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """The seam a real provider plugs into."""

    dimensions: int

    def embed(self, text: str) -> tuple[float, ...]:
        ...

    def name(self) -> str:
        ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Deterministic bag-of-tokens embedder with L2 normalization.

    Each token is hashed to a dimension and a sign, then accumulated. Two texts
    sharing tokens therefore share components, and cosine similarity behaves
    the way retrieval needs it to.

    The signed hash matters: without it, every token adds positively and every
    long text drifts toward the same direction, which would make similarity a
    proxy for length rather than for content.
    """

    def __init__(self, dimensions: int = DIMENSIONS, seed: str = "boundary-v1"):
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
        tokens = tokenize(text)
        for tok in tokens:
            idx, sign = self._slot(tok)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Both inputs are expected to be normalized already.

    Returns 0.0 for a zero vector rather than raising: an empty chunk is a real
    thing in a corpus and should simply never match, not crash a sweep.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embed_all(embedder: Embedder, texts: Iterable[str]) -> list[tuple[float, ...]]:
    return [embedder.embed(t) for t in texts]


def default_embedder() -> Embedder:
    return HashingEmbedder()
