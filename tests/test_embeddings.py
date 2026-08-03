"""The embedder is experimental apparatus, so its parameters are tested."""
from __future__ import annotations

import pytest

from boundary.corpus import build_corpus
from boundary.embeddings import (
    DIMENSIONS,
    HashingEmbedder,
    cosine,
    default_embedder,
    tokenize,
)
from boundary.erasure import (
    ErasureEngine,
    ErasureStrategy,
    build_index,
)
from boundary.verify import ErasureVerifier, separation_band


def test_the_embedder_is_deterministic():
    a, b = HashingEmbedder(), HashingEmbedder()
    assert a.embed("Jordan Rivera") == b.embed("Jordan Rivera")


def test_texts_sharing_tokens_are_closer_than_texts_that_do_not():
    """The property the neighbor probe depends on.

    Without it the probe could detect nothing and the erasure demonstration
    would be a tautology.
    """
    emb = default_embedder()
    q = emb.embed("jordan rivera payroll")
    near = emb.embed("jordan rivera holds payroll access")
    far = emb.embed("quarterly revenue projections for the northeast region")
    assert cosine(q, near) > cosine(q, far)


def test_the_signed_hash_stops_similarity_from_tracking_length():
    """Unsigned accumulation would make every long text point the same way."""
    emb = default_embedder()
    long_a = emb.embed(" ".join(f"alpha{i}" for i in range(200)))
    long_b = emb.embed(" ".join(f"beta{i}" for i in range(200)))
    assert cosine(long_a, long_b) < 0.2


def test_the_dimension_sweep_still_justifies_the_pinned_value():
    """The measurement behind DIMENSIONS, re-run rather than trusted.

    At 64 dimensions the two populations must still fail to separate, and at
    256 they must still succeed. If this breaks, the constant needs
    re-deriving rather than nudging.
    """
    corpus = build_corpus()

    small = HashingEmbedder(dimensions=64)
    small_band = separation_band(build_index(build_corpus().store, small), small,
                                 corpus.subjects[0])
    assert small_band["separated"] == 0.0

    chosen = HashingEmbedder(dimensions=DIMENSIONS)
    chosen_band = separation_band(build_index(build_corpus().store, chosen), chosen,
                                  corpus.subjects[0])
    assert chosen_band["separated"] == 1.0


def test_a_complete_erasure_verifies_clean_at_the_pinned_dimension():
    """The other half of the sweep: 256 must leave no false positives."""
    emb = HashingEmbedder(dimensions=DIMENSIONS)
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]
    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        subject, ErasureStrategy.FULL
    )
    assert not ErasureVerifier(corpus.store, index, emb).verify(subject).leaked


def test_cosine_of_a_zero_vector_is_zero_not_an_error():
    """An empty chunk is a real thing in a corpus and should never match."""
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_cosine_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_an_empty_text_embeds_to_a_zero_vector():
    emb = default_embedder()
    assert set(emb.embed("")) == {0.0}


def test_tokenize_is_case_insensitive_and_drops_punctuation():
    assert tokenize("Jordan Rivera, SSN 482-19-7734!") == [
        "jordan", "rivera", "ssn", "482", "19", "7734"
    ]
