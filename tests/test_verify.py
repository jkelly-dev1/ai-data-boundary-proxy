"""The verifier is an instrument, so it gets measured too."""
from __future__ import annotations

import pytest

from boundary.corpus import build_corpus
from boundary.embeddings import HashingEmbedder, default_embedder
from boundary.erasure import ErasureEngine, ErasureStrategy, build_index
from boundary.index import VectorIndex
from boundary.models import IndexEntry
from boundary.verify import (
    COMMON_TOKEN_FRACTION,
    NEIGHBOR_THRESHOLD,
    ErasureVerifier,
    separation_band,
    subject_query,
)


def test_the_neighbor_threshold_sits_between_the_two_populations(indexed):
    """The calibration invariant.

    max_without < threshold < min_with. Above min_with the probe misses real
    mentions; below max_without it reports strangers. A corpus or embedder
    change that invalidates the constant fails here rather than silently
    weakening the probe.
    """
    corpus, index, emb = indexed
    band = separation_band(index, emb, corpus.subjects[0])
    assert band["max_without"] < NEIGHBOR_THRESHOLD, band
    assert NEIGHBOR_THRESHOLD < band["min_with"], band
    assert band["separated"] == 1.0


def test_the_probe_query_drops_the_email_domain(indexed):
    """An email is two identifiers; only the personal half belongs in a probe.

    Mutation check, executed in-test: the raw identifier join is computed here
    and asserted to contain the domain token the real query excludes.
    """
    corpus, index, _ = indexed
    subject = corpus.subjects[0]
    query = subject_query(index, subject)
    assert "rivera" in query
    assert "jrivera" in query
    assert "acme" not in query.split(), query
    assert "example" not in query.split(), query

    naive = " ".join(sorted(subject.identifiers)).lower()
    assert "acme" in naive, "the mutation control failed: naive join should leak it"


def test_the_frequency_baseline_excludes_the_subjects_own_mentions(indexed):
    """A heavily-mentioned subject must not be filtered out as background.

    Mutation check, executed in-test: document frequency computed over the
    WHOLE index is shown to exceed the cutoff for the subject's own surname,
    which is what a whole-corpus baseline would discard.
    """
    corpus, index, _ = indexed
    subject = corpus.subjects[0]
    query = subject_query(index, subject).split()
    assert "rivera" in query, "the subject's own name must survive the filter"

    from boundary.embeddings import tokenize

    entries = list(index.entries.values())
    whole_corpus_freq = sum(
        1 for e in entries if "rivera" in set(tokenize(e.text))
    )
    cutoff = max(1, int(len(entries) * COMMON_TOKEN_FRACTION))
    assert whole_corpus_freq > cutoff, (
        "the mutation control failed: a whole-corpus baseline should have "
        "discarded this token"
    )


def test_the_neighbor_probe_finds_a_vector_whose_text_is_gone():
    """The probe's whole reason to exist.

    An orphaned vector, text deleted everywhere, embedding left behind, is
    invisible to every string search and must still be caught.
    """
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]

    # Delete the subject's text from every store but leave the vectors alone.
    for chunk in list(corpus.store.chunks_mentioning(subject)):
        corpus.store.remove_chunk(chunk.chunk_id)
    for doc in list(corpus.store.documents_mentioning(subject)):
        corpus.store.remove_document(doc.doc_id)
    for entry in corpus.store.cache_entries_mentioning(subject):
        corpus.store.remove_cache_entry(entry.key)
    for entry in corpus.store.prompt_entries_mentioning(subject):
        corpus.store.remove_prompt_entry(entry.entry_id)

    report = ErasureVerifier(corpus.store, index, emb).verify(subject)
    probes = {p.probe: p for p in report.probes}
    assert not probes["source_scan"].leaked
    assert not probes["chunk_scan"].leaked
    assert probes["neighbor_probe"].leaked, "the orphaned vector must be found"


def test_the_verifier_does_not_consult_the_lineage_table(indexed):
    """Independence claim: erasing lineage does not blind the verifier.

    If verification walked store.derivations it would confirm the eraser's
    blind spots. Wiping the lineage entirely must not change any probe result.
    """
    corpus, index, emb = indexed
    subject = corpus.subjects[0]
    before = ErasureVerifier(corpus.store, index, emb).verify(subject)
    corpus.store.derivations.clear()
    after = ErasureVerifier(corpus.store, index, emb).verify(subject)
    assert [p.leaked for p in before.probes] == [p.leaked for p in after.probes]


def test_probes_report_hits_not_just_a_boolean(indexed):
    """A count is believable; a list is checkable.

    Every leaking probe must name what it found, because inspecting the hits is
    the only way to tell a real leak from a probe artifact.
    """
    corpus, index, emb = indexed
    report = ErasureVerifier(corpus.store, index, emb).verify(corpus.subjects[0])
    for probe in report.leaking_probes:
        if probe.probe == "cross_tenant_probe":
            continue
        assert probe.hits, f"{probe.probe} reported a leak with no hits"


def test_a_low_dimension_embedder_manufactures_false_neighbors():
    """Collision floor: the probe's precision depends on the embedder.

    At 64 dimensions hash collisions create similarity between unrelated texts,
    so the populations stop separating. This is why DIMENSIONS is pinned by a
    measured sweep rather than chosen for speed.
    """
    corpus = build_corpus()
    small = HashingEmbedder(dimensions=64)
    index = build_index(corpus.store, small)
    band = separation_band(index, small, corpus.subjects[0])
    assert band["separated"] == 0.0, (
        "64 dimensions is expected to overlap; if this now separates, re-run "
        "the sweep in boundary/embeddings.py and update DIMENSIONS"
    )


def test_an_empty_index_does_not_crash_the_probe_query():
    from boundary.models import Subject

    subject = Subject(
        subject_id="s", identifiers=frozenset({"Nobody"}), tenant_id="acme"
    )
    assert subject_query(VectorIndex(), subject) == "Nobody"


def test_the_published_leak_counts_hold_for_every_plaintext_strategy():
    """The headline of this repository IS a count of leaking probes.

    A probe that stops finding anything makes the count fall, the erasure look
    better and the report read cleaner, so the count is exactly the number that
    a broken probe improves. Pinning the whole table, rather than one probe,
    is what makes any single probe non-removable.

    Mutation check, executed against the verifier: replace the hit list of
    `_source_scan`, `_chunk_scan`, `_index_text_scan`, `_prompt_log_probe` or
    `_cross_tenant_probe` with an empty list and the corresponding row here
    falls, which this catches. The named probes below are asserted
    individually so a failure says WHICH probe went quiet.
    """
    expected = {
        ErasureStrategy.SOURCE_ONLY: 9,
        ErasureStrategy.SOURCE_AND_INDEX: 9,
        ErasureStrategy.MENTION_SWEEP: 4,
        ErasureStrategy.FULL: 0,
    }
    embedder = default_embedder()

    for strategy, count in expected.items():
        corpus = build_corpus()
        index = build_index(corpus.store, embedder)
        subject = corpus.store.subjects["subj-rivera"]
        ErasureEngine(
            store=corpus.store, index=index, embedder=embedder
        ).erase(subject, strategy)
        report = ErasureVerifier(corpus.store, index, embedder).verify(subject)
        leaking = {p.probe for p in report.probes if p.leaked}
        assert len(leaking) == count, (
            f"{strategy.value}: expected {count} leaking probes, got "
            f"{len(leaking)}: {sorted(leaking)}"
        )

    corpus = build_corpus()
    index = build_index(corpus.store, embedder)
    subject = corpus.store.subjects["subj-rivera"]
    ErasureEngine(store=corpus.store, index=index, embedder=embedder).erase(
        subject, ErasureStrategy.SOURCE_ONLY
    )
    leaking = {
        p.probe
        for p in ErasureVerifier(corpus.store, index, embedder).verify(subject).probes
        if p.leaked
    }
    for probe in (
        "source_scan",
        "chunk_scan",
        "index_text_scan",
        "retrieval_probe",
        "tombstone_probe",
        "neighbor_probe",
        "cache_probe",
        "prompt_log_probe",
        "cross_tenant_probe",
    ):
        assert probe in leaking, f"{probe} found nothing after a source-only delete"
