"""Erasure: the claims this project stands on.

Every test here maps to a row in the README's claims table.
"""
from __future__ import annotations

import pytest

from boundary.corpus import build_corpus
from boundary.embeddings import default_embedder
from boundary.erasure import (
    PLAINTEXT_STRATEGIES,
    ErasureEngine,
    ErasureStrategy,
    build_index,
    rechunk_and_reindex,
    reindex,
)
from boundary.models import Subject
from boundary.verify import ErasureVerifier


def _run(strategy, subject_index: int = 0):
    """Build a fresh corpus, erase, and return (store, index, subject, report)."""
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[subject_index]
    cert = ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        subject, strategy
    )
    verifier = ErasureVerifier(corpus.store, index, emb)
    return corpus.store, index, subject, verifier.verify(subject), cert, emb, verifier


def test_deleting_the_source_row_leaves_the_subject_fully_retrievable():
    """The headline claim: "delete the user" does not erase the user.

    Mutation check, executed in-test: the same corpus under FULL erasure is
    asserted to leak nothing, so this test cannot pass because the probes are
    broken or the subject was never present.
    """
    _, _, _, report, cert, _, _ = _run(ErasureStrategy.SOURCE_ONLY)
    assert report.leaked
    probes = {p.probe for p in report.leaking_probes}
    # Every reachability path is still open, including the two that no string
    # search would ever find.
    assert "retrieval_probe" in probes
    assert "neighbor_probe" in probes
    assert "cache_probe" in probes
    assert cert.total_artifacts() == len(cert.documents_removed)

    _, _, _, full_report, _, _, _ = _run(ErasureStrategy.FULL)
    assert not full_report.leaked, "the mutation control failed: FULL should be clean"


def test_tombstoning_vectors_is_not_removing_them():
    """SOURCE_AND_INDEX marks vectors deleted; the data is still there.

    Mutation check, executed in-test: FULL, which compacts the index after
    tombstoning, leaves no tombstoned entry behind and the tombstone probe
    goes quiet. Remove the `compact()` call from the FULL path and the second
    block fails, because a tombstone that is never reclaimed is still there.
    """
    _, index, subject, report, _, _, _ = _run(ErasureStrategy.SOURCE_AND_INDEX)
    tombstoned = [e for e in index.entries.values() if e.deleted]
    assert tombstoned, "expected tombstones, not removals"
    assert any(subject.matches(e.text) for e in tombstoned)
    assert any(p.probe == "tombstone_probe" and p.leaked for p in report.probes)

    _, full_index, _, full_report, _, _, _ = _run(ErasureStrategy.FULL)
    assert not [e for e in full_index.entries.values() if e.deleted]
    assert not [p for p in full_report.probes if p.probe == "tombstone_probe" and p.leaked]


def test_a_mention_inside_another_partys_document_survives_owner_scoped_erasure():
    """The subject is named in documents they do not own.

    Owner-scoped erasure cannot see them, so the corpus contains a support
    ticket filed by one subject that names another.
    """
    store, _, subject, _, _, _, _ = _run(ErasureStrategy.SOURCE_AND_INDEX)
    survivors = store.chunks_mentioning(subject)
    assert survivors, "expected mentions in other parties' documents to survive"
    owners = {store.chunks[c.chunk_id].doc_id for c in survivors}
    assert "acme-ticket-114" in owners or "acme-note-220" in owners


def test_mention_sweep_rewrites_rather_than_deletes_another_partys_record():
    """Deleting a colleague's ticket to erase one subject destroys their record.

    Mutation check, executed in-test: with redact_mentions=False the engine
    deletes those chunks instead, and the other party's document loses content.
    """
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]
    before = len(corpus.store.chunks_for_document("acme-ticket-114"))
    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        subject, ErasureStrategy.MENTION_SWEEP
    )
    after = len(corpus.store.chunks_for_document("acme-ticket-114"))
    assert after == before, "the colleague's record should be rewritten, not removed"
    remaining = " ".join(c.text for c in corpus.store.chunks_for_document("acme-ticket-114"))
    assert "Priya Raman" in remaining, "the other party's own data must survive"
    assert not subject.matches(remaining)

    corpus2 = build_corpus()
    index2 = build_index(corpus2.store, emb)
    ErasureEngine(
        store=corpus2.store, index=index2, embedder=emb, redact_mentions=False
    ).erase(corpus2.subjects[0], ErasureStrategy.MENTION_SWEEP)
    assert len(corpus2.store.chunks_for_document("acme-ticket-114")) < before


def test_rewriting_a_chunk_without_reembedding_leaves_the_old_vector():
    """The vector is the disclosure, not just the text.

    Mutation check, executed in-test: the engine's own path re-embeds, and the
    resulting vector differs from the one computed from the original text.
    """
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]
    target = next(
        c for c in corpus.store.chunks_mentioning(subject) if c.doc_id == "acme-ticket-114"
    )
    original_vector = index.get(target.chunk_id).vector

    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        subject, ErasureStrategy.MENTION_SWEEP
    )
    new_vector = index.get(target.chunk_id).vector
    assert new_vector != original_vector, "the vector must be recomputed"

    # And the recomputed vector matches the redacted text, not the old text.
    redacted_text = corpus.store.chunks[target.chunk_id].text
    assert new_vector == emb.embed(redacted_text)


def test_full_erasure_verifies_clean_against_every_probe():
    _, _, _, report, cert, _, _ = _run(ErasureStrategy.FULL)
    assert not report.leaked, report.summary()
    assert cert.total_artifacts() > 0


def test_an_incomplete_erasure_is_undone_by_a_routine_reingest():
    """The finding: a re-ingest restores a subject swept only from derived data.

    Mutation check, executed in-test: FULL, which also rewrites the source
    documents, survives the identical re-ingest with zero leaks.
    """
    emb = default_embedder()

    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]
    engine = ErasureEngine(store=corpus.store, index=index, embedder=emb)
    engine.erase(subject, ErasureStrategy.MENTION_SWEEP)
    verifier = ErasureVerifier(corpus.store, index, emb)
    before = len(verifier.verify(subject).leaking_probes)

    rechunk_and_reindex(corpus.store, index, emb)
    after = len(verifier.verify(subject).leaking_probes)
    assert after > before, "re-ingest should resurrect the subject"

    corpus2 = build_corpus()
    index2 = build_index(corpus2.store, emb)
    subject2 = corpus2.subjects[0]
    ErasureEngine(store=corpus2.store, index=index2, embedder=emb).erase(
        subject2, ErasureStrategy.FULL
    )
    rechunk_and_reindex(corpus2.store, index2, emb)
    assert not ErasureVerifier(corpus2.store, index2, emb).verify(subject2).leaked


def test_strategies_are_ordered_by_thoroughness():
    """Each strategy leaks no more than the weaker one before it."""
    counts = []
    for strat in PLAINTEXT_STRATEGIES:
        _, _, _, report, _, _, _ = _run(strat)
        counts.append(len(report.leaking_probes))
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], "the corpus must discriminate between strategies"


def test_erasing_one_subject_does_not_erase_another():
    """A sweep keyed on identifiers must not catch a different person.

    Mutation check, executed in-test: the second subject is asserted still
    fully reachable, so a sweep that deleted everything would fail here.
    """
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    target, bystander = corpus.subjects[0], corpus.subjects[1]
    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        target, ErasureStrategy.FULL
    )
    verifier = ErasureVerifier(corpus.store, index, emb)
    assert not verifier.verify(target).leaked
    assert verifier.verify(bystander).leaked, "the bystander must be untouched"


def test_reindex_from_chunks_is_the_expensive_path_that_always_works():
    """Rebuilding derived data from cleaned chunks removes every stale vector."""
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    subject = corpus.subjects[0]
    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        subject, ErasureStrategy.MENTION_SWEEP
    )
    reindex(corpus.store, index, emb)
    stale = [e for e in index.entries.values() if subject.matches(e.text)]
    assert not stale


def test_a_subject_with_no_identifiers_matches_nothing():
    """Guard against a sweep that erases the corpus when given an empty set."""
    empty = Subject(subject_id="s0", identifiers=frozenset(), tenant_id="acme")
    assert not empty.matches("Jordan Rivera worked here")
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    before = len(corpus.store.chunks)
    ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
        empty, ErasureStrategy.FULL
    )
    assert len(corpus.store.chunks) == before
