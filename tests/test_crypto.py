"""Crypto-shredding: the envelope model, key custody, and rotation.

What these tests protect is not the cipher. It is the three places the
technique leaks: an artifact wrapped for more than one subject, a key copy
nobody purged, and a maintenance job that re-wraps from a backup.
"""
from __future__ import annotations

import pytest

from boundary.corpus import build_corpus
from boundary.crypto import LIVE, KeyStore, seal_store
from boundary.embeddings import default_embedder
from boundary.erasure import ErasureEngine, ErasureStrategy, build_index
from boundary.verify import ErasureVerifier


def sealed_corpus(snapshot: str | None = None, seal_vectors: bool = False):
    corpus = build_corpus()
    emb = default_embedder()
    index = build_index(corpus.store, emb)
    keys = KeyStore()
    sealed = seal_store(
        corpus.store, index, keys, corpus.subjects, seal_vectors=seal_vectors
    )
    if snapshot:
        keys.snapshot(snapshot)
    return corpus, index, emb, keys, sealed


# --------------------------------------------------------------- envelopes


def test_sealing_removes_every_identifier_from_the_stored_bytes():
    corpus, _, _, _, _ = sealed_corpus()
    rivera = corpus.subjects[0]
    assert not any(
        rivera.matches(d.text) for d in corpus.store.documents.values()
    )
    assert not any(rivera.matches(c.text) for c in corpus.store.chunks.values())


def test_an_envelope_opens_for_every_subject_it_wraps():
    """A record concerning two people is wrapped twice, for the same content
    key, so either subject's key opens it."""
    corpus, _, _, keys, sealed = sealed_corpus()
    ticket = sealed.envelopes["acme-ticket-114"]
    assert ticket.concerns() == ("subj-raman", "subj-rivera")
    assert keys.open(ticket, LIVE) is not None


def test_destroying_one_subjects_key_leaves_another_subjects_wrap():
    """The multi-subject leak, stated as a test.

    Mutation check, executed in-test: destroying BOTH keys does close it, so
    the surviving wrap is the reason and not some other property of the record.
    """
    corpus, _, _, keys, sealed = sealed_corpus()
    ticket = sealed.envelopes["acme-ticket-114"]

    keys.destroy("subj-rivera")
    opened = keys.open(ticket, LIVE)
    assert opened is not None, "Raman's wrap should still open her own ticket"
    assert b"Jordan Rivera" in opened, "and Rivera is still inside it"

    keys.destroy("subj-raman")
    assert keys.open(ticket, LIVE) is None


def test_a_subject_only_artifact_becomes_unreadable():
    corpus, _, _, keys, sealed = sealed_corpus()
    hr = sealed.envelopes["acme-hr-001"]
    assert hr.concerns() == ("subj-rivera",)
    keys.destroy("subj-rivera")
    assert keys.open(hr, LIVE) is None


# ------------------------------------------------------------- key custody


def test_a_snapshot_taken_before_the_destroy_still_opens_the_data():
    _, _, _, keys, sealed = sealed_corpus(snapshot="nightly")
    hr = sealed.envelopes["acme-hr-001"]
    receipt = keys.destroy("subj-rivera")

    assert receipt.removed_from == (LIVE,)
    assert receipt.surviving_sources == ("snapshot:nightly",)
    assert not receipt.complete
    assert keys.open(hr, LIVE) is None
    assert keys.open(hr, "snapshot:nightly") is not None


def test_an_escrowed_key_is_found_by_the_key_material_probe():
    """The probe tries the escrow account too, not only the live keyring and
    the snapshots, and a key escrowed before the destroy still opens the
    subject's data."""
    corpus, index, emb, keys, sealed = sealed_corpus()
    rivera = corpus.subjects[0]
    keys.escrow_key(rivera.subject_id)
    keys.destroy(rivera.subject_id)
    verifier = ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)
    probe = next(
        p for p in verifier.verify(rivera).probes if p.probe == "key_material_probe"
    )
    assert probe.leaked
    assert "escrow" in probe.hits


def test_purging_snapshots_completes_the_destroy():
    """Mutation check for the test above: with no backup, destroy is complete."""
    _, _, _, keys, sealed = sealed_corpus(snapshot="nightly")
    keys.purge_snapshots()
    receipt = keys.destroy("subj-rivera")
    assert receipt.surviving_sources == ()
    assert receipt.complete


def test_the_key_probe_is_scoped_to_the_subjects_own_wrap():
    """Not "can anything open this"; "can THIS subject's wrap be unwrapped".

    The broad question would report a leak for Rivera every time Raman's key
    opened her own ticket, which is true of the record and false of the key.
    """
    _, _, _, keys, sealed = sealed_corpus()
    ticket = sealed.envelopes["acme-ticket-114"]
    keys.destroy("subj-rivera")

    assert keys.wrap_sources("subj-rivera", ticket) == ()
    assert keys.wrap_sources("subj-raman", ticket) == (LIVE,)
    # The envelope itself is still openable, via the other subject.
    assert keys.open(ticket, LIVE) is not None


def test_the_key_material_probe_attempts_a_real_decrypt():
    """Mutation check on the probe's independence.

    A key store that LIES about what it removed must not fool the probe. The
    subclass below reports a complete destroy while silently keeping the key;
    the probe still finds it, because it decrypts instead of asking.
    """

    class LyingKeyStore(KeyStore):
        def destroy(self, subject_id):
            from boundary.crypto import DestroyReceipt

            return DestroyReceipt(subject_id=subject_id, removed_from=(LIVE,))

        def sources_holding(self, subject_id):
            return ()

    corpus = build_corpus()
    emb = default_embedder()
    index = build_index(corpus.store, emb)
    keys = LyingKeyStore()
    sealed = seal_store(corpus.store, index, keys, corpus.subjects)
    rivera = corpus.subjects[0]

    receipt = keys.destroy(rivera.subject_id)
    assert receipt.complete, "the store claims a clean destroy"

    verifier = ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)
    probe = next(
        p for p in verifier.verify(rivera).probes if p.probe == "key_material_probe"
    )
    assert probe.leaked, "the probe must not believe the key store"
    assert LIVE in probe.hits


def test_the_key_probe_reports_not_applicable_on_a_plaintext_store():
    corpus = build_corpus()
    emb = default_embedder()
    index = build_index(corpus.store, emb)
    verifier = ErasureVerifier(corpus.store, index, emb)
    probe = next(
        p
        for p in verifier.verify(corpus.subjects[0]).probes
        if p.probe == "key_material_probe"
    )
    assert not probe.leaked
    assert "plaintext posture" in probe.detail


# ----------------------------------------------------------------- probes


def test_a_probe_reads_what_the_running_system_can_read():
    """The false negative this design exists to avoid.

    Ciphertext contains no identifier substring, so a probe matching on stored
    bytes would call an encrypted store clean while the key sits in the
    keyring and any query can open it. `readable()` resolves the artifact
    first, so the store reports leaking until the key is actually gone.
    """
    corpus, index, emb, keys, sealed = sealed_corpus()
    rivera = corpus.subjects[0]
    verifier = ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)

    # Nothing erased yet: the data is encrypted and entirely readable.
    before = verifier.verify(rivera)
    assert before.leaked, "an encrypted store with live keys is not erased"

    # And the stored bytes really are ciphertext, so a naive byte match would
    # have reported clean. This is the control for what follows.
    assert not any(
        rivera.matches(d.text) for d in corpus.store.documents.values()
    )

    # `readable()` itself, asserted directly rather than through whichever
    # probe happens to call it: the stored bytes do not name the subject and
    # what the running system reads out of them does. Mutation check,
    # executed against the verifier: return the stored bytes from
    # `readable()` and this is the assertion that fails, while the string
    # probes above go quiet and the report reads cleaner than it is.
    hr = corpus.store.documents["acme-hr-001"]
    assert rivera.matches(verifier.readable(hr.doc_id, hr.text))
    string_probes = {
        p.probe: p.leaked
        for p in before.probes
        if p.probe in ("source_scan", "chunk_scan", "index_text_scan")
    }
    assert all(string_probes.values()), string_probes

    # Once the live key is gone the subject's own record resolves to nothing,
    # which is what crypto-shredding delivers, unreadability and not absence.
    # The ticket another subject owns stays readable through that subject's
    # wrap, so it is the HR record's hit that must disappear, not the probe.
    keys.destroy(rivera.subject_id)
    assert verifier.readable(hr.doc_id, hr.text) == ""
    source_scan = next(
        p for p in verifier.verify(rivera).probes if p.probe == "source_scan"
    )
    assert hr.doc_id not in source_scan.hits, source_scan


def test_encrypting_a_chunk_does_not_change_its_vector():
    """P1's mechanism, pinned as an invariant rather than as an outcome.

    Encryption at rest leaves the derived embedding exactly where it was, so
    the neighbor probe survives crypto-shredding.
    """
    corpus = build_corpus()
    emb = default_embedder()
    index = build_index(corpus.store, emb)
    before = {c: tuple(e.vector) for c, e in index.entries.items()}

    keys = KeyStore()
    seal_store(corpus.store, index, keys, corpus.subjects)

    after = {c: tuple(e.vector) for c, e in index.entries.items()}
    assert before == after


# --------------------------------------------------------------- rotation


def test_a_rotation_from_the_live_keyring_cannot_resurrect_an_erased_subject():
    """The safe case, and the mutation check for the test below."""
    _, _, _, keys, sealed = sealed_corpus(snapshot="nightly")
    keys.destroy("subj-rivera")
    receipt, rewrapped = keys.rotate(
        "subj-rivera", sealed.for_subject("subj-rivera"), source=LIVE
    )
    assert receipt.envelopes_rewrapped == 0
    assert not receipt.resurrected
    assert not rewrapped


def test_a_rotation_from_a_snapshot_resurrects_an_erased_subject():
    """THE FINDING. Routine maintenance undoes a completed erasure.

    Same shape as the re-ingest result one layer up: no code changed, no
    operator did anything wrong, and the subject is readable again.
    """
    corpus, index, emb, keys, sealed = sealed_corpus(snapshot="nightly")
    rivera = corpus.subjects[0]
    verifier = ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)

    ErasureEngine(
        store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
    ).erase(rivera, ErasureStrategy.CRYPTO_SHRED_TEXT)
    erased = len(verifier.verify(rivera).leaking_probes)

    receipt, rewrapped = keys.rotate(
        rivera.subject_id,
        sealed.for_subject(rivera.subject_id),
        source="snapshot:nightly",
    )
    sealed.substitute(rewrapped)
    restored = len(verifier.verify(rivera).leaking_probes)

    assert receipt.resurrected
    assert receipt.envelopes_rewrapped > 0
    assert restored > erased, "the rotation put the subject back"
    assert keys.open(sealed.envelopes["acme-hr-001"], LIVE) is not None


def test_rotation_does_not_re_encrypt_the_body():
    """Envelope encryption's whole point: only the wrapped key moves."""
    _, _, _, keys, sealed = sealed_corpus()
    before = sealed.envelopes["acme-hr-001"].ciphertext
    _, rewrapped = keys.rotate("subj-rivera", sealed.for_subject("subj-rivera"))
    sealed.substitute(rewrapped)
    assert sealed.envelopes["acme-hr-001"].ciphertext == before


# ---------------------------------------------------------------- strategy


def test_crypto_shredding_removes_nothing():
    """The certificate's first five fields are empty and that is correct."""
    corpus, index, emb, keys, sealed = sealed_corpus()
    cert = ErasureEngine(
        store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
    ).erase(corpus.subjects[0], ErasureStrategy.CRYPTO_SHRED_TEXT)

    assert cert.documents_removed == ()
    assert cert.chunks_removed == ()
    assert cert.vectors_removed == ()
    assert cert.keys_destroyed == (LIVE,)
    assert len(corpus.store.documents) == 8, "every byte is still there"


def test_an_envelope_is_orphaned_only_when_no_source_can_open_it():
    """A key surviving in a backup is not erasure, and is not counted as it."""
    _, index, emb, keys, sealed = sealed_corpus(snapshot="nightly")
    corpus = build_corpus()
    engine = ErasureEngine(
        store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
    )
    with_backup = engine.erase(corpus.subjects[0], ErasureStrategy.CRYPTO_SHRED_TEXT)
    assert with_backup.envelopes_orphaned == ()

    keys.purge_snapshots()
    after_purge = engine.erase(corpus.subjects[0], ErasureStrategy.CRYPTO_SHRED_TEXT)
    assert after_purge.envelopes_orphaned, "now nothing can open them"


def test_crypto_shredding_refuses_to_run_on_a_plaintext_store():
    corpus = build_corpus()
    emb = default_embedder()
    index = build_index(corpus.store, emb)
    engine = ErasureEngine(store=corpus.store, index=index, embedder=emb)
    with pytest.raises(ValueError, match="encrypted at rest"):
        engine.erase(corpus.subjects[0], ErasureStrategy.CRYPTO_SHRED_TEXT)


# --------------------------------------------------- variant B, sealed vectors


def test_sealing_a_vector_removes_the_plaintext_copy():
    _, index, _, _, _ = sealed_corpus(seal_vectors=True)
    entry = index.get("acme-hr-001#0")
    assert entry.sealed
    assert entry.vector == ()


def test_variant_b_closes_the_neighbor_path_for_subject_only_vectors():
    """THE CORRECTED P2. I predicted the neighbor probe would go clean; it
    does not, and the reason is the same multi-subject wrap that limits
    everything else.

    Sealing the vector removes it from search for artifacts wrapped for the
    erased subject ALONE. A vector derived from a chunk that concerns two
    people is still openable with the other person's key, so it stays
    searchable and stays near the erased subject's identifiers.
    """
    corpus, index, emb, keys, sealed = sealed_corpus(seal_vectors=True)
    rivera = corpus.subjects[0]
    ErasureEngine(
        store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
    ).erase(rivera, ErasureStrategy.CRYPTO_SHRED_FULL)

    probe = next(
        p
        for p in ErasureVerifier(
            corpus.store, index, emb, keys=keys, sealed=sealed
        ).verify(rivera).probes
        if p.probe == "neighbor_probe"
    )
    # Rivera's own vectors are gone from the ranking entirely.
    assert "acme-hr-001#0" not in probe.hits
    assert "acme-benefits-002#0" not in probe.hits
    # The shared ones remain, and every survivor is wrapped for someone else.
    assert probe.leaked
    for chunk_id in probe.hits:
        wraps = sealed.envelopes[f"vector:{chunk_id}"].concerns()
        assert len(wraps) > 1, f"{chunk_id} survived without a co-subject"


def test_an_unopenable_vector_is_skipped_by_search_not_scored_zero():
    corpus, index, emb, keys, sealed = sealed_corpus(seal_vectors=True)
    keys.destroy("subj-rivera")
    hits = index.search(emb.embed("employee record"), k=20, min_score=-1.0)
    returned = {h.chunk_id for h in hits}
    assert "acme-hr-001#0" not in returned
    assert index.decrypt_failures > 0


def test_a_sealed_index_decrypts_every_candidate_on_every_query():
    """Variant B's cost, counted rather than timed.

    Eleven chunks cannot demonstrate a complexity difference in wall-clock, so
    the honest figure is the number of opens: one per candidate, per query,
    forever. That is why an encrypted index cannot carry an ANN structure: the
    comparison it would need to build one is the thing that requires a key.
    """
    corpus, index, emb, keys, sealed = sealed_corpus(seal_vectors=True)
    index.reset_counters()
    index.search(emb.embed("payroll access"), k=5, tenant_id="acme")

    # One open per SEALED candidate. The corpus also holds a chunk concerning
    # no subject at all (the retention policy note), which is never encrypted
    # and therefore costs nothing, so the count is the sealed candidates
    # rather than every candidate, and a mixed index is the realistic case.
    sealed_acme = sum(
        1 for e in index.entries.values() if e.tenant_id == "acme" and e.sealed
    )
    plain_acme = sum(
        1 for e in index.entries.values() if e.tenant_id == "acme" and not e.sealed
    )
    assert plain_acme >= 1, "the corpus should contain an unsealed chunk"
    assert index.decrypt_attempts == sealed_acme

    plain_corpus = build_corpus()
    plain_index = build_index(plain_corpus.store, emb)
    plain_index.search(emb.embed("payroll access"), k=5, tenant_id="acme")
    assert plain_index.decrypt_attempts == 0


# ------------------------------------------------------------- blind sweep


def test_a_string_matching_sweep_goes_blind_on_an_encrypted_store():
    """Encryption at rest disables every string-matching governance tool.

    The same `full` strategy, on the same sealed corpus, with and without keys.
    Blind, it finds only what it can reach by foreign key, the subject's own
    documents, misses every mention inside another party's record, and reports
    a certificate an operator would read as success.
    """
    corpus, index, emb, keys, sealed = sealed_corpus()
    rivera = corpus.subjects[0]
    blind = ErasureEngine(
        store=corpus.store, index=index, embedder=emb
    ).erase(rivera, ErasureStrategy.FULL)
    blind_leaks = len(
        ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)
        .verify(rivera)
        .leaking_probes
    )

    corpus2, index2, emb2, keys2, sealed2 = sealed_corpus()
    aware = ErasureEngine(
        store=corpus2.store, index=index2, embedder=emb2, keys=keys2, sealed=sealed2
    ).erase(corpus2.subjects[0], ErasureStrategy.FULL)
    aware_leaks = len(
        ErasureVerifier(corpus2.store, index2, emb2, keys=keys2, sealed=sealed2)
        .verify(corpus2.subjects[0])
        .leaking_probes
    )

    assert aware.total_artifacts() > blind.total_artifacts()
    assert blind_leaks > aware_leaks
    # The key is untouched by either, so the content sweep never fully erases.
    assert aware_leaks >= 1


# ----------------------------------------------------------- key tombstone


def test_a_destroyed_key_cannot_be_re_issued():
    _, _, _, keys, _ = sealed_corpus()
    keys.destroy("subj-rivera")
    assert keys.ensure("subj-rivera") is None
    with pytest.raises(Exception, match="erased"):
        keys.create("subj-rivera")


def test_sealing_omits_an_erased_subjects_wrap_and_records_it():
    """A destroyed key is never silently re-issued.

    A re-ingest re-seals each chunk for "whoever it mentions". Without the
    tombstone that path would call ensure(), be handed a fresh live key for an
    erased subject, and un-erase them. `refused_wraps` records the omission
    because a pipeline processing an erased subject's data is a compliance
    event rather than a silent skip.
    """
    _, _, _, keys, _ = sealed_corpus()
    keys.destroy("subj-rivera")
    envelope = keys.seal("chunk", b"a note mentioning Jordan Rivera",
                         ["subj-rivera", "subj-raman"])
    assert envelope.concerns() == ("subj-raman",)
    assert ("chunk", "subj-rivera") in keys.refused_wraps


def test_rotation_can_override_the_tombstone_because_that_is_the_finding():
    _, _, _, keys, sealed = sealed_corpus(snapshot="nightly")
    keys.destroy("subj-rivera")
    receipt, _ = keys.rotate(
        "subj-rivera", sealed.for_subject("subj-rivera"),
        source="snapshot:nightly",
    )
    assert receipt.resurrected
    assert "subj-rivera" not in keys.destroyed


# --------------------------------------------------------- key-aware reingest


def test_a_key_aware_reingest_skips_what_it_cannot_decrypt():
    """You cannot re-ingest what you cannot read, which is exactly why
    crypto-shredding beats a content sweep on this axis."""
    from boundary.crypto import rechunk_and_reindex_sealed

    corpus, index, emb, keys, sealed = sealed_corpus()
    rivera = corpus.subjects[0]
    ErasureEngine(
        store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
    ).erase(rivera, ErasureStrategy.CRYPTO_SHRED_TEXT)
    verifier = ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)
    before = len(verifier.verify(rivera).leaking_probes)

    done, skipped = rechunk_and_reindex_sealed(
        corpus.store, index, emb, keys, sealed, corpus.subjects
    )
    after = len(verifier.verify(rivera).leaking_probes)

    assert skipped == 2, "Rivera's own two documents are undecryptable"
    assert done == 6
    assert after == before, "a re-ingest must not resurrect an erased subject"


def test_the_key_aware_reingest_is_deterministic():
    """The key-BLIND version was not: it re-chunked ciphertext, and a random
    IV meant different chunk boundaries on every run."""
    from boundary.crypto import rechunk_and_reindex_sealed

    counts = []
    for _ in range(2):
        corpus, index, emb, keys, sealed = sealed_corpus()
        ErasureEngine(
            store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
        ).erase(corpus.subjects[0], ErasureStrategy.CRYPTO_SHRED_TEXT)
        rechunk_and_reindex_sealed(
            corpus.store, index, emb, keys, sealed, corpus.subjects
        )
        counts.append(
            (
                len(corpus.store.chunks),
                len(
                    ErasureVerifier(corpus.store, index, emb, keys=keys, sealed=sealed)
                    .verify(corpus.subjects[0])
                    .leaking_probes
                ),
            )
        )
    assert counts[0] == counts[1]
