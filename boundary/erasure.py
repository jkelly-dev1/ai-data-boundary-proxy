"""Right-to-erasure, implemented at four levels of thoroughness.

The four strategies exist so the demo can MEASURE the difference between them
rather than assert it. Each one is a real implementation that a team could
plausibly ship, and the first three all leak:

  SOURCE_ONLY      delete the subject's own documents. What "delete the user"
                   usually means in a ticket. Leaves chunks, vectors, caches
                   and logs untouched, and never looks at documents owned by
                   someone else that mention the subject.

  SOURCE_AND_INDEX delete the documents, their chunks, and TOMBSTONE the
                   vectors. Better, and still leaks: tombstoned vectors remain
                   reachable through a query path that ignores the flag, the
                   caches still hold quoted text, and mentions inside other
                   parties' documents are untouched.

  MENTION_SWEEP    additionally find every chunk that mentions the subject
                   ANYWHERE, including in other tenants' documents, and either
                   remove or rewrite it. Purges caches and prompt logs. Closes
                   the string-reachable paths.

  FULL             everything in MENTION_SWEEP, plus purge (not tombstone) the
                   vectors and compact the index, so no derived vector for a
                   removed chunk survives.

A rewrite-vs-delete decision is built in. When a chunk mentioning the subject
belongs to a document the subject does not own, deleting it destroys another
party's record, which is its own compliance problem. The sweep rewrites those
chunks with the subject redacted and RE-EMBEDS them, because the old vector was
computed from text that named the subject and is itself the disclosure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from boundary.detectors import DetectorSuite
from boundary.embeddings import Embedder
from boundary.index import VectorIndex
from boundary.models import (
    Chunk,
    Document,
    ErasureCertificate,
    IndexEntry,
    Subject,
)
from boundary.store import DataStore

# The live keyring's name. A plain string; importing it pulls in no cipher.
_LIVE = "live"


class ErasureStrategy(str, Enum):
    SOURCE_ONLY = "source_only"
    SOURCE_AND_INDEX = "source_and_index"
    MENTION_SWEEP = "mention_sweep"
    FULL = "full"
    CRYPTO_SHRED_TEXT = "crypto_shred_text"
    CRYPTO_SHRED_FULL = "crypto_shred_full"


# Two postures, and the distinction is not cosmetic. The first four strategies
# operate on a PLAINTEXT store and erase by removing rows. The crypto
# strategies operate on a store that is already encrypted at rest and erase by
# destroying a key, so they require a KeyStore and a SealedCorpus and cannot be
# run against a plaintext corpus at all. Callers that sweep "every strategy"
# iterate one of these tuples rather than the enum, because the two postures
# need different fixtures and are not interchangeable.
PLAINTEXT_STRATEGIES = (
    ErasureStrategy.SOURCE_ONLY,
    ErasureStrategy.SOURCE_AND_INDEX,
    ErasureStrategy.MENTION_SWEEP,
    ErasureStrategy.FULL,
)
CRYPTO_STRATEGIES = (
    ErasureStrategy.CRYPTO_SHRED_TEXT,
    ErasureStrategy.CRYPTO_SHRED_FULL,
)


@dataclass
class ErasureEngine:
    """Executes an erasure and returns a certificate of what it did."""

    store: DataStore
    index: VectorIndex
    embedder: Embedder
    detectors: Optional[DetectorSuite] = None
    redact_mentions: bool = True
    # Supplied only in the sealed posture. See CRYPTO_STRATEGIES.
    keys: Optional[object] = None
    sealed: Optional[object] = None

    # ------------------------------------------------------------------ api

    def erase(
        self, subject: Subject, strategy: ErasureStrategy
    ) -> ErasureCertificate:
        if strategy is ErasureStrategy.SOURCE_ONLY:
            return self._source_only(subject)
        if strategy is ErasureStrategy.SOURCE_AND_INDEX:
            return self._source_and_index(subject, purge=False)
        if strategy is ErasureStrategy.MENTION_SWEEP:
            return self._mention_sweep(
                subject, purge=False, compact=False, sweep_documents=False
            )
        if strategy in (
            ErasureStrategy.CRYPTO_SHRED_TEXT,
            ErasureStrategy.CRYPTO_SHRED_FULL,
        ):
            return self._crypto_shred(subject, strategy)
        return self._mention_sweep(
            subject, purge=True, compact=True, sweep_documents=True
        )

    # ------------------------------------------------------------ strategies

    def _source_only(self, subject: Subject) -> ErasureCertificate:
        """Delete the subject's own documents. Nothing else."""
        removed = []
        for doc in self.store.documents_owned_by(subject):
            if self.store.remove_document(doc.doc_id):
                removed.append(doc.doc_id)
        return ErasureCertificate(
            subject_id=subject.subject_id,
            tenant_id=subject.tenant_id,
            documents_removed=tuple(sorted(removed)),
            strategy=ErasureStrategy.SOURCE_ONLY.value,
        )

    def _source_and_index(self, subject: Subject, purge: bool) -> ErasureCertificate:
        """Documents, their chunks, and their vectors.

        Note what is NOT here: any document owned by another subject, the
        response cache, and the prompt log. This is the implementation that
        looks complete in a code review because it touches three stores.
        """
        docs_removed: list[str] = []
        chunks_removed: list[str] = []
        vectors_removed: list[str] = []

        for doc in self.store.documents_owned_by(subject):
            for chunk in self.store.chunks_for_document(doc.doc_id):
                if self.store.remove_chunk(chunk.chunk_id):
                    chunks_removed.append(chunk.chunk_id)
                if purge:
                    if self.index.purge(chunk.chunk_id):
                        vectors_removed.append(chunk.chunk_id)
                elif self.index.delete(chunk.chunk_id):
                    vectors_removed.append(chunk.chunk_id)
            if self.store.remove_document(doc.doc_id):
                docs_removed.append(doc.doc_id)

        return ErasureCertificate(
            subject_id=subject.subject_id,
            tenant_id=subject.tenant_id,
            documents_removed=tuple(sorted(docs_removed)),
            chunks_removed=tuple(sorted(chunks_removed)),
            vectors_removed=tuple(sorted(vectors_removed)),
            strategy=ErasureStrategy.SOURCE_AND_INDEX.value,
        )

    def _mention_sweep(
        self, subject: Subject, purge: bool, compact: bool, sweep_documents: bool
    ) -> ErasureCertificate:
        """Follow the subject everywhere they are string-reachable.

        `sweep_documents` is the difference between MENTION_SWEEP and FULL.
        Cleaning the chunks, vectors, caches and logs while leaving the SOURCE
        documents intact produces a system that passes every retrieval probe
        and silently restores the subject on the next re-ingest, because
        re-chunking reads the documents. See `rechunk_and_reindex`.
        """
        base = self._source_and_index(subject, purge=purge)

        chunks_removed = list(base.chunks_removed)
        vectors_removed = list(base.vectors_removed)
        rewritten: list[str] = []
        caches: list[str] = []

        # Chunks anywhere that still mention the subject, including ones
        # belonging to documents owned by other people.
        for chunk in self._mentioning_chunks(subject):
            owner = self.store.documents.get(chunk.doc_id)
            subject_owns = owner is not None and owner.owner_subject_id == subject.subject_id
            if subject_owns or not self.redact_mentions:
                if self.store.remove_chunk(chunk.chunk_id):
                    chunks_removed.append(chunk.chunk_id)
                if purge:
                    if self.index.purge(chunk.chunk_id):
                        vectors_removed.append(chunk.chunk_id)
                elif self.index.delete(chunk.chunk_id):
                    vectors_removed.append(chunk.chunk_id)
            else:
                if self._rewrite_and_reembed(chunk, subject):
                    rewritten.append(chunk.chunk_id)

        # Caches and prompt logs: copies of the text that no chunk owns.
        for entry in self._mentioning_caches(subject):
            if self.store.remove_cache_entry(entry.key):
                caches.append(entry.key)
        for entry in self._mentioning_prompts(subject):
            if self.store.remove_prompt_entry(entry.entry_id):
                caches.append(entry.entry_id)

        # Source documents owned by OTHER subjects that still name this one.
        # Rewritten rather than deleted: the document is another party's
        # record, and destroying it to satisfy this subject's erasure request
        # trades one compliance failure for another.
        if sweep_documents:
            for doc in self._mentioning_documents(subject):
                plaintext = self._readable(doc.doc_id, doc.text)
                redacted = _strip_identifiers(plaintext, subject)
                if redacted == plaintext:
                    continue
                stored = redacted
                if self.sealed is not None and self.keys is not None:
                    envelope = self.sealed.envelopes.get(doc.doc_id)
                    if envelope is not None:
                        remaining = [
                            sid
                            for sid in envelope.wraps
                            if sid != subject.subject_id
                        ]
                        if remaining:
                            resealed = self.keys.seal(
                                "document", redacted.encode("utf-8"), remaining
                            )
                            self.sealed.envelopes[doc.doc_id] = resealed
                            stored = resealed.as_text()
                        else:
                            self.sealed.envelopes.pop(doc.doc_id, None)
                self.store.documents[doc.doc_id] = Document(
                    doc_id=doc.doc_id,
                    tenant_id=doc.tenant_id,
                    text=stored,
                    sensitivity=doc.sensitivity,
                    owner_subject_id=doc.owner_subject_id,
                )
                rewritten.append(doc.doc_id)

        if compact:
            self.index.compact()

        return ErasureCertificate(
            subject_id=subject.subject_id,
            tenant_id=subject.tenant_id,
            documents_removed=base.documents_removed,
            chunks_removed=tuple(sorted(set(chunks_removed))),
            vectors_removed=tuple(sorted(set(vectors_removed))),
            caches_purged=tuple(sorted(set(caches))),
            mentions_rewritten=tuple(sorted(set(rewritten))),
            strategy=(
                ErasureStrategy.FULL.value if compact else ErasureStrategy.MENTION_SWEEP.value
            ),
        )

    def _readable(self, artifact_id: str, stored: str) -> str:
        """What THIS ERASER can read out of an artifact.

        The eraser needs this for the same reason the verifier does. Every
        sweep strategy finds its targets by string-matching document and
        chunk text. On a store encrypted at rest that text is a base64 token,
        nothing matches, the sweep removes nothing, and the certificate
        truthfully reports zero artifacts, which an operator reads as "there
        was nothing to remove".

        Encryption at rest silently disables every string-matching governance
        tool: the retention sweep, the DLP scanner, the erasure job. Supplying
        `keys` and `sealed` to this engine is what makes the sweep key-aware;
        omitting them reproduces the blind version, which the demo shows side
        by side because the difference is the lesson.
        """
        if self.sealed is None or self.keys is None:
            return stored
        envelope = self.sealed.envelopes.get(artifact_id)
        if envelope is None:
            return stored
        opened = self.keys.open(envelope, source=_LIVE)
        return opened.decode("utf-8", errors="replace") if opened is not None else ""

    def _mentioning_documents(self, subject: Subject):
        return [
            d
            for d in self.store.documents.values()
            if subject.matches(self._readable(d.doc_id, d.text))
        ]

    def _mentioning_chunks(self, subject: Subject):
        return [
            c
            for c in self.store.chunks.values()
            if subject.matches(self._readable(c.chunk_id, c.text))
        ]

    def _mentioning_caches(self, subject: Subject):
        return [
            e
            for e in self.store.response_cache.values()
            if subject.matches(self._readable(e.key, e.text))
        ]

    def _mentioning_prompts(self, subject: Subject):
        return [
            e
            for e in self.store.prompt_log.values()
            if subject.matches(self._readable(e.entry_id, e.text))
        ]

    def _crypto_shred(
        self, subject: Subject, strategy: "ErasureStrategy"
    ) -> ErasureCertificate:
        """Erase by destroying the subject's key. Remove nothing.

        The certificate's first five fields come back empty and that is
        correct: no document was deleted, no chunk was rewritten, no vector was
        tombstoned. Every byte of the subject's data is exactly where it was.
        What changed is that the running system can no longer read it.

        `envelopes_orphaned` counts the artifacts the subject can no longer
        open FROM ANY SOURCE. It is deliberately not "artifacts wrapped for the
        subject": an envelope whose key survives in last night's snapshot is
        not orphaned, it is merely inconvenient, and counting it as erased is
        the exact self-flattering error this project exists to catch.
        """
        if self.keys is None or self.sealed is None:
            raise ValueError(
                "crypto shredding needs a KeyStore and a SealedCorpus; the "
                "store must already be encrypted at rest. See "
                "boundary.crypto.seal_store."
            )
        receipt = self.keys.destroy(subject.subject_id)
        orphaned = tuple(
            artifact_id
            for artifact_id, envelope in sorted(self.sealed.envelopes.items())
            if subject.subject_id in envelope.wraps
            and not self.keys.wrap_sources(subject.subject_id, envelope)
        )
        return ErasureCertificate(
            subject_id=subject.subject_id,
            tenant_id=subject.tenant_id,
            keys_destroyed=receipt.removed_from,
            envelopes_orphaned=orphaned,
            strategy=strategy.value,
        )

    # ------------------------------------------------------------- internals

    def _rewrite_and_reembed(self, chunk: Chunk, subject: Subject) -> bool:
        """Redact the subject out of a chunk and RECOMPUTE its vector.

        Recomputing is the part that is easy to skip and fatal to skip. The
        stored vector was derived from text containing the subject's name;
        leaving it in place while rewriting the text means the index still
        answers "who is near this person" correctly, and the neighbor probe in
        verify.py will find it.
        """
        plaintext = self._readable(chunk.chunk_id, chunk.text)
        redacted = _strip_identifiers(plaintext, subject)
        if redacted == plaintext:
            return False

        # On a sealed store the rewrite is decrypt, strip, RE-SEAL, and the
        # re-seal drops this subject's wrap, because a chunk that no longer
        # names them is no longer theirs to unlock. Writing the plaintext back
        # would silently un-encrypt a record in the name of erasing someone
        # from it, which is the sort of fix that creates the next finding.
        stored = redacted
        if self.sealed is not None and self.keys is not None:
            envelope = self.sealed.envelopes.get(chunk.chunk_id)
            if envelope is not None:
                remaining = [
                    sid for sid in envelope.wraps if sid != subject.subject_id
                ]
                if remaining:
                    resealed = self.keys.seal(
                        "chunk", redacted.encode("utf-8"), remaining
                    )
                    self.sealed.envelopes[chunk.chunk_id] = resealed
                    stored = resealed.as_text()
                else:
                    self.sealed.envelopes.pop(chunk.chunk_id, None)

        if not self.store.rewrite_chunk(chunk.chunk_id, stored):
            return False
        entry = self.index.get(chunk.chunk_id)
        if entry is not None:
            self.index.upsert(
                IndexEntry(
                    chunk_id=chunk.chunk_id,
                    tenant_id=entry.tenant_id,
                    vector=self.embedder.embed(redacted),
                    deleted=entry.deleted,
                    text=stored,
                )
            )
        return True


def _strip_identifiers(text: str, subject: Subject) -> str:
    """Case-insensitively remove every identifier string for the subject.

    Longest identifier first, so removing "Jordan" does not leave a fragment of
    "Jordan Rivera" behind for the next pass to miss.
    """
    out = text
    for ident in sorted(subject.identifiers, key=len, reverse=True):
        if not ident:
            continue
        lowered = out.lower()
        target = ident.lower()
        start = lowered.find(target)
        while start != -1:
            out = out[:start] + "[ERASED]" + out[start + len(ident):]
            lowered = out.lower()
            start = lowered.find(target)
    return out


def reindex(
    store: DataStore, index: VectorIndex, embedder: Embedder
) -> VectorIndex:
    """Rebuild the index from the current chunks.

    Used by the demo to show the one erasure path that always works and is
    almost never affordable: throw the derived data away and recompute it. At
    real corpus sizes this is the option finance vetoes, so the incremental
    strategies above have to be correct.
    """
    index.entries.clear()
    for chunk in store.chunks.values():
        index.upsert(
            IndexEntry(
                chunk_id=chunk.chunk_id,
                tenant_id=chunk.tenant_id,
                vector=embedder.embed(chunk.text),
                text=chunk.text,
            )
        )
    return index


def rechunk_and_reindex(
    store: DataStore, index: VectorIndex, embedder: Embedder
) -> VectorIndex:
    """Re-ingest: rebuild every chunk from the source documents, then re-embed.

    This is not an exotic operation. It is what happens on a chunking-strategy
    change, an embedding-model upgrade, a corpus migration, or a restore from
    backup, and it runs on a schedule in plenty of systems.

    It is also what undoes an incomplete erasure. If a subject was swept out of
    the chunks, vectors, caches and logs but their name is still in a source
    document owned by somebody else, this puts them back into the index, and
    probes that passed an hour ago start failing with no code change and no
    operator action. That is why FULL rewrites source documents and
    MENTION_SWEEP does not: the difference is invisible until this runs.
    """
    from boundary.store import chunk_document

    store.chunks.clear()
    index.entries.clear()
    for doc in store.documents.values():
        for chunk in chunk_document(doc):
            store.add_chunk(chunk)
    for chunk in store.chunks.values():
        index.upsert(
            IndexEntry(
                chunk_id=chunk.chunk_id,
                tenant_id=chunk.tenant_id,
                vector=embedder.embed(chunk.text),
                text=chunk.text,
            )
        )
    return index


def build_index(
    store: DataStore, embedder: Embedder, index: Optional[VectorIndex] = None
) -> VectorIndex:
    idx = index if index is not None else VectorIndex()
    for chunk in store.chunks.values():
        idx.upsert(
            IndexEntry(
                chunk_id=chunk.chunk_id,
                tenant_id=chunk.tenant_id,
                vector=embedder.embed(chunk.text),
                text=chunk.text,
            )
        )
        store.record_derivation("chunk", chunk.chunk_id, "vector", chunk.chunk_id)
    return idx
