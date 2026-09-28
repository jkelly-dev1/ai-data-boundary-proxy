"""Crypto-shredding: erase a subject by destroying their key, not their rows.

The promise. Encrypt each subject's data under a key that is theirs alone. To
erase them, destroy the key. Every copy of the ciphertext, including copies in
backups you do not control and derived artifacts you never enumerated, becomes
inert at once. It is the standard answer to derived-data erasure and it is a
good one.

The two places it does not reach, so this module exists.

First, encrypting a Chunk does not change the vector computed from it.
Encryption at rest exists so the system keeps working: retrieval still finds
the right chunk and an authorized reader decrypts what comes back.
Re-embedding the ciphertext would destroy retrieval for every subject at once,
so no real system does it, and neither does `seal_store` unless you ask for
`seal_vectors`. The derived vector therefore survives the erasure untouched,
and `neighbor_probe` in verify.py finds it. Sealing the vector too closes that
path and costs a decrypt on every candidate of every query: the difference
between an index and a scan.

Second, a key is only destroyed when every copy is. `destroy` clears the live
keyring and nothing else, which is not a shortcut but the modeled behavior of a
real key manager: a delete removes the working key and leaves whatever the
backup policy already captured. That gap is the whole security argument, so
`KeyStore` models snapshots and escrow explicitly and `key_material_probe`
attempts a real decrypt against every one of them.

Envelope encryption is not boilerplate here. Each artifact gets a random
content key; the content key is wrapped once per subject the artifact concerns;
the body is encrypted once under the content key. That is standard practice,
and in this corpus it is also the only honest way to express a real conflict: a
support ticket owned by one subject that mentions another cannot be encrypted
under either subject's key alone. Destroying one subject's key removes one wrap
and leaves the other, so the record survives for its owner and the mentioned
subject stays inside it in plaintext. That is not a defect in the
implementation. It is what crypto-shredding does to shared records, and the
wrap map states it more precisely than prose can.

The cipher is ordinary and is not hand-rolled: Fernet from `cryptography`
(AES-128-CBC with an HMAC-SHA256 tag). A stdlib keystream would keep this
repository dependency-free and would be exactly the wrong thing to demonstrate
in a project about doing security properly.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Optional

from cryptography.fernet import Fernet, InvalidToken

from boundary.index import VectorIndex
from boundary.models import Chunk, Document, Subject
from boundary.store import DataStore

# The name of the working keyring, as reported by probes and receipts. Real
# key managers distinguish the live key from its backups; so does this.
LIVE = "live"


class KeyDestroyedError(RuntimeError):
    """Raised when something tries to re-issue a key that was destroyed."""


@dataclass(frozen=True)
class Envelope:
    """Ciphertext, plus one wrapped content key per subject it concerns.

    `wraps` is the interesting field. An artifact concerning two subjects has
    two wraps of the SAME content key, so either subject's key opens it and
    destroying one leaves the other working.
    """

    ciphertext: bytes
    wraps: Mapping[str, bytes]
    kind: str

    def concerns(self) -> tuple[str, ...]:
        return tuple(sorted(self.wraps))

    def as_text(self) -> str:
        """The stored form: a Fernet token is URL-safe base64, so it is ASCII.

        Real systems keep an encrypted blob in the column the plaintext used to
        occupy, and so does `seal_store`. It matters that this contains no
        identifier substring, because that is what makes a string scan of the
        store come back empty, so a probe has to ask whether the artifact can
        be OPENED rather than whether the stored bytes match.
        """
        return self.ciphertext.decode("ascii")


@dataclass(frozen=True)
class DestroyReceipt:
    """What a destroy actually accomplished, which is rarely everything."""

    subject_id: str
    removed_from: tuple[str, ...] = ()
    surviving_sources: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.surviving_sources


@dataclass(frozen=True)
class RotationReceipt:
    """What a key rotation did, and whether it brought anyone back.

    `resurrected` is the finding this type exists for: true when the subject
    had NO live key before the rotation and has one after. A rotation job that
    sources its old key material from a backup will do that to an erased
    subject without anyone asking it to.
    """

    subject_id: str
    source: str
    envelopes_rewrapped: int = 0
    resurrected: bool = False


class KeyStore:
    """Per-subject key encryption keys, and every place copies of them live.

    Three sources, because three is what a real program has and because the
    difference between them is the entire finding:

        live       the working keyring, what the running system reads
        snapshots  point-in-time backups, named, taken by policy
        escrow     break-glass copies, usually in another account entirely
    """

    def __init__(self) -> None:
        self.live: dict[str, bytes] = {}
        self.snapshots: dict[str, dict[str, bytes]] = {}
        self.escrow: dict[str, bytes] = {}
        # Subjects whose key has been destroyed. THE INVARIANT: a destroyed
        # key id is never silently re-issued. Any path that seals new data for
        # a subject must consult this first, or it hands out a fresh live key
        # to someone who was erased and un-erases them. Real key managers keep
        # the same tombstone for the same reason.
        self.destroyed: set[str] = set()
        # (kind, subject_id) for every wrap omitted because the subject was
        # erased. Each entry is a compliance event: the pipeline just processed
        # data concerning someone who exercised their right to erasure.
        self.refused_wraps: list[tuple[str, str]] = []

    # ------------------------------------------------------------- lifecycle

    def create(self, subject_id: str, force: bool = False) -> bytes:
        """Issue (or re-issue) the live key encryption key for a subject.

        `Force` exists so rotation can override the tombstone, because a
        rotation job sourcing old material from a backup overriding an erasure
        is the finding rather than a case to prevent. Every other caller gets
        the refusal.
        """
        if subject_id in self.destroyed and not force:
            raise KeyDestroyedError(
                f"{subject_id} has been erased; a destroyed key must not be "
                f"re-issued. Seal for the remaining subjects instead."
            )
        key = Fernet.generate_key()
        self.live[subject_id] = key
        self.destroyed.discard(subject_id)
        return key

    def ensure(self, subject_id: str) -> Optional[bytes]:
        """The live key, or None when the subject has been erased.

        Returns None instead of raising so `seal` can omit the wrap and carry
        on, which is what a real pipeline does: it seals the record for the
        people it still has keys for and notices nothing.
        """
        if subject_id in self.destroyed:
            return None
        return self.live.get(subject_id) or self.create(subject_id)

    def snapshot(self, name: str) -> None:
        """Model a key-manager backup: a full copy of the live keyring."""
        self.snapshots[name] = dict(self.live)

    def escrow_key(self, subject_id: str) -> None:
        """Model a break-glass copy held outside the live keyring."""
        if subject_id in self.live:
            self.escrow[subject_id] = self.live[subject_id]

    def purge_snapshots(self) -> tuple[str, ...]:
        """Destroy every backup. The step real programs forget."""
        names = tuple(sorted(self.snapshots))
        self.snapshots.clear()
        return names

    def destroy(self, subject_id: str) -> DestroyReceipt:
        """Delete the LIVE key and nothing else.

        This is not a thorough delete. A `DELETE` against a key manager removes
        the working key; it does not reach into last night's snapshot or into
        the escrow account. Modeling the narrow version is what gives
        `key_material_probe` something true to find.
        """
        removed = []
        if self.live.pop(subject_id, None) is not None:
            removed.append(LIVE)
        self.destroyed.add(subject_id)
        return DestroyReceipt(
            subject_id=subject_id,
            removed_from=tuple(removed),
            surviving_sources=self.sources_holding(subject_id),
        )

    # ---------------------------------------------------------------- lookup

    def sources(self) -> tuple[tuple[str, Mapping[str, bytes]], ...]:
        """Every place key material lives, named."""
        out: list[tuple[str, Mapping[str, bytes]]] = [(LIVE, self.live)]
        for name in sorted(self.snapshots):
            out.append((f"snapshot:{name}", self.snapshots[name]))
        if self.escrow:
            out.append(("escrow", self.escrow))
        return tuple(out)

    def sources_holding(self, subject_id: str) -> tuple[str, ...]:
        """Which named sources still hold this subject's key.

        Reporting only. The verifier must not call this: asking the key store
        whether it thinks the key is gone is the verifier reading the eraser's
        own bookkeeping, which verify.py forbids. The probe attempts a real
        decrypt instead.
        """
        return tuple(
            name for name, mapping in self.sources() if subject_id in mapping
        )

    def get(self, subject_id: str, source: str = LIVE) -> Optional[bytes]:
        for name, mapping in self.sources():
            if name == source:
                return mapping.get(subject_id)
        return None

    # ------------------------------------------------------------ seal/open

    def seal(
        self, kind: str, plaintext: bytes, subject_ids: Iterable[str]
    ) -> Envelope:
        """Encrypt a body once and wrap its content key per subject."""
        cek = Fernet.generate_key()
        ciphertext = Fernet(cek).encrypt(plaintext)
        wraps: dict[str, bytes] = {}
        for subject_id in sorted(set(subject_ids)):
            kek = self.ensure(subject_id)
            if kek is None:
                self.refused_wraps.append((kind, subject_id))
                continue
            wraps[subject_id] = Fernet(kek).encrypt(cek)
        if not wraps:
            raise KeyDestroyedError(
                f"sealing a {kind} for {sorted(set(subject_ids))} would "
                f"produce an envelope nobody can open"
            )
        return Envelope(ciphertext=ciphertext, wraps=wraps, kind=kind)

    def open(
        self, envelope: Envelope, source: str = LIVE
    ) -> Optional[bytes]:
        """Open an envelope using key material from ONE named source.

        Returns None when no wrap can be unwrapped from that source, which is
        the state a successful erasure produces. Callers treat None as "not
        readable" rather than as an error, because an unreadable artifact is
        the goal and not a failure.
        """
        for subject_id, wrapped in envelope.wraps.items():
            kek = self.get(subject_id, source)
            if kek is None:
                continue
            try:
                cek = Fernet(kek).decrypt(wrapped)
                return Fernet(cek).decrypt(envelope.ciphertext)
            except InvalidToken:
                continue
        return None

    def can_open_wrap(
        self, subject_id: str, envelope: Envelope, source: str
    ) -> bool:
        """Can THIS subject's wrap be unwrapped from THIS source?

        Deliberately narrower than `open`. An envelope concerning two subjects
        opens for either of them, so asking "can anything open this" would
        report a leak for Rivera whenever Raman's key happens to work, which
        is true of the record and false of the key. The tenth probe needs the
        narrow question, and it needs it answered by attempting the decrypt
        rather than by consulting a lookup table.
        """
        wrapped = envelope.wraps.get(subject_id)
        kek = self.get(subject_id, source)
        if wrapped is None or kek is None:
            return False
        try:
            Fernet(kek).decrypt(wrapped)
            return True
        except InvalidToken:
            return False

    def wrap_sources(self, subject_id: str, envelope: Envelope) -> tuple[str, ...]:
        """Every named source whose key material opens this subject's wrap."""
        return tuple(
            name
            for name, _ in self.sources()
            if self.can_open_wrap(subject_id, envelope, name)
        )

    # -------------------------------------------------------------- rotation

    def rotate(
        self,
        subject_id: str,
        envelopes: Iterable[Envelope],
        source: str = LIVE,
    ) -> tuple[RotationReceipt, dict[int, Envelope]]:
        """Re-wrap a subject's content keys under a fresh key encryption key.

        No body is re-encrypted. That is the point of envelope encryption: the
        content key is unwrapped with the old key and re-wrapped with the new
        one, and the ciphertext never moves.

        `Source` is the whole finding. A rotation job has to get the OLD key
        material from somewhere. Point it at the live keyring and an erased
        subject stays erased, because there is nothing to unwrap with. Point it
        at a snapshot, which is what a disaster-recovery rebuild does by
        definition, and the job cheerfully unwraps an erased subject's content
        keys and re-wraps them under a fresh, LIVE key. The subject is readable
        again on the running system, no code changed, nobody did anything
        wrong.

        Returns the receipt and a map of {id(old_envelope): new_envelope}, so
        the caller can substitute them. Envelopes are frozen, so rotation
        produces new ones instead of mutating in place.
        """
        old_kek = self.get(subject_id, source)
        was_live = subject_id in self.live
        if old_kek is None:
            return (
                RotationReceipt(subject_id=subject_id, source=source),
                {},
            )

        new_kek = self.create(subject_id, force=True)
        rewrapped: dict[int, Envelope] = {}
        for envelope in envelopes:
            wrapped = envelope.wraps.get(subject_id)
            if wrapped is None:
                continue
            try:
                cek = Fernet(old_kek).decrypt(wrapped)
            except InvalidToken:
                continue
            wraps = dict(envelope.wraps)
            wraps[subject_id] = Fernet(new_kek).encrypt(cek)
            rewrapped[id(envelope)] = replace(envelope, wraps=wraps)

        return (
            RotationReceipt(
                subject_id=subject_id,
                source=source,
                envelopes_rewrapped=len(rewrapped),
                resurrected=not was_live,
            ),
            rewrapped,
        )


# --------------------------------------------------------------- vectors


def pack_vector(vector: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 8}d", blob)


# ------------------------------------------------------------- sealing


@dataclass
class SealedCorpus:
    """A plaintext store converted to ciphertext, plus the envelopes to open it.

    The envelopes are held BESIDE the store rather than inside it, the way a
    real system keeps wrapped keys in a key manager and blobs in a database.
    Artifact ids are the store's own ids, with vectors under a `vector:` prefix
    so a chunk's text and its embedding are separately addressable.
    """

    envelopes: dict[str, Envelope] = field(default_factory=dict)
    sealed_vectors: bool = False

    def for_subject(self, subject_id: str) -> tuple[Envelope, ...]:
        return tuple(
            env
            for _, env in sorted(self.envelopes.items())
            if subject_id in env.wraps
        )

    def substitute(self, rewrapped: Mapping[int, Envelope]) -> int:
        """Apply a rotation's output back into the map."""
        count = 0
        for artifact_id, envelope in list(self.envelopes.items()):
            new = rewrapped.get(id(envelope))
            if new is not None:
                self.envelopes[artifact_id] = new
                count += 1
        return count


def _concerned(text: str, subjects: Iterable[Subject], owner: Optional[str]) -> set[str]:
    """Which subjects an artifact is about: everyone named, plus the owner.

    The owner is included even when unnamed, because a record can be about a
    person without repeating their name in every chunk of it.
    """
    ids = {s.subject_id for s in subjects if s.matches(text)}
    if owner:
        ids.add(owner)
    return ids


def seal_store(
    store: DataStore,
    index: VectorIndex,
    keys: KeyStore,
    subjects: Iterable[Subject],
    seal_vectors: bool = False,
) -> SealedCorpus:
    """Convert a plaintext store to encryption at rest, in place.

    Every document, chunk, index payload, cache entry and prompt log entry is
    replaced by its ciphertext. `seal_vectors` additionally seals the embedding
    itself, which is variant B and is what closes the neighbor path at the cost
    of a decrypt per candidate per query.

    Note what is not re-computed: vectors. Under variant A the embedding stays
    exactly as it was computed from the plaintext, because that is what
    encryption at rest means. See this module's docstring.
    """
    subjects = tuple(subjects)
    sealed = SealedCorpus(sealed_vectors=seal_vectors)
    if seal_vectors:
        index.opener = lambda env: vector_opener(keys, env)

    for doc_id, doc in list(store.documents.items()):
        ids = _concerned(doc.text, subjects, doc.owner_subject_id)
        if not ids:
            continue
        env = keys.seal("document", doc.text.encode("utf-8"), ids)
        sealed.envelopes[doc_id] = env
        store.documents[doc_id] = replace(doc, text=env.as_text())

    for chunk_id, chunk in list(store.chunks.items()):
        owner = store.documents.get(chunk.doc_id)
        ids = _concerned(
            chunk.text, subjects, owner.owner_subject_id if owner else None
        )
        if not ids:
            continue
        env = keys.seal("chunk", chunk.text.encode("utf-8"), ids)
        sealed.envelopes[chunk_id] = env
        store.chunks[chunk_id] = replace(chunk, text=env.as_text())

        entry = index.get(chunk_id)
        if entry is not None:
            entry.text = env.as_text()
            if seal_vectors:
                venv = keys.seal("vector", pack_vector(entry.vector), ids)
                sealed.envelopes[f"vector:{chunk_id}"] = venv
                entry.sealed_vector = venv
                entry.vector = ()

    for key, cache in list(store.response_cache.items()):
        ids = _concerned(cache.text, subjects, None)
        if not ids:
            continue
        env = keys.seal("cache", cache.text.encode("utf-8"), ids)
        sealed.envelopes[key] = env
        cache.text = env.as_text()

    for entry_id, prompt in list(store.prompt_log.items()):
        ids = _concerned(prompt.text, subjects, None)
        if not ids:
            continue
        env = keys.seal("prompt", prompt.text.encode("utf-8"), ids)
        sealed.envelopes[entry_id] = env
        prompt.text = env.as_text()

    return sealed


def vector_opener(keys: KeyStore, envelope: object) -> Optional[tuple[float, ...]]:
    """Open a sealed vector with LIVE key material, or return None.

    Wired into `VectorIndex.opener` by `seal_store` when sealing vectors. LIVE
    only: an index that silently reached into a key backup to answer a query
    would be a worse bug than the one this project measures.
    """
    blob = keys.open(envelope, source=LIVE)  # type: ignore[arg-type]
    return unpack_vector(blob) if blob is not None else None


def rechunk_and_reindex_sealed(
    store: DataStore,
    index: VectorIndex,
    embedder,
    keys: KeyStore,
    sealed: SealedCorpus,
    subjects: Iterable[Subject],
) -> tuple[int, int]:
    """Re-ingest a SEALED corpus: decrypt, re-chunk, re-seal, re-embed.

    USE THIS AND NOT `erasure.rechunk_and_reindex` on a sealed store. That one
    re-chunks whatever sits in `Document.text`, which here is a base64 token,
    so it would slice ciphertext into fragments and embed those. Corruption
    that resembles a re-ingest, and non-deterministic besides, since a random
    IV puts the chunk boundaries somewhere new on every run.

    The real operation decrypts each document, re-chunks the PLAINTEXT,
    re-seals each chunk for the subjects it concerns, and re-embeds from the
    plaintext.

    A Document that cannot be decrypted is skipped, and that is the whole
    reason crypto-shredding beats a content sweep on this axis: you cannot
    re-ingest what you cannot read, so an erased subject stays erased across a
    re-chunk instead of being restored by one. Returns
    (documents_reingested, documents_skipped).
    """
    from boundary.models import IndexEntry
    from boundary.store import chunk_document

    subjects = tuple(subjects)
    store.chunks.clear()
    index.entries.clear()
    for chunk_id in [k for k in sealed.envelopes if not k.startswith("vector:")]:
        if chunk_id not in store.documents and chunk_id not in store.response_cache:
            sealed.envelopes.pop(chunk_id, None)

    reingested = 0
    skipped = 0
    for doc in list(store.documents.values()):
        envelope = sealed.envelopes.get(doc.doc_id)
        if envelope is None:
            plaintext = doc.text
        else:
            opened = keys.open(envelope, source=LIVE)
            if opened is None:
                skipped += 1
                continue
            plaintext = opened.decode("utf-8")
        reingested += 1
        for chunk in chunk_document(replace(doc, text=plaintext)):
            ids = _concerned(chunk.text, subjects, doc.owner_subject_id)
            if ids:
                env = keys.seal("chunk", chunk.text.encode("utf-8"), ids)
                sealed.envelopes[chunk.chunk_id] = env
                store.add_chunk(replace(chunk, text=env.as_text()))
            else:
                store.add_chunk(chunk)
            entry = IndexEntry(
                chunk_id=chunk.chunk_id,
                tenant_id=chunk.tenant_id,
                vector=embedder.embed(chunk.text),
                text=store.chunks[chunk.chunk_id].text,
            )
            if sealed.sealed_vectors and ids:
                venv = keys.seal("vector", pack_vector(entry.vector), ids)
                sealed.envelopes[f"vector:{chunk.chunk_id}"] = venv
                entry.sealed_vector = venv
                entry.vector = ()
            index.upsert(entry)
    return reingested, skipped
