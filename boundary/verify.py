"""Independent probes that a subject is really gone.

The design rule here: verification must not reuse the erasure engine's idea of
where the subject lives. If the verifier walked the same lineage table the
eraser walked, it would confirm the eraser's blind spots rather than find them.
Every probe below reaches for the subject a DIFFERENT way, and several of them
deliberately bypass the application's own filters, because "the app no longer
shows it" and "it is no longer there" are different claims and only the second
one is erasure.

The probes, and the distinct reachability path each one covers:

    source_scan        the row itself
    chunk_scan         derived text, including chunks in other people's docs
    index_text_scan    the copy of the text stored beside the vector
    retrieval_probe    what the application's own search returns
    tombstone_probe    what a query path that ignores delete flags returns
    neighbor_probe     the EMBEDDING, which no string search can find
    cache_probe        answers that quoted the subject
    prompt_log_probe   retained prompts and completions
    cross_tenant_probe reachability from a neighboring tenant's scope

`neighbor_probe` is the one that matters most and the one a real system almost
never runs. An embedding cannot be grepped. If the vector derived from text
naming a subject is still in the index, the subject is still in the index, and
the only way to notice is to embed their identifiers and look at what comes
back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# LIVE is a plain string constant; importing it does not pull in the cipher.
from boundary.crypto import LIVE
from boundary.embeddings import Embedder
from boundary.index import VectorIndex
from boundary.models import LeakProbe, Subject, VerificationReport
from boundary.store import DataStore

# Similarity at or above which a neighbor is treated as "about" the subject.
#
# Similarity at or above which a neighbor is treated as "about" the subject.
#
# The invariant: this value must sit strictly between the WORST-scoring chunk
# that mentions the subject and the BEST-scoring chunk that does not. Above
# min_with the probe misses genuine mentions; below max_without it reports
# strangers. On the shipped corpus at 256 dimensions those bounds are
# max_without = 0.071 and min_with = 0.120.
#
# `separation_band()` returns both bounds and tests/test_verify.py asserts
# max_without < NEIGHBOR_THRESHOLD < min_with, so a corpus or embedder change
# that invalidates this value fails the suite rather than silently weakening
# the probe.
NEIGHBOR_THRESHOLD = 0.10


# A token appearing in more than this fraction of index entries is treated as
# corpus background rather than as something that identifies a subject.
COMMON_TOKEN_FRACTION = 0.25


def subject_query(index: VectorIndex, subject: Subject) -> str:
    """Build the probe query, dropping tokens that identify the CORPUS.

    A probe query is not just the identifier list joined together, for two
    reasons that both cause false positives:

    An identifier set is not homogeneous. An email address is two identifiers
    glued together: `jrivera` names a person, `acme.example` names an employer.
    Keeping the domain scores every colleague as "near the subject", so only
    the local part is used.

    The frequency baseline must exclude the Subject. Rare tokens identify a
    person, so high-document-frequency tokens are dropped, but the frequency
    has to be measured over entries that do NOT mention the subject. Measured
    over the whole corpus, a heavily-mentioned subject looks like background
    and the filter discards the very tokens worth probing for. Scoped to the
    background it is correct in both directions, and it still behaves when
    erasure has succeeded and every entry is background.
    """
    from boundary.embeddings import tokenize

    entries = list(index.entries.values())
    if not entries:
        return " ".join(sorted(subject.identifiers))

    background = [e for e in entries if not subject.matches(e.text)] or entries
    doc_freq: dict[str, int] = {}
    for entry in background:
        for tok in set(tokenize(entry.text)):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    cutoff = max(1, int(len(background) * COMMON_TOKEN_FRACTION))
    kept: list[str] = []
    for ident in sorted(subject.identifiers):
        # Local part only: the domain is an organizational identifier and
        # the frequency filter cannot remove it on a small corpus, where the
        # background set is too small for the cutoff to bite.
        personal = ident.split("@", 1)[0] if "@" in ident else ident
        for tok in tokenize(personal):
            if doc_freq.get(tok, 0) <= cutoff and tok not in kept:
                kept.append(tok)
    # If every token was filtered out the subject is indistinguishable from
    # the corpus background; fall back rather than probe with an empty query.
    return " ".join(kept) if kept else " ".join(sorted(subject.identifiers))


@dataclass
class ErasureVerifier:
    """Runs every probe and reports which ones still find the subject."""

    store: DataStore
    index: VectorIndex
    embedder: Embedder
    neighbor_threshold: float = NEIGHBOR_THRESHOLD
    # Supplied only in the sealed posture. A `boundary.crypto.KeyStore` and the
    # `SealedCorpus` returned by `seal_store`. Left as `object` so this module
    # does not import crypto, and absent entirely for a plaintext store.
    keys: Optional[object] = None
    sealed: Optional[object] = None

    def verify(self, subject: Subject) -> VerificationReport:
        probes = (
            self._source_scan(subject),
            self._chunk_scan(subject),
            self._index_text_scan(subject),
            self._retrieval_probe(subject),
            self._tombstone_probe(subject),
            self._neighbor_probe(subject),
            self._cache_probe(subject),
            self._prompt_log_probe(subject),
            self._cross_tenant_probe(subject),
            self._key_material_probe(subject),
        )
        return VerificationReport(subject_id=subject.subject_id, probes=probes)

    # ------------------------------------------------------------- readable

    def readable(self, artifact_id: str, stored: str) -> str:
        """What the RUNNING SYSTEM can actually read out of this artifact.

        This method is why the string probes are still honest under encryption,
        and getting it wrong would have been the worst defect in the project.

        `Subject.matches(ciphertext)` is False, so a probe that matched on
        stored bytes would report an encrypted store clean even when the key is
        sitting right there in the keyring and any query can open it. That is a
        false negative in the measuring instrument, failing in the direction
        that flatters the control, which is the direction a wrong measurement
        gets believed in.

        So a sealed artifact is resolved to its plaintext before matching, and
        an artifact that CANNOT be opened resolves to the empty string, which
        matches nothing. Erasure then shows up as unreadability rather than as
        an absence of bytes, which is what crypto-shredding actually is.

        Live key material only. Snapshots and escrow are not consulted here;
        they are `_key_material_probe`'s job. Letting a surviving backup light
        up all seven string probes would report one failure seven times and
        bury the finding that produced it.
        """
        if self.sealed is None or self.keys is None:
            return stored
        envelope = self.sealed.envelopes.get(artifact_id)
        if envelope is None:
            return stored
        opened = self.keys.open(envelope, source=LIVE)
        return opened.decode("utf-8", errors="replace") if opened is not None else ""

    # -------------------------------------------------------------- probes

    def _source_scan(self, subject: Subject) -> LeakProbe:
        hits = [
            d.doc_id
            for d in self.store.documents.values()
            if subject.matches(self.readable(d.doc_id, d.text))
        ]
        return LeakProbe(
            probe="source_scan",
            leaked=bool(hits),
            detail="documents whose text still names the subject",
            hits=tuple(sorted(hits)),
        )

    def _chunk_scan(self, subject: Subject) -> LeakProbe:
        hits = [
            c.chunk_id
            for c in self.store.chunks.values()
            if subject.matches(self.readable(c.chunk_id, c.text))
        ]
        return LeakProbe(
            probe="chunk_scan",
            leaked=bool(hits),
            detail="derived chunks that still name the subject",
            hits=tuple(sorted(hits)),
        )

    def _index_text_scan(self, subject: Subject) -> LeakProbe:
        """The text stored alongside each vector.

        Vector stores keep a payload copy for display. Erasing the chunk store
        and forgetting the payload leaves a full-text copy of the subject
        sitting in the index, reachable by anyone who can query it.
        """
        hits = [
            entry.chunk_id
            for entry in self.index.entries.values()
            if subject.matches(self.readable(entry.chunk_id, entry.text))
        ]
        return LeakProbe(
            probe="index_text_scan",
            leaked=bool(hits),
            detail="index payloads that still contain the subject",
            hits=tuple(sorted(hits)),
        )

    def _retrieval_probe(self, subject: Subject) -> LeakProbe:
        """What the application's own tenant-scoped search returns."""
        qv = self.embedder.embed(subject_query(self.index, subject))
        hits = self.index.search(qv, k=5, tenant_id=subject.tenant_id)
        leaking = [
            h.chunk_id
            for h in hits
            if subject.matches(self.readable(h.chunk_id, h.text))
        ]
        return LeakProbe(
            probe="retrieval_probe",
            leaked=bool(leaking),
            detail="the app's own search still returns the subject",
            hits=tuple(sorted(leaking)),
        )

    def _tombstone_probe(self, subject: Subject) -> LeakProbe:
        """A query path that ignores delete flags.

        Not hypothetical: a tombstone is honored by the query layer, and
        anything reading the segment underneath it, a backup, a replica built
        before compaction, an admin tool, a bulk export, does not see the flag
        at all.
        """
        qv = self.embedder.embed(subject_query(self.index, subject))
        hits = self.index.search(qv, k=10, tenant_id=None, honor_tombstones=False)
        leaking = [
            h.chunk_id
            for h in hits
            if subject.matches(self.readable(h.chunk_id, h.text))
        ]
        return LeakProbe(
            probe="tombstone_probe",
            leaked=bool(leaking),
            detail="tombstoned vectors still carry the subject's text",
            hits=tuple(sorted(leaking)),
        )

    def _neighbor_probe(self, subject: Subject) -> LeakProbe:
        """The probe that finds what string search cannot.

        Embed the subject's identifiers, then look at the nearest vectors
        regardless of tenant and regardless of tombstones. A vector computed
        from text that named the subject stays close to that subject's
        embedding even after the text has been deleted from every store, and
        no grep will ever find it.

        A hit here with no corresponding text hit is the signature case: the
        chunk is gone, the payload is gone, and the DERIVED VECTOR is still
        answering questions about the person.
        """
        qv = self.embedder.embed(subject_query(self.index, subject))
        near = self.index.neighbors(qv, k=10, include_deleted=True)
        leaking = [
            h.chunk_id for h in near if h.score >= self.neighbor_threshold
        ]
        detail = (
            f"vectors within {self.neighbor_threshold:.2f} cosine of the "
            f"subject's identifiers"
        )
        return LeakProbe(
            probe="neighbor_probe",
            leaked=bool(leaking),
            detail=detail,
            hits=tuple(sorted(leaking)),
        )

    def _cache_probe(self, subject: Subject) -> LeakProbe:
        hits = [
            e.key
            for e in self.store.response_cache.values()
            if subject.matches(self.readable(e.key, e.text))
        ]
        return LeakProbe(
            probe="cache_probe",
            leaked=bool(hits),
            detail="cached answers still quote the subject",
            hits=tuple(sorted(hits)),
        )

    def _prompt_log_probe(self, subject: Subject) -> LeakProbe:
        hits = [
            e.entry_id
            for e in self.store.prompt_log.values()
            if subject.matches(self.readable(e.entry_id, e.text))
        ]
        return LeakProbe(
            probe="prompt_log_probe",
            leaked=bool(hits),
            detail="retained prompts still contain the subject",
            hits=tuple(sorted(hits)),
        )

    def _cross_tenant_probe(self, subject: Subject) -> LeakProbe:
        """Reachability from a NEIGHBORING tenant's scope.

        Runs the same search another tenant would run. A hit means the subject
        was reachable across the tenant boundary as well as surviving erasure,
        which is two findings at once and the worse of the two.
        """
        others = [t for t in self.index.tenants() if t != subject.tenant_id]
        if not others:
            # Nobody to leak to is not the same as nothing leaked, and the
            # probe says which one it is reporting.
            return LeakProbe(
                probe="cross_tenant_probe",
                leaked=False,
                detail="not applicable: no other tenant in the index",
                applicable=False,
            )
        qv = self.embedder.embed(subject_query(self.index, subject))
        leaking: list[str] = []
        for tenant in others:
            for hit in self.index.search(qv, k=5, tenant_id=tenant):
                if subject.matches(self.readable(hit.chunk_id, hit.text)):
                    leaking.append(f"{tenant}:{hit.chunk_id}")
        return LeakProbe(
            probe="cross_tenant_probe",
            leaked=bool(leaking),
            detail="the subject is reachable from another tenant's scope",
            hits=tuple(sorted(leaking)),
        )


    def _key_material_probe(self, subject: Subject) -> LeakProbe:
        """The probe for the thing that actually breaks crypto-shredding.

        Destroying a key erases a subject only if every copy of the key is
        gone. Real programs miss the copies: last night's key-manager snapshot,
        the escrow account somebody set up for break-glass access, the backup
        taken minutes before the destroy ran. The ciphertext is untouched and
        perfectly readable to anyone holding any one of them.

        It does not ask the key store whether the key was destroyed. That would
        be the verifier reading the eraser's own bookkeeping, which the
        docstring at the top of this file forbids, and a key store that lied
        about its own state would be confirmed rather than caught. The probe
        attempts the decrypt against every named source and reports which ones
        worked.

        It asks a subject-scoped question, not "can anything open this". An
        envelope concerning two subjects opens for either of them, so the broad
        question would report a leak for Rivera every time Raman's key happened
        to work: true of the record, false of the key. `wrap_sources` unwraps
        this subject's wrap specifically.
        """
        if self.sealed is None or self.keys is None:
            return LeakProbe(
                probe="key_material_probe",
                leaked=False,
                detail="not applicable: plaintext posture, no key material exists to survive",
                applicable=False,
            )
        envelopes = self.sealed.for_subject(subject.subject_id)
        if not envelopes:
            return LeakProbe(
                probe="key_material_probe",
                leaked=False,
                detail="no sealed artifact is wrapped for this subject",
            )
        sources: list[str] = []
        for envelope in envelopes:
            for name in self.keys.wrap_sources(subject.subject_id, envelope):
                if name not in sources:
                    sources.append(name)
        return LeakProbe(
            probe="key_material_probe",
            leaked=bool(sources),
            detail="key material that opens the subject's data still exists",
            hits=tuple(sorted(sources)),
        )


def separation_band(
    index: VectorIndex, embedder: Embedder, subject: Subject
) -> dict[str, float]:
    """The four numbers that justify the neighbor threshold.

    Returns the min and mean similarity of chunks that DO mention the subject
    and the max and mean of those that do not. A defensible threshold sits
    strictly between `max_without` and `min_with`. When those two cross, no
    threshold separates the populations and the neighbor probe has stopped
    measuring anything, which the demo prints rather than hides.
    """
    from boundary.embeddings import cosine

    qv = embedder.embed(subject_query(index, subject))
    with_subject: list[float] = []
    without: list[float] = []
    for entry in index.entries.values():
        score = cosine(qv, entry.vector)
        (with_subject if subject.matches(entry.text) else without).append(score)
    return {
        "min_with": min(with_subject) if with_subject else 0.0,
        "mean_with": sum(with_subject) / len(with_subject) if with_subject else 0.0,
        "max_without": max(without) if without else 0.0,
        "mean_without": sum(without) / len(without) if without else 0.0,
        "separated": float(
            bool(with_subject and without and min(with_subject) > max(without))
        ),
    }


def separation(
    store: DataStore, index: VectorIndex, embedder: Embedder, subject: Subject
) -> tuple[float, float]:
    """Mean similarity to the subject for chunks that mention them vs those that do not.

    Printed by the demo so the neighbor threshold is visibly between the two
    populations rather than asserted to be. If these two numbers converge, the
    neighbor probe has stopped discriminating and any zero it reports is
    meaningless: the same trap a reduced benchmark subset falls into when it
    stops separating.
    """
    qv = embedder.embed(subject_query(index, subject))
    from boundary.embeddings import cosine

    with_subject: list[float] = []
    without: list[float] = []
    for entry in index.entries.values():
        score = cosine(qv, entry.vector)
        if subject.matches(entry.text):
            with_subject.append(score)
        else:
            without.append(score)
    mean_with = sum(with_subject) / len(with_subject) if with_subject else 0.0
    mean_without = sum(without) / len(without) if without else 0.0
    return mean_with, mean_without
