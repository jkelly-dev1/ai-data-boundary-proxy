"""The source of truth, its derivatives, and the lineage between them.

A production RAG system spreads one document across at least six places:

    documents      the source row
    chunks         slices of it, often in a different store
    index          vectors derived from the chunks
    response_cache answers that quoted it
    prompt_log     the prompts that contained it, kept for debugging
    eval_set       cases promoted from real traffic (named, not modeled here)

`delete from documents where id = ?` removes ONE of those six. The other five
are what this module exists to model, because an erasure implementation that
has never been tested against them will pass its own tests and still leave the
subject reachable.

The `derivations` table is the only thing that makes complete erasure tractable.
Without recorded lineage, finding every artifact derived from a document means
scanning every store for a needle you may not be able to describe, and an
embedding cannot be string-matched at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from boundary.models import Chunk, Derivation, Document, Subject


@dataclass
class CacheEntry:
    """A cached answer, and the chunks that were in its context."""

    key: str
    tenant_id: str
    text: str
    source_chunk_ids: tuple[str, ...] = ()


@dataclass
class PromptLogEntry:
    """A retained prompt or completion.

    Prompt logs are the most commonly forgotten copy of production data. They
    are written for debugging, retained by default, and almost never covered by
    an erasure path, so this store keeps them and the verifier probes them.
    """

    entry_id: str
    tenant_id: str
    text: str
    redacted: bool = False


@dataclass
class DataStore:
    """Everything inside the boundary."""

    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    subjects: dict[str, Subject] = field(default_factory=dict)
    derivations: list[Derivation] = field(default_factory=list)
    response_cache: dict[str, CacheEntry] = field(default_factory=dict)
    prompt_log: dict[str, PromptLogEntry] = field(default_factory=dict)

    # ------------------------------------------------------------- mutation

    def add_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self.derivations.append(
            Derivation("document", chunk.doc_id, "chunk", chunk.chunk_id)
        )

    def add_subject(self, subject: Subject) -> None:
        self.subjects[subject.subject_id] = subject

    def record_derivation(
        self, source_kind: str, source_id: str, derived_kind: str, derived_id: str
    ) -> None:
        self.derivations.append(
            Derivation(source_kind, source_id, derived_kind, derived_id)
        )

    def cache_response(self, entry: CacheEntry) -> None:
        self.response_cache[entry.key] = entry

    def log_prompt(self, entry: PromptLogEntry) -> None:
        self.prompt_log[entry.entry_id] = entry

    # -------------------------------------------------------------- queries

    def chunks_for_document(self, doc_id: str) -> tuple[Chunk, ...]:
        return tuple(c for c in self.chunks.values() if c.doc_id == doc_id)

    def derived_from(self, source_kind: str, source_id: str) -> tuple[Derivation, ...]:
        return tuple(
            d
            for d in self.derivations
            if d.source_kind == source_kind and d.source_id == source_id
        )

    def documents_for_tenant(self, tenant_id: str) -> tuple[Document, ...]:
        return tuple(d for d in self.documents.values() if d.tenant_id == tenant_id)

    def chunks_for_tenant(self, tenant_id: str) -> tuple[Chunk, ...]:
        return tuple(c for c in self.chunks.values() if c.tenant_id == tenant_id)

    # ---------------------------------------------------- subject reachability

    def documents_owned_by(self, subject: Subject) -> tuple[Document, ...]:
        """Documents whose owner field names the subject.

        This is what a naive erasure uses, and it is not enough. It finds the
        subject's own records and misses every OTHER document that happens to
        mention them: a support ticket filed by a colleague, an audit entry, a
        meeting note. Those are the chunks that keep a "deleted" person
        retrievable.
        """
        return tuple(
            d for d in self.documents.values() if d.owner_subject_id == subject.subject_id
        )

    def documents_mentioning(self, subject: Subject) -> tuple[Document, ...]:
        """Every document whose text contains any of the subject's identifiers."""
        return tuple(d for d in self.documents.values() if subject.matches(d.text))

    def chunks_mentioning(self, subject: Subject) -> tuple[Chunk, ...]:
        return tuple(c for c in self.chunks.values() if subject.matches(c.text))

    def cache_entries_mentioning(self, subject: Subject) -> tuple[CacheEntry, ...]:
        return tuple(e for e in self.response_cache.values() if subject.matches(e.text))

    def prompt_entries_mentioning(self, subject: Subject) -> tuple[PromptLogEntry, ...]:
        return tuple(e for e in self.prompt_log.values() if subject.matches(e.text))

    # --------------------------------------------------------------- removal

    def remove_document(self, doc_id: str) -> bool:
        return self.documents.pop(doc_id, None) is not None

    def remove_chunk(self, chunk_id: str) -> bool:
        return self.chunks.pop(chunk_id, None) is not None

    def remove_cache_entry(self, key: str) -> bool:
        return self.response_cache.pop(key, None) is not None

    def remove_prompt_entry(self, entry_id: str) -> bool:
        return self.prompt_log.pop(entry_id, None) is not None

    def rewrite_chunk(self, chunk_id: str, new_text: str) -> bool:
        """Replace a chunk's text in place, keeping its id and lineage.

        Used by the redact-in-place erasure strategy, where a chunk that
        mentions the subject belongs to somebody else's document and cannot
        simply be deleted without destroying another party's record.
        """
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return False
        self.chunks[chunk_id] = Chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            tenant_id=chunk.tenant_id,
            text=new_text,
            ordinal=chunk.ordinal,
        )
        return True

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "subjects": len(self.subjects),
            "cache_entries": len(self.response_cache),
            "prompt_log": len(self.prompt_log),
            "derivations": len(self.derivations),
        }


def chunk_document(doc: Document, size: int = 240, overlap: int = 40) -> list[Chunk]:
    """Split a document on whitespace into overlapping windows.

    Overlap is why erasure is harder than it looks. A subject's name near a
    chunk boundary lands in TWO chunks, so removing the one chunk that "owns"
    the mention leaves a second copy behind. The default overlap here is
    non-zero specifically so the corpus contains that case.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = doc.text
    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    tenant_id=doc.tenant_id,
                    text=piece,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_store(documents: Iterable[Document], subjects: Iterable[Subject] = ()) -> DataStore:
    store = DataStore()
    for subj in subjects:
        store.add_subject(subj)
    for doc in documents:
        store.add_document(doc)
        for chunk in chunk_document(doc):
            store.add_chunk(chunk)
    return store
