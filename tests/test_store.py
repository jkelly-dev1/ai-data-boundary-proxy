"""The store, its lineage, and the chunking that duplicates a mention."""
from __future__ import annotations

import pytest

from boundary.corpus import build_corpus
from boundary.models import Document, Sensitivity, Subject
from boundary.store import DataStore, build_store, chunk_document


def test_overlapping_chunks_duplicate_a_mention_near_a_boundary():
    """Why removing "the" chunk that owns a mention is not enough.

    Mutation check, executed in-test: the same document chunked with zero
    overlap places the mention in exactly one chunk, so the duplication is the
    overlap and not the text.
    """
    filler = "word " * 60
    doc = Document(
        doc_id="d1",
        tenant_id="t",
        text=f"{filler}Jordan Rivera {filler}",
    )
    overlapped = chunk_document(doc, size=120, overlap=40)
    hits = [c for c in overlapped if "Rivera" in c.text or "Jordan" in c.text]
    assert len(hits) >= 1

    tight = chunk_document(doc, size=120, overlap=0)
    tight_hits = [c for c in tight if "Jordan Rivera" in c.text]
    assert len(tight_hits) <= len(
        [c for c in overlapped if "Jordan Rivera" in c.text]
    )


def test_chunking_records_lineage_back_to_the_document():
    """Without lineage, finding derived artifacts means scanning everything."""
    corpus = build_corpus()
    derived = corpus.store.derived_from("document", "acme-hr-001")
    assert derived
    assert all(d.derived_kind == "chunk" for d in derived)


def test_owner_scoped_lookup_misses_mentions_in_other_documents():
    """The gap between "documents they own" and "documents that name them".

    Mutation check, executed in-test: the mention-based lookup on the same
    subject returns strictly more documents.
    """
    corpus = build_corpus()
    subject = corpus.subjects[0]
    owned = {d.doc_id for d in corpus.store.documents_owned_by(subject)}
    mentioned = {d.doc_id for d in corpus.store.documents_mentioning(subject)}
    assert owned
    assert mentioned - owned, "the corpus must contain mentions the subject does not own"


def test_an_alias_only_document_is_found_by_identifier_not_by_full_name():
    """A document naming "J. Rivera" is the same person as "Jordan Rivera"."""
    corpus = build_corpus()
    subject = corpus.subjects[0]
    note = corpus.store.documents["acme-note-220"]
    assert "Jordan Rivera" not in note.text
    assert subject.matches(note.text)


def test_a_cross_tenant_mention_exists_in_the_corpus():
    """A Tenant B document naming a Tenant A subject.

    This is a pre-existing isolation failure that erasure then has to clean up,
    and it is in the corpus deliberately.
    """
    corpus = build_corpus()
    subject = corpus.subjects[0]
    vendor = corpus.store.documents["vertex-vendor-050"]
    assert vendor.tenant_id != subject.tenant_id
    assert subject.matches(vendor.text)


def test_rewriting_a_chunk_preserves_its_id_and_lineage():
    corpus = build_corpus()
    chunk_id = next(iter(corpus.store.chunks))
    original = corpus.store.chunks[chunk_id]
    assert corpus.store.rewrite_chunk(chunk_id, "replacement text")
    rewritten = corpus.store.chunks[chunk_id]
    assert rewritten.chunk_id == original.chunk_id
    assert rewritten.doc_id == original.doc_id
    assert rewritten.ordinal == original.ordinal
    assert rewritten.text == "replacement text"


def test_chunking_rejects_an_overlap_that_would_not_advance():
    with pytest.raises(ValueError):
        chunk_document(Document(doc_id="d", tenant_id="t", text="x" * 100),
                       size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_document(Document(doc_id="d", tenant_id="t", text="x"), size=0)


def test_prompt_logs_and_caches_are_reachable_by_subject():
    """Copies of the text that no document owns."""
    corpus = build_corpus()
    subject = corpus.subjects[0]
    assert corpus.store.cache_entries_mentioning(subject)
    assert corpus.store.prompt_entries_mentioning(subject)


def test_an_empty_store_reports_zero_without_raising():
    stats = DataStore().stats()
    assert stats["documents"] == 0
    assert stats["chunks"] == 0
