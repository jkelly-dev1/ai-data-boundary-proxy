"""The index, including the query paths that leak."""
from __future__ import annotations

import pytest

from boundary.embeddings import default_embedder
from boundary.erasure import build_index
from boundary.index import VectorIndex
from boundary.models import IndexEntry


def test_post_filtering_by_tenant_returns_fewer_results_than_requested(indexed):
    """The shortfall is the disclosure.

    Ranking the whole index and then dropping other tenants' rows tells the
    caller those rows exist and roughly how well they scored.

    Mutation check, executed in-test: the pre-filtered query with identical
    arguments returns the full k, so the shortfall is the post-filter and not
    a thin corpus.
    """
    corpus, index, emb = indexed
    qv = emb.embed("employee record payroll access")
    pre = index.search(qv, k=5, tenant_id="acme", post_filter=False)
    post = index.search(qv, k=5, tenant_id="acme", post_filter=True)
    assert len(pre) == 5
    assert len(post) < len(pre)
    assert all(h.tenant_id == "acme" for h in pre + post)


def test_delete_is_a_tombstone_and_purge_is_a_removal():
    index = VectorIndex()
    index.upsert(IndexEntry(chunk_id="c1", tenant_id="t", vector=(1.0, 0.0), text="x"))
    assert index.delete("c1")
    assert index.get("c1") is not None
    assert index.get("c1").deleted
    assert len(index) == 0
    assert "c1" in index.all_ids()

    assert index.purge("c1")
    assert index.get("c1") is None


def test_a_query_that_ignores_tombstones_still_returns_deleted_vectors():
    """The path a backup, replica or admin tool takes.

    Mutation check, executed in-test: the same query honoring tombstones
    returns nothing, so the difference is the flag and not the vector.
    """
    index = VectorIndex()
    index.upsert(IndexEntry(chunk_id="c1", tenant_id="t", vector=(1.0, 0.0), text="secret"))
    index.delete("c1")
    assert index.search((1.0, 0.0), k=5, honor_tombstones=True) == []
    ignored = index.search((1.0, 0.0), k=5, honor_tombstones=False)
    assert [h.chunk_id for h in ignored] == ["c1"]
    assert ignored[0].deleted


def test_compaction_reclaims_tombstones():
    index = VectorIndex()
    for i in range(3):
        index.upsert(
            IndexEntry(chunk_id=f"c{i}", tenant_id="t", vector=(1.0, 0.0), text="x")
        )
    index.delete("c0")
    index.delete("c1")
    assert index.compact() == 2
    assert index.all_ids() == ("c2",)


def test_ranking_is_stable_when_scores_tie():
    """An unstable ordering would make the gate flap for unrelated reasons."""
    index = VectorIndex()
    for cid in ("cb", "ca", "cc"):
        index.upsert(
            IndexEntry(chunk_id=cid, tenant_id="t", vector=(1.0, 0.0), text="same")
        )
    first = [h.chunk_id for h in index.search((1.0, 0.0), k=3)]
    second = [h.chunk_id for h in index.search((1.0, 0.0), k=3)]
    assert first == second == ["ca", "cb", "cc"]


def test_an_orphaned_entry_is_still_searchable(indexed):
    """A source-only delete leaves exactly this state."""
    corpus, index, emb = indexed
    chunk_id = next(iter(corpus.store.chunks))
    corpus.store.remove_chunk(chunk_id)
    assert index.get(chunk_id) is not None
    hits = index.search(index.get(chunk_id).vector, k=1)
    assert hits and hits[0].chunk_id == chunk_id


def test_search_rejects_a_dimension_mismatch(indexed):
    corpus, index, emb = indexed
    with pytest.raises(ValueError):
        index.search((1.0, 0.0, 0.0), k=1)


def test_stats_separate_live_from_tombstoned(indexed):
    corpus, index, emb = indexed
    total_before = index.stats()["total"]
    index.delete(next(iter(index.entries)))
    stats = index.stats()
    assert stats["total"] == total_before
    assert stats["tombstoned"] == 1
    assert stats["live"] == total_before - 1
