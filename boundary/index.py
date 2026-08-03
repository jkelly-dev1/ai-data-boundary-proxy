"""A vector index with the failure modes a real one has.

This is a brute-force in-memory index, and being brute-force is not the point.
It reproduces three behaviors that cause real leaks and that a toy index
usually papers over:

  1. Tombstones, not removal. `delete()` marks an entry deleted. The bytes stay
     until `compact()` runs. A query path that forgets to check the flag still
     returns the vector, and `search(honor_tombstones=False)` exists to make
     that path testable rather than hypothetical.

  2. PRE-FILTER vs POST-FILTER. Filtering by tenant BEFORE ranking and
     filtering AFTER ranking give different results, and the post-filter
     variant leaks. It returns fewer than k results when another tenant's
     documents outranked yours, which discloses that those documents exist and
     roughly how well they match: a side channel that survives the fact that
     their content was never shown.

  3. Orphaned entries. An index entry whose chunk no longer exists in the store
     is still searchable. That is exactly the state a source-only delete
     leaves behind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from boundary.embeddings import cosine
from boundary.models import IndexEntry


@dataclass
class SearchHit:
    chunk_id: str
    tenant_id: str
    score: float
    text: str
    deleted: bool = False


@dataclass
class VectorIndex:
    """In-memory vector store keyed by chunk id.

    `Opener` is what makes a sealed index searchable, and what makes it
    expensive. When entries carry `sealed_vector` instead of a plaintext
    vector, the index cannot compare a query against them without decrypting
    first, so `search` opens every candidate it can and skips the ones it
    cannot. Set by `boundary.crypto.seal_store` when sealing vectors; None on a
    plaintext index, where the cost is zero because there is nothing to open.

    `decrypt_attempts` counts the opens. It is the honest cost figure for
    variant B and it is a counter rather than a timing, because eleven chunks
    cannot demonstrate a complexity difference in wall-clock and pretending
    otherwise would be theater.
    """

    entries: dict[str, IndexEntry] = field(default_factory=dict)
    opener: Optional[Callable[[object], Optional[tuple[float, ...]]]] = None
    decrypt_attempts: int = 0
    decrypt_failures: int = 0

    def reset_counters(self) -> None:
        self.decrypt_attempts = 0
        self.decrypt_failures = 0

    def _usable_vector(self, entry: IndexEntry) -> Optional[tuple[float, ...]]:
        """The vector to compare against, opening it first when it is sealed.

        Returns None for an entry this index cannot read, which is exactly the
        state a completed crypto-shred produces: the entry is still physically
        present, still tombstone-free, still enumerable by an admin tool, and
        no longer answers a similarity question.
        """
        if entry.sealed_vector is None:
            return entry.vector
        if self.opener is None:
            return None
        self.decrypt_attempts += 1
        opened = self.opener(entry.sealed_vector)
        if opened is None:
            self.decrypt_failures += 1
        return opened

    # ---------------------------------------------------------------- writes

    def upsert(self, entry: IndexEntry) -> None:
        self.entries[entry.chunk_id] = entry

    def delete(self, chunk_id: str) -> bool:
        """Tombstone an entry. Returns False if it was not present.

        Deliberately does NOT drop the dict key. Real indexes defer reclamation,
        and the window between tombstone and compaction is where the leak lives.
        """
        entry = self.entries.get(chunk_id)
        if entry is None:
            return False
        entry.deleted = True
        return True

    def purge(self, chunk_id: str) -> bool:
        """Actually remove the entry and its vector."""
        return self.entries.pop(chunk_id, None) is not None

    def compact(self) -> int:
        """Drop every tombstoned entry. Returns how many were reclaimed."""
        dead = [cid for cid, e in self.entries.items() if e.deleted]
        for cid in dead:
            del self.entries[cid]
        return len(dead)

    # ---------------------------------------------------------------- reads

    def __len__(self) -> int:
        return sum(1 for e in self.entries.values() if not e.deleted)

    def live_ids(self) -> tuple[str, ...]:
        return tuple(sorted(cid for cid, e in self.entries.items() if not e.deleted))

    def all_ids(self) -> tuple[str, ...]:
        """Every id including tombstoned ones. What a compaction sweep sees."""
        return tuple(sorted(self.entries))

    def get(self, chunk_id: str) -> Optional[IndexEntry]:
        return self.entries.get(chunk_id)

    def search(
        self,
        query_vector: Iterable[float],
        k: int = 5,
        tenant_id: Optional[str] = None,
        honor_tombstones: bool = True,
        post_filter: bool = False,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Rank by cosine similarity.

        `post_filter=True` reproduces the broken pattern: rank the WHOLE index,
        take the top k, then drop the rows the caller may not see. The caller
        gets fewer than k results, and the shortfall is itself information
        about the other tenant's data. `post_filter=False` restricts the
        candidate set first and is the correct behavior.
        """
        qv = tuple(query_vector)
        candidates = list(self.entries.values())

        if honor_tombstones:
            candidates = [e for e in candidates if not e.deleted]

        if tenant_id is not None and not post_filter:
            candidates = [e for e in candidates if e.tenant_id == tenant_id]

        scored = []
        for e in candidates:
            vector = self._usable_vector(e)
            if vector is None:
                # Sealed and unopenable. Skipped rather than scored zero: a
                # zero would still occupy a rank and could still be returned by
                # a query with a low min_score.
                continue
            scored.append(
                SearchHit(
                    chunk_id=e.chunk_id,
                    tenant_id=e.tenant_id,
                    score=cosine(qv, vector),
                    text=e.text,
                    deleted=e.deleted,
                )
            )
        scored = [h for h in scored if h.score >= min_score]
        # Ties broken by chunk_id so results are stable across runs; an
        # unstable ordering would make the gate flap for reasons unrelated to
        # any defense.
        scored.sort(key=lambda h: (-h.score, h.chunk_id))

        top = scored[:k]
        if tenant_id is not None and post_filter:
            top = [h for h in top if h.tenant_id == tenant_id]
        return top

    def neighbors(
        self,
        query_vector: Iterable[float],
        k: int = 10,
        include_deleted: bool = True,
    ) -> list[SearchHit]:
        """Unfiltered nearest neighbors, tombstones included by default.

        This is the adversarial view: what someone with direct index access can
        still reach. Erasure verification uses it because honoring the
        application's filters would only prove the application's filters work.
        """
        return self.search(
            query_vector,
            k=k,
            tenant_id=None,
            honor_tombstones=not include_deleted,
            post_filter=False,
        )

    def tenants(self) -> tuple[str, ...]:
        return tuple(sorted({e.tenant_id for e in self.entries.values()}))

    def stats(self) -> dict[str, int]:
        return {
            "live": len(self),
            "tombstoned": sum(1 for e in self.entries.values() if e.deleted),
            "total": len(self.entries),
            "tenants": len(self.tenants()),
        }
