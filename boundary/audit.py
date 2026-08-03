"""A hash-chained, tamper-evident record of every boundary decision.

Each record carries the hash of the one before it, so the log is a chain rather
than a list. Editing a record, reordering two, or excising one and relinking
the survivor all break the chain at a detectable point, and `verify_chain()`
reports the first index where it breaks.

What is deliberately not hashed: nothing. The hash covers the canonical JSON of
every field including the timestamp, which means two runs produce different
record hashes. That is the honest choice. Freezing the clock to make the log
byte-reproducible would defeat the purpose of an audit trail, whose entire
value is that it says WHEN a decision was made. Reproducibility comes from
canonical serialization with sorted keys, not from pinning time.

The log never contains the data it is about. Records store counts, class names,
rule ids and decisions: never the text, never a detected value. An audit trail
of privacy decisions that quotes the personal data it was protecting is a second
copy of the thing you were trying to contain, and it is usually retained longer
and read by more people than the system it audits.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

GENESIS = "0" * 64

# Fields covered by the record hash. Listed explicitly so a future field cannot
# be added to the payload and silently excluded from the integrity guarantee.
HASHED_FIELDS = ("seq", "timestamp", "event", "tenant_id", "detail", "prev_hash")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class AuditRecord:
    seq: int
    timestamp: str
    event: str
    tenant_id: str
    detail: dict[str, Any]
    prev_hash: str
    record_hash: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "tenant_id": self.tenant_id,
            "detail": self.detail,
            "prev_hash": self.prev_hash,
        }

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self.payload()).encode("utf-8")).hexdigest()


@dataclass
class AuditLog:
    """Append-only chain of decisions."""

    records: list[AuditRecord] = field(default_factory=list)
    clock: Optional[Any] = None

    def _now(self) -> str:
        if self.clock is not None:
            return self.clock()
        return datetime.now(timezone.utc).isoformat()

    def record(
        self, event: str, tenant_id: str, detail: Optional[dict[str, Any]] = None
    ) -> AuditRecord:
        prev = self.records[-1].record_hash if self.records else GENESIS
        rec = AuditRecord(
            seq=len(self.records),
            timestamp=self._now(),
            event=event,
            tenant_id=tenant_id,
            detail=dict(detail or {}),
            prev_hash=prev,
        )
        rec.record_hash = rec.compute_hash()
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------ integrity

    def verify_chain(self) -> tuple[bool, Optional[int], str]:
        """Return (intact, first_bad_index, reason).

        Checks three things, because they fail differently: each record's own
        hash still matches its contents, each record's prev_hash matches the
        previous record's hash, and the sequence numbers are contiguous from
        zero. A deletion that relinks the survivors passes the first two checks
        and fails the third.
        """
        expected_prev = GENESIS
        for i, rec in enumerate(self.records):
            if rec.seq != i:
                return False, i, f"sequence gap: record claims seq {rec.seq} at index {i}"
            if rec.prev_hash != expected_prev:
                return False, i, "prev_hash does not match the preceding record"
            if rec.record_hash != rec.compute_hash():
                return False, i, "record contents do not match its hash"
            expected_prev = rec.record_hash
        return True, None, "chain intact"

    def events(self, event: Optional[str] = None) -> tuple[AuditRecord, ...]:
        if event is None:
            return tuple(self.records)
        return tuple(r for r in self.records if r.event == event)

    def to_jsonl(self) -> str:
        return "\n".join(
            _canonical({**r.payload(), "record_hash": r.record_hash})
            for r in self.records
        )

    def __len__(self) -> int:
        return len(self.records)
