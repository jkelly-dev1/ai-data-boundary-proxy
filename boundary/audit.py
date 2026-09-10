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

The trail can be written to disk. A log built with `path` appends each record
to that file as one canonical JSON line at the moment it is made, so the file
is the same chain the process holds and `read()` brings it back for
verification. Without a path the chain lives only in the process.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

GENESIS = "0" * 64

# Fields covered by the record hash. The hashed payload and the serialized
# line are both BUILT FROM this tuple, so a field cannot be added to the
# record and reach the file without also entering the hash; `payload()` reads
# these names off the record rather than listing them a second time.
HASHED_FIELDS = ("seq", "timestamp", "event", "tenant_id", "detail", "prev_hash")

# Everything a serialized record carries: the hashed fields and the hash
# itself. The hash is the one field that cannot cover itself.
SERIALIZED_FIELDS = HASHED_FIELDS + ("record_hash",)


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
        return {name: getattr(self, name) for name in HASHED_FIELDS}

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self.payload()).encode("utf-8")).hexdigest()

    def serialize(self) -> str:
        """The record as one canonical JSON line: the payload plus its hash."""
        return _canonical({**self.payload(), "record_hash": self.record_hash})

    @classmethod
    def from_json(cls, line: str) -> "AuditRecord":
        data = json.loads(line)
        return cls(**{f.name: data[f.name] for f in fields(cls)})


@dataclass
class AuditLog:
    """Append-only chain of decisions.

    `path`, when given, is where the chain is written. The file holds THIS
    log's chain: it is created on the first record and appended to on every
    record after that, so a reader sees the same sequence the process holds,
    starting at GENESIS, with nothing from any other process in front of it.
    """

    records: list[AuditRecord] = field(default_factory=list)
    clock: Optional[Any] = None
    path: Optional[Union[str, Path]] = None

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
        if self.path is not None:
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if len(self.records) == 1 else "a"
            with target.open(mode, encoding="utf-8") as fh:
                fh.write(rec.serialize() + "\n")
        return rec

    @classmethod
    def from_jsonl(cls, text: str) -> "AuditLog":
        """Rebuild a log from serialized lines. Verify it with `verify_chain`."""
        records = [AuditRecord.from_json(line) for line in text.splitlines() if line]
        return cls(records=records)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "AuditLog":
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))

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
        return "\n".join(r.serialize() for r in self.records)

    def __len__(self) -> int:
        return len(self.records)
