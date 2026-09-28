"""The audit chain, and the three ways it can be attacked."""
from __future__ import annotations

import json
from dataclasses import fields

import pytest

from boundary.audit import (
    GENESIS,
    HASHED_FIELDS,
    SERIALIZED_FIELDS,
    AuditLog,
    AuditRecord,
)


def _seed(log, n=4):
    for i in range(n):
        log.record(event="egress", tenant_id="acme", detail={"i": i, "action": "allow"})
    return log


def test_editing_a_record_breaks_the_chain(frozen_audit):
    """Contents must not be changeable without detection.

    Mutation check, executed in-test: the edit is reverted and the chain is
    asserted intact again, so the failure is the edit and not a fragile check.
    """
    log = _seed(frozen_audit)
    assert log.verify_chain()[0]

    original = log.records[1].detail["action"]
    log.records[1].detail["action"] = "deny"
    intact, index, reason = log.verify_chain()
    assert not intact
    assert index == 1
    assert "hash" in reason

    log.records[1].detail["action"] = original
    assert log.verify_chain()[0]


def test_reordering_two_records_breaks_the_chain(frozen_audit):
    log = _seed(frozen_audit)
    log.records[1], log.records[2] = log.records[2], log.records[1]
    intact, index, reason = log.verify_chain()
    assert not intact
    assert index == 1


def test_excising_a_record_from_the_middle_breaks_the_chain(frozen_audit):
    """A deletion that relinks survivors passes the hash checks and fails on seq.

    This is why verify_chain checks sequence contiguity as well as hashes:
    the first two checks alone would accept a truncated log.
    """
    log = _seed(frozen_audit)
    log.records.pop(1)
    intact, index, reason = log.verify_chain()
    assert not intact
    assert "sequence" in reason


def test_the_first_record_links_to_genesis(frozen_audit):
    log = _seed(frozen_audit, n=1)
    assert log.records[0].prev_hash == GENESIS


def test_the_timestamp_is_inside_the_hash(frozen_audit):
    """A faked timestamp is worse than an honest one that varies.

    Mutation check, executed in-test: changing only the timestamp is shown to
    invalidate the record hash.
    """
    assert "timestamp" in HASHED_FIELDS
    log = _seed(frozen_audit, n=1)
    rec = log.records[0]
    before = rec.record_hash
    rec.timestamp = "2020-01-01T00:00:00+00:00"
    assert rec.compute_hash() != before


def test_the_log_never_stores_the_data_it_is_about(frozen_audit):
    """An audit trail that quotes the personal data is a second copy of it."""
    log = frozen_audit
    log.record(
        event="egress",
        tenant_id="acme",
        detail={"findings": 3, "classes": ["ssn", "email"], "action": "redact"},
    )
    dumped = log.to_jsonl()
    assert "482-19-7734" not in dumped
    assert "jrivera@acme.example" not in dumped
    assert "ssn" in dumped


def test_jsonl_round_trips_as_canonical_json(frozen_audit):
    log = _seed(frozen_audit, n=2)
    lines = log.to_jsonl().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert set(HASHED_FIELDS).issubset(parsed.keys())


def test_an_empty_log_is_intact():
    assert AuditLog().verify_chain()[0]


def test_a_serialized_record_carries_exactly_the_hashed_fields_and_the_hash(frozen_audit):
    """HASHED_FIELDS is the whole record, not a subset of it.

    A field that reaches the file without entering the hash is content the
    chain reports intact while it can be edited freely. The serialized keys
    are therefore asserted EQUAL to the hashed fields plus the hash, and the
    record's own fields are asserted equal to the same set, so a field added
    to either place without the other fails here.

    Mutation check, executed against the serializer: add a `note` key to the
    serialized dict outside the payload and the first assertion fails; add a
    dataclass field to AuditRecord without adding it to HASHED_FIELDS and the
    second one does.
    """
    log = _seed(frozen_audit, n=2)
    for line in log.to_jsonl().splitlines():
        assert set(json.loads(line)) == set(SERIALIZED_FIELDS)
    assert {f.name for f in fields(AuditRecord)} == set(SERIALIZED_FIELDS)
    assert set(HASHED_FIELDS) == set(SERIALIZED_FIELDS) - {"record_hash"}


def test_a_second_process_resumes_the_trail_instead_of_erasing_it(tmp_path):
    """Two logs pointed at one path, one after the other, as two processes
    would be: the file holds both, as one chain. A file whose chain is broken
    is refused rather than extended."""
    target = tmp_path / "boundary.audit.jsonl"
    first = AuditLog(path=target)
    first.record(event="egress", tenant_id="acme", detail={"n": 1})
    first.record(event="egress", tenant_id="acme", detail={"n": 2})
    second = AuditLog(path=target)
    second.record(event="egress", tenant_id="acme", detail={"n": 3})

    on_disk = AuditLog.read(target)
    assert [r.detail["n"] for r in on_disk.records] == [1, 2, 3]
    assert on_disk.verify_chain() == (True, None, "chain intact")

    lines = target.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"n":1', '"n":9')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to extend"):
        AuditLog(path=target).record(event="egress", tenant_id="acme")


def test_a_configured_path_receives_every_record_as_it_is_made(tmp_path):
    """The file is the chain, written record by record, and it reads back.

    Mutation check, executed against `record`: stop the write and the file is
    never created; write only on the first record and the line count falls.
    """
    target = tmp_path / "trail" / "boundary.audit.jsonl"
    log = AuditLog(path=target)
    assert not target.exists()

    log.record(event="egress", tenant_id="acme", detail={"action": "allow"})
    assert target.read_text(encoding="utf-8").splitlines() == log.to_jsonl().splitlines()
    log.record(event="ingress", tenant_id="acme", detail={"findings": 0})
    log.record(event="retrieval", tenant_id="acme", detail={"k": 5})
    on_disk = target.read_text(encoding="utf-8").splitlines()
    assert on_disk == log.to_jsonl().splitlines()
    assert len(on_disk) == 3

    reread = AuditLog.read(target)
    assert reread.verify_chain() == (True, None, "chain intact")
    assert [r.record_hash for r in reread.records] == [r.record_hash for r in log.records]

    # An edit on disk is caught on re-read, so the file is evidence and not
    # a convenience copy.
    lines = on_disk[:]
    lines[1] = lines[1].replace('"findings":0', '"findings":9')
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    intact, index, _ = AuditLog.read(target).verify_chain()
    assert not intact and index == 1


def test_a_record_spliced_from_another_log_breaks_the_chain(frozen_audit, request):
    """The LINK is what turns this into a chain rather than a list of sealed
    rows.

    Every other attack here also disturbs the sequence numbers, so the seq
    check alone catches them and the prev_hash comparison is never the thing
    that fires. A record lifted from a DIFFERENT log at the same position keeps
    its seq and its own valid contents hash, and only the link exposes it.

    Mutation check, executed against the verifier: delete the prev_hash
    comparison in `verify_chain` and this is the test that goes red. The
    reordering and excision tests stay green without it.
    """
    log = _seed(frozen_audit)
    other = _seed(AuditLog(clock=frozen_audit.clock), n=4)

    donor = other.records[2]
    assert donor.seq == log.records[2].seq
    assert donor.record_hash == donor.compute_hash(), "donor is internally valid"

    log.records[2] = donor
    intact, index, reason = log.verify_chain()
    assert not intact, "a spliced record with a valid seq and hash was accepted"
    assert index == 2
    assert "prev_hash" in reason
