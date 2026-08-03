"""The audit chain, and the three ways it can be attacked."""
from __future__ import annotations

import json

import pytest

from boundary.audit import GENESIS, HASHED_FIELDS, AuditLog


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
