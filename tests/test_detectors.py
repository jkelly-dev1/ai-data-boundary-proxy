"""Detectors, including the things they are supposed to miss."""
from __future__ import annotations

import pytest

from boundary.corpus import LABELED, ROSTER, UNROSTERED, build_corpus
from boundary.detectors import (
    DetectorSuite,
    NameDetector,
    dedupe,
    default_suite,
    detect_context,
    detect_loose,
    detect_strict,
    luhn_ok,
    merge_overlaps,
)
from boundary.metrics import correlate, score_detector
from boundary.models import DataClass


def test_luhn_rejects_a_number_that_only_looks_like_a_card():
    """Without the checksum, any 16 digits is reported as a credit card.

    Mutation check, executed in-test: the same string is shown to match the
    card PATTERN, so it is the validator and not the regex doing the work.
    """
    import re

    from boundary.detectors import _STRICT_PATTERNS

    good, bad = "4111111111111111", "4111111111111112"
    assert luhn_ok(good)
    assert not luhn_ok(bad)

    pattern = next(p for c, p, _ in _STRICT_PATTERNS if c is DataClass.CREDIT_CARD)
    assert pattern.search(bad), "the mutation control failed: pattern should match"
    assert not any(f.data_class is DataClass.CREDIT_CARD for f in detect_strict(bad))


def test_structurally_impossible_ssns_are_not_reported():
    """000-00-0000 appears in every template and is never a real SSN."""
    assert not detect_strict("SSN 000-00-0000 is a placeholder.")
    assert not detect_strict("SSN 666-12-3456 was never issued.")
    assert detect_strict("SSN 482-19-7734 on file.")


def test_person_name_recall_is_below_one_and_that_is_the_finding():
    """A roster only finds the names it was given.

    Mutation check, executed in-test: adding the unrostered names to the roster
    raises recall to 1.0, proving the miss is the roster's coverage rather than
    a broken matcher.
    """
    detector = NameDetector(ROSTER)
    assert detector("Jordan Rivera signed the form.")
    for name in UNROSTERED:
        assert not detector(f"{name} signed the form."), name

    widened = NameDetector(tuple(ROSTER) + UNROSTERED)
    for name in UNROSTERED:
        assert widened(f"{name} signed the form."), name


def test_overall_recall_stays_below_one_on_the_labeled_set(suite):
    """The gate depends on this staying honest."""
    score = score_detector("all", LABELED, suite.run)
    assert score.recall.point < 1.0
    assert score.false_negatives >= 1
    assert score.false_positives == 0


def test_a_second_detector_is_not_independent_of_the_first():
    """Stacked detectors share blind spots; the marginal gain is measured.

    Mutation check, executed in-test: a deliberately independent second
    detector (the context family, which keys on labels rather than values) is
    shown to have a different marginal gain from the loose family.
    """
    loose_gain = correlate(LABELED, detect_strict, detect_loose, "s", "l").marginal_gain
    context_gain = correlate(
        LABELED, detect_strict, detect_context, "s", "c"
    ).marginal_gain
    assert loose_gain < 1.0
    assert loose_gain != context_gain


def test_dedupe_collapses_the_same_claim_from_two_detectors():
    """Two detectors finding one SSN is one fact, not two.

    Counting it twice inflates apparent coverage, which is the arithmetic error
    that makes stacked detectors look additive.
    """
    text = "SSN 482-19-7734 on file."
    combined = detect_strict(text) + detect_loose(text)
    ssn = [f for f in combined if f.data_class is DataClass.SSN]
    assert len(ssn) == 2, "expected both families to fire"
    assert len(dedupe(ssn)) == 1


def test_merge_overlaps_keeps_the_longer_span():
    """Nested matches would corrupt redaction by rewriting inside a rewrite."""
    text = "Call +1 512-555-0143 now."
    findings = merge_overlaps(detect_strict(text) + detect_loose(text))
    phones = [f for f in findings if f.data_class is DataClass.PHONE]
    assert len(phones) == 1
    assert len(phones[0].span) == max(
        len(f.span)
        for f in detect_strict(text) + detect_loose(text)
        if f.data_class is DataClass.PHONE
    )


def test_context_detector_finds_a_value_that_is_weak_on_its_own():
    """"1982-03-14" is just a date until a DOB label sits next to it."""
    assert detect_context("DOB: 1982-03-14 recorded.")
    assert not detect_context("The release shipped on 1982-03-14.")


def test_quasi_identifiers_are_classified_separately_from_direct_ones():
    assert DataClass.SSN.is_direct
    assert not DataClass.SSN.is_quasi
    assert DataClass.POSTAL_CODE.is_quasi
    assert DataClass.DATE_OF_BIRTH.is_quasi


def test_an_empty_suite_finds_nothing_and_does_not_raise():
    assert DetectorSuite({}).run("SSN 482-19-7734") == []


def test_every_label_rule_class_fires_somewhere_in_the_corpus():
    """A rule that matches nothing is indistinguishable from a rule that works.

    The label-driven rules, JOB_TITLE, EMPLOYER, MRN, are the ones whose
    patterns can stop matching without any recall figure moving, because
    `LABELED` carries no case for them. Their only exercise is the corpus
    itself, so the corpus is where they have to be held.

    Mutation check, executed against the detector: drop `re.I` from the
    EMPLOYER rule and its two corpus hits disappear, which this catches. The
    same applies to a label rule whose pattern is edited into one that cannot
    match.
    """
    corpus = build_corpus()
    found: dict[str, set[str]] = {}
    for doc in corpus.store.documents.values():
        for finding in detect_context(doc.text):
            value = doc.text[finding.span.start : finding.span.end]
            found.setdefault(finding.data_class.value, set()).add(value)

    for expected in ("job_title", "employer"):
        assert expected in found, (
            f"the {expected} rule matched nothing in the whole corpus; "
            f"context detector produced only {sorted(found)}"
        )

    assert found["employer"] == {"Acme Robotics", "Vertex Supply"}, found["employer"]
    assert found["job_title"] == {
        "Staff Platform Engineer",
        "Director of Procurement",
    }, found["job_title"]
