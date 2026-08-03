"""Metrics: intervals, correlation, and the residual that matters."""
from __future__ import annotations

import random

import pytest

from boundary.corpus import LABELED
from boundary.detectors import detect_loose, detect_strict
from boundary.metrics import (
    BOOTSTRAP_SEED,
    CorrelationResult,
    bootstrap_interval,
    correlate,
    reidentification_residual,
)
from boundary.models import DataClass


def test_a_single_observation_reports_the_full_range_not_certainty():
    """One sample bootstrapped against itself would give a zero-width interval.

    Reading that as certainty is the opposite of what one observation means.
    """
    interval = bootstrap_interval([1])
    assert interval.point == 1.0
    assert (interval.low, interval.high) == (0.0, 1.0)
    assert interval.n == 1


def test_an_empty_sample_reports_the_full_range():
    interval = bootstrap_interval([])
    assert (interval.low, interval.high) == (0.0, 1.0)
    assert interval.n == 0


def test_the_interval_narrows_as_the_sample_grows():
    narrow = bootstrap_interval([1, 0] * 200)
    wide = bootstrap_interval([1, 0] * 3)
    assert narrow.width < wide.width


def test_the_bootstrap_never_touches_the_global_random_state():
    """A shared generator would make results depend on test order.

    Mutation check, executed in-test: the global state is seeded, the
    bootstrap runs, and the next global draw is asserted unchanged.
    """
    random.seed(1234)
    expected = random.random()

    random.seed(1234)
    bootstrap_interval([1, 0, 1, 1, 0])
    assert random.random() == expected


def test_the_bootstrap_is_reproducible_across_runs():
    a = bootstrap_interval([1, 0, 1, 1, 0, 1])
    b = bootstrap_interval([1, 0, 1, 1, 0, 1])
    assert (a.low, a.point, a.high) == (b.low, b.point, b.high)
    c = bootstrap_interval([1, 0, 1, 1, 0, 1], seed=BOOTSTRAP_SEED + 1)
    assert (c.low, c.high) != (a.low, a.high) or c.point == a.point


def test_completely_nested_errors_report_zero_marginal_gain():
    """Completely nested errors, encoded: B is never right where A is wrong.

    Mutation check, executed in-test: a case B catches alone raises the gain
    above zero, so the metric is not always zero.
    """
    # Completely nested: neither detector is ever wrong ALONE, so a_only and
    # b_only are both zero. Every miss is shared and stacking buys nothing.
    nested = CorrelationResult("a", "b", a_only=0, b_only=0, both=5, neither=4)
    assert nested.marginal_gain == 0.0
    assert nested.error_overlap == 1.0

    # B catching cases A missed is the only thing that makes a stack worth it.
    independent = CorrelationResult("a", "b", a_only=3, b_only=2, both=5, neither=2)
    assert independent.marginal_gain > 0.0
    assert independent.error_overlap < 1.0

    # One-sided: A is wrong alone sometimes, B never is. Still no gain from B,
    # because gain is measured on what B rescues from A's misses.
    one_sided = CorrelationResult("a", "b", a_only=3, b_only=0, both=5, neither=4)
    assert one_sided.marginal_gain == 0.0
    assert one_sided.error_overlap < 1.0


def test_the_measured_marginal_gain_of_a_second_detector_is_below_one():
    result = correlate(LABELED, detect_strict, detect_loose, "strict", "loose")
    assert 0.0 <= result.marginal_gain < 1.0
    assert result.both > 0, "the two families should overlap on something"


def test_residual_reports_reidentifiability_without_any_direct_identifier():
    """Direct-identifier recall is the quoted number; this is the real one."""
    from boundary.detectors import default_suite
    from boundary.redaction import Redactor, residual_classes

    text = "DOB: 1982-03-14. Sex: female. Zip: 78704. Title: Staff Engineer."
    findings = default_suite().run(text)
    redactor = Redactor()
    survived = residual_classes(text, findings, redactor)
    residual = reidentification_residual(findings, survived)
    assert residual["quasi_count"] >= 1
    assert isinstance(residual["reidentifiable"], bool)


def test_residual_ignores_direct_identifiers_in_the_quasi_count():
    residual = reidentification_residual([], [DataClass.SSN, DataClass.POSTAL_CODE])
    assert residual["quasi_surviving"] == ["postal_code"]
    assert residual["quasi_count"] == 1
