"""The gate protects the measurement, so its own failure modes are tested."""
from __future__ import annotations

import pytest

from boundary.config import Settings
from boundary.gate import (
    EXPECTED_ERROR_OVERLAP,
    EXPECTED_MARGINAL_GAIN,
    MAX_RECALL,
    MIN_RECALL,
    MIN_WEAK_LEAKS,
    main,
    run_checks,
)


def test_the_gate_passes_on_the_shipped_corpus():
    results = run_checks()
    failures = [r for r in results if not r.passed]
    assert not failures, [f"{r.name}: {r.detail}" for r in failures]
    assert main() == 0


def test_the_gate_checks_both_directions():
    """It fails on regression AND on everything suddenly passing.

    A suite where nothing can fail has been fitted to its own answers, so the
    gate carries checks that assert imperfection: weak strategies must still
    leak and detector recall must stay below 1.0. The same recall check also
    carries a floor, and the precision check carries a control that must
    count one false positive, so degrading detection is a failure and not a
    move toward the middle of a band.
    """
    names = {r.name for r in run_checks()}
    assert "weak strategies still leak" in names
    assert "detector recall is above the floor and below 1.0" in names
    assert "the misses are exactly the unrostered names" in names
    assert "detector precision is perfect on the labeled set" in names
    assert "full erasure verifies clean" in names


def test_the_gate_runs_on_the_mock_even_with_a_real_provider_configured():
    """A regression gate has to be reproducible.

    Any live model moves the numbers between runs, so real-model behavior
    belongs in SAMPLE_RUN.md and never in an exit code. The gate does not
    consult the provider at all, which this test pins by configuring one fully.
    """
    settings = Settings(
        BOUNDARY_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-not-a-real-key"
    )
    assert settings.effective_provider() == "anthropic"
    # The gate still passes, and it passes because it never asked.
    assert main() == 0


def test_the_thresholds_are_pinned_not_incidental():
    assert MIN_WEAK_LEAKS >= 4
    assert MAX_RECALL < 1.0
    # The band is real: a floor above 10/13 catches the loss of any one
    # class, and it sits below the shipped 11/13.
    assert 10 / 13 < MIN_RECALL < 11 / 13 < MAX_RECALL
    assert EXPECTED_MARGINAL_GAIN == 1 / 6
    assert EXPECTED_ERROR_OVERLAP == 5 / 11


def test_a_provider_without_its_key_falls_back_to_the_mock():
    """A missing key is a configuration state, not an error.

    Mutation check, executed in-test: supplying the key flips the effective
    provider, so the fallback is the missing key and not a hardcoded mock.
    """
    without = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY=None)
    assert without.effective_provider() == "mock"
    assert without.is_mock()

    with_key = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY="sk-not-real")
    assert with_key.effective_provider() == "openai"
    assert not with_key.is_mock()
