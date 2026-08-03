"""Egress policy: what may leave, decided before anything is rewritten."""
from __future__ import annotations

import pytest

from boundary.detectors import default_suite
from boundary.models import Action, DataClass, Destination, EgressRequest
from boundary.policy import (
    NEVER_EXTERNAL,
    QUASI_THRESHOLD,
    PolicyEngine,
    Rule,
    sensitivity_of,
)
from boundary.models import Sensitivity

SSN_TEXT = "SSN 482-19-7734 on file."
CLEAN_TEXT = "The quarterly report is attached."


def _decide(policy, text, destination):
    findings = default_suite().run(text)
    return policy.decide(
        EgressRequest(tenant_id="acme", text=text, destination=destination), findings
    )


def test_restricted_identifiers_never_reach_an_external_vendor(policy):
    """No redaction makes an SSN acceptable at a default-terms vendor.

    Mutation check, executed in-test: the same text to the internal model is
    allowed, so the denial is the destination and not the content alone.
    """
    verdict = _decide(policy, SSN_TEXT, Destination.VENDOR_DEFAULT)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R2-restricted-class-external"

    internal = _decide(policy, SSN_TEXT, Destination.INTERNAL_MODEL)
    assert internal.action is Action.ALLOW


def test_a_third_party_tool_receives_no_personal_data_at_all(policy):
    verdict = _decide(policy, "Contact jrivera@acme.example", Destination.THIRD_PARTY_TOOL)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R1-third-party-any-pii"


def test_clean_text_is_allowed_everywhere(policy):
    for dest in Destination:
        verdict = _decide(policy, CLEAN_TEXT, dest)
        assert verdict.action is Action.ALLOW, dest


def test_a_mosaic_of_quasi_identifiers_is_denied_without_any_direct_one(policy):
    """Three quasi-identifiers single out a person with no direct id present.

    Mutation check, executed in-test: dropping to two quasi-identifiers is
    shown to produce a different action, so the threshold is doing the work.
    """
    three = "DOB: 1982-03-14. Sex: female. Zip: 78704."
    verdict = _decide(policy, three, Destination.VENDOR_DEFAULT)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R4-quasi-identifier-mosaic"

    two = "DOB: 1982-03-14. Zip: 78704."
    assert _decide(policy, two, Destination.VENDOR_DEFAULT).action is not Action.DENY


def test_the_policy_fails_closed_when_no_rule_matches():
    """An egress policy whose default is "send it" is not a boundary.

    Mutation check, executed in-test: an engine with the real rules allows the
    same request, so the denial comes from the empty rule set.
    """
    empty = PolicyEngine(rules=())
    verdict = _decide(empty, CLEAN_TEXT, Destination.INTERNAL_MODEL)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R0-no-matching-rule"

    assert _decide(PolicyEngine(), CLEAN_TEXT, Destination.INTERNAL_MODEL).action is Action.ALLOW


def test_first_matching_rule_wins_and_is_named(policy):
    """The verdict names the rule so an audit entry can be acted on."""
    verdict = _decide(policy, SSN_TEXT, Destination.THIRD_PARTY_TOOL)
    assert verdict.rule_id == "R1-third-party-any-pii"
    assert verdict.reason


def test_rule_order_is_load_bearing():
    """Reordering the rules changes the decision, so order is fixed."""
    engine = PolicyEngine()
    reversed_engine = PolicyEngine(rules=tuple(reversed(engine.rules)))
    a = _decide(engine, SSN_TEXT, Destination.THIRD_PARTY_TOOL)
    b = _decide(reversed_engine, SSN_TEXT, Destination.THIRD_PARTY_TOOL)
    assert a.rule_id != b.rule_id


def test_sensitivity_reflects_the_worst_class_present():
    suite = default_suite()
    assert sensitivity_of(suite.run(SSN_TEXT)) is Sensitivity.RESTRICTED
    assert sensitivity_of(suite.run("Zip: 78704.")) is Sensitivity.INTERNAL
    assert sensitivity_of(suite.run(CLEAN_TEXT)) is Sensitivity.PUBLIC


def test_every_never_external_class_is_a_direct_identifier():
    """A quasi-identifier in the never-external set would be a modeling error."""
    assert all(c.is_direct for c in NEVER_EXTERNAL)


def test_the_quasi_threshold_is_documented_and_used():
    assert QUASI_THRESHOLD == 3
