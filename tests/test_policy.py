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


def test_restricted_identifiers_never_reach_a_default_or_third_party_destination(policy):
    """No redaction makes an SSN acceptable at a default-terms vendor.

    A vendor under no-training terms is outside this rule: it receives the
    SSN redacted (R5), and the README says so.

    Mutation check, executed in-test: the same text to the internal model is
    allowed, so the denial is the destination and not the content alone.
    """
    verdict = _decide(policy, SSN_TEXT, Destination.VENDOR_DEFAULT)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R2-restricted-class-external"

    internal = _decide(policy, SSN_TEXT, Destination.INTERNAL_MODEL)
    assert internal.action is Action.ALLOW

    no_train = _decide(policy, SSN_TEXT, Destination.VENDOR_NO_TRAIN)
    assert no_train.action is Action.REDACT
    assert no_train.rule_id == "R5-no-train-vendor-redact"


def test_a_direct_identifier_outside_never_external_is_denied_by_r3(policy):
    """R3 is the rule that stops a direct identifier at a default-terms vendor.

    An email is direct but not restricted, so R2 does not apply and R3 is the
    only rule standing between it and a redact-and-send under R6. The rule id
    is asserted, not only the outcome, because the outcome alone survives R3
    being switched off only until it does not.

    Mutation check, executed against the policy: drop R3 and this request
    becomes `redact / R6-vendor-default-redact` with the text leaving.
    """
    verdict = _decide(policy, "Contact jrivera@acme.example", Destination.VENDOR_DEFAULT)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R3-vendor-default-direct-id"

    no_train = _decide(policy, "Contact jrivera@acme.example", Destination.VENDOR_NO_TRAIN)
    assert no_train.action is Action.REDACT
    assert no_train.rule_id == "R5-no-train-vendor-redact"


def test_every_never_external_class_is_denied_by_r2_specifically(policy):
    """NEVER_EXTERNAL membership is pinned by the rule that fires, per class.

    Each restricted class is denied at a default-terms vendor by R2 and not by
    a later rule that happens to deny direct identifiers too, so removing a
    class from the set is visible even though R3 would still deny it.

    Mutation check, executed against the policy: remove CREDIT_CARD from
    NEVER_EXTERNAL and the card sample reports R3, which this catches.
    """
    samples = {
        DataClass.SSN: "SSN 482-19-7734 on file.",
        DataClass.MRN: "MRN 4471902 admitted Tuesday.",
        DataClass.CREDIT_CARD: "Card 4111 1111 1111 1111 expires soon.",
        DataClass.ACCOUNT_NUMBER: "ACCT-88120455 was reconciled.",
    }
    assert NEVER_EXTERNAL == frozenset(samples)
    for data_class, text in samples.items():
        findings = default_suite().run(text)
        assert data_class in {f.data_class for f in findings}, data_class
        verdict = _decide(policy, text, Destination.VENDOR_DEFAULT)
        assert verdict.action is Action.DENY, data_class
        assert verdict.rule_id == "R2-restricted-class-external", data_class


def test_a_third_party_tool_receives_no_detected_personal_data(policy):
    verdict = _decide(policy, "Contact jrivera@acme.example", Destination.THIRD_PARTY_TOOL)
    assert verdict.action is Action.DENY
    assert verdict.rule_id == "R1-third-party-any-pii"


def test_an_undetected_name_reaches_a_third_party_tool(policy):
    """The documented limit of R1, pinned: it denies what was DETECTED, and a
    name that is not on the roster is not detected, so it goes out unchanged.
    If this starts failing, the destination is denied outright and the
    README's R1 row can say "no personal data at all" again."""
    verdict = _decide(policy, "Please look up Ilya Sorokin's payroll history.",
                      Destination.THIRD_PARTY_TOOL)
    assert verdict.action is Action.ALLOW


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
