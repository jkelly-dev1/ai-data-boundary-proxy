"""What may leave the boundary, decided before anything is redacted.

The ordering is the design. Policy runs on the DETECTED CLASSES, not on the
redacted text, and it can DENY outright. A proxy that redacts first and asks
questions later has already decided that everything is sendable as long as it
has been rewritten, which is the wrong default for restricted data: some
material must not leave at all, in any form, and no redaction makes it
acceptable.

So the pipeline is detect -> decide -> (redact) -> forward. A DENY short
circuits before any text is prepared for transmission.

Rules are evaluated in declared order and the FIRST MATCH WINS, which makes the
policy readable top to bottom and makes the winning rule reportable. The
verdict carries `rule_id` so an audit entry can name the rule that produced it;
"the policy blocked it" is not an answer anyone can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from boundary.models import (
    Action,
    DataClass,
    Destination,
    EgressRequest,
    Finding,
    PolicyVerdict,
    Sensitivity,
)


@dataclass(frozen=True)
class Rule:
    """One policy rule.

    `applies` is a predicate over (request, findings) so a rule can reason
    about the request's destination and the data classes present at the same
    time. That combination is the case this handles: an SSN to an internal
    model and an SSN to an arbitrary third-party tool are different decisions.
    """

    rule_id: str
    action: Action
    reason: str
    applies: Callable[[EgressRequest, Sequence[Finding]], bool]


def _classes(findings: Sequence[Finding]) -> frozenset[DataClass]:
    return frozenset(f.data_class for f in findings)


def _has_direct(findings: Sequence[Finding]) -> bool:
    return any(f.data_class.is_direct for f in findings)


def _quasi_count(findings: Sequence[Finding]) -> int:
    return len({f.data_class for f in findings if f.data_class.is_quasi})


# Classes that must never leave the boundary toward an uncontrolled
# destination, whatever the redaction strategy would have done with them.
NEVER_EXTERNAL = frozenset(
    {DataClass.SSN, DataClass.MRN, DataClass.CREDIT_CARD, DataClass.ACCOUNT_NUMBER}
)

# How many distinct quasi-identifiers constitute a re-identification risk on
# their own. Three is the classic combination, date of birth, postal code and
# sex, which is enough to single out a large share of a population even with
# every direct identifier removed.
QUASI_THRESHOLD = 3


def default_rules() -> tuple[Rule, ...]:
    """The shipped policy, ordered most restrictive first."""
    return (
        Rule(
            rule_id="R1-third-party-any-pii",
            action=Action.DENY,
            reason=(
                "a third-party tool destination receives no detected personal "
                "data in any form"
            ),
            applies=lambda req, f: (
                req.destination is Destination.THIRD_PARTY_TOOL and bool(f)
            ),
        ),
        Rule(
            rule_id="R2-restricted-class-external",
            action=Action.DENY,
            reason=(
                "restricted identifiers do not leave the boundary toward an "
                "external vendor, redacted or not"
            ),
            applies=lambda req, f: (
                req.destination
                in (Destination.VENDOR_DEFAULT, Destination.THIRD_PARTY_TOOL)
                and bool(_classes(f) & NEVER_EXTERNAL)
            ),
        ),
        Rule(
            rule_id="R3-vendor-default-direct-id",
            action=Action.DENY,
            reason=(
                "a vendor on default terms may train on submitted content, so "
                "direct identifiers are not sent"
            ),
            applies=lambda req, f: (
                req.destination is Destination.VENDOR_DEFAULT and _has_direct(f)
            ),
        ),
        Rule(
            rule_id="R4-quasi-identifier-mosaic",
            action=Action.DENY,
            reason=(
                f"{QUASI_THRESHOLD} or more quasi-identifiers together permit "
                "re-identification even with direct identifiers removed"
            ),
            applies=lambda req, f: (
                req.destination
                in (Destination.VENDOR_DEFAULT, Destination.THIRD_PARTY_TOOL)
                and _quasi_count(f) >= QUASI_THRESHOLD
            ),
        ),
        Rule(
            rule_id="R5-no-train-vendor-redact",
            action=Action.REDACT,
            reason="external vendor under no-training terms receives redacted text",
            applies=lambda req, f: (
                req.destination is Destination.VENDOR_NO_TRAIN and bool(f)
            ),
        ),
        Rule(
            rule_id="R6-vendor-default-redact",
            action=Action.REDACT,
            reason="external vendor receives redacted text",
            applies=lambda req, f: (
                req.destination is Destination.VENDOR_DEFAULT and bool(f)
            ),
        ),
        Rule(
            rule_id="R7-internal-allow",
            action=Action.ALLOW,
            reason="internal model inside the boundary receives the text unchanged",
            applies=lambda req, f: req.destination is Destination.INTERNAL_MODEL,
        ),
        Rule(
            rule_id="R8-default-allow-clean",
            action=Action.ALLOW,
            reason="no personal data detected",
            applies=lambda req, f: not f,
        ),
    )


@dataclass
class PolicyEngine:
    """First-match-wins evaluation over an ordered rule list."""

    rules: tuple[Rule, ...] = field(default_factory=default_rules)

    def decide(
        self, request: EgressRequest, findings: Sequence[Finding]
    ) -> PolicyVerdict:
        for rule in self.rules:
            if rule.applies(request, findings):
                return PolicyVerdict(
                    action=rule.action,
                    rule_id=rule.rule_id,
                    reason=rule.reason,
                    findings=tuple(findings),
                )
        # A request that matches nothing is DENIED, not allowed. An egress
        # policy whose default is "send it" fails open, and a boundary that
        # fails open is not a boundary. Reaching this line means the rule set
        # has a gap, and the rule_id says so rather than pretending the denial
        # was intentional.
        return PolicyVerdict(
            action=Action.DENY,
            rule_id="R0-no-matching-rule",
            reason="no rule matched; egress policy fails closed",
            findings=tuple(findings),
        )

    def explain(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (r.rule_id, r.action.value, r.reason) for r in self.rules
        )


def sensitivity_of(findings: Iterable[Finding]) -> Sensitivity:
    """The tier implied by the classes present.

    Used for reporting rather than for decisions. The rules above reason about
    specific classes because "restricted" is too coarse to act on: an SSN and a
    job title can both appear in a restricted document and they are not the
    same egress question.
    """
    classes = {f.data_class for f in findings}
    if classes & NEVER_EXTERNAL:
        return Sensitivity.RESTRICTED
    if any(c.is_direct for c in classes):
        return Sensitivity.CONFIDENTIAL
    if classes:
        return Sensitivity.INTERNAL
    return Sensitivity.PUBLIC
