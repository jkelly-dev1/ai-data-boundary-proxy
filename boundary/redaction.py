"""Rewriting text on its way out of the boundary.

The interesting content here is not the string substitution, it is the two
properties a redactor can silently fail to have:

  REVERSIBILITY. A redactor that produces a stable token for a given input
  keeps the recipient able to LINK requests: they cannot read the SSN, but they
  can tell that request 4 and request 91 concern the same person, and they can
  count how many distinct people you have. That is a disclosure, and
  `PSEUDONYM` is implemented here specifically so the demo can measure it
  instead of describing it.

  Idempotence on re-redaction. Redacting already-redacted text must not corrupt
  it. Spans are applied right-to-left for that reason: rewriting left-to-right
  invalidates every offset after the first substitution, and the resulting
  corruption looks like a redaction that worked.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from boundary.models import DataClass, Finding, RedactionStrategy

# A per-class default. RESTRICTED-tier classes get MASK because a partial
# reveal of an SSN is still a reveal; contact details get PARTIAL because
# operators need to recognize the record they are looking at.
DEFAULT_STRATEGY: Mapping[DataClass, RedactionStrategy] = {
    DataClass.SSN: RedactionStrategy.MASK,
    DataClass.CREDIT_CARD: RedactionStrategy.PARTIAL,
    DataClass.EMAIL: RedactionStrategy.PSEUDONYM,
    DataClass.PHONE: RedactionStrategy.PARTIAL,
    DataClass.MRN: RedactionStrategy.MASK,
    DataClass.ACCOUNT_NUMBER: RedactionStrategy.MASK,
    DataClass.PERSON_NAME: RedactionStrategy.PSEUDONYM,
    DataClass.IP_ADDRESS: RedactionStrategy.MASK,
    DataClass.DATE_OF_BIRTH: RedactionStrategy.MASK,
    DataClass.POSTAL_CODE: RedactionStrategy.PARTIAL,
    DataClass.SEX: RedactionStrategy.MASK,
    DataClass.JOB_TITLE: RedactionStrategy.NONE,
    DataClass.EMPLOYER: RedactionStrategy.NONE,
}


@dataclass
class RedactionResult:
    """Redacted text plus what it took to get there."""

    text: str
    applied: tuple[Finding, ...]
    tokens: Mapping[str, str]

    @property
    def count(self) -> int:
        return len(self.applied)


class Pseudonymizer:
    """Stable HMAC tokens for redacted values.

    This is pseudonymization, not anonymization. Given the same key, the same
    input always produces the same token, so:

      - the recipient can link every request about one subject,
      - the recipient can count distinct subjects,
      - and anyone who obtains the key and a candidate value can CONFIRM the
        value by recomputing the token, because the input space of an SSN is
        only 10^9 and of an email is a dictionary.

    A rotating salt breaks linkage across rotations but also breaks the
    usability that motivated pseudonymization in the first place. That tension
    is real and the demo reports both sides rather than picking a winner.
    """

    def __init__(self, key: bytes, salt: str = ""):
        if not key:
            raise ValueError("pseudonymizer requires a non-empty key")
        self.key = key
        self.salt = salt

    def token(self, data_class: DataClass, value: str) -> str:
        msg = f"{self.salt}|{data_class.value}|{value.strip().lower()}".encode("utf-8")
        digest = hmac.new(self.key, msg, hashlib.sha256).hexdigest()[:10]
        return f"[{data_class.value.upper()}_{digest}]"

    def links(self, a: str, b: str, data_class: DataClass) -> bool:
        """True when two raw values would produce the same token."""
        return self.token(data_class, a) == self.token(data_class, b)


def _mask(data_class: DataClass) -> str:
    return f"[REDACTED:{data_class.value}]"


def _partial(data_class: DataClass, value: str) -> str:
    """Keep a recognizable tail, hide the rest.

    Deliberately conservative: 4 characters for a card (the industry norm) and
    2 for anything else. Even this leaks, a ZIP+4 reduced to its last two
    digits still narrows a population, so the residual re-identification
    measurement counts PARTIAL spans as surviving quasi identifiers rather
    than as removed.
    """
    digits = re.sub(r"\D", "", value)
    keep = 4 if data_class is DataClass.CREDIT_CARD else 2
    if len(digits) <= keep:
        return _mask(data_class)
    return "*" * (len(digits) - keep) + digits[-keep:]


class Redactor:
    """Applies a strategy per data class over a set of findings."""

    def __init__(
        self,
        pseudonymizer: Optional[Pseudonymizer] = None,
        strategy: Optional[Mapping[DataClass, RedactionStrategy]] = None,
        override: Optional[RedactionStrategy] = None,
    ):
        self.pseudonymizer = pseudonymizer
        self.strategy = dict(strategy or DEFAULT_STRATEGY)
        self.override = override

    def strategy_for(self, data_class: DataClass) -> RedactionStrategy:
        if self.override is not None:
            return self.override
        return self.strategy.get(data_class, RedactionStrategy.MASK)

    def _replacement(self, finding: Finding, value: str) -> tuple[str, Optional[str]]:
        strat = self.strategy_for(finding.data_class)
        if strat is RedactionStrategy.NONE:
            return value, None
        if strat is RedactionStrategy.DROP:
            return "", None
        if strat is RedactionStrategy.PARTIAL:
            return _partial(finding.data_class, value), None
        if strat is RedactionStrategy.PSEUDONYM:
            if self.pseudonymizer is None:
                return _mask(finding.data_class), None
            tok = self.pseudonymizer.token(finding.data_class, value)
            return tok, tok
        return _mask(finding.data_class), None

    def apply(self, text: str, findings: Iterable[Finding]) -> RedactionResult:
        """Rewrite `text`, replacing each finding's span.

        Spans are applied in DESCENDING start order. Applying them forward
        would shift every later offset by the length delta of each replacement,
        so the second substitution would land in the wrong place and the third
        would land inside the second. The failure is silent, the output still
        looks redacted, so it has its own test.
        """
        ordered = sorted(findings, key=lambda f: f.span.start, reverse=True)
        out = text
        applied: list[Finding] = []
        tokens: dict[str, str] = {}
        for f in ordered:
            value = text[f.span.start:f.span.end]
            replacement, token = self._replacement(f, value)
            if replacement == value:
                continue
            out = out[: f.span.start] + replacement + out[f.span.end:]
            applied.append(f)
            if token:
                tokens[token] = value
        applied.reverse()
        return RedactionResult(text=out, applied=tuple(applied), tokens=tokens)


def residual_classes(
    text: str, findings: Iterable[Finding], redactor: Redactor
) -> tuple[DataClass, ...]:
    """Which data classes SURVIVE redaction in the outgoing text.

    Counts both classes the redactor was told to leave alone (strategy NONE)
    and classes reduced to a partial value. This is the input to the
    re-identification measurement: a record with DOB, ZIP and sex intact is
    identifying for a large fraction of a population even with every direct
    identifier removed, a result that goes back to Sweeney's work on the 1990
    census and is the reason quasi-identifiers are modeled separately here.
    """
    survived: set[DataClass] = set()
    for f in findings:
        strat = redactor.strategy_for(f.data_class)
        if strat in (RedactionStrategy.NONE, RedactionStrategy.PARTIAL):
            survived.add(f.data_class)
    return tuple(sorted(survived, key=lambda c: c.value))


def quasi_identifier_set(findings: Iterable[Finding]) -> frozenset[DataClass]:
    """The quasi-identifiers present, regardless of redaction."""
    return frozenset(f.data_class for f in findings if f.data_class.is_quasi)
