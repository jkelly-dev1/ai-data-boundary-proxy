"""PII detectors, and the honest accounting of what they miss.

Three detector FAMILIES are implemented, deliberately overlapping, because the
question this project answers is not "can a regex find an SSN" but "how much
does a second detector add once you already have the first". The answer, which
metrics.py measures rather than asserts, is usually much less than the vendor
math implies: detectors built on the same idea fail on the same inputs.

  strict   -- anchored, low false-positive patterns. What a cautious team ships.
  loose    -- wider patterns that catch more formats and fire more often.
  context  -- looks for a labeling keyword near a weak value ("DOB: 03/14/1982"),
              catching things a value-only pattern cannot and missing the same
              value when it appears unlabeled.

A detector returns FINDINGS, not truth. Every finding carries the detector name
so recall, precision and inter-detector agreement are computable per detector.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from boundary.models import DataClass, Finding, Span

# --------------------------------------------------------------------------
# Primitive validators
# --------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    """Luhn checksum. Turns a 16-digit-number pattern into a card detector.

    Without this, any 16 consecutive digits, an order id, a tracking number, is
    reported as a credit card, and the false-positive rate makes the whole
    redactor untrustworthy. With it, a card number that has been transcribed
    with a typo is MISSED, which is the trade this function makes explicit.
    """
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 12:
        return False
    total = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _invalid_ssn(value: str) -> bool:
    """SSA rules for structurally impossible SSNs.

    Area 000/666/900-999, group 00 and serial 0000 are never issued. Filtering
    them keeps the strict detector from firing on 000-00-0000 placeholders,
    which appear constantly in test fixtures and templates.
    """
    d = re.sub(r"\D", "", value)
    if len(d) != 9:
        return True
    area, group, serial = d[:3], d[3:5], d[5:]
    if area in {"000", "666"} or area[0] == "9":
        return True
    return group == "00" or serial == "0000"


# --------------------------------------------------------------------------
# Pattern tables
# --------------------------------------------------------------------------

_STRICT_PATTERNS: list[tuple[DataClass, re.Pattern, Callable[[str], bool] | None]] = [
    (DataClass.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     lambda v: not _invalid_ssn(v)),
    (DataClass.EMAIL,
     re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    (DataClass.PHONE,
     re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"), None),
    # Separators are allowed only BETWEEN digits, so the span ends on the last
    # digit and the redaction does not swallow the space after the number.
    (DataClass.CREDIT_CARD,
     re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), luhn_ok),
    (DataClass.IP_ADDRESS,
     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
     lambda v: all(0 <= int(p) <= 255 for p in v.split("."))),
    (DataClass.MRN, re.compile(r"\bMRN[-: ]?\d{6,10}\b", re.I), None),
    (DataClass.ACCOUNT_NUMBER,
     re.compile(r"\bACCT[-: ]?\d{6,12}\b", re.I), None),
]

# Looser: catches formats the strict table refuses, at a real false-positive
# cost. Bare nine-digit runs are SSNs often enough to matter and order numbers
# often enough to hurt. Shipping this alone would be irresponsible; measuring
# what it adds on top of strict is the point.
_LOOSE_PATTERNS: list[tuple[DataClass, re.Pattern, Callable[[str], bool] | None]] = [
    (DataClass.SSN, re.compile(r"\b\d{3}[-. ]?\d{2}[-. ]?\d{4}\b"),
     lambda v: not _invalid_ssn(v)),
    (DataClass.PHONE,
     re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), None),
    (DataClass.POSTAL_CODE, re.compile(r"\b\d{5}(?:-\d{4})?\b"), None),
    (DataClass.DATE_OF_BIRTH,
     re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"),
     None),
]

# Context detectors: a label near a weak value. "DOB: 1982-03-14" is findable
# by the label even though the value alone is just a date.
_CONTEXT_RULES: list[tuple[DataClass, re.Pattern]] = [
    (DataClass.DATE_OF_BIRTH,
     re.compile(r"(?:\bDOB\b|date of birth)\s*[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9/.\-]{6,10})", re.I)),
    (DataClass.SEX, re.compile(r"\b(?:sex|gender)\s*[:\-]\s*(male|female|m|f)\b", re.I)),
    (DataClass.POSTAL_CODE,
     re.compile(r"(?:\bzip\b|postal code)\s*[:\-]?\s*(\d{5}(?:-\d{4})?)", re.I)),
    # Case-insensitivity is scoped to the LABEL with (?i:...): real documents
    # write "Title:" and "TITLE:", so a case-sensitive label silently never
    # fires. The captured VALUE stays case-sensitive and must start with a
    # capital, which is what keeps "title: see attached" from being read as
    # a job title and "patient id: pending" from being read as a record
    # number. A pattern-wide re.I would reach the value class too and turn
    # [A-Z] into any letter.
    (DataClass.JOB_TITLE,
     re.compile(r"\b(?i:title|role|position)\s*[:\-]\s*([A-Z][A-Za-z ]{3,30})")),
    (DataClass.EMPLOYER,
     re.compile(r"\b(?i:employer|works at|employed by)\s*[:\-]?\s*([A-Z][A-Za-z ]{2,30})")),
    (DataClass.MRN,
     re.compile(r"(?i:medical record(?: number)?|patient id)\s*[:\-]?\s*([A-Z0-9\-]{5,12})")),
]


def _scan(text: str, table, detector: str) -> list[Finding]:
    out: list[Finding] = []
    for data_class, pattern, validator in table:
        for m in pattern.finditer(text):
            value = m.group(0)
            if validator is not None and not validator(value):
                continue
            out.append(
                Finding(
                    data_class=data_class,
                    span=Span(m.start(), m.end()),
                    detector=detector,
                    text=value,
                )
            )
    return out


def detect_strict(text: str) -> list[Finding]:
    """Anchored, conservative patterns. High precision, misses variants."""
    return _scan(text, _STRICT_PATTERNS, "strict")


def detect_loose(text: str) -> list[Finding]:
    """Wider patterns. Catches more formats, fires on more innocent text."""
    return _scan(text, _LOOSE_PATTERNS, "loose")


def detect_context(text: str) -> list[Finding]:
    """Label-driven detection for values that are weak on their own."""
    out: list[Finding] = []
    for data_class, pattern in _CONTEXT_RULES:
        for m in pattern.finditer(text):
            if m.lastindex:
                start, end = m.start(m.lastindex), m.end(m.lastindex)
            else:
                start, end = m.start(), m.end()
            out.append(
                Finding(
                    data_class=data_class,
                    span=Span(start, end),
                    detector="context",
                    text=text[start:end],
                )
            )
    return out


class NameDetector:
    """Roster-based person-name detection.

    This is the honest weak point of the whole detector suite and it is
    implemented this way ON PURPOSE. Real name detection needs a model; a
    roster only finds the names it was given. The corpus therefore contains
    names that are NOT on the roster, and the measured recall for PERSON_NAME
    is well under 1.0 as a result. That number is a headline finding, not a
    defect to be tuned away: a boundary that depends on catching every name is
    a boundary that will leak, and the fix is policy (deny the destination),
    not a better regex.
    """

    def __init__(self, roster: Iterable[str]):
        self.roster = tuple(sorted({r for r in roster if r.strip()}, key=len, reverse=True))
        self._patterns = [
            (name, re.compile(r"\b" + re.escape(name) + r"\b", re.I))
            for name in self.roster
        ]

    def __call__(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        for name, pattern in self._patterns:
            for m in pattern.finditer(text):
                out.append(
                    Finding(
                        data_class=DataClass.PERSON_NAME,
                        span=Span(m.start(), m.end()),
                        detector="roster",
                        text=m.group(0),
                    )
                )
        return out


def dedupe(findings: Iterable[Finding]) -> list[Finding]:
    """Collapse identical claims from different detectors, keeping the first.

    Two detectors reporting the same class at the same span is ONE fact about
    the text, not two. Counting it twice inflates apparent coverage, which is
    precisely the arithmetic error that makes stacked detectors look additive
    when they are not.
    """
    seen: set[tuple] = set()
    out: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.span.start, -len(f.span))):
        if f.key() in seen:
            continue
        seen.add(f.key())
        out.append(f)
    return out


def merge_overlaps(findings: Iterable[Finding]) -> list[Finding]:
    """Drop findings fully contained inside a longer finding of the same class,
    and keep ONE finding where two classes claim the identical span.

    A loose phone pattern and a strict one can match nested spans of the same
    number. Redacting both would corrupt the text by replacing part of an
    already-replaced span, so the longer span wins.

    Two classes can also claim exactly the same characters: "78704-1234" is a
    ZIP+4 to the postal pattern and a nine-digit SSN to the loose one.
    Redacting both rewrites the second replacement into the first and leaves
    debris. One claim is kept, and a direct identifier is preferred over a
    quasi-identifier because it is the stricter of the two redactions.
    """
    ordered = sorted(
        findings,
        key=lambda f: (
            f.span.start,
            -len(f.span),
            not f.data_class.is_direct,
            f.data_class.value,
        ),
    )
    kept: list[Finding] = []
    for f in ordered:
        if any(
            k.data_class == f.data_class
            and k.span.start <= f.span.start
            and f.span.end <= k.span.end
            for k in kept
        ):
            continue
        if any(k.span == f.span for k in kept):
            continue
        kept.append(f)
    return kept


class DetectorSuite:
    """A named bundle of detectors, run together.

    Kept as an explicit object so a run can be configured with a SUBSET,
    strict only, strict+loose, everything, and the measured difference
    between those configurations is the experiment.
    """

    def __init__(self, detectors: dict[str, Callable[[str], list[Finding]]]):
        self.detectors = dict(detectors)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.detectors))

    def run_each(self, text: str) -> dict[str, list[Finding]]:
        """Per-detector findings, unmerged. The input to correlation analysis."""
        return {name: fn(text) for name, fn in self.detectors.items()}

    def run(self, text: str) -> list[Finding]:
        """Everything, deduped and overlap-merged. The input to redaction."""
        all_findings: list[Finding] = []
        for fn in self.detectors.values():
            all_findings.extend(fn(text))
        return merge_overlaps(dedupe(all_findings))


def default_suite(roster: Iterable[str] = ()) -> DetectorSuite:
    """The suite the demo and the gate use."""
    detectors: dict[str, Callable[[str], list[Finding]]] = {
        "strict": detect_strict,
        "loose": detect_loose,
        "context": detect_context,
    }
    roster = tuple(roster)
    if roster:
        detectors["roster"] = NameDetector(roster)
    return DetectorSuite(detectors)
