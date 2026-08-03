"""Measuring the instrument, not just the system.

Three things are computed here, and each exists because a sibling project paid
to learn that the naive version misleads.

Bootstrap confidence intervals. A recall of 0.86 over 18 labeled cases is not
meaningfully different from 0.79. Reporting the point estimate alone invites a
comparison the sample size does not support, so every rate here carries an
interval computed from a SEEDED generator that never touches the global random
state.

Detector correlation. The tempting arithmetic is that two detectors at 0.8
recall give 0.96 together. That is only true if their errors are independent,
and detectors built on the same idea fail on the same inputs. This module
reports the pairwise agreement on ERRORS and, more usefully, how much recall a
second detector ADDS over the first. Ensembles of judges built on a shared
design exhibit the same failure: where errors are completely nested and no
member is ever wrong alone, stacking buys nothing while appearing to buy
redundancy.

Re-identification residual. Recall on direct identifiers is the number people
quote. It is not the number that matters. A record with every direct identifier
removed and date of birth, postal code and sex intact is still identifying for
a large share of a population, so the residual quasi-identifier count is
reported beside recall rather than underneath it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from boundary.models import DataClass, Finding

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260731


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] n={self.n}"


def bootstrap_interval(
    outcomes: Sequence[int],
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap over binary outcomes.

    With fewer than two observations the FULL range [0, 1] is returned rather
    than a tight interval around a single point. A one-sample bootstrap
    resamples the same value every time and produces a zero-width interval,
    which reads as certainty and is the opposite of what one observation means.
    """
    n = len(outcomes)
    if n == 0:
        return Interval(0.0, 0.0, 1.0, 0)
    point = sum(outcomes) / n
    if n < 2:
        return Interval(point, 0.0, 1.0, n)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            total += outcomes[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2) * samples)]
    hi = means[min(samples - 1, int((1 - alpha / 2) * samples))]
    return Interval(point, lo, hi, n)


@dataclass
class DetectorScore:
    detector: str
    recall: Interval
    precision: Interval
    true_positives: int
    false_positives: int
    false_negatives: int


def score_detector(
    detector: str,
    cases: Sequence[tuple[str, tuple[str, ...]]],
    run: "callable",
) -> DetectorScore:
    """Score one detector against hand-labeled cases.

    Scored at the level of (case, data class) rather than exact span, because
    the question the policy engine asks is "is there an SSN in this text", not
    "does the span start at offset 14". Span accuracy matters for redaction and
    is tested separately.
    """
    tp = fp = fn = 0
    recall_outcomes: list[int] = []
    precision_outcomes: list[int] = []
    for text, expected in cases:
        want = set(expected)
        got = {f.data_class.value for f in run(text)}
        for cls in want:
            hit = 1 if cls in got else 0
            recall_outcomes.append(hit)
            tp += hit
            fn += 1 - hit
        for cls in got:
            hit = 1 if cls in want else 0
            precision_outcomes.append(hit)
            if not hit:
                fp += 1
    return DetectorScore(
        detector=detector,
        recall=bootstrap_interval(recall_outcomes),
        precision=bootstrap_interval(precision_outcomes),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


@dataclass
class CorrelationResult:
    """How much a second detector adds over a first."""

    detector_a: str
    detector_b: str
    a_only: int
    b_only: int
    both: int
    neither: int

    @property
    def marginal_gain(self) -> float:
        """Fraction of A's misses that B catches.

        The honest number. If B catches nothing A missed, this is 0.0 and the
        second detector is latency with extra steps, whatever its standalone
        recall says.
        """
        a_missed = self.b_only + self.neither
        if a_missed == 0:
            return 0.0
        return self.b_only / a_missed

    @property
    def error_overlap(self) -> float:
        """Fraction of all misses on which BOTH detectors failed.

        A value of 1.0 means completely nested errors: the detectors
        never fail independently, and stacking them provides no
        redundancy at all.
        """
        total_missed = self.a_only + self.b_only + self.neither
        if total_missed == 0:
            return 0.0
        return self.neither / total_missed


def correlate(
    cases: Sequence[tuple[str, tuple[str, ...]]],
    run_a: "callable",
    run_b: "callable",
    name_a: str,
    name_b: str,
) -> CorrelationResult:
    a_only = b_only = both = neither = 0
    for text, expected in cases:
        got_a = {f.data_class.value for f in run_a(text)}
        got_b = {f.data_class.value for f in run_b(text)}
        for cls in expected:
            in_a, in_b = cls in got_a, cls in got_b
            if in_a and in_b:
                both += 1
            elif in_a:
                a_only += 1
            elif in_b:
                b_only += 1
            else:
                neither += 1
    return CorrelationResult(name_a, name_b, a_only, b_only, both, neither)


def reidentification_residual(
    findings: Iterable[Finding], surviving: Iterable[DataClass]
) -> dict[str, object]:
    """What is still identifying after redaction.

    Reports the surviving quasi-identifier set and whether it reaches the
    mosaic threshold at which a record singles out an individual without any
    direct identifier present.
    """
    from boundary.policy import QUASI_THRESHOLD

    survived = {c for c in surviving if c.is_quasi}
    direct_removed = {f.data_class for f in findings if f.data_class.is_direct}
    return {
        "quasi_surviving": sorted(c.value for c in survived),
        "quasi_count": len(survived),
        "direct_removed": sorted(c.value for c in direct_removed),
        "reidentifiable": len(survived) >= QUASI_THRESHOLD,
    }


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
