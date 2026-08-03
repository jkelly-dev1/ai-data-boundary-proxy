"""The CI gate.

It protects the MEASUREMENT, not a product, and it fails in both directions.

Failing when erasure regresses is obvious. Failing when EVERYTHING PASSES is
the part that matters: if the weak erasure strategies stop leaking, or the
detector suite reaches perfect recall, or the neighbor probe stops separating
its two populations, then the corpus has stopped discriminating and every
subsequent green run is meaningless. A suite where nothing can fail is a suite
that has been fitted to its own answers.

The gate always runs against the deterministic mock. A regression gate has to
be reproducible, and a live model moves the numbers between runs, so real-model
behavior is reported in SAMPLE_RUN.md and never reaches an exit code.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from boundary.corpus import LABELED, ROSTER, build_corpus
from boundary.detectors import default_suite, detect_loose, detect_strict
from boundary.embeddings import default_embedder
from boundary.erasure import (
    PLAINTEXT_STRATEGIES,
    ErasureEngine,
    ErasureStrategy,
    build_index,
    rechunk_and_reindex,
)
from boundary.metrics import correlate, score_detector
from boundary.verify import ErasureVerifier, separation_band, NEIGHBOR_THRESHOLD

# The weak strategies must still leak by at least this much, or the corpus has
# stopped being able to tell a thorough erasure from a careless one.
MIN_WEAK_LEAKS = 4
# Overall detector recall must stay strictly below 1.0. Perfect recall on this
# corpus means the unrostered names were quietly added to the roster, which
# would delete the project's most honest finding.
MAX_RECALL = 0.999


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def run_checks() -> list[GateResult]:
    results: list[GateResult] = []
    emb = default_embedder()

    # 1. The four strategies must remain ordered by thoroughness.
    leaks: dict[str, int] = {}
    after: dict[str, int] = {}
    for strat in PLAINTEXT_STRATEGIES:
        c = build_corpus()
        idx = build_index(c.store, emb)
        subj = c.subjects[0]
        ErasureEngine(store=c.store, index=idx, embedder=emb).erase(subj, strat)
        v = ErasureVerifier(c.store, idx, emb)
        leaks[strat.value] = len(v.verify(subj).leaking_probes)
        rechunk_and_reindex(c.store, idx, emb)
        after[strat.value] = len(v.verify(subj).leaking_probes)

    ordered = (
        leaks["source_only"]
        >= leaks["source_and_index"]
        >= leaks["mention_sweep"]
        >= leaks["full"]
    )
    results.append(
        GateResult(
            "erasure strategies ordered by thoroughness",
            ordered,
            " >= ".join(f"{k}:{v}" for k, v in leaks.items()),
        )
    )

    results.append(
        GateResult(
            "full erasure verifies clean",
            leaks["full"] == 0,
            f"{leaks['full']} leaking probes after FULL",
        )
    )

    results.append(
        GateResult(
            "weak strategies still leak",
            leaks["source_only"] >= MIN_WEAK_LEAKS,
            f"source_only leaks {leaks['source_only']} (floor {MIN_WEAK_LEAKS})",
        )
    )

    results.append(
        GateResult(
            "incomplete erasure is undone by a re-ingest",
            after["mention_sweep"] > leaks["mention_sweep"],
            f"mention_sweep {leaks['mention_sweep']} -> {after['mention_sweep']} after re-ingest",
        )
    )

    results.append(
        GateResult(
            "complete erasure survives a re-ingest",
            after["full"] == 0,
            f"full {leaks['full']} -> {after['full']} after re-ingest",
        )
    )

    # 1b. The crypto posture. Every assertion here was written AFTER the first
    # measurement, not before it. A gate written to a prediction encodes the
    # prediction: P2 said sealing the vectors would close the neighbor probe,
    # it does not, and a gate asserting it would have gone red on the first run
    # and invited someone to "fix" the code until it agreed.
    crypto = _crypto_checks(emb)
    results.extend(crypto)

    # 2. The neighbor probe must still separate its two populations.
    c = build_corpus()
    idx = build_index(c.store, emb)
    band = separation_band(idx, emb, c.subjects[0])
    separated = band["max_without"] < NEIGHBOR_THRESHOLD < band["min_with"]
    results.append(
        GateResult(
            "neighbor threshold sits between the populations",
            separated,
            f"{band['max_without']:.3f} < {NEIGHBOR_THRESHOLD} < {band['min_with']:.3f}",
        )
    )

    # 3. Detection must remain imperfect and honest about it.
    suite = default_suite(ROSTER)
    overall = score_detector("all", LABELED, suite.run)
    results.append(
        GateResult(
            "detector recall is below 1.0",
            overall.recall.point < MAX_RECALL,
            f"recall {overall.recall}",
        )
    )
    results.append(
        GateResult(
            "detector precision is perfect on the labeled set",
            overall.false_positives == 0,
            f"{overall.false_positives} false positives",
        )
    )

    # 4. Stacked detectors must not be assumed independent.
    corr = correlate(LABELED, detect_strict, detect_loose, "strict", "loose")
    results.append(
        GateResult(
            "second detector's marginal gain is reported, not assumed",
            corr.marginal_gain < 1.0,
            f"loose adds {corr.marginal_gain:.3f} over strict; "
            f"error overlap {corr.error_overlap:.3f}",
        )
    )

    return results



def _crypto_checks(emb) -> list[GateResult]:
    """The sealed posture, protected the same way the plaintext one is.

    These fail in both directions like everything else in this file. Two of
    them assert that crypto-shredding DOES NOT WORK in a specific way, because
    those are the project's findings and a corpus that stopped showing them
    would have stopped measuring anything.
    """
    from boundary.crypto import KeyStore, rechunk_and_reindex_sealed, seal_store

    def sealed(snapshot=None, vectors=False):
        c = build_corpus()
        idx = build_index(c.store, emb)
        keys = KeyStore()
        sc = seal_store(c.store, idx, keys, c.subjects, seal_vectors=vectors)
        if snapshot:
            keys.snapshot(snapshot)
        return c, idx, keys, sc

    def shred(strategy, snapshot=None, vectors=False):
        c, idx, keys, sc = sealed(snapshot, vectors)
        subj = c.subjects[0]
        ErasureEngine(
            store=c.store, index=idx, embedder=emb, keys=keys, sealed=sc
        ).erase(subj, strategy)
        v = ErasureVerifier(c.store, idx, emb, keys=keys, sealed=sc)
        return c, idx, keys, sc, subj, v, v.verify(subj)

    out: list[GateResult] = []

    # Sealing must not be mistakable for erasure. An encrypted store whose keys
    # are all live is fully readable, and the probes must say so.
    c, idx, keys, sc = sealed()
    live_report = ErasureVerifier(
        c.store, idx, emb, keys=keys, sealed=sc
    ).verify(c.subjects[0])
    out.append(
        GateResult(
            "an encrypted store with live keys is not erased",
            live_report.leaked,
            f"{len(live_report.leaking_probes)} probes still reach the subject",
        )
    )

    # The headline finding: the derived vector outlives the key.
    _, _, _, _, _, _, text_report = shred(ErasureStrategy.CRYPTO_SHRED_TEXT)
    neighbor = {p.probe: p.leaked for p in text_report.probes}["neighbor_probe"]
    out.append(
        GateResult(
            "crypto_shred_text leaves the derived vector reachable",
            neighbor,
            "neighbor_probe still finds vectors after the key is destroyed",
        )
    )

    # Variant B closes it for subject-only vectors and NOT for shared ones.
    _, _, _, sc_full, _, _, full_report = shred(
        ErasureStrategy.CRYPTO_SHRED_FULL, vectors=True
    )
    hits = {p.probe: p.hits for p in full_report.probes}["neighbor_probe"]
    shared_only = bool(hits) and all(
        len(sc_full.envelopes[f"vector:{h}"].concerns()) > 1 for h in hits
    )
    out.append(
        GateResult(
            "sealed vectors survive only via a co-subject's key",
            shared_only,
            f"{len(hits)} surviving vectors, every one wrapped for someone else",
        )
    )

    # A surviving key backup defeats the erasure, and purging it does not.
    _, _, _, _, _, _, backed_up = shred(
        ErasureStrategy.CRYPTO_SHRED_TEXT, snapshot="nightly"
    )
    with_backup = {p.probe: p for p in backed_up.probes}["key_material_probe"]
    _, _, _, _, _, _, purged = shred(ErasureStrategy.CRYPTO_SHRED_TEXT)
    without = {p.probe: p for p in purged.probes}["key_material_probe"]
    out.append(
        GateResult(
            "a surviving key backup defeats crypto-shredding",
            with_backup.leaked and not without.leaked,
            f"with backup {list(with_backup.hits)}; without, clean",
        )
    )

    # A string-matching sweep goes blind on ciphertext.
    c, idx, keys, sc = sealed()
    blind = ErasureEngine(store=c.store, index=idx, embedder=emb).erase(
        c.subjects[0], ErasureStrategy.FULL
    )
    c2, idx2, keys2, sc2 = sealed()
    aware = ErasureEngine(
        store=c2.store, index=idx2, embedder=emb, keys=keys2, sealed=sc2
    ).erase(c2.subjects[0], ErasureStrategy.FULL)
    out.append(
        GateResult(
            "a key-blind sweep finds less than a key-aware one",
            blind.total_artifacts() < aware.total_artifacts(),
            f"blind {blind.total_artifacts()} vs key-aware "
            f"{aware.total_artifacts()} artifacts",
        )
    )

    # A key-aware re-ingest must not resurrect, and must be deterministic.
    trials = []
    for _ in range(2):
        c, idx, keys, sc, subj, v, before = shred(ErasureStrategy.CRYPTO_SHRED_TEXT)
        n_before = len(before.leaking_probes)
        done, skipped = rechunk_and_reindex_sealed(
            c.store, idx, emb, keys, sc, c.subjects
        )
        trials.append((n_before, len(v.verify(subj).leaking_probes), done, skipped))
    stable = trials[0] == trials[1]
    no_resurrection = trials[0][1] <= trials[0][0]
    out.append(
        GateResult(
            "a key-aware re-ingest is deterministic and does not resurrect",
            stable and no_resurrection,
            f"{trials[0][0]} -> {trials[0][1]} leaking, {trials[0][3]} documents "
            f"skipped as undecryptable, identical across two runs",
        )
    )

    return out


def main() -> int:
    print("=" * 78)
    print("ai-data-boundary-proxy gate: deterministic mock, offline")
    print("=" * 78)
    results = run_checks()
    width = max(len(r.name) for r in results)
    failed = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if not r.passed:
            failed += 1
        print(f"  [{status}] {r.name.ljust(width)}  {r.detail}")
    print()
    if failed:
        print(f"GATE FAILED: {failed} of {len(results)} checks")
        return 1
    print(f"GATE PASSED ({len(results)} checks)")
    print("  erasure is still graded, detection is still imperfect, and the")
    print("  corpus can still tell a thorough erasure from a careless one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
