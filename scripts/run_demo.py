#!/usr/bin/env python3
"""End-to-end demonstration, offline and deterministic by default.

Two runs produce identical output except for the audit record hashes, which
cover a live timestamp on purpose. See boundary/audit.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Python puts the SCRIPT's directory on sys.path, not the working directory, so
# `python scripts/run_demo.py` cannot see the package sitting beside `scripts/`
# without help. The README documents exactly that command, so the bootstrap
# lives here rather than in a PYTHONPATH the reader has to remember.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.audit import AuditLog  # noqa: E402
from boundary.config import load_settings  # noqa: E402
from boundary.corpus import LABELED, ROSTER, build_corpus  # noqa: E402
from boundary.detectors import (  # noqa: E402
    default_suite,
    detect_context,
    detect_loose,
    detect_strict,
)
from boundary.embeddings import default_embedder  # noqa: E402
from boundary.crypto import (  # noqa: E402
    KeyStore,
    rechunk_and_reindex_sealed,
    seal_store,
)
from boundary.erasure import (  # noqa: E402
    PLAINTEXT_STRATEGIES,
    ErasureEngine,
    ErasureStrategy,
    build_index,
    rechunk_and_reindex,
)
from boundary.metrics import (  # noqa: E402
    correlate,
    reidentification_residual,
    score_detector,
)
from boundary.models import Destination, EgressRequest  # noqa: E402
from boundary.policy import PolicyEngine  # noqa: E402
from boundary.proxy import build_proxy  # noqa: E402
from boundary.redaction import Pseudonymizer, Redactor, residual_classes  # noqa: E402
from boundary.verify import (  # noqa: E402
    ErasureVerifier,
    separation_band,
    NEIGHBOR_THRESHOLD,
)

RULE = "=" * 78
THIN = "-" * 78


def header(settings) -> None:
    corpus = build_corpus()
    print(RULE)
    print("ai-data-boundary-proxy demo")
    print(RULE)
    print(f"  provider (effective)                   {settings.effective_provider()}")
    print(f"  tenants                                {len(set(d.tenant_id for d in corpus.store.documents.values()))}")
    print(f"  documents                              {len(corpus.store.documents)}")
    print(f"  chunks                                 {len(corpus.store.chunks)}")
    print(f"  subjects                               {len(corpus.subjects)}")
    print(f"  labeled detector cases                 {len(LABELED)}")
    print(f"  names on the detector roster           {len(ROSTER)}")
    print()
    print("  The corpus is synthetic. The organizations are fictional, every")
    print("  host uses an .example TLD, and no real personal data is present.")
    print()


def section_detection() -> None:
    print(RULE)
    print("1. Detection, and what it misses")
    print(RULE)
    suite = default_suite(ROSTER)
    print(f"  {'detector':10s} {'recall':28s} {'precision':28s}")
    for name in suite.names():
        s = score_detector(name, LABELED, suite.detectors[name])
        print(f"  {name:10s} {str(s.recall):28s} {str(s.precision):28s}")
    overall = score_detector("all", LABELED, suite.run)
    print(f"  {'ALL':10s} {str(overall.recall):28s} {str(overall.precision):28s}")
    print()
    print(f"  false negatives: {overall.false_negatives}   false positives: {overall.false_positives}")
    print()
    print("  RECALL IS BELOW 1.0 ON PURPOSE AND THAT IS THE HEADLINE. Person-name")
    print("  detection here is a roster, so it finds the names it was given and")
    print("  misses the two the corpus contains deliberately. Real name detection")
    print("  needs a model, and a boundary that depends on catching every name is")
    print("  a boundary that leaks. The control is policy -- deny the destination")
    print("  -- not a better pattern.")
    print()

    corr = correlate(LABELED, detect_strict, detect_loose, "strict", "loose")
    print(THIN)
    print("  Stacking two detectors is not additive")
    print(THIN)
    print(f"  cases both caught                      {corr.both}")
    print(f"  caught only by loose                   {corr.b_only}")
    print(f"  missed by both                         {corr.neither}")
    print(f"  marginal gain of loose over strict     {corr.marginal_gain:.3f}")
    print(f"  error overlap                          {corr.error_overlap:.3f}")
    print()
    print("  The tempting arithmetic says two detectors at 0.5 recall give 0.75.")
    print("  They share a design, so they share their blind spots, and the")
    print("  measured marginal gain is what a second detector actually buys.")
    print()


def section_policy(settings) -> None:
    print(RULE)
    print("2. Egress policy: what may leave at all")
    print(RULE)
    audit = AuditLog()
    proxy = build_proxy(key=settings.key_bytes(), roster=ROSTER, audit=audit)

    # The real HR record from the corpus, not a hand-written sample, so what
    # the policy decides here is what it decides on the actual data.
    sample = build_corpus().store.documents["acme-hr-001"].text
    print(f"  {'destination':22s} {'action':8s} {'rule':28s}")
    for dest in Destination:
        req = EgressRequest(tenant_id="acme", text=sample, destination=dest)
        result = proxy.evaluate(req)
        print(
            f"  {dest.value:22s} {result.verdict.action.value:8s} "
            f"{result.verdict.rule_id:28s}"
        )
    print()
    print("  POLICY RUNS BEFORE REDACTION, and a DENY short circuits before any")
    print("  text is prepared for transmission. Some material must not leave in")
    print("  any form, and no redaction makes it acceptable.")
    print()

    print(THIN)
    print("  What survives redaction")
    print(THIN)
    suite = default_suite(ROSTER)
    findings = suite.run(sample)
    redactor = Redactor(pseudonymizer=Pseudonymizer(settings.key_bytes()))
    out = redactor.apply(sample, findings)
    survived = residual_classes(sample, findings, redactor)
    residual = reidentification_residual(findings, survived)
    print(f"  spans rewritten                        {out.count}")
    print(f"  direct identifiers removed             {len(residual['direct_removed'])}")
    print(f"  quasi-identifiers surviving            {residual['quasi_count']} {residual['quasi_surviving']}")
    print(f"  re-identifiable by mosaic              {residual['reidentifiable']}")
    print()
    print("  Every direct identifier can be removed and the record can still")
    print("  single out a person. Recall on direct identifiers is the number")
    print("  people quote; the surviving quasi-identifier set is the number that")
    print("  decides whether the subject is still identifiable.")
    print()

    print(THIN)
    print("  Pseudonyms preserve linkage")
    print(THIN)
    ps = Pseudonymizer(settings.key_bytes())
    from boundary.models import DataClass

    a = ps.token(DataClass.EMAIL, "jrivera@acme.example")
    b = ps.token(DataClass.EMAIL, "JRivera@Acme.Example")
    print(f"  token for the address                  {a}")
    print(f"  token for the same address, recased    {b}")
    print(f"  the recipient can link the two         {a == b}")
    print()
    print("  A stable token hides the value and preserves the RELATIONSHIP. The")
    print("  recipient still counts distinct subjects and still knows which")
    print("  requests concern the same person. That is a disclosure.")
    print()


def section_tenancy() -> None:
    print(RULE)
    print("3. Tenant scoping: pre-filter versus post-filter")
    print(RULE)
    corpus = build_corpus()
    emb = default_embedder()
    idx = build_index(corpus.store, emb)
    qv = emb.embed("employee record payroll access")

    pre = idx.search(qv, k=5, tenant_id="acme", post_filter=False)
    post = idx.search(qv, k=5, tenant_id="acme", post_filter=True)
    print(f"  pre-filtered results returned          {len(pre)} of 5 requested")
    print(f"  post-filtered results returned         {len(post)} of 5 requested")
    print()
    print("  The post-filter ranks the WHOLE index, takes the top k, then drops")
    print("  the rows the caller may not see. The shortfall is itself the")
    print("  disclosure: the caller learns that other tenants hold documents")
    print("  matching this query, and roughly how well they matched, without")
    print("  ever being shown their content.")
    print()


def section_erasure() -> None:
    print(RULE)
    print("4. Right to erasure, and the derived data it forgets")
    print(RULE)
    emb = default_embedder()
    corpus = build_corpus()
    idx = build_index(corpus.store, emb)
    band = separation_band(idx, emb, corpus.subjects[0])
    print(f"  neighbor probe threshold               {NEIGHBOR_THRESHOLD}")
    print(f"  best score among non-mentions          {band['max_without']:.3f}")
    print(f"  worst score among real mentions        {band['min_with']:.3f}")
    print(f"  populations separated                  {bool(band['separated'])}")
    print()

    print(f"  {'strategy':18s} {'artifacts':>9s} {'leaking probes':>15s} {'after re-ingest':>16s}")
    rows = []
    for strat in PLAINTEXT_STRATEGIES:
        c = build_corpus()
        index = build_index(c.store, emb)
        subj = c.subjects[0]
        cert = ErasureEngine(store=c.store, index=index, embedder=emb).erase(subj, strat)
        v = ErasureVerifier(c.store, index, emb)
        report = v.verify(subj)
        before = len(report.leaking_probes)
        rechunk_and_reindex(c.store, index, emb)
        after = len(v.verify(subj).leaking_probes)
        rows.append((strat, report, before, after))
        print(
            f"  {strat.value:18s} {cert.total_artifacts():>9d} "
            f"{before:>15d} {after:>16d}"
        )
    print()

    weakest = rows[0][1]
    print("  Probes still finding the subject after SOURCE_ONLY:")
    for p in weakest.leaking_probes:
        print(f"    {p.probe:20s} {p.detail}")
    print()
    print("  THE THIRD COLUMN IS THE FINDING. MENTION_SWEEP cleans the chunks,")
    print("  the vectors, the caches and the logs, and passes every probe the")
    print("  application can see -- then the subject comes BACK when the corpus")
    print("  is re-ingested, because their name was still in a source document")
    print("  owned by somebody else and re-chunking reads documents. No code")
    print("  changed and no operator did anything.")
    print()
    print("  The neighbor probe is the one a real system never runs. An")
    print("  embedding cannot be grepped, so a vector derived from text that")
    print("  named a subject stays reachable after every copy of that text is")
    print("  gone, and only an embedding-space probe will find it.")
    print()



def section_crypto() -> None:
    print(RULE)
    print("5. Crypto-shredding: erase the key, not the rows")
    print(RULE)
    emb = default_embedder()

    def sealed(snapshot=None, vectors=False):
        c = build_corpus()
        idx = build_index(c.store, emb)
        keys = KeyStore()
        sc = seal_store(c.store, idx, keys, c.subjects, seal_vectors=vectors)
        if snapshot:
            keys.snapshot(snapshot)
        return c, idx, keys, sc

    c, idx, keys, sc = sealed()
    subj = c.subjects[0]
    print(f"  artifacts sealed                       {len(sc.envelopes)}")
    print(f"  subject's name found by a string scan  "
          f"{any(subj.matches(d.text) for d in c.store.documents.values())}")
    live = ErasureVerifier(c.store, idx, emb, keys=keys, sealed=sc).verify(subj)
    print(f"  probes still reaching the subject      {len(live.leaking_probes)}")
    print()
    print("  ENCRYPTED IS NOT ERASED, AND THE PROBES HAVE TO KNOW THAT. Every")
    print("  identifier is gone from the stored bytes and the subject is fully")
    print("  reachable, because the keys are live. A probe that matched on")
    print("  stored bytes would call this clean, which is why every string")
    print("  probe resolves an artifact to what the system can READ first.")
    print()

    print(THIN)
    print("  What destroying the key does and does not reach")
    print(THIN)
    print(f"  {'strategy':28s} {'artifacts':>7s} {'leaking':>8s} {'after re-ingest':>16s}")
    for label, strat, snap, vec in (
        ("crypto_shred_text", ErasureStrategy.CRYPTO_SHRED_TEXT, "nightly", False),
        ("crypto_shred_text, no backup", ErasureStrategy.CRYPTO_SHRED_TEXT, None, False),
        ("crypto_shred_full", ErasureStrategy.CRYPTO_SHRED_FULL, "nightly", True),
        ("crypto_shred_full, no backup", ErasureStrategy.CRYPTO_SHRED_FULL, None, True),
    ):
        c, idx, keys, sc = sealed(snap, vec)
        subj = c.subjects[0]
        cert = ErasureEngine(
            store=c.store, index=idx, embedder=emb, keys=keys, sealed=sc
        ).erase(subj, strat)
        v = ErasureVerifier(c.store, idx, emb, keys=keys, sealed=sc)
        before = len(v.verify(subj).leaking_probes)
        rechunk_and_reindex_sealed(c.store, idx, emb, keys, sc, c.subjects)
        after = len(v.verify(subj).leaking_probes)
        print(f"  {label:28s} {cert.total_artifacts():>7d} {before:>8d} {after:>16d}")
    print()
    print("  IT REMOVES NOTHING AND ERASES SOMETHING. Every byte is still on")
    print("  disk; the certificate's document, chunk and vector columns are")
    print("  empty. What changed is that the running system cannot read them.")
    print()
    print("  THE LIMIT IS THE SHARED RECORD, AT EVERY LAYER. The two probes")
    print("  that go clean are the ones covering artifacts about this subject")
    print("  ALONE -- the cached answer and the retained prompt. A support")
    print("  ticket another person owns is wrapped for both of them, so their")
    print("  key still opens it and the subject is still named inside. Sealing")
    print("  the vectors too closes the neighbor path for the subject's own")
    print("  embeddings and leaves every embedding derived from a shared chunk.")
    print()
    print("  UNLIKE A CONTENT SWEEP, A RE-INGEST CANNOT UNDO IT. Re-chunking")
    print("  reads documents, and two of the eight no longer decrypt, so they")
    print("  are skipped. You cannot re-ingest what you cannot read.")
    print()

    print(THIN)
    print("  A string-matching sweep goes blind on ciphertext")
    print(THIN)
    for aware in (False, True):
        c, idx, keys, sc = sealed()
        eng = ErasureEngine(
            store=c.store, index=idx, embedder=emb,
            keys=keys if aware else None, sealed=sc if aware else None,
        )
        cert = eng.erase(c.subjects[0], ErasureStrategy.FULL)
        v = ErasureVerifier(c.store, idx, emb, keys=keys, sealed=sc)
        n = len(v.verify(c.subjects[0]).leaking_probes)
        tag = "key-aware" if aware else "key-blind"
        plural = "probe " if n == 1 else "probes"
        print(f"  full sweep, {tag:10s}             "
              f"{cert.total_artifacts():>2d} artifacts, {n:>2d} {plural} leaking")
    print()
    print("  THE SAME STRATEGY ON THE SAME CORPUS. The blind sweep still finds")
    print("  what it can reach by FOREIGN KEY -- the subject's own documents --")
    print("  and misses every mention inside another party's record, because")
    print("  that path matches on text and ciphertext matches nothing. It does")
    print("  not error. It reports artifacts touched and looks like it worked.")
    print("  ENCRYPTION AT REST SILENTLY DISABLES EVERY STRING-MATCHING")
    print("  GOVERNANCE TOOL YOU OWN: the retention sweep, the DLP scanner,")
    print("  the erasure job.")
    print()

    print(THIN)
    print("  A routine key rotation undoes a completed erasure")
    print(THIN)
    c, idx, keys, sc = sealed("nightly")
    subj = c.subjects[0]
    ErasureEngine(
        store=c.store, index=idx, embedder=emb, keys=keys, sealed=sc
    ).erase(subj, ErasureStrategy.CRYPTO_SHRED_TEXT)
    v = ErasureVerifier(c.store, idx, emb, keys=keys, sealed=sc)
    before = len(v.verify(subj).leaking_probes)
    receipt, rewrapped = keys.rotate(
        subj.subject_id, sc.for_subject(subj.subject_id), source="snapshot:nightly"
    )
    sc.substitute(rewrapped)
    after = len(v.verify(subj).leaking_probes)
    print(f"  leaking probes after the erasure       {before}")
    print(f"  rotation sourced its old key from      {receipt.source}")
    print(f"  envelopes re-wrapped                   {receipt.envelopes_rewrapped}")
    print(f"  subject resurrected                    {receipt.resurrected}")
    print(f"  leaking probes after the rotation      {after}")
    print()
    print("  THE KEY WAS DESTROYED AN HOUR AGO AND THE SUBJECT IS READABLE")
    print("  AGAIN. A rotation job has to read the old key material from")
    print("  somewhere, and a disaster-recovery rebuild reads it from a backup")
    print("  by definition. It unwrapped an erased subject's content keys and")
    print("  re-wrapped them under a fresh LIVE key. No code changed and no")
    print("  operator did anything wrong -- the same sentence this demo already")
    print("  uses about a re-ingest, one layer down.")
    print()


def section_audit(settings) -> None:
    print(RULE)
    print("6. Audit trail")
    print(RULE)
    # The file is where AUDIT_LOG_PATH points, and the demo says which. A
    # configuration key that names a file nothing writes is worse than no key.
    audit = AuditLog(path=settings.audit_log_path)
    proxy = build_proxy(key=settings.key_bytes(), roster=ROSTER, audit=audit)
    for dest in (Destination.INTERNAL_MODEL, Destination.VENDOR_NO_TRAIN):
        proxy.send(
            EgressRequest(
                tenant_id="acme",
                text="Contact jrivera@acme.example about the ticket.",
                destination=dest,
            )
        )
    intact, bad, reason = audit.verify_chain()
    print(f"  records                                {len(audit)}")
    print(f"  chain intact                           {intact} ({reason})")
    if settings.audit_log_path:
        on_disk = AuditLog.read(settings.audit_log_path)
        print(
            f"  written to                             {settings.audit_log_path} "
            f"({len(on_disk)} records, re-read chain intact "
            f"{on_disk.verify_chain()[0]})"
        )
    else:
        print("  written to                             nowhere: AUDIT_LOG_PATH is unset")

    # Three tampers, because they break the chain in three different places
    # and a log that only detects one of them is not tamper-evident.
    before = audit.records[0].detail.get("action")
    audit.records[0].detail["action"] = "allow-TAMPERED"
    intact2, bad2, reason2 = audit.verify_chain()
    print(f"  edit a record's contents               {intact2} at index {bad2}: {reason2}")
    audit.records[0].detail["action"] = before

    audit.records[0], audit.records[1] = audit.records[1], audit.records[0]
    intact3, bad3, reason3 = audit.verify_chain()
    print(f"  reorder two records                    {intact3} at index {bad3}: {reason3}")
    audit.records[0], audit.records[1] = audit.records[1], audit.records[0]

    excised = audit.records.pop(1)
    intact4, bad4, reason4 = audit.verify_chain()
    print(f"  excise a record from the middle        {intact4} at index {bad4}: {reason4}")
    audit.records.insert(1, excised)

    intact5, _, reason5 = audit.verify_chain()
    print(f"  chain restored after undoing all three {intact5} ({reason5})")
    print()
    print("  The log records counts, classes, rule ids and decisions. It never")
    print("  records the text or a detected value: an audit trail of privacy")
    print("  decisions that quotes the data it protected is a second copy of")
    print("  the thing you were containing, usually retained longer.")
    print()


def main() -> int:
    settings = load_settings()
    header(settings)
    section_detection()
    section_policy(settings)
    section_tenancy()
    section_erasure()
    section_crypto()
    section_audit(settings)
    print(RULE)
    print("What this demo does NOT claim")
    print(RULE)
    print("  - The embedder is a deterministic hash model, not a semantic one.")
    print("    Every number here is evidence about the harness, not about any")
    print("    real embedding model.")
    print("  - Person-name detection is a roster and its recall is below 1.0.")
    print("  - The vector index is brute force and in memory. The adapter seam")
    print("    for a real store exists; no real store has been measured.")
    print("  - Erasure is verified against ten reachability paths. Ten is not")
    print("    all of them: backups, replicas and downstream exports are named")
    print("    in the README as out of scope and are not probed.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
