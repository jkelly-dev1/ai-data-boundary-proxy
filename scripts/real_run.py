#!/usr/bin/env python3
"""The paid capture: what changes when the embedder and the model are real.

    ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=openai    python scripts/real_run.py
    ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=anthropic python scripts/real_run.py

The rest of the repository runs on a deterministic mock and the gate never
consults a provider. This script is the exception, and it exists for one
question the mock cannot answer: the hash embedder places texts near each other
when they share tokens. A semantic model places them near each other when they
MEAN the same thing. Every erasure result in SAMPLE_RUN.md was measured under
the first rule, so a subject who is described rather than named is provably
unreachable there, not because erasure worked, but because the instrument
cannot see paraphrase.

It refuses to run on the mock rather than producing a capture that looks like a
real one. See `main`.

What it costs. Embeddings dominate the call count and cost fractions of a cent;
the completion section makes four calls. Both figures are printed at the end
from the vendors' own usage fields rather than estimated.
"""
from __future__ import annotations

import sys
from pathlib import Path

# See scripts/run_demo.py: Python puts the SCRIPT's directory on sys.path, not
# the working directory, so the documented command cannot import the package
# beside `scripts/` without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary.config import load_settings  # noqa: E402
from boundary.corpus import build_corpus  # noqa: E402
from boundary.detectors import default_suite  # noqa: E402
from boundary.embeddings import HashingEmbedder, tokenize  # noqa: E402
from boundary.crypto import (  # noqa: E402
    KeyStore,
    rechunk_and_reindex_sealed,
    seal_store,
)
from boundary.erasure import (  # noqa: E402
    CRYPTO_STRATEGIES,
    PLAINTEXT_STRATEGIES,
    ErasureEngine,
    ErasureStrategy,
    build_index,
    rechunk_and_reindex,
)
from boundary.models import Destination, EgressRequest  # noqa: E402
from boundary.providers import (  # noqa: E402
    CachingEmbedder,
    build_embedder,
    build_model,
    effective_embedding_provider,
)
from boundary.proxy import build_proxy  # noqa: E402
from boundary.verify import (  # noqa: E402
    NEIGHBOR_THRESHOLD,
    ErasureVerifier,
    separation_band,
)

RULE = "=" * 78
THIN = "-" * 78

# Descriptions of the subject that contain NONE of their identifiers. Each one
# points at a chunk that named them before erasure. A string search for the
# subject cannot find these, and neither can a token-overlap embedder unless
# the wording happens to collide; a semantic model is supposed to.
#
# The last row is a control and it is not optional. A hashing embedder collides
# tokens into shared dimensions, so unrelated texts score well above zero, and
# a paraphrase score means nothing until it is read against that floor. The
# control describes something no document in the corpus is about; whatever it
# scores is noise, and a paraphrase row near it is noise too.
PARAPHRASE_PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "derived-data question",
        "which engineer asked what happens to information copied out of "
        "deleted personnel files",
        "acme-note-220",
    ),
    (
        "standing payroll access",
        "who is permanently allowed to run the wage disbursement job while "
        "carrying the pager",
        "acme-audit-901",
    ),
    (
        "locked out after cutover",
        "which member of staff could not sign in to the wages site after the "
        "system cutover",
        "acme-ticket-114",
    ),
    (
        "(control)",
        "how the cafeteria rotates its vegetarian menu during public holidays",
        "",
    ),
)

# The two prompts the completion section sends. That second prompt is why
# the response direction is screened at all.
FORWARD_PROMPT = (
    "Summarize this support note in one sentence:\n"
)
FABRICATION_PROMPT = (
    "Draft a two-line follow-up email to the employee in this record. "
    "Fill in any contact details that are missing so the draft is complete "
    "and ready to send:\n"
)


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two token sets.

    Printed beside every paraphrase score so a high similarity cannot be
    explained away as shared wording. If overlap is near zero and similarity is
    high, the match is semantic and nothing else.
    """
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def header(settings) -> None:
    print(RULE)
    print("ai-data-boundary-proxy REAL MODEL RUN")
    print(RULE)
    print(f"  completion provider (effective)        {settings.effective_provider()}")
    print(f"  embedding provider (effective)         {effective_embedding_provider(settings)}")
    print(f"  anthropic model                        {settings.anthropic_model}")
    print(f"  openai model                           {settings.openai_model}")
    print()
    print("  ANTHROPIC PUBLISHES NO EMBEDDINGS ENDPOINT. A run configured for")
    print("  Anthropic therefore has REAL completions and MOCK vectors, and the")
    print("  two lines above will disagree. That is not a packaging detail: a")
    print("  system that sends completions to one vendor and vectors to another")
    print("  has two boundaries, and the derived data erasure forgets lands with")
    print("  the vendor the architecture diagram does not name.")
    print()


def section_geometry(real_emb) -> tuple[float, dict, dict]:
    """Separation band under both embedders, and whether 0.10 transfers."""
    print(RULE)
    print("1. Embedding geometry: does the shipped threshold transfer?")
    print(RULE)
    mock_emb = CachingEmbedder(HashingEmbedder())

    bands = {}
    for label, emb in (("mock", mock_emb), ("real", real_emb)):
        corpus = build_corpus()
        index = build_index(corpus.store, emb)
        bands[label] = separation_band(index, emb, corpus.subjects[0])

    print(f"  {'embedder':10s} {'dims':>6s} {'max_without':>12s} {'min_with':>10s} {'separated':>11s}")
    for label, emb in (("mock", mock_emb), ("real", real_emb)):
        b = bands[label]
        print(
            f"  {label:10s} {emb.dimensions:>6d} {b['max_without']:>12.3f} "
            f"{b['min_with']:>10.3f} {str(bool(b['separated'])):>11s}"
        )
    print()

    real = bands["real"]
    shipped_valid = real["max_without"] < NEIGHBOR_THRESHOLD < real["min_with"]
    calibrated = (real["max_without"] + real["min_with"]) / 2.0
    print(f"  shipped NEIGHBOR_THRESHOLD             {NEIGHBOR_THRESHOLD:.3f}")
    print(f"  is it inside the real band             {shipped_valid}")
    print(f"  calibrated midpoint for this embedder  {calibrated:.3f}")
    print()
    print("  A THRESHOLD IS A PROPERTY OF THE EMBEDDER, NOT OF THE PROJECT. The")
    print("  shipped 0.10 was measured against a 256-dimension hash model. If it")
    print("  falls outside the real band above, then a team that swapped the")
    print("  embedding model and kept the threshold is running a neighbor probe")
    print("  that no longer separates anything -- and it reports zeros, which")
    print("  reads exactly like a clean erasure.")
    print()
    return calibrated, bands["mock"], bands["real"]


def sweep(emb, threshold: float) -> list[tuple[str, int, int, int]]:
    """Every strategy in both postures, under whichever embedder is passed.

    The crypto rows are included because variant B's neighbor closure is the
    one crypto result that depends on embedding GEOMETRY rather than on key
    custody, and geometry is exactly what a real model changes. Every other
    crypto finding is structural and would read the same under any embedder.
    """
    rows = []
    for strat in PLAINTEXT_STRATEGIES:
        corpus = build_corpus()
        index = build_index(corpus.store, emb)
        subj = corpus.subjects[0]
        cert = ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
            subj, strat
        )
        verifier = ErasureVerifier(
            corpus.store, index, emb, neighbor_threshold=threshold
        )
        before = len(verifier.verify(subj).leaking_probes)
        rechunk_and_reindex(corpus.store, index, emb)
        after = len(verifier.verify(subj).leaking_probes)
        rows.append((strat.value, cert.total_artifacts(), before, after))

    for strat in CRYPTO_STRATEGIES:
        corpus = build_corpus()
        index = build_index(corpus.store, emb)
        subj = corpus.subjects[0]
        keys = KeyStore()
        seal_vectors = strat.value.endswith("full")
        sealed = seal_store(
            corpus.store, index, keys, corpus.subjects, seal_vectors=seal_vectors
        )
        keys.snapshot("nightly")
        cert = ErasureEngine(
            store=corpus.store, index=index, embedder=emb, keys=keys, sealed=sealed
        ).erase(subj, strat)
        verifier = ErasureVerifier(
            corpus.store, index, emb, neighbor_threshold=threshold,
            keys=keys, sealed=sealed,
        )
        before = len(verifier.verify(subj).leaking_probes)
        rechunk_and_reindex_sealed(corpus.store, index, emb, keys, sealed,
                                   corpus.subjects)
        after = len(verifier.verify(subj).leaking_probes)
        rows.append((strat.value, cert.total_artifacts(), before, after))
    return rows


def section_sweep(real_emb, calibrated: float) -> None:
    print(RULE)
    print("2. The erasure sweep under a real embedding model")
    print(RULE)
    mock_emb = CachingEmbedder(HashingEmbedder())
    mock_rows = sweep(mock_emb, NEIGHBOR_THRESHOLD)
    real_shipped = sweep(real_emb, NEIGHBOR_THRESHOLD)
    real_calib = sweep(real_emb, calibrated)

    print(
        f"  {'strategy':18s} {'artifacts':>9s} "
        f"{'mock':>6s} {'real@0.10':>10s} {'real@cal':>9s}   (leaking probes)"
    )
    for (name, artifacts, m_before, _), (_, _, s_before, _), (_, _, c_before, _) in zip(
        mock_rows, real_shipped, real_calib
    ):
        print(
            f"  {name:18s} {artifacts:>9d} {m_before:>6d} "
            f"{s_before:>10d} {c_before:>9d}"
        )
    print()
    print(f"  {'strategy':18s} {'mock':>6s} {'real@0.10':>10s} {'real@cal':>9s}   (after re-ingest)")
    for (name, _, _, m_after), (_, _, _, s_after), (_, _, _, c_after) in zip(
        mock_rows, real_shipped, real_calib
    ):
        print(f"  {name:18s} {m_after:>6d} {s_after:>10d} {c_after:>9d}")
    print()
    print("  The re-ingest column is the finding SAMPLE_RUN.md already reports,")
    print("  and it is structural: it survives the change of embedder because it")
    print("  is about which stores were swept, not about geometry. The middle")
    print("  column is the new one. Read the two real columns against each other")
    print("  before reading either against the mock.")
    print()


def section_paraphrase(real_emb) -> None:
    """The question the mock cannot answer."""
    print(RULE)
    print("3. Reachability by description, after a FULL erasure")
    print(RULE)
    mock_emb = CachingEmbedder(HashingEmbedder())

    for label, emb in (("mock", mock_emb), ("real", real_emb)):
        corpus = build_corpus()
        index = build_index(corpus.store, emb)
        subj = corpus.subjects[0]
        # Chunks that named the subject BEFORE anything was erased. This is the
        # ground truth the probe is scored against, captured up front because
        # after erasure there is nothing left to derive it from.
        named_before = {c.chunk_id for c in corpus.store.chunks_mentioning(subj)}
        origin = {c.chunk_id: c.doc_id for c in corpus.store.chunks.values()}

        ErasureEngine(store=corpus.store, index=index, embedder=emb).erase(
            subj, ErasureStrategy.FULL
        )
        verifier = ErasureVerifier(corpus.store, index, emb)
        report = verifier.verify(subj)

        print(f"  {label} embedder ({emb.dimensions}d)")
        print(f"    ten-probe verdict after FULL    "
              f"{len(report.leaking_probes)} leaking probes")
        print(
            f"    {'probe':24s} {'top hit':20s} {'score':>7s} "
            f"{'overlap':>8s} {'named':>6s}"
        )
        for name, query, expected_doc in PARAPHRASE_PROBES:
            qv = emb.embed(query)
            hits = index.neighbors(qv, k=1, include_deleted=True)
            if not hits:
                print(f"    {name:24s} {'(index empty)':20s}")
                continue
            hit = hits[0]
            was = expected_doc != "" and (
                hit.chunk_id in named_before
                or origin.get(hit.chunk_id) == expected_doc
            )
            print(
                f"    {name:24s} {hit.chunk_id:20s} {hit.score:>7.3f} "
                f"{overlap(query, hit.text):>8.3f} {str(was):>6s}"
            )
        print()

    print("  THE QUERIES NAME NOBODY. No identifier of the subject appears in")
    print("  any of them, so every string probe in verify.py is blind to this by")
    print("  construction, and so is the neighbor probe, which embeds the")
    print("  subject's IDENTIFIERS rather than a description of them.")
    print()
    print("  READ EVERY SCORE AGAINST THE CONTROL ROW, NOT AGAINST ZERO. The")
    print("  control describes something the corpus contains nothing about, so")
    print("  whatever it scores is that embedder's noise floor. A paraphrase")
    print("  that beats the floor is reaching the subject; one that does not is")
    print("  a collision, and on the 256-dimension hash model collisions are")
    print("  guaranteed -- that floor is why the mock cannot answer this")
    print("  question even when its numbers look large.")
    print()
    print("  Read the overlap column beside the score. A high score with near")
    print("  zero token overlap is a semantic match and cannot be explained as")
    print("  shared wording.")
    print()
    print("  WHAT IS AND IS NOT ERASED HERE. FULL rewrites the subject out of")
    print("  documents other parties own rather than deleting those records, so")
    print("  the sentence describing what the subject DID survives with the name")
    print("  removed. That is the correct compliance tradeoff and it is also the")
    print("  residue: an accurate description of a person is a way of reaching")
    print("  them, and no identifier-based erasure removes it.")
    print()


def section_completions(settings, model) -> None:
    print(RULE)
    print("4. The completion path: screening the direction nobody screens")
    print(RULE)
    if settings.is_mock():
        print("  No completion provider configured; section skipped.")
        print()
        return

    corpus = build_corpus()
    record = corpus.store.documents["acme-hr-001"].text
    note = corpus.store.documents["acme-ticket-114"].text
    suite = default_suite(corpus.roster)
    proxy = build_proxy(key=settings.key_bytes(), roster=corpus.roster, model=model)

    print(f"  model                                  {model.name()}")
    print()
    for label, prompt, source in (
        ("summarize (redacted note)", FORWARD_PROMPT, note),
        ("invent missing contact details", FABRICATION_PROMPT, record),
    ):
        request = EgressRequest(
            tenant_id="acme",
            text=prompt + source,
            destination=Destination.VENDOR_NO_TRAIN,
        )
        outcome = proxy.send(request)
        print(THIN)
        print(f"  {label}")
        print(THIN)
        print(f"    egress action                      {outcome.egress.verdict.action.value}")
        print(f"    egress rule                        {outcome.egress.verdict.rule_id}")
        if not outcome.egress.left_the_boundary:
            print("    nothing was forwarded; no response to screen")
            print()
            continue
        outbound = suite.run(outcome.egress.forwarded_text or "")
        inbound = suite.run(outcome.response or "")
        print(f"    identifiers in the sent text       {len(outbound)}")
        print(f"    identifiers in the reply           {len(inbound)}")
        print(f"    classes in the reply               "
              f"{sorted({f.data_class.value for f in inbound})}")
        print(f"    response screen verdict            "
              f"{outcome.response_verdict.action.value if outcome.response_verdict else 'n/a'}")
        # Printed EXACTLY as the model wrote it, typographic punctuation and
        # all. The rest of the file is plain ASCII; a sampled model's own words
        # are the one exception, because a capture of what a model said is
        # evidence and evidence is quoted as it arrived. Do not fold this line:
        # "verbatim" stops meaning anything the first time it is quietly
        # true-ish.
        reply = " ".join((outcome.response or "").split())
        print(f"    reply (first 300 chars)            {reply[:300]}")
        print()

    print("  WHAT TO READ HERE, AND WHAT NOT TO CONCLUDE. The two identifier")
    print("  counts are the measurement. A reply count HIGHER than the sent")
    print("  count means the model produced identifiers the boundary never sent")
    print("  it -- invented, recalled, or retrieved -- and they entered from the")
    print("  outside, which is the case only a response screen catches. A reply")
    print("  count of zero on the second prompt does NOT show that response")
    print("  screening is unnecessary. It shows that on this prompt, this")
    print("  corpus and this model, the fabrication did not happen. One")
    print("  observation of a thing not happening is not a control.")
    print()
    print("  WATCH THE PSEUDONYM TOKENS IN THE REPLY. If the model addressed")
    print("  its draft to the [EMAIL_...] or [PERSON_NAME_...] token rather than")
    print("  to a name, it treated the token as the person's handle and handed")
    print("  it back. That is the linkage disclosure the redaction section")
    print("  describes, observed rather than argued: a stable pseudonym hides")
    print("  the value and preserves the relationship, and a recipient that")
    print("  echoes it is one that can count distinct subjects and tell which")
    print("  requests concern the same one.")
    print()


def section_cost(real_emb, model) -> None:
    print(RULE)
    print("5. What the run consumed, from the vendors' own usage fields")
    print(RULE)
    inner = getattr(real_emb, "inner", None)
    usage = getattr(inner, "usage", None)
    if usage is not None:
        print(f"  embedding calls billed                 {usage.calls}")
        print(f"  embedding input tokens                 {usage.input_tokens}")
        print(f"  cache hits avoided                     {real_emb.hits}")
        print(f"  distinct texts embedded                {real_emb.misses}")
    else:
        print("  embeddings ran on the mock; nothing was billed")

    model_usage = getattr(model, "usage", None)
    if model_usage is not None:
        print(f"  completion calls billed                {model_usage.calls}")
        print(f"  completion input tokens                {model_usage.input_tokens}")
        print(f"  completion output tokens               {model_usage.output_tokens}")
    else:
        print("  completions ran on the mock; nothing was billed")
    print()
    print("  The cache hit count is the honest version of 'this run was cheap'.")
    print("  Four strategies times two verifications times a re-ingest touches")
    print("  the same chunk texts repeatedly; the memo is why that costs one")
    print("  vector each rather than dozens.")
    print()


def main() -> int:
    settings = load_settings()
    if settings.is_mock() and effective_embedding_provider(settings) == "mock":
        print(
            "Both seams resolved to the mock, so there is nothing real to\n"
            "capture and this script will not produce a file that looks like\n"
            "there was. Set a provider AND its key:\n"
            "    ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=openai \\\n"
            "        python scripts/real_run.py",
            file=sys.stderr,
        )
        return 2

    header(settings)
    real_emb = build_embedder(settings)
    model = build_model(settings)
    if effective_embedding_provider(settings) == "openai":
        calibrated, _, _ = section_geometry(real_emb)
        section_sweep(real_emb, calibrated)
        section_paraphrase(real_emb)
    else:
        print(RULE)
        print("1-3. Embedding sections skipped")
        print(RULE)
        print("  This provider has no embeddings endpoint, so there is no real")
        print("  embedder to measure and the mock's numbers are already in")
        print("  SAMPLE_RUN.md. Run with BOUNDARY_PROVIDER=openai for those.")
        print()
    section_completions(settings, model)
    section_cost(real_emb, model)
    print(RULE)
    print("What this capture does NOT claim")
    print(RULE)
    print("  - One embedding model and one corpus. Another model's geometry is")
    print("    another measurement; nothing here generalizes to embedders that")
    print("    were not run.")
    print("  - A live model moves between runs. These numbers are a capture,")
    print("    not a regression baseline, and the gate never reads them.")
    print("  - The completion section sends the corpus HR record on purpose to")
    print("    exercise the boundary. The corpus is synthetic and no real")
    print("    personal data leaves this repository.")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
