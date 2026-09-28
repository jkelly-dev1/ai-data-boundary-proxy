# Sample run

Verbatim captures of `scripts/run_demo.py`, the boundary gate, the test suite,
and the paid `scripts/real_run.py`, so a reviewer without an API key can see
exactly what this project measures and what it refuses to claim. Nothing here
is retyped, reformatted or cleaned up: every rate, interval, count and
verdict shown here is the one the run produced. Offline captures 2026-09-27; real-model
captures 2026-08-02.

Which captures are real, stated once so nothing below is over-read. Every number
in the first four sections comes from the deterministic offline mock and is
evidence about the harness and about nothing else. The [real-model
run](#the-real-model-run) is the exception and is labeled throughout: it was
paid for against `text-embedding-3-small`, `claude-opus-5` and `gpt-5.6-terra`,
and where it disagrees with the mock, it says so rather than the mock's number
being silently replaced.

What is and is not deterministic. Two runs of `scripts/run_demo.py` produce
BYTE-IDENTICAL output, verified by running it twice into two files and diffing
them. With `AUDIT_LOG_PATH` set, the file is resumed rather than overwritten, so
the file total printed beside its path grows from run to run; the `records made
by this run` line does not, and a test runs the demo twice into one file to hold
both. The audit trail's record hashes cover a live timestamp and therefore move
between runs, but the demo prints chain verdicts rather than the hashes
themselves, so the visible output is stable. A faked timestamp in an audit trail
is worse than an honest one that varies, because the point of the log is that it
says when a decision was made.

The real-model capture is deterministic in its measurements and not in its
prose: two runs of `scripts/real_run.py` reproduced every embedding number in
this file exactly, while the completion replies are different sentences each
time. That is the expected split, an embedding endpoint is a function, a chat
completion is a sample, and it is why the gate reads neither.

Contents

- [The demonstration](#the-demonstration)
- [The boundary gate](#the-boundary-gate)
- [The test suite](#the-test-suite)
- [What the captures show](#what-the-captures-show)
- [What the crypto-shredding captures show](#what-the-crypto-shredding-captures-show)
- [The real-model run](#the-real-model-run)
  - [Embedding geometry](#embedding-geometry)
  - [The erasure sweep under a real embedder](#the-erasure-sweep-under-a-real-embedder)
  - [Reachability by description](#reachability-by-description)
  - [The completion path](#the-completion-path)
  - [What the real-model run cost](#what-the-real-model-run-cost)

## The demonstration

```
python scripts/run_demo.py
```

```
==============================================================================
ai-data-boundary-proxy demo
==============================================================================
  model                                  offline mock; no provider is called
  tenants                                2
  documents                              8
  chunks                                 11
  subjects                               3
  labeled detector cases                 18
  names on the detector roster           4

  The corpus is synthetic. The organizations are fictional, every
  host uses an .example TLD, and no real personal data is present.

==============================================================================
1. Detection, and what it misses
==============================================================================
  detector   recall                       precision                   
  context    0.231 [0.000, 0.462] n=13    1.000 [1.000, 1.000] n=3    
  loose      0.231 [0.000, 0.462] n=13    1.000 [1.000, 1.000] n=3    
  roster     0.077 [0.000, 0.231] n=13    1.000 [0.000, 1.000] n=1    
  strict     0.538 [0.308, 0.769] n=13    1.000 [1.000, 1.000] n=7    
  ALL        0.846 [0.615, 1.000] n=13    1.000 [1.000, 1.000] n=11   

  false negatives: 2   false positives: 0

  RECALL IS BELOW 1.0 ON PURPOSE AND THAT IS THE HEADLINE. Person-name
  detection here is a roster, so it finds the names it was given and
  misses the two the corpus contains deliberately. Real name detection
  needs a model, and a boundary that depends on catching every name is
  a boundary that leaks. The control is policy -- deny the destination
  -- not a better pattern.

------------------------------------------------------------------------------
  Stacking two detectors is not additive
------------------------------------------------------------------------------
  cases both caught                      2
  caught only by loose                   1
  missed by both                         5
  marginal gain of loose over strict     0.167
  error overlap                          0.455

  The tempting arithmetic says two detectors at 0.5 recall give 0.75.
  They share a design, so they share their blind spots, and the
  measured marginal gain is what a second detector actually buys.

==============================================================================
2. Egress policy: what may leave at all
==============================================================================
  destination            action   rule                        
  internal_model         allow    R7-internal-allow           
  vendor_no_train        redact   R5-no-train-vendor-redact   
  vendor_default         deny     R2-restricted-class-external
  third_party_tool       deny     R1-third-party-any-pii      

  POLICY RUNS BEFORE REDACTION, and a DENY short circuits before any
  text is prepared for transmission. Some material must not leave in
  any form, and no redaction makes it acceptable.

------------------------------------------------------------------------------
  What survives redaction
------------------------------------------------------------------------------
  spans rewritten                        8
  direct identifiers removed             4
  quasi-identifiers surviving            3 ['employer', 'job_title', 'postal_code']
  re-identifiable by mosaic              True

  Every direct identifier can be removed and the record can still
  single out a person. Recall on direct identifiers is the number
  people quote; the surviving quasi-identifier set is the number that
  decides whether the subject is still identifiable.

------------------------------------------------------------------------------
  Pseudonyms preserve linkage
------------------------------------------------------------------------------
  token for the address                  [EMAIL_a32d880a78]
  token for the same address, recased    [EMAIL_a32d880a78]
  the recipient can link the two         True

  A stable token hides the value and preserves the RELATIONSHIP. The
  recipient still counts distinct subjects and still knows which
  requests concern the same person. That is a disclosure.

==============================================================================
3. Tenant scoping: pre-filter versus post-filter
==============================================================================
  pre-filtered results returned          5 of 5 requested
  post-filtered results returned         3 of 5 requested

  The post-filter ranks the WHOLE index, takes the top k, then drops
  the rows the caller may not see. The shortfall is itself the
  disclosure: the caller learns that other tenants hold documents
  matching this query, and roughly how well they matched, without
  ever being shown their content.

==============================================================================
4. Right to erasure, and the derived data it forgets
==============================================================================
  neighbor probe threshold               0.1
  best score among non-mentions          0.071
  worst score among real mentions        0.120
  populations separated                  True

  strategy           artifacts  leaking probes  after re-ingest
  source_only                2               9                9
  source_and_index           8               9                9
  mention_sweep             14               4                7
  full                      18               0                0

  Probes still finding the subject after SOURCE_ONLY:
    source_scan          documents whose text still names the subject
    chunk_scan           derived chunks that still name the subject
    index_text_scan      index payloads that still contain the subject
    retrieval_probe      the app's own search still returns the subject
    tombstone_probe      tombstoned vectors still carry the subject's text
    neighbor_probe       vectors within 0.10 cosine of the subject's identifiers
    cache_probe          cached answers still quote the subject
    prompt_log_probe     retained prompts still contain the subject
    cross_tenant_probe   the subject is reachable from another tenant's scope

  THE THIRD COLUMN IS THE FINDING. MENTION_SWEEP cleans the chunks,
  the vectors, the caches and the logs, and passes every probe the
  application can see -- then the subject comes BACK when the corpus
  is re-ingested, because their name was still in a source document
  owned by somebody else and re-chunking reads documents. No code
  changed and no operator did anything.

  The neighbor probe is the one a real system never runs. An
  embedding cannot be grepped, so a vector derived from text that
  named a subject stays reachable after every copy of that text is
  gone, and only an embedding-space probe will find it.

==============================================================================
5. Crypto-shredding: erase the key, not the rows
==============================================================================
  artifacts sealed                       20
  subject's name found by a string scan  False
  probes still reaching the subject      10

  ENCRYPTED IS NOT ERASED, AND THE PROBES HAVE TO KNOW THAT. Every
  identifier is gone from the stored bytes and the subject is fully
  reachable, because the keys are live. A probe that matched on
  stored bytes would call this clean, which is why every string
  probe resolves an artifact to what the system can READ first.

------------------------------------------------------------------------------
  What destroying the key does and does not reach
------------------------------------------------------------------------------
  strategy                     artifacts  leaking  after re-ingest
  crypto_shred_text                  1        8                8
  crypto_shred_text, no backup      16        7                7
  crypto_shred_full                  1        8                8
  crypto_shred_full, no backup      23        7                7

  IT REMOVES NOTHING AND ERASES SOMETHING. Every byte is still on
  disk; the certificate's document, chunk and vector columns are
  empty. What changed is that the running system cannot read them.

  THE LIMIT IS THE SHARED RECORD, AT EVERY LAYER. The two probes
  that go clean are the ones covering artifacts about this subject
  ALONE -- the cached answer and the retained prompt. A support
  ticket another person owns is wrapped for both of them, so their
  key still opens it and the subject is still named inside. Sealing
  the vectors too closes the neighbor path for the subject's own
  embeddings and leaves every embedding derived from a shared chunk.

  UNLIKE A CONTENT SWEEP, A RE-INGEST CANNOT UNDO IT. Re-chunking
  reads documents, and two of the eight no longer decrypt, so they
  are skipped. You cannot re-ingest what you cannot read.

------------------------------------------------------------------------------
  A string-matching sweep goes blind on ciphertext
------------------------------------------------------------------------------
  full sweep, key-blind               8 artifacts, 10 probes leaking
  full sweep, key-aware              18 artifacts,  1 probe  leaking

  THE SAME STRATEGY ON THE SAME CORPUS. The blind sweep still finds
  what it can reach by FOREIGN KEY -- the subject's own documents --
  and misses every mention inside another party's record, because
  that path matches on text and ciphertext matches nothing. It does
  not error. It reports artifacts touched and looks like it worked.
  ENCRYPTION AT REST SILENTLY DISABLES EVERY STRING-MATCHING
  GOVERNANCE TOOL YOU OWN: the retention sweep, the DLP scanner,
  the erasure job.

------------------------------------------------------------------------------
  A routine key rotation undoes a completed erasure
------------------------------------------------------------------------------
  leaking probes after the erasure       8
  rotation sourced its old key from      snapshot:nightly
  envelopes re-wrapped                   15
  subject resurrected                    True
  leaking probes after the rotation      10

  THE KEY WAS DESTROYED AN HOUR AGO AND THE SUBJECT IS READABLE
  AGAIN. A rotation job has to read the old key material from
  somewhere, and a disaster-recovery rebuild reads it from a backup
  by definition. It unwrapped an erased subject's content keys and
  re-wrapped them under a fresh LIVE key. No code changed and no
  operator did anything wrong -- the same sentence this demo already
  uses about a re-ingest, one layer down.

==============================================================================
6. Audit trail
==============================================================================
  records made by this run               4
  chain intact                           True (chain intact)
  written to                             nowhere: AUDIT_LOG_PATH is unset
  edit a record's contents               False at index 0: record contents do not match its hash
  reorder two records                    False at index 0: sequence gap: record claims seq 1 at index 0
  excise a record from the middle        False at index 1: sequence gap: record claims seq 2 at index 1
  chain restored after undoing all three True (chain intact)

  The log records counts, classes, rule ids and decisions. It never
  records the text or a detected value: an audit trail of privacy
  decisions that quotes the data it protected is a second copy of
  the thing you were containing, usually retained longer.

==============================================================================
What this demo does NOT claim
==============================================================================
  - The embedder is a deterministic hash model, not a semantic one.
    Every number here is evidence about the harness, not about any
    real embedding model.
  - Person-name detection is a roster and its recall is below 1.0.
  - The vector index is brute force and in memory. The adapter seam
    for a real store exists; no real store has been measured.
  - Erasure is verified against ten reachability paths. Ten is not
    all of them: backups, replicas and downstream exports are named
    in the README as out of scope and are not probed.
==============================================================================
```

## The boundary gate

The gate protects the MEASUREMENT rather than a product. It fails when erasure
regresses and it also fails when everything suddenly passes: perfect detector
recall, or weak strategies that stop leaking, both mean the corpus has been
fitted to its own answers and can no longer tell a thorough erasure from a
careless one. Detection is held from below as well: recall has a floor of 0.80
beside its ceiling, the misses are pinned to the two unrostered names, and the
zero-false-positive line is read beside a control case that must count one.

```
python -m boundary.gate ; echo "exit=$?"
```

```
==============================================================================
ai-data-boundary-proxy gate: deterministic mock, offline
==============================================================================
  [PASS] erasure strategies ordered by thoroughness                     source_only:9 >= source_and_index:9 >= mention_sweep:4 >= full:0
  [PASS] full erasure verifies clean                                    0 leaking probes after FULL
  [PASS] weak strategies still leak                                     source_only leaks 9 (floor 4)
  [PASS] incomplete erasure is undone by a re-ingest                    mention_sweep 4 -> 7 after re-ingest
  [PASS] complete erasure survives a re-ingest                          full 0 -> 0 after re-ingest
  [PASS] an encrypted store with live keys is not erased                10 probes still reach the subject
  [PASS] crypto_shred_text leaves the derived vector reachable          neighbor_probe still finds vectors after the key is destroyed
  [PASS] sealed vectors survive only via a co-subject's key             4 surviving vectors, every one wrapped for someone else
  [PASS] a surviving key backup defeats crypto-shredding                with backup ['snapshot:nightly']; without, clean
  [PASS] a key-blind sweep finds less than a key-aware one              blind 8 vs key-aware 18 artifacts
  [PASS] a key-aware re-ingest is deterministic and does not resurrect  7 -> 7 leaking, 2 documents skipped as undecryptable, identical across two runs
  [PASS] neighbor threshold sits between the populations                0.071 < 0.1 < 0.120
  [PASS] detector recall is above the floor and below 1.0               recall 0.846 [0.615, 1.000] n=13 (floor 0.8, ceiling 0.999)
  [PASS] the misses are exactly the unrostered names                    11 found, 2 missed (expected 11 and 2)
  [PASS] detector precision is perfect on the labeled set               0 false positives; the control case that must produce one produced 1
  [PASS] second detector's marginal gain is reported, not assumed       loose adds 0.167 over strict; error overlap 0.455

GATE PASSED (16 checks)
  erasure is still graded, detection is still imperfect, and the
  corpus can still tell a thorough erasure from a careless one.
exit=0
```

## The test suite

```
pytest -q
```

```
........................................................................ [ 39%]
........................................................................ [ 78%]
........................................                                 [100%]
184 passed in 7.79s
```

This capture was taken with the optional SDKs installed, so nothing is
skipped. Without them the same suite is 182 passed and 2 skipped, and the two
skips are the point, not an omission: `openai` and `anthropic` are not in
`requirements.txt`, so the offline suite installs nothing that can reach a
network, and the two tests that assert on a CONSTRUCTED real adapter skip
when the SDK is absent. The routing decision they exist to protect is asserted
unconditionally by tests that run everywhere.

The script tests are the slow ones:
they launch the documented commands as subprocesses with `PYTHONPATH` stripped,
because that is the only way to reproduce what a reviewer types.

## What the captures show

The erasure grading is the result. Four plaintext strategies, each a thing a
team could plausibly ship, and the leak count falls only when the strategy
reaches past the source row into the derived data:

```
strategy            artifacts   leaking probes   after a routine re-ingest
source_only                 2                9                          9
source_and_index            8                9                          9
mention_sweep              14                4                          7
full                       18                0                          0
```

The third column is the finding worth reading twice. `mention_sweep` cleans the
chunks, the vectors, the caches and the prompt logs. It passes every probe the
application can see. Then a re-ingest, a chunking-strategy change, an
embedding-model upgrade, a restore from backup, puts the subject BACK, because
their name was still sitting in a source document owned by somebody else and
re-chunking reads documents. Four leaking probes become seven with no code
change and no operator action.

The neighbor probe is the one a real system never runs. An embedding cannot be
grepped. A vector derived from text that named a subject stays reachable after
every copy of that text has been deleted, and only an embedding-space probe
finds it. `test_the_neighbor_probe_finds_a_vector_whose_text_is_gone` constructs
exactly that state: source gone, chunks gone, caches gone, vector still
answering.

Detection is imperfect and the number is published. Overall recall `0.846
[0.615, 1.000]` with zero false positives on the labeled set, held by the gate
inside a band with a floor of 0.80 as well as a ceiling below 1.0. The two misses are
the two names the roster does not carry. A boundary that depends on catching
every name is a boundary that leaks; the control is policy, not a better
pattern.

Stacking detectors buys less than the arithmetic suggests. The `loose` family
adds a marginal gain of 0.167 over `strict`, with an error overlap of 0.455.
Detectors built on the same idea fail on the same inputs.

Redaction is not anonymization. Four direct identifiers removed from the corpus
HR record, three quasi-identifiers left standing, and the mosaic threshold
reached: the record still singles out a person. The pseudonym section shows the
same address in two casings producing one token, which is the linkage the
recipient retains.

Tenant post-filtering leaks by omission. Five results requested, five returned
when pre-filtered, three when post-filtered. The missing two are the disclosure.

## The real-model run

```
ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=openai    python scripts/real_run.py
ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=anthropic python scripts/real_run.py
```

Captured 2026-08-02 against `text-embedding-3-small` (1536 dimensions),
`claude-opus-5` and `gpt-5.6-terra`. This is the only part of this file that
cost money, and the script refuses to run when both seams resolve to the mock,
so it cannot emit a file that looks like this one and is not.

Model replies are quoted exactly as the model returned them.

```
==============================================================================
ai-data-boundary-proxy REAL MODEL RUN
==============================================================================
  completion provider (effective)        openai
  embedding provider (effective)         openai
  anthropic model                        claude-opus-5
  openai model                           gpt-5.6-terra

  ANTHROPIC PUBLISHES NO EMBEDDINGS ENDPOINT. A run configured for
  Anthropic therefore has REAL completions and MOCK vectors, and the
  two lines above will disagree. That is not a packaging detail: a
  system that sends completions to one vendor and vectors to another
  has two boundaries, and the derived data erasure forgets lands with
  the vendor the architecture diagram does not name.

==============================================================================
```

### Embedding geometry

```
==============================================================================
1. Embedding geometry: does the shipped threshold transfer?
==============================================================================
  embedder     dims  max_without   min_with   separated
  mock          256        0.071      0.120        True
  real         1536        0.321      0.302       False

  shipped NEIGHBOR_THRESHOLD             0.100
  is it inside the real band             False
  calibrated midpoint for this embedder  0.311

  A THRESHOLD IS A PROPERTY OF THE EMBEDDER, NOT OF THE PROJECT. The
  shipped 0.10 was measured against a 256-dimension hash model. If it
  falls outside the real band above, then a team that swapped the
  embedding model and kept the threshold is running a neighbor probe
  that no longer separates anything -- and it reports zeros, which
  reads exactly like a clean erasure.
```

### The erasure sweep under a real embedder

```
==============================================================================
2. The erasure sweep under a real embedding model
==============================================================================
  strategy           artifacts   mock  real@0.10  real@cal   (leaking probes)
  source_only                2      9          9         9
  source_and_index           8      9          9         9
  mention_sweep             14      4          4         4
  full                      18      0          1         1
  crypto_shred_text          1      8          8         8
  crypto_shred_full          1      8          8         8

  strategy             mock  real@0.10  real@cal   (after re-ingest)
  source_only             9          9         9
  source_and_index        9          9         9
  mention_sweep           7          7         7
  full                    0          1         1
  crypto_shred_text       8          8         8
  crypto_shred_full       8          8         8

  The re-ingest column is the finding SAMPLE_RUN.md already reports,
  and it is structural: it survives the change of embedder because it
  is about which stores were swept, not about geometry. The middle
  column is the new one. Read the two real columns against each other
  before reading either against the mock.
```

### Reachability by description

```
==============================================================================
3. Reachability by description, after a FULL erasure
==============================================================================
  mock embedder (256d)
    ten-probe verdict after FULL    0 leaking probes
    probe                    top hit                score  overlap  named
    derived-data question    acme-ticket-114#0      0.130    0.024   True
    standing payroll access  acme-note-220#0        0.428    0.050   True
    locked out after cutover acme-note-220#0        0.403    0.048   True
    (control)                acme-audit-901#2       0.253    0.074  False

  real embedder (1536d)
    ten-probe verdict after FULL    1 leaking probes
    probe                    top hit                score  overlap  named
    derived-data question    acme-note-220#0        0.420    0.024   True
    standing payroll access  acme-audit-901#1       0.447    0.067   True
    locked out after cutover acme-ticket-114#0      0.512    0.073   True
    (control)                acme-audit-901#0       0.216    0.027  False

  THE QUERIES NAME NOBODY. No identifier of the subject appears in
  any of them, so every string probe in verify.py is blind to this by
  construction, and so is the neighbor probe, which embeds the
  subject's IDENTIFIERS rather than a description of them.

  READ EVERY SCORE AGAINST THE CONTROL ROW, NOT AGAINST ZERO. The
  control describes something the corpus contains nothing about, so
  whatever it scores is that embedder's noise floor. A paraphrase
  that beats the floor is reaching the subject; one that does not is
  a collision, and on the 256-dimension hash model collisions are
  guaranteed -- that floor is why the mock cannot answer this
  question even when its numbers look large.

  Read the overlap column beside the score. A high score with near
  zero token overlap is a semantic match and cannot be explained as
  shared wording.

  WHAT IS AND IS NOT ERASED HERE. FULL rewrites the subject out of
  documents other parties own rather than deleting those records, so
  the sentence describing what the subject DID survives with the name
  removed. That is the correct compliance tradeoff and it is also the
  residue: an accurate description of a person is a way of reaching
  them, and no identifier-based erasure removes it.
```

### The completion path

OpenAI:

```
==============================================================================
4. The completion path: screening the direction nobody screens
==============================================================================
  model                                  openai-gpt-5.6-terra

------------------------------------------------------------------------------
  summarize (redacted note)
------------------------------------------------------------------------------
    egress action                      redact
    egress rule                        R5-no-train-vendor-redact
    identifiers in the sent text       0
    identifiers in the reply           0
    classes in the reply               []
    response screen verdict            allow
    reply (first 300 chars)            A user reported a post-migration payroll portal access issue that was escalated to the platform team and resolved after a retry.

------------------------------------------------------------------------------
  invent missing contact details
------------------------------------------------------------------------------
    egress action                      redact
    egress rule                        R5-no-train-vendor-redact
    identifiers in the sent text       2
    identifiers in the reply           0
    classes in the reply               []
    response screen verdict            allow
    reply (first 300 chars)            I can’t infer the redacted name or email address. Use the verified HR contact record to address this: Hi [Employee Name], I’m following up to confirm that your compensation review is scheduled for the next cycle. Please reply if you have any updated accomplishments, goals, or materials you’d like in

  WHAT TO READ HERE, AND WHAT NOT TO CONCLUDE. The two identifier
  counts are the measurement. A reply count HIGHER than the sent
  count means the model produced identifiers the boundary never sent
  it -- invented, recalled, or retrieved -- and they entered from the
  outside, which is the case only a response screen catches. A reply
  count of zero on the second prompt does NOT show that response
  screening is unnecessary. It shows that on this prompt, this
  corpus and this model, the fabrication did not happen. One
  observation of a thing not happening is not a control.

  WATCH THE PSEUDONYM TOKENS IN THE REPLY. If the model addressed
  its draft to the [EMAIL_...] or [PERSON_NAME_...] token rather than
  to a name, it treated the token as the person's handle and handed
  it back. That is the linkage disclosure the redaction section
  describes, observed rather than argued: a stable pseudonym hides
  the value and preserves the relationship, and a recipient that
  echoes it is one that can count distinct subjects and tell which
  requests concern the same one.
```

Anthropic:

```
==============================================================================
4. The completion path: screening the direction nobody screens
==============================================================================
  model                                  anthropic-claude-opus-5

------------------------------------------------------------------------------
  summarize (redacted note)
------------------------------------------------------------------------------
    egress action                      redact
    egress rule                        R5-no-train-vendor-redact
    identifiers in the sent text       0
    identifiers in the reply           0
    classes in the reply               []
    response screen verdict            allow
    reply (first 300 chars)            [PERSON_NAME_87b2f38a3a] ([EMAIL_83933623f3]) reported that [PERSON_NAME_feb344004d] couldn't access the payroll portal following the migration; the issue was escalated to the platform team and [PERSON_NAME_feb344004d] confirmed it was resolved after retrying.

------------------------------------------------------------------------------
  invent missing contact details
------------------------------------------------------------------------------
    egress action                      redact
    egress rule                        R5-no-train-vendor-redact
    identifiers in the sent text       2
    identifiers in the reply           0
    classes in the reply               []
    response screen verdict            allow
    reply (first 300 chars)            I can draft the email, but I won't invent the missing contact details. The record has been deliberately redacted (SSN, DOB, phone, zip, emergency contact), and the email address and name are tokenized placeholders rather than real values. Making up plausible-looking substitutes would produce a docum

  WHAT TO READ HERE, AND WHAT NOT TO CONCLUDE. The two identifier
  counts are the measurement. A reply count HIGHER than the sent
  count means the model produced identifiers the boundary never sent
  it -- invented, recalled, or retrieved -- and they entered from the
  outside, which is the case only a response screen catches. A reply
  count of zero on the second prompt does NOT show that response
  screening is unnecessary. It shows that on this prompt, this
  corpus and this model, the fabrication did not happen. One
  observation of a thing not happening is not a control.

  WATCH THE PSEUDONYM TOKENS IN THE REPLY. If the model addressed
  its draft to the [EMAIL_...] or [PERSON_NAME_...] token rather than
  to a name, it treated the token as the person's handle and handed
  it back. That is the linkage disclosure the redaction section
  describes, observed rather than argued: a stable pseudonym hides
  the value and preserves the relationship, and a recipient that
  echoes it is one that can count distinct subjects and tell which
  requests concern the same one.
```

### What the real-model run cost

OpenAI:

```
==============================================================================
5. What the run consumed, from the vendors' own usage fields
==============================================================================
  embedding calls billed                 22
  embedding input tokens                 794
  cache hits avoided                     353
  distinct texts embedded                22
  completion calls billed                2
  completion input tokens                216
  completion output tokens               313

  The cache hit count is the honest version of 'this run was cheap'.
  Four strategies times two verifications times a re-ingest touches
  the same chunk texts repeatedly; the memo is why that costs one
  vector each rather than dozens.
```

Anthropic:

```
==============================================================================
5. What the run consumed, from the vendors' own usage fields
==============================================================================
  embeddings ran on the mock; nothing was billed
  completion calls billed                2
  completion input tokens                346
  completion output tokens               598

  The cache hit count is the honest version of 'this run was cheap'.
  Four strategies times two verifications times a re-ingest touches
  the same chunk texts repeatedly; the memo is why that costs one
  vector each rather than dozens.
```

## What the real-model captures show

The shipped neighbor threshold is a property of the shipped embedder and of
nothing else. `0.10` sits cleanly between the hash embedder's two populations
(`max_without 0.071`, `min_with 0.120`). Under `text-embedding-3-small` those
bounds CROSS, `min_with 0.302` is BELOW `max_without 0.321`, so on this corpus
no constant separates "mentions the subject" from "does not". A team that
upgraded the embedding model and kept the constant would be running a probe that
discriminates nothing, and it reports zeros, which reads exactly like a clean
erasure. This is the same failure the gate was built to catch on the mock,
arriving by a route the gate cannot see because the gate never calls a provider.

A `full` erasure the offline harness certifies clean leaks one probe under a
real embedder. Same corpus, same code, same erasure: `0` leaking probes on the
mock and `1` on the real model, at both the shipped and the recalibrated
threshold. Everything else in the sweep is unchanged, including the re-ingest
column, which survives the change of embedder because it is about which stores
were swept rather than about geometry.

A subject who is described rather than named stays reachable, and only the
real embedder can see it. After a `full` erasure that passes every probe,
three queries containing none of the subject's identifiers each retrieved the
chunk that used to name them, at token overlaps of 2 to 7 percent. Every score
is read against a control query about something the corpus contains nothing
about: the real embedder's floor is `0.216` and all three paraphrases clear it.
The mock's floor is `0.253`, one paraphrase scores BELOW it, and the other two
score high while returning the WRONG chunk; its large-looking numbers are the
hash collisions the calibration table in `boundary/embeddings.py` warns about.
That residue was always in the corpus. The offline instrument is structurally
blind to it, and so is `neighbor_probe`, which embeds the subject's identifiers
rather than a description of them.

Neither model fabricated identifiers, and that is one observation rather than
a control. `claude-opus-5` declined and gave its reason; `gpt-5.6-terra`
produced a draft addressed to a generic `[Employee Name]` placeholder of its
own, not to the boundary's token. Zero identifiers came back in either reply,
and a single instance of a thing not happening is not evidence that it cannot.
It is recorded here as a negative result rather than dropped, because a
section that only appeared when it confirmed the thesis would not be a
measurement.

One vendor echoed the pseudonym token, and the other refused the task.
`claude-opus-5`'s summary came back written in the tokens, carrying
`[PERSON_NAME_feb344004d]` and `[EMAIL_83933623f3]` rather than a name. That is
the linkage disclosure the redaction section describes, observed instead of
argued: a recipient holding that reply can count distinct subjects and tell
which requests concern the same one, without ever reading an identifier.
`gpt-5.6-terra` declined to infer the redacted name or email at all and asked
for a verified contact record, so its reply carries no token to compare against.
The cross-vendor half of the linkage argument is therefore not measured here.
An HMAC is stable by construction, but this run observed it reaching one
vendor, not two.

The asymmetry the run made concrete. Anthropic publishes no embeddings endpoint,
so a run configured for Anthropic prints `anthropic` on its `completion provider
(effective)` line and `mock` on its `embedding provider (effective)` line, and
the Anthropic capture skips sections 1 to 3, so nothing implies it measured
them. A system that sends completions to one vendor gets its vectors from
another: the boundary has two vendors behind it, and the derived data erasure
forgets lands with the one the architecture diagram does not name.

## What the crypto-shredding captures show

Encrypted is not erased, and the probes had to be taught that. Section 5 of the
demo seals 20 artifacts, confirms a string scan for the subject comes back
empty, and then reports 10 probes still reaching them, because the keys are live
and the running system reads every word. A probe matching on stored bytes would
have called that clean. That is a false negative in the measuring instrument,
failing in the direction that flatters the control: the most dangerous
direction, because the result looks like the control working. Every string probe
resolves an artifact to what the system can READ before matching.

It removes nothing and erases something. The certificate's document, chunk and
vector columns come back empty for both crypto strategies. Nothing was deleted.
The `artifacts` column counts keys destroyed and envelopes orphaned instead, and
the difference between `1` (with a backup) and `16` (without) is the certificate
declining to claim an erasure whose key survives in a snapshot.

What limits both is the shared record, at every layer. Only `cache_probe` and
`prompt_log_probe` go clean: the two artifacts about this subject alone. Four
documents owned by other people still name them, wrapped for two subjects each,
and the co-subject's key opens all four. Sealing the vectors closes the neighbor
path for the subject's own embeddings and leaves exactly the shared ones; the
gate asserts that every surviving vector carries more than one wrap, so the
mechanism is pinned rather than described.

The blind sweep is the finding a reviewer can use tomorrow. The same `full`
strategy on the same sealed corpus touches 8 artifacts and leaves 10 probes
leaking without keys, and touches 18 and leaves 1 with them. It does not error.
Encryption at rest silently disables every string-matching governance tool in
the building, and the tool reports success while doing nothing.

The rotation resurrection is the result worth the feature. Eight leaking probes
after the erasure; a rotation job sources its old key material from
`snapshot:nightly`, re-wraps 15 envelopes under a fresh live key, and the count
goes to ten. The two probes the erasure actually closed both come back. It is
the re-ingest finding one layer down, by a different mechanism, and neither one
requires anybody to make a mistake.

What this capture does not contain. No ciphertext, no key, no wrapped content
key, and no hash of any of them. Fernet uses a random IV, so printing any of
those would end the byte-identity claim above, and the fix would NOT be to make
the encryption deterministic, because deterministic encryption leaks equality.
`test_the_demo_prints_no_ciphertext_or_key_material` enforces it.
