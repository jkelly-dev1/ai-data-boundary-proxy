# ai-data-boundary-proxy

[![CI](https://github.com/jkelly-dev1/ai-data-boundary-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/jkelly-dev1/ai-data-boundary-proxy/actions/workflows/ci.yml)

A gateway between an application and a model provider, built as a personal
learning project, that enforces four controls and then measures what each one
misses: PII detection and redaction on the way out, an egress policy deciding
what may leave the boundary at all, per-tenant retrieval scoping, and a
right-to-erasure operation verified against the derived embeddings rather than
just the source rows.

The headline result is a control that fails. Deleting a subject from the source
database touches exactly one of the six places they live, and does not clear
even that one: four documents belonging to other people still name the subject.
On the shipped corpus, `DELETE FROM documents` leaves the subject reachable
through nine of the ten probes this repository runs, including the one that no
string search can reach.

The second result is worse, because it looks like success. An erasure that
cleans the chunks, the vectors, the caches and the prompt logs passes every
probe the application can see. Then a routine re-ingest (a chunking change, an
embedding-model upgrade, a restore from backup) restores the subject, because
their name was still in a source document owned by somebody else and re-chunking
reads documents. No code changed. Nobody did anything wrong.

```
strategy            artifacts   leaking probes   after a routine re-ingest
source_only                 2                9                          9
source_and_index            8                9                          9
mention_sweep              14                4                          7
full                       18                0                          0
```

The third result is that crypto-shredding, the recommended answer, does not
close it either. Encrypt every artifact under a per-subject key, destroy the
key, and the subject is erased from the records they own, and stays fully
readable inside the records other people own that happen to mention them,
because those are wrapped for two subjects and only one key went away. That
limit holds at every layer, including the embedding.

## The problem it addresses

"Delete the user" is a database operation. A RAG system spreads one document
across at least six stores, and only one of them is the database:

```
documents        the source row               <- what DELETE removes
chunks           slices of it
index            vectors derived from chunks  <- cannot be grepped
response_cache   answers that quoted it
prompt_log       prompts retained for debugging
eval_set         cases promoted from real traffic   <- NOT MODELED HERE
```

Five of those six are modeled and probed. `eval_set` is named because a real
system has one and erasure has to reach it; this repository does not implement
it, so every count below is over the five it does model.

An embedding is derived data. It is computed from text, stored separately, and
survives the deletion of its source. A vector built from a chunk that named a
person still answers questions about that person after every copy of the text
is gone, and no string search will ever find it. That is the gap this project
measures.

## What it demonstrates

Erasure is graded, not binary. Six strategies are implemented and five of them
leak, because all five are things teams actually ship. Four operate on a
plaintext store and erase by removing rows; two encrypt at rest and erase by
destroying a key. Taking the plaintext four first: `source_only` is what a
ticket usually means; `source_and_index` touches three stores and looks complete
in review; `mention_sweep` closes every string-reachable path and still loses to
a re-ingest. Only `full` survives, and only against the probes that read
content: see the crypto section for what the other two reach.

Ten probes, written independently of the eraser. Whether they are independent
of each other is measured here. The verifier does not walk the eraser's lineage
table. A verifier that reuses the eraser's map confirms its blind spots instead
of finding them, so each probe reaches for the subject a different way. What
they do share is how a subject is recognized: both match on the subject's
`identifiers`, so an alias missing from that set would blind the eraser and
every probe at once. The shipped corpus has none.

Here is each probe and the reachability path it covers:

| probe | reachability path it covers |
|---|---|
| `source_scan` | the row itself |
| `chunk_scan` | derived text, including chunks in other people's documents |
| `index_text_scan` | the payload copy stored beside the vector |
| `retrieval_probe` | what the application's own search returns |
| `tombstone_probe` | a query path that ignores delete flags |
| `neighbor_probe` | **the embedding** |
| `cache_probe` | answers that quoted the subject |
| `prompt_log_probe` | retained prompts and completions |
| `cross_tenant_probe` | reachability from a neighboring tenant's scope |
| `key_material_probe` | **a surviving copy of the subject's key** |

What the independence actually measures, on this corpus. Written independently
is not the same as behaving independently, and this repository argues elsewhere
that two detectors which always agree are one detector. The same test applied to
its own probes, across the four plaintext strategies:

```
                    source_only  source_and_index  mention_sweep  full
source_scan              leak          leak            leak       clean
index_text_scan          leak          leak            leak       clean
tombstone_probe          leak          leak            leak       clean
neighbor_probe           leak          leak            leak       clean
chunk_scan               leak          leak           clean       clean
retrieval_probe          leak          leak           clean       clean
cross_tenant_probe       leak          leak           clean       clean
cache_probe              leak          leak           clean       clean
prompt_log_probe         leak          leak           clean       clean
key_material_probe      clean         clean           clean       clean
```

Ten probes produce three distinct outcomes here, and `neighbor_probe`, the one
that reaches what no string search can, never once fires alone. Its verdict is
predicted exactly by three string probes in every plaintext strategy. That does
not make it redundant: it is the only probe that WOULD still fire if the text
were gone and the vector remained, which is the whole argument of this
repository. It means the shipped corpus never produces that state, so the
probe's distinctness is argued rather than demonstrated. A strategy that purges
text while leaving vectors intact is what would demonstrate it, and nothing here
builds one.

`key_material_probe` returns clean unconditionally on a plaintext store, so on
these four strategies only nine probes can fire at all. "Nine of ten" is nine
of the nine that are live.

Detection recall is below 1.0. Overall recall on the labeled set is `0.846
[0.615, 1.000]` with zero false positives. Person-name detection is a roster, so
it finds the names it was given and misses the two the corpus contains
deliberately. Real name detection needs a model. The correct control is policy,
which denies the destination outright, rather than a better regular
expression. The shipped R1 denies a third-party tool only what was detected,
so an unrostered name reaches it; a test pins that.

That number is held from both sides. The gate keeps recall inside a band,
`0.80 <= recall < 0.999`, and pins the misses to exactly the two unrostered
names, so a detector that stops matching is a failure and not a move toward
the middle of the band: deleting the one-line ACCOUNT_NUMBER pattern takes
recall to 0.769 and the gate goes red. The zero-false-positive line is read
beside a control case that must count one, so a counter that has stopped
counting cannot pass as perfect precision.

Stacked detectors are not additive. The tempting arithmetic says two detectors
at 0.5 recall give 0.75. They share a design, so they share their blind spots:
the measured marginal gain of the `loose` family over `strict` is 0.167 for the
`loose` family, with an error overlap of 0.455. Pairing `strict` with `context`
instead measures 0.500 at an overlap of 0.231, so what a second detector buys
depends on whether it shares the first one's design rather than on there being
two of them. This check exists because detectors that share a design produce
nested errors, and a majority vote over nested errors corrects nothing.

Every direct identifier can be removed and the record still identifies someone.
On the corpus HR record, redaction removes 4 direct identifiers and leaves 3
quasi-identifiers standing (employer, job title, postal code), which reaches the
mosaic threshold. Recall on direct identifiers is the number people quote; the
surviving quasi-identifier set is the number that decides whether the subject is
still identifiable.

Pseudonymization preserves linkage, which is a disclosure. A stable HMAC token
hides the value and keeps the relationship: the recipient cannot read the email
address but can still tell which requests concern the same person, and can count
distinct subjects. In the real-model run below, `claude-opus-5` did exactly
that: its summary came back written in the pseudonym tokens, carrying
`[PERSON_NAME_feb344004d]` for the subject. `gpt-5.6-terra` declined to use the
redacted identity at all and asked for a verified contact record instead, so the
run shows one vendor echoing the token rather than two.

Post-filtering by tenant leaks existence. Ranking the whole index and then
dropping the rows a caller may not see returns fewer than `k` results. The
shortfall tells the caller that other tenants hold matching documents and
roughly how well they matched, without ever showing the content. The application
path pre-filters; the broken variant exists so it can be tested.

## What the real-model run found

Everything above is measured against a deterministic hash embedder and a mock
model, offline. `scripts/real_run.py` runs the same measurements against
`text-embedding-3-small` and against `claude-opus-5` and `gpt-5.6-terra`. The
verbatim captures are in [SAMPLE_RUN.md](SAMPLE_RUN.md). Four things came back
that the offline harness could not have told me.

1. The shipped neighbor threshold does not transfer, and the real embedder
does not separate at all. The hash embedder's two populations are cleanly
apart. Worst genuine mention `0.120`, best non-mention `0.071`, so `0.10` sits
between them. Under `text-embedding-3-small` the bounds cross: `min_with =
0.302`, `max_without = 0.321`. No threshold separates them. A team that upgraded
the embedding model and kept the constant would be running a neighbor probe that
discriminates nothing, and it reports zeros, which reads exactly like a clean
erasure.

2. A `full` erasure the offline harness certifies clean leaks one probe under
a real embedder. Same corpus, same code, same erasure: `0` leaking probes on
the mock, `1` on the real model, at both the shipped and the recalibrated
threshold. The re-ingest finding survives the change of embedder unchanged,
because it is about which stores were swept rather than about geometry.

3. A subject who is described rather than named is reachable, and only the
real embedder can see it. After a `full` erasure that passes every probe,
three queries containing none of the subject's identifiers each retrieved the
chunk that used to name them. Every score is read against a control query about
something the corpus contains nothing about, which is that embedder's noise
floor:

```
probe                     real@1536d   mock@256d   token overlap
derived-data question       0.420 *      0.130 x       0.024
standing payroll access     0.447 *      0.428 x       0.067
locked out after cutover    0.512 *      0.403 x       0.073
(control) noise floor       0.216        0.253
                            * correct chunk    x wrong chunk
```

The real embedder ranks all three correctly and well clear of its floor, at
token overlaps of 2-7%: semantic matches, not shared wording. The mock returns
the wrong chunk for all three, and scores one of them below its own noise floor:
its large-looking numbers are hash collisions, which is what the 256-dim
calibration table in `boundary/embeddings.py` warns about. That residue was
always there. The offline instrument is structurally blind to it, and so is
`neighbor_probe`, which embeds the subject's identifiers rather than a
description of them. This is a reachability path none of the ten cover.

4. Neither model fabricated identifiers. Asked to fill in missing contact
details, `claude-opus-5` declined and said why ("fabricating an SSN, email
address, phone number, or emergency contact would create false personal data");
`gpt-5.6-terra` produced a draft addressed to a generic `[Employee Name]`
placeholder of its own. Zero identifiers came back in either reply. That does
not show a response screen is unnecessary. One instance of a thing not happening
is not evidence that it cannot happen.

One asymmetry is worth naming. Anthropic publishes no embeddings endpoint, so a
system that sends completions to Anthropic gets its vectors somewhere else. The
boundary has two vendors behind it, and the derived data erasure forgets lands
with the one the architecture diagram does not name.
`effective_embedding_provider()` reports the two seams separately for exactly
this reason, and a run configured for Anthropic prints `anthropic` on its
`completion provider (effective)` line and `mock` on its
`embedding provider (effective)` line.

What the run does not claim. One embedding model, one corpus, one prompt pair. A
live model moves between runs, so these are captures rather than a baseline: the
gate never reads them and never calls a provider.

## What crypto-shredding does and does not reach

The obvious objection to everything above is "so use crypto-shredding":
encrypt each subject's data under a key that is theirs alone, and erase them by
destroying the key. Every copy of the ciphertext goes inert at once, including
copies in backups you do not control and derived artifacts you never
enumerated. It is the standard answer and it is a good one.

It is implemented here, in two variants, measured against all ten probes.

```
strategy                     artifacts   leaking probes   after a re-ingest
crypto_shred_text                    1                8                  8
crypto_shred_text, no backup        16                7                  7
crypto_shred_full                    1                8                  8
crypto_shred_full, no backup        23                7                  7
```

Crypto-shredding removes nothing and erases something. Every byte is still on disk. The
certificate's document, chunk and vector columns come back empty, because no
document was deleted, no chunk rewritten, no vector tombstoned. What changed is
that the running system cannot read them.

The limit is the shared record, and it holds at every layer. The only two probes
that go clean are the ones covering artifacts about this subject alone: the
cached answer and the retained prompt. A support ticket another person owns is
wrapped for both of them under envelope encryption, so the owner's key still
opens it and the subject is still named inside. Sealing the vectors too
(`crypto_shred_full`) closes the neighbor path for the subject's own
embeddings, which vanish from the ranking entirely. It leaves every embedding
derived from a shared chunk, each one still openable with the co-subject's key.
A record about two people cannot be erased for one of them by destroying a key.

What it beats a content sweep at: a re-ingest cannot undo it. Re-chunking reads
documents, and two of the eight no longer decrypt, so they are skipped. You
cannot re-ingest what you cannot read.

What variant B costs. One decrypt per sealed candidate, per query, forever:
eight opens for a single tenant-scoped query on this corpus. The structural
consequence is that an encrypted index cannot carry an ANN structure, because
the comparison that would build one is the thing that needs a key. That is
reasoning, not a measurement: eleven chunks cannot demonstrate a complexity
difference and the tests count decrypt operations instead of pretending to time
one. Searchable encryption, homomorphic similarity and secure enclaves are real
answers to this and none of them are implemented here.

A string-matching sweep goes blind on ciphertext. The same `full` strategy, the
same sealed corpus, with and without keys:

```
full sweep, key-blind    8 artifacts,  10 probes leaking
full sweep, key-aware   18 artifacts,   1 probe  leaking
```

The blind sweep still finds what it reaches by foreign key, the subject's own
documents, and misses every mention inside another party's record, because that
path matches on text and ciphertext matches nothing. It does not error. It
reports artifacts touched and looks like it worked. Encryption at rest
silently disables every string-matching governance tool you own: the retention
sweep, the DLP scanner, the erasure job.

And a routine key rotation undoes a completed erasure. This is the result worth
the whole feature:

```
leaking probes after the erasure        8
rotation sourced its old key from       snapshot:nightly
envelopes re-wrapped                    15
subject resurrected                     True
leaking probes after the rotation      10
```

A rotation job has to read the old key material from somewhere, and a
disaster-recovery rebuild reads it from a backup by definition. It unwrapped an
erased subject's content keys and re-wrapped them under a fresh, live key. The
two probes the erasure had actually closed both came back. No code changed and
nobody did anything wrong: the same sentence this README already uses about a
routine re-ingest, arriving one layer down by a completely different mechanism.

`key_material_probe` is the tenth probe and it exists for this. It does not ask
the key store whether the key was destroyed, because that would be the verifier
reading the eraser's own bookkeeping; it attempts a real decrypt against the
live keyring, every snapshot and escrow, and reports which sources worked.

The cipher is not hand-rolled. Fernet from `cryptography`, used the boring way,
with envelope encryption so a shared record can be wrapped once per subject.
There are no primitives implemented in this repository.

## Claims backed by tests

| claim | test |
|---|---|
| Deleting the source row leaves the subject fully retrievable | `test_deleting_the_source_row_leaves_the_subject_fully_retrievable` |
| An incomplete erasure is undone by a routine re-ingest | `test_an_incomplete_erasure_is_undone_by_a_routine_reingest` |
| Full erasure verifies clean against every probe | `test_full_erasure_verifies_clean_against_every_probe` |
| Tombstoning a vector is not removing it | `test_tombstoning_vectors_is_not_removing_them` |
| A mention in another party's document survives owner-scoped erasure | `test_a_mention_inside_another_partys_document_survives_owner_scoped_erasure` |
| The sweep rewrites rather than deletes another party's record | `test_mention_sweep_rewrites_rather_than_deletes_another_partys_record` |
| Rewriting text without re-embedding leaves the old vector | `test_rewriting_a_chunk_without_reembedding_leaves_the_old_vector` |
| Erasing one subject does not erase another | `test_erasing_one_subject_does_not_erase_another` |
| Strategies are ordered by thoroughness | `test_strategies_are_ordered_by_thoroughness` |
| The neighbor probe finds a vector whose text is gone | `test_the_neighbor_probe_finds_a_vector_whose_text_is_gone` |
| The verifier does not consult the lineage table | `test_the_verifier_does_not_consult_the_lineage_table` |
| The published leaking-probe count holds for every plaintext strategy, and no probe goes quiet | `test_the_published_leak_counts_hold_for_every_plaintext_strategy` (mutation-checked: empty any probe's hit list and the count falls) |
| The neighbor threshold sits between the two populations | `test_the_neighbor_threshold_sits_between_the_two_populations` |
| A low-dimension embedder manufactures false neighbors | `test_a_low_dimension_embedder_manufactures_false_neighbors` |
| The dimension sweep still justifies the pinned value | `test_the_dimension_sweep_still_justifies_the_pinned_value` |
| The probe query drops the email domain | `test_the_probe_query_drops_the_email_domain` |
| The frequency baseline excludes the subject's own mentions | `test_the_frequency_baseline_excludes_the_subjects_own_mentions` |
| Person-name recall is below 1.0 | `test_person_name_recall_is_below_one_and_that_is_the_finding` |
| Overall recall stays inside its band on the labeled set: below 1.0, above the 0.80 floor, with the misses pinned to the two unrostered names and the false-positive counter shown live on a control case | `test_overall_recall_stays_inside_its_band_on_the_labeled_set` (mutation-checked: delete the ACCOUNT_NUMBER detector and recall falls to 0.769; stub the false-positive counter and the control reads zero) |
| The label rules match the label loosely and the value strictly, so "patient id: pending" is not a record number | `test_label_rules_match_the_label_loosely_and_the_value_strictly` |
| Two classes claiming one span are reduced to one before redaction | `test_two_classes_claiming_one_span_are_reduced_to_one` |
| The card span ends on its last digit | `test_the_card_span_ends_on_its_last_digit` |
| A second detector is not independent of the first | `test_a_second_detector_is_not_independent_of_the_first` |
| Completely nested errors report zero marginal gain | `test_completely_nested_errors_report_zero_marginal_gain` |
| Luhn rejects a number that only looks like a card | `test_luhn_rejects_a_number_that_only_looks_like_a_card` |
| Structurally impossible SSNs are not reported | `test_structurally_impossible_ssns_are_not_reported` |
| Restricted identifiers never reach a default-terms vendor or a third-party tool; a no-training vendor receives them redacted (R5) | `test_restricted_identifiers_never_reach_a_default_or_third_party_destination` |
| Every NEVER_EXTERNAL class is denied by R2 specifically, and a direct identifier outside the set by R3, so the rule that fires is pinned and not only the outcome | `test_every_never_external_class_is_denied_by_r2_specifically`, `test_a_direct_identifier_outside_never_external_is_denied_by_r3` (mutation-checked: disable R3 and the email case becomes redact-and-send) |
| A third-party tool receives no DETECTED personal data. A name the roster does not hold is not detected and goes out unchanged, and that limit is pinned too | `test_a_third_party_tool_receives_no_detected_personal_data`, `test_an_undetected_name_reaches_a_third_party_tool` |
| A mosaic of quasi-identifiers is denied with no direct one present | `test_a_mosaic_of_quasi_identifiers_is_denied_without_any_direct_one` |
| The policy fails closed when no rule matches | `test_the_policy_fails_closed_when_no_rule_matches` |
| A denied request forwards nothing | `test_a_denied_request_forwards_nothing` |
| The egress result reports the strategies actually applied, per class | `test_the_egress_result_reports_the_strategies_actually_applied` |
| A detector that raises fails closed, on the request and on the reply, and the decision is audited | `test_a_detector_that_raises_fails_closed`, `test_a_detector_failure_is_audited_on_both_directions` |
| Every label-driven rule -- job title, employer -- fires somewhere in the corpus, so a rule that stops matching is visible | `test_every_label_rule_class_fires_somewhere_in_the_corpus` (mutation-checked: make the employer rule's label case-sensitive and its two corpus hits disappear) |
| The response is screened as well as the request | `test_the_response_is_screened_as_well_as_the_request` |
| The response screen keys on the destination the reply came from, and the raw reply is delivered only when policy allows it | `test_the_response_screen_keys_on_the_destination_the_reply_came_from` |
| Post-filtering by tenant returns fewer results than requested | `test_post_filtering_by_tenant_returns_fewer_results_than_requested` |
| A query ignoring tombstones still returns deleted vectors | `test_a_query_that_ignores_tombstones_still_returns_deleted_vectors` |
| Spans are applied right to left so offsets stay valid | `test_spans_are_applied_right_to_left_so_offsets_stay_valid` |
| No value a detector found survives redaction in cleartext, across every corpus document | `test_no_detected_value_survives_redaction_in_any_corpus_document` (mutation-checked: apply the spans forward, or return the digits from `_partial`, and it fails) |
| A stable pseudonym lets the recipient link two requests | `test_a_stable_pseudonym_lets_the_recipient_link_two_requests` |
| Quasi-identifiers survive redaction by default | `test_quasi_identifiers_survive_redaction_by_default` |
| PARTIAL keeps exactly the last four of a card, PSEUDONYM with no key masks, DROP removes the span | `test_partial_redaction_of_a_card_keeps_only_the_last_four`, `test_pseudonym_without_a_key_falls_back_to_a_mask`, `test_drop_removes_the_span_entirely` |
| A probe with nobody to leak to reports not-applicable rather than clean | `test_a_probe_with_nobody_to_leak_to_says_so_rather_than_clean` |
| Overlapping chunks duplicate a mention near a boundary | `test_overlapping_chunks_duplicate_a_mention_near_a_boundary` |
| Owner-scoped lookup misses mentions in other documents | `test_owner_scoped_lookup_misses_mentions_in_other_documents` |
| Editing an audit record breaks the chain | `test_editing_a_record_breaks_the_chain` |
| Excising a record from the middle breaks the chain | `test_excising_a_record_from_the_middle_breaks_the_chain` |
| A record spliced in from another log, with a valid sequence number and its own valid hash, breaks the chain | `test_a_record_spliced_from_another_log_breaks_the_chain` (mutation-checked: delete the `prev_hash` comparison and only this row fails) |
| The audit log never stores the data it is about | `test_the_log_never_stores_the_data_it_is_about` |
| A serialized record carries exactly the hashed fields and the hash, so nothing reaches the file outside the integrity guarantee | `test_a_serialized_record_carries_exactly_the_hashed_fields_and_the_hash` |
| A configured AUDIT_LOG_PATH receives every record as it is made, and the demo writes the file it names | `test_a_configured_path_receives_every_record_as_it_is_made`, `test_the_demo_writes_the_audit_file_the_configuration_names` |
| A second process resumes the trail at the same path instead of erasing it, and a broken trail is not extended | `test_a_second_process_resumes_the_trail_instead_of_erasing_it` |
| SAMPLE_RUN.md holds the demo and gate output the code prints now, so a moved figure fails the suite | `test_sample_run_holds_the_demo_output_the_code_prints`, `test_sample_run_holds_the_gate_output_the_code_prints` |
| The README's tables and figures equal the demo output or the SAMPLE_RUN.md capture they are read from, and the strict/context pair equals its computation | `test_the_readme_erasure_tables_are_the_demo_output`, `test_the_readme_detection_figures_are_the_demo_output`, `test_the_readme_real_run_figures_are_the_captured_ones` |
| The suite sizes stated with and without the provider SDKs are measured with the SDKs hidden | `test_the_no_sdk_suite_figures_are_derived` |
| A second demo run into the same audit file prints the same records figure; only the file total grows | `test_a_second_run_into_the_same_audit_file_prints_the_same_records_figure` |
| The key-material probe finds a key held in escrow | `test_an_escrowed_key_is_found_by_the_key_material_probe` |
| BOUNDARY_PSEUDONYM_SALT reaches the pseudonymizer, including the proxy's own redactor | `test_the_demo_uses_the_configured_pseudonym_salt`, `test_the_proxy_redactor_applies_the_configured_salt` |
| A single observation reports the full range, not certainty | `test_a_single_observation_reports_the_full_range_not_certainty` |
| The bootstrap never touches the global random state | `test_the_bootstrap_never_touches_the_global_random_state` |
| The gate runs on the mock even with a real provider configured | `test_the_gate_runs_on_the_mock_even_with_a_real_provider_configured` |
| A provider without its key falls back to the mock | `test_a_provider_without_its_key_falls_back_to_the_mock` |
| Anthropic with a key still embeds on the mock (no embeddings endpoint) | `test_anthropic_with_a_key_still_embeds_on_the_mock` |
| A key alone never selects a provider | `test_a_key_alone_never_selects_a_provider` |
| No key means the mock on both seams | `test_no_key_means_the_mock_on_both_seams` |
| The embedding cache pays once per distinct text | `test_the_cache_pays_once_per_distinct_text` |
| A declared embedding width mismatch raises rather than being adopted | `test_a_width_mismatch_raises_rather_than_being_adopted` |
| An empty text returns a zero vector without calling the API | `test_an_empty_text_returns_a_zero_vector_without_calling_the_api` |
| The Anthropic adapter sends no sampling parameters | `test_the_anthropic_adapter_sends_no_sampling_parameters` |
| The OpenAI adapter uses `max_completion_tokens` | `test_the_openai_adapter_uses_max_completion_tokens` |
| A refusal is an outcome, not an exception | `test_a_refusal_is_an_outcome_not_an_exception` |
| A filtered completion is reported, not raised | `test_a_filtered_completion_is_reported_not_raised` |
| The documented demo command runs as documented | `test_the_documented_demo_command_runs` |
| The demo runs from any working directory | `test_the_demo_runs_from_any_working_directory` |
| The real run refuses to capture a mock | `test_the_real_run_refuses_to_capture_a_mock` |
| Sealing removes every identifier from the stored bytes | `test_sealing_removes_every_identifier_from_the_stored_bytes` |
| An envelope opens for every subject it wraps | `test_an_envelope_opens_for_every_subject_it_wraps` |
| Destroying one subject's key leaves another subject's wrap | `test_destroying_one_subjects_key_leaves_another_subjects_wrap` |
| A subject-only artifact becomes unreadable | `test_a_subject_only_artifact_becomes_unreadable` |
| A snapshot taken before the destroy still opens the data | `test_a_snapshot_taken_before_the_destroy_still_opens_the_data` |
| Purging snapshots completes the destroy | `test_purging_snapshots_completes_the_destroy` |
| The key probe is scoped to the subject's own wrap | `test_the_key_probe_is_scoped_to_the_subjects_own_wrap` |
| The key material probe attempts a real decrypt | `test_the_key_material_probe_attempts_a_real_decrypt` |
| The key probe reports not-applicable on a plaintext store | `test_the_key_probe_reports_not_applicable_on_a_plaintext_store` |
| A probe reads what the running system can read: `readable()` is asserted directly, plaintext with a live key and empty after the destroy | `test_a_probe_reads_what_the_running_system_can_read` (mutation-checked: return the stored bytes from `readable()` and it fails) |
| Encrypting a chunk does not change its vector | `test_encrypting_a_chunk_does_not_change_its_vector` |
| Sealing a vector removes the plaintext copy | `test_sealing_a_vector_removes_the_plaintext_copy` |
| Variant B closes the neighbor path for subject-only vectors | `test_variant_b_closes_the_neighbor_path_for_subject_only_vectors` |
| An unopenable vector is skipped by search, not scored zero | `test_an_unopenable_vector_is_skipped_by_search_not_scored_zero` |
| A sealed index decrypts every candidate on every query | `test_a_sealed_index_decrypts_every_candidate_on_every_query` |
| A string-matching sweep goes blind on an encrypted store | `test_a_string_matching_sweep_goes_blind_on_an_encrypted_store` |
| A destroyed key cannot be re-issued | `test_a_destroyed_key_cannot_be_re_issued` |
| Sealing omits an erased subject's wrap and records it | `test_sealing_omits_an_erased_subjects_wrap_and_records_it` |
| Rotation can override the tombstone, because that is the finding | `test_rotation_can_override_the_tombstone_because_that_is_the_finding` |
| A key-aware re-ingest skips what it cannot decrypt | `test_a_key_aware_reingest_skips_what_it_cannot_decrypt` |
| The key-aware re-ingest is deterministic | `test_the_key_aware_reingest_is_deterministic` |
| A rotation from the live keyring cannot resurrect | `test_a_rotation_from_the_live_keyring_cannot_resurrect_an_erased_subject` |
| A rotation from a snapshot resurrects an erased subject | `test_a_rotation_from_a_snapshot_resurrects_an_erased_subject` |
| Rotation does not re-encrypt the body | `test_rotation_does_not_re_encrypt_the_body` |
| Crypto-shredding removes nothing | `test_crypto_shredding_removes_nothing` |
| An envelope is orphaned only when no source can open it | `test_an_envelope_is_orphaned_only_when_no_source_can_open_it` |
| Crypto-shredding refuses to run on a plaintext store | `test_crypto_shredding_refuses_to_run_on_a_plaintext_store` |
| The demo prints no ciphertext or key material | `test_the_demo_prints_no_ciphertext_or_key_material` |
| The demo is byte-identical across runs | `test_the_demo_is_byte_identical_across_runs` |

## Quickstart

```
pip install -r requirements.txt
pytest -q                          # 184 tests, offline; 2 skip without
                                   # the optional provider SDKs
python -m boundary.gate            # the CI gate
python scripts/run_demo.py         # the full demonstration
```

Everything above runs offline against a deterministic mock. No key is needed and
no network call is made. The two skips are the provider-SDK tests: `openai` and
`anthropic` are optional and are not in `requirements.txt`, so nothing the
offline suite installs can reach a network. Install the extras and the count is
184 with nothing skipped. The number is re-derived by the suite itself:
`test_the_readme_test_count_is_the_collected_count` compares this paragraph to
`pytest --collect-only`.

Set `AUDIT_LOG_PATH` to write the hash-chained audit trail to a file as each
record is made; the demo's audit section names the file it wrote and re-reads
it, and says `nowhere` when the variable is unset. A file that already holds a
chain is resumed, not overwritten, so the file total the demo prints beside the
path grows with each run into the same file; its `records made by this run` line
counts only this run's records. The chain is a plain SHA-256 with no key and no
external anchor: it catches an edit made in place, not an editor who recomputes
every hash after the change, and not the removal of the last records.

The paid capture is separate, opt-in, and refuses to run on the mock rather
than producing a file that looks real:

```
pip install openai anthropic
ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=openai    python scripts/real_run.py
ENV_FILE=~/.secrets/ai.env BOUNDARY_PROVIDER=anthropic python scripts/real_run.py
```

The full run is 2 completion calls per provider and, on OpenAI only, 22
embedding calls: Anthropic publishes no embeddings endpoint, so its run embeds
on the mock. The script prints the counts from the vendors' own usage fields at
the end, so the cost claim can be checked rather than trusted.

## How a request flows

```
application
    |
    v
detect ......... three detector families plus a name roster
    |
    v
decide ......... egress policy, first match wins, DENY short circuits
    |
    v
redact ......... per-class strategy, spans applied right to left
    |
    v
forward ........ model provider
    |
    v
screen ......... the SAME detectors on the response
```

Policy runs before redaction. A proxy that redacts first has already decided
everything is sendable once rewritten, which is the wrong default: some material
must not leave in any form. The response is screened too, because personal data
arriving from a model is the same disclosure as personal data leaving; it just
crossed the boundary the other way.

## Design notes and honest limits

1. The embedder is a deterministic hash model, not a semantic one. Every number
   here is evidence about the harness and about nothing else. It places texts
   sharing tokens near each other, which is enough to demonstrate retrieval,
   neighbor probes and erasure leaks, and is not enough to claim anything about
   real embedding behavior.

2. **`DIMENSIONS = 256` is measured, not chosen.** A signed hashing embedder
   collides tokens into shared dimensions, and at 64 dimensions those collisions
   manufacture similarity between unrelated texts: putting a floor under the
   verifier's false-positive rate and making a fully-erased subject look
   reachable. The sweep is in `boundary/embeddings.py` and re-runs as a test.

3. **`NEIGHBOR_THRESHOLD = 0.10` is calibrated, not picked.** It must sit
   strictly between the worst-scoring genuine mention and the best-scoring
   non-mention (`0.071 < 0.10 < 0.120` on this corpus). A corpus or embedder
   change that invalidates it fails the suite.

4. **Person-name detection is a roster.** Measured recall is below 1.0 by
   construction. That is the finding, and tuning it away would hide it.

5. **The vector index is brute force and in memory.** The adapter seam for a
   real store exists; no real store has been measured, and pgvector, Qdrant and
   Pinecone all have their own tombstone and compaction behavior that would
   change the tombstone probe's results.

6. Ten probes are not all the probes. Backups, read replicas, downstream
   exports, warehouse copies, and any vendor that retained a prompt are all out
   of scope and unprobed. A real erasure program has to reach them, and this
   repository does not pretend to.

7. **The mosaic threshold of three quasi-identifiers is a convention**, not a
   derivation from this corpus. Real re-identification risk depends on
   population size and attribute distribution.

8. **The real-model run is one model, one corpus, one prompt pair.** Its
   findings are captures, not baselines; the gate never reads them and never
   calls a provider. In particular, "neither model fabricated identifiers" is a
   single observation of a thing not happening, which is not a control.

9. **The neighbor probe has no valid threshold under the embedder actually
   measured.** On `text-embedding-3-small` the two populations overlap
   (`min_with 0.302` below `max_without 0.321`), so no constant separates them
   on this corpus. The shipped `0.10` is correct for the shipped hash embedder
   and for nothing else. A real deployment has to derive its own from
   `separation_band()` and re-derive it whenever the embedding model changes.

10. Description-based reachability is measured but not defended against. The
    real-model run shows a subject remains retrievable by a paraphrase that
    names nobody, after an erasure every probe calls clean. There is no probe
    for it in `verify.py` and no mitigation in the eraser. Naming the gap is not
    closing it.

11. **Crypto-shredding is implemented and does not close the gap.** See the
    section above for what it does and does not reach. Key rotation models one
    operation and a disaster-recovery rebuild; there is no key hierarchy, no
    KMS adapter and no re-encryption of bodies, and none of those are needed
    for the finding.

## TODO / not yet wired (honest scope)

- No probe for description-based reachability, which the real-model run shows
  is a live path (limit 10).
- No real vector store adapter. The seam exists, unexercised.
- Crypto-shredding reaches artifacts concerning one subject and not shared
  records; nothing here closes that, and nothing can without deciding whose
  record a shared one is (limit 11).
- Key rotation models one operation and a disaster-recovery rebuild. No key
  hierarchy, no KMS adapter, no re-encryption of bodies.
- The eval-set store is NOT implemented. It is named in the six-store list
  because a production system has one, and it would be the next store to
  model; nothing in the corpus or the probes represents it today.
- Erasure is single-subject. Bulk erasure and its cost model are unaddressed.

## Scope

One of several small projects on the theme of AI systems you can trust and prove.
All fifteen siblings link back to this one:

- [prompt-injection-benchmark](https://github.com/jkelly-dev1/prompt-injection-benchmark)
- [federated-retrieval-router](https://github.com/jkelly-dev1/federated-retrieval-router)
- [least-privilege-agent](https://github.com/jkelly-dev1/least-privilege-agent)
- [llm-eval-gate](https://github.com/jkelly-dev1/llm-eval-gate)
- [citation-abstention-rag](https://github.com/jkelly-dev1/citation-abstention-rag)
- [agentic-review-gate](https://github.com/jkelly-dev1/agentic-review-gate)
- [temporal-multi-agent](https://github.com/jkelly-dev1/temporal-multi-agent)
- [typed-agent-service](https://github.com/jkelly-dev1/typed-agent-service)
- [hardened-mcp-server](https://github.com/jkelly-dev1/hardened-mcp-server)
- [vlm-extraction-integrity](https://github.com/jkelly-dev1/vlm-extraction-integrity)
- [llm-observability-stack](https://github.com/jkelly-dev1/llm-observability-stack)
- [ai-compliance-checker](https://github.com/jkelly-dev1/ai-compliance-checker)
- [airgapped-ai-bundle](https://github.com/jkelly-dev1/airgapped-ai-bundle)
- [agent-sandbox-escape](https://github.com/jkelly-dev1/agent-sandbox-escape)
- [parser-eval](https://github.com/jkelly-dev1/parser-eval)

The corpus is synthetic. The organizations are fictional, every host uses an
`.example` TLD reserved by RFC 2606, every SSN is a synthetic number not known
to belong to anyone (the areas are issuable, because the strict detector
rejects never-issued ones), and every card number is a published test value. No real personal
data is in this repository.

## License

MIT. See [LICENSE](LICENSE).
