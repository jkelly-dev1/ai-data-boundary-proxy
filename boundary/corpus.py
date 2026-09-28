"""A synthetic corpus built to contain the cases that break erasure.

Everything here is invented. The organizations are Acme and Vertex, every host
is under an .example TLD reserved by RFC 2606, every SSN is drawn from ranges
the SSA never issues, and every card number is a published test number. No real
personal data is in this repository and none is needed: the failure modes this
project measures are structural, and a structural failure reproduces perfectly
on synthetic data.

The Corpus is adversarial by construction. Each of these is a case a
same-tenant, same-document, delete-the-row erasure gets wrong, and each is here
on purpose:

  MENTIONED-ELSEWHERE   a subject named inside a document another subject owns.
                        Deleting the subject's own records leaves it.
  BOUNDARY-STRADDLING   a mention positioned so chunking duplicates it into two
                        overlapping chunks. Removing one leaves the other.
  ALIAS-ONLY            a document that refers to the subject only by initials
                        or email local-part, so a full-name search misses it.
  CACHED-ANSWER         a cached response quoting the subject, owned by no
                        document at all.
  LOGGED-PROMPT         a retained prompt containing the subject, which no
                        content store knows about.
  CROSS-TENANT          a mention of a Tenant A subject inside a Tenant B
                        document, which is a pre-existing isolation failure
                        that erasure then has to clean up.
  UNROSTERED-NAME       a person whose name is absent from the
                        detector roster, so measured name recall is below 1.0
                        and the README's headline stays honest.
"""
from __future__ import annotations

from dataclasses import dataclass

from boundary.models import Document, Sensitivity, Subject
from boundary.store import CacheEntry, DataStore, PromptLogEntry, build_store

TENANT_A = "acme"
TENANT_B = "vertex"

# Names the roster detector KNOWS. Kept separate from the corpus text so the
# gap between "names present" and "names detectable" is explicit.
ROSTER = (
    "Jordan Rivera",
    "Priya Raman",
    "Marcus Webb",
    "Dana Okafor",
)

# Names present in the corpus and deliberately NOT on the roster. These are why
# PERSON_NAME recall is measured below 1.0 rather than tuned to 1.0.
UNROSTERED = (
    "Ilya Sorokin",
    "Fatima Zahra",
)


def subjects() -> tuple[Subject, ...]:
    return (
        Subject(
            subject_id="subj-rivera",
            tenant_id=TENANT_A,
            identifiers=frozenset(
                {
                    "Jordan Rivera",
                    "J. Rivera",
                    "jrivera@acme.example",
                    "jrivera",
                    "482-19-7734",
                }
            ),
        ),
        Subject(
            subject_id="subj-raman",
            tenant_id=TENANT_A,
            identifiers=frozenset(
                {"Priya Raman", "praman@acme.example", "531-44-2210"}
            ),
        ),
        Subject(
            subject_id="subj-webb",
            tenant_id=TENANT_B,
            identifiers=frozenset(
                {"Marcus Webb", "mwebb@vertex.example", "605-33-9182"}
            ),
        ),
    )


def documents() -> tuple[Document, ...]:
    return (
        # ---- Tenant A, owned by Rivera. What a naive erasure finds. --------
        Document(
            doc_id="acme-hr-001",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.RESTRICTED,
            owner_subject_id="subj-rivera",
            text=(
                "Employee record for Jordan Rivera. SSN 482-19-7734. "
                "Email jrivera@acme.example, phone 512-555-0143. "
                "DOB: 1982-03-14. Sex: female. Zip: 78704. "
                "Title: Staff Platform Engineer. Employer: Acme Robotics. "
                "Compensation review scheduled for the next cycle. "
                "Emergency contact on file at 512-555-0199."
            ),
        ),
        Document(
            doc_id="acme-benefits-002",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.RESTRICTED,
            owner_subject_id="subj-rivera",
            text=(
                "Benefits enrollment for Jordan Rivera, MRN 4471902. "
                "Card on file 4111 1111 1111 1111 for dependent coverage. "
                "Account ACCT-88120455. Plan selection confirmed."
            ),
        ),
        # ---- MENTIONED-ELSEWHERE: owned by Raman, names Rivera. -----------
        Document(
            doc_id="acme-ticket-114",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.CONFIDENTIAL,
            owner_subject_id="subj-raman",
            text=(
                "Support ticket filed by Priya Raman (praman@acme.example). "
                "Reporter states that Jordan Rivera was unable to access the "
                "payroll portal after the migration. Escalated to platform. "
                "Jordan Rivera confirmed the issue resolved on retry."
            ),
        ),
        # ---- ALIAS-ONLY: refers to Rivera without the full name. ----------
        Document(
            doc_id="acme-note-220",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.INTERNAL,
            owner_subject_id="subj-raman",
            text=(
                "Design review notes. J. Rivera walked the team through the "
                "retention proposal and raised the derived-data question. "
                "Follow-up assigned to jrivera for the erasure path. "
                "Ilya Sorokin took the action on the index rebuild."
            ),
        ),
        # ---- BOUNDARY-STRADDLING: long text placing a mention near a split.
        Document(
            doc_id="acme-audit-901",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.CONFIDENTIAL,
            owner_subject_id="subj-raman",
            text=(
                "Quarterly access audit for the payroll system covering all "
                "privileged accounts and their justifications, reviewed by the "
                "security team and retained for the compliance window as "
                "required by policy and the retention schedule agreed with "
                "legal during the prior cycle of this same review process. "
                "Jordan Rivera holds standing access to the payroll export "
                "job, justified by on-call duties, and this grant was "
                "re-approved without changes during the current review window "
                "alongside every other standing grant in the same category."
            ),
        ),
        # ---- CROSS-TENANT: Tenant B document naming a Tenant A subject. ---
        Document(
            doc_id="vertex-vendor-050",
            tenant_id=TENANT_B,
            sensitivity=Sensitivity.CONFIDENTIAL,
            owner_subject_id="subj-webb",
            text=(
                "Vendor coordination log maintained by Marcus Webb "
                "(mwebb@vertex.example). Joint integration call included "
                "Jordan Rivera from Acme Robotics as the platform contact. "
                "Fatima Zahra attended for procurement."
            ),
        ),
        # ---- Tenant B, unrelated. Gives retrieval something to confuse. ----
        Document(
            doc_id="vertex-hr-003",
            tenant_id=TENANT_B,
            sensitivity=Sensitivity.RESTRICTED,
            owner_subject_id="subj-webb",
            text=(
                "Employee record for Marcus Webb. SSN 605-33-9182. "
                "Email mwebb@vertex.example. DOB: 1979-11-02. Zip: 60614. "
                "Title: Director of Procurement. Employer: Vertex Supply."
            ),
        ),
        Document(
            doc_id="acme-policy-777",
            tenant_id=TENANT_A,
            sensitivity=Sensitivity.INTERNAL,
            owner_subject_id=None,
            text=(
                "Data retention policy. Records are retained for the period "
                "required by the applicable schedule and then destroyed. "
                "Derived artifacts including search indexes and caches are in "
                "scope for destruction. Contact Dana Okafor for questions."
            ),
        ),
    )


def seed_caches(store: DataStore) -> None:
    """Cached answers and retained prompts: copies no document owns."""
    store.cache_response(
        CacheEntry(
            key="cache-payroll-access",
            tenant_id=TENANT_A,
            text=(
                "Answer: Jordan Rivera has standing access to the payroll "
                "export job, justified by on-call duties."
            ),
            source_chunk_ids=("acme-audit-901#1",),
        )
    )
    store.cache_response(
        CacheEntry(
            key="cache-benefits",
            tenant_id=TENANT_A,
            text="Answer: enrollment for MRN 4471902 is confirmed.",
            source_chunk_ids=("acme-benefits-002#0",),
        )
    )
    store.log_prompt(
        PromptLogEntry(
            entry_id="prompt-0041",
            tenant_id=TENANT_A,
            text=(
                "Context: Employee record for Jordan Rivera. SSN 482-19-7734. "
                "Question: what is the on-call rotation?"
            ),
        )
    )
    store.log_prompt(
        PromptLogEntry(
            entry_id="prompt-0042",
            tenant_id=TENANT_B,
            text="Context: Marcus Webb procurement summary. Question: renewal date?",
        )
    )


@dataclass(frozen=True)
class Corpus:
    store: DataStore
    subjects: tuple[Subject, ...]
    roster: tuple[str, ...]


def build_corpus() -> Corpus:
    subs = subjects()
    store = build_store(documents(), subs)
    seed_caches(store)
    return Corpus(store=store, subjects=subs, roster=ROSTER)


# ---------------------------------------------------------------------------
# Ground truth for detector measurement.
# ---------------------------------------------------------------------------
#
# (text, expected data classes present). Kept SEPARATE from the documents above
# so recall is measured against a hand-labeled set rather than against whatever
# the detectors happen to find, which would be circular.

LABELED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SSN 482-19-7734 on file.", ("ssn",)),
    ("Reach me at jrivera@acme.example.", ("email",)),
    ("Call 512-555-0143 after noon.", ("phone",)),
    ("Card 4111 1111 1111 1111 expires soon.", ("credit_card",)),
    ("Order 4111111111111112 shipped.", ()),          # fails Luhn: not a card
    ("Server at 10.4.19.200 is down.", ("ip_address",)),
    ("Host at 999.1.1.1 is invalid.", ()),            # invalid octets
    ("MRN 4471902 admitted Tuesday.", ("mrn",)),
    ("ACCT-88120455 was reconciled.", ("account_number",)),
    ("DOB: 1982-03-14 recorded.", ("date_of_birth",)),
    ("Zip: 78704 for delivery.", ("postal_code",)),
    ("Sex: female per the intake form.", ("sex",)),
    ("SSN 000-00-0000 is a placeholder.", ()),        # never issued
    ("Jordan Rivera signed the form.", ("person_name",)),
    ("Ilya Sorokin signed the form.", ("person_name",)),   # NOT on the roster
    ("Fatima Zahra approved the change.", ("person_name",)),  # NOT on the roster
    ("The quarterly report is attached.", ()),
    ("Ticket 4471902 was closed without an MRN label.", ()),
)
