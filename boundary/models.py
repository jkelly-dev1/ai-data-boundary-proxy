"""Domain types for the data-boundary proxy.

Everything that crosses the boundary is described here, and the vocabulary is
deliberate. Three distinctions in this file carry the whole project:

1. DATA CLASS vs FINDING. A data class is what a piece of text IS (an SSN, an
   email). A finding is a CLAIM by one detector that a specific span holds one.
   Findings are evidence, not truth: they carry the detector that produced them
   so recall and inter-detector correlation can be measured per detector rather
   than assumed.

2. REDACTION vs ERASURE. Redaction rewrites text on its way OUT of the
   boundary. Erasure removes a subject from everything already INSIDE it,
   including artifacts derived from the original text. The second is a
   different and much harder operation, and conflating them is the mistake this
   project exists to demonstrate.

3. SOURCE vs DERIVED. A chunk is derived from a document; an embedding is
   derived from a chunk; an index entry is derived from an embedding. Deleting
   the source leaves every derivative in place. `Derivation` records that
   lineage so erasure has something to follow.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DataClass(str, Enum):
    """What a span of text is, for policy purposes.

    Split into DIRECT identifiers, which name a person on their own, and
    QUASI identifiers, which do not; until they are combined. The split is
    load-bearing: a redactor that removes every direct identifier and leaves
    the quasi-identifiers intact still permits re-identification, and
    `boundary/metrics.py` measures exactly that residual.
    """

    # Direct identifiers.
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    MRN = "mrn"                      # medical record number
    ACCOUNT_NUMBER = "account_number"
    PERSON_NAME = "person_name"
    IP_ADDRESS = "ip_address"

    # Quasi-identifiers: harmless alone, identifying in combination.
    DATE_OF_BIRTH = "date_of_birth"
    POSTAL_CODE = "postal_code"
    SEX = "sex"
    JOB_TITLE = "job_title"
    EMPLOYER = "employer"

    @property
    def is_direct(self) -> bool:
        return self in _DIRECT_CLASSES

    @property
    def is_quasi(self) -> bool:
        return not self.is_direct


_DIRECT_CLASSES = frozenset(
    {
        DataClass.SSN,
        DataClass.EMAIL,
        DataClass.PHONE,
        DataClass.CREDIT_CARD,
        DataClass.MRN,
        DataClass.ACCOUNT_NUMBER,
        DataClass.PERSON_NAME,
        DataClass.IP_ADDRESS,
    }
)


class Sensitivity(str, Enum):
    """Policy tier. Ordered, and the ordering is used for comparisons."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return _SENSITIVITY_RANK[self]


_SENSITIVITY_RANK = {
    Sensitivity.PUBLIC: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.CONFIDENTIAL: 2,
    Sensitivity.RESTRICTED: 3,
}


class Destination(str, Enum):
    """Where an egress request is going.

    A destination is not just a URL: it carries the trust properties the policy
    engine reasons about. A vendor that trains on submitted data and one that
    contractually does not are different destinations even at the same
    sensitivity tier.
    """

    INTERNAL_MODEL = "internal_model"          # self-hosted, inside the boundary
    VENDOR_NO_TRAIN = "vendor_no_train"        # external, contractual no-training
    VENDOR_DEFAULT = "vendor_default"          # external, default terms
    THIRD_PARTY_TOOL = "third_party_tool"      # webhook, plugin, arbitrary host


class Action(str, Enum):
    """What the policy engine decided to do with a request."""

    ALLOW = "allow"
    REDACT = "redact"
    DENY = "deny"


class RedactionStrategy(str, Enum):
    """How a detected span is rewritten.

    PSEUDONYM is deliberately included and deliberately called out as unsafe in
    a specific way: a stable pseudonym preserves linkage, so the same subject
    is still trackable across requests by whoever receives them. That is a real
    finding this project measures rather than a bug in the implementation.
    """

    MASK = "mask"                # [REDACTED:ssn]
    PARTIAL = "partial"          # ***-**-6789, keeps a suffix for usability
    PSEUDONYM = "pseudonym"      # stable HMAC token, preserves linkage
    DROP = "drop"                # remove the span entirely
    NONE = "none"                # leave it, used to measure the baseline


@dataclass(frozen=True)
class Span:
    """A half-open character range [start, end) inside a specific text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span: [{self.start}, {self.end})")

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Finding:
    """One detector's claim that `span` in some text holds `data_class`.

    `detector` is recorded on every finding because the honest question is not
    "did we find the SSN" but "which detector found it, and would a second
    detector have found it independently". Two detectors that always agree are
    one detector with extra latency, and metrics.py reports that correlation
    instead of assuming independence, because nested errors are the normal
    outcome for two detectors that share a design.
    """

    data_class: DataClass
    span: Span
    detector: str
    confidence: float = 1.0
    text: str = ""

    def key(self) -> tuple:
        """Identity for dedup across detectors: same class, same place."""
        return (self.data_class, self.span.start, self.span.end)


@dataclass(frozen=True)
class Subject:
    """A data subject: the person an erasure request is about.

    `identifiers` is the set of literal strings that denote this person in the
    corpus (name variants, email, SSN, account number). Erasure has to find the
    subject through ALL of them, because a chunk that says "J. Rivera" and one
    that says "jrivera@acme.example" are the same person and only one of them
    matches a naive name search.
    """

    subject_id: str
    identifiers: frozenset[str]
    tenant_id: str

    def matches(self, text: str) -> bool:
        low = text.lower()
        return any(ident.lower() in low for ident in self.identifiers)


@dataclass(frozen=True)
class Document:
    """A source record. The thing a naive erasure deletes and stops."""

    doc_id: str
    tenant_id: str
    text: str
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    owner_subject_id: Optional[str] = None

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """A slice of a document, and the unit that actually gets embedded.

    `doc_id` is the lineage link back to the source. The subtle case this
    project is built around: a chunk can MENTION a subject without belonging to
    that subject's document, so erasing by `owner_subject_id` alone misses it.
    """

    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str
    ordinal: int = 0


@dataclass
class IndexEntry:
    """A vector plus the metadata retrieval filters on.

    `deleted` is a TOMBSTONE rather than a removal. Real vector indexes mark
    deleted and reclaim later, which means a "deleted" vector can still be
    returned by a search that ignores the flag. Modeling the tombstone
    explicitly is what lets `verify.py` catch that class of leak instead of
    assuming the delete worked.

    `Sealed_vector` holds the only copy when it is set, and `vector` is then
    empty. That is the shape of encrypting the embedding itself: the index can
    no longer compare a query against the entry without decrypting it first, so
    search has to open every candidate it can and skip the ones it cannot.
    Typed as `object` to keep this module free of a crypto import. The value is
    a `boundary.crypto.Envelope`, and only the index and the sealing code ever
    look inside it.
    """

    chunk_id: str
    tenant_id: str
    vector: tuple[float, ...]
    deleted: bool = False
    text: str = ""
    sealed_vector: Optional[object] = None

    @property
    def sealed(self) -> bool:
        return self.sealed_vector is not None


@dataclass(frozen=True)
class Derivation:
    """One lineage edge: `derived_id` was computed from `source_id`.

    Erasure walks these edges. Without a lineage record, finding every artifact
    derived from a document means scanning everything, which is exactly what
    production systems fail to do.
    """

    source_kind: str
    source_id: str
    derived_kind: str
    derived_id: str


@dataclass
class PolicyVerdict:
    """The policy engine's decision about one egress request."""

    action: Action
    rule_id: str
    reason: str
    findings: tuple[Finding, ...] = ()
    residual_classes: tuple[DataClass, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.action is Action.DENY


@dataclass
class EgressRequest:
    """What the application wants to send outside the boundary."""

    tenant_id: str
    text: str
    destination: Destination
    purpose: str = "inference"
    request_id: str = ""


@dataclass
class EgressResult:
    """What the proxy actually did, and what left."""

    verdict: PolicyVerdict
    forwarded_text: Optional[str]
    original_text: str
    strategy: RedactionStrategy

    @property
    def left_the_boundary(self) -> bool:
        return self.forwarded_text is not None


@dataclass
class ErasureCertificate:
    """Evidence that an erasure ran, and what it claims to have removed.

    A certificate is NOT proof. It records what the erasure engine believes it
    did; `verify.py` independently probes whether the subject is still
    reachable, and the two are compared. A certificate that disagrees with the
    probes is the interesting case, and the demo prints both.
    """

    subject_id: str
    tenant_id: str
    documents_removed: tuple[str, ...] = ()
    chunks_removed: tuple[str, ...] = ()
    vectors_removed: tuple[str, ...] = ()
    caches_purged: tuple[str, ...] = ()
    mentions_rewritten: tuple[str, ...] = ()
    keys_destroyed: tuple[str, ...] = ()
    envelopes_orphaned: tuple[str, ...] = ()
    strategy: str = ""

    def total_artifacts(self) -> int:
        """How much this erasure actually touched.

        The crypto strategies REMOVE NOTHING, so their first five fields are
        empty and the count comes entirely from the last two. That asymmetry is
        the point rather than an accounting quirk: destroying one key can
        neutralize more artifacts than a sweep reaches, and the column stays
        comparable across both postures.
        """
        return (
            len(self.documents_removed)
            + len(self.chunks_removed)
            + len(self.vectors_removed)
            + len(self.caches_purged)
            + len(self.mentions_rewritten)
            + len(self.keys_destroyed)
            + len(self.envelopes_orphaned)
        )


@dataclass
class LeakProbe:
    """One independent check that a subject is no longer reachable.

    `leaked` true means the subject WAS still reachable by this probe after
    erasure claimed to be complete. Each probe names a different reachability
    path, because they fail independently: an index tombstone can be honored by
    one query path and ignored by another.
    """

    probe: str
    leaked: bool
    detail: str = ""
    hits: tuple[str, ...] = ()


@dataclass
class VerificationReport:
    """The result of probing an erasure from several directions."""

    subject_id: str
    probes: tuple[LeakProbe, ...] = field(default_factory=tuple)

    @property
    def leaked(self) -> bool:
        return any(p.leaked for p in self.probes)

    @property
    def leaking_probes(self) -> tuple[LeakProbe, ...]:
        return tuple(p for p in self.probes if p.leaked)

    def summary(self) -> str:
        n = len(self.leaking_probes)
        if n == 0:
            return f"{self.subject_id}: no probe found the subject ({len(self.probes)} run)"
        names = ", ".join(p.probe for p in self.leaking_probes)
        return f"{self.subject_id}: STILL REACHABLE via {n} probe(s): {names}"
