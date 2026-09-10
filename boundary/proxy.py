"""The gateway itself: the only sanctioned path out of the boundary.

    detect -> decide -> redact -> forward -> screen the response

Two properties are enforced here rather than documented.

Fail closed. Every failure path returns DENY. A detector that raises, a rule
set with a gap, a destination the policy has no opinion about: all of them
end with nothing leaving. The alternative, where an unexpected condition
forwards the original text, is how boundaries leak in production, and it is
always the result of an exception handler written for availability.

The response is screened too. A proxy that only inspects the outbound direction
misses the case where the model returns personal data that was never sent to
it: retrieved by a tool the model called, recalled from the conversation, or
reconstructed from context. Screening one direction and calling it a boundary
is the most common design error in this category. The reply is screened
against the DESTINATION IT CAME FROM, with the same rules the request met: a
restricted identifier in a reply from a default-terms vendor is a disclosure
that vendor now holds, so the reply is refused; the same reply from the
internal model is allowed through. What the application receives is
`delivered`, which is the raw reply only when the policy allows it.

Retrieval is tenant-scoped at the SOURCE, using a pre-filter, never a
post-filter over a global ranking. See index.py for why the post-filter variant
discloses the existence of other tenants' documents even when it never shows
their content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from boundary.audit import AuditLog
from boundary.detectors import DetectorSuite, default_suite
from boundary.embeddings import Embedder
from boundary.index import SearchHit, VectorIndex
from boundary.models import (
    Action,
    Destination,
    EgressRequest,
    EgressResult,
    PolicyVerdict,
)
from boundary.policy import PolicyEngine
from boundary.redaction import Redactor, residual_classes


class ModelClient(Protocol):
    """The provider seam. The mock is the default so nothing needs a network."""

    def complete(self, prompt: str) -> str:
        ...

    def name(self) -> str:
        ...


class EchoModel:
    """Deterministic offline model.

    It ECHOES A FRAGMENT of the prompt back, which is deliberate: it gives the
    inbound screen something real to catch when a prompt carrying personal data
    is forwarded, so the response path is exercised offline rather than only
    against a paid provider.
    """

    def name(self) -> str:
        return "mock-echo-v1"

    def complete(self, prompt: str) -> str:
        head = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"Acknowledged. Context began: {head[:160]}"


@dataclass
class ProxyOutcome:
    """One full round trip through the boundary.

    `response` is the reply exactly as the model produced it, kept because the
    inbound measurement (how many identifiers came back that were never sent)
    is made on it. `delivered` is what the application may use: the reply
    unchanged when the response screen allows it, the redacted reply when the
    screen redacts, and None when the screen denies.
    """

    egress: EgressResult
    response: Optional[str]
    response_verdict: Optional[PolicyVerdict]
    response_redacted: Optional[str]
    delivered: Optional[str] = None

    @property
    def denied(self) -> bool:
        return self.egress.verdict.action is Action.DENY

    @property
    def response_blocked(self) -> bool:
        return (
            self.response_verdict is not None
            and self.response_verdict.action is Action.DENY
        )


@dataclass
class BoundaryProxy:
    """Mediates every request between the application and a model."""

    policy: PolicyEngine
    redactor: Redactor
    detectors: DetectorSuite = field(default_factory=default_suite)
    model: ModelClient = field(default_factory=EchoModel)
    audit: Optional[AuditLog] = None
    screen_responses: bool = True

    # ------------------------------------------------------------ egress

    def evaluate(self, request: EgressRequest) -> EgressResult:
        """Detect, decide, and redact, without forwarding anything."""
        try:
            findings = self.detectors.run(request.text)
        except Exception as exc:  # fail closed, deliberately broad
            verdict = PolicyVerdict(
                action=Action.DENY,
                rule_id="R0-detector-error",
                reason=f"detector raised, failing closed: {type(exc).__name__}",
            )
            return EgressResult(
                verdict=verdict,
                forwarded_text=None,
                original_text=request.text,
            )

        verdict = self.policy.decide(request, findings)

        if verdict.action is Action.DENY:
            result = EgressResult(
                verdict=verdict,
                forwarded_text=None,
                original_text=request.text,
            )
        elif verdict.action is Action.ALLOW:
            result = EgressResult(
                verdict=verdict,
                forwarded_text=request.text,
                original_text=request.text,
            )
        else:
            redacted = self.redactor.apply(request.text, findings)
            verdict.residual_classes = residual_classes(
                request.text, findings, self.redactor
            )
            applied = {
                (f.data_class, self.redactor.strategy_for(f.data_class))
                for f in redacted.applied
            }
            result = EgressResult(
                verdict=verdict,
                forwarded_text=redacted.text,
                original_text=request.text,
                strategies=tuple(sorted(applied, key=lambda pair: pair[0].value)),
            )

        if self.audit is not None:
            self.audit.record(
                event="egress",
                tenant_id=request.tenant_id,
                detail={
                    "destination": request.destination.value,
                    "action": verdict.action.value,
                    "rule_id": verdict.rule_id,
                    "findings": len(verdict.findings),
                    "residual": [c.value for c in verdict.residual_classes],
                },
            )
        return result

    def send(self, request: EgressRequest) -> ProxyOutcome:
        """Full round trip: evaluate, forward if permitted, screen the reply."""
        egress = self.evaluate(request)
        if not egress.left_the_boundary:
            return ProxyOutcome(egress, None, None, None)

        response = self.model.complete(egress.forwarded_text or "")

        if not self.screen_responses:
            return ProxyOutcome(egress, response, None, None, delivered=response)

        # The inbound direction is screened with the SAME detectors and the
        # SAME destination, because personal data arriving from the model is
        # the same disclosure as personal data leaving; it just crossed the
        # boundary the other way, and the party on the far side is the party
        # the request went to. Screening every reply as if it came from the
        # internal model would allow every reply, whatever it carried.
        inbound_findings = self.detectors.run(response)
        inbound_request = EgressRequest(
            tenant_id=request.tenant_id,
            text=response,
            destination=request.destination,
            purpose="response_screen",
        )
        verdict = self.policy.decide(inbound_request, inbound_findings)
        redacted = None
        if inbound_findings:
            redacted = self.redactor.apply(response, inbound_findings).text
        if verdict.action is Action.DENY:
            delivered = None
        elif verdict.action is Action.REDACT:
            delivered = redacted if redacted is not None else response
        else:
            delivered = response

        if self.audit is not None:
            self.audit.record(
                event="ingress",
                tenant_id=request.tenant_id,
                detail={
                    "destination": request.destination.value,
                    "action": verdict.action.value,
                    "rule_id": verdict.rule_id,
                    "findings": len(inbound_findings),
                    "classes": sorted(
                        {f.data_class.value for f in inbound_findings}
                    ),
                },
            )
        return ProxyOutcome(egress, response, verdict, redacted, delivered=delivered)

    # --------------------------------------------------------- retrieval

    def retrieve(
        self,
        index: VectorIndex,
        embedder: Embedder,
        tenant_id: str,
        query: str,
        k: int = 5,
    ) -> list[SearchHit]:
        """Tenant-scoped retrieval, pre-filtered at the source.

        `post_filter` is not exposed here on purpose. The broken variant exists
        in index.py so it can be TESTED, not so it can be called by the
        application path.
        """
        qv = embedder.embed(query)
        hits = index.search(qv, k=k, tenant_id=tenant_id, post_filter=False)
        if self.audit is not None:
            self.audit.record(
                event="retrieval",
                tenant_id=tenant_id,
                detail={"k": k, "returned": len(hits)},
            )
        return hits


def build_proxy(
    key: bytes = b"boundary-demo-key",
    roster: tuple[str, ...] = (),
    audit: Optional[AuditLog] = None,
    model: Optional[ModelClient] = None,
) -> BoundaryProxy:
    from boundary.redaction import Pseudonymizer

    return BoundaryProxy(
        policy=PolicyEngine(),
        redactor=Redactor(pseudonymizer=Pseudonymizer(key)),
        detectors=default_suite(roster),
        model=model or EchoModel(),
        audit=audit,
    )
