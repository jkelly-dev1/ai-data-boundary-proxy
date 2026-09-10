"""The gateway: fail closed, screen both directions."""
from __future__ import annotations

import pytest

from boundary.audit import AuditLog
from boundary.corpus import ROSTER, build_corpus
from boundary.detectors import DetectorSuite
from boundary.embeddings import default_embedder
from boundary.erasure import build_index
from boundary.models import (
    Action,
    DataClass,
    Destination,
    EgressRequest,
    RedactionStrategy,
)
from boundary.proxy import BoundaryProxy, EchoModel, build_proxy

KEY = b"test-key"
SSN_TEXT = "Employee Jordan Rivera, SSN 482-19-7734, jrivera@acme.example."


def test_a_denied_request_forwards_nothing():
    """DENY short circuits before any text is prepared for transmission.

    Mutation check, executed in-test: the same text to an allowed destination
    does leave, so the block is the policy and not a broken forward path.
    """
    proxy = build_proxy(key=KEY, roster=ROSTER)
    denied = proxy.send(
        EgressRequest(tenant_id="acme", text=SSN_TEXT, destination=Destination.THIRD_PARTY_TOOL)
    )
    assert denied.denied
    assert denied.egress.forwarded_text is None
    assert denied.response is None

    allowed = proxy.send(
        EgressRequest(tenant_id="acme", text=SSN_TEXT, destination=Destination.INTERNAL_MODEL)
    )
    assert allowed.egress.left_the_boundary


def test_a_detector_that_raises_fails_closed():
    """An exception must not become an allow.

    This is the failure mode that leaks in production, and it is always the
    result of an exception handler written for availability.
    """

    def explode(text):
        raise RuntimeError("detector exploded")

    proxy = build_proxy(key=KEY)
    proxy.detectors = DetectorSuite({"boom": explode})
    result = proxy.evaluate(
        EgressRequest(tenant_id="acme", text=SSN_TEXT, destination=Destination.INTERNAL_MODEL)
    )
    assert result.verdict.action is Action.DENY
    assert result.forwarded_text is None
    assert "detector" in result.verdict.rule_id


def test_forwarded_text_to_a_vendor_is_redacted_not_original():
    proxy = build_proxy(key=KEY, roster=ROSTER)
    result = proxy.evaluate(
        EgressRequest(
            tenant_id="acme", text=SSN_TEXT, destination=Destination.VENDOR_NO_TRAIN
        )
    )
    assert result.verdict.action is Action.REDACT
    assert result.forwarded_text is not None
    assert "482-19-7734" not in result.forwarded_text
    assert result.original_text == SSN_TEXT


def test_the_response_is_screened_as_well_as_the_request():
    """Personal data arriving from the model is the same disclosure.

    Mutation check, executed in-test: with screening disabled the same reply
    comes back unexamined, so the finding is the screen and not the model.
    """

    class Leaky:
        def name(self):
            return "leaky"

        def complete(self, prompt):
            return "Here is the record: SSN 482-19-7734."

    proxy = build_proxy(key=KEY, roster=ROSTER, model=Leaky())
    outcome = proxy.send(
        EgressRequest(
            tenant_id="acme", text="nothing sensitive here", destination=Destination.INTERNAL_MODEL
        )
    )
    assert outcome.response_verdict is not None
    assert outcome.response_redacted is not None
    assert "482-19-7734" not in outcome.response_redacted

    proxy.screen_responses = False
    unscreened = proxy.send(
        EgressRequest(
            tenant_id="acme", text="nothing sensitive here", destination=Destination.INTERNAL_MODEL
        )
    )
    assert unscreened.response_verdict is None


class _Leaky:
    """A model that returns a restricted identifier the request never sent."""

    def name(self):
        return "leaky"

    def complete(self, prompt):
        return "Here is the record: SSN 482-19-7734."


def test_the_response_screen_keys_on_the_destination_the_reply_came_from():
    """A reply is screened with the rules of the party that produced it.

    A restricted identifier in a reply from a default-terms vendor is refused,
    from a no-training vendor it is redacted, and from the internal model it
    is allowed, exactly as the same text would be on the way out. The raw
    reply is never what the application receives unless the policy allows
    it.

    Mutation check, executed against the proxy: screen every reply as if it
    came from the internal model and the vendor cases below report
    R7-internal-allow, which fails on the first assertion.
    """
    proxy = build_proxy(key=KEY, roster=ROSTER, model=_Leaky())
    clean = "nothing sensitive here"

    refused = proxy.send(
        EgressRequest(tenant_id="acme", text=clean, destination=Destination.VENDOR_DEFAULT)
    )
    assert refused.egress.left_the_boundary
    assert refused.response_verdict.rule_id == "R2-restricted-class-external"
    assert refused.response_blocked
    assert refused.delivered is None

    redacted = proxy.send(
        EgressRequest(tenant_id="acme", text=clean, destination=Destination.VENDOR_NO_TRAIN)
    )
    assert redacted.response_verdict.rule_id == "R5-no-train-vendor-redact"
    assert not redacted.response_blocked
    assert redacted.delivered is not None
    assert "482-19-7734" not in redacted.delivered
    assert "[REDACTED:ssn]" in redacted.delivered

    allowed = proxy.send(
        EgressRequest(tenant_id="acme", text=clean, destination=Destination.INTERNAL_MODEL)
    )
    assert allowed.response_verdict.rule_id == "R7-internal-allow"
    assert allowed.delivered == allowed.response


def test_the_egress_result_reports_the_strategies_actually_applied():
    """The result names what the redactor did per class, not a constant.

    Mutation check, executed against the proxy: report a fixed strategy and
    the three-class request below lists fewer than three different ones.
    """
    proxy = build_proxy(key=KEY)
    mixed = "SSN 482-19-7734, phone 512-555-0143, jrivera@acme.example."
    result = proxy.evaluate(
        EgressRequest(tenant_id="acme", text=mixed, destination=Destination.VENDOR_NO_TRAIN)
    )
    assert result.verdict.action is Action.REDACT
    assert result.strategies == (
        (DataClass.EMAIL, RedactionStrategy.PSEUDONYM),
        (DataClass.PHONE, RedactionStrategy.PARTIAL),
        (DataClass.SSN, RedactionStrategy.MASK),
    )

    denied = proxy.evaluate(
        EgressRequest(tenant_id="acme", text=mixed, destination=Destination.THIRD_PARTY_TOOL)
    )
    assert denied.verdict.action is Action.DENY
    assert denied.strategies == ()


def test_retrieval_is_pre_filtered_by_tenant():
    """The application path never uses the post-filter variant."""
    emb = default_embedder()
    corpus = build_corpus()
    index = build_index(corpus.store, emb)
    proxy = build_proxy(key=KEY, roster=ROSTER)
    hits = proxy.retrieve(index, emb, tenant_id="acme", query="employee record", k=5)
    assert hits
    assert all(h.tenant_id == "acme" for h in hits)
    assert len(hits) == 5


def test_every_decision_is_audited():
    audit = AuditLog()
    proxy = build_proxy(key=KEY, roster=ROSTER, audit=audit)
    proxy.send(
        EgressRequest(tenant_id="acme", text=SSN_TEXT, destination=Destination.INTERNAL_MODEL)
    )
    events = {r.event for r in audit.records}
    assert "egress" in events
    assert "ingress" in events
    assert audit.verify_chain()[0]
    for rec in audit.records:
        assert "482-19-7734" not in str(rec.detail)


def test_the_mock_model_is_deterministic():
    model = EchoModel()
    assert model.complete("hello world") == model.complete("hello world")
