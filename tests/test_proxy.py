"""The gateway: fail closed, screen both directions."""
from __future__ import annotations

import pytest

from boundary.audit import AuditLog
from boundary.corpus import ROSTER, build_corpus
from boundary.detectors import DetectorSuite
from boundary.embeddings import default_embedder
from boundary.erasure import build_index
from boundary.models import Action, Destination, EgressRequest
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
