"""Shared fixtures.

Every fixture builds a FRESH corpus. The erasure tests mutate the store and the
index destructively, so a session-scoped corpus would make test order
significant, and an erasure suite whose results depend on execution order is
measuring the wrong thing.
"""
from __future__ import annotations

import pytest

from boundary.audit import AuditLog
from boundary.corpus import ROSTER, build_corpus
from boundary.detectors import default_suite
from boundary.embeddings import HashingEmbedder, default_embedder
from boundary.erasure import build_index
from boundary.policy import PolicyEngine
from boundary.redaction import Pseudonymizer, Redactor

DEMO_KEY = b"test-key-not-secret"


@pytest.fixture
def embedder():
    return default_embedder()


@pytest.fixture
def corpus():
    return build_corpus()


@pytest.fixture
def indexed(corpus, embedder):
    """A corpus with its index built. Returns (corpus, index, embedder)."""
    index = build_index(corpus.store, embedder)
    return corpus, index, embedder


@pytest.fixture
def suite():
    return default_suite(ROSTER)


@pytest.fixture
def redactor():
    return Redactor(pseudonymizer=Pseudonymizer(DEMO_KEY))


@pytest.fixture
def policy():
    return PolicyEngine()


@pytest.fixture
def audit():
    return AuditLog()


@pytest.fixture
def frozen_audit():
    """An audit log with a fixed clock, for assertions about record hashes."""
    counter = {"n": 0}

    def clock() -> str:
        counter["n"] += 1
        return f"2026-07-31T00:00:{counter['n']:02d}+00:00"

    return AuditLog(clock=clock)
