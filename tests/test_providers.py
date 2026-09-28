"""The real-provider seam, exercised entirely offline with fake clients.

Every adapter here takes an injected `client`, so the request shape and the
response handling are testable without a key and without a network. What these
tests protect is the part a live run cannot check for you: that the fallbacks
fall back, that the cache does not pay twice, and that the embedding provider
is reported separately from the completion provider.
"""
from __future__ import annotations

import builtins
import importlib.util

import pytest

from boundary.config import Settings
from boundary.embeddings import HashingEmbedder
from boundary.providers import (
    AnthropicModel,
    CachingEmbedder,
    OpenAIEmbedder,
    OpenAIModel,
    build_embedder,
    build_model,
    effective_embedding_provider,
)
from boundary.proxy import EchoModel


# --------------------------------------------------------------- fake clients


class _FakeEmbeddings:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.calls: list[str] = []

    def create(self, *, model, input, dimensions):
        # The width this fake returns is its OWN, not the requested one, so a
        # caller that asks for a width the model does not serve gets a mismatch
        # to detect rather than silent agreement.
        self.calls.append(input)
        vector = [0.0] * self.dimensions
        # Deterministic and text-dependent, so two different texts get two
        # different vectors and a test can tell a cache hit from a real call.
        vector[len(input) % self.dimensions] = 1.0
        return type(
            "Response",
            (),
            {
                "data": [type("Datum", (), {"embedding": vector})()],
                "usage": type("Usage", (), {"prompt_tokens": len(input)})(),
            },
        )()


class _FakeOpenAIClient:
    def __init__(self, dimensions: int = 8):
        self.embeddings = _FakeEmbeddings(dimensions)


class _FakeAnthropicMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = type("Block", (), {"type": "text", "text": self.text})()
        return type(
            "Message",
            (),
            {
                "content": [block],
                "stop_reason": self.stop_reason,
                "usage": type("Usage", (), {"input_tokens": 11, "output_tokens": 7})(),
            },
        )()


class _FakeAnthropicClient:
    def __init__(self, text: str = "ok", stop_reason: str = "end_turn"):
        self.messages = _FakeAnthropicMessages(text, stop_reason)


class _FakeChatCompletions:
    def __init__(self, text: str, finish_reason: str = "stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Msg", (), {"content": self.text})()
        choice = type("Choice", (), {"message": message, "finish_reason": self.finish_reason})()
        return type(
            "Response",
            (),
            {
                "choices": [choice],
                "usage": type(
                    "Usage", (), {"prompt_tokens": 13, "completion_tokens": 5}
                )(),
            },
        )()


class _FakeOpenAIChatClient:
    def __init__(self, text: str = "ok", finish_reason: str = "stop"):
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(text, finish_reason)})()


# ------------------------------------------------------------------- caching


def test_the_cache_pays_once_per_distinct_text():
    """Mutation check, executed in-test.

    The erasure sweep embeds the same chunk texts across four strategies and
    again after every re-ingest. Without the memo the run pays for each vector
    dozens of times; the uncached call below is what makes that concrete rather
    than asserted.
    """
    inner = HashingEmbedder()
    cached = CachingEmbedder(inner)

    first = cached.embed("Jordan Rivera")
    second = cached.embed("Jordan Rivera")
    third = cached.embed("Priya Raman")

    assert first == second
    assert cached.misses == 2
    assert cached.hits == 1
    assert third != first
    assert cached.dimensions == inner.dimensions


def test_the_cache_is_transparent_to_the_embedder_protocol():
    cached = CachingEmbedder(HashingEmbedder())
    assert cached.embed("text") == HashingEmbedder().embed("text")
    assert "hashing" in cached.name()


# ---------------------------------------------------------------- embeddings


def test_the_openai_embedder_declares_its_width_and_checks_it():
    client = _FakeOpenAIClient(dimensions=8)
    emb = OpenAIEmbedder("sk-not-real", dimensions=8, client=client)

    vector = emb.embed("hello")
    assert len(vector) == 8
    assert emb.dimensions == 8
    assert emb.usage.calls == 1
    assert client.embeddings.calls == ["hello"]


def test_a_width_mismatch_raises_rather_than_being_adopted():
    """`cosine` raises on mismatched widths, so an index half built at one
    width and half at another fails deep inside a sweep. Failing on the first
    vector instead makes the misconfiguration obvious at the call that caused it.
    """
    client = _FakeOpenAIClient(dimensions=8)
    emb = OpenAIEmbedder("sk-not-real", dimensions=16, client=client)
    with pytest.raises(ValueError, match="dimensions"):
        emb.embed("hello")


def test_an_empty_text_returns_a_zero_vector_without_calling_the_api():
    """The API rejects empty input, and an empty chunk is a real thing in a
    corpus. `cosine` returns 0.0 for a zero vector, so it simply never matches.
    """
    client = _FakeOpenAIClient(dimensions=8)
    emb = OpenAIEmbedder("sk-not-real", dimensions=8, client=client)
    assert emb.embed("   ") == tuple([0.0] * 8)
    assert client.embeddings.calls == []
    assert emb.usage.calls == 0


# -------------------------------------------------------------- model clients


def test_the_anthropic_adapter_sends_no_sampling_parameters():
    """temperature, top_p and top_k are removed on claude-opus-5 and return a
    400 if sent. Pinned here because the failure is a live-only failure.
    """
    client = _FakeAnthropicClient("Acknowledged.")
    model = AnthropicModel("sk-ant-not-real", client=client)
    assert model.complete("hello") == "Acknowledged."
    sent = client.messages.kwargs
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "top_k" not in sent
    assert sent["max_tokens"] > 0
    assert model.usage.input_tokens == 11
    assert model.usage.output_tokens == 7


def test_a_refusal_is_an_outcome_not_an_exception():
    """A refusal is HTTP 200 with no text block. For a boundary proxy the
    egress decision already happened before the model saw anything, so a
    refused completion is recorded rather than raised.
    """
    client = _FakeAnthropicClient("", stop_reason="refusal")
    model = AnthropicModel("sk-ant-not-real", client=client)
    assert model.complete("hello") == "[model refused]"
    assert model.usage.calls == 1


def test_the_openai_adapter_uses_max_completion_tokens():
    """The gpt-5 family rejects `max_tokens`. Live-only failure, pinned here."""
    client = _FakeOpenAIChatClient("Acknowledged.")
    model = OpenAIModel("sk-not-real", client=client)
    assert model.complete("hello") == "Acknowledged."
    sent = client.chat.completions.kwargs
    assert "max_completion_tokens" in sent
    assert "max_tokens" not in sent


def test_a_filtered_completion_is_reported_not_raised():
    client = _FakeOpenAIChatClient("", finish_reason="content_filter")
    model = OpenAIModel("sk-not-real", client=client)
    assert model.complete("hello") == "[content filtered]"


# ------------------------------------------------------------------ factories


# `Openai` and `anthropic` are optional and absent from
# requirements.txt, so the offline suite installs nothing that can reach a
# network. The suite must therefore pass without them. Anything that asserts on
# a CONSTRUCTED real adapter needs the SDK and is skipped when it is missing;
# the routing DECISION is what matters and is asserted unconditionally.
_HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
_HAS_OPENAI = importlib.util.find_spec("openai") is not None


def test_anthropic_with_a_key_still_embeds_on_the_mock():
    """THE ASYMMETRY. Anthropic publishes no embeddings endpoint, so a run
    configured for Anthropic has REAL completions and MOCK vectors. Reporting a
    single provider line for both seams would claim an embedding run that did
    not happen.
    """
    settings = Settings(
        BOUNDARY_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-not-real"
    )
    assert settings.effective_provider() == "anthropic"
    assert effective_embedding_provider(settings) == "mock"
    # Needs no SDK: the mock embedder is what this configuration selects.
    assert isinstance(build_embedder(settings).inner, HashingEmbedder)


@pytest.mark.skipif(not _HAS_ANTHROPIC, reason="optional SDK not installed")
def test_anthropic_with_a_key_builds_the_anthropic_adapter():
    settings = Settings(
        BOUNDARY_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-not-real"
    )
    assert isinstance(build_model(settings), AnthropicModel)


def test_openai_with_a_key_embeds_for_real():
    settings = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY="sk-not-real")
    assert effective_embedding_provider(settings) == "openai"


@pytest.mark.skipif(not _HAS_OPENAI, reason="optional SDK not installed")
def test_openai_with_a_key_builds_the_openai_adapter():
    settings = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY="sk-not-real")
    assert isinstance(build_model(settings), OpenAIModel)


def test_a_missing_sdk_is_a_sentence_not_a_traceback(monkeypatch):
    """The optional dependency is missing far more often than it is present.

    A reviewer who cloned the repo and set a provider name gets this path, so
    it returns one actionable sentence rather than six lines of import
    traceback. Simulated rather than skipped, so it runs everywhere, and it
    is the test that would have caught the CI break, because it exercises the
    absent-SDK branch on a machine where the SDK is installed.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("openai", "anthropic"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    settings = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY="sk-not-real")
    with pytest.raises(RuntimeError, match="pip install openai"):
        build_model(settings)


def test_no_key_means_the_mock_on_both_seams():
    """Mutation check, executed in-test: the same provider name with a key
    selects the real adapter, so the fallback is the missing key and not a
    hardcoded mock.
    """
    without = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY=None)
    assert effective_embedding_provider(without) == "mock"
    assert isinstance(build_embedder(without).inner, HashingEmbedder)
    assert isinstance(build_model(without), EchoModel)

    with_key = Settings(BOUNDARY_PROVIDER="openai", OPENAI_API_KEY="sk-not-real")
    assert effective_embedding_provider(with_key) == "openai"


def test_a_key_alone_never_selects_a_provider():
    """A stray environment variable must not be able to send a run to a paid
    API. The NAME selects; the key only enables.
    """
    stray = Settings(BOUNDARY_PROVIDER="mock", OPENAI_API_KEY="sk-not-real")
    assert effective_embedding_provider(stray) == "mock"
    assert isinstance(build_model(stray), EchoModel)
