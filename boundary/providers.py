"""Real provider adapters for the two seams, and one asymmetry worth naming.

The rest of this package runs on a deterministic mock so the suite, the demo
and the gate never touch a network. This module is the other side of both
seams: a real embedding model behind `Embedder`, and a real completion model
behind `ModelClient`. Nothing here is imported by the gate, the demo or any
test that runs offline.

The asymmetry, because it is a data-boundary fact and not a packaging detail.
Anthropic publishes no embeddings endpoint. A system that sends completions to
Anthropic and needs vectors gets them somewhere else, so the boundary this
project is about has TWO vendors behind it, not one, and the derived data, the
part erasure forgets, lands with the embedding vendor rather than the one the
architecture diagram names. `effective_embedding_provider` reports that
explicitly instead of letting a caller read "provider = anthropic" and assume
the vectors went there too.

Caching is correctness here, not just thrift. The erasure sweep re-embeds the
same chunk texts across four strategies and again after each re-ingest. Without
a memo the run pays for the same vector dozens of times AND, worse for a
measurement, lets a vendor's own nondeterminism move the numbers between two
sections that are supposed to be comparing strategies, not runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from boundary.config import Settings
from boundary.embeddings import Embedder, HashingEmbedder
from boundary.proxy import EchoModel, ModelClient

# text-embedding-3-small's native width. Named rather than inlined because the
# repository's own mock is 256 and the difference between the two is one of the
# things the real run measures.
OPENAI_EMBED_DIMENSIONS = 1536


@dataclass
class Usage:
    """What a vendor said the calls consumed.

    Carried alongside the adapter rather than returned from `embed`/`complete`,
    because both of those are Protocol methods that the mock also implements
    and widening the seam for a figure only the real path has would fork it.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


# --------------------------------------------------------------- embedders


@dataclass
class CachingEmbedder:
    """Memoize an embedder by exact text.

    `misses` is the number of vectors that were actually paid for and `hits` is
    the number that were not; a capture prints both so the reported cost can be
    checked against the number of distinct texts in the corpus rather than
    trusted.
    """

    inner: Embedder
    cache: dict[str, tuple[float, ...]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def name(self) -> str:
        return f"cached({self.inner.name()})"

    def embed(self, text: str) -> tuple[float, ...]:
        cached = self.cache.get(text)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed(text)
        self.cache[text] = vector
        return vector


class OpenAIEmbedder:
    """text-embedding-3-* behind the `Embedder` protocol.

    Two deliberate behaviors:

    An empty text returns a zero vector instead of calling the API, which
    rejects empty input. `cosine` already returns 0.0 for a zero vector, so an
    empty chunk simply never matches. The same thing the mock does, and the
    behavior the index needs, since an empty chunk is a real thing in a corpus.

    `Dimensions` is declared, not discovered. The index and the verifier need a
    width before the first call, and `cosine` raises on a mismatch, so the
    declared value is asserted against the first vector the API returns instead
    of being silently adopted from it.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = OPENAI_EMBED_DIMENSIONS,
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.usage = Usage()
        if client is not None:
            self._client = client
            return
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "A real embedding run needs the 'openai' package, which is "
                "commented out of requirements.txt on purpose so the offline "
                "suite installs nothing that can reach a network.\n"
                "    pip install openai"
            ) from exc
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def name(self) -> str:
        return f"openai-{self.model}-{self.dimensions}d"

    def embed(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            return tuple([0.0] * self.dimensions)
        response = self._client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimensions
        )
        usage = getattr(response, "usage", None)
        self.usage.add(input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0))
        vector = tuple(float(v) for v in response.data[0].embedding)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"{self.model} returned {len(vector)} dimensions, "
                f"expected {self.dimensions}"
            )
        return vector


# ------------------------------------------------------------ model clients


class _RealModel(Protocol):  # pragma: no cover - typing only
    usage: Usage

    def complete(self, prompt: str) -> str: ...

    def name(self) -> str: ...


class AnthropicModel:
    """Claude behind the `ModelClient` protocol.

    No temperature, top_p or top_k: removed on claude-opus-5 and returned as a
    400 if sent. `stop_reason` is checked before `content`, because a refusal is
    a successful HTTP 200 with nothing to read, and for a boundary proxy a
    refused completion is an outcome to record rather than an exception to
    raise: the egress decision already happened before the model saw anything.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5",
        max_tokens: int = 1024,
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.usage = Usage()
        if client is not None:
            self._client = client
            return
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BOUNDARY_PROVIDER=anthropic needs the 'anthropic' package, "
                "which is commented out of requirements.txt on purpose.\n"
                "    pip install anthropic"
            ) from exc
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)

    def name(self) -> str:
        return f"anthropic-{self.model}"

    def complete(self, prompt: str) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Low effort: this is a one-sentence screening reply, not a reasoning
        # task, and effort is the only lever that bounds spend now that
        # budget_tokens is gone. Falls back on an SDK too old to know the
        # parameter instead of failing a capture over a cost control.
        try:
            message = self._client.messages.create(
                **kwargs, output_config={"effort": "low"}
            )
        except TypeError:
            message = self._client.messages.create(**kwargs)
        usage = getattr(message, "usage", None)
        self.usage.add(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        if getattr(message, "stop_reason", None) == "refusal":
            return "[model refused]"
        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        )


class OpenAIModel:
    """GPT behind the `ModelClient` protocol.

    `max_completion_tokens`, not `max_tokens`: the gpt-5 family rejects the
    older name. A content-filtered completion is reported the same way a
    refusal is on the Anthropic path: as a recorded outcome, not an error.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.6-terra",
        max_tokens: int = 1024,
        client: Optional[object] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.usage = Usage()
        if client is not None:
            self._client = client
            return
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BOUNDARY_PROVIDER=openai needs the 'openai' package, which is "
                "commented out of requirements.txt on purpose.\n"
                "    pip install openai"
            ) from exc
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def name(self) -> str:
        return f"openai-{self.model}"

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(response, "usage", None)
        self.usage.add(
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "") == "content_filter":
            return "[content filtered]"
        return choice.message.content or ""


# ----------------------------------------------------------------- factories


def effective_embedding_provider(settings: Settings) -> str:
    """Which vendor will actually compute the vectors: "openai" or "mock".

    NOT the same question as `Settings.effective_provider()`. Anthropic has no
    embeddings endpoint, so BOUNDARY_PROVIDER=anthropic embeds on the mock even
    though completions are real. A capture that printed one provider line for
    both seams would be claiming a real embedding run that did not happen.
    """
    if settings.provider == "openai" and settings.openai_api_key:
        return "openai"
    return "mock"


def build_embedder(settings: Settings, dimensions: int = OPENAI_EMBED_DIMENSIONS) -> Embedder:
    """The embedder for this configuration, memoized.

    Falls back to the hashing mock for every configuration that is not OpenAI
    with a key, including anthropic-with-a-key. See
    `effective_embedding_provider`.
    """
    if effective_embedding_provider(settings) == "openai":
        assert settings.openai_api_key is not None  # narrowed by the check above
        return CachingEmbedder(
            OpenAIEmbedder(settings.openai_api_key, dimensions=dimensions)
        )
    return CachingEmbedder(HashingEmbedder())


def build_model(settings: Settings) -> ModelClient:
    """The completion client for this configuration.

    Same iron rule as everywhere else in this repository: a provider NAME
    without its matching KEY returns the mock. A stray environment variable
    must not be able to send a demo run to a paid API.
    """
    provider = settings.effective_provider()
    if provider == "anthropic":
        assert settings.anthropic_api_key is not None
        return AnthropicModel(settings.anthropic_api_key, settings.anthropic_model)
    if provider == "openai":
        assert settings.openai_api_key is not None
        return OpenAIModel(settings.openai_api_key, settings.openai_model)
    return EchoModel()
