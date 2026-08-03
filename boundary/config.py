"""Runtime configuration.

The provider switch follows the house rule used across these repos: a provider
NAME without its matching KEY silently falls back to the deterministic mock, so
the tests, the demo and the CI gate never depend on the network. A missing key
is a configuration state, not an error, and treating it as an error is how a
suite ends up green only on the maintainer's laptop.
"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # mock | anthropic | openai
    provider: str = Field(default="mock", alias="BOUNDARY_PROVIDER")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_model: str = Field(default="claude-opus-5", alias="ANTHROPIC_MODEL")
    openai_model: str = Field(default="gpt-5.6-terra", alias="OPENAI_MODEL")

    # The HMAC key for pseudonymization. Supplied by the operator; the default
    # is a published demo value and is safe only because the corpus is
    # synthetic. A real deployment that reuses it makes every token forgeable.
    pseudonym_key: str = Field(
        default="boundary-demo-key-not-secret", alias="BOUNDARY_PSEUDONYM_KEY"
    )
    pseudonym_salt: str = Field(default="", alias="BOUNDARY_PSEUDONYM_SALT")

    audit_log_path: Optional[str] = Field(default=None, alias="AUDIT_LOG_PATH")

    def effective_provider(self) -> str:
        """The provider that will actually be used.

        Returns "mock" when the named provider has no key: callers report THIS
        value rather than `provider`, so a capture can never claim a
        real-model run that did not happen.
        """
        if self.provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"
        if self.provider == "openai" and self.openai_api_key:
            return "openai"
        return "mock"

    def is_mock(self) -> bool:
        return self.effective_provider() == "mock"

    def key_bytes(self) -> bytes:
        return self.pseudonym_key.encode("utf-8")


def load_settings() -> Settings:
    return Settings()
