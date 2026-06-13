from __future__ import annotations

import os

from openai import AsyncOpenAI

API_KEY_ENV_VARS = ("OPENAI_API_KEY",)


def resolve_api_key() -> str:
    """Resolve the OpenAI API key from the supported environment variables."""

    for env_var in API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    raise ValueError(
        "Missing OpenAI API key. Set OPENAI_API_KEY.",
    )


def build_client(api_key: str | None = None) -> AsyncOpenAI:
    """Create an OpenAI async client."""

    resolved_api_key = api_key or resolve_api_key()
    return AsyncOpenAI(api_key=resolved_api_key)
