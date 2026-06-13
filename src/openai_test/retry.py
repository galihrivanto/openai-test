from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIStatusError

T = TypeVar("T")
logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {404, 408, 409, 425, 429, 500, 502, 503, 504}


def describe_error(exc: APIStatusError) -> str:
    """Format OpenAI error metadata for logs and CLI output."""

    status_code = getattr(exc, "status_code", None)
    error_type = exc.__class__.__name__
    request_id = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id")

    return (
        f"status_code={status_code} error_type={error_type} "
        f"request_id={request_id} error={exc}"
    )


async def retry_until_success(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int | None = None,
    initial_delay_seconds: float = 1.0,
    max_delay_seconds: float = 60.0,
) -> T:
    """Retry retryable OpenAI file operations until they succeed."""

    delay = initial_delay_seconds
    last_error: APIStatusError | None = None
    attempt = 1

    while attempts is None or attempt <= attempts:
        try:
            return await operation()
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code not in RETRYABLE_STATUS_CODES:
                raise
            if attempts is not None and attempt >= attempts:
                raise

            last_error = exc
            logger.warning(
                "OpenAI request retrying attempt=%s delay_seconds=%.1f status_code=%s",
                attempt + 1,
                delay,
                status_code,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_seconds)
            attempt += 1

    if last_error is not None:
        raise last_error

    raise RuntimeError("Retry loop exited unexpectedly")
