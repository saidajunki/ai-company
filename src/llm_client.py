"""LLM Client — OpenRouter Chat Completions API wrapper.

Provides:
- LLMResponse / LLMError dataclasses
- LLMClient: synchronous httpx client with retry logic
- is_retryable: helper to classify retryable HTTP status codes

Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 1.7
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 3
BACKOFF_DELAYS = [1, 2, 4]  # seconds — exponential backoff


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """LLM呼び出しの成功結果."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str


@dataclass
class LLMError:
    """LLM呼び出しのエラー."""

    error_type: str  # "timeout", "rate_limit", "api_error", "budget_exceeded"
    message: str
    status_code: int | None = None
    retryable: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_retryable(status_code: int) -> bool:
    """Return True for 429 (rate limit) and 5xx (server errors)."""
    return status_code == 429 or 500 <= status_code <= 599


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Synchronous OpenRouter Chat Completions API client.

    Uses httpx for HTTP, with exponential-backoff retry for transient errors.

    Args:
        api_key: OpenRouter API key (``OPENROUTER_API_KEY``).
        model: Model identifier (e.g. ``google/gemini-3-flash``).
        timeout: Request timeout in seconds (default 30).
    """

    def __init__(self, api_key: str, model: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse | LLMError:
        """Send a chat completion request to OpenRouter.

        Retries up to 3 times on retryable errors (429, 5xx) with
        exponential backoff (1s, 2s, 4s).

        Args:
            messages: OpenAI-compatible message list
                      (e.g. ``[{"role": "user", "content": "hello"}]``).

        Returns:
            ``LLMResponse`` on success, ``LLMError`` on failure.
        """
        payload = {
            "model": self.model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: LLMError | None = None

        for attempt in range(MAX_RETRIES + 1):  # 0, 1, 2, 3 → initial + 3 retries
            try:
                response = self._client.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers=headers,
                )
            except httpx.TimeoutException:
                logger.warning("OpenRouter request timed out (attempt %d)", attempt + 1)
                return LLMError(
                    error_type="timeout",
                    message=f"Request timed out after {self.timeout}s",
                    retryable=False,
                )
            except httpx.HTTPError as exc:
                # Connection errors are retryable
                logger.warning(
                    "OpenRouter connection error (attempt %d): %s", attempt + 1, exc,
                )
                last_error = LLMError(
                    error_type="api_error",
                    message=str(exc),
                    retryable=True,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_DELAYS[attempt])
                    continue
                return last_error

            # --- Got an HTTP response ---
            if response.status_code == 200:
                return self._parse_success(response)

            # Build error for non-200
            retryable = is_retryable(response.status_code)
            error_type = "rate_limit" if response.status_code == 429 else "api_error"
            last_error = LLMError(
                error_type=error_type,
                message=f"HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
                retryable=retryable,
            )

            if retryable and attempt < MAX_RETRIES:
                logger.warning(
                    "Retryable error %d (attempt %d/%d), backing off %ds",
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES + 1,
                    BACKOFF_DELAYS[attempt],
                )
                time.sleep(BACKOFF_DELAYS[attempt])
                continue

            # Non-retryable or retries exhausted
            logger.error(
                "OpenRouter API error: %d (retryable=%s)", response.status_code, retryable,
            )
            return last_error

        # Should not reach here, but just in case
        assert last_error is not None
        return last_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_success(self, response: httpx.Response) -> LLMResponse | LLMError:
        """Extract content and token usage from a successful API response."""
        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["message"]["content"],
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=data.get("model", self.model),
                finish_reason=choice.get("finish_reason", "unknown"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Failed to parse OpenRouter response: %s", exc)
            return LLMError(
                error_type="api_error",
                message=f"Failed to parse response: {exc}",
                status_code=200,
                retryable=False,
            )
