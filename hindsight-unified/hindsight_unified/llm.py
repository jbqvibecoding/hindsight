"""Minimal stdlib LLM client for the sidecar.

The sidecar has no third-party dependencies and this keeps it that way: an
``urllib`` POST to an Anthropic-Messages-shaped endpoint, nothing else. It is
**off unless configured** — no key or no model means every LLM-dependent
feature degrades to its non-LLM path rather than erroring.

Two conventions the rest of the package relies on:

* ``complete()`` returns ``None`` on failure and a string on success, and the
  distinction is load-bearing. ``None`` means "the call did not happen"; an
  empty string means "the model answered with nothing". Consolidation must not
  treat an outage as "nothing durable to keep" and seal entries on the strength
  of calls that never ran.
* ``complete_json()`` parses defensively through three tiers, because a model
  asked for JSON returns a fenced block, or a bare object with prose around it,
  more often than anyone would like.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_API_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _env(name: str, *fallbacks: str) -> str:
    for key in (name, *fallbacks):
        value = os.environ.get(key)
        if value and value.strip():
            return value.strip()
    return ""


class LLMClient:
    """Env-gated completion client. Never raises from ``complete``."""

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = api_key or _env("UNIFIED_MEMORY_LLM_API_KEY", "ANTHROPIC_API_KEY")
        # Deliberately no default model: pinning one here would silently commit
        # every deployment to a model the operator never chose, and go stale.
        self._model = model or _env("UNIFIED_MEMORY_LLM_MODEL")
        self._base_url = base_url or _env("UNIFIED_MEMORY_LLM_BASE_URL") or DEFAULT_BASE_URL
        self._timeout = timeout
        self._max_retries = max_retries

    def available(self) -> bool:
        """True when both a key and a model are configured."""
        return bool(self._api_key and self._model)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str | None:
        """One completion. ``None`` means the call failed, not that it was empty."""
        if not self.available():
            return None

        body = json.dumps(
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            request = urllib.request.Request(
                self._base_url,
                data=body,
                headers={
                    "content-type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": DEFAULT_API_VERSION,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return _text_of(payload)
            except urllib.error.HTTPError as e:
                detail = ""
                # The body is for the log line only, so failing to read it must
                # not mask the HTTP error we are actually handling.
                with contextlib.suppress(Exception):
                    detail = e.read().decode("utf-8", errors="replace")[:300]
                last_error = e
                # 4xx other than 429 will not get better by retrying.
                if e.code != 429 and 400 <= e.code < 500:
                    logger.warning("LLM call rejected (%s): %s", e.code, detail)
                    return None
                logger.debug("LLM call failed (%s), retrying: %s", e.code, detail)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.debug("LLM call failed, retrying: %s", e)
            if attempt < self._max_retries - 1:
                time.sleep(2**attempt)

        logger.warning("LLM call gave up after %d attempts: %s", self._max_retries, last_error)
        return None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        """A completion parsed as a JSON object, or ``None``.

        ``None`` covers both a failed call and an unparseable answer: in either
        case the caller learned nothing, which is what it must not confuse with
        a successful empty result.
        """
        raw = self.complete(
            system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )
        if raw is None:
            return None
        parsed = parse_json_object(raw)
        if parsed is None:
            logger.warning("LLM answer was not parseable JSON: %.200s", raw)
        return parsed


def _text_of(payload: dict[str, Any]) -> str:
    """Concatenate the text blocks of a Messages response."""
    blocks = payload.get("content") or []
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "".join(parts).strip()


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Three-tier JSON extraction: exact, fenced block, then first bare object.

    Models asked for JSON wrap it in prose or a code fence often enough that a
    single ``json.loads`` throws away usable answers.
    """
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    candidates = [text]
    fenced = _FENCED_JSON.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    bare = _BARE_JSON.search(text)
    if bare:
        candidates.append(bare.group(0))
    return candidates
