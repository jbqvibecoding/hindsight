"""Hindsight brain adapter — the retrieval/storage core.

Wraps ``hindsight_api.MemoryEngine`` in-process (constructed exactly as
``hindsight_api/server.py`` does) and drives it through the shared
``AsyncRunner`` loop. This is the *only* semantic index in the unified system:
capture routes each verbatim turn into ``retain_async`` (L1 fact extraction +
entity/temporal graph), and recall runs ``recall_async`` (4-strategy RRF +
cross-encoder rerank + MMR).

If ``hindsight_api`` cannot be imported or the engine cannot initialize (no
database, missing torch, ...), the adapter marks itself unavailable and the
engine falls back to the always-on verbatim substrate. Nothing raises.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..asyncrunner import AsyncRunner
from ..types import CaptureEvent, Recalled, RecallRequest
from .base import UnifiedAdapter

logger = logging.getLogger(__name__)


class HindsightCoreAdapter(UnifiedAdapter):
    def __init__(self, runner: AsyncRunner) -> None:
        self._runner = runner
        self._engine: Any = None
        self._request_context: Any = None
        self._available = False
        self._known_banks: set[str] = set()

    @property
    def name(self) -> str:
        return "hindsight"

    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        try:
            from hindsight_api import MemoryEngine  # type: ignore
            from hindsight_api.models import RequestContext  # type: ignore
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "hindsight brain unavailable (import failed: %s); "
                "unified memory will run on the verbatim substrate only.",
                e,
            )
            return
        try:
            # Same construction path as hindsight_api/server.py — reads config
            # (DB url, LLM/embeddings providers) from the environment.
            engine = MemoryEngine(run_migrations=True)
            self._runner.run(engine.initialize(), timeout=120)
            self._engine = engine
            # internal=True skips tenant-extension auth; user_initiated marks it
            # as originating from a real request for async bookkeeping.
            self._request_context = RequestContext(internal=True, user_initiated=True)
            self._available = True
            logger.info("hindsight brain ready (in-process MemoryEngine)")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "hindsight brain failed to initialize (%s); falling back to substrate.",
                e,
            )
            self._engine = None

    def stop(self) -> None:
        if self._engine is not None:
            try:
                self._runner.run(self._engine.close(), timeout=30)
            except Exception as e:  # noqa: BLE001
                logger.debug("hindsight close failed: %s", e)
        self._engine = None
        self._available = False

    # -- capture (L1) --------------------------------------------------------

    def capture(self, event: CaptureEvent, bank_dir: Path) -> None:
        if not self._available or self._engine is None:
            return
        content = self._format_turn(event)
        if not content.strip():
            return
        # Fire-and-forget: extraction is slow and must never block the turn.
        self._runner.submit(
            self._engine.retain_async(
                bank_id=event.bank,
                content=content,
                context=f"conversation turn (session={event.session_key})",
                request_context=self._request_context,
            )
        )

    @staticmethod
    def _format_turn(event: CaptureEvent) -> str:
        parts = []
        if event.user:
            parts.append(f"User said: {event.user}")
        if event.assistant:
            parts.append(f"Assistant replied: {event.assistant}")
        return "\n".join(parts)

    # -- recall --------------------------------------------------------------

    def recall_enrich(self, req: RecallRequest, bank_dir: Path) -> list[Recalled]:
        if not self._available or self._engine is None or not req.query:
            return []
        try:
            question_date = self._parse_date(req.question_date)
            fact_types = [req.fact_type] if req.fact_type else None
            result = self._runner.run(
                self._engine.recall_async(
                    bank_id=req.bank,
                    query=req.query,
                    max_tokens=4096,
                    fact_type=fact_types,
                    question_date=question_date,
                    request_context=self._request_context,
                ),
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("hindsight recall failed: %s", e)
            return []
        out: list[Recalled] = []
        for rank, fact in enumerate(getattr(result, "results", []) or []):
            text = getattr(fact, "text", "") or ""
            if not text:
                continue
            out.append(
                Recalled(
                    text=text,
                    source="hindsight",
                    score=float(getattr(fact, "activation", 0.0) or 0.0),
                    fact_type=getattr(fact, "fact_type", "") or "",
                    metadata={"rank": rank},
                )
            )
            if len(out) >= req.limit:
                break
        return out

    @staticmethod
    def _parse_date(value: str) -> Any:
        if not value:
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    # -- reflect (used by the /reflect endpoint) -----------------------------

    def reflect(self, bank: str, query: str) -> str:
        if not self._available or self._engine is None:
            return ""
        try:
            result = self._runner.run(
                self._engine.reflect_async(
                    bank_id=bank, query=query, request_context=self._request_context
                ),
                timeout=60,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("hindsight reflect failed: %s", e)
            return ""
        return getattr(result, "answer", "") or getattr(result, "response", "") or ""
