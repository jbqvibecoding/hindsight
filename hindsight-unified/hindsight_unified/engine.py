"""UnifiedEngine — lifecycle owner and public API for the sidecar.

Composes the always-on verbatim substrate with the contributor adapters behind
one interface. Owns:
  * the background ``AsyncRunner`` (drives the async Hindsight brain),
  * adapter start/stop,
  * the L0→L3 ``LayeredPipeline``,
  * the synchronous methods the HTTP layer calls (capture/recall/search/…).

Everything degrades: if the Hindsight brain (or any contributor) cannot load,
the substrate keeps capture and keyword recall working, and ``health`` reports
``degraded`` instead of ``ok`` so the supervisor still considers the sidecar
usable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .adapters import (
    EverosSubstrateAdapter,
    HindsightCoreAdapter,
    MemosPortabilityAdapter,
    MempalaceIndexAdapter,
    OpenVikingInjectionAdapter,
)
from .asyncrunner import AsyncRunner
from .config import Settings
from .pipeline import LayeredPipeline
from .substrate import MarkdownSubstrate, SingletonLockHeld
from .types import CaptureEvent, RecallRequest

logger = logging.getLogger(__name__)


class UnifiedEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.from_env()
        self._runner = AsyncRunner()
        self._substrate = MarkdownSubstrate(self._settings.home)

        # Instantiate adapters (respecting enable flags).
        self._hindsight = (
            HindsightCoreAdapter(self._runner)
            if self._settings.enable_hindsight
            else None
        )
        self._everos = (
            EverosSubstrateAdapter() if self._settings.enable_everos else None
        )
        self._mempalace = (
            MempalaceIndexAdapter() if self._settings.enable_mempalace else None
        )
        self._openviking = (
            OpenVikingInjectionAdapter() if self._settings.enable_openviking else None
        )
        self._memos = (
            MemosPortabilityAdapter() if self._settings.enable_memos else None
        )

        # Adapters that participate in capture/recall/consolidate fan-out.
        self._adapters = [
            a
            for a in (self._hindsight, self._everos, self._mempalace, self._memos)
            if a is not None
        ]
        self._pipeline = LayeredPipeline(
            self._substrate,
            self._adapters,
            mempalace=self._mempalace,
            openviking=self._openviking,
        )
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    def start(self, *, require_singleton: bool = False) -> None:
        """Start adapters and, when asked, claim the memory root exclusively.

        ``require_singleton`` raises :class:`SingletonLockHeld` when another live
        process already owns this root. The HTTP server passes it so a sidecar
        resurrected beside a hung one exits instead of interleaving writes into
        the markdown we call truth. In-process embedders (tests, a library
        caller) leave it off and rely on the per-bank append lock.
        """
        self._settings.home.mkdir(parents=True, exist_ok=True)
        if require_singleton and not self._substrate.acquire_singleton():
            raise SingletonLockHeld(
                f"another unified-memory process already owns {self._settings.home}"
            )
        for adapter in (
            self._hindsight,
            self._everos,
            self._mempalace,
            self._openviking,
            self._memos,
        ):
            if adapter is None:
                continue
            try:
                adapter.start()
            except Exception as e:  # noqa: BLE001
                logger.warning("adapter %s start failed: %s", adapter.name, e)
        self._started = True
        logger.info(
            "UnifiedEngine started (brain=%s, home=%s)",
            "on" if self.brain_available else "off (substrate-only)",
            self._settings.home,
        )

    def stop(self) -> None:
        for adapter in (
            self._hindsight,
            self._everos,
            self._mempalace,
            self._openviking,
            self._memos,
        ):
            if adapter is None:
                continue
            try:
                adapter.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("adapter %s stop failed: %s", adapter.name, e)
        self._runner.close()
        self._substrate.release_singleton()
        self._started = False

    @property
    def brain_available(self) -> bool:
        return self._hindsight is not None and self._hindsight.available()

    # -- health --------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Supervisor contract: status is 'ok' when the brain is up, else
        'degraded' (substrate still serves capture + keyword recall)."""
        adapters = {
            a.name: a.available()
            for a in (
                self._hindsight,
                self._everos,
                self._mempalace,
                self._openviking,
                self._memos,
            )
            if a is not None
        }
        status = "ok" if self.brain_available else "degraded"
        return {
            "status": status,
            "brain": self.brain_available,
            "adapters": adapters,
            "home": str(self._settings.home),
        }

    # -- public API (called by the HTTP layer) ------------------------------

    def capture(
        self,
        *,
        bank: str,
        session_key: str,
        user_content: str,
        assistant_content: str,
        user_id: str = "",
    ) -> dict[str, Any]:
        bank_dir = self._settings.bank_dir(bank)
        event = CaptureEvent(
            bank=bank,
            session_key=session_key,
            user=user_content,
            assistant=assistant_content,
            user_id=user_id,
        )
        entry_id = self._pipeline.capture(event, bank_dir)
        return {"ok": True, "entry_id": entry_id, "brain": self.brain_available}

    def recall(
        self,
        *,
        bank: str,
        query: str,
        session_key: str = "",
        user_id: str = "",
        limit: int = 8,
        fact_type: str = "",
        question_date: str = "",
    ) -> dict[str, Any]:
        bank_dir = self._settings.bank_dir(bank)
        req = RecallRequest(
            bank=bank,
            query=query,
            session_key=session_key,
            user_id=user_id,
            limit=limit,
            fact_type=fact_type,
            question_date=question_date,
        )
        context, fused, meta = self._pipeline.recall(req, bank_dir)
        return {
            "context": context,
            "results": [
                {
                    "text": r.text,
                    "source": r.source,
                    "fact_type": r.fact_type,
                    "score": r.score,
                }
                for r in fused
            ],
            "meta": meta,
        }

    def search_conversations(
        self, *, bank: str, query: str, limit: int = 5, session_key: str = ""
    ) -> dict[str, Any]:
        """Verbatim L0 search over the substrate — exact past dialogue."""
        bank_dir = self._settings.bank_dir(bank)
        hits = self._substrate.search(
            bank_dir, query, limit=limit, session_key=session_key
        )
        return {
            "results": [
                {
                    "text": entry.as_text(),
                    "session_key": entry.session_key,
                    "ts": entry.ts,
                    "score": score,
                }
                for entry, score in hits
            ]
        }

    def end_session(self, *, bank: str, session_key: str = "") -> dict[str, Any]:
        bank_dir = self._settings.bank_dir(bank)
        self._pipeline.consolidate(bank, bank_dir)
        return {"ok": True}

    def reflect(self, *, bank: str, query: str) -> dict[str, Any]:
        if self._hindsight is not None and self._hindsight.available():
            answer = self._hindsight.reflect(bank, query)
            if answer:
                return {"answer": answer, "source": "hindsight"}
        # Degraded fallback: return the top verbatim recalls as raw material.
        recall = self.recall(bank=bank, query=query, limit=5)
        return {"answer": recall["context"], "source": "substrate"}

    def export(self, *, bank: str) -> dict[str, Any]:
        bank_dir = self._settings.bank_dir(bank)
        path = None
        if self._memos is not None:
            path = self._memos.export(bank, bank_dir)
        return {"ok": path is not None, "manifest": str(path) if path else ""}

    def backup_paths(self, *, bank: str) -> list[str]:
        bank_dir = self._settings.bank_dir(bank)
        paths = [str(bank_dir)]
        for adapter in self._adapters:
            try:
                paths.extend(adapter.export_paths(bank, bank_dir))
            except Exception:  # noqa: BLE001
                continue
        return sorted(set(paths))
