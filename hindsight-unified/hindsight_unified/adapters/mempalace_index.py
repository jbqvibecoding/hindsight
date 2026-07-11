"""mempalace adapter — AAAK index-triage & embedder-identity safety.

mempalace's reusable ideas, mapped into the unified system:

* **AAAK compression** (``mempalace.dialect.Dialect``): a compact symbolic
  index card per memory so an LLM can triage many entries cheaply before
  "opening the drawer" (reading the verbatim text). We use it in the injection
  path to prefix recalled items with a scannable digest.
* **Embedder-identity guard** (``mempalace.backends.base.check_embedder_identity``):
  refuse to silently serve recall under a swapped embedding model, which would
  degrade relevance without any error. We record the active embedder identity
  and raise on mismatch.
* **As-of temporal** semantics: surfaced by passing ``question_date`` through to
  the Hindsight brain's already-temporal recall (handled in the engine), so no
  second temporal graph is maintained here.

Every capability degrades to a dependency-free fallback when ``mempalace`` is
not importable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..types import Recalled
from .base import UnifiedAdapter

logger = logging.getLogger(__name__)


class MempalaceIndexAdapter(UnifiedAdapter):
    def __init__(self) -> None:
        self._available = False
        self._dialect = None

    @property
    def name(self) -> str:
        return "mempalace"

    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        self._available = True  # fallback AAAK works with zero deps
        try:
            from mempalace.dialect import Dialect  # type: ignore

            self._dialect = Dialect()
            logger.info("mempalace Dialect present; AAAK triage active (native)")
        except Exception:  # noqa: BLE001
            logger.info("mempalace absent; AAAK triage active (lean fallback)")

    # -- embedder-identity guard --------------------------------------------

    def check_embedder_identity(self, bank_dir: Path, identity: str) -> None:
        """Persist and verify the embedding-model identity for a bank.

        Raises ``RuntimeError`` if a previously recorded identity differs from
        the current one — a swapped embedder means the stored vectors are no
        longer comparable, and silently continuing would degrade recall.
        """
        if not identity:
            return
        marker = bank_dir / ".embedder_identity"
        try:
            if marker.exists():
                prev = marker.read_text(encoding="utf-8").strip()
                if prev and prev != identity:
                    raise RuntimeError(
                        f"Embedder identity mismatch for bank: stored={prev!r} "
                        f"current={identity!r}. Re-embed or restore the model to "
                        f"avoid silent recall degradation."
                    )
                return
            bank_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(identity, encoding="utf-8")
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("mempalace embedder-identity check skipped: %s", e)

    # -- AAAK triage compression --------------------------------------------

    def aaak_cards(self, recalled: list[Recalled], max_cards: int = 12) -> str:
        """Render recalled items as compact triage cards.

        Native path uses mempalace's ``Dialect`` when it exposes a ``compress``
        method; otherwise emits a lean card (source · type · keyworded gist).
        """
        cards: list[str] = []
        for i, r in enumerate(recalled[:max_cards]):
            gist = self._compress_one(r.text)
            tag = r.fact_type or r.source
            cards.append(f"[{i}] ({tag}) {gist}")
        if not cards:
            return ""
        return "AAAK index (scan, then read full entries below):\n" + "\n".join(cards)

    def _compress_one(self, text: str) -> str:
        if self._dialect is not None:
            for meth in ("compress", "encode", "to_aaak"):
                fn = getattr(self._dialect, meth, None)
                if callable(fn):
                    try:
                        out = fn(text)
                        if isinstance(out, str) and out:
                            return out
                    except Exception:  # noqa: BLE001
                        break
        # Lean fallback: first ~120 chars, whitespace-collapsed.
        gist = " ".join(text.split())
        return gist[:120] + ("…" if len(gist) > 120 else "")

    def export_paths(self, bank: str, bank_dir: Path) -> list[str]:
        marker = bank_dir / ".embedder_identity"
        return [str(marker)] if marker.exists() else []
