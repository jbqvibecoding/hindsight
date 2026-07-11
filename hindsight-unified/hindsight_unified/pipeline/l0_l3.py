"""L0→L3 layered pipeline — the memory lifecycle spine.

Ported in shape from tencentdb-agent-memory's four-layer design, but driven by
the unified adapter set rather than a TS engine:

* **L0 — capture (verbatim):** append the exact turn to the markdown substrate
  (truth). Always runs, zero deps.
* **L1 — extract (episodic):** hand the turn to the Hindsight brain's
  ``retain_async`` for fact/entity/temporal-graph extraction into the single
  semantic index. Skipped (gracefully) when the brain is unavailable.
* **L2 — consolidate:** on session end / background tick, let each adapter
  reconcile and re-organize (EverOS md↔index reconciliation, brain mental-model
  refresh, MemOS scheduler cadence).
* **L3 — persona:** synthesized on demand via the brain's ``reflect_async``
  (exposed through the engine's ``reflect``).

The pipeline owns *sequencing*; the engine owns *lifecycle*. Adapters never
raise, so a missing contributor degrades the relevant layer without breaking
the others.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..adapters.base import UnifiedAdapter
from ..adapters.mempalace_index import MempalaceIndexAdapter
from ..adapters.openviking_injection import OpenVikingInjectionAdapter
from ..substrate import MarkdownSubstrate
from ..types import CaptureEvent, Recalled, RecallRequest

logger = logging.getLogger(__name__)

# Layer identifiers, kept explicit so logs/trajectory are self-describing.
L0_CAPTURE = "L0"
L1_EXTRACT = "L1"
L2_CONSOLIDATE = "L2"
L3_PERSONA = "L3"


def rrf_fuse(
    ranked_lists: list[list[Recalled]], *, k: int = 60, limit: int = 8
) -> list[Recalled]:
    """Reciprocal Rank Fusion across heterogeneous adapter result lists.

    Fuses on *rank* (not score) so adapters with incomparable score scales —
    the brain's activation vs the substrate's tf overlap — combine cleanly.
    Deduplicates by normalized text, keeping the highest-fused instance.
    """
    fused: dict[str, tuple[float, Recalled]] = {}
    for results in ranked_lists:
        for rank, item in enumerate(results):
            key = " ".join(item.text.split()).lower()[:200]
            contrib = 1.0 / (k + rank + 1)
            if key in fused:
                prev_score, prev_item = fused[key]
                fused[key] = (prev_score + contrib, prev_item)
            else:
                fused[key] = (contrib, item)
    ordered = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)
    out: list[Recalled] = []
    for score, item in ordered[:limit]:
        item.score = score
        out.append(item)
    return out


class LayeredPipeline:
    """Sequences the L0→L3 flow over the substrate + adapter set."""

    def __init__(
        self,
        substrate: MarkdownSubstrate,
        adapters: list[UnifiedAdapter],
        *,
        mempalace: MempalaceIndexAdapter | None = None,
        openviking: OpenVikingInjectionAdapter | None = None,
    ) -> None:
        self._substrate = substrate
        self._adapters = adapters
        self._mempalace = mempalace
        self._openviking = openviking

    # -- L0 + L1 -------------------------------------------------------------

    def capture(self, event: CaptureEvent, bank_dir: Path) -> str:
        """L0 verbatim append, then fan out L1 extraction to adapters."""
        entry_id = self._substrate.append(
            bank_dir,
            session_key=event.session_key,
            user=event.user,
            assistant=event.assistant,
            ts=event.ts,
            metadata=event.metadata,
        )
        for adapter in self._adapters:
            try:
                adapter.capture(event, bank_dir)
            except Exception as e:  # noqa: BLE001 - adapters must not break capture
                logger.warning("adapter %s capture failed: %s", adapter.name, e)
        return entry_id

    # -- recall (fuse brain + substrate, then tier + AAAK) -------------------

    def recall(self, req: RecallRequest, bank_dir: Path) -> tuple[str, list, dict]:
        """Return ``(context_text, recalled_items, trajectory_meta)``.

        Fuses every adapter's ``recall_enrich`` with the substrate keyword
        search (RRF), compresses to AAAK triage cards (mempalace), and assembles
        a tiered, budgeted payload (OpenViking).
        """
        ranked_lists: list[list[Recalled]] = []

        # Adapter contributions (Hindsight brain, etc.).
        for adapter in self._adapters:
            try:
                items = adapter.recall_enrich(req, bank_dir)
            except Exception as e:  # noqa: BLE001
                logger.debug("adapter %s recall failed: %s", adapter.name, e)
                items = []
            if items:
                ranked_lists.append(items)

        # Always-on substrate keyword recall (verbatim safety net).
        sub_hits = self._substrate.search(
            bank_dir, req.query, limit=req.limit, session_key=""
        )
        if sub_hits:
            ranked_lists.append(
                [
                    Recalled(
                        text=entry.as_text(),
                        source="substrate",
                        score=score,
                        metadata={"entry_id": entry.entry_id},
                    )
                    for entry, score in sub_hits
                ]
            )

        fused = rrf_fuse(ranked_lists, limit=req.limit)

        aaak = ""
        if self._mempalace is not None:
            aaak = self._mempalace.aaak_cards(fused)

        if self._openviking is not None:
            context, trajectory = self._openviking.assemble(fused, aaak_header=aaak)
        else:
            context = "\n".join(f"- {r.text}" for r in fused)
            trajectory = [{"source": r.source, "included": True} for r in fused]

        meta = {
            "num_fused": len(fused),
            "sources": sorted({r.source for r in fused}),
            "trajectory": trajectory,
        }
        return context, fused, meta

    # -- L2 ------------------------------------------------------------------

    def consolidate(self, bank: str, bank_dir: Path) -> None:
        for adapter in self._adapters:
            try:
                adapter.consolidate(bank, bank_dir)
            except Exception as e:  # noqa: BLE001
                logger.debug("adapter %s consolidate failed: %s", adapter.name, e)
