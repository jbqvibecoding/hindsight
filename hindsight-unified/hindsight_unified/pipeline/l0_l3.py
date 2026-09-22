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
import time
from pathlib import Path

from ..adapters.base import UnifiedAdapter
from ..adapters.mempalace_index import MempalaceIndexAdapter
from ..adapters.openviking_injection import OpenVikingInjectionAdapter
from ..substrate import MarkdownSubstrate
from ..types import CaptureEvent, Recalled, RecallRequest
from .stages import (
    ConsolidationRun,
    RunOutcome,
    Stage,
    StageResult,
    StageStatus,
    debounce,
    execute_stage,
    read_state,
    validate_fatal_stage_policy,
    write_state,
)

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
        self._stages = self._build_stages()
        # Fails construction, not a later run, if the one-fatal-stage contract
        # is ever broken by adding a second.
        validate_fatal_stage_policy(self._stages)

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
                        metadata={"entry_id": entry.entry_id, "seq": entry.seq},
                    )
                    for entry, score in sub_hits
                ]
            )

        fused = rrf_fuse(ranked_lists, limit=req.limit)

        # Order for presentation once, before the cards are cut, so card [0]
        # names the same entry the body shows first. Building the index from the
        # fused order and the body from the recency order would break the
        # "scan the cards, then open that drawer" contract they exist for.
        if self._openviking is not None:
            fused = self._openviking.order_for_presentation(fused)

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

    def _build_stages(self) -> list[Stage]:
        """One stage per adapter, plus the integrity stage that may fail a run.

        ``reconcile`` is the single fatal stage: it is what keeps the markdown
        truth and the derived index in agreement, so its failure means entries
        exist that recall cannot reach. Everything else is enrichment and fails
        open — the same contract that lets the sidecar run with no optional
        engine installed at all.
        """
        stages: list[Stage] = []

        everos = next((a for a in self._adapters if a.name == "everos"), None)
        if everos is not None:
            stages.append(
                Stage(
                    name="reconcile",
                    run=lambda bank, bank_dir: bool(everos.reconcile(bank_dir)),
                    fatal=True,
                    # An integrity check must run even when no new entries
                    # arrived: drift can come from outside the append path.
                    ignores_watermark=True,
                )
            )

        for adapter in self._adapters:
            if adapter.name == "everos":
                continue  # already represented by the reconcile stage
            stages.append(
                Stage(
                    name=adapter.name,
                    run=self._adapter_stage(adapter),
                    gate=self._adapter_gate(adapter),
                )
            )
        return stages

    @staticmethod
    def _adapter_stage(adapter: UnifiedAdapter):
        def _run(bank: str, bank_dir: Path) -> bool:
            adapter.consolidate(bank, bank_dir)
            return True

        return _run

    @staticmethod
    def _adapter_gate(adapter: UnifiedAdapter):
        def _gate(bank: str, bank_dir: Path) -> str:
            # Checked before any work: an adapter whose engine never loaded has
            # nothing to consolidate, and saying so is cheaper than finding out.
            return "" if adapter.available() else "adapter_unavailable"

        return _gate

    def consolidate(
        self,
        bank: str,
        bank_dir: Path,
        *,
        min_entries: int = 0,
        min_seconds: float = 0.0,
        force: bool = False,
    ) -> ConsolidationRun:
        """Run the L2 stages, honouring watermarks and the debounce.

        Returns a per-stage report rather than nothing, so a partial failure is
        visible and a caller can tell work from a no-op.
        """
        state = read_state(bank_dir)
        entries = self._substrate.count(bank_dir)

        if not force:
            decision = debounce(
                state, entries=entries, min_entries=min_entries, min_seconds=min_seconds
            )
            if not decision.due:
                return ConsolidationRun(RunOutcome.NOOP, reason=decision.reason)

        results: list[StageResult] = []
        fatal_error: Exception | None = None
        for stage in self._stages:
            try:
                result = execute_stage(stage, bank, bank_dir, entries, state)
            except Exception as e:  # noqa: BLE001 - only a fatal stage reaches here
                fatal_error = e
                results.append(
                    StageResult(
                        stage.name,
                        StageStatus.ERRORED,
                        reason=type(e).__name__,
                        detail=str(e),
                    )
                )
                break
            results.append(result)
            if result.status is StageStatus.COMPLETED:
                stages_state = state.setdefault("stages", {})
                stages_state[stage.name] = {"entries_at": entries, "ts": time.time()}

        did_work = any(r.status is StageStatus.COMPLETED for r in results)
        if fatal_error is not None:
            outcome, reason = RunOutcome.FAILED, "fatal_stage_errored"
        elif did_work:
            outcome, reason = RunOutcome.SUCCEEDED, ""
        else:
            # Nothing ran: do not let a watermark advance over work that never
            # happened, so a later run still picks these entries up.
            outcome, reason = RunOutcome.NOOP, "nothing_to_do"

        if outcome is not RunOutcome.NOOP:
            state["entries_at_last_run"] = entries
            state["last_run_ts"] = time.time()
        state["last_outcome"] = outcome.value
        write_state(bank_dir, state)

        if fatal_error is not None:
            logger.error("consolidation failed on a fatal stage: %s", fatal_error)
        return ConsolidationRun(outcome, results, reason)
