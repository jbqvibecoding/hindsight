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
from ..distill import LessonStore
from ..llm import LLMClient
from ..substrate import MarkdownSubstrate
from ..triggers import TriggerStore
from ..types import CaptureEvent, Recalled, RecallRequest
from .conversational import (
    RECENT_TURNS,
    build_conversational_query,
    conversational_reserve,
    merge_ranked,
)
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


def rrf_fuse(ranked_lists: list[list[Recalled]], *, k: int = 60, limit: int = 8) -> list[Recalled]:
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
        llm: LLMClient | None = None,
    ) -> None:
        self._substrate = substrate
        self._adapters = adapters
        self._mempalace = mempalace
        self._openviking = openviking
        self._llm = llm if llm is not None else LLMClient()
        self._triggers = TriggerStore()
        self._lessons = LessonStore()
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

        # Always-on substrate recall (verbatim safety net), in two lanes: the
        # question as asked, and a deterministic rewrite that folds in the last
        # couple of turns. An anaphoric follow-up ("and the other one?") names
        # nothing a lexical scorer can match, and the rewrite costs no model
        # call — see pipeline/conversational.py.
        raw_lane = self._substrate_lane(bank_dir, req.query, req.limit)
        rewrite_lane: list[Recalled] = []
        expanded = build_conversational_query(
            req.query,
            self._substrate.recent_turns(bank_dir, session_key=req.session_key, limit=RECENT_TURNS),
        )
        if expanded:
            rewrite_lane = self._substrate_lane(bank_dir, expanded, req.limit)

        substrate_hits = merge_ranked(
            raw_lane,
            rewrite_lane,
            limit=req.limit,
            secondary_reserve=conversational_reserve(req.limit),
        )
        if substrate_hits:
            ranked_lists.append(substrate_hits)

        # Third lane: write-time triggers — the phrases a future question was
        # predicted to use. It votes on ranking only: every hit is dereferenced
        # back to its verbatim entry before it can be shown, so a trigger can
        # change what surfaces and never what is read.
        trigger_lane = self._trigger_lane(bank_dir, req.query, req.limit)
        if trigger_lane:
            ranked_lists.append(trigger_lane)

        # Fourth lane: distilled lessons. Unlike a summary, a lesson IS shown —
        # it is a legitimate derived memory — so it is labelled as such and
        # carries the ids of the entries it came from, keeping the verbatim
        # source one lookup away.
        lesson_lane = self._lesson_lane(bank_dir, req.query, req.limit)
        if lesson_lane:
            ranked_lists.append(lesson_lane)

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

    def _substrate_lane(self, bank_dir: Path, query: str, limit: int) -> list[Recalled]:
        """One substrate retrieval lane, as a ranked Recalled list."""
        return [
            Recalled(
                text=entry.as_text(),
                source="substrate",
                score=score,
                metadata={"entry_id": entry.entry_id, "seq": entry.seq},
            )
            for entry, score in self._substrate.search(bank_dir, query, limit=limit)
        ]

    def _trigger_lane(self, bank_dir: Path, query: str, limit: int) -> list[Recalled]:
        """Trigger hits, resolved to the verbatim entries they point at."""
        hits = self._triggers.search(bank_dir, query, limit=limit)
        if not hits:
            return []
        by_id = {e.entry_id: e for e in self._substrate._load(bank_dir)}
        lane: list[Recalled] = []
        for entry_id, score in hits:
            entry = by_id.get(entry_id)
            if entry is None:
                # The trigger outlived its entry; it is a derivative, so the
                # missing source wins and the hit is dropped.
                continue
            lane.append(
                Recalled(
                    text=entry.as_text(),
                    source="trigger",
                    score=score,
                    metadata={"entry_id": entry.entry_id, "seq": entry.seq},
                )
            )
        return lane

    def _lesson_lane(self, bank_dir: Path, query: str, limit: int) -> list[Recalled]:
        """Distilled lessons as a ranked lane, tagged so they are never
        mistaken for something the user said."""
        return [
            Recalled(
                text=lesson.render(),
                source="lesson",
                score=score,
                fact_type="lesson",
                metadata={
                    "lesson_id": lesson.lesson_id,
                    "member_entry_ids": lesson.member_entry_ids,
                },
            )
            for lesson, score in self._lessons.search(bank_dir, query, limit=limit)
        ]

    # -- L2 ------------------------------------------------------------------

    def _trigger_stage(self, bank: str, bank_dir: Path) -> bool:
        """Generate write-time triggers for entries that lack current ones.

        Raises on an LLM outage so the stage reports ``errored`` and its
        watermark stays put: an unreachable model is not the same as "nothing
        left to anticipate", and sealing entries on the strength of calls that
        never ran would leave them permanently unreachable by this lane.
        """
        written = self._triggers.generate_missing(
            bank_dir, self._substrate._load(bank_dir), self._llm
        )
        if written is None:
            raise RuntimeError("trigger generation aborted: LLM unreachable")
        return written > 0

    def _distill_stage(self, bank: str, bank_dir: Path) -> bool:
        """Distil the newest slice into durable lessons.

        Raises on an outage so the stage reports ``errored`` and its watermark
        stays put — an unreachable model must never mark these entries
        distilled forever.
        """
        accepted = self._lessons.distill(bank_dir, self._substrate._load(bank_dir), self._llm)
        if accepted is None:
            raise RuntimeError("distillation aborted: LLM unreachable")
        return accepted > 0

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

        stages.append(
            Stage(
                name="triggers",
                run=self._trigger_stage,
                # Gated before any cost: with no model configured the whole
                # lane is simply absent, which is a supported deployment
                # rather than a degraded one.
                gate=lambda bank, bank_dir: "" if self._llm.available() else "no_llm_configured",
            )
        )

        stages.append(
            Stage(
                name="distill",
                run=self._distill_stage,
                gate=lambda bank, bank_dir: "" if self._llm.available() else "no_llm_configured",
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
