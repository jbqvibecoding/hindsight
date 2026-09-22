"""L2 consolidation as gated, watermarked stages.

Consolidation used to be one loop that re-did every adapter's work on every
session end, reported nothing, and ran only at session end — so a long session
never consolidated at all and a partial failure was invisible.

The shape here is ported from cognee's ``modules/improve`` registry, with four
disciplines worth naming because each one is load-bearing:

* **Watermark.** A stage records the entry count it last completed at. Nothing
  new since then is ``ALREADY_COMPLETED`` — an explicit status, never silence,
  so "skipped because done" cannot be confused with "did not run".
* **Gate before cost.** A gate runs before any expensive work and returns a
  typed skip reason. **A gate that itself raises is treated as open**: gates
  exist to save work, so a broken gate may cost time but must never change
  what the run produces.
* **Exactly one fatal stage.** Only the stage that guards the truth-to-index
  relationship may fail the run; every enrichment stage must fail open. This is
  the L0-vs-L1/L2/L3 contract, and it is checked at import time rather than
  left as a convention.
* **A real NOOP.** A run that legitimately did nothing — everything already at
  its watermark, or held back by the debounce — is distinguishable from one that
  succeeded at doing work, so no reader that gates on "succeeded" advances a
  watermark over work that never ran.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILE = "consolidate.json"


class StageStatus(str, Enum):
    COMPLETED = "completed"
    # Nothing new since this stage's watermark. Distinct from COMPLETED so a
    # caller can tell "ran and did work" from "ran and had nothing to do".
    ALREADY_COMPLETED = "already_completed"
    SKIPPED = "skipped"
    ERRORED = "errored"


class RunOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Terminal state for a run that executed nothing. Without it, a reader
    # gating on "succeeded" would treat a no-op as work that happened.
    NOOP = "noop"


@dataclass(slots=True)
class StageResult:
    name: str
    status: StageStatus
    reason: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ConsolidationRun:
    outcome: RunOutcome
    stages: list[StageResult] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "stages": [s.as_dict() for s in self.stages],
        }


@dataclass(slots=True)
class Stage:
    """One consolidation step: a gate, a call, and a watermark.

    ``run`` returns True when it did work and False when there was nothing to
    do. ``gate`` returns a skip reason, or "" to proceed.
    """

    name: str
    run: Callable[[str, Path], bool]
    gate: Callable[[str, Path], str] | None = None
    fatal: bool = False
    # A stage that must run on every consolidation regardless of whether new
    # entries arrived (e.g. an integrity check).
    ignores_watermark: bool = False


def validate_fatal_stage_policy(stages: list[Stage]) -> None:
    """At most one stage may fail a consolidation run.

    Checked when the pipeline is constructed, not reviewed by convention. The
    fatal stage is the one guarding the truth-to-index relationship: if it
    fails, entries exist in the markdown that recall cannot reach. Every other
    stage is enrichment and must degrade quietly — the same contract that lets
    the whole sidecar run with no optional engine installed.

    Zero fatal stages is legal but notable: it means the integrity guard was
    switched off (``UNIFIED_MEMORY_ENABLE_EVEROS=false``), so nothing will fail
    a run and index drift goes unreported. Two or more is always a mistake —
    that is how an enrichment failure starts taking consolidation down with it.
    """
    fatal = [s.name for s in stages if s.fatal]
    if len(fatal) > 1:
        raise AssertionError(
            f"at most one fatal consolidation stage is allowed, found {len(fatal)}: {fatal}"
        )
    if not fatal:
        logger.warning(
            "no fatal consolidation stage: truth-to-index integrity is not guarded, "
            "so index drift will not fail a run."
        )


# -- state ---------------------------------------------------------------------


def _state_path(bank_dir: Path) -> Path:
    return bank_dir / ".state" / _STATE_FILE


def read_state(bank_dir: Path) -> dict[str, Any]:
    """Load consolidation state. Never raises; a lost state file replays work."""
    try:
        with open(_state_path(bank_dir), encoding="utf-8") as fh:
            state = json.load(fh)
        return state if isinstance(state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(bank_dir: Path, state: dict[str, Any]) -> None:
    """Persist consolidation state atomically (write-then-rename)."""
    path = _state_path(bank_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{_STATE_FILE}-", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stage_watermark(state: dict[str, Any], name: str) -> int:
    """Entry count a stage last completed at, tolerant of both record shapes.

    Accepts a bare integer and a ``{"entries_at": n}`` dict so a state file
    written before a field existed stays readable — the same forward-compatible
    reader discipline as the substrate index.
    """
    slot = (state.get("stages") or {}).get(name)
    if isinstance(slot, dict):
        try:
            return int(slot.get("entries_at") or 0)
        except (TypeError, ValueError):
            return 0
    try:
        return int(slot or 0)
    except (TypeError, ValueError):
        return 0


# -- debounce ------------------------------------------------------------------


@dataclass(slots=True)
class DebounceDecision:
    due: bool
    reason: str
    new_entries: int = 0
    elapsed: float = 0.0


def debounce(
    state: dict[str, Any],
    *,
    entries: int,
    min_entries: int,
    min_seconds: float,
    now: float | None = None,
) -> DebounceDecision:
    """Whether a consolidation should run now.

    Fires on **either** trigger: enough new entries, or enough elapsed time.
    Both at their off value means every call runs. There is no timer — the
    decision is made inline on the call that asks — so entries below the
    threshold wait for the next call rather than being dropped; the watermarks
    mean whichever run comes next picks them up.

    Reading state is **fail-open**: an unreadable state file consolidates rather
    than silently stalling enrichment forever.
    """
    now = time.time() if now is None else now
    if min_entries <= 0 and min_seconds <= 0:
        return DebounceDecision(True, "no_debounce")
    if not state:
        return DebounceDecision(True, "first_run")

    try:
        last_entries = int(state.get("entries_at_last_run") or 0)
        last_ts = float(state.get("last_run_ts") or 0.0)
    except (TypeError, ValueError):
        return DebounceDecision(True, "state_unreadable")

    new_entries = max(0, entries - last_entries)
    elapsed = max(0.0, now - last_ts)

    if min_entries > 0 and new_entries >= min_entries:
        return DebounceDecision(True, "entries", new_entries, elapsed)
    if min_seconds > 0 and elapsed >= min_seconds:
        return DebounceDecision(True, "elapsed", new_entries, elapsed)
    return DebounceDecision(False, "debounced", new_entries, elapsed)


# -- execution -----------------------------------------------------------------


def execute_stage(stage: Stage, bank: str, bank_dir: Path, entries: int, state: dict) -> StageResult:
    """Run one stage. Never raises for a non-fatal stage."""
    if stage.gate is not None:
        try:
            skip_reason = stage.gate(bank, bank_dir)
        except Exception as e:  # noqa: BLE001
            # A gate exists only to save work, so a broken one costs time, never
            # correctness: treat it as open.
            logger.debug("stage %s gate raised, treating as open: %s", stage.name, e)
            skip_reason = ""
        if skip_reason:
            return StageResult(stage.name, StageStatus.SKIPPED, reason=skip_reason)

    if not stage.ignores_watermark and entries <= stage_watermark(state, stage.name):
        return StageResult(stage.name, StageStatus.ALREADY_COMPLETED, reason="no_new_entries")

    try:
        did_work = stage.run(bank, bank_dir)
    except Exception as e:  # noqa: BLE001
        if stage.fatal:
            raise
        logger.warning("consolidation stage %s errored (continuing): %s", stage.name, e)
        return StageResult(stage.name, StageStatus.ERRORED, reason=type(e).__name__, detail=str(e))

    return StageResult(
        stage.name,
        StageStatus.COMPLETED if did_work else StageStatus.ALREADY_COMPLETED,
        reason="" if did_work else "nothing_to_do",
    )
