"""Consolidation stage-machine tests (workstream B4).

The statuses are the point: a run that did nothing, a run that had nothing to
do, and a run that failed must all be distinguishable, and a watermark must
never advance over work that did not happen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.pipeline.stages import (
    DebounceDecision,
    RunOutcome,
    Stage,
    StageStatus,
    debounce,
    execute_stage,
    read_state,
    stage_watermark,
    validate_fatal_stage_policy,
    write_state,
)


def _engine(tmp_path: Path, **overrides) -> UnifiedEngine:
    settings = Settings(
        host="127.0.0.1",
        port=0,
        home=tmp_path,
        enable_hindsight=False,
        enable_everos=True,
        enable_mempalace=True,
        enable_openviking=True,
        enable_memos=True,
        **overrides,
    )
    engine = UnifiedEngine(settings)
    engine.start()
    return engine


# -- policy --------------------------------------------------------------------


def test_two_fatal_stages_is_rejected() -> None:
    noop = lambda bank, bank_dir: True  # noqa: E731
    with pytest.raises(AssertionError, match="at most one fatal"):
        validate_fatal_stage_policy(
            [Stage("a", noop, fatal=True), Stage("b", noop, fatal=True)]
        )


def test_zero_fatal_stages_is_allowed(caplog: pytest.LogCaptureFixture) -> None:
    # Legal: it means the integrity guard was switched off deliberately.
    validate_fatal_stage_policy([Stage("a", lambda b, d: True)])
    assert "integrity is not guarded" in caplog.text


def test_live_pipeline_has_exactly_one_fatal_stage(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    try:
        fatal = [s.name for s in engine._pipeline._stages if s.fatal]
        assert fatal == ["reconcile"]
    finally:
        engine.stop()


# -- stage execution -----------------------------------------------------------


def test_a_gate_that_raises_is_treated_as_open(tmp_path: Path) -> None:
    """Gates save work; a broken one must cost time, never correctness."""
    ran: list[str] = []

    def boom(bank: str, bank_dir: Path) -> str:
        raise RuntimeError("gate is broken")

    stage = Stage("enrich", run=lambda b, d: bool(ran.append("ran")) or True, gate=boom)
    result = execute_stage(stage, "b", tmp_path, entries=1, state={})
    assert ran == ["ran"]
    assert result.status is StageStatus.COMPLETED


def test_a_closed_gate_skips_with_its_reason(tmp_path: Path) -> None:
    stage = Stage("enrich", run=lambda b, d: True, gate=lambda b, d: "adapter_unavailable")
    result = execute_stage(stage, "b", tmp_path, entries=1, state={})
    assert result.status is StageStatus.SKIPPED
    assert result.reason == "adapter_unavailable"


def test_watermark_reports_already_completed_not_silence(tmp_path: Path) -> None:
    stage = Stage("enrich", run=lambda b, d: True)
    state = {"stages": {"enrich": {"entries_at": 5}}}
    result = execute_stage(stage, "b", tmp_path, entries=5, state=state)
    assert result.status is StageStatus.ALREADY_COMPLETED
    assert result.reason == "no_new_entries"


def test_a_non_fatal_stage_error_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    def boom(bank: str, bank_dir: Path) -> bool:
        raise ValueError("enrichment exploded")

    result = execute_stage(Stage("enrich", run=boom), "b", tmp_path, entries=1, state={})
    assert result.status is StageStatus.ERRORED
    assert result.reason == "ValueError"


def test_a_fatal_stage_error_propagates(tmp_path: Path) -> None:
    def boom(bank: str, bank_dir: Path) -> bool:
        raise ValueError("integrity gone")

    with pytest.raises(ValueError):
        execute_stage(Stage("reconcile", run=boom, fatal=True), "b", tmp_path, 1, {})


def test_watermark_reader_tolerates_both_record_shapes() -> None:
    assert stage_watermark({"stages": {"s": {"entries_at": 7}}}, "s") == 7
    assert stage_watermark({"stages": {"s": 7}}, "s") == 7  # pre-dict shape
    assert stage_watermark({}, "s") == 0
    assert stage_watermark({"stages": {"s": "junk"}}, "s") == 0


# -- debounce ------------------------------------------------------------------


def test_debounce_fires_on_either_trigger() -> None:
    state = {"entries_at_last_run": 10, "last_run_ts": 1_000.0}
    entries_due = debounce(state, entries=30, min_entries=20, min_seconds=999_999, now=1_001.0)
    assert entries_due.due and entries_due.reason == "entries"

    time_due = debounce(state, entries=11, min_entries=999, min_seconds=60, now=2_000.0)
    assert time_due.due and time_due.reason == "elapsed"

    held = debounce(state, entries=11, min_entries=20, min_seconds=60, now=1_001.0)
    assert not held.due and held.reason == "debounced"


def test_debounce_is_fail_open_on_unreadable_state() -> None:
    bad = {"entries_at_last_run": "not a number", "last_run_ts": 0.0}
    decision = debounce(bad, entries=1, min_entries=20, min_seconds=60)
    assert decision.due and decision.reason == "state_unreadable"


def test_debounce_off_by_default_values() -> None:
    assert debounce({"a": 1}, entries=1, min_entries=0, min_seconds=0).reason == "no_debounce"
    assert debounce({}, entries=1, min_entries=20, min_seconds=60).reason == "first_run"


def test_state_roundtrips_atomically(tmp_path: Path) -> None:
    write_state(tmp_path, {"entries_at_last_run": 3})
    assert read_state(tmp_path)["entries_at_last_run"] == 3
    # A corrupt state file replays work rather than crashing.
    (tmp_path / ".state" / "consolidate.json").write_text("{oops", encoding="utf-8")
    assert read_state(tmp_path) == {}


# -- end to end through the engine ---------------------------------------------


def test_second_consolidation_with_no_new_entries_is_a_noop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        first = engine.end_session(bank="b")
        assert first["outcome"] == RunOutcome.SUCCEEDED.value

        second = engine.end_session(bank="b")
        # Nothing new: no stage did work, so the run is a NOOP, not a success.
        assert second["outcome"] == RunOutcome.NOOP.value
        assert second["reason"] == "nothing_to_do"
        by_name = {s["name"]: s for s in second["stages"]}
        assert by_name["memos"]["status"] == StageStatus.ALREADY_COMPLETED.value
    finally:
        engine.stop()


def test_mid_session_consolidation_is_debounced_but_session_end_forces(tmp_path: Path) -> None:
    engine = _engine(tmp_path, consolidate_min_entries=50, consolidate_min_seconds=99_999)
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        engine.end_session(bank="b")  # establishes the debounce baseline
        engine.capture(bank="b", session_key="s", user_content="another", assistant_content="ok")

        held = engine.maybe_consolidate(bank="b")
        assert held["outcome"] == RunOutcome.NOOP.value
        assert held["reason"] == "debounced"

        # Session end is the one moment we know no more turns are coming, so
        # the debounce must not hold work back to a run that may never happen.
        forced = engine.end_session(bank="b")
        assert forced["outcome"] == RunOutcome.SUCCEEDED.value
    finally:
        engine.stop()


def test_an_unavailable_adapter_is_skipped_with_a_reason(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    try:
        memos = next(a for a in engine._pipeline._adapters if a.name == "memos")
        memos._available = False  # engine never loaded / was disabled
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        by_name = {s["name"]: s for s in run["stages"]}
        assert by_name["memos"]["status"] == StageStatus.SKIPPED.value
        assert by_name["memos"]["reason"] == "adapter_unavailable"
    finally:
        engine.stop()


def test_a_noop_run_does_not_advance_the_debounce_baseline(tmp_path: Path) -> None:
    """A watermark must never move over work that never happened."""
    engine = _engine(tmp_path)
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        engine.end_session(bank="b")
        baseline = read_state(tmp_path / "banks" / "b")["entries_at_last_run"]

        engine.end_session(bank="b")  # NOOP
        after = read_state(tmp_path / "banks" / "b")
        assert after["entries_at_last_run"] == baseline
        assert after["last_outcome"] == RunOutcome.NOOP.value
    finally:
        engine.stop()


def test_debounce_decision_is_reported_not_inferred() -> None:
    decision = debounce({}, entries=0, min_entries=1, min_seconds=0)
    assert isinstance(decision, DebounceDecision)
    assert decision.reason  # every decision carries its reason
