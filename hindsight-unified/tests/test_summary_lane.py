"""Retrieval-summary lane tests (workstream D4).

The lane needs a model to produce anything, and no key is configured here, so
these drive it with a stub client. That covers the plumbing and — more
importantly — the failure semantics, which is where the bugs in this kind of
layer actually live. The prompt's behaviour with a real model is **not**
validated by these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.pipeline.stages import RunOutcome, StageStatus
from hindsight_unified.substrate import SubstrateEntry, content_hash
from hindsight_unified.summarize import SummaryStore, summary_id


class StubLLM:
    """Stands in for LLMClient. ``None`` from complete() means outage."""

    def __init__(self, replies: list[str | None] | None = None, *, available: bool = True):
        self._replies = list(replies or [])
        self._available = available
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    @property
    def model(self) -> str:
        return "stub"

    def complete(self, *, system: str, user: str, **_kw) -> str | None:
        self.calls.append(user)
        if self._replies:
            return self._replies.pop(0)
        return f"This chunk is about:\n- Topics: stubbed\n\nFacts:\n- {user[:40]}"


def _engine(tmp_path: Path, llm=None) -> UnifiedEngine:
    engine = UnifiedEngine(
        Settings(
            host="127.0.0.1",
            port=0,
            home=tmp_path,
            enable_hindsight=False,
            enable_everos=True,
            enable_mempalace=True,
            enable_openviking=True,
            enable_memos=True,
        )
    )
    if llm is not None:
        engine._pipeline._llm = llm
        engine._pipeline._stages = engine._pipeline._build_stages()
    engine.start()
    return engine


def _entry(entry_id: str, user: str) -> SubstrateEntry:
    return SubstrateEntry(
        entry_id=entry_id,
        session_key="s",
        user=user,
        assistant="",
        ts=1.0,
        metadata={},
        digest=content_hash(user, ""),
    )


# -- identity and storage ------------------------------------------------------


def test_summary_id_is_derived_not_stored() -> None:
    # Given an entry you can always recompute its summary id, so the sidecar
    # needs no join table and stays rebuildable.
    assert summary_id("abc") == summary_id("abc")
    assert summary_id("abc") != summary_id("abd")


def test_store_roundtrips_and_skips_torn_lines(tmp_path: Path) -> None:
    store = SummaryStore()
    store.append(tmp_path, [_summary("e1", "about queues")])
    with open(store.path(tmp_path), "a", encoding="utf-8") as fh:
        fh.write('{"entry_id": "tor')  # simulated torn write
    loaded = store.load(tmp_path)
    assert set(loaded) == {"e1"}


def _summary(entry_id: str, text: str, digest: str = ""):
    from hindsight_unified.summarize import EntrySummary

    return EntrySummary(entry_id=entry_id, text=text, source_digest=digest)


# -- generation semantics ------------------------------------------------------


def test_generation_is_skipped_without_a_model(tmp_path: Path) -> None:
    store = SummaryStore()
    written = store.generate_missing(tmp_path, [_entry("e1", "x")], StubLLM(available=False))
    # 0, not None: "no model configured" is a supported deployment, not an outage.
    assert written == 0
    assert store.load(tmp_path) == {}


def test_an_outage_returns_none_not_zero(tmp_path: Path) -> None:
    """The distinction the watermark depends on.

    ``0`` means "nothing left to summarise" and may advance a watermark;
    ``None`` means the calls never happened and must not.
    """
    store = SummaryStore()
    result = store.generate_missing(
        tmp_path, [_entry("e1", "x"), _entry("e2", "y")], StubLLM([None])
    )
    assert result is None


def test_partial_progress_is_published_before_an_outage(tmp_path: Path) -> None:
    store = SummaryStore()
    result = store.generate_missing(
        tmp_path,
        [_entry("e1", "first"), _entry("e2", "second"), _entry("e3", "third")],
        StubLLM(["a real summary", None]),
    )
    assert result is None  # the run still reports the outage...
    # ...but what succeeded was kept, and the rest is simply still pending.
    assert set(store.load(tmp_path)) == {"e1"}


def test_existing_summaries_are_not_regenerated(tmp_path: Path) -> None:
    store = SummaryStore()
    entry = _entry("e1", "x")
    llm = StubLLM()
    assert store.generate_missing(tmp_path, [entry], llm) == 1
    assert store.generate_missing(tmp_path, [entry], llm) == 0
    assert len(llm.calls) == 1


def test_a_stale_summary_is_regenerated(tmp_path: Path) -> None:
    """A digest mismatch is stale, which is a different problem from missing."""
    store = SummaryStore()
    store.append(tmp_path, [_summary("e1", "old summary", digest="stale-digest")])
    assert store.generate_missing(tmp_path, [_entry("e1", "new text")], StubLLM()) == 1


# -- retrieval discipline ------------------------------------------------------


def test_search_returns_entry_ids_never_summary_text(tmp_path: Path) -> None:
    store = SummaryStore()
    store.append(tmp_path, [_summary("e1", "This chunk is about storage cleanup and blobs")])
    hits = store.search(tmp_path, "storage cleanup")
    assert hits and hits[0][0] == "e1"
    assert all(isinstance(entry_id, str) for entry_id, _score in hits)


def test_the_lane_dereferences_to_verbatim_text(tmp_path: Path) -> None:
    """A summary may change what ranks; it must never change what is shown."""
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        engine.capture(
            bank="b",
            session_key="s",
            user_content="The nightly job that trims old blobs is called reaper.",
            assistant_content="Noted.",
        )
        # Hand-write a summary carrying vocabulary the entry does not have.
        bank_dir = tmp_path / "banks" / "b"
        entry_id = engine._substrate._load(bank_dir)[0].entry_id
        SummaryStore().append(
            bank_dir,
            [_summary(entry_id, "This chunk is about:\n- Topics: storage cleanup, retention")],
        )

        context = engine.recall(bank="b", query="storage cleanup")["context"]
        # The verbatim entry surfaces...
        assert "reaper" in context
        # ...and no summary wording leaks into what the caller reads.
        assert "storage cleanup" not in context
    finally:
        engine.stop()


def test_a_summary_whose_entry_is_gone_is_dropped(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        bank_dir = tmp_path / "banks" / "b"
        bank_dir.mkdir(parents=True, exist_ok=True)
        SummaryStore().append(bank_dir, [_summary("nonexistent", "about widgets")])
        # The summary is a derivative, so a missing source wins.
        assert engine.recall(bank="b", query="widgets")["results"] == []
    finally:
        engine.stop()


def test_rederive_drops_the_summary_sidecar(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        bank_dir = tmp_path / "banks" / "b"
        SummaryStore().append(bank_dir, [_summary("whatever", "a summary")])
        out = engine.rederive(bank="b")
        assert out["summaries_dropped"] is True
        assert SummaryStore().load(bank_dir) == {}
    finally:
        engine.stop()


# -- the consolidation stage ---------------------------------------------------


def test_the_stage_gates_off_without_a_model(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM(available=False))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "summarize")
        # Gated before any cost, with a reason rather than silence.
        assert stage["status"] == StageStatus.SKIPPED.value
        assert stage["reason"] == "no_llm_configured"
    finally:
        engine.stop()


def test_the_stage_runs_and_reports_completed(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "summarize")
        assert stage["status"] == StageStatus.COMPLETED.value
        assert run["outcome"] == RunOutcome.SUCCEEDED.value
    finally:
        engine.stop()


def test_an_outage_errors_the_stage_without_advancing_its_watermark(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM([None]))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "summarize")
        assert stage["status"] == StageStatus.ERRORED.value
        # Enrichment fails open: the run as a whole still succeeded on the
        # stages that did work.
        assert run["outcome"] == RunOutcome.SUCCEEDED.value

        from hindsight_unified.pipeline.stages import read_state, stage_watermark

        state = read_state(tmp_path / "banks" / "b")
        # The watermark must not have moved over calls that never happened.
        assert stage_watermark(state, "summarize") == 0
    finally:
        engine.stop()


def test_summarize_is_not_the_fatal_stage(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        fatal = [s.name for s in engine._pipeline._stages if s.fatal]
        assert fatal == ["reconcile"]
        assert "summarize" in [s.name for s in engine._pipeline._stages]
    finally:
        engine.stop()


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_queries_return_no_summary_hits(tmp_path: Path, query: str) -> None:
    store = SummaryStore()
    store.append(tmp_path, [_summary("e1", "about queues")])
    assert store.search(tmp_path, query) == []


def test_the_lane_closes_the_measured_vocabulary_gap(tmp_path: Path) -> None:
    """The reason this layer exists, demonstrated on the real failing case.

    The eval's `vocabulary_gap` category sits at 0.33 because a question and
    the entry that answers it can share no term at all: "what cleans up storage
    we no longer need?" against "the nightly job that trims old blobs is called
    reaper". No expansion derivable *from the entry text* can invent the
    synonym, which is why the deterministic attempt was abandoned.

    The summary below is hand-written to stand in for the first section of the
    prompt (a category -> names list). So this verifies the **mechanism** — a
    summary hit dereferencing to a verbatim entry the question could not reach
    — and not the prompt's output quality, which needs a real model.
    """
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        # The answering turn must sit OUTSIDE the two-turn rewrite window, as
        # it does in the real case. With it inside, the rewrite query contains
        # the entry verbatim and retrieves it by self-similarity, which would
        # make this test pass for a reason that has nothing to do with the lane.
        engine.capture(
            bank="b",
            session_key="s",
            user_content="The nightly job that trims old blobs is called reaper.",
            assistant_content="Noted.",
        )
        for filler in ("Unrelated: dashboards refresh every five minutes.",
                       "Unrelated: we label flaky tests."):
            engine.capture(bank="b", session_key="s", user_content=filler, assistant_content="Noted.")

        question = "what cleans up storage we no longer need?"
        bank_dir = tmp_path / "banks" / "b"
        # A tight limit so ranking actually binds. With a limit above the bank
        # size every entry that scores at all comes back, and the lane's effect
        # is invisible — the same saturation the eval needed distractors to fix.
        limit = 2

        # Precondition: unreachable without the lane.
        before = engine.recall(bank="b", query=question, limit=limit)["context"]
        assert "reaper" not in before

        entry = engine._substrate._load(bank_dir)[0]
        SummaryStore().append(
            bank_dir,
            [
                _summary(
                    entry.entry_id,
                    "This chunk is about:\n- Topics: storage cleanup, retention, background jobs",
                    digest=entry.digest,
                )
            ],
        )
        after = engine.recall(bank="b", query=question, limit=limit)["context"]
        assert "reaper" in after
    finally:
        engine.stop()
