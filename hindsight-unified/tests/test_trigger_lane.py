"""Write-time trigger lane tests (workstream E3).

The lane needs a model to produce anything and no key is configured here, so
these drive it with a stub. That covers the plumbing, the deterministic
quality filters — which is where this layer's bugs actually live — and the
failure semantics. The prompt's behaviour with a real model is **not**
validated by these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.pipeline.stages import RunOutcome, StageStatus
from hindsight_unified.substrate import SubstrateEntry, content_hash
from hindsight_unified.triggers import (
    MIN_CONFIDENCE,
    Trigger,
    TriggerStore,
    bridge_is_grounded,
    trigger_id,
)


class StubLLM:
    """Stands in for LLMClient. ``None`` in the queue means an outage."""

    def __init__(self, replies: list[dict | None] | None = None, *, available: bool = True):
        self._replies = list(replies or [])
        self._available = available
        self.systems: list[str] = []
        self.users: list[str] = []

    def available(self) -> bool:
        return self._available

    @property
    def model(self) -> str:
        return "stub"

    def complete(self, *, system: str, user: str, **_kw) -> str | None:
        payload = self.complete_json(system=system, user=user)
        return None if payload is None else json.dumps(payload)

    def complete_json(self, *, system: str, user: str, **_kw) -> dict | None:
        self.systems.append(system)
        self.users.append(user)
        return self._replies.pop(0) if self._replies else {"triggers": []}


def _entry(entry_id: str, user: str, assistant: str = "Understood.") -> SubstrateEntry:
    return SubstrateEntry(
        entry_id=entry_id,
        session_key="s",
        user=user,
        assistant=assistant,
        ts=1.0,
        metadata={},
        digest=content_hash(user, assistant),
    )


def _proposal(*triggers: tuple[str, str, float]) -> dict:
    return {
        "triggers": [{"concept": c, "bridge": b, "confidence": conf} for c, b, conf in triggers]
    }


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


# -- identity ------------------------------------------------------------------


def test_trigger_id_is_derived_from_entry_and_concept() -> None:
    assert trigger_id("e1", "storage cleanup") == trigger_id("e1", "  Storage   Cleanup ")
    assert trigger_id("e1", "storage cleanup") != trigger_id("e2", "storage cleanup")


# -- the deterministic filters, which are the point of this layer --------------


def test_an_ungrounded_bridge_is_rejected() -> None:
    """The cheapest guard here: a deterministic check on LLM output.

    The prompt demands the cue come from the turn's own wording, so a bridge
    citing nothing in the turn means the model invented the link rather than
    derived it — and an invented link is exactly what drags an unrelated
    memory into an answer.
    """
    turn = "The nightly job that trims old blobs is called reaper."
    assert bridge_is_grounded("trims old blobs -> trimming stored data is cleanup", turn)
    assert not bridge_is_grounded("quarterly revenue -> finance planning", turn)
    assert not bridge_is_grounded("", turn)
    # A cue of nothing but stopwords is not grounding either.
    assert not bridge_is_grounded("the it is -> something", turn)


def test_a_low_confidence_trigger_is_not_indexed(tmp_path: Path) -> None:
    store = TriggerStore()
    llm = StubLLM(
        [
            _proposal(
                ("storage cleanup", "trims old blobs -> trimming data is cleanup", 0.9),
                ("infrastructure", "job -> it is infrastructure", MIN_CONFIDENCE - 0.01),
            )
        ]
    )
    entry = _entry("e1", "The nightly job that trims old blobs is called reaper.")
    assert store.generate_missing(tmp_path, [entry], llm) == 1
    assert [t.concept for t in store.load(tmp_path)["e1"]] == ["storage cleanup"]


def test_an_ungrounded_trigger_is_dropped_at_write_time(tmp_path: Path) -> None:
    store = TriggerStore()
    llm = StubLLM([_proposal(("quarterly revenue", "board deck -> finance planning", 0.95))])
    entry = _entry("e1", "The nightly job that trims old blobs is called reaper.")
    assert store.generate_missing(tmp_path, [entry], llm) == 0
    assert store.load(tmp_path) == {}


def test_the_prompt_carries_the_three_disqualifiers(tmp_path: Path) -> None:
    """What separates a trigger from a paraphrase.

    A restatement adds no retrieval reach at all, and that is the failure mode
    a summarisation prompt walks straight into.
    """
    store = TriggerStore()
    llm = StubLLM()
    store.generate_missing(tmp_path, [_entry("e1", "anything at all")], llm)
    system = llm.systems[0]
    assert "Restatement" in system
    assert "Over-general label" in system
    assert "Weakly-predictive scene" in system
    assert "is-a ladder" in system


# -- retrieval discipline ------------------------------------------------------


def test_search_returns_entry_ids_never_trigger_text(tmp_path: Path) -> None:
    store = TriggerStore()
    store.append(
        tmp_path,
        [
            Trigger(
                entry_id="e1", concept="storage cleanup", bridge="blobs -> cleanup", confidence=0.9
            )
        ],
    )
    hits = store.search(tmp_path, "storage cleanup")
    assert hits and hits[0][0] == "e1"
    assert all(isinstance(entry_id, str) for entry_id, _score in hits)


def test_an_entry_scores_its_best_trigger_not_their_sum(tmp_path: Path) -> None:
    """Summing would reward a turn merely for having had many triggers made."""
    store = TriggerStore()
    store.append(
        tmp_path,
        [
            Trigger(
                entry_id="one", concept="storage cleanup", bridge="blobs -> cleanup", confidence=0.9
            ),
            Trigger(
                entry_id="many",
                concept="storage cleanup",
                bridge="blobs -> cleanup",
                confidence=0.9,
            ),
            Trigger(
                entry_id="many",
                concept="storage retention",
                bridge="blobs -> retention",
                confidence=0.9,
            ),
            Trigger(
                entry_id="many", concept="storage policy", bridge="blobs -> policy", confidence=0.9
            ),
        ],
    )
    scores = dict(store.search(tmp_path, "storage cleanup"))
    assert scores["one"] == pytest.approx(scores["many"])


@pytest.mark.parametrize("query", ["", "   ", "the and of"])
def test_empty_or_all_stopword_queries_return_nothing(tmp_path: Path, query: str) -> None:
    store = TriggerStore()
    store.append(tmp_path, [Trigger(entry_id="e1", concept="storage cleanup", confidence=0.9)])
    assert store.search(tmp_path, query) == []


def test_a_trigger_whose_entry_is_gone_is_dropped(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        bank_dir = tmp_path / "banks" / "b"
        bank_dir.mkdir(parents=True, exist_ok=True)
        TriggerStore().append(
            bank_dir, [Trigger(entry_id="nonexistent", concept="widget handling", confidence=0.9)]
        )
        # The trigger is a derivative, so a missing source wins.
        assert engine.recall(bank="b", query="widget handling")["results"] == []
    finally:
        engine.stop()


def test_rederive_drops_the_trigger_sidecar(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        bank_dir = tmp_path / "banks" / "b"
        TriggerStore().append(
            bank_dir, [Trigger(entry_id="whatever", concept="a thing", confidence=0.9)]
        )
        out = engine.rederive(bank="b")
        assert "triggers" in out["derived_dropped"]
        assert TriggerStore().load(bank_dir) == {}
    finally:
        engine.stop()


# -- generation semantics ------------------------------------------------------


def test_generation_is_skipped_without_a_model(tmp_path: Path) -> None:
    store = TriggerStore()
    # 0, not None: "no model configured" is a supported deployment, not an outage.
    assert store.generate_missing(tmp_path, [_entry("e1", "x")], StubLLM(available=False)) == 0


def test_an_outage_returns_none_not_zero(tmp_path: Path) -> None:
    store = TriggerStore()
    assert store.generate_missing(tmp_path, [_entry("e1", "x")], StubLLM([None])) is None


def test_a_malformed_answer_is_not_an_outage(tmp_path: Path) -> None:
    store = TriggerStore()
    assert store.generate_missing(tmp_path, [_entry("e1", "x")], StubLLM([{"junk": 1}])) == 0


def test_partial_progress_is_published_before_an_outage(tmp_path: Path) -> None:
    store = TriggerStore()
    result = store.generate_missing(
        tmp_path,
        [_entry("e1", "trims old blobs nightly"), _entry("e2", "second"), _entry("e3", "third")],
        StubLLM([_proposal(("storage cleanup", "trims old blobs -> cleanup", 0.9)), None]),
    )
    assert result is None  # the run still reports the outage...
    assert set(store.load(tmp_path)) == {"e1"}  # ...but what succeeded was kept


def test_existing_triggers_are_not_regenerated(tmp_path: Path) -> None:
    store = TriggerStore()
    entry = _entry("e1", "trims old blobs nightly")
    llm = StubLLM([_proposal(("storage cleanup", "trims old blobs -> cleanup", 0.9))])
    assert store.generate_missing(tmp_path, [entry], llm) == 1
    assert store.generate_missing(tmp_path, [entry], llm) == 0
    assert len(llm.users) == 1


def test_stale_triggers_are_regenerated(tmp_path: Path) -> None:
    """A digest mismatch is stale, a different problem from missing."""
    store = TriggerStore()
    store.append(
        tmp_path,
        [Trigger(entry_id="e1", concept="old idea", confidence=0.9, source_digest="stale")],
    )
    llm = StubLLM([_proposal(("storage cleanup", "trims old blobs -> cleanup", 0.9))])
    assert store.generate_missing(tmp_path, [_entry("e1", "trims old blobs nightly")], llm) == 1


# -- the consolidation stage ---------------------------------------------------


def test_the_stage_gates_off_without_a_model(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM(available=False))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "triggers")
        assert stage["status"] == StageStatus.SKIPPED.value
        assert stage["reason"] == "no_llm_configured"
    finally:
        engine.stop()


def test_an_outage_errors_the_stage_without_advancing_its_watermark(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM([None, None]))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "triggers")
        assert stage["status"] == StageStatus.ERRORED.value
        # Enrichment fails open; the run as a whole is not taken down.
        assert run["outcome"] == RunOutcome.SUCCEEDED.value

        from hindsight_unified.pipeline.stages import read_state, stage_watermark

        state = read_state(tmp_path / "banks" / "b")
        assert stage_watermark(state, "triggers") == 0
    finally:
        engine.stop()


def test_triggers_is_not_the_fatal_stage(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        assert [s.name for s in engine._pipeline._stages if s.fatal] == ["reconcile"]
        assert "triggers" in [s.name for s in engine._pipeline._stages]
    finally:
        engine.stop()


# -- the reason this layer exists ----------------------------------------------


def test_the_lane_closes_the_measured_vocabulary_gap(tmp_path: Path) -> None:
    """The gap no deterministic lane can reach, on the real failing case.

    "what cleans up storage we no longer need?" shares no term with "the
    nightly job that trims old blobs is called reaper", so no expansion
    derivable from the entry text can connect them. A write-time trigger can,
    because it is generated from the entry before any question exists.

    The trigger below is hand-written to stand in for what the prompt should
    produce, so this verifies the **mechanism** — a trigger hit dereferencing
    to a verbatim entry the question could not otherwise reach — and not the
    prompt's output quality, which needs a real model.
    """
    engine = _engine(tmp_path, llm=StubLLM())
    try:
        engine.capture(
            bank="b",
            session_key="s",
            user_content="The nightly job that trims old blobs is called reaper.",
            assistant_content="Understood.",
        )
        # Filler after the answer, so the two-turn rewrite window cannot carry
        # the entry verbatim into the query and retrieve it by self-similarity.
        for filler in (
            "Unrelated: the board refreshes every five minutes.",
            "Unrelated: we tag unreliable tests.",
        ):
            engine.capture(bank="b", session_key="s", user_content=filler, assistant_content="Ack.")

        question = "what cleans up storage we no longer need?"
        bank_dir = tmp_path / "banks" / "b"
        limit = 2  # tight, so ranking actually binds

        before = engine.recall(bank="b", query=question, limit=limit)["context"]
        assert "reaper" not in before

        entry = engine._substrate._load(bank_dir)[0]
        TriggerStore().append(
            bank_dir,
            [
                Trigger(
                    entry_id=entry.entry_id,
                    concept="storage cleanup",
                    bridge="trims old blobs -> trimming stored data is cleanup",
                    confidence=0.9,
                    source_digest=entry.digest,
                )
            ],
        )
        after = engine.recall(bank="b", query=question, limit=limit)["context"]
        assert "reaper" in after
        # The verbatim turn is what surfaces; no trigger wording leaks in.
        assert "storage cleanup" not in after
    finally:
        engine.stop()
