"""Two-stage distillation tests (workstream D2).

Driven by a stub client, because no key is configured here. That covers the
plumbing and the failure semantics — which is the half where the bugs live, and
the half a real model would not exercise reliably anyway. The prompts'
behaviour with a real model is **not** validated by these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from hindsight_unified.config import Settings
from hindsight_unified.distill import Lesson, LessonStore, lesson_id
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.pipeline.stages import (
    RunOutcome,
    StageStatus,
    read_state,
    stage_watermark,
)
from hindsight_unified.substrate import SubstrateEntry, content_hash


class ScriptedLLM:
    """Returns queued JSON payloads. ``None`` in the queue means an outage."""

    def __init__(self, replies: list[dict | None], *, available: bool = True):
        self._replies = list(replies)
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
        return self._replies.pop(0) if self._replies else {}


def _entry(entry_id: str, user: str) -> SubstrateEntry:
    return SubstrateEntry(
        entry_id=entry_id,
        session_key="s",
        user=user,
        assistant="Noted.",
        ts=1.0,
        metadata={},
        digest=content_hash(user, "Noted."),
    )


def _proposal(statement: str, ids: list[str]) -> dict:
    return {"lessons": [{"working_statement": statement, "member_entry_ids": ids}]}


def _accept(statement: str, entities: list[str] | None = None) -> dict:
    return {
        "accept": True,
        "reason": None,
        "statement": statement,
        "entities": entities or [],
        "why_learned": "while reviewing a deploy",
    }


def _reject(reason: str) -> dict:
    return {"accept": False, "reason": reason, "statement": "", "entities": []}


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


def test_lesson_id_is_content_derived_and_time_free() -> None:
    """A lesson re-accepted later must be absorbed, not stored twice.

    This is why the rendered lesson carries no run date: cognee's own note is
    that an identical lesson must hash identically across runs.
    """
    assert lesson_id("We deploy to eu-west-3.") == lesson_id("  we  DEPLOY to eu-west-3. ")
    assert lesson_id("a") != lesson_id("b")
    rendered = Lesson(statement="X happens.", why_learned="during an incident").render()
    assert "X happens." in rendered
    assert "20" not in rendered  # no year, no timestamp


# -- the two stages ------------------------------------------------------------


def test_accepted_lesson_is_persisted_with_its_sources(tmp_path: Path) -> None:
    store = LessonStore()
    llm = ScriptedLLM(
        [
            _proposal("The team deploys to eu-west-3.", ["e1"]),
            _accept("The team deploys to eu-west-3.", ["eu-west-3"]),
        ]
    )
    assert store.distill(tmp_path, [_entry("e1", "We deploy to eu-west-3 now.")], llm) == 1
    lessons = list(store.load(tmp_path).values())
    assert lessons[0].entities == ["eu-west-3"]
    # Provenance kept, so the verbatim source stays one lookup away.
    assert lessons[0].member_entry_ids == ["e1"]


def test_a_rejected_lesson_is_not_persisted(tmp_path: Path) -> None:
    store = LessonStore()
    llm = ScriptedLLM([_proposal("Session-local trivia.", ["e1"]), _reject("not_durable")])
    assert store.distill(tmp_path, [_entry("e1", "please rerun that")], llm) == 0
    assert store.load(tmp_path) == {}


def test_the_judge_is_shown_competing_lessons_and_the_glossary(tmp_path: Path) -> None:
    """Novelty is checked by retrieval, not by instructing the model to be novel."""
    store = LessonStore()
    store.append(
        tmp_path,
        [Lesson(statement="The team deploys to eu-west-3.", entities=["eu-west-3"])],
    )
    llm = ScriptedLLM(
        [
            _proposal("Deployments target eu-west-3.", ["e1"]),
            _reject("already_known"),
        ]
    )
    store.distill(tmp_path, [_entry("e1", "we deploy to eu-west-3")], llm)

    judge_prompt = llm.users[-1]
    assert "SIMILAR EXISTING LESSONS" in judge_prompt
    assert "The team deploys to eu-west-3." in judge_prompt
    assert "ENTITY GLOSSARY" in judge_prompt
    assert "eu-west-3" in judge_prompt


def test_an_identical_lesson_is_absorbed_not_duplicated(tmp_path: Path) -> None:
    store = LessonStore()
    existing = Lesson(statement="The team deploys to eu-west-3.")
    store.append(tmp_path, [existing])
    llm = ScriptedLLM(
        [
            _proposal("x", ["e1"]),
            _accept("The team deploys to eu-west-3."),  # judge re-writes the same text
        ]
    )
    assert store.distill(tmp_path, [_entry("e1", "anything")], llm) == 0
    assert len(store.load(tmp_path)) == 1


def test_the_curator_prompt_carries_the_anti_hallucination_rule(tmp_path: Path) -> None:
    """The line that stops consolidation learning the agent's own guesses.

    We store both halves of every turn, so without it a claim that exists only
    in an assistant answer becomes a durable fact — and then gets learned from.
    """
    store = LessonStore()
    llm = ScriptedLLM([{"lessons": []}])
    store.distill(tmp_path, [_entry("e1", "x")], llm)
    curator = llm.systems[0]
    assert "only in an assistant answer" in curator
    assert "DURABLE ONLY" in curator
    assert "fine to return none" in curator


# -- failure semantics: the reason this module has tests ----------------------


def test_nothing_durable_is_zero_not_none(tmp_path: Path) -> None:
    store = LessonStore()
    assert store.distill(tmp_path, [_entry("e1", "x")], ScriptedLLM([{"lessons": []}])) == 0


def test_a_curator_outage_returns_none(tmp_path: Path) -> None:
    store = LessonStore()
    assert store.distill(tmp_path, [_entry("e1", "x")], ScriptedLLM([None])) is None


def test_a_judge_outage_returns_none_and_keeps_what_was_accepted(tmp_path: Path) -> None:
    store = LessonStore()
    llm = ScriptedLLM(
        [
            {
                "lessons": [
                    {"working_statement": "first", "member_entry_ids": ["e1"]},
                    {"working_statement": "second", "member_entry_ids": ["e1"]},
                ]
            },
            _accept("The first lesson."),
            None,  # the judge dies on the second
        ]
    )
    assert store.distill(tmp_path, [_entry("e1", "x")], llm) is None
    # Partial progress is real progress; the rest is simply still pending.
    assert [lesson.statement for lesson in store.load(tmp_path).values()] == ["The first lesson."]


def test_a_malformed_curator_answer_is_not_an_outage(tmp_path: Path) -> None:
    """The call happened and said nothing usable — that is an empty slice, not
    a failure, so the watermark may legitimately advance."""
    store = LessonStore()
    assert store.distill(tmp_path, [_entry("e1", "x")], ScriptedLLM([{"junk": 1}])) == 0


def test_distillation_is_skipped_without_a_model(tmp_path: Path) -> None:
    store = LessonStore()
    result = store.distill(tmp_path, [_entry("e1", "x")], ScriptedLLM([], available=False))
    assert result == 0  # a supported deployment, not an outage


# -- the consolidation stage ---------------------------------------------------


def test_the_stage_gates_off_without_a_model(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=ScriptedLLM([], available=False))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "distill")
        assert stage["status"] == StageStatus.SKIPPED.value
        assert stage["reason"] == "no_llm_configured"
    finally:
        engine.stop()


def test_an_outage_errors_the_stage_without_advancing_its_watermark(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=ScriptedLLM([None, None]))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        run = engine.end_session(bank="b")
        stage = next(s for s in run["stages"] if s["name"] == "distill")
        assert stage["status"] == StageStatus.ERRORED.value
        # Enrichment fails open — the run as a whole is not taken down.
        assert run["outcome"] == RunOutcome.SUCCEEDED.value
        state = read_state(tmp_path / "banks" / "b")
        assert stage_watermark(state, "distill") == 0
    finally:
        engine.stop()


def test_distill_is_not_the_fatal_stage(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=ScriptedLLM([]))
    try:
        assert [s.name for s in engine._pipeline._stages if s.fatal] == ["reconcile"]
        assert "distill" in [s.name for s in engine._pipeline._stages]
    finally:
        engine.stop()


# -- the reader ----------------------------------------------------------------


def test_a_lesson_is_recalled_and_labelled(tmp_path: Path) -> None:
    """A lesson IS shown, unlike a summary — so it must be unmistakable."""
    engine = _engine(tmp_path, llm=ScriptedLLM([]))
    try:
        bank_dir = tmp_path / "banks" / "b"
        bank_dir.mkdir(parents=True, exist_ok=True)
        LessonStore().append(
            bank_dir,
            [
                Lesson(
                    statement="The team standardised on four-space indentation.",
                    why_learned="during a review",
                    member_entry_ids=["e1"],
                )
            ],
        )
        out = engine.recall(bank="b", query="indentation standard")
        assert any(r["source"] == "lesson" for r in out["results"])
        assert "four-space" in out["context"]
        # Tagged as a lesson, so it cannot be read as something the user said.
        assert "(lesson)" in out["context"]
    finally:
        engine.stop()


def test_rederive_drops_lessons_too(tmp_path: Path) -> None:
    engine = _engine(tmp_path, llm=ScriptedLLM([]))
    try:
        engine.capture(bank="b", session_key="s", user_content="a fact", assistant_content="ok")
        bank_dir = tmp_path / "banks" / "b"
        LessonStore().append(bank_dir, [Lesson(statement="Something durable.")])
        out = engine.rederive(bank="b")
        assert "lessons" in out["derived_dropped"]
        assert LessonStore().load(bank_dir) == {}
    finally:
        engine.stop()
