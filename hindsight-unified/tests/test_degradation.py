"""Degradation-honesty tests (workstream B5).

"Nothing matched" and "the layer that would have answered is offline" must be
distinguishable, and nothing we assemble for triage may ever be handed back as
though it were memory content.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine

# The scaffolding our own context assembler prepends. It must never appear in
# anything a caller would paste into a prompt as "memory".
AAAK_SCAFFOLD = "AAAK index"


def _engine(tmp_path: Path, *, brain: bool = False) -> UnifiedEngine:
    settings = Settings(
        host="127.0.0.1",
        port=0,
        home=tmp_path,
        enable_hindsight=brain,
        enable_everos=True,
        enable_mempalace=True,
        enable_openviking=True,
        enable_memos=True,
    )
    engine = UnifiedEngine(settings)
    engine.start()
    return engine


def test_empty_recall_is_distinguishable_from_a_disabled_brain(tmp_path: Path) -> None:
    engine = _engine(tmp_path, brain=False)
    try:
        out = engine.recall(bank="b", query="anything at all")
        assert out["results"] == []
        marker = out["marker"]
        # The whole point: an empty result carries WHY it is empty.
        assert marker["status"] == "degraded"
        assert marker["reason"] == "brain_disabled"
        assert marker["brain"] is False
        assert marker["text"]
    finally:
        engine.stop()


def test_marker_is_present_on_a_hit_too(tmp_path: Path) -> None:
    engine = _engine(tmp_path, brain=False)
    try:
        engine.capture(
            bank="b",
            session_key="s",
            user_content="my cat is called Miso",
            assistant_content="noted",
        )
        out = engine.recall(bank="b", query="cat name")
        assert out["results"], "expected a substrate hit"
        # Degradation is a property of the layers consulted, not of the result
        # count, so the caller never has to infer it from an empty list.
        assert out["marker"]["status"] == "degraded"
        assert out["marker"]["num_results"] == len(out["results"])
    finally:
        engine.stop()


def test_reflect_never_returns_our_triage_scaffolding(tmp_path: Path) -> None:
    """The degraded `answer` must be raw recalled material, not our framing.

    Cognee documents the same leak at ``agent_memory/runtime.py:421``: handing
    back a payload that contains your own question framing means the caller
    pastes your scaffolding into its prompt as if it were content.
    """
    engine = _engine(tmp_path, brain=False)
    try:
        engine.capture(
            bank="b",
            session_key="s",
            user_content="I value concise code reviews",
            assistant_content="noted",
        )
        assembled = engine.recall(bank="b", query="what do I value?")["context"]
        assert AAAK_SCAFFOLD in assembled, "precondition: the assembler adds a header"

        out = engine.reflect(bank="b", query="what do I value?")
        assert "concise" in out["answer"]
        assert AAAK_SCAFFOLD not in out["answer"]
        assert out["synthesized"] is False  # nothing reasoned over this
        assert out["marker"]["status"] == "degraded"
    finally:
        engine.stop()


def test_rederive_rebuilds_the_index_and_keeps_the_markdown(tmp_path: Path) -> None:
    engine = _engine(tmp_path, brain=False)
    try:
        for text in ("first fact", "second fact"):
            engine.capture(bank="b", session_key="s", user_content=text, assistant_content="ok")
        bank_dir = tmp_path / "banks" / "b"
        md_before = sorted(p.read_text(encoding="utf-8") for p in (bank_dir / "log").glob("*.md"))

        out = engine.rederive(bank="b")
        assert out["ok"] is True
        assert out["entries_before"] == 2
        assert out["entries_rebuilt"] == 2

        # The derived layer is back...
        assert "second fact" in engine.recall(bank="b", query="second fact")["context"]
        # ...and the truth was never touched.
        md_after = sorted(p.read_text(encoding="utf-8") for p in (bank_dir / "log").glob("*.md"))
        assert md_after == md_before
    finally:
        engine.stop()


def test_rederive_recovers_a_bank_whose_index_drifted(tmp_path: Path) -> None:
    engine = _engine(tmp_path, brain=False)
    try:
        engine.capture(
            bank="b", session_key="s", user_content="durable fact", assistant_content="ok"
        )
        # Simulate a derivation change / corruption: index holds garbage.
        (tmp_path / "banks" / "b" / "index.jsonl").write_text("{not json\n", encoding="utf-8")
        engine.rederive(bank="b")
        assert "durable fact" in engine.recall(bank="b", query="durable fact")["context"]
    finally:
        engine.stop()
