"""Presentation-order tests (workstream A4).

Relevance decides what is selected; recency decides what leads. These pin the
contract that a superseded fact which is still retrievable cannot be read as
current, and that the index cards and the body never disagree about order.
"""

from __future__ import annotations

from pathlib import Path

from hindsight_unified.adapters.openviking_injection import OpenVikingInjectionAdapter
from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.types import Recalled


def _adapter() -> OpenVikingInjectionAdapter:
    adapter = OpenVikingInjectionAdapter()
    adapter.start()
    return adapter


def _engine(tmp_path: Path) -> UnifiedEngine:
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
    engine.start()
    return engine


def test_newest_entry_leads_regardless_of_relevance_order() -> None:
    adapter = _adapter()
    # Arrives relevance-ranked with the OLD entry first, as RRF would.
    recalled = [
        Recalled(text="I use tabs", source="substrate", score=9.0, metadata={"seq": 1}),
        Recalled(text="4 spaces now", source="substrate", score=1.0, metadata={"seq": 7}),
    ]
    context, _ = adapter.assemble(recalled)
    assert context.index("4 spaces now") < context.index("I use tabs")


def test_the_conflict_rule_is_stated_not_implied() -> None:
    adapter = _adapter()
    context, _ = adapter.assemble(
        [Recalled(text="a fact", source="substrate", metadata={"seq": 0})]
    )
    assert "Most recent first" in context
    assert "nearer the top" in context


def test_append_position_beats_the_timestamp_within_one_millisecond() -> None:
    """Same-millisecond entries must still order deterministically.

    Entry ids carry a ms prefix, which cannot separate entries written inside
    one millisecond — measured as run-to-run variance in the eval before the
    append position was carried through.
    """
    adapter = _adapter()
    same_ms = "0001700000000"
    recalled = [
        Recalled(text="older", source="substrate", metadata={"seq": 3, "entry_id": f"{same_ms}-aaa"}),
        Recalled(text="newer", source="substrate", metadata={"seq": 4, "entry_id": f"{same_ms}-bbb"}),
    ]
    context, _ = adapter.assemble(recalled)
    assert context.index("newer") < context.index("older")
    # And the order is stable across repeated calls on shuffled input.
    reversed_input = list(reversed(recalled))
    assert adapter.assemble(reversed_input)[0] == context


def test_items_without_a_recency_key_keep_their_relevance_order() -> None:
    adapter = _adapter()
    # Typed facts from the brain carry no entry position; reordering them would
    # discard the only ranking they have.
    recalled = [
        Recalled(text="first by relevance", source="hindsight", fact_type="world"),
        Recalled(text="second by relevance", source="hindsight", fact_type="world"),
    ]
    context, _ = adapter.assemble(recalled)
    assert context.index("first by relevance") < context.index("second by relevance")
    assert "Most recent first" not in context  # no ordering claim without keys


def test_index_cards_and_body_agree_on_what_leads(tmp_path: Path) -> None:
    """Card [0] must name the entry the body shows first.

    The cards exist so a model can scan and then open one drawer; building the
    index from the fused order and the body from the recency order would break
    exactly that.
    """
    engine = _engine(tmp_path)
    try:
        engine.capture(
            bank="b", session_key="s", user_content="I always indent with tabs", assistant_content="ok"
        )
        engine.capture(
            bank="b", session_key="s", user_content="we now indent with 4 spaces", assistant_content="ok"
        )
        context = engine.recall(bank="b", query="indent")["context"]
        card_zero = next(line for line in context.splitlines() if line.startswith("[0]"))
        body_first = next(line for line in context.splitlines() if line.startswith("- ("))
        assert "4 spaces" in card_zero
        assert "4 spaces" in body_first
    finally:
        engine.stop()


def test_a_superseded_fact_stays_retrievable_but_does_not_lead(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    try:
        engine.capture(
            bank="b", session_key="s", user_content="deploy to us-east-1", assistant_content="ok"
        )
        engine.capture(
            bank="b", session_key="s", user_content="deploy to eu-central-1 now", assistant_content="ok"
        )
        context = engine.recall(bank="b", query="deploy region")["context"]
        # Nothing is deleted — history and provenance stay readable...
        assert "us-east-1" in context
        # ...but the current fact leads.
        assert context.index("eu-central-1") < context.index("us-east-1")
    finally:
        engine.stop()
