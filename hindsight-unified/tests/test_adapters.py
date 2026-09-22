"""Adapter tests — EverOS crash recovery, mempalace guard/AAAK, OpenViking
tiering, MemOS export. All deterministic (lean fallback paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hindsight_unified.adapters.everos_substrate import EverosSubstrateAdapter
from hindsight_unified.adapters.memos_portability import MemosPortabilityAdapter
from hindsight_unified.adapters.mempalace_index import MempalaceIndexAdapter
from hindsight_unified.adapters.openviking_injection import OpenVikingInjectionAdapter
from hindsight_unified.substrate import MarkdownSubstrate
from hindsight_unified.types import Recalled

# -- EverOS: index rebuilt from md truth (crash recovery) ---------------------


def test_everos_reconcile_rebuilds_lost_index(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    sub.append(bank, session_key="s1", user="fact one", assistant="ack one")
    sub.append(bank, session_key="s1", user="fact two", assistant="ack two")
    # Simulate index loss (crash / deletion). md logs remain — they are truth.
    (bank / "index.jsonl").unlink()
    assert sub.count(bank) == 0

    adapter = EverosSubstrateAdapter()
    adapter.start()
    recovered = adapter.reconcile(bank)
    assert recovered == 2
    # Substrate search works again, from the rebuilt index.
    assert sub.count(bank) == 2
    hits = sub.search(bank, "fact two")
    assert hits and "fact two" in hits[0][0].as_text()


def test_everos_reconcile_noop_when_consistent(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    sub.append(bank, session_key="s1", user="hello", assistant="hi")
    adapter = EverosSubstrateAdapter()
    adapter.start()
    assert adapter.reconcile(bank) == 0


# -- mempalace: embedder-identity guard + AAAK cards ---------------------------


def test_embedder_identity_mismatch_raises(tmp_path: Path) -> None:
    adapter = MempalaceIndexAdapter()
    adapter.start()
    bank = tmp_path / "bank"
    adapter.check_embedder_identity(bank, "all-MiniLM-L6-v2")
    adapter.check_embedder_identity(bank, "all-MiniLM-L6-v2")  # same: fine
    with pytest.raises(RuntimeError, match="mismatch"):
        adapter.check_embedder_identity(bank, "bge-large-en")


def test_aaak_cards_compact_and_indexed(tmp_path: Path) -> None:
    adapter = MempalaceIndexAdapter()
    adapter.start()
    recalled = [
        Recalled(text="A" * 500, source="hindsight", fact_type="world"),
        Recalled(text="short fact", source="substrate"),
    ]
    cards = adapter.aaak_cards(recalled)
    assert "[0]" in cards and "[1]" in cards
    # Cards are triage-sized, not full drawers.
    for line in cards.splitlines()[1:]:
        assert len(line) < 200


# -- OpenViking: tiered budget + trajectory ------------------------------------


def test_openviking_budget_and_trajectory() -> None:
    adapter = OpenVikingInjectionAdapter()
    adapter.start()
    recalled = [
        Recalled(text="fact " * 50, source="hindsight", fact_type="world"),
        Recalled(text="verbatim " * 200, source="substrate"),  # expensive L2
    ]
    context, trajectory = adapter.assemble(recalled, max_tokens=80)
    included = [t for t in trajectory if t.get("included")]
    excluded = [t for t in trajectory if not t.get("included")]
    # L1 fact fits; the oversized L2 verbatim is budgeted out — and the
    # trajectory says exactly why.
    assert any(t["tier"] == "L1" for t in included)
    assert excluded and excluded[0]["reason"] == "budget_exhausted"
    assert "fact" in context


def test_openviking_l1_before_l2() -> None:
    adapter = OpenVikingInjectionAdapter()
    adapter.start()
    recalled = [
        Recalled(text="verbatim excerpt", source="substrate"),  # no fact_type = L2
        Recalled(text="typed fact", source="hindsight", fact_type="experience"),
    ]
    context, _ = adapter.assemble(recalled, max_tokens=1000)
    assert context.index("typed fact") < context.index("verbatim excerpt")


# -- MemOS: portable export envelope --------------------------------------------


def test_memos_export_manifest(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    sub.append(bank, session_key="s1", user="exportable", assistant="ok")
    adapter = MemosPortabilityAdapter()
    adapter.start()
    manifest_path = adapter.export("bank-x", bank)
    assert manifest_path is not None and manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"].startswith("unified-memory/portable-envelope/")
    assert manifest["bank"] == "bank-x"
    assert adapter.export_paths("bank-x", bank)
