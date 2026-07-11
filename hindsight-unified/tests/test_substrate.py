"""Substrate (L0 verbatim) tests — deterministic, no external deps."""

from __future__ import annotations

from pathlib import Path

from hindsight_unified.substrate import MarkdownSubstrate


def _make(tmp_path: Path) -> tuple[MarkdownSubstrate, Path]:
    sub = MarkdownSubstrate(tmp_path)
    return sub, tmp_path / "bank"


def test_append_writes_md_and_index(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    entry_id = sub.append(
        bank, session_key="s1", user="hello world", assistant="hi there"
    )
    assert entry_id
    md_files = list((bank / "log").glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    # Verbatim discipline: the exact words appear, unmodified.
    assert "hello world" in content
    assert "hi there" in content
    assert (bank / "index.jsonl").exists()


def test_search_finds_verbatim_entry(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s1", user="my dog is called Loki", assistant="noted")
    sub.append(bank, session_key="s1", user="I like PostgreSQL", assistant="ok")
    hits = sub.search(bank, "what is my dog's name?")
    assert hits
    assert "Loki" in hits[0][0].as_text()


def test_search_stopwords_do_not_outrank_content_terms(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s1", user="review the PR tomorrow", assistant="will do")
    sub.append(bank, session_key="s1", user="favorite database is PostgreSQL", assistant="ok")
    hits = sub.search(bank, "what database does the user like?")
    assert hits
    assert "PostgreSQL" in hits[0][0].as_text()


def test_search_session_filter(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="a", user="alpha topic", assistant="")
    sub.append(bank, session_key="b", user="alpha topic again", assistant="")
    hits = sub.search(bank, "alpha topic", session_key="a")
    assert len(hits) == 1
    assert hits[0][0].session_key == "a"


def test_torn_index_line_is_skipped(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s1", user="good entry", assistant="ok")
    with open(bank / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"id": "torn-entr')  # simulated torn write
    assert sub.count(bank) == 1
    assert sub.search(bank, "good entry")
