"""Substrate (L0 verbatim) tests — deterministic, no external deps."""

from __future__ import annotations

from pathlib import Path

from hindsight_unified.substrate import MarkdownSubstrate


def _make(tmp_path: Path) -> tuple[MarkdownSubstrate, Path]:
    sub = MarkdownSubstrate(tmp_path)
    return sub, tmp_path / "bank"


def test_append_writes_md_and_index(tmp_path: Path) -> None:
    sub, bank = _make(tmp_path)
    entry_id = sub.append(bank, session_key="s1", user="hello world", assistant="hi there")
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


def test_serialization_labels_are_not_searchable(tmp_path: Path) -> None:
    """Our own field labels must not be terms.

    ``as_text()`` renders "User: ...\\nAssistant: ...", and indexing that
    verbatim made every entry in a bank match any query containing the words
    "user" or "assistant" — which the conversational rewrite lane emits on
    every single call ("Prior user:", "Prior assistant:"). The lane could
    therefore never come back empty, and a question memory cannot answer still
    returned a full slate.
    """
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s1", user="the reaper trims blobs", assistant="ok")
    assert sub.search(bank, "user assistant") == []
    # The turn's own words still match, so this removes noise, not signal.
    assert sub.search(bank, "reaper")


def test_an_inflected_query_term_matches_its_stem(tmp_path: Path) -> None:
    """Exact-token BM25 cannot match a word against its own inflection, and a
    question almost never reuses the tense and number of the turn answering
    it."""
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s", user="Our migration runner picks files", assistant="ok")
    assert sub.search(bank, "how are migrations ordered")
    sub.append(bank, session_key="s", user="Snapshots are pruned nightly", assistant="ok")
    assert sub.search(bank, "when is a snapshot pruning done")


def test_stemming_keeps_distinct_words_apart(tmp_path: Path) -> None:
    """Over-stemming manufactures matches, which is the failure mode that the
    anchor matcher in the eval already had to undo once."""
    sub, bank = _make(tmp_path)
    sub.append(bank, session_key="s", user="the land registry export", assistant="ok")
    assert sub.search(bank, "lane") == []

