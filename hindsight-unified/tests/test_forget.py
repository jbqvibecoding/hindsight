"""Forgetting tests.

Every test here targets a *reader*. A tombstone nothing consults is the
write-only failure this project has found in other codebases and in its own,
so the question each test asks is whether a particular path still surfaces a
forgotten turn.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import hindsight_unified.server as server_mod
from hindsight_unified import forget as forget_mod
from hindsight_unified.config import Settings
from hindsight_unified.distill import Lesson, LessonStore
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.triggers import Trigger, TriggerStore

SECRET = "My staging database password is tangerine-42."


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


@pytest.fixture()
def engine(tmp_path: Path):
    eng = _engine(tmp_path)
    yield eng
    eng.stop()


def _seed(engine: UnifiedEngine) -> str:
    """The secret, then unrelated turns so it sits outside the rewrite window."""
    out = engine.capture(bank="b", session_key="s", user_content=SECRET, assistant_content="Ack.")
    for filler in ("The grafana board refreshes every five minutes.", "We tag flaky tests."):
        engine.capture(bank="b", session_key="s", user_content=filler, assistant_content="Okay.")
    return out["entry_id"]


def _bank_dir(engine: UnifiedEngine) -> Path:
    return engine._settings.bank_dir("b")


# -- the two-phase contract ----------------------------------------------------


def test_preview_changes_nothing(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    preview = engine.forget_preview(bank="b", query="staging database password")
    assert entry_id in [c["entry_id"] for c in preview["candidates"]]
    # Still recallable: a preview is a look, never an action.
    assert "tangerine" in engine.recall(bank="b", query="staging database password")["context"]


def test_forget_takes_ids_and_reports_what_it_could_not_do(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    out = engine.forget(bank="b", entry_ids=[entry_id, "no-such-id"], reason="user asked")
    assert out["forgotten"] == [entry_id]
    # A mistyped id must not come back as "forgotten".
    assert out["unknown"] == ["no-such-id"]
    assert out["batch_id"]
    again = engine.forget(bank="b", entry_ids=[entry_id])
    assert again["forgotten"] == [] and again["already_forgotten"] == [entry_id]


def test_the_response_never_lets_a_caller_call_it_erasure(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    out = engine.forget(bank="b", entry_ids=[entry_id])
    assert out["still_on_disk"] is True
    # And it is true: the markdown truth is never rewritten.
    logs = "".join(p.read_text("utf-8") for p in (_bank_dir(engine) / "log").glob("*.md"))
    assert "tangerine-42" in logs


# -- every reader --------------------------------------------------------------


def test_a_forgotten_turn_is_not_recalled(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    out = engine.recall(bank="b", query="staging database password")
    assert "tangerine" not in out["context"]
    assert all("tangerine" not in r["text"] for r in out["results"])


def test_a_forgotten_turn_is_not_found_by_conversation_search(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    hits = engine.search_conversations(bank="b", query="staging database password")["results"]
    assert all("tangerine" not in h["text"] for h in hits)


def test_a_forgotten_turn_does_not_steer_the_rewrite_lane(engine: UnifiedEngine) -> None:
    """Hiding results is not enough if the forgotten words still shape the query.

    The rewrite lane folds the last turns into the query; a forgotten turn
    among them would keep pulling its own neighbours up.
    """
    out = engine.capture(bank="b", session_key="s", user_content=SECRET, assistant_content="Ack.")
    engine.forget(bank="b", entry_ids=[out["entry_id"]])
    recent = engine._substrate.recent_turns(_bank_dir(engine), limit=2)
    assert all("tangerine" not in user for user, _assistant in recent)


def test_a_trigger_for_a_forgotten_turn_does_not_vote(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    TriggerStore().append(
        _bank_dir(engine),
        [Trigger(entry_id=entry_id, concept="database credentials", confidence=0.9)],
    )
    assert "tangerine" in engine.recall(bank="b", query="database credentials")["context"]
    engine.forget(bank="b", entry_ids=[entry_id])
    assert "tangerine" not in engine.recall(bank="b", query="database credentials")["context"]


def test_a_lesson_drawn_from_a_forgotten_turn_is_hidden(engine: UnifiedEngine) -> None:
    """A lesson can restate its source outright, which would defeat the forget."""
    entry_id = _seed(engine)
    LessonStore().append(
        _bank_dir(engine),
        [
            Lesson(
                statement="The staging database password is tangerine-42.",
                member_entry_ids=[entry_id],
            )
        ],
    )
    engine.forget(bank="b", entry_ids=[entry_id])
    assert "tangerine" not in engine.recall(bank="b", query="staging password")["context"]


def test_consolidation_derives_nothing_new_from_a_forgotten_turn(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    live_ids = [e.entry_id for e in engine._substrate.live(_bank_dir(engine))]
    assert entry_id not in live_ids


# -- durability ----------------------------------------------------------------


def test_forgetting_survives_rederive(engine: UnifiedEngine) -> None:
    """Tombstones are intent, not a derivative, and must outlive a rebuild.

    They can only do so because entry ids do: the markdown header carries each
    id, so the rebuild reproduces them. If that ever stopped being true, a
    rederive would quietly bring forgotten memories back.
    """
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    out = engine.rederive(bank="b")
    assert "forgotten" not in out["derived_dropped"]
    assert forget_mod.forgotten_ids(_bank_dir(engine)) == {entry_id}
    assert "tangerine" not in engine.recall(bank="b", query="staging database password")["context"]


def test_restore_brings_it_back(engine: UnifiedEngine) -> None:
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    out = engine.restore(bank="b", entry_ids=[entry_id, "never-forgotten"])
    assert out["restored"] == [entry_id]
    assert out["not_forgotten"] == ["never-forgotten"]
    assert "tangerine" in engine.recall(bank="b", query="staging database password")["context"]


def test_saying_it_again_after_forgetting_makes_a_new_memory(engine: UnifiedEngine) -> None:
    """Forgetting a statement must not stop the user making it again."""
    entry_id = _seed(engine)
    engine.forget(bank="b", entry_ids=[entry_id])
    again = engine.capture(bank="b", session_key="s", user_content=SECRET, assistant_content="Ack.")
    assert again["entry_id"] != entry_id
    assert "tangerine" in engine.recall(bank="b", query="staging database password")["context"]


def test_occurrence_counting_still_sees_forgotten_entries(tmp_path: Path) -> None:
    """The one case where this decides an id: same millisecond, cold cache.

    In practice the timestamp prefix already separates a statement from its
    repetition. But two identical turns inside one millisecond are told apart
    only by their occurrence ordinal, and a fresh process rebuilds that count
    from the log. If the count skipped forgotten entries, the repeat would mint
    the forgotten entry's own id and inherit its tombstone.
    """
    from hindsight_unified.substrate import MarkdownSubstrate

    bank = tmp_path / "b"
    first = MarkdownSubstrate(tmp_path).append(
        bank, session_key="s", user="x", assistant="y", ts=1000.0
    )
    forget_mod.forget(bank, [first])
    # A fresh instance, so the ordinal is rebuilt from disk rather than cached.
    second = MarkdownSubstrate(tmp_path).append(
        bank, session_key="s", user="x", assistant="y", ts=1000.0
    )
    assert second != first
    assert second not in forget_mod.forgotten_ids(bank)


def test_a_torn_tombstone_line_is_skipped(tmp_path: Path) -> None:
    forget_mod.forget(tmp_path, ["e1"])
    with open(forget_mod.path(tmp_path), "a", encoding="utf-8") as fh:
        fh.write('{"entry_id": "e2", "act')
    assert forget_mod.forgotten_ids(tmp_path) == {"e1"}


# -- over the wire ---------------------------------------------------------------


def _post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def test_the_http_flow_is_preview_then_forget_by_id(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    server_mod._engine = eng
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _seed(eng)
        candidates = _post(base, "/forget/preview", {"bank": "b", "query": "staging password"})
        ids = [c["entry_id"] for c in candidates["candidates"] if "tangerine" in c["text"]]
        out = _post(base, "/forget", {"bank": "b", "entry_ids": ids, "reason": "rotated"})
        assert out["forgotten"] == ids
        recall = _post(base, "/recall", {"bank": "b", "query": "staging database password"})
        assert "tangerine" not in recall["context"]
    finally:
        httpd.shutdown()
        eng.stop()
        server_mod._engine = None
