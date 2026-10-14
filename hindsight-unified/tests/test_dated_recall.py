"""Dated recall tests.

A recalled turn used to carry no time at all, and an imported conversation was
stamped with the moment of import. Both are about the same thing - a memory
knowing when it was said - and both are tested at the reader: what the model
is shown, and in what order.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import hindsight_unified.server as server_mod
from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine, parse_ts


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


def _body(context: str) -> list[str]:
    return [line for line in context.splitlines() if line.startswith("- ")]


# -- parsing -------------------------------------------------------------------


def test_a_bare_date_is_noon_utc_so_it_keeps_its_day_everywhere() -> None:
    ts = parse_ts("2026-03-14")
    assert datetime.fromtimestamp(ts, UTC) == datetime(2026, 3, 14, 12, tzinfo=UTC)


def test_timestamps_parse_from_iso_and_epoch() -> None:
    assert parse_ts("2026-03-14T09:30:00Z") == datetime(2026, 3, 14, 9, 30, tzinfo=UTC).timestamp()
    # No offset is read as UTC rather than as whatever the server's clock says.
    assert parse_ts("2026-03-14T09:30:00") == parse_ts("2026-03-14T09:30:00Z")
    assert parse_ts(1_700_000_000) == 1_700_000_000.0
    assert parse_ts("") == 0.0 and parse_ts(None) == 0.0


@pytest.mark.parametrize("bad", ["yesterday", "2026-13-40", True])
def test_an_unusable_timestamp_is_rejected(bad) -> None:
    with pytest.raises(ValueError):
        parse_ts(bad)


# -- what the model is shown -----------------------------------------------------


def test_a_recalled_turn_carries_the_date_it_was_said(engine: UnifiedEngine) -> None:
    """Without it, "next Monday" inside a memory is unanchored."""
    engine.capture(
        bank="b",
        session_key="s",
        user_content="I'll rotate the signing keys next Monday.",
        assistant_content="Okay.",
        ts=parse_ts("2026-01-20"),
    )
    context = engine.recall(bank="b", query="rotate signing keys")["context"]
    assert "**2026-01-20**" in context


# -- order -----------------------------------------------------------------------


def test_an_imported_older_statement_does_not_present_as_newest(engine: UnifiedEngine) -> None:
    """Recency used to be append position alone. An older turn imported after a
    newer one then led the context, and the conflict rule ("follow the one
    nearer the top") would follow the outdated statement."""
    engine.capture(
        bank="b",
        session_key="s",
        user_content="We now indent with 4 spaces.",
        assistant_content="Okay.",
        ts=parse_ts("2026-06-01"),
    )
    engine.capture(  # appended later, said earlier
        bank="b",
        session_key="s",
        user_content="I always indent with tabs.",
        assistant_content="Okay.",
        ts=parse_ts("2026-01-01"),
    )
    body = _body(engine.recall(bank="b", query="indent")["context"])
    assert "4 spaces" in body[0]
    assert "tabs" in body[1]


def test_turns_said_in_the_same_millisecond_keep_append_order(engine: UnifiedEngine) -> None:
    """Position still breaks ties, which is the nondeterminism it was added to
    remove: identical timestamps must not reorder between runs."""
    for text in ("first statement about deploys", "second statement about deploys"):
        engine.capture(
            bank="b",
            session_key="s",
            user_content=text,
            assistant_content="Okay.",
            ts=1_700_000_000.0,
        )
    body = _body(engine.recall(bank="b", query="statement deploys")["context"])
    assert "second" in body[0] and "first" in body[1]


# -- over the wire ---------------------------------------------------------------


@pytest.fixture()
def http_base(tmp_path: Path):
    eng = _engine(tmp_path)
    server_mod._engine = eng
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    eng.stop()
    server_mod._engine = None


def _post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def test_imported_history_keeps_its_own_dates(http_base: str) -> None:
    """/seed used to stamp every imported turn with the moment of import, so a
    year-old conversation read as the newest thing the user had said."""
    _post(
        http_base,
        "/seed",
        {
            "bank": "b",
            "data": [
                {"user": "The billing migration finished this morning.", "ts": "2025-11-03"},
            ],
        },
    )
    recall = _post(http_base, "/recall", {"bank": "b", "query": "billing migration"})
    assert "**2025-11-03**" in recall["context"]


def test_a_bad_timestamp_is_the_callers_error(http_base: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(
            http_base,
            "/capture",
            {"bank": "b", "user_content": "x", "assistant_content": "y", "ts": "soon"},
        )
    assert excinfo.value.code == 400
