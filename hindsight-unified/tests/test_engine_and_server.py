"""Engine + HTTP server tests — the full L0→L3 loop over real HTTP."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import hindsight_unified.server as server_mod
from hindsight_unified.config import Settings
from hindsight_unified.engine import UnifiedEngine
from hindsight_unified.pipeline import rrf_fuse
from hindsight_unified.types import Recalled


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=0,
        home=tmp_path,
        enable_hindsight=False,  # deterministic: substrate-only brain-off mode
        enable_everos=True,
        enable_mempalace=True,
        enable_openviking=True,
        enable_memos=True,
    )


# -- RRF fusion -----------------------------------------------------------------


def test_rrf_fuse_merges_and_dedups() -> None:
    a = [
        Recalled(text="Shared Fact", source="hindsight"),
        Recalled(text="only a", source="hindsight"),
    ]
    b = [
        Recalled(text="shared  fact", source="substrate"),
        Recalled(text="only b", source="substrate"),
    ]
    fused = rrf_fuse([a, b], limit=10)
    texts = [" ".join(r.text.split()).lower() for r in fused]
    assert texts.count("shared fact") == 1  # dedup across adapters
    assert fused[0].text.lower().replace("  ", " ") == "shared fact"  # rank-1 in both wins


# -- engine ------------------------------------------------------------------------


def test_engine_capture_recall_roundtrip(tmp_path: Path) -> None:
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    try:
        engine.capture(
            bank="b1",
            session_key="s1",
            user_content="the deploy password hint is 'sunflower'",
            assistant_content="stored",
        )
        out = engine.recall(bank="b1", query="deploy password hint")
        assert "sunflower" in out["context"]
        assert out["meta"]["sources"] == ["substrate"]
        # health is degraded (brain off) but functional.
        assert engine.health()["status"] == "degraded"
    finally:
        engine.stop()


def test_engine_bank_isolation(tmp_path: Path) -> None:
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    try:
        engine.capture(
            bank="alice", session_key="s", user_content="alice secret zebra", assistant_content=""
        )
        engine.capture(
            bank="bob", session_key="s", user_content="bob topic yak", assistant_content=""
        )
        out = engine.recall(bank="bob", query="secret zebra")
        assert "zebra" not in out["context"]  # strict bank isolation
    finally:
        engine.stop()


def test_engine_reflect_degraded_fallback(tmp_path: Path) -> None:
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    try:
        engine.capture(
            bank="b", session_key="s", user_content="I value concise reviews", assistant_content=""
        )
        out = engine.reflect(bank="b", query="what do I value?")
        assert out["source"] == "substrate"
        assert "concise" in out["answer"]
    finally:
        engine.stop()


# -- HTTP server ---------------------------------------------------------------------


@pytest.fixture()
def http_sidecar(tmp_path: Path):
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    server_mod._engine = engine
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    engine.stop()
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


def test_http_health(http_sidecar: str) -> None:
    with urllib.request.urlopen(f"{http_sidecar}/health", timeout=5) as resp:
        health = json.loads(resp.read().decode())
    assert health["status"] in ("ok", "degraded")
    assert "adapters" in health


def test_http_capture_then_recall(http_sidecar: str) -> None:
    cap = _post(
        http_sidecar,
        "/capture",
        {
            "session_key": "bank-h",
            "user_content": "my cat is named Miso",
            "assistant_content": "noted: Miso",
        },
    )
    assert cap["ok"] is True
    rec = _post(http_sidecar, "/recall", {"session_key": "bank-h", "query": "cat name"})
    assert "Miso" in rec["context"]


def test_http_search_conversations_and_session_end(http_sidecar: str) -> None:
    _post(
        http_sidecar,
        "/capture",
        {
            "session_key": "bank-c",
            "user_content": "exact phrase alpha bravo",
            "assistant_content": "",
        },
    )
    conv = _post(
        http_sidecar,
        "/search/conversations",
        {
            "session_key": "bank-c",
            "query": "alpha bravo",
        },
    )
    assert conv["results"] and "alpha bravo" in conv["results"][0]["text"]
    end = _post(http_sidecar, "/session/end", {"session_key": "bank-c"})
    assert end["ok"] is True


def test_http_seed_batch(http_sidecar: str) -> None:
    out = _post(
        http_sidecar,
        "/seed",
        {
            "session_key": "bank-s",
            "data": [
                {"user_content": "seeded one", "assistant_content": "ok"},
                {"user_content": "seeded two", "assistant_content": "ok"},
            ],
        },
    )
    assert out["rounds_processed"] == 2
    rec = _post(http_sidecar, "/recall", {"session_key": "bank-s", "query": "seeded two"})
    assert "seeded two" in rec["context"]


def test_http_unknown_endpoint_404(http_sidecar: str) -> None:
    req = urllib.request.Request(
        f"{http_sidecar}/nope",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404


# -- idempotent capture, end to end --------------------------------------------


class _RecordingAdapter:
    """Counts the turns fanned out to L1, which is what a replay must not reach."""

    name = "recording"

    def __init__(self) -> None:
        self.captured: list[str] = []

    def available(self) -> bool:
        return True

    def capture(self, event, bank_dir) -> None:
        self.captured.append(event.user)

    def recall_enrich(self, req, bank_dir) -> list:
        return []

    def consolidate(self, bank, bank_dir) -> None:
        return None

    def export_paths(self, bank, bank_dir) -> list:
        return []


def test_a_retried_capture_over_http_is_written_once(http_sidecar: str) -> None:
    """The key was supported by the substrate for a long time and passed by no
    caller above it. This is the path a real retry takes."""
    body = {
        "bank": "b",
        "session_key": "s",
        "user_content": "the staging db listens on 6543",
        "assistant_content": "Understood.",
        "idempotency_key": "sess-1:turn-3",
    }
    first = _post(http_sidecar, "/capture", body)
    second = _post(http_sidecar, "/capture", body)
    assert first["entry_id"] == second["entry_id"]
    assert (first["duplicate"], second["duplicate"]) == (False, True)
    assert server_mod._engine._substrate.count(server_mod._engine._settings.bank_dir("b")) == 1


def test_a_replayed_turn_is_not_fanned_out_to_the_adapters(tmp_path: Path) -> None:
    """Otherwise the brain ingests the turn twice while the log holds it once."""
    engine = UnifiedEngine(_settings(tmp_path))
    recorder = _RecordingAdapter()
    engine._pipeline._adapters.append(recorder)
    engine.start()
    try:
        for _ in range(3):
            engine.capture(
                bank="b",
                session_key="s",
                user_content="a fact",
                assistant_content="ok",
                idempotency_key="k1",
            )
        assert recorder.captured == ["a fact"]
    finally:
        engine.stop()


def test_identical_turns_without_a_key_stay_distinct(tmp_path: Path) -> None:
    """Content is not identity: a user who says the same thing twice said it twice."""
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    try:
        a = engine.capture(bank="b", session_key="s", user_content="yes", assistant_content="ok")
        b = engine.capture(bank="b", session_key="s", user_content="yes", assistant_content="ok")
        assert a["entry_id"] != b["entry_id"]
        assert not a["duplicate"] and not b["duplicate"]
    finally:
        engine.stop()


def test_a_retry_racing_its_original_writes_once(tmp_path: Path) -> None:
    """The realistic case: a request times out client-side while the sidecar is
    still completing it, and the retry arrives concurrently."""
    engine = UnifiedEngine(_settings(tmp_path))
    recorder = _RecordingAdapter()
    engine._pipeline._adapters.append(recorder)
    engine.start()
    try:
        results: list[dict] = []
        barrier = threading.Barrier(8)

        def _send() -> None:
            barrier.wait()
            results.append(
                engine.capture(
                    bank="b",
                    session_key="s",
                    user_content="same turn",
                    assistant_content="ok",
                    idempotency_key="race-1",
                )
            )

        threads = [threading.Thread(target=_send) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({r["entry_id"] for r in results}) == 1
        assert sum(not r["duplicate"] for r in results) == 1
        assert len(recorder.captured) == 1
    finally:
        engine.stop()
