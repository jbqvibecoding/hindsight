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
    a = [Recalled(text="Shared Fact", source="hindsight"), Recalled(text="only a", source="hindsight")]
    b = [Recalled(text="shared  fact", source="substrate"), Recalled(text="only b", source="substrate")]
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
            bank="b1", session_key="s1",
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
        engine.capture(bank="alice", session_key="s", user_content="alice secret zebra", assistant_content="")
        engine.capture(bank="bob", session_key="s", user_content="bob topic yak", assistant_content="")
        out = engine.recall(bank="bob", query="secret zebra")
        assert "zebra" not in out["context"]  # strict bank isolation
    finally:
        engine.stop()


def test_engine_reflect_degraded_fallback(tmp_path: Path) -> None:
    engine = UnifiedEngine(_settings(tmp_path))
    engine.start()
    try:
        engine.capture(bank="b", session_key="s", user_content="I value concise reviews", assistant_content="")
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
    cap = _post(http_sidecar, "/capture", {
        "session_key": "bank-h", "user_content": "my cat is named Miso",
        "assistant_content": "noted: Miso",
    })
    assert cap["ok"] is True
    rec = _post(http_sidecar, "/recall", {"session_key": "bank-h", "query": "cat name"})
    assert "Miso" in rec["context"]


def test_http_search_conversations_and_session_end(http_sidecar: str) -> None:
    _post(http_sidecar, "/capture", {
        "session_key": "bank-c", "user_content": "exact phrase alpha bravo",
        "assistant_content": "",
    })
    conv = _post(http_sidecar, "/search/conversations", {
        "session_key": "bank-c", "query": "alpha bravo",
    })
    assert conv["results"] and "alpha bravo" in conv["results"][0]["text"]
    end = _post(http_sidecar, "/session/end", {"session_key": "bank-c"})
    assert end["ok"] is True


def test_http_seed_batch(http_sidecar: str) -> None:
    out = _post(http_sidecar, "/seed", {
        "session_key": "bank-s",
        "data": [
            {"user_content": "seeded one", "assistant_content": "ok"},
            {"user_content": "seeded two", "assistant_content": "ok"},
        ],
    })
    assert out["rounds_processed"] == 2
    rec = _post(http_sidecar, "/recall", {"session_key": "bank-s", "query": "seeded two"})
    assert "seeded two" in rec["context"]


def test_http_unknown_endpoint_404(http_sidecar: str) -> None:
    req = urllib.request.Request(
        f"{http_sidecar}/nope", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404
