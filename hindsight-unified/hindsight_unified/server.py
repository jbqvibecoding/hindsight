"""Unified memory sidecar — a dependency-free stdlib HTTP server.

Deliberately built on ``http.server`` rather than FastAPI so the sidecar has
zero third-party requirements of its own: the heavy dependencies belong to the
optional contributor engines (Hindsight, MemOS, ...), which load lazily inside
the ``UnifiedEngine`` and degrade when absent. A dependency-free sidecar is also
far simpler for the Hermes supervisor to launch and health-check.

Launch:  ``python -m hindsight_unified.server``
Endpoints mirror the tencentdb Gateway surface (so the Hermes thin provider is a
near-copy) plus ``/reflect``:

    GET  /health
    POST /recall               {query, bank|session_key, user_id, limit, ...}
    POST /capture              {user_content, assistant_content, bank|session_key}
    POST /search/memories      {query, limit, type}
    POST /search/conversations {query, limit, session_key}
    POST /session/end          {bank|session_key}
    POST /reflect              {query, bank|session_key}
    POST /seed                 {data: [...]}
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import Settings
from .engine import UnifiedEngine
from .substrate import SingletonLockHeld

logger = logging.getLogger(__name__)

# Module-level engine so every threaded handler shares one instance (and one
# warm brain / connection pool).
_engine: UnifiedEngine | None = None


def _bank_of(body: dict[str, Any]) -> str:
    """The bank id is the tenant key. Hermes sends ``session_key`` (its bank);
    accept an explicit ``bank`` too. Falls back to 'default'."""
    return (body.get("bank") or body.get("session_key") or "default").strip() or "default"


class Handler(BaseHTTPRequestHandler):
    server_version = "UnifiedMemory/1.0"

    # Quieter logging — one line per request at debug level.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send(200, _get_engine().health())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        engine = _get_engine()
        body = self._read_body()
        path = self.path.rstrip("/") or "/"
        try:
            if path == "/recall":
                self._send(200, engine.recall(
                    bank=_bank_of(body),
                    query=body.get("query", ""),
                    session_key=body.get("session_key", ""),
                    user_id=body.get("user_id", ""),
                    limit=int(body.get("limit", 8) or 8),
                    fact_type=body.get("type", "") or body.get("fact_type", ""),
                    question_date=body.get("question_date", ""),
                ))
            elif path == "/capture":
                self._send(200, engine.capture(
                    bank=_bank_of(body),
                    session_key=body.get("session_key", ""),
                    user_content=body.get("user_content", ""),
                    assistant_content=body.get("assistant_content", ""),
                    user_id=body.get("user_id", ""),
                ))
            elif path == "/search/memories":
                self._send(200, engine.recall(
                    bank=_bank_of(body),
                    query=body.get("query", ""),
                    limit=int(body.get("limit", 5) or 5),
                    fact_type=body.get("type", ""),
                ))
            elif path == "/search/conversations":
                self._send(200, engine.search_conversations(
                    bank=_bank_of(body),
                    query=body.get("query", ""),
                    limit=int(body.get("limit", 5) or 5),
                    session_key=body.get("session_key", ""),
                ))
            elif path == "/session/end":
                self._send(200, engine.end_session(
                    bank=_bank_of(body), session_key=body.get("session_key", ""),
                ))
            elif path == "/reflect":
                self._send(200, engine.reflect(
                    bank=_bank_of(body), query=body.get("query", ""),
                ))
            elif path == "/seed":
                self._send(200, _handle_seed(engine, body))
            else:
                self._send(404, {"error": f"unknown endpoint: {path}"})
        except Exception as e:  # noqa: BLE001 - never leak a stack to the client
            logger.exception("request to %s failed", path)
            self._send(500, {"error": str(e)})


def _handle_seed(engine: UnifiedEngine, body: dict[str, Any]) -> dict[str, Any]:
    """Batch-ingest historical turns. Accepts ``{"data": [...]}`` where each
    item is ``{user_content, assistant_content, session_key?}``."""
    data = body.get("data") or []
    if isinstance(data, dict):
        data = data.get("sessions", []) or data.get("rounds", [])
    count = 0
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        bank = _bank_of({**body, **item})
        engine.capture(
            bank=bank,
            session_key=item.get("session_key", body.get("session_key", "")),
            user_content=item.get("user_content", "") or item.get("user", ""),
            assistant_content=item.get("assistant_content", "") or item.get("assistant", ""),
            user_id=item.get("user_id", ""),
        )
        count += 1
    return {"ok": True, "rounds_processed": count}


def _get_engine() -> UnifiedEngine:
    global _engine
    if _engine is None:
        _engine = UnifiedEngine()
        _engine.start()
    return _engine


def serve(settings: Settings | None = None) -> None:
    global _engine
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [unified-memory] %(message)s",
    )
    _engine = UnifiedEngine(settings)
    try:
        _engine.start(require_singleton=True)
    except SingletonLockHeld as e:
        # Not a crash: the supervisor's watchdog can resurrect a sidecar beside
        # a hung-but-alive one, and the loser must decline rather than write.
        logger.warning("unified memory sidecar not starting: %s", e)
        _engine = None
        raise SystemExit(0) from None
    httpd = ThreadingHTTPServer((settings.host, settings.port), Handler)
    logger.info("unified memory sidecar listening on http://%s:%d", settings.host, settings.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down unified memory sidecar")
    finally:
        httpd.shutdown()
        if _engine is not None:
            _engine.stop()


if __name__ == "__main__":
    serve()
