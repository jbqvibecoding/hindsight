"""A background asyncio loop for driving async engines from sync HTTP code.

The sidecar's HTTP layer is stdlib ``http.server`` (synchronous, zero-dep) but
the Hindsight brain is fully ``async``. Rather than spin up an event loop per
request, we run one persistent loop in a daemon thread and submit coroutines to
it with ``run_coroutine_threadsafe``. This keeps a single engine instance and
its connection pool warm across requests.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncRunner:
    """Owns a dedicated event loop running in a background daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="unified-async-loop")
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the loop and block for its result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Fire-and-forget: schedule ``coro`` without waiting. Errors logged."""

        def _wrap() -> Coroutine[Any, Any, Any]:
            return coro

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _done(f: Any) -> None:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001 - background task must not crash loop
                logger.warning("unified async background task failed: %s", e)

        fut.add_done_callback(_done)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
