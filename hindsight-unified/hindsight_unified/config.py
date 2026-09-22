"""Configuration for the unified memory sidecar.

All settings resolve from the environment so the sidecar can be launched by
the Hermes supervisor with nothing but env vars. Every getter is
exception-safe: a malformed value logs a warning and falls back to the
default, because the supervisor's ``is_available`` contract forbids throwing
during resolution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766  # distinct from hindsight-api (8888) and tencentdb (8420)


def _home() -> Path:
    """Data root: verbatim md substrate + exports live here.

    Priority: UNIFIED_MEMORY_HOME > $HERMES_HOME/unified > ~/.hermes/unified.
    """
    env = os.environ.get("UNIFIED_MEMORY_HOME")
    if env and env.strip():
        return Path(env.strip()).expanduser()
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home and hermes_home.strip():
        return Path(hermes_home.strip()).expanduser() / "unified"
    return Path.home() / ".hermes" / "unified"


def _port(default: int = DEFAULT_PORT) -> int:
    raw = os.environ.get("UNIFIED_MEMORY_GATEWAY_PORT")
    if raw is None or not raw.strip():
        return default
    try:
        port = int(raw.strip())
    except ValueError:
        logger.warning("Invalid UNIFIED_MEMORY_GATEWAY_PORT=%r; using %d", raw, default)
        return default
    if not (1 <= port <= 65535):
        logger.warning("UNIFIED_MEMORY_GATEWAY_PORT=%d out of range; using %d", port, default)
        return default
    return port


def _host(default: str = DEFAULT_HOST) -> str:
    raw = os.environ.get("UNIFIED_MEMORY_GATEWAY_HOST")
    return raw.strip() if raw and raw.strip() else default


def _int(name: str, default: int) -> int:
    """Non-negative int from the environment; a bad value warns and defaults."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d is negative; using %d", name, value, default)
        return default
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(slots=True)
class Settings:
    """Resolved sidecar settings."""

    host: str
    port: int
    home: Path
    # Per-adapter enable flags. Off means "never even attempt the import",
    # which keeps a broken/heavy optional dep from slowing startup.
    enable_hindsight: bool
    enable_everos: bool
    enable_mempalace: bool
    enable_openviking: bool
    enable_memos: bool
    # L2 debounce: a mid-session consolidation fires when EITHER enough new
    # entries accumulated OR enough time passed since the last run. Both at 0
    # means every call consolidates. Session end forces a run regardless.
    consolidate_min_entries: int = 20
    consolidate_min_seconds: float = 900.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            host=_host(),
            port=_port(),
            home=_home(),
            enable_hindsight=_bool("UNIFIED_MEMORY_ENABLE_HINDSIGHT", True),
            enable_everos=_bool("UNIFIED_MEMORY_ENABLE_EVEROS", True),
            enable_mempalace=_bool("UNIFIED_MEMORY_ENABLE_MEMPALACE", True),
            enable_openviking=_bool("UNIFIED_MEMORY_ENABLE_OPENVIKING", True),
            enable_memos=_bool("UNIFIED_MEMORY_ENABLE_MEMOS", True),
            consolidate_min_entries=_int("UNIFIED_MEMORY_CONSOLIDATE_MIN_ENTRIES", 20),
            consolidate_min_seconds=float(_int("UNIFIED_MEMORY_CONSOLIDATE_MIN_SECONDS", 900)),
        )

    def bank_dir(self, bank: str) -> Path:
        """Directory holding one bank's verbatim md substrate."""
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in bank) or "default"
        return self.home / "banks" / safe
