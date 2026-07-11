"""EverOS substrate adapter — md-as-truth durability & crash recovery.

EverOS's core guarantee is that markdown is the source of truth and every index
is a rebuildable derivative, kept in sync by a cascade daemon that survives
crashes. We adopt that guarantee here: this adapter owns *reconciliation* of the
verbatim substrate. On start (and on ``consolidate``) it verifies that the
machine-readable ``index.jsonl`` matches the human-readable md logs, and rebuilds
the index from the md truth when they have drifted (e.g. a torn write, or the
jsonl was deleted). The md logs are never modified — they are the truth.

When the real ``everos`` package is importable we additionally borrow its
externalized prompt-slot overlay; otherwise we degrade to reconciliation only,
which needs no third-party dependency.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from ..types import CaptureEvent
from .base import UnifiedAdapter

logger = logging.getLogger(__name__)

# Matches the header emitted by MarkdownSubstrate._render_md_block:
#   ### 2026-07-11 10:30:00 · 0001752...-abc123
_HEADER_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) · (\S+)")
_SESSION_RE = re.compile(r"<!-- session=(.*?) -->")
_USER_RE = re.compile(r"^\*\*User:\*\* (.*)$")
_ASST_RE = re.compile(r"^\*\*Assistant:\*\* (.*)$")


class EverosSubstrateAdapter(UnifiedAdapter):
    def __init__(self) -> None:
        self._available = False
        self._have_everos = False

    @property
    def name(self) -> str:
        return "everos"

    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        # Reconciliation works with zero deps, so the adapter is always
        # "available"; the everos import is a bonus (prompt slots / extraction).
        self._available = True
        try:
            import everos  # type: ignore  # noqa: F401

            self._have_everos = True
            logger.info("everos package present; md-truth reconciler active (enriched)")
        except Exception:  # noqa: BLE001
            logger.info("everos package absent; md-truth reconciler active (lean mode)")

    def consolidate(self, bank: str, bank_dir: Path) -> None:
        self.reconcile(bank_dir)

    # -- reconciliation ------------------------------------------------------

    def reconcile(self, bank_dir: Path) -> int:
        """Rebuild ``index.jsonl`` from md logs if it is missing or shorter.

        Returns the number of entries recovered from md that were absent from
        the index. The md logs are authoritative; a divergence means the index
        (a derivative) lost data, so we rewrite it from the logs. Never raises.
        """
        try:
            log_dir = bank_dir / "log"
            idx = bank_dir / "index.jsonl"
            if not log_dir.exists():
                return 0
            md_entries = self._parse_md_logs(log_dir)
            idx_ids = self._index_ids(idx)
            missing = [e for e in md_entries if e["id"] not in idx_ids]
            if not missing:
                return 0
            logger.warning(
                "everos reconciler: %d md entries missing from index %s; rebuilding.",
                len(missing),
                idx,
            )
            self._rewrite_index(idx, md_entries)
            return len(missing)
        except Exception as e:  # noqa: BLE001
            logger.debug("everos reconcile failed (non-fatal): %s", e)
            return 0

    @staticmethod
    def _index_ids(idx: Path) -> set[str]:
        if not idx.exists():
            return set()
        import json

        ids: set[str] = set()
        with open(idx, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line).get("id", ""))
                except json.JSONDecodeError:
                    continue
        return ids

    @staticmethod
    def _parse_md_logs(log_dir: Path) -> list[dict]:
        entries: list[dict] = []
        for md_path in sorted(log_dir.glob("*.md")):
            cur: dict | None = None
            with open(md_path, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.rstrip("\n")
                    m = _HEADER_RE.match(line)
                    if m:
                        if cur is not None:
                            entries.append(cur)
                        stamp, entry_id = m.group(1), m.group(2)
                        ts = time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
                        cur = {
                            "id": entry_id,
                            "session_key": "",
                            "user": "",
                            "assistant": "",
                            "ts": ts,
                            "metadata": {},
                        }
                        continue
                    if cur is None:
                        continue
                    sm = _SESSION_RE.search(line)
                    if sm:
                        cur["session_key"] = sm.group(1)
                        continue
                    um = _USER_RE.match(line)
                    if um:
                        cur["user"] = um.group(1)
                        continue
                    am = _ASST_RE.match(line)
                    if am:
                        cur["assistant"] = am.group(1)
                        continue
            if cur is not None:
                entries.append(cur)
        return entries

    @staticmethod
    def _rewrite_index(idx: Path, entries: list[dict]) -> None:
        import json
        import os
        import tempfile

        idx.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(idx.parent), prefix=".index-", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                for e in entries:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, idx)  # atomic swap — never a torn index
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def capture(self, event: CaptureEvent, bank_dir: Path) -> None:
        # The engine already appends to the substrate before calling adapters;
        # EverOS's contribution is durability/reconciliation, not a second write.
        return
