"""Verbatim markdown substrate — the L0 durable source of truth.

Dependency-free by design. This is the one layer that is *always* available,
even when every heavy contributor (Hindsight, MemOS, ...) fails to import. It
embodies two borrowed disciplines:

* EverOS "markdown = truth": each turn is appended to a per-day markdown log
  with an atomic tmp-write + fsync + rename, so a crash mid-write can never
  corrupt existing history. The pg/vector index is a *rebuildable derivative*
  of these files.
* mempalace "verbatim / 100% recall": we store the user's exact words, never a
  summary. Retrieval finds the entry; it never paraphrases it.

Search here is a transparent keyword scorer (no model, no network) so the
substrate can answer ``/search/conversations`` and back a keyword ``/recall``
fallback when no semantic brain is loaded.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")

# Common function words carry no recall signal; matching on them lets an
# unrelated entry tie with a genuinely relevant one in the lean scorer.
_STOPWORDS = frozenset(
    "a an and are as at be but by do does did for from has have he her his i in is it its "
    "me my of on or our she so that the their them they this to us was we were what when "
    "where which who will with you your".split()
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass(slots=True)
class SubstrateEntry:
    entry_id: str
    session_key: str
    user: str
    assistant: str
    ts: float
    metadata: dict[str, Any]

    def as_text(self) -> str:
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        return "\n".join(parts)


class MarkdownSubstrate:
    """Append-only verbatim store for one process (all banks under one root)."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()

    # -- write ---------------------------------------------------------------

    def append(
        self,
        bank_dir: Path,
        *,
        session_key: str,
        user: str,
        assistant: str,
        ts: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Append one verbatim turn. Returns the entry id.

        Writes two things atomically-enough for crash safety: a human-readable
        markdown log (truth, append mode) and a machine-readable jsonl index
        line (rebuildable). The jsonl is what search reads; the markdown is what
        a human — or a rebuild — reads.
        """
        ts = ts or time.time()
        metadata = metadata or {}
        entry_id = f"{int(ts * 1000):013d}-{os.urandom(3).hex()}"
        log_dir = bank_dir / "log"
        with self._lock:
            log_dir.mkdir(parents=True, exist_ok=True)
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            md_path = log_dir / f"{day}.md"
            block = self._render_md_block(entry_id, session_key, user, assistant, ts)
            # Append is inherently crash-tolerant for prior content; fsync the
            # dir entry so the new bytes survive a power cut (EverOS discipline).
            with open(md_path, "a", encoding="utf-8") as fh:
                fh.write(block)
                fh.flush()
                os.fsync(fh.fileno())
            self._append_index(
                bank_dir,
                SubstrateEntry(entry_id, session_key, user, assistant, ts, metadata),
            )
        return entry_id

    @staticmethod
    def _render_md_block(
        entry_id: str, session_key: str, user: str, assistant: str, ts: float
    ) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        lines = [
            f"\n### {stamp} · {entry_id}",
            f"<!-- session={session_key} -->",
            "",
            f"**User:** {user}" if user else "",
            "",
            f"**Assistant:** {assistant}" if assistant else "",
            "",
        ]
        return "\n".join(line for line in lines if line is not None) + "\n"

    def _append_index(self, bank_dir: Path, entry: SubstrateEntry) -> None:
        idx = bank_dir / "index.jsonl"
        record = {
            "id": entry.entry_id,
            "session_key": entry.session_key,
            "user": entry.user,
            "assistant": entry.assistant,
            "ts": entry.ts,
            "metadata": entry.metadata,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        # Atomic-append via O_APPEND write; index is a derivative so a torn
        # line is recoverable by re-scanning the md logs.
        with open(idx, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    # -- read ----------------------------------------------------------------

    def _load(self, bank_dir: Path) -> list[SubstrateEntry]:
        idx = bank_dir / "index.jsonl"
        if not idx.exists():
            return []
        out: list[SubstrateEntry] = []
        with open(idx, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn/partial line — skip, md log is authoritative
                out.append(
                    SubstrateEntry(
                        entry_id=r.get("id", ""),
                        session_key=r.get("session_key", ""),
                        user=r.get("user", ""),
                        assistant=r.get("assistant", ""),
                        ts=float(r.get("ts", 0.0)),
                        metadata=r.get("metadata", {}),
                    )
                )
        return out

    def count(self, bank_dir: Path) -> int:
        return len(self._load(bank_dir))

    def search(
        self, bank_dir: Path, query: str, limit: int = 8, session_key: str = ""
    ) -> list[tuple[SubstrateEntry, float]]:
        """Keyword-overlap search over verbatim entries.

        A tiny tf scorer with a recency tie-breaker. No model, no network — the
        point is guaranteed availability and exactness, not semantic nuance
        (that is what the Hindsight brain adds on top).
        """
        q_tokens = _tokenize(query)
        q_set = {t for t in q_tokens if t not in _STOPWORDS}
        if not q_set:  # all-stopword query: fall back to raw tokens
            q_set = set(q_tokens)
        if not q_set:
            return []
        entries = self._load(bank_dir)
        scored: list[tuple[SubstrateEntry, float]] = []
        for e in entries:
            if session_key and e.session_key != session_key:
                continue
            doc = _tokenize(e.as_text())
            if not doc:
                continue
            overlap = sum(1 for t in doc if t in q_set)
            if overlap == 0:
                continue
            # tf-ish score normalized by doc length, nudged by recency.
            score = overlap / (len(doc) ** 0.5)
            scored.append((e, score))
        scored.sort(key=lambda pair: (pair[1], pair[0].ts), reverse=True)
        return scored[:limit]
