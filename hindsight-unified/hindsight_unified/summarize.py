"""Retrieval-oriented summaries as a second lane over the same verbatim store.

Verbatim text and a summary of it fail differently: verbatim matches exact
phrasing, a summary matches abstracted intent. So index both, search both, and
**always return the verbatim entry** — the summary only ever votes. That keeps
the promise this system is built on: markdown is truth, the summary is a
rebuildable derivative that affects ranking and never content.

This is the layer that addresses the one gap the deterministic work cannot.
Measured: asked "what cleans up storage we no longer need?", nothing retrieves
"the nightly job that trims old blobs is called reaper", because the two share
no term and no expansion derivable *from the entry text* can invent the
synonym. Cognee's summary prompt is kept near-verbatim because it summarises
**for retrieval, not for a human**: its first section is a category→names list,
which is keyword expansion by hand, and its second section is self-contained
facts, which survive being packed into a token budget.

Summary ids derive from the entry id, so there is no join table and the whole
sidecar is rebuildable — the same property `rederive` relies on.

Requires an LLM. With none configured, generation is skipped and the lane is
simply empty; nothing here degrades recall below the substrate-only behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMClient
from .substrate import SubstrateEntry, _bm25, _corpus_stats, _tokenize

logger = logging.getLogger(__name__)

_PROMPT = Path(__file__).parent / "prompts" / "summarize_content.txt"
SUMMARY_FILE = "summaries.jsonl"
# The prompt caps itself at 200 tokens; leave room and no more, so a summary
# stays budget-plannable rather than quietly becoming a second copy of the turn.
MAX_SUMMARY_TOKENS = 400


def summary_id(entry_id: str) -> str:
    """Derived, not stored — given an entry you can always recompute this."""
    return uuid.uuid5(uuid.NAMESPACE_OID, f"Summary:{entry_id}").hex


@dataclass(slots=True)
class EntrySummary:
    entry_id: str
    text: str
    # The content hash of the entry this was derived from. A mismatch means the
    # summary is stale rather than missing, which are different problems.
    source_digest: str = ""

    @property
    def summary_id(self) -> str:
        return summary_id(self.entry_id)


class SummaryStore:
    """Append-only sidecar index of derived summaries, one file per bank."""

    def __init__(self) -> None:
        self._prompt = _PROMPT.read_text(encoding="utf-8") if _PROMPT.exists() else ""

    # -- storage ------------------------------------------------------------

    @staticmethod
    def path(bank_dir: Path) -> Path:
        return bank_dir / SUMMARY_FILE

    def load(self, bank_dir: Path) -> dict[str, EntrySummary]:
        """Summaries by entry id. A torn line is skipped, as in the index."""
        path = self.path(bank_dir)
        if not path.exists():
            return {}
        out: dict[str, EntrySummary] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_id = str(record.get("entry_id") or "")
                if not entry_id:
                    continue
                out[entry_id] = EntrySummary(
                    entry_id=entry_id,
                    text=str(record.get("text") or ""),
                    source_digest=str(record.get("source_digest") or ""),
                )
        return out

    def append(self, bank_dir: Path, summaries: list[EntrySummary]) -> None:
        if not summaries:
            return
        path = self.path(bank_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for summary in summaries:
                fh.write(
                    json.dumps(
                        {
                            "summary_id": summary.summary_id,
                            "entry_id": summary.entry_id,
                            "text": summary.text,
                            "source_digest": summary.source_digest,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())

    def rewrite(self, bank_dir: Path, summaries: list[EntrySummary]) -> None:
        """Replace the sidecar atomically (used when re-deriving)."""
        path = self.path(bank_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{SUMMARY_FILE}-", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                for summary in summaries:
                    fh.write(
                        json.dumps(
                            {
                                "summary_id": summary.summary_id,
                                "entry_id": summary.entry_id,
                                "text": summary.text,
                                "source_digest": summary.source_digest,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- generation ---------------------------------------------------------

    def generate_missing(
        self,
        bank_dir: Path,
        entries: list[SubstrateEntry],
        client: LLMClient,
        *,
        limit: int = 50,
    ) -> int | None:
        """Summarise entries that have no current summary.

        Returns the number written, ``0`` when there was nothing to do, and
        **``None`` when the LLM was unreachable** — the same distinction
        consolidation relies on, so an outage never looks like "nothing to
        summarise" and never advances a watermark over work that did not run.
        """
        if not client.available() or not self._prompt:
            return 0

        existing = self.load(bank_dir)
        pending = [
            entry
            for entry in entries
            if entry.entry_id
            and (
                entry.entry_id not in existing
                # A digest mismatch means stale, not missing. Re-deriving is
                # correct; leaving it would rank against text that changed.
                or (entry.digest and existing[entry.entry_id].source_digest != entry.digest)
            )
        ][:limit]
        if not pending:
            return 0

        written: list[EntrySummary] = []
        failed = False
        for entry in pending:
            text = client.complete(
                system=self._prompt,
                user=entry.as_text(),
                max_tokens=MAX_SUMMARY_TOKENS,
            )
            if text is None:
                # Stop on the first outage rather than hammering a dead
                # endpoint for every remaining entry.
                failed = True
                break
            if text.strip():
                written.append(
                    EntrySummary(
                        entry_id=entry.entry_id, text=text.strip(), source_digest=entry.digest
                    )
                )

        # Whatever succeeded is published — partial progress is real progress,
        # and the next run picks up the rest because pending is recomputed.
        self.append(bank_dir, written)
        if failed:
            return None
        return len(written)

    # -- retrieval ----------------------------------------------------------

    def search(
        self, bank_dir: Path, query: str, *, limit: int = 8
    ) -> list[tuple[str, float]]:
        """BM25 over summary text. Returns ``(entry_id, score)`` — never text.

        Returning ids rather than summaries is the whole discipline: the caller
        dereferences to the verbatim entry, so a summary can change what ranks
        but can never change what is shown.
        """
        summaries = list(self.load(bank_dir).values())
        if not summaries:
            return []
        q_tokens = {t for t in _tokenize(query)}
        if not q_tokens:
            return []

        docs = [(s, _tokenize(s.text)) for s in summaries]
        docs = [(s, tokens) for s, tokens in docs if tokens]
        if not docs:
            return []
        idf, avg_len = _corpus_stats([tokens for _s, tokens in docs])

        scored = [
            (s.entry_id, _bm25(q_tokens, tokens, idf=idf, avg_len=avg_len))
            for s, tokens in docs
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]
