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

Write safety is layered, because ``threading.Lock`` alone is a lie as soon as a
second process opens the same bank (and the provider's watchdog can resurrect a
sidecar alongside a hung one):

* a **singleton** ``flock`` on the memory root, held for the process lifetime —
  a second sidecar over the same root fails to take it and must exit;
* an exclusive ``flock`` per bank around each append, so even two processes that
  legitimately share a root cannot interleave an entry;
* the in-process ``threading.Lock``, which ``flock`` does *not* replace —
  advisory locks are per-open-file-description and do not serialize threads
  sharing one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # POSIX advisory locking; absent on Windows.
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - platform-dependent
    _HAVE_FCNTL = False

_TOKEN_RE = re.compile(r"[A-Za-z0-9_一-鿿]+")

# Common function words carry no recall signal; matching on them lets an
# unrelated entry tie with a genuinely relevant one in the lean scorer.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "does",
        "did",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _entry_document(entry: SubstrateEntry) -> str:
    """The text an entry is *indexed* by, which is not the text it is shown as.

    ``as_text()`` renders ``"User: ...\\nAssistant: ..."`` for display, and
    indexing that verbatim put our own serialization labels into the term
    space. That was not cosmetic: the conversational rewrite lane builds a
    query containing the literal words ``Prior user:`` and ``Prior assistant:``,
    so *every entry in the bank* matched it on ``user`` and ``assistant``. The
    rewrite lane could therefore never return an empty result, and a question
    memory genuinely cannot answer still came back with a full slate of
    entries — which is the mechanism behind the relevance floor we measured as
    unreachable.

    A field label is part of how we serialize a turn, never part of what the
    user said, so it has no business being searchable.
    """
    return "\n".join(part for part in (entry.user, entry.assistant) if part)


# Okapi BM25 parameters. k1 controls how fast term frequency saturates, b how
# strongly document length is normalised; these are the standard defaults and
# there is no corpus here large enough to justify tuning them.
_BM25_K1 = 1.5
_BM25_B = 0.75


def _corpus_stats(documents: list[list[str]]) -> tuple[dict[str, float], float]:
    """Per-term IDF and mean document length for one bank."""
    total = len(documents)
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for tokens in documents:
        total_length += len(tokens)
        document_frequency.update(set(tokens))
    idf = {
        term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
        for term, freq in document_frequency.items()
    }
    return idf, (total_length / total if total else 0.0)


def _bm25(
    query_terms: set[str], tokens: list[str], *, idf: dict[str, float], avg_len: float
) -> float:
    """BM25 score of one document, summed over the distinct query terms."""
    if not tokens or avg_len <= 0:
        return 0.0
    frequencies = Counter(tokens)
    length_norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * len(tokens) / avg_len)
    score = 0.0
    for term in query_terms:
        tf = frequencies.get(term, 0)
        if tf == 0:
            continue
        score += idf.get(term, 0.0) * (tf * (_BM25_K1 + 1)) / (tf + length_norm)
    return score


def content_hash(user: str, assistant: str) -> str:
    """Stable digest of one turn's exact text.

    The two halves are joined with ``\\x1f`` rather than concatenated: a
    separatorless join makes field boundaries ambiguous, so ("ab", "c") and
    ("a", "bc") would hash identically. (Borrowed from cognee's canonicalization
    fix in ``modules/provenance/integrity.py``, which documents exactly that bug
    in the code it ports from.)
    """
    payload = f"{user}\x1f{assistant}".encode()
    return hashlib.sha256(payload).hexdigest()


def derive_entry_id(*, ts: float, bank: str, session_key: str, digest: str, occurrence: int) -> str:
    """Deterministic, time-sortable entry id.

    The prefix is the zero-padded ms timestamp, so ids sort lexicographically by
    time and give recall a free total recency order. The suffix is derived from
    the turn's content rather than ``os.urandom``, so re-deriving an entry (an
    index rebuild, a replayed capture) reproduces the same id instead of forking
    a duplicate.

    ``occurrence`` is what keeps this honest: two genuinely identical turns in
    one session must stay two entries, so identity is
    ``(bank, session, content, nth occurrence)`` — the part naive content-hash
    dedup gets wrong by collapsing real repetition.
    """
    key = f"SubstrateEntry:{bank}:{session_key}:{digest}:{occurrence}"
    suffix = uuid.uuid5(uuid.NAMESPACE_OID, key).hex[:12]
    return f"{int(ts * 1000):013d}-{suffix}"


@contextmanager
def _flock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Exclusive advisory lock on ``path``, yielding whether it was taken.

    The kernel releases it when the fd closes or the process dies, so there is
    no stale-lock recovery to write. A no-op yielding True where ``fcntl`` is
    unavailable — degrading to the in-process lock is the documented behaviour
    on such platforms, not silent breakage.
    """
    if not _HAVE_FCNTL:
        yield True
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@dataclass(slots=True)
class SubstrateEntry:
    entry_id: str
    session_key: str
    user: str
    assistant: str
    ts: float
    metadata: dict[str, Any]
    digest: str = ""
    superseded_by: str = ""
    last_served_ts: float = 0.0
    # Position in the bank's append-only index. The true recency order:
    # the ms timestamp in entry_id cannot separate entries written inside
    # the same millisecond, which measurably made presentation order vary
    # between runs. Append position is total and deterministic.
    seq: int = 0

    def as_text(self) -> str:
        parts = []
        if self.user:
            parts.append(f"User: {self.user}")
        if self.assistant:
            parts.append(f"Assistant: {self.assistant}")
        return "\n".join(parts)


class SingletonLockHeld(RuntimeError):
    """Another process already owns this memory root."""


class MarkdownSubstrate:
    """Append-only verbatim store for one process (all banks under one root)."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._singleton_fd: int | None = None
        # content digest -> how many entries already carry it, per bank dir.
        # Populated lazily from the index; safe to hold in memory because the
        # singleton lock guarantees this process is the only writer.
        self._occurrences: dict[str, dict[str, int]] = {}
        self._idem_keys: dict[str, dict[str, str]] = {}
        # Index size the caches above were built at. The index is append-only,
        # so a size that no longer matches means somebody else wrote to this
        # bank and our private counts are stale — see _next_occurrence.
        self._index_size: dict[str, int] = {}
        # Locks held by the current thread, so a nested append cannot
        # self-deadlock on a non-reentrant lock (cognee guards the same hazard
        # with a ContextVar in dataset_lock.py:103-112).
        self._held = threading.local()

    # -- singleton ownership -------------------------------------------------

    def acquire_singleton(self) -> bool:
        """Claim exclusive ownership of the memory root for this process.

        Returns False when another live process holds it — the caller must then
        exit rather than write, which is what closes the window the provider's
        watchdog opens when it resurrects a sidecar beside a hung one.
        """
        if not _HAVE_FCNTL:
            return True
        if self._singleton_fd is not None:
            return True
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self._root / ".sidecar.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return True  # cannot lock (read-only fs, odd mount) — do not block startup
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.write(fd, f"{os.getpid()}\n".encode())
        self._singleton_fd = fd
        return True

    def release_singleton(self) -> None:
        fd, self._singleton_fd = self._singleton_fd, None
        if fd is None or not _HAVE_FCNTL:
            return
        # Both are best-effort: closing the fd releases the lock anyway, and a
        # dying process has it released by the kernel.
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

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
        idempotency_key: str = "",
    ) -> str:
        """Append one verbatim turn. Returns the entry id.

        Writes two things atomically-enough for crash safety: a human-readable
        markdown log (truth, append mode) and a machine-readable jsonl index
        line (rebuildable). The jsonl is what search reads; the markdown is what
        a human — or a rebuild — reads.

        ``idempotency_key`` is how a caller gets exactly-once out of an
        at-least-once delivery path: a retried capture (a sync thread that timed
        out on a request the sidecar actually completed) passes the same key and
        gets the original entry id back without a second append. Content alone
        cannot carry this — a user who genuinely says the same thing twice must
        get two entries — so the key has to come from whoever knows the turn's
        identity, and dedup stays a *lookup*, never an identity.
        """
        ts = ts or time.time()
        metadata = metadata or {}
        bank = bank_dir.name
        digest = content_hash(user, assistant)
        log_dir = bank_dir / "log"

        with self._lock, self._bank_lock(bank_dir):
            if idempotency_key:
                existing = self._lookup_idem(bank_dir, idempotency_key)
                if existing:
                    return existing

            occurrence = self._next_occurrence(bank_dir, digest)
            entry_id = derive_entry_id(
                ts=ts,
                bank=bank,
                session_key=session_key,
                digest=digest,
                occurrence=occurrence,
            )

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

            if idempotency_key:
                metadata = {**metadata, "idempotency_key": idempotency_key}
            self._append_index(
                bank_dir,
                SubstrateEntry(
                    entry_id,
                    session_key,
                    user,
                    assistant,
                    ts,
                    metadata,
                    digest=digest,
                ),
            )
            # Keep the cache usable for the next append by advancing both the
            # count and the size stamp it is valid at; a foreign write will move
            # the size to a value we never recorded and force a re-read.
            key = str(bank_dir)
            self._occurrences.setdefault(key, {})[digest] = occurrence + 1
            self._index_size[key] = self._index_bytes(bank_dir)
            if idempotency_key:
                self._idem_keys.setdefault(key, {})[idempotency_key] = entry_id
        return entry_id

    @contextmanager
    def _bank_lock(self, bank_dir: Path) -> Iterator[None]:
        """Cross-process exclusive lock for one bank's append path.

        Re-entrant per thread: a nested append (an adapter that captures while
        the pipeline is mid-capture) must not block on a lock this thread
        already owns.
        """
        held: set[str] = getattr(self._held, "banks", None) or set()
        self._held.banks = held
        key = str(bank_dir)
        if key in held:
            yield
            return
        with _flock(bank_dir / ".append.lock"):
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)

    def _index_bytes(self, bank_dir: Path) -> int:
        try:
            return (bank_dir / "index.jsonl").stat().st_size
        except OSError:
            return -1

    def _next_occurrence(self, bank_dir: Path, digest: str) -> int:
        """How many entries in this bank already carry ``digest``.

        Must be answered from the *index on disk*, not from a counter this
        process has been incrementing privately. Two sidecars appending the same
        text would each believe ``occurrence == 0`` and mint the same entry id
        for two distinct appends — measured, and the per-bank lock alone does not
        prevent it, because it serializes the write while leaving each process
        its own stale view. The index is append-only, so its byte size is a
        sufficient and cheap staleness signal: any foreign write changes it.
        """
        key = str(bank_dir)
        size = self._index_bytes(bank_dir)
        if self._occurrences.get(key) is None or self._index_size.get(key) != size:
            cache: dict[str, int] = {}
            for entry in self._load(bank_dir):
                digest_key = entry.digest or content_hash(entry.user, entry.assistant)
                cache[digest_key] = cache.get(digest_key, 0) + 1
            self._occurrences[key] = cache
            self._index_size[key] = size
        return self._occurrences[key].get(digest, 0)

    def _lookup_idem(self, bank_dir: Path, key: str) -> str:
        """Entry id previously written under this idempotency key, if any."""
        cache = self._idem_keys.get(str(bank_dir))
        if cache is None:
            cache = {}
            for entry in self._load(bank_dir):
                stored = (entry.metadata or {}).get("idempotency_key")
                if stored:
                    cache.setdefault(str(stored), entry.entry_id)
            self._idem_keys[str(bank_dir)] = cache
        return cache.get(key, "")

    def forget_caches(self, bank_dir: Path | None = None) -> None:
        """Drop the occurrence/idempotency caches (after an index rebuild)."""
        if bank_dir is None:
            self._occurrences.clear()
            self._idem_keys.clear()
            self._index_size.clear()
            return
        self._occurrences.pop(str(bank_dir), None)
        self._idem_keys.pop(str(bank_dir), None)
        self._index_size.pop(str(bank_dir), None)

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
            # Carried as a field, not folded into the id, so dedup stays a
            # lookup and re-enrichment of an entry updates in place.
            "content_hash": entry.digest,
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
            for position, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn/partial line — skip, md log is authoritative
                meta = r.get("metadata") or {}
                out.append(
                    SubstrateEntry(
                        entry_id=r.get("id", ""),
                        session_key=r.get("session_key", ""),
                        user=r.get("user", ""),
                        assistant=r.get("assistant", ""),
                        ts=float(r.get("ts", 0.0)),
                        metadata=meta,
                        # Tolerant reader: fields added after a bank was written
                        # default cleanly, so an existing index stays readable
                        # (cognee's DataItemStatus.py:5-22 discipline).
                        digest=str(r.get("content_hash") or ""),
                        superseded_by=str(meta.get("superseded_by") or ""),
                        last_served_ts=float(meta.get("last_served_ts") or 0.0),
                        seq=position,
                    )
                )
        return out

    def recent_turns(
        self, bank_dir: Path, *, session_key: str = "", limit: int = 2
    ) -> list[tuple[str, str]]:
        """The last ``limit`` turns, oldest-first, as ``(user, assistant)``.

        Recall needs conversation history to expand an anaphoric question, and
        the substrate already holds it — so the history costs one read and no
        new API surface for the client to pass it in.
        """
        entries = self._load(bank_dir)
        if session_key:
            entries = [e for e in entries if e.session_key == session_key]
        return [(e.user, e.assistant) for e in entries[-limit:]]

    def count(self, bank_dir: Path) -> int:
        return len(self._load(bank_dir))

    def search(
        self, bank_dir: Path, query: str, limit: int = 8, session_key: str = ""
    ) -> list[tuple[SubstrateEntry, float]]:
        """Okapi BM25 over verbatim entries. No model, no network.

        Replaces a plain term-overlap score, which had neither of the two things
        that make lexical retrieval work: **IDF**, so a rare term counts for more
        than a common one, and **length normalisation** that saturates, so a long
        entry cannot win by sheer term count. Corpus statistics come from the
        bank being searched — the entries are loaded for scoring anyway, so this
        costs one extra pass and no persistence.

        The point of this layer remains guaranteed availability and exactness,
        not semantic nuance; that is what the Hindsight brain adds on top.
        """
        q_tokens = _tokenize(query)
        q_set = {t for t in q_tokens if t not in _STOPWORDS}
        if not q_set:  # all-stopword query: fall back to raw tokens
            q_set = set(q_tokens)
        if not q_set:
            return []

        entries = [
            e for e in self._load(bank_dir) if not session_key or e.session_key == session_key
        ]
        docs = [(e, _tokenize(_entry_document(e))) for e in entries]
        docs = [(e, tokens) for e, tokens in docs if tokens]
        if not docs:
            return []

        idf, avg_len = _corpus_stats([tokens for _e, tokens in docs])

        scored: list[tuple[SubstrateEntry, float]] = []
        for entry, tokens in docs:
            score = _bm25(q_set, tokens, idf=idf, avg_len=avg_len)
            if score <= 0.0:
                continue
            scored.append((entry, score))
        # Recency breaks ties only; relevance still decides selection.
        scored.sort(key=lambda pair: (pair[1], pair[0].ts), reverse=True)
        return scored[:limit]
