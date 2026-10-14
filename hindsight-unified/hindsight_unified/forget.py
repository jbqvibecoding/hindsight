"""Forgetting, as a recall filter over a log that is never rewritten.

The design is supermemory's documented one, and deliberately not the shape of
its own Hermes plugin. Its docs specify a **soft delete** — a forgotten memory
"will no longer appear in search results but remains in the database" — with a
recorded reason, and for anything matched by a query, a **preview first**:
"Always dryRun first to review what would be forgotten, then re-run" with the
exact ids, because "the match is semantic, so a too-broad query can select more
than you intend". Its Hermes plugin instead searches, takes the top hit, and
forgets it unseen; asked to forget an old address, it forgets whichever address
ranked first. Here a query can only ever *preview*; forgetting takes ids.

A tombstone is user intent, not a derivative. Triggers, lessons and the index
are all rebuildable from the markdown and go on ``rederive``; this file is not,
and rederive leaves it alone. It survives because entry ids do: the markdown
header carries each entry's id, so a rebuild reproduces them exactly.

Two honest limits, both reported in the API response rather than left implicit:

* The verbatim text **stays on disk** in the markdown log. Forgetting removes a
  memory from what the agent can recall; it is not erasure, and must never be
  described to a user as erasure.
* The semantic brain, when installed, keeps its own extracted facts in its own
  store, which is keyed by its own ids. Forgetting here does not reach it.

The log is append-only and the latest action per entry wins, so a forget can be
undone with ``restore`` — the reason soft deletion is safe to offer at all.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

FORGET_FILE = "forgotten.jsonl"

_FORGET = "forget"
_RESTORE = "restore"


def path(bank_dir: Path) -> Path:
    return bank_dir / FORGET_FILE


def forgotten_ids(bank_dir: Path) -> frozenset[str]:
    """Entry ids currently forgotten. A torn line is skipped, as in the index."""
    tombstones = path(bank_dir)
    if not tombstones.exists():
        return frozenset()
    state: dict[str, str] = {}
    with open(tombstones, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_id = str(row.get("entry_id") or "")
            action = row.get("action")
            if entry_id and action in (_FORGET, _RESTORE):
                state[entry_id] = action
    return frozenset(entry_id for entry_id, action in state.items() if action == _FORGET)


def new_batch_id() -> str:
    """One id per forget call, stamped on every row it writes, so a bulk
    forget can be traced and reviewed as the unit it was issued as."""
    return uuid.uuid4().hex


def record(
    bank_dir: Path,
    entry_ids: Iterable[str],
    *,
    action: str,
    reason: str = "",
    batch_id: str = "",
) -> None:
    """Append one row per entry. Durable before it returns.

    fsync matters more here than for the derived sidecars: losing a tombstone
    in a crash would bring a memory back that the user asked to be forgotten.
    """
    if action not in (_FORGET, _RESTORE):
        raise ValueError(f"unknown forget action: {action!r}")
    ids = [entry_id for entry_id in entry_ids if entry_id]
    if not ids:
        return
    tombstones = path(bank_dir)
    tombstones.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with open(tombstones, "a", encoding="utf-8") as fh:
        for entry_id in ids:
            row = {
                "entry_id": entry_id,
                "action": action,
                "reason": reason,
                "batch_id": batch_id,
                "ts": now,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def forget(bank_dir: Path, entry_ids: Iterable[str], *, reason: str = "") -> str:
    """Tombstone these entries. Returns the batch id."""
    batch_id = new_batch_id()
    record(bank_dir, entry_ids, action=_FORGET, reason=reason, batch_id=batch_id)
    return batch_id


def restore(bank_dir: Path, entry_ids: Iterable[str]) -> str:
    """Undo a forget. Returns the batch id."""
    batch_id = new_batch_id()
    record(bank_dir, entry_ids, action=_RESTORE, batch_id=batch_id)
    return batch_id
