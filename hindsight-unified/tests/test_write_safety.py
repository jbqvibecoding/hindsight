"""Write-safety and identity tests (workstream B).

These cover the guarantees the substrate makes about *not losing or corrupting*
verbatim history: cross-process append safety, exclusive root ownership, and
deterministic content-derived identity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from hindsight_unified.substrate import (
    MarkdownSubstrate,
    content_hash,
    derive_entry_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- identity ------------------------------------------------------------------


def test_content_hash_is_field_separated() -> None:
    # A separatorless join would make these two turns hash identically.
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_entry_id_is_deterministic_and_time_sortable(tmp_path: Path) -> None:
    args = dict(bank="b", session_key="s", digest="d" * 64, occurrence=0)
    assert derive_entry_id(ts=1_000.0, **args) == derive_entry_id(ts=1_000.0, **args)
    early = derive_entry_id(ts=1_000.0, **args)
    later = derive_entry_id(ts=2_000.0, **args)
    assert early < later  # lexicographic order == chronological order


def test_occurrence_keeps_identical_turns_distinct(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    first = sub.append(bank, session_key="s", user="ping", assistant="pong")
    second = sub.append(bank, session_key="s", user="ping", assistant="pong")
    # Real repetition must survive: naive content-hash dedup would collapse it.
    assert first != second
    assert sub.count(bank) == 2


def test_idempotency_key_makes_a_retry_a_noop(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    first = sub.append(
        bank, session_key="s", user="hello", assistant="hi", idempotency_key="turn-7"
    )
    again = sub.append(
        bank, session_key="s", user="hello", assistant="hi", idempotency_key="turn-7"
    )
    assert again == first
    assert sub.count(bank) == 1


def test_idempotency_key_survives_a_fresh_process_view(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    first = MarkdownSubstrate(tmp_path).append(
        bank, session_key="s", user="hello", assistant="hi", idempotency_key="turn-7"
    )
    # A new instance rebuilds the key cache from the index, not from memory.
    again = MarkdownSubstrate(tmp_path).append(
        bank, session_key="s", user="hello", assistant="hi", idempotency_key="turn-7"
    )
    assert again == first


def test_occurrence_is_re_read_when_another_writer_touched_the_index(
    tmp_path: Path,
) -> None:
    """A stale private occurrence count must never mint a duplicate id.

    This is the deterministic form of the cross-process defect. ``first`` holds
    an in-memory count from its own append; ``other`` (standing in for a second
    sidecar) then appends the same text behind its back. ``first``'s next append
    must notice the index moved and re-read the count from disk, or it reissues
    an id that already exists.

    The timestamp is pinned so the id's time prefix cannot mask the defect.
    That prefix is real collision resistance — two appends in different
    milliseconds differ regardless of occurrence — which is why the concurrent
    two-process test only caught this in a tight same-millisecond loop. Here
    ``occurrence`` is the only thing that can distinguish the ids, so the
    mechanism is tested rather than the clock.
    """
    bank = tmp_path / "bank"
    first = MarkdownSubstrate(tmp_path)
    other = MarkdownSubstrate(tmp_path)
    turn = dict(session_key="s", user="same text", assistant="same reply", ts=1_700_000.0)

    id_a = first.append(bank, **turn)
    id_b = other.append(bank, **turn)  # a second sidecar, behind first's back
    id_c = first.append(bank, **turn)  # first must notice the index moved

    assert len({id_a, id_b, id_c}) == 3, "occurrence read from a stale cache"
    assert len(_index_ids(bank)) == 3


def test_content_hash_is_persisted_for_lookup(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    sub.append(bank, session_key="s", user="hello", assistant="hi")
    record = json.loads((bank / "index.jsonl").read_text(encoding="utf-8").strip())
    assert record["content_hash"] == content_hash("hello", "hi")


# -- cross-process write safety ------------------------------------------------

_WRITER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from hindsight_unified.substrate import MarkdownSubstrate

    root, mode, marker, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sub = MarkdownSubstrate(Path(root))
    bank = Path(root) / "bank"
    if mode == "identical":
        # Same text from both writers: the case where two processes each compute
        # occurrence == 0 and would mint the same entry id.
        user, assistant = "same text", "same reply"
    else:
        # Well past PIPE_BUF, to exercise large-append behaviour.
        body = (marker * 40 + " ") * 400
        user, assistant = f"{marker} {body}", body
    time.sleep(0.3)  # line both processes up on roughly the same instant
    for _ in range(count):
        sub.append(bank, session_key="s", user=user, assistant=assistant)
    """
)


def _run_writer(root: Path, mode: str, marker: str, count: int) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.Popen(
        [sys.executable, "-c", _WRITER, str(root), mode, marker, str(count)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _await(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        _, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err.decode()


def _index_ids(bank: Path) -> list[str]:
    lines = (bank / "index.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["id"] for line in lines if line.strip()]


def test_concurrent_identical_appends_never_collide_on_an_id(tmp_path: Path) -> None:
    """Two processes writing the same text must not mint the same entry id.

    An integration check over the real two-process path. It is deliberately
    *not* the regression guard for the stale-count bug — that race needs both
    writers to read the index before either's first write lands, which does not
    reproduce reliably (measured: 2 duplicates unlocked, 1 with the per-bank
    lock alone, 0 on a later run of the same buggy build). The deterministic
    guard is ``test_occurrence_is_re_read_when_another_writer_touched_the_index``.
    """
    _await(
        [
            _run_writer(tmp_path, "identical", "A", 5),
            _run_writer(tmp_path, "identical", "B", 5),
        ]
    )
    ids = _index_ids(tmp_path / "bank")
    assert len(ids) == 10
    assert len(set(ids)) == 10, f"{len(ids) - len(set(ids))} colliding entry id(s)"


def test_concurrent_large_appends_keep_entries_intact(tmp_path: Path) -> None:
    """Large concurrent appends leave every entry individually parseable.

    A portability guard, not a reproduction: Linux holds the inode lock for the
    duration of a ``write(2)``, so large ``O_APPEND`` writes to a local regular
    file do not interleave in practice (measured: clean without any locking).
    POSIX only promises this below ``PIPE_BUF`` for pipes, and NFS does not
    promise it at all — which is what the per-append lock is for.
    """
    _await(
        [
            _run_writer(tmp_path, "large", "A", 12),
            _run_writer(tmp_path, "large", "B", 12),
        ]
    )
    bank = tmp_path / "bank"
    entries = MarkdownSubstrate(tmp_path)._load(bank)
    assert len(entries) == 24

    # No entry may carry the other writer's bytes: that is a torn append.
    for entry in entries:
        marker = entry.user[:1]
        other = "B" if marker == "A" else "A"
        assert other * 40 not in entry.user
        assert other * 40 not in entry.assistant

    # The markdown log — the truth — must parse back to the same count.
    md = "".join(path.read_text(encoding="utf-8") for path in sorted((bank / "log").glob("*.md")))
    assert md.count("### ") == 24


def test_singleton_lock_excludes_a_second_owner(tmp_path: Path) -> None:
    first = MarkdownSubstrate(tmp_path)
    assert first.acquire_singleton() is True
    second = MarkdownSubstrate(tmp_path)
    # Same process, different fd — flock is per open-file-description, so this
    # models the two-sidecar case the provider's watchdog can create.
    assert second.acquire_singleton() is False
    first.release_singleton()
    assert second.acquire_singleton() is True
    second.release_singleton()


def test_bank_lock_is_reentrant_within_a_thread(tmp_path: Path) -> None:
    sub = MarkdownSubstrate(tmp_path)
    bank = tmp_path / "bank"
    # Deliberately nested rather than combined into one `with`: the nesting IS
    # what is under test, since a nested append must not deadlock on a lock
    # this thread already holds.
    with sub._bank_lock(bank):  # noqa: SIM117
        with sub._bank_lock(bank):
            pass
    assert sub.append(bank, session_key="s", user="x", assistant="y")
