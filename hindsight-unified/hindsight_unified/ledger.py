"""Per-call cost ledger. Off unless asked for, and never in the way.

Consolidation is the only thing here that spends money, and it spent it
invisibly: a run reported which stages did work, never what they cost. T-Mem
keeps a ledger for the same reason and states the principle worth copying —
log at the **finest granularity available**, because anything coarser can be
re-aggregated from these rows offline while the reverse is impossible. So one
row per successful call, never per stage or per run.

Three properties make it safe to leave in the code path:

* **Off by default.** No ``UNIFIED_MEMORY_COST_LOG`` path, no work at all.
* **Every exception is swallowed.** Accounting must never be able to fail a
  memory write. A ledger that can take down the thing it measures is worse
  than no ledger.
* **Missing is recorded as missing.** Token counts come from the provider's
  own ``usage`` when it returns one and are ``null`` when it does not. There
  is no local estimator and no silent zero — a fabricated token count is
  indistinguishable from a real one once it is in the file, and this codebase
  has spent its whole life keeping "absent" apart from "zero".
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

ENV_PATH = "UNIFIED_MEMORY_COST_LOG"

# Appends are single-line and short, so O_APPEND plus this lock is enough to
# keep rows intact; the substrate's flock machinery would be overkill for a
# file nothing reads back during a run.
_WRITE_LOCK = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get(ENV_PATH, "").strip())


def record(
    *,
    call_site: str,
    model: str,
    latency_s: float,
    prompt_chars: int,
    completion_chars: int,
    usage: dict[str, Any] | None = None,
) -> None:
    """Append one row for a successful completion. Silent when disabled.

    ``usage`` is the provider's own block if it returned one. Its fields are
    copied through as-is, and left ``null`` when absent.
    """
    path_text = os.environ.get(ENV_PATH, "").strip()
    if not path_text:
        return
    try:
        row = {
            "ts": time.time(),
            "pid": os.getpid(),
            "call_site": call_site,
            "model": model,
            "latency_s": round(latency_s, 4),
            "prompt_chars": prompt_chars,
            "completion_chars": completion_chars,
            "prompt_tokens": _usage_field(usage, "input_tokens"),
            "completion_tokens": _usage_field(usage, "output_tokens"),
            "token_source": "api" if usage else None,
        }
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _WRITE_LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:  # noqa: BLE001 - accounting must never break a memory write
        return


def _usage_field(usage: dict[str, Any] | None, name: str) -> int | None:
    if not usage:
        return None
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
