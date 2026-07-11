"""Shared data types for the unified memory sidecar.

Small, dependency-free dataclasses passed between the engine, the adapters,
and the HTTP layer. Keeping them here (rather than importing any contributor's
models) is what lets an adapter degrade to a no-op without dragging its
package's imports into the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CaptureEvent:
    """One conversation turn to persist.

    ``bank`` is the tenant/namespace key (Hindsight ``bank_id``). ``user`` and
    ``assistant`` are the verbatim turn halves — the L0 layer stores them
    exactly, never summarized.
    """

    bank: str
    session_key: str
    user: str
    assistant: str
    user_id: str = ""
    ts: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Recalled:
    """A single recalled item contributed by an adapter.

    ``score`` is adapter-local and only meaningful for that adapter's own
    ranking; cross-adapter merging uses Reciprocal Rank Fusion on *rank*, so
    heterogeneous score scales never need normalizing.
    """

    text: str
    source: str  # adapter name that produced it, e.g. "hindsight" / "substrate"
    score: float = 0.0
    fact_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecallRequest:
    bank: str
    query: str
    session_key: str = ""
    user_id: str = ""
    limit: int = 8
    fact_type: str = ""
    question_date: str = ""  # ISO-8601; enables mempalace as-of temporal filtering
