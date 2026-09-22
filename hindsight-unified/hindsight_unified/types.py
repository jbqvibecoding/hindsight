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
class RecallMarker:
    """Why a recall returned what it did — carried even when results are empty.

    "Nothing matched" and "the layer that would have matched is offline" must
    not be the same value. Without this an L0-only degraded recall is
    indistinguishable from a healthy empty memory, so neither the agent nor the
    user can tell that an answer is missing rather than absent.

    ``status`` is ``ok`` when every enabled layer answered, ``degraded`` when a
    layer was skipped (typically the semantic brain), and ``unavailable`` when
    nothing could be consulted at all.
    """

    status: str = "ok"
    reason: str = ""
    text: str = ""
    brain: bool = False
    num_results: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "text": self.text,
            "brain": self.brain,
            "num_results": self.num_results,
        }


@dataclass(slots=True)
class RecallRequest:
    bank: str
    query: str
    session_key: str = ""
    user_id: str = ""
    limit: int = 8
    fact_type: str = ""
    question_date: str = ""  # ISO-8601; enables mempalace as-of temporal filtering
