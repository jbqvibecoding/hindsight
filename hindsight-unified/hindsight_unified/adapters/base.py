"""Uniform adapter contract.

Each contributor repo is wrapped in one adapter under this package. The
``UnifiedEngine`` composes them; every method has a safe default so an adapter
only implements the layer it owns and stays a no-op elsewhere. Adapters must
never raise out of these methods — degrade and log instead. That is what makes
"all six wired from the start" survivable when a heavy dependency is missing on
a given host.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..types import CaptureEvent, Recalled, RecallRequest

logger = logging.getLogger(__name__)


class UnifiedAdapter(ABC):
    """Base class for a contributor adapter.

    Only ``name`` is abstract. Every hook below is optional with a no-op or
    empty default, which is what lets an adapter contribute to one phase and
    ignore the rest — and what lets a missing engine degrade to the verbatim
    substrate instead of failing. So B027 ("empty method in an ABC, but not
    abstract") is describing the intended design, not a defect.
    """

    # ruff: noqa: B027

    @property
    @abstractmethod
    def name(self) -> str:
        """Short adapter id (matches the config enable-flag suffix)."""

    def available(self) -> bool:
        """Whether this adapter is loaded and ready. Default: False."""
        return False

    def start(self) -> None:
        """Attempt to load/connect the underlying engine. Must not raise."""

    def stop(self) -> None:
        """Release resources. Must not raise."""

    # -- L0/L1 write ---------------------------------------------------------

    def capture(self, event: CaptureEvent, bank_dir: Path) -> None:
        """Ingest a turn (verbatim store, fact extraction, ...). Default no-op."""

    # -- retrieval -----------------------------------------------------------

    def recall_enrich(self, req: RecallRequest, bank_dir: Path) -> list[Recalled]:
        """Contribute recalled items for the query. Default: nothing."""
        return []

    # -- L2/L3 consolidate ---------------------------------------------------

    def consolidate(self, bank: str, bank_dir: Path) -> None:
        """Session-end / background consolidation. Default no-op."""

    # -- portability ---------------------------------------------------------

    def export_paths(self, bank: str, bank_dir: Path) -> list[str]:
        """On-disk paths to include in a backup. Default: none."""
        return []
