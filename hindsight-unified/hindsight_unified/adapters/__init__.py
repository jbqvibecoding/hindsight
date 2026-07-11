"""Contributor adapters — one per integrated repo."""

from __future__ import annotations

from .base import UnifiedAdapter
from .everos_substrate import EverosSubstrateAdapter
from .hindsight_core import HindsightCoreAdapter
from .memos_portability import MemosPortabilityAdapter
from .mempalace_index import MempalaceIndexAdapter
from .openviking_injection import OpenVikingInjectionAdapter

__all__ = [
    "UnifiedAdapter",
    "HindsightCoreAdapter",
    "EverosSubstrateAdapter",
    "MempalaceIndexAdapter",
    "OpenVikingInjectionAdapter",
    "MemosPortabilityAdapter",
]
