"""hindsight-unified — one memory mechanism unifying six systems.

A supervised sidecar that fuses the strengths of six memory projects behind a
single interface for the Hermes agent:

* **Hindsight** — retrieval/storage brain (fact + entity + temporal graph,
  4-strategy RRF recall, cross-encoder rerank, mental models).
* **EverOS** — markdown-as-truth durable substrate + crash-recovery reconciler.
* **mempalace** — verbatim / 100%-recall discipline, AAAK index triage,
  embedder-identity safety.
* **OpenViking** — tiered, token-budgeted context assembly + observable
  retrieval trajectory.
* **MemOS** — MemCube portability (dump/load) for backup & transfer.
* **tencentdb-agent-memory** — the L0→L3 pipeline shape + sidecar/thin-provider
  integration pattern + reliability engineering.

The sidecar itself is dependency-free (stdlib HTTP); each contributor loads
lazily and degrades gracefully, with an always-on verbatim substrate as the
floor.
"""

from __future__ import annotations

from .config import Settings
from .engine import UnifiedEngine

__all__ = ["UnifiedEngine", "Settings"]
__version__ = "0.1.0"
