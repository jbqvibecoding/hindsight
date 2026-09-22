"""OpenViking adapter — tiered context assembly & observable trajectory.

OpenViking rejects flat vector dumps in favour of a *filesystem-as-context*
model with tiered L0/L1/L2 loading (load only what the turn needs, to save
tokens) and an observable retrieval trajectory (you can see why each item was
chosen). We adopt the two ideas that matter for injection:

* **Tiered assembly**: given the fused recall set, lay it out in tiers within a
  token budget — L1 facts (dense, cheap) first, then L2 verbatim excerpts
  (expensive) only while budget remains. This is the injection-time counterpart
  to OpenViking's on-demand directory loading.
* **Trajectory**: emit a machine-readable list of what was included, its source
  adapter, and why, so recall is debuggable.

Falls back to a pure-Python assembler when the ``openviking`` SDK is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..types import Recalled
from .base import UnifiedAdapter

logger = logging.getLogger(__name__)

# Rough token estimate: ~4 chars/token. Good enough for budgeting; the brain
# already enforces its own precise token filter upstream.
_CHARS_PER_TOKEN = 4


class OpenVikingInjectionAdapter(UnifiedAdapter):
    def __init__(self) -> None:
        self._available = False
        self._have_sdk = False

    @property
    def name(self) -> str:
        return "openviking"

    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        self._available = True  # pure-Python tiering always works
        try:
            import openviking  # type: ignore  # noqa: F401

            self._have_sdk = True
            logger.info("openviking SDK present; tiered assembly active (enriched)")
        except Exception:  # noqa: BLE001
            logger.info("openviking SDK absent; tiered assembly active (lean mode)")

    def assemble(
        self,
        recalled: list[Recalled],
        *,
        aaak_header: str = "",
        max_tokens: int = 1500,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Assemble a token-budgeted, tiered context block + trajectory.

        Returns ``(context_text, trajectory)``. The trajectory is a list of
        ``{index, source, fact_type, tier, tokens}`` dicts describing exactly
        what made it into the payload and why it stopped.

        Relevance decides **selection**; recency decides **presentation**.
        Within each tier the newest entry leads and the header states the
        tie-break rule, so a superseded fact that is still retrievable cannot
        be read as current. This is the cheapest half of conflict resolution:
        no extraction, no LLM call, no deletion — order the material and tell
        the reader how to use the order.
        """
        budget = max_tokens
        trajectory: list[dict[str, Any]] = []
        lines: list[str] = []

        if aaak_header:
            cost = self._tokens(aaak_header)
            if cost < budget:
                lines.append(aaak_header)
                lines.append("")
                budget -= cost

        if any(self._recency_key(r) for r in recalled):
            rule = self.CONFLICT_RULE
            cost = self._tokens(rule)
            if cost < budget:
                lines.append(rule)
                lines.append("")
                budget -= cost

        # Tier 1: dense fact-type items (world/experience/observation) first.
        # Tier 2: everything else (verbatim conversation excerpts) after.
        tier1 = self._newest_first([r for r in recalled if r.fact_type])
        tier2 = self._newest_first([r for r in recalled if not r.fact_type])

        for tier_name, bucket in (("L1", tier1), ("L2", tier2)):
            for r in bucket:
                snippet = self._render(r)
                cost = self._tokens(snippet)
                if cost > budget:
                    trajectory.append(
                        {
                            "source": r.source,
                            "fact_type": r.fact_type,
                            "tier": tier_name,
                            "tokens": cost,
                            "included": False,
                            "reason": "budget_exhausted",
                        }
                    )
                    continue
                lines.append(snippet)
                budget -= cost
                trajectory.append(
                    {
                        "source": r.source,
                        "fact_type": r.fact_type,
                        "tier": tier_name,
                        "tokens": cost,
                        "included": True,
                    }
                )

        return "\n".join(lines).strip(), trajectory

    # One line of instruction, in place of trying to infer which of two
    # conflicting statements is true. Cognee arrives at the same answer twice
    # independently (user_preferences/constants.py:60 and
    # session_context_builder.py:70), which is a strong hint that ordering plus
    # a stated rule beats inference at this price point.
    CONFLICT_RULE = "Most recent first. When two entries conflict, follow the one nearer the top."

    @staticmethod
    def _recency_key(r: Recalled) -> str:
        """Sortable recency key, empty for items that have none.

        Prefers the entry's **append position**, which is the true recency
        order: the ms prefix in an entry id cannot separate entries written
        inside the same millisecond, and that measurably made presentation
        order vary between otherwise identical eval runs. Falls back to the
        entry id, then to nothing — items with no key (typed facts from the
        brain) keep the relevance order they arrived with.
        """
        meta = r.metadata or {}
        seq = meta.get("seq")
        if isinstance(seq, int):
            return f"{seq:012d}"
        return str(meta.get("entry_id") or "")

    @classmethod
    def order_for_presentation(cls, items: list[Recalled]) -> list[Recalled]:
        """Public entry point: tier-agnostic recency ordering.

        The pipeline calls this before the index cards are cut, so the cards and
        the assembled body agree on which entry is first.
        """
        return cls._newest_first(items)

    @classmethod
    def _newest_first(cls, items: list[Recalled]) -> list[Recalled]:
        """Order by recency, keeping relevance order among items with no id.

        A stable sort on a constant key is a no-op, so items without an entry id
        (typed facts from the brain) retain the fused ranking they arrived with
        rather than being shuffled to one end.
        """
        if not any(cls._recency_key(r) for r in items):
            return items
        return sorted(items, key=cls._recency_key, reverse=True)

    @staticmethod
    def _render(r: Recalled) -> str:
        tag = r.fact_type or r.source
        return f"- ({tag}) {r.text}"

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, len(text) // _CHARS_PER_TOKEN)

    def export_paths(self, bank: str, bank_dir: Path) -> list[str]:
        return []
