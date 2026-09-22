"""LLM-as-judge for the memory eval.

The prompt is cognee's ``direct_llm_eval_system.txt`` plus a JSON instruction,
kept near-verbatim because its anti-bias clauses are the whole asset: compare by
common-sense meaning rather than wording, do not penalise length, extra detail
is fine. Those three lines are the difference between a judge that measures
correctness and one that measures phrasing.

The judge is **optional**. With no LLM configured every case still gets its
deterministic scores (exact match, token F1, retrieval hit, abstention), so the
harness is useful before anyone sets a key — and a judge outage produces
``None``, never a 0.0, so a failed call cannot be averaged in as a bad answer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hindsight_unified.llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent / "prompts"


def _read(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


class Judge:
    """Scores one answer against a golden answer. Never raises."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client if client is not None else LLMClient()
        self._system = _read("judge_system.txt")
        self._user_template = _read("judge_user.txt")

    def available(self) -> bool:
        return self._client.available()

    def score(self, *, question: str, answer: str, golden_answer: str) -> tuple[float | None, str]:
        """``(score, explanation)``; ``score`` is ``None`` when the call failed.

        ``None`` must not become 0.0 anywhere downstream: a judge that could not
        be reached is missing data, and averaging it in as a wrong answer would
        make an outage look like a regression.
        """
        if not self.available():
            return None, "judge_unconfigured"

        user = self._user_template.format(
            question=question, answer=answer, golden_answer=golden_answer
        )
        payload = self._client.complete_json(system=self._system, user=user, max_tokens=300)
        if payload is None:
            return None, "judge_unavailable"

        raw_score = payload.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            logger.warning("judge returned a non-numeric score: %r", raw_score)
            return None, "judge_unparseable_score"

        explanation = str(payload.get("explanation") or "")
        return max(0.0, min(1.0, score)), explanation
