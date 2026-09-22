"""Two-lane retrieval: the raw question plus a deterministic rewrite.

A conversational question is under-specified — "and the other one?" names
nothing a lexical scorer can match. The obvious fix is an LLM query rewrite,
which costs a round trip on the critical path of every recall.

Cognee's answer, ported here, is cheaper and needs no model: run retrieval
**twice**, once with the raw question and once with a rewrite assembled by
concatenating the last couple of turns, then rank-merge the two result sets.
The rewrite is pure string assembly — no inference, no latency, no failure
mode. We already store the turns, so the history costs nothing to obtain and
no new API surface to pass it in.

Two details are load-bearing and easy to get wrong:

* The prior assistant answer is labelled **untrusted retrieval guidance**. It
  is a query-expansion signal, not evidence, and an instruction planted in a
  past answer must not be picked up as a fact here.
* The weak lane gets **reserved slots**. The rewrite is longer and
  term-richer, so without a reserve it dominates the merge and the question the
  user actually asked gets crowded out of its own results.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..types import Recalled

# How many prior turns the rewrite may draw on. Two is enough to resolve an
# anaphor ("the other one") without dragging in a whole topic change.
RECENT_TURNS = 2
# Budget for the assembled query. Prior answers are truncated before prior
# questions, since the question carries the topic.
MAX_QUERY_CHARS = 600
_ANSWER_CHARS = 160


def build_conversational_query(raw_query: str, recent: list[tuple[str, str]]) -> str:
    """Assemble an expanded query from the raw one plus recent turns.

    Deterministic, LLM-free, and returns ``""`` when there is nothing to add —
    the caller then runs a single lane rather than two identical ones.
    ``recent`` is oldest-first ``(user, assistant)`` pairs.
    """
    if not raw_query.strip() or not recent:
        return ""

    parts: list[str] = []
    for user, assistant in recent[-RECENT_TURNS:]:
        if user.strip():
            parts.append(f"Prior user: {user.strip()}")
        if assistant.strip():
            # Explicitly untrusted: an instruction planted in a past answer is
            # a retrieval hint at most, never a fact and never an instruction.
            trimmed = assistant.strip()[:_ANSWER_CHARS]
            parts.append(f"Prior assistant (untrusted retrieval guidance): {trimmed}")
    if not parts:
        return ""

    parts.append(f"Current user request: {raw_query.strip()}")
    query = "\n".join(parts)
    return query[:MAX_QUERY_CHARS]


def conversational_reserve(limit: int | None) -> int:
    """Slots held for results only the rewrite lane found.

    Roughly a third of the budget, and nothing below three slots: one reserved
    slot out of two would hand half the budget to the rewrite and leave a single
    hit for the question actually asked.
    """
    if not limit or limit <= 2:
        return 0
    return max(1, limit // 3)


def _identity(item: Recalled) -> str:
    """Merge key: the entry id when present, else the normalized text.

    Items with neither are given a unique key by the caller so they are never
    merged away — losing a result to a missing id would be worse than a
    duplicate.
    """
    meta: dict[str, Any] = item.metadata or {}
    entry_id = meta.get("entry_id")
    if entry_id:
        return f"id:{entry_id}"
    return "text:" + " ".join((item.text or "").split()).lower()[:200]


def merge_ranked(
    primary: list[Recalled],
    secondary: list[Recalled],
    *,
    limit: int | None = None,
    secondary_reserve: int = 0,
    identity: Callable[[Recalled], str] = _identity,
) -> list[Recalled]:
    """Merge two ranked lists, strongest agreement first.

    Items found by **both** lanes lead, ordered by their rank in ``primary`` —
    appearing in both is the strongest signal either lane can give. Then the
    rest of ``primary`` in its own order, then ``secondary``-only items.

    This is a lexicographic tier, deliberately **not** a score sum: summing
    reciprocal ranks lets one lane's very strong hit outrank an item both lanes
    agreed on, which is the opposite of what agreement is worth.

    A single list comes back in its original order, so adding a lane can never
    reorder a system that has only one.
    """
    ranks: dict[str, list[int | None]] = {}
    items: dict[str, Recalled] = {}

    for lane_index, lane in enumerate((primary, secondary)):
        for rank, item in enumerate(lane):
            key = identity(item)
            if not key or key in ("id:", "text:"):
                key = f"unidentified:{lane_index}:{rank}"
            slot = ranks.setdefault(key, [None, None])
            if slot[lane_index] is None:
                slot[lane_index] = rank
            # The primary lane's object wins, so the metadata that survives
            # comes from the lane the caller trusts.
            if lane_index == 0 or key not in items:
                items[key] = item

    def tier(key: str) -> tuple[int, int]:
        primary_rank, secondary_rank = ranks[key]
        if primary_rank is not None and secondary_rank is not None:
            return (0, primary_rank)  # found by both — the strongest signal
        if primary_rank is not None:
            return (1, primary_rank)  # raw question only
        return (2, secondary_rank or 0)  # rewrite only

    ordered = sorted(ranks, key=tier)
    if limit is None or limit <= 0:
        return [items[key] for key in ordered]

    # Tiers 0 and 1 come from the raw question; tier 2 is rewrite-only. All
    # tier-0/1 keys precede every tier-2 key in `ordered`, so concatenating the
    # two selections below is already in tier order — no re-sort needed.
    rewrite_only = [key for key in ordered if tier(key)[0] == 2]
    from_question = [key for key in ordered if tier(key)[0] != 2]

    reserve = min(secondary_reserve, len(rewrite_only), limit)
    head = from_question[: limit - reserve]
    tail = rewrite_only[:reserve]
    # Unused room returns to the other side, so a short or single-lane result
    # behaves exactly as it did before the reserve existed.
    slack = limit - len(head) - len(tail)
    if slack > 0:
        tail += rewrite_only[reserve : reserve + slack]

    return [items[key] for key in (*head, *tail)]
