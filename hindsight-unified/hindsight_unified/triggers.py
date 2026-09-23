"""Write-time retrieval triggers — the memory reaches toward the question.

Every lane before this one expands the *query* toward what is stored. That runs
out exactly where this system measured its ceiling: asked "what cleans up
storage we no longer need?", nothing retrieves "the nightly job that trims old
blobs is called reaper", because the two share no term and no expansion
derivable from the entry text can invent the synonym.

T-Mem (EMNLP 2026) inverts the move. At write time it generates the phrases a
future question is likely to use — "storage cleanup", "data retention" — and
indexes those alongside the memory, so the memory is reachable from a question
it shares no wording with. Its own framing is that memory is otherwise
"reachability-bounded by the similarity between a query and stored content".

Four things from its Entity/Bridge prompt do the real work, and the summary
prompt this replaces had none of them:

* **Route A**, naming the memory "one or two rungs up the is-a ladder" — a
  generative recipe for the hypernym the gap needs, rather than a request to
  summarise.
* **Route B**, a concrete scene where the memory is likely to decide the answer.
* **Three disqualifiers** — restatement, over-general label, weakly-predictive
  scene — which are what stop the output collapsing into paraphrase. A trigger
  that restates the turn adds no reach at all, and that is the failure mode a
  summary prompt walks straight into.
* **Confidence calibrated by rung height**, which gives a write-time quality
  gate that means something.

Two adaptations, both deliberate. T-Mem scores triggers by embedding cosine
behind a hard 0.85 gate; this substrate has no embeddings and stays
zero-dependency, so triggers are scored lexically — which is T-Mem's own
treatment of its other anticipated-query field, ``query_patterns``, indexed
into BM25 at a field weight. And its tri-view ``nanmax`` over concept / bridge
/ joint embeddings collapses to one document here: the views exist to spread a
short phrase and a longer sentence across vector space, whereas lexically the
joint document already contains both token sets.

The invariant is unchanged and is the reason this is safe: a trigger only ever
**votes**. ``search`` returns entry ids, never trigger text, so what is shown
is always the verbatim turn. Triggers are a rebuildable derivative and go with
the index on ``rederive``.

Requires an LLM. With none configured the lane is simply empty and recall falls
back to exactly the substrate-only behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .llm import LLMClient
from .substrate import _STOPWORD_STEMS, SubstrateEntry, _bm25, _corpus_stats, _tokenize

logger = logging.getLogger(__name__)

_PROMPT = Path(__file__).parent / "prompts" / "entity_bridge_triggers.txt"
TRIGGER_FILE = "triggers.jsonl"

# How many triggers to ask for per turn. T-Mem passes this to the prompt too;
# five spans both routes and a couple of abstraction rungs without the model
# padding out near-duplicates.
TRIGGERS_PER_ENTRY = 5
MAX_TRIGGER_TOKENS = 700

# Write-time quality gate. T-Mem filters a trigger's item links to
# ``conf >= 0.70`` when building its index, so a weak association never reaches
# the retriever at all. Below this, a trigger is noise wearing a confidence.
MIN_CONFIDENCE = 0.70


def trigger_id(entry_id: str, concept: str) -> str:
    """Derived, so the sidecar needs no join table and stays rebuildable."""
    normalized = " ".join((concept or "").split()).lower()
    return uuid.uuid5(uuid.NAMESPACE_OID, f"Trigger:{entry_id}:{normalized}").hex


@dataclass(slots=True)
class Trigger:
    entry_id: str
    concept: str
    bridge: str = ""
    confidence: float = 0.0
    # Content hash of the turn this was derived from. A mismatch means stale
    # rather than missing, which are different problems.
    source_digest: str = ""

    @property
    def trigger_id(self) -> str:
        return trigger_id(self.entry_id, self.concept)

    def document(self) -> str:
        """What the trigger is matched against: concept plus its justification.

        The bridge is included because it carries the turn's own wording on one
        side and the inference on the other, so it matches questions phrased
        either way.
        """
        return f"{self.concept} {self.bridge}".strip()


def bridge_is_grounded(bridge: str, entry_text: str) -> bool:
    """Whether the bridge's cue really comes from the turn's own wording.

    The prompt demands ``<cue lifted from the turn> -> <one-step inference>``,
    and that is checkable without a model: if no content word on the cue side
    occurs in the turn, the model invented the link rather than derived it.

    This is the cheapest guard in the whole system — a deterministic filter on
    LLM output, of the kind that was missing when consolidation could only be
    trusted to the prompt. A trigger whose justification is ungrounded is
    exactly the trigger that will pull an unrelated memory into an answer.
    """
    if not bridge:
        return False
    cue = bridge.split("->")[0] if "->" in bridge else bridge
    cue_terms = {t for t in _tokenize(cue) if t not in _STOPWORD_STEMS}
    if not cue_terms:
        return False
    return bool(cue_terms & set(_tokenize(entry_text)))


class TriggerStore:
    """Append-only sidecar of write-time triggers, one file per bank."""

    def __init__(self) -> None:
        self._prompt = _PROMPT.read_text(encoding="utf-8") if _PROMPT.exists() else ""

    # -- storage ------------------------------------------------------------

    @staticmethod
    def path(bank_dir: Path) -> Path:
        return bank_dir / TRIGGER_FILE

    def load(self, bank_dir: Path) -> dict[str, list[Trigger]]:
        """Triggers grouped by entry id. A torn line is skipped, as in the index."""
        path = self.path(bank_dir)
        if not path.exists():
            return {}
        out: dict[str, list[Trigger]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_id = str(record.get("entry_id") or "")
                concept = str(record.get("concept") or "")
                if not entry_id or not concept:
                    continue
                out.setdefault(entry_id, []).append(
                    Trigger(
                        entry_id=entry_id,
                        concept=concept,
                        bridge=str(record.get("bridge") or ""),
                        confidence=float(record.get("confidence") or 0.0),
                        source_digest=str(record.get("source_digest") or ""),
                    )
                )
        return out

    @staticmethod
    def _row(trigger: Trigger) -> str:
        return json.dumps(
            {
                "trigger_id": trigger.trigger_id,
                "entry_id": trigger.entry_id,
                "concept": trigger.concept,
                "bridge": trigger.bridge,
                "confidence": round(trigger.confidence, 4),
                "source_digest": trigger.source_digest,
            },
            ensure_ascii=False,
        )

    def append(self, bank_dir: Path, triggers: list[Trigger]) -> None:
        if not triggers:
            return
        path = self.path(bank_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for trigger in triggers:
                fh.write(self._row(trigger) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def rewrite(self, bank_dir: Path, triggers: list[Trigger]) -> None:
        """Replace the sidecar atomically (used when re-deriving)."""
        path = self.path(bank_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{TRIGGER_FILE}-", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                for trigger in triggers:
                    fh.write(self._row(trigger) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- generation ---------------------------------------------------------

    def generate_missing(
        self,
        bank_dir: Path,
        entries: list[SubstrateEntry],
        client: LLMClient,
        *,
        limit: int = 50,
    ) -> int | None:
        """Generate triggers for entries that have none current.

        Returns the number of triggers written, ``0`` when there was nothing to
        do, and **``None`` when the LLM was unreachable** — the distinction
        consolidation depends on, so an outage never looks like "nothing to
        anticipate" and never advances a watermark over work that never ran.
        """
        if not client.available() or not self._prompt:
            return 0

        existing = self.load(bank_dir)
        pending = [
            entry
            for entry in entries
            if entry.entry_id
            and (
                entry.entry_id not in existing
                or (
                    entry.digest
                    and any(t.source_digest != entry.digest for t in existing[entry.entry_id])
                )
            )
        ][:limit]
        if not pending:
            return 0

        system = self._prompt.replace("{trigger_count}", str(TRIGGERS_PER_ENTRY))
        written: list[Trigger] = []
        failed = False
        for entry in pending:
            payload = client.complete_json(
                system=system, user=entry.as_text(), max_tokens=MAX_TRIGGER_TOKENS
            )
            if payload is None:
                # Stop on the first outage rather than hammering a dead endpoint
                # for every remaining entry.
                failed = True
                break
            written.extend(self._accept(entry, payload))

        # Whatever succeeded is published: partial progress is real progress,
        # and the next run recomputes `pending` and picks up the rest.
        self.append(bank_dir, written)
        if failed:
            return None
        return len(written)

    def _accept(self, entry: SubstrateEntry, payload: dict) -> list[Trigger]:
        """Filter one turn's proposed triggers down to the ones worth indexing."""
        raw = payload.get("triggers")
        if not isinstance(raw, list):
            # A malformed answer is not an outage: the call happened and said
            # nothing usable, so this turn simply has no triggers.
            logger.warning("trigger generation returned no usable list")
            return []

        entry_text = entry.as_text()
        kept: list[Trigger] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept") or "").strip()
            bridge = str(item.get("bridge") or "").strip()
            if not concept:
                continue
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < MIN_CONFIDENCE:
                continue
            if not bridge_is_grounded(bridge, entry_text):
                logger.debug("dropped ungrounded trigger %r", concept[:60])
                continue
            trigger = Trigger(
                entry_id=entry.entry_id,
                concept=concept,
                bridge=bridge,
                confidence=confidence,
                source_digest=entry.digest,
            )
            if trigger.trigger_id in seen:
                continue
            seen.add(trigger.trigger_id)
            kept.append(trigger)
        return kept

    # -- retrieval ----------------------------------------------------------

    def search(self, bank_dir: Path, query: str, *, limit: int = 8) -> list[tuple[str, float]]:
        """BM25 over trigger documents. Returns ``(entry_id, score)`` — never text.

        An entry scores the **best** of its triggers, not their sum: a memory
        should surface if *any* anticipated phrasing matches, and summing would
        reward a turn merely for having had many triggers generated. T-Mem takes
        the same maximum over each channel's texts.
        """
        by_entry = self.load(bank_dir)
        if not by_entry:
            return []
        q_tokens = {t for t in _tokenize(query) if t not in _STOPWORD_STEMS}
        if not q_tokens:
            return []

        docs: list[tuple[Trigger, list[str]]] = []
        for triggers in by_entry.values():
            for trigger in triggers:
                tokens = _tokenize(trigger.document())
                if tokens:
                    docs.append((trigger, tokens))
        if not docs:
            return []
        idf, avg_len = _corpus_stats([tokens for _t, tokens in docs])

        best: dict[str, float] = {}
        for trigger, tokens in docs:
            score = _bm25(q_tokens, tokens, idf=idf, avg_len=avg_len)
            if score <= 0.0:
                continue
            if score > best.get(trigger.entry_id, 0.0):
                best[trigger.entry_id] = score

        ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]
