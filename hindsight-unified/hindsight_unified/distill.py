"""Two-stage distillation: propose, then judge.

Turning a session into durable knowledge is two different jobs, and cognee
splits them because they need different information:

* a **curator** sees a chronological slice and proposes candidate lessons,
  merging duplicates within the slice;
* a **writer/judge** sees *one* proposal plus the lessons already stored and an
  entity glossary, and must either reject it with a typed reason or write the
  final prose.

The novelty check is therefore a *retrieval* step, not a prompt instruction —
the judge is shown the actual competing lessons rather than told to be novel.

Three lines in these prompts carry most of the value:

* *"do not promote a claim that exists only in an assistant answer unless the
  user backs it"* — an anti-hallucination-compounding rule. We store both halves
  of every turn, so without it consolidation learns the agent's own guesses as
  facts, and then learns from those.
* the typed rejection reasons (``already_known`` / ``not_durable`` /
  ``unsupported``), which make a rejection debuggable instead of silent.
* *"never paraphrase, shorten, or rename a glossary entity"* — vocabulary drift
  hurts exact-token keyword search **more** than it hurts a vector store, so
  feeding the existing names back in at write time is worth more here than it is
  in cognee.

A lesson is shown to the reader, unlike a summary — it is a legitimate derived
memory. It is therefore always labelled and always carries the ids of the
entries it came from, so the verbatim source stays one lookup away. The
verbatim record is never replaced, paraphrased in place, or removed; that is
what "never summarise user content" protects, and it still holds.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient
from .substrate import SubstrateEntry, _bm25, _corpus_stats, _tokenize

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent / "prompts"
LESSON_FILE = "lessons.jsonl"

# How many turns the curator sees at once, and how many prior lessons the judge
# is shown. Both bounded so one call's cost cannot grow with the bank.
SLICE_SIZE = 20
SIMILAR_LESSONS = 5
GLOSSARY_SIZE = 40


def lesson_id(statement: str) -> str:
    """Content-derived, and deliberately **not** time-derived.

    A lesson re-accepted on a later run must hash identically so it is absorbed
    rather than stored twice — which is also why the rendered lesson carries no
    run date.
    """
    normalized = " ".join((statement or "").split()).lower()
    return uuid.uuid5(uuid.NAMESPACE_OID, f"Lesson:{normalized}").hex


@dataclass(slots=True)
class Lesson:
    statement: str
    why_learned: str = ""
    entities: list[str] = field(default_factory=list)
    member_entry_ids: list[str] = field(default_factory=list)

    @property
    def lesson_id(self) -> str:
        return lesson_id(self.statement)

    def render(self) -> str:
        """Human-readable form. Carries no timestamp, by design."""
        parts = [self.statement]
        if self.why_learned:
            parts.append(f"(learned when: {self.why_learned})")
        return " ".join(parts)


class LessonStore:
    """Append-only sidecar of distilled lessons, one file per bank."""

    def __init__(self) -> None:
        self._curator = self._read("distill_curator.txt")
        self._writer = self._read("distill_writer.txt")

    @staticmethod
    def _read(name: str) -> str:
        path = _PROMPTS / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @staticmethod
    def path(bank_dir: Path) -> Path:
        return bank_dir / LESSON_FILE

    # -- storage ------------------------------------------------------------

    def load(self, bank_dir: Path) -> dict[str, Lesson]:
        path = self.path(bank_dir)
        if not path.exists():
            return {}
        out: dict[str, Lesson] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                statement = str(record.get("statement") or "")
                if not statement:
                    continue
                lesson = Lesson(
                    statement=statement,
                    why_learned=str(record.get("why_learned") or ""),
                    entities=list(record.get("entities") or []),
                    member_entry_ids=list(record.get("member_entry_ids") or []),
                )
                out[lesson.lesson_id] = lesson
        return out

    def append(self, bank_dir: Path, lessons: list[Lesson]) -> None:
        if not lessons:
            return
        path = self.path(bank_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for lesson in lessons:
                fh.write(
                    json.dumps(
                        {
                            "lesson_id": lesson.lesson_id,
                            "statement": lesson.statement,
                            "why_learned": lesson.why_learned,
                            "entities": lesson.entities,
                            "member_entry_ids": lesson.member_entry_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            fh.flush()
            os.fsync(fh.fileno())

    def glossary(self, bank_dir: Path) -> list[str]:
        """Entity names already in use, so the judge can reuse rather than rename."""
        seen: list[str] = []
        for lesson in self.load(bank_dir).values():
            for name in lesson.entities:
                if name and name not in seen:
                    seen.append(name)
        return seen[:GLOSSARY_SIZE]

    # -- retrieval ----------------------------------------------------------

    def search(self, bank_dir: Path, query: str, *, limit: int = 5) -> list[tuple[Lesson, float]]:
        """BM25 over lesson statements. A lesson is shown, so it is returned whole."""
        lessons = list(self.load(bank_dir).values())
        if not lessons:
            return []
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return []
        docs = [(lesson, _tokenize(lesson.render())) for lesson in lessons]
        docs = [(lesson, tokens) for lesson, tokens in docs if tokens]
        if not docs:
            return []
        idf, avg_len = _corpus_stats([tokens for _l, tokens in docs])
        scored = [
            (lesson, _bm25(q_tokens, tokens, idf=idf, avg_len=avg_len)) for lesson, tokens in docs
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # -- distillation -------------------------------------------------------

    def distill(
        self,
        bank_dir: Path,
        entries: list[SubstrateEntry],
        client: LLMClient,
    ) -> int | None:
        """Run curator then judge over the newest slice.

        Returns the number of lessons accepted, ``0`` when nothing was durable,
        and **``None`` when the LLM was unreachable**. That distinction is the
        whole point: an outage is not "nothing durable to keep", and treating it
        as such would seal these entries as distilled forever on the strength of
        calls that never ran.
        """
        if not client.available() or not self._curator or not self._writer:
            return 0
        if not entries:
            return 0

        slice_ = entries[-SLICE_SIZE:]
        proposals = self._propose(slice_, client)
        if proposals is None:
            return None  # curator call failed
        if not proposals:
            return 0  # genuinely nothing durable — distinct from an outage

        existing = self.load(bank_dir)
        glossary = self.glossary(bank_dir)
        by_id = {entry.entry_id: entry for entry in slice_}

        accepted: list[Lesson] = []
        for proposal in proposals:
            written = self._judge(proposal, by_id, existing, glossary, client)
            if written is None:
                # Publish what was accepted so far, then report the outage.
                self.append(bank_dir, accepted)
                return None
            # A rejection comes back as an empty statement; an already-known
            # lesson hashes to an id we hold, and is absorbed rather than
            # appended a second time.
            if written.statement and written.lesson_id not in existing:
                accepted.append(written)
                existing[written.lesson_id] = written

        self.append(bank_dir, accepted)
        return len(accepted)

    def _propose(self, slice_: list[SubstrateEntry], client: LLMClient) -> list[dict] | None:
        rendered = "\n\n".join(f"[{e.entry_id}]\n{e.as_text()}" for e in slice_)
        payload = client.complete_json(
            system=self._curator, user=rendered, max_tokens=1200, call_site="distill.curator"
        )
        if payload is None:
            return None
        lessons = payload.get("lessons")
        if not isinstance(lessons, list):
            # A malformed answer is not an outage: the call happened and said
            # nothing usable, so the slice is treated as holding nothing.
            logger.warning("curator returned no usable lessons list")
            return []
        return [item for item in lessons if isinstance(item, dict)]

    def _judge(
        self,
        proposal: dict,
        by_id: dict[str, SubstrateEntry],
        existing: dict[str, Lesson],
        glossary: list[str],
        client: LLMClient,
    ) -> Lesson | None:
        statement = str(proposal.get("working_statement") or "").strip()
        if not statement:
            return Lesson(statement="")  # nothing to judge; not an outage

        member_ids = [str(i) for i in (proposal.get("member_entry_ids") or [])]
        members = [by_id[i].as_text() for i in member_ids if i in by_id]
        similar = [
            lesson.statement
            for lesson, _score in self._rank_similar(statement, existing)[:SIMILAR_LESSONS]
        ]

        user = "\n\n".join(
            [
                f"PROPOSED LESSON:\n{statement}",
                "MEMBER ENTRIES:\n" + ("\n---\n".join(members) if members else "(none)"),
                "SIMILAR EXISTING LESSONS:\n"
                + ("\n".join(f"- {s}" for s in similar) if similar else "(none)"),
                "ENTITY GLOSSARY:\n" + (", ".join(glossary) if glossary else "(empty)"),
            ]
        )
        payload = client.complete_json(
            system=self._writer, user=user, max_tokens=800, call_site="distill.judge"
        )
        if payload is None:
            return None

        if not payload.get("accept"):
            reason = str(payload.get("reason") or "unspecified")
            logger.debug("lesson rejected (%s): %s", reason, statement[:80])
            return Lesson(statement="")

        return Lesson(
            statement=str(payload.get("statement") or "").strip(),
            why_learned=str(payload.get("why_learned") or "").strip(),
            entities=[str(e) for e in (payload.get("entities") or [])],
            member_entry_ids=member_ids,
        )

    @staticmethod
    def _rank_similar(statement: str, existing: dict[str, Lesson]) -> list[tuple[Lesson, float]]:
        """Cheap lexical neighbours, so the judge sees real competitors."""
        if not existing:
            return []
        lessons = list(existing.values())
        docs = [(lesson, _tokenize(lesson.statement)) for lesson in lessons]
        docs = [(lesson, tokens) for lesson, tokens in docs if tokens]
        if not docs:
            return []
        idf, avg_len = _corpus_stats([tokens for _l, tokens in docs])
        q_tokens = set(_tokenize(statement))
        scored = [
            (lesson, _bm25(q_tokens, tokens, idf=idf, avg_len=avg_len)) for lesson, tokens in docs
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
