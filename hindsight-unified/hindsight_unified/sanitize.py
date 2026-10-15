"""Neutralise instruction-like text in recalled memory before it reaches a prompt.

Recall injects stored text into the model's context every turn, without the
agent asking for it, and the host frames that block as "authoritative
reference data". Stored text is verbatim by design - including whatever an
assistant reply quoted from a web page or a file. So a stored "ignore all
previous instructions" arrives in the prompt as authoritative memory.

The patterns are gbrain's (``src/core/think/sanitize.ts``, ``INJECTION_PATTERNS``),
which it also applies to retrieved chat content in its LongMemEval harness.
Its own caveat applies unchanged: "We don't pretend this is bulletproof." The
structural defence is the host's fence around recalled memory; this removes the
trivial attacks that a fence alone would still deliver.

Ported selectively. gbrain's set is tuned for short claims; this memory holds
a coding agent's verbatim turns. Every pattern was run against ordinary
engineering sentences, and each one below rewrote at least one of them:

* ``eval-shell`` (``eval(``, ``system(``, ``shell(``): "the deploy script calls
  os.system(cmd)". Dropped.
* ``do-anything-now``'s bare ``DAN`` is case-insensitive, so it matches the
  name: "Dan owns the billing service". The ``DAN`` alternative is dropped;
  "do anything now" and "developer mode enabled" are kept.
* ``forget-everything``: "Forget everything about the old staging cluster",
  which is a genuine request for the forget tool. Narrowed to the jailbreak
  form (forget everything/all *above*, *before*, *previous*, *you were told*).
* ``system-prompt``: "System: always run migrations first". Narrowed to
  "System: you are / you must", which no longer catches "System: never ...".
* ``print-system``: "Print instructions for the on-call rotation". Narrowed to
  the system prompt and hidden instructions.
* ``role-jailbreak``: "You are now an admin on the infra repo", a statement
  this memory should hold. Dropped.
* ``open-instructions``: "<instructions> tags in the prompt template".
  ``verbatim``: "the WebSocket server will echo back each frame".
  ``new-instructions``: "New instructions: setup on macOS". This one also
  strips only the label and leaves the payload, so it was not worth its false
  positives. All three dropped.
* ``xml-attr-inject``: ``kind="Deployment"`` in a manifest. It protects
  attributes on gbrain's own ``<take>`` tags, and so do ``close-take`` and
  ``close/open-trajectory``. This pipeline renders no such tags. Dropped.

gbrain's 500-character cap is not ported, because the assembler already
budgets every entry.

Applied to what recall injects. The markdown truth is never rewritten, and
conversation search still returns the exact words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_I = re.IGNORECASE


@dataclass(frozen=True, slots=True)
class _Pattern:
    name: str
    regex: re.Pattern[str]
    replacement: str


INJECTION_PATTERNS: tuple[_Pattern, ...] = (
    _Pattern(
        "ignore-prior",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:prior|previous|above|earlier)\s+"
            r"(?:instructions?|prompts?|messages?)",
            _I,
        ),
        "[redacted]",
    ),
    _Pattern(
        "disregard",
        re.compile(
            r"disregard\s+(?:all\s+)?(?:prior|previous|above|earlier)\s+"
            r"(?:instructions?|prompts?)",
            _I,
        ),
        "[redacted]",
    ),
    _Pattern(
        "forget-above",
        re.compile(
            r"forget\s+(?:everything|all)(?:\s+of)?(?:\s+the)?\s+"
            r"(?:above|before|previous|prior|you\s+(?:were|have\s+been)\s+told)",
            _I,
        ),
        "[redacted]",
    ),
    _Pattern("system-you", re.compile(r"system\s*:\s*you\s+(?:are|must)", _I), "[redacted]"),
    _Pattern(
        "do-anything-now",
        re.compile(r"\b(?:do\s+anything\s+now|developer\s+mode\s+enabled?)\b", _I),
        "[redacted]",
    ),
    _Pattern("open-system", re.compile(r"<\s*system\s*>", _I), "&lt;system&gt;"),
    _Pattern(
        "print-system",
        re.compile(
            r"(?:print|output|reveal|show)\s+(?:your\s+)?"
            r"(?:system\s+prompt|hidden\s+instructions?)",
            _I,
        ),
        "[redacted]",
    ),
)


def neutralize(text: str) -> tuple[str, list[str]]:
    """``(cleaned_text, names_of_patterns_that_fired)``."""
    matched: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.regex.search(text):
            matched.append(pattern.name)
            text = pattern.regex.sub(pattern.replacement, text)
    return text, matched
