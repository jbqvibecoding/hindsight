"""Scoring and aggregation for the memory eval.

Two deterministic scorers and the statistics that decide whether a change did
anything. All stdlib — cognee computes its bootstrap CI with numpy, which is
not worth a dependency for this.

The statistics matter more than the scorers. With thirty cases and an LLM
judge, a mean moving 0.65 -> 0.70 is usually noise, and there are two different
variances to respect:

* ``ci_lower``/``ci_upper`` — spread across *cases within one run*.
* ``run_std`` — spread across *repeated identical runs*, i.e. judge and
  retrieval nondeterminism. **If ``run_std`` exceeds the delta you are looking
  at, the change did nothing.**

So: repeat every run, report both, and refuse to call overlapping intervals an
improvement.
"""

from __future__ import annotations

import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def exact_match(answer: str, golden: str) -> float:
    return 1.0 if normalize(answer) == normalize(golden) else 0.0


def token_f1(answer: str, golden: str) -> float:
    """Token-level F1 over multiset intersection (the SQuAD/HotPotQA measure)."""
    actual = Counter(_WORD.findall(normalize(answer)))
    expected = Counter(_WORD.findall(normalize(golden)))
    if not actual or not expected:
        return 1.0 if not actual and not expected else 0.0

    true_positive = sum(min(actual[t], expected[t]) for t in actual)
    if true_positive == 0:
        return 0.0
    precision = true_positive / sum(actual.values())
    recall = true_positive / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


# Inflections an anchor may appear under. Deliberately short and **explicitly
# enumerated** rather than a regex suffix wildcard: over-stemming manufactures
# hits, which is the failure the word-boundary fix already had to undo once.
_STRIPPABLE = ("ing", "es", "ed", "s")
_MIN_STEM = 4


def _stem(token: str) -> str:
    """Strip one common suffix, but only when a real word is left behind.

    The ``_MIN_STEM`` guard is what keeps this honest: without it "tabs" stems
    to "tab" and "is" to "i", and short stems match far too much.
    """
    for suffix in _STRIPPABLE:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def _inflections(stem: str) -> list[str]:
    """Forms a stem may legitimately appear as, enumerated not guessed.

    Includes the silent-e drop, because appending a suffix to the full stem
    gives "cacheing" rather than "caching" — the kind of gap that quietly costs
    a true hit. Enumeration keeps the set auditable; a wildcard would not.
    """
    forms = {stem, stem + "s", stem + "es", stem + "ed", stem + "d", stem + "ing"}
    if stem.endswith("e") and len(stem) - 1 >= _MIN_STEM - 1:
        root = stem[:-1]
        forms.update({root + "ing", root + "ed"})
    # Longest first so the alternation prefers the most specific form.
    return sorted(forms, key=len, reverse=True)


def _anchor_pattern(anchor: str) -> str:
    """Regex for one anchor: word-bounded, inflection-tolerant, punctuation-safe.

    Each alphabetic token of the anchor is stemmed and allowed to reappear under
    a common inflection, so ``drain`` matches "drained" and ``migration``
    matches "migrations" — both cases where the system retrieved the right entry
    and ranked it first, and only the scorer disagreed. Numeric and mixed tokens
    are matched literally, so "100" still does not match "1000". Tokens are
    joined on non-word runs, which is how a punctuated anchor like
    ``eu-central-1`` or ``release.sh`` keeps matching as written.
    """
    tokens = re.findall(r"\w+", normalize(anchor))
    if not tokens:
        return ""
    parts = []
    for token in tokens:
        if token.isalpha() and len(token) >= _MIN_STEM:
            variants = "|".join(re.escape(f) for f in _inflections(_stem(token)))
            parts.append(f"(?:{variants})")
        else:
            parts.append(re.escape(token))
    return r"(?<!\w)" + r"\W+".join(parts) + r"(?!\w)"


def contains_anchor(haystack: str, anchor: str) -> bool:
    """Case-insensitive, word-bounded, inflection-tolerant match.

    Plain substring matching is wrong for short factual anchors ("Wen" matches
    inside "when", "100" inside "1000"), and a purely literal word-bounded
    match is wrong for inflections ("drain" vs "drained"). Both mistakes were
    made here in turn; this is the form that survived both.
    """
    pattern = _anchor_pattern(anchor)
    if not pattern:
        return False
    return re.search(pattern, normalize(haystack)) is not None


def recall_hit(context: str, must_contain: list[str]) -> float:
    """Fraction of required anchors the retrieved context actually contained.

    Scores *retrieval* rather than generation, which is the difference between
    "the answer got worse" and "the answer got worse because the retrieved set
    changed".
    """
    if not must_contain:
        return 1.0
    hits = sum(1 for needle in must_contain if contains_anchor(context, needle))
    return hits / len(must_contain)


def no_false_recall(context: str, *, expect_nothing: bool) -> float | None:
    """1.0 when a question memory cannot answer surfaces nothing.

    **Not currently scored by the harness.** Kept because the measurement is
    correct and becomes meaningful the moment a semantic lane exists — but with
    only a lexical signal the requirement it encodes is unachievable, so
    reporting it would pin a metric at 0.00 forever. Three discriminators were
    measured and all three failed:

    * *Raw-question lane empty* — fails: legitimate anaphoric follow-ups
      ("who should I ask instead?") also match zero question terms.
    * *Absolute fused score* — fails: RRF scores are ``1/(k+rank+1)`` by
      construction, identical across queries, and carry no relevance
      information at all.
    * *Absolute BM25 score* — fails, decisively: a no-answer question scored
      **20.31** at the top against **18.58** for a follow-up that should
      answer. The distributions overlap because the rewrite query *contains the
      recent turns verbatim*, so those turns score highly against it whether or
      not they relate to the question. The score measures similarity to the
      borrowed text, not relevance to the question.

    Returns ``None`` for a case that has an answer, so it is excluded rather
    than scored — the metric only means something where silence is correct.
    """
    if not expect_nothing:
        return None
    return 1.0 if not context_blocks(context) else 0.0


def context_blocks(context: str) -> list[str]:
    """Split an assembled context into its per-entry blocks.

    Every rendered entry — index card or body line — starts with ``- (`` or
    ``[n] (``, so the block boundary is recoverable from the text alone without
    the harness needing the structured result.
    """
    parts = re.split(r"(?m)^(?:- \(|\[\d+\] \()", context or "")
    return [p for p in parts if p.strip()]


def ordering_correct(context: str, must_precede: list[str]) -> float:
    """1.0 when the entry holding the current fact leads the one it replaced.

    The measure for "the user changed their mind". Superseded facts are tagged,
    not deleted, so the older statement legitimately remains retrievable — what
    must hold is that the *current* one is presented first, since that is the
    signal the reading model acts on. Absence would be the wrong assertion.

    Compared by **entry position, not character offset**. Found by inspecting a
    case rather than trusting the score: "Ravi handed on-call over to Mira"
    mentions the superseded holder before the current one *inside the current
    entry*, so an offset comparison failed a correctly ordered context. Two
    anchors landing in the same entry is therefore a pass — the leading entry is
    the current one, which is all this claims.

    Scores 0.0 when the current fact is missing entirely: a worse failure than
    bad ordering, not an exemption from it.
    """
    if len(must_precede) != 2:
        return 1.0
    blocks = context_blocks(context)
    current = next((i for i, b in enumerate(blocks) if contains_anchor(b, must_precede[0])), -1)
    stale = next((i for i, b in enumerate(blocks) if contains_anchor(b, must_precede[1])), -1)
    if current < 0:
        return 0.0  # the current fact was not recalled at all
    if stale < 0:
        return 1.0  # only the current fact present — trivially ordered
    return 1.0 if current <= stale else 0.0


def must_not_appear(context: str, forbidden: list[str]) -> float:
    """1.0 when nothing forbidden leaked into the context.

    The abstention case: a memory system is also judged on what it does *not*
    surface, and nothing else here would catch a confident wrong recall.

    **Never read this in isolation.** A system that retrieves nothing at all
    scores a perfect 1.0 here — verified by crippling recall, which sent
    ``recall_hit`` to 0.07 and ``abstention`` to 1.00. It is the precision half
    of a pair, and only means something next to ``recall_hit``.
    """
    if not forbidden:
        return 1.0
    return 0.0 if any(contains_anchor(context, n) for n in forbidden) else 1.0


# -- statistics ----------------------------------------------------------------


def bootstrap_ci(
    scores: list[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """``(mean, ci_lower, ci_upper)`` by resampling with replacement.

    Seeded, so the interval is reproducible for a fixed score list — otherwise
    the tool meant to tell signal from noise adds noise of its own. 2000 samples
    is statistically indistinguishable from cognee's 10000 at this scale.
    """
    if not scores:
        return 0.0, 0.0, 0.0
    if len(scores) == 1:
        return scores[0], scores[0], scores[0]

    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(samples))
    lower_index = int((1 - confidence) / 2 * samples)
    upper_index = min(samples - 1, int((1 + confidence) / 2 * samples))
    return statistics.fmean(scores), means[lower_index], means[upper_index]


@dataclass(slots=True)
class MetricSummary:
    metric: str
    mean: float
    ci_lower: float
    ci_upper: float
    n: int
    run_std: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": round(self.mean, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "n": self.n,
            "run_std": None if self.run_std is None else round(self.run_std, 4),
        }

    def line(self) -> str:
        std = "" if self.run_std is None else f"  run_std={self.run_std:.4f}"
        return (
            f"{self.metric:<16} mean={self.mean:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]  n={self.n}{std}"
        )


@dataclass(slots=True)
class Aggregate:
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    runs: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "metrics": {k: v.as_dict() for k, v in self.metrics.items()},
            "by_category": self.by_category,
        }


def validate_runs(runs: list[list[dict[str, Any]]]) -> None:
    """Every repeat must cover the same cases in the same order.

    Cheap, and it catches a whole class of silently meaningless comparison —
    two runs over different case sets whose means look comparable.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    reference = [row.get("case_id") for row in runs[0]]
    for index, run in enumerate(runs[1:], start=1):
        if [row.get("case_id") for row in run] != reference:
            raise ValueError(f"run {index} covers different cases than run 0 — not comparable")


def aggregate(runs: list[list[dict[str, Any]]]) -> Aggregate:
    """Fold repeated runs into per-metric summaries plus a category breakdown."""
    validate_runs(runs)

    metric_names: list[str] = []
    for run in runs:
        for row in run:
            for name in row.get("metrics") or {}:
                if name not in metric_names:
                    metric_names.append(name)

    out = Aggregate(runs=len(runs))
    for name in metric_names:
        pooled: list[float] = []
        per_run_means: list[float] = []
        for run in runs:
            scores = [
                float(row["metrics"][name])
                for row in run
                if name in (row.get("metrics") or {}) and row["metrics"][name] is not None
            ]
            pooled.extend(scores)
            if scores:
                per_run_means.append(statistics.fmean(scores))
        if not pooled:
            continue
        mean, lower, upper = bootstrap_ci(pooled)
        out.metrics[name] = MetricSummary(
            metric=name,
            mean=mean,
            ci_lower=lower,
            ci_upper=upper,
            n=len(pooled),
            # Needs at least two repeats to say anything about run-to-run spread.
            run_std=statistics.stdev(per_run_means) if len(per_run_means) >= 2 else None,
        )

    # Per-category means: this is where "helped multi-session reasoning, hurt
    # abstention" becomes visible instead of averaging out.
    buckets: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for row in run:
            category = row.get("category") or "uncategorized"
            for name, score in (row.get("metrics") or {}).items():
                if score is None:
                    continue
                buckets.setdefault(category, {}).setdefault(name, []).append(float(score))
    out.by_category = {
        category: {name: round(statistics.fmean(scores), 4) for name, scores in metrics.items()}
        for category, metrics in sorted(buckets.items())
    }
    return out


# -- regression gate -----------------------------------------------------------


@dataclass(slots=True)
class GateResult:
    passed: bool
    findings: list[str] = field(default_factory=list)

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        return "\n".join([f"regression gate: {head}", *(f"  - {f}" for f in self.findings)])


def check_regression(
    current: Aggregate, baseline: dict[str, Any], *, metrics: list[str] | None = None
) -> GateResult:
    """Fail when a metric's mean drops below the baseline's lower bound.

    Cognee's harness has no equivalent — nothing in its eval framework compares
    a run to the previous one. Comparing against the baseline's ``ci_lower``
    rather than its mean is what keeps the gate from firing on ordinary noise:
    a drop has to clear the baseline's own spread to count.
    """
    baseline_metrics = (baseline or {}).get("metrics") or {}
    names = metrics or sorted(set(current.metrics) & set(baseline_metrics))
    findings: list[str] = []
    passed = True

    for name in names:
        summary = current.metrics.get(name)
        recorded = baseline_metrics.get(name)
        if summary is None or not recorded:
            findings.append(
                f"{name}: missing from {'current run' if summary is None else 'baseline'}"
            )
            continue
        floor = float(recorded.get("ci_lower", recorded.get("mean", 0.0)))
        if summary.mean < floor:
            passed = False
            findings.append(
                f"{name}: {summary.mean:.4f} below baseline floor {floor:.4f} — regression"
            )
        else:
            findings.append(f"{name}: {summary.mean:.4f} >= baseline floor {floor:.4f}")

    return GateResult(passed, findings)


def describe_delta(current: Aggregate, previous: dict[str, Any], metric: str) -> str:
    """State a change honestly, including when it is indistinguishable from noise."""
    summary = current.metrics.get(metric)
    recorded = ((previous or {}).get("metrics") or {}).get(metric)
    if summary is None or not recorded:
        return f"{metric}: no comparison available"

    before = float(recorded.get("mean", 0.0))
    delta = summary.mean - before
    noise = summary.run_std or 0.0
    overlaps = summary.ci_lower <= float(recorded.get("ci_upper", before)) and (
        float(recorded.get("ci_lower", before)) <= summary.ci_upper
    )

    if noise and abs(delta) <= noise:
        verdict = f"within run-to-run noise (run_std={noise:.4f}) — no conclusion"
    elif overlaps:
        verdict = "confidence intervals overlap — no conclusion"
    else:
        verdict = "improvement" if delta > 0 else "regression"
    return f"{metric}: {before:.4f} -> {summary.mean:.4f} ({delta:+.4f}) — {verdict}"
