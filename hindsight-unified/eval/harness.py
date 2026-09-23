"""The memory eval harness.

Four stages that talk to each other **only through JSON files on disk** — load
cases, answer them, score them, aggregate — plus a thin runner. That shape is
why cognee's harness is portable, and it is the whole reason this one fits in a
few hundred lines.

Run it:

    python -m eval.harness --runs 3
    python -m eval.harness --runs 3 --write-baseline
    python -m eval.harness --runs 3 --gate        # fail on a regression

What makes it honest, and is not optional:

* **A fresh store per run.** Each run seeds a brand-new memory root, so runs
  cannot contaminate each other. Cognee's harness instead destroys the whole
  store before every run, which also makes it impossible to evaluate against a
  snapshot of real memory.
* **Repeats.** One run is not a measurement. ``--runs`` defaults to 3 and the
  report carries ``run_std`` next to the confidence interval.
* **Errors are data.** A case that blows up scores 0 and says so; it never
  aborts the run.
* **A resolved-config dump** next to the artifacts, so a number can be traced
  back to the thing that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M  # noqa: E402
from eval.judge import Judge  # noqa: E402
from hindsight_unified.config import Settings  # noqa: E402
from hindsight_unified.engine import UnifiedEngine  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "cases.jsonl"
DEFAULT_DISTRACTORS = EVAL_DIR / "distractors.jsonl"
DEFAULT_RESULTS = EVAL_DIR / "results"
BASELINE_PATH = EVAL_DIR / "baseline.json"


# -- stage 1: load -------------------------------------------------------------


@dataclass(slots=True)
class Case:
    case_id: str
    category: str
    seed_turns: list[dict[str, str]]
    question: str
    answer: str
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    must_precede: list[str] = field(default_factory=list)
    # True when memory genuinely holds no answer, so surfacing anything is
    # wrong. Scored by no_false_recall; every other case skips that metric.
    expect_nothing: bool = False


def load_cases(path: Path, *, only: list[str] | None = None) -> list[Case]:
    """Read cases, optionally pinned to specific ids.

    Pinning is what makes iterating on a failure affordable — cognee's
    ``_filter_instances`` exists for the same reason.
    """
    cases: list[Case] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            raw = json.loads(line)
            cases.append(
                Case(
                    case_id=raw["case_id"],
                    category=raw.get("category", "uncategorized"),
                    seed_turns=raw.get("seed_turns", []),
                    question=raw["question"],
                    answer=raw.get("answer", ""),
                    must_contain=raw.get("must_contain", []),
                    must_not_contain=raw.get("must_not_contain", []),
                    must_precede=raw.get("must_precede", []),
                    expect_nothing=bool(raw.get("expect_nothing", False)),
                )
            )
    # A duplicate id is a silent double-count, and worse under --only, where
    # both cases are selected and the category breakdown reports whichever was
    # read last. That is not hypothetical: a new category was added with an id
    # prefix already in use, and the run reported n=6 for three cases under
    # another category's name. Fail loudly instead.
    counts = Counter(case.case_id for case in cases)
    duplicates = sorted(case_id for case_id, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate case_id in {path}: {', '.join(duplicates)}")

    if only:
        wanted = set(only)
        cases = [c for c in cases if c.case_id in wanted]
    return cases


# -- stage 2: answer -----------------------------------------------------------


def _fresh_engine(root: Path) -> UnifiedEngine:
    settings = Settings(
        host="127.0.0.1",
        port=0,
        home=root,
        enable_hindsight=False,
        enable_everos=True,
        enable_mempalace=True,
        enable_openviking=True,
        enable_memos=True,
    )
    engine = UnifiedEngine(settings)
    engine.start()
    return engine


def load_distractors(path: Path = DEFAULT_DISTRACTORS) -> list[dict[str, str]]:
    """Unrelated turns seeded into every case bank before the case's own.

    Without them the harness cannot see a ranking change at all: a bank holding
    three entries against a limit of eight returns everything, so relevance
    never binds. This was not theory — BM25 was measured against the first
    baseline and moved all three metrics by exactly 0.0000, because the
    instrument could not tell ranking from retrieval. The distractors share the
    domain's vocabulary so they compete for terms rather than being trivially
    filtered.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def answer_cases(
    cases: list[Case],
    root: Path,
    *,
    cache_dir: Path | None = None,
    distractors: list[dict[str, str]] | None = None,
) -> list[dict]:
    """Seed one bank per case, recall, and record what was retrieved.

    Retrieval and answering are kept apart: ``retrieval_context`` is stored as a
    first-class artifact so a score can be attributed to the retrieval changing
    rather than to the answer changing. That distinction is the main reason to
    capture it at all.
    """
    engine = _fresh_engine(root)
    noise = load_distractors() if distractors is None else distractors
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            cached = _read_cached(cache_dir, case.case_id)
            if cached is not None:
                rows.append(cached)
                continue
            try:
                bank = f"eval-{case.case_id}"
                # Distractors first, so the case's own turns are also the most
                # recent — the ordering metric must not be won by accident.
                for turn in noise:
                    engine.capture(
                        bank=bank,
                        session_key="eval-noise",
                        user_content=turn.get("user", ""),
                        assistant_content=turn.get("assistant", ""),
                    )
                for turn in case.seed_turns:
                    engine.capture(
                        bank=bank,
                        session_key="eval",
                        user_content=turn.get("user", ""),
                        assistant_content=turn.get("assistant", ""),
                    )
                recall = engine.recall(bank=bank, query=case.question, limit=8)
                row = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "golden_answer": case.answer,
                    "retrieval_context": recall.get("context", ""),
                    "marker": recall.get("marker", {}),
                    "num_results": len(recall.get("results", [])),
                }
            except Exception as e:  # noqa: BLE001 - one bad case must not end the run
                row = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "golden_answer": case.answer,
                    "retrieval_context": f"ERROR: {e}",
                    "marker": {},
                    "num_results": 0,
                }
            rows.append(row)
            # Failures are deliberately not cached, so a retry re-attempts them.
            if not str(row["retrieval_context"]).startswith("ERROR:"):
                _write_cached(cache_dir, row)
    finally:
        engine.stop()
    return rows


def _cache_path(cache_dir: Path | None, case_id: str) -> Path | None:
    return None if cache_dir is None else cache_dir / f"{case_id}.json"


def _read_cached(cache_dir: Path | None, case_id: str) -> dict | None:
    path = _cache_path(cache_dir, case_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cached(cache_dir: Path | None, row: dict) -> None:
    path = _cache_path(cache_dir, row["case_id"])
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


# -- stage 3: score ------------------------------------------------------------


def score_rows(rows: list[dict], cases: list[Case], *, judge: Judge | None = None) -> list[dict]:
    """Attach deterministic metrics, plus a judge score when one is configured.

    The three deterministic metrics all score *retrieval*, because retrieval is
    what this system produces: the sidecar returns a context block, not an
    answer. Exact match and token F1 are deliberately absent — comparing a
    multi-line verbatim context against a short golden answer makes EM
    structurally zero and F1 a length artifact, and a metric that cannot move
    is worse than no metric. They belong with a generation step, which lives in
    the agent, not here. The judge fills that gap when configured: its prompt
    explicitly tolerates extra detail, so it can score a context against a
    golden answer meaningfully.
    """
    by_id = {c.case_id: c for c in cases}
    for row in rows:
        case = by_id[row["case_id"]]
        context = str(row.get("retrieval_context") or "")
        scored: dict[str, float | None] = {
            "abstention": M.must_not_appear(context, case.must_not_contain),
            "ordering": M.ordering_correct(context, case.must_precede),
        }
        # A no-answer case has nothing to recall, so recall_hit would score a
        # meaningless 1.0 for it. It is also NOT scored for false recall: that
        # metric was tried and removed, because suppressing a no-answer recall
        # is not achievable with a lexical signal — see eval/README.md. The
        # cases stay in the set as probes for when a semantic lane lands.
        if not case.expect_nothing:
            scored["recall_hit"] = M.recall_hit(context, case.must_contain)
        if judge is not None and judge.available():
            score, explanation = judge.score(
                question=case.question, answer=context, golden_answer=case.answer
            )
            # None stays None: a judge outage is missing data, not a zero.
            scored["judged"] = score
            row["judge_explanation"] = explanation
        row["metrics"] = scored
    return rows


# -- stage 4: aggregate + report ----------------------------------------------


def run_once(cases: list[Case], run_dir: Path, *, judge: Judge | None) -> list[dict]:
    store = run_dir / "store"
    rows = answer_cases(cases, store, cache_dir=run_dir / "answers")
    rows = score_rows(rows, cases, judge=judge)
    M_write(run_dir / "scored.json", rows)
    return rows


def M_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified memory eval harness")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--runs", type=int, default=3, help="repeats; one run is not a measurement")
    parser.add_argument("--only", nargs="*", default=None, help="pin specific case ids")
    parser.add_argument("--no-judge", action="store_true", help="deterministic metrics only")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--gate", action="store_true", help="exit non-zero on a regression")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases, only=args.only)
    if not cases:
        print("no cases loaded", file=sys.stderr)
        return 2

    judge = None if args.no_judge else Judge()
    judging = judge is not None and judge.available()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.results / stamp
    runs: list[list[dict]] = []
    for index in range(args.runs):
        # A fresh store per run: runs must not contaminate each other, and
        # nothing outside the run directory is touched.
        runs.append(run_once(cases, out_dir / f"run-{index}", judge=judge))

    summary = M.aggregate(runs)
    M_write(out_dir / "aggregate.json", summary.as_dict())
    M_write(
        out_dir / "eval_config.json",
        {
            "cases": str(args.cases),
            "num_cases": len(cases),
            "runs": args.runs,
            "judging": judging,
            "judge_model": judge._client.model if judging else "",
            "pinned": args.only or [],
        },
    )

    print(f"\n{len(cases)} cases x {args.runs} runs -> {out_dir}")
    print(f"judge: {'on' if judging else 'off (deterministic metrics only)'}\n")
    for name in sorted(summary.metrics):
        print("  " + summary.metrics[name].line())
    print("\nby category:")
    for category, scores in summary.by_category.items():
        compact = " ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items()))
        print(f"  {category:<26} {compact}")

    exit_code = 0
    if args.write_baseline:
        M_write(BASELINE_PATH, summary.as_dict())
        print(f"\nbaseline written to {BASELINE_PATH}")
    elif BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        gate = M.check_regression(summary, baseline)
        print("\n" + gate.report())
        for name in sorted(summary.metrics):
            print("  " + M.describe_delta(summary, baseline, name))
        if args.gate and not gate.passed:
            exit_code = 1
    else:
        print("\nno baseline recorded yet — run with --write-baseline to set one")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
