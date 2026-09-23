"""Tests for the eval harness (workstream C).

A measurement instrument needs its own tests more than the code it measures:
a harness that silently mis-scores produces confident wrong conclusions. These
cover the scorers, the statistics, and — most importantly — that the regression
gate actually fires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import metrics as M
from eval.harness import Case, answer_cases, load_cases, score_rows

# -- scorers -------------------------------------------------------------------


def test_recall_hit_is_a_fraction_of_required_anchors() -> None:
    assert M.recall_hit("port 6543 and RELEASE_TOKEN", ["6543", "RELEASE_TOKEN"]) == 1.0
    assert M.recall_hit("port 6543 only", ["6543", "RELEASE_TOKEN"]) == 0.5
    assert M.recall_hit("nothing useful", ["6543"]) == 0.0
    assert M.recall_hit("anything", []) == 1.0  # no requirement is trivially met


def test_anchors_match_on_word_boundaries_not_substrings() -> None:
    """Substring matching would report hits the system never made."""
    assert M.recall_hit("when did that happen", ["Wen"]) == 0.0
    assert M.recall_hit("Wen owns the tokenizer", ["Wen"]) == 1.0
    assert M.recall_hit("the limit is 1000", ["100"]) == 0.0
    assert M.recall_hit("the limit is 100", ["100"]) == 1.0
    # Anchors carrying punctuation still match as written.
    assert M.recall_hit("deploy to eu-central-1 now", ["eu-central-1"]) == 1.0
    assert M.recall_hit("run ops/deploy/release.sh", ["release.sh"]) == 1.0


def test_abstention_catches_a_leak() -> None:
    assert M.must_not_appear("it is in the vault", ["hunter2"]) == 1.0
    assert M.must_not_appear("the password is hunter2", ["hunter2"]) == 0.0


def _ctx(*entries: str) -> str:
    return "\n".join(f"- (substrate) {e}" for e in entries)


def test_ordering_scores_recency_not_absence() -> None:
    # A superseded fact may legitimately still be present — what matters is
    # that the entry holding the current one leads.
    assert M.ordering_correct(_ctx("4 spaces now", "I use tabs"), ["4 spaces", "tabs"]) == 1.0
    assert M.ordering_correct(_ctx("I use tabs", "4 spaces now"), ["4 spaces", "tabs"]) == 0.0
    # Only the current fact present: trivially ordered.
    assert M.ordering_correct(_ctx("4 spaces now"), ["4 spaces", "tabs"]) == 1.0
    # Current fact missing entirely is the worse failure, not an exemption.
    assert M.ordering_correct(_ctx("I use tabs"), ["4 spaces", "tabs"]) == 0.0


def test_ordering_compares_entries_not_character_offsets() -> None:
    """Both anchors inside the leading entry is a pass, not a failure.

    Real case: "Ravi handed on-call over to Mira" names the superseded holder
    before the current one within the current entry. A character-offset
    comparison failed a correctly ordered context.
    """
    correct = _ctx("Ravi handed on-call over to Mira", "Ravi is the on-call")
    assert M.ordering_correct(correct, ["Mira", "Ravi"]) == 1.0

    stale_leads = _ctx("Ravi is the on-call", "Ravi handed on-call over to Mira")
    assert M.ordering_correct(stale_leads, ["Mira", "Ravi"]) == 0.0


def test_token_f1_and_exact_match() -> None:
    assert M.exact_match("  4 Spaces ", "4 spaces") == 1.0
    assert M.exact_match("tabs", "4 spaces") == 0.0
    assert M.token_f1("the cat sat", "the cat sat") == 1.0
    assert M.token_f1("nothing alike", "the cat sat") == 0.0
    assert 0.0 < M.token_f1("the cat", "the cat sat") < 1.0


# -- statistics ----------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    scores = [1.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.75]
    mean, lower, upper = M.bootstrap_ci(scores)
    assert lower <= mean <= upper
    # Seeded: the tool that tells signal from noise must not add noise.
    assert M.bootstrap_ci(scores) == (mean, lower, upper)


def test_bootstrap_ci_degenerate_inputs() -> None:
    assert M.bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert M.bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_run_std_is_zero_for_identical_repeats_and_positive_otherwise() -> None:
    def run(score: float) -> list[dict]:
        return [{"case_id": "a", "category": "c", "metrics": {"m": score}}]

    identical = M.aggregate([run(1.0), run(1.0), run(1.0)])
    assert identical.metrics["m"].run_std == 0.0

    varying = M.aggregate([run(1.0), run(0.0), run(0.5)])
    assert varying.metrics["m"].run_std and varying.metrics["m"].run_std > 0.0


def test_a_single_run_reports_no_run_std() -> None:
    single = M.aggregate([[{"case_id": "a", "category": "c", "metrics": {"m": 1.0}}]])
    # One run cannot say anything about run-to-run spread, so it must not claim to.
    assert single.metrics["m"].run_std is None


def test_mismatched_runs_are_refused() -> None:
    a = [{"case_id": "a", "category": "c", "metrics": {"m": 1.0}}]
    b = [{"case_id": "b", "category": "c", "metrics": {"m": 1.0}}]
    with pytest.raises(ValueError, match="different cases"):
        M.aggregate([a, b])


def test_none_scores_are_excluded_not_counted_as_zero() -> None:
    """A judge outage is missing data; averaging it as 0.0 fakes a regression."""
    rows = [
        {"case_id": "a", "category": "c", "metrics": {"judged": 1.0}},
        {"case_id": "b", "category": "c", "metrics": {"judged": None}},
    ]
    summary = M.aggregate([rows])
    assert summary.metrics["judged"].mean == 1.0
    assert summary.metrics["judged"].n == 1


# -- regression gate -----------------------------------------------------------


def _aggregate_at(mean: float) -> M.Aggregate:
    rows = [
        {"case_id": f"c{i}", "category": "x", "metrics": {"recall_hit": mean}} for i in range(5)
    ]
    return M.aggregate([rows])


def test_gate_fails_on_a_drop_below_the_baseline_floor() -> None:
    baseline = _aggregate_at(0.9).as_dict()
    gate = M.check_regression(_aggregate_at(0.4), baseline)
    assert gate.passed is False
    assert any("regression" in f for f in gate.findings)


def test_gate_passes_when_the_metric_holds() -> None:
    baseline = _aggregate_at(0.9).as_dict()
    assert M.check_regression(_aggregate_at(0.95), baseline).passed is True


def test_delta_refuses_a_verdict_inside_the_noise_floor() -> None:
    def run(score: float) -> list[dict]:
        return [{"case_id": "a", "category": "c", "metrics": {"m": score}}]

    noisy = M.aggregate([run(0.9), run(0.5), run(0.7)])  # run_std ~0.2
    previous = M.aggregate([run(0.68), run(0.70), run(0.72)]).as_dict()
    verdict = M.describe_delta(noisy, previous, "m")
    assert "no conclusion" in verdict


# -- end to end ----------------------------------------------------------------


def test_harness_sees_the_changed_mind_case_end_to_end(tmp_path: Path) -> None:
    """The case that pinned the original defect, now pinning the fix.

    Written when both assertions were 0.0: keyword overlap matched only the
    stale turn (which contains "indent") and missed the correction entirely,
    which then also made ordering impossible. The two-lane rewrite retrieves
    the correction and recency presentation puts it first.
    """
    case = Case(
        case_id="cr-live",
        category="contradiction_resolution",
        seed_turns=[
            {"user": "I always indent with tabs.", "assistant": "Noted."},
            {"user": "Actually we standardised on 4 spaces.", "assistant": "Updated."},
        ],
        question="How do I indent?",
        answer="4 spaces",
        must_contain=["spaces"],
        must_precede=["4 spaces", "tabs"],
    )
    rows = score_rows(answer_cases([case], tmp_path / "store"), [case])
    assert rows[0]["metrics"]["recall_hit"] == 1.0
    assert rows[0]["metrics"]["ordering"] == 1.0
    # The superseded fact is still there — tagged by order, never deleted.
    assert "tabs" in rows[0]["retrieval_context"]


def test_errors_become_data_not_an_aborted_run(tmp_path: Path, monkeypatch) -> None:
    case = Case(
        case_id="boom",
        category="x",
        seed_turns=[],
        question="q",
        answer="6543",
        must_contain=["6543"],
    )
    monkeypatch.setattr(
        "hindsight_unified.engine.UnifiedEngine.recall",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recall exploded")),
    )
    rows = answer_cases([case], tmp_path / "store")
    assert rows[0]["retrieval_context"].startswith("ERROR:")
    # Scoring still works: the case scores 0 instead of taking the run down.
    scored = score_rows(rows, [case])
    assert scored[0]["metrics"]["recall_hit"] == 0.0


def test_failures_are_not_cached_but_successes_are(tmp_path: Path) -> None:
    case = Case(
        case_id="ok-01",
        category="x",
        seed_turns=[{"user": "the port is 6543", "assistant": "noted"}],
        question="port?",
        answer="6543",
        must_contain=["6543"],
    )
    cache = tmp_path / "answers"
    answer_cases([case], tmp_path / "store", cache_dir=cache)
    assert (cache / "ok-01.json").exists()

    # A second pass reads the cache rather than re-seeding the store.
    again = answer_cases([case], tmp_path / "store2", cache_dir=cache)
    assert again[0]["case_id"] == "ok-01"


def test_shipped_case_set_is_wellformed() -> None:
    cases = load_cases(Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl")
    assert len(cases) == 45
    assert len({c.case_id for c in cases}) == 45
    # Ten BEAM-style skill categories plus five of our own, three cases each.
    categories = {}
    for case in cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    assert len(categories) == 15
    assert set(categories.values()) == {3}
    for case in cases:
        assert case.question and case.answer
        # A no-answer case asserts silence, so it carries no anchors by design.
        if not case.expect_nothing:
            assert case.must_contain or case.must_not_contain or case.must_precede


def test_pinning_selects_a_subset() -> None:
    path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    assert [c.case_id for c in load_cases(path, only=["cr-01", "ku-02"])] == ["cr-01", "ku-02"]


def test_a_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    """A duplicate id double-counts silently, which is worse than a crash.

    Found the hard way: a new category reused an id prefix already in use, and
    ``--only`` then selected both cases, reporting n=6 for three cases under
    the other category's name.
    """
    path = tmp_path / "cases.jsonl"
    row = {"case_id": "dup-01", "category": "x", "question": "q?", "answer": "a"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dup-01"):
        load_cases(path)


def test_baseline_on_disk_is_loadable_and_has_the_retrieval_metrics() -> None:
    baseline = json.loads(
        (Path(__file__).resolve().parents[1] / "eval" / "baseline.json").read_text("utf-8")
    )
    assert baseline["runs"] >= 3, "a baseline from a single run is not a measurement"
    assert {"recall_hit", "ordering", "abstention"} <= set(baseline["metrics"])


def test_anchors_tolerate_inflections() -> None:
    """The system retrieved these and ranked them first; only the scorer disagreed."""
    assert M.recall_hit("First we drained the queue", ["drain"]) == 1.0
    assert M.recall_hit("Run migrations before starting the API", ["migration"]) == 1.0
    assert M.recall_hit("we are caching the model", ["cache"]) == 1.0
    # Still conservative: a short stem must not open the floodgates.
    assert M.recall_hit("when did that happen", ["Wen"]) == 0.0
    assert M.recall_hit("the limit is 1000", ["100"]) == 0.0
    assert M.recall_hit("I prefer tab characters", ["tabs"]) == 0.0


def test_no_false_recall_only_scores_no_answer_cases() -> None:
    # A case that has an answer is excluded, not scored.
    assert M.no_false_recall("- (substrate) anything", expect_nothing=False) is None
    # Silence is the right answer when memory holds nothing...
    assert M.no_false_recall("", expect_nothing=True) == 1.0
    # ...and handing back top-k regardless of score is a confident wrong answer.
    assert M.no_false_recall("- (substrate) an unrelated entry", expect_nothing=True) == 0.0
