"""Cost ledger tests (workstream E4).

All three tested properties are about the ledger staying out of the way:
off by default, unable to fail a memory write, and honest about what it does
not know.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hindsight_unified import ledger


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_disabled_by_default_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ledger.ENV_PATH, raising=False)
    assert ledger.enabled() is False
    ledger.record(call_site="x", model="m", latency_s=0.1, prompt_chars=1, completion_chars=1)
    assert list(tmp_path.iterdir()) == []


def test_one_row_per_call_with_the_reported_usage(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "cost.jsonl"
    monkeypatch.setenv(ledger.ENV_PATH, str(path))
    ledger.record(
        call_site="triggers.generate",
        model="m",
        latency_s=1.25,
        prompt_chars=100,
        completion_chars=20,
        usage={"input_tokens": 90, "output_tokens": 15},
    )
    (row,) = _rows(path)
    assert row["call_site"] == "triggers.generate"
    assert (row["prompt_tokens"], row["completion_tokens"]) == (90, 15)
    assert row["token_source"] == "api"


def test_absent_usage_is_null_never_zero(tmp_path: Path, monkeypatch) -> None:
    """A fabricated token count is indistinguishable from a real one on disk.

    The same rule the rest of this system runs on: missing is not zero. There
    is no local estimator here precisely so a row can never imply a
    measurement that was never taken.
    """
    path = tmp_path / "cost.jsonl"
    monkeypatch.setenv(ledger.ENV_PATH, str(path))
    ledger.record(call_site="x", model="m", latency_s=0.1, prompt_chars=10, completion_chars=2)
    (row,) = _rows(path)
    assert row["prompt_tokens"] is None
    assert row["completion_tokens"] is None
    assert row["token_source"] is None
    # Chars are always known, so they are always recorded.
    assert row["prompt_chars"] == 10


@pytest.mark.parametrize("usage", [{"input_tokens": "ninety"}, {"input_tokens": True}, {}])
def test_unusable_usage_fields_are_null(tmp_path: Path, monkeypatch, usage: dict) -> None:
    path = tmp_path / "cost.jsonl"
    monkeypatch.setenv(ledger.ENV_PATH, str(path))
    ledger.record(
        call_site="x", model="m", latency_s=0.1, prompt_chars=1, completion_chars=1, usage=usage
    )
    assert _rows(path)[0]["prompt_tokens"] is None


def test_an_unwritable_path_never_raises(tmp_path: Path, monkeypatch) -> None:
    """Accounting must never be able to fail the memory write it is measuring."""
    monkeypatch.setenv(ledger.ENV_PATH, str(tmp_path / "a-file" / "nested" / "cost.jsonl"))
    (tmp_path / "a-file").write_text("not a directory", encoding="utf-8")
    ledger.record(call_site="x", model="m", latency_s=0.1, prompt_chars=1, completion_chars=1)


def test_rows_accumulate_so_they_can_be_aggregated_offline(tmp_path: Path, monkeypatch) -> None:
    """The reason to log per call: coarser can be derived, finer cannot."""
    path = tmp_path / "cost.jsonl"
    monkeypatch.setenv(ledger.ENV_PATH, str(path))
    for site in ("triggers.generate", "distill.curator", "distill.judge"):
        ledger.record(call_site=site, model="m", latency_s=0.1, prompt_chars=1, completion_chars=1)
    assert [r["call_site"] for r in _rows(path)] == [
        "triggers.generate",
        "distill.curator",
        "distill.judge",
    ]
