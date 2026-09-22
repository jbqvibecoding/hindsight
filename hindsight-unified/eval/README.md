# Memory eval harness

Measures whether a change to recall made memory **better or worse**. Before
this existed there was no way to tell, which meant every quality change was an
act of faith.

```bash
python -m eval.harness --runs 3 --no-judge          # deterministic only
python -m eval.harness --runs 3                     # + LLM judge, if configured
python -m eval.harness --runs 3 --write-baseline    # record a new baseline
python -m eval.harness --runs 3 --gate              # exit 1 on a regression
python -m eval.harness --only cr-01 ku-02           # iterate on failures
```

## Shape

Four stages that communicate **only through JSON files on disk**, plus a thin
runner — load cases → answer them → score them → aggregate. Artifacts land in
`eval/results/<timestamp>/`, with the resolved config dumped beside them so a
number can be traced back to what produced it.

## The case set

30 hand-written cases in our own domain (`cases.jsonl`), three in each of the
ten skill categories BEAM uses — the closest published taxonomy to what an
agent memory actually does:

`information_extraction`, `temporal_reasoning`, `multi_session_reasoning`,
`contradiction_resolution`, `event_ordering`, `knowledge_update`,
`summarization`, `abstention`, `preference_following`, `instruction_following`.

Hand-written beats a public benchmark here. HotPotQA measures multi-hop
Wikipedia QA, not whether an agent remembers that you moved off Slack.

## Metrics

All three score **retrieval**, because retrieval is what the sidecar produces —
it returns a context block, not an answer.

| Metric | Measures |
|---|---|
| `recall_hit` | Fraction of a case's required anchors present in the retrieved context. Word-boundary matched, so "Wen" does not match inside "when" and "100" does not match inside "1000". |
| `ordering` | For a changed-mind case, whether the **current** fact precedes the superseded one. Superseded facts are tagged rather than deleted, so absence would be the wrong assertion — what matters is which one leads. |
| `abstention` | Nothing forbidden leaked in. **Never read alone**: a system that retrieves nothing scores 1.0 here. It is the precision half of a pair with `recall_hit`. |
| `judged` | Optional LLM judge, using cognee's `direct_llm_eval_system.txt` near-verbatim for its anti-bias clauses (compare by meaning, do not penalise length, extra detail is fine). A judge outage yields `None`, never `0.0` — missing data must not average in as a wrong answer. |

Exact match and token F1 are deliberately **absent**: comparing a multi-line
verbatim context against a short golden answer makes EM structurally zero and
F1 a length artifact. A metric that cannot move is worse than no metric. They
belong with a generation step, which lives in the agent.

## Reading the output honestly

Two different variances, and they answer different questions:

- `ci_lower`/`ci_upper` — spread across **cases within one run** (bootstrap,
  2000 resamples, seeded so the interval itself is reproducible).
- `run_std` — spread across **repeated identical runs**. **If `run_std` exceeds
  the delta you are looking at, the change did nothing.**

So: `--runs` defaults to 3 because one run is not a measurement, and
`describe_delta` refuses a verdict when intervals overlap or the delta sits
inside the noise floor. On the current substrate-only path `run_std` is
`0.0000` — the keyword lane is fully deterministic — so any delta there is
signal. That will stop being true the moment an LLM or a semantic index enters
the path, which is exactly when the number starts earning its keep.

The gate compares against the baseline's `ci_lower`, not its mean, so a drop
has to clear the baseline's own spread before it counts as a regression.

## Baseline as of the workstream-B commits

`baseline.json`, 30 cases × 3 runs, substrate-only (no semantic brain, no judge):

```
abstention       mean=0.9667 [0.9222, 1.0000]  run_std=0.0000
ordering         mean=0.8000 [0.7111, 0.8778]  run_std=0.0000
recall_hit       mean=0.7667 [0.6778, 0.8556]  run_std=0.0000
```

The category breakdown is the useful part, and it already names the known
defects rather than averaging them away:

| Category | `recall_hit` | `ordering` | What it says |
|---|---|---|---|
| `contradiction_resolution` | 0.33 | **0.00** | The changed-mind defect. Asked "how do I indent?", keyword overlap matches the stale turn (which contains "indent") and misses the correction entirely. |
| `knowledge_update` | 1.00 | **0.00** | Both facts retrieved, stale one leading. |
| `summarization` | 0.33 | 1.00 | Conclusions spread across turns are not joined. |
| `event_ordering` | 0.33 | 1.00 | Causal chains phrased once are missed by term overlap. |
| `instruction_following` | 0.67 | 1.00 | |
| `abstention` | 1.00 | 1.00 | 0.67 on the abstention metric — one case leaks. |

`ordering = 0.00` on both changed-mind categories is workstream A4/D1's target;
the `recall_hit` gaps are what A1 (conversational rewrite) and A2 (BM25) are
for. Re-run with `--gate` after each and compare, rather than assuming.
