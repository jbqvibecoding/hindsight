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

33 hand-written cases in our own domain (`cases.jsonl`), three in each of the
ten skill categories BEAM uses — the closest published taxonomy to what an
agent memory actually does:

`information_extraction`, `temporal_reasoning`, `multi_session_reasoning`,
`contradiction_resolution`, `event_ordering`, `knowledge_update`,
`summarization`, `abstention`, `preference_following`, `instruction_following`.

Plus `followup_reference`, three anaphoric follow-ups ("and the other one?")
added because every other case asks its question cold, which made the
conversational-rewrite lane invisible to the instrument.

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

## Baseline

`baseline.json`, 33 cases x 3 runs, substrate-only (no semantic brain, no judge):

```
abstention       mean=0.9697 [0.9293, 1.0000]  run_std=0.0000
ordering         mean=1.0000 [1.0000, 1.0000]  run_std=0.0000
recall_hit       mean=0.9394 [0.8889, 0.9798]  run_std=0.0000
```

Per category, which is where the actionable signal lives:

| Category | `recall_hit` | `ordering` |
|---|---|---|
| `event_ordering` | **0.33** | 1.00 |
| `abstention` | 1.00 | 1.00 (0.67 on the abstention metric) |
| everything else | 1.00 | 1.00 |

### What the instrument has actually decided so far

It has been wrong-footed twice and has overruled two plausible changes, which
is the only reason to trust the numbers it now reports.

| Change | Verdict |
|---|---|
| Okapi BM25 replacing term overlap | **0.0000 on every metric, twice.** Inspecting the retrieved context showed why: every failing case failed on *zero term overlap*, not bad ranking, and no lexical scorer can rank an entry it never matched. Kept because it is strictly better and free, not because it was measured to help. |
| Distractor corpus | Not a system change — an instrument fix. Cases seeded one to three entries against a limit of eight, so everything was returned and relevance never bound. |
| Recency ordering + conflict rule | `ordering` 0.8000 -> 0.9333; `knowledge_update` 0.00 -> 1.00. Also surfaced real nondeterminism (`run_std` 0.0000 -> 0.0192) from same-millisecond entry ids, fixed by keying recency on append position. |
| Two-lane conversational rewrite | `recall_hit` 0.7667 -> 0.9394, `ordering` -> 1.0000, both non-overlapping. Needed three new follow-up cases first: every existing case was a cold question, so the instrument could not see it — the same gap that hid BM25. |

Two metric bugs were also caught by inspecting cases rather than trusting
scores: substring anchors reported hits the system never made ("Wen" inside
"when"), and comparing character offsets failed a correctly ordered context
where the current entry mentions the superseded holder first ("Ravi handed
on-call over to Mira").

`run_std` is `0.0000` because the substrate path is fully deterministic, so any
delta there is signal. That stops being true the moment an LLM or a semantic
index enters the path — which is exactly when the number starts earning its
keep.
