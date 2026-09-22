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

42 hand-written cases in our own domain (`cases.jsonl`), three in each of
fourteen categories. Ten are the skill categories BEAM uses — the closest
published taxonomy to what an agent memory actually does:

`information_extraction`, `temporal_reasoning`, `multi_session_reasoning`,
`contradiction_resolution`, `event_ordering`, `knowledge_update`,
`summarization`, `abstention`, `preference_following`, `instruction_following`.

The other four were added here, and **every one of them was added because the
system failed it at the time of writing.** That is the rule: a category that
passes on arrival measures nothing.

| Category | Added because |
|---|---|
| `followup_reference` | Every other case asks its question cold, which made the conversational-rewrite lane invisible to the instrument — the same blind spot that had already hidden BM25. |
| `vocabulary_gap` | A question and the entry answering it can share **no term at all** ("what cleans up storage we no longer need?" vs "the nightly job that trims old blobs is called reaper"). No expansion derivable from the entry text can invent the synonym, so this is the case no deterministic lane can pass. |
| `predecessor_query` | "Which region did we use *before* the move?" The question's terms match the *current* entry; the superseded one shares nothing with the question, so it is never retrieved. |
| `no_answer` | A question the bank genuinely cannot answer, over a distractor-only corpus. Top-k returns its k regardless of score, so this asks whether the system can decline. |

Hand-written beats a public benchmark here. HotPotQA measures multi-hop
Wikipedia QA, not whether an agent remembers that you moved off Slack.

## Metrics

All three score **retrieval**, because retrieval is what the sidecar produces —
it returns a context block, not an answer.

| Metric | Measures |
|---|---|
| `recall_hit` | Fraction of a case's required anchors present in the retrieved context. Word-boundary matched, so "Wen" does not match inside "when" and "100" does not match inside "1000"; and inflection-matched, so `drain` finds "drained" and `migration` finds "migrations". |
| `ordering` | For a changed-mind case, whether the **current** fact precedes the superseded one, compared by **entry block position**, not character offset. Superseded facts are tagged rather than deleted, so absence would be the wrong assertion — what matters is which one leads. |
| `abstention` | Nothing forbidden leaked in. **Never read alone**: a system that retrieves nothing scores 1.0 here. It is the precision half of a pair with `recall_hit`. |
| `judged` | Optional LLM judge, using cognee's `direct_llm_eval_system.txt` near-verbatim for its anti-bias clauses (compare by meaning, do not penalise length, extra detail is fine). A judge outage yields `None`, never `0.0` — missing data must not average in as a wrong answer. |
| `no_false_recall` | **Implemented but deliberately unscored.** See below. |

The inflections are an **explicit enumeration**, not a suffix wildcard. A
wildcard was tried first and got `cache`→`cacheing`, which is not a word; the
list therefore includes the silent-e drop. Over-stemming manufactures hits the
system never made, which is the exact failure the word-boundary fix already had
to undo once — so this errs narrow.

### Why `no_false_recall` is not scored

The metric asks a fair question — for a question memory cannot answer, the
context should not parade distractors as though they were hits — and there is
no honest way to compute it here. Three discriminators were implemented and
measured, and all three failed:

1. **Raw-lane emptiness** — the rewrite lane fills in behind it, so the signal
   never appears.
2. **Fused rank score** — post-fusion scores are ranks, so they carry no
   absolute magnitude to threshold.
3. **Absolute BM25** — the honest attempt, and the one that settles it: a
   no-answer question's top hit scored **20.31** while a genuine case's scored
   **18.58**. Any threshold separating them would fail a real case.

So the metric ships computed-but-unscored, with the numbers recorded. A metric
pinned at 0.00 is worse than an absent one: it looks like a standing defect and
invites someone to "fix" the system to satisfy an instrument that cannot
measure it. The relevance floor needs a semantic signal, not a lexical one.

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

`baseline.json`, 42 cases x 3 runs, substrate-only (no semantic brain, no LLM
lane, no judge):

```
abstention       mean=1.0000 [1.0000, 1.0000]  n=126  run_std=0.0000
ordering         mean=1.0000 [1.0000, 1.0000]  n=126  run_std=0.0000
recall_hit       mean=0.9231 [0.8718, 0.9658]  n=117  run_std=0.0000
```

Per category, which is where the actionable signal lives:

| Category | `recall_hit` | Why |
|---|---|---|
| `vocabulary_gap` | **0.33** | Zero term overlap between question and answer. Unreachable by any lexical lane — this is what the LLM summary lane exists for. |
| `predecessor_query` | **0.67** | The failing case's subject ("region") appears in *neither* entry, so no deterministic link can be derived. |
| everything else (12) | 1.00 | — |

`recall_hit` counts 117, not 126, because the `no_answer` cases have no anchors
to find — a case that should return nothing cannot contribute to a recall mean
without corrupting it.

**This is the deterministic ceiling.** Both remaining gaps need semantics —
synonym and domain knowledge ("cleans up storage" → "trims old blobs",
"database connections" → "pgbouncer") — not better ranking. That is the whole
reason the LLM lanes were built, and the reason they ship labelled unvalidated
rather than claimed.

### What the instrument has actually decided so far

It has been wrong-footed twice and has overruled two plausible changes, which
is the only reason to trust the numbers it now reports.

| Change | Verdict |
|---|---|
| Okapi BM25 replacing term overlap | **0.0000 on every metric, twice.** Inspecting the retrieved context showed why: every failing case failed on *zero term overlap*, not bad ranking, and no lexical scorer can rank an entry it never matched. Kept because it is strictly better and free, not because it was measured to help. |
| Distractor corpus | Not a system change — an instrument fix. Cases seeded one to three entries against a limit of eight, so everything was returned and relevance never bound. |
| Recency ordering + conflict rule | `ordering` 0.8000 -> 0.9333; `knowledge_update` 0.00 -> 1.00. Also surfaced real nondeterminism (`run_std` 0.0000 -> 0.0192) from same-millisecond entry ids, fixed by keying recency on append position. |
| Two-lane conversational rewrite | `recall_hit` 0.7667 -> 0.9394, `ordering` -> 1.0000, both non-overlapping. Needed three new follow-up cases first: every existing case was a cold question, so the instrument could not see it — the same gap that hid BM25. |
| Instrument hardening before D | **Not a system change.** All three remaining "failures" turned out to be instrument defects: two cases retrieved the right entry and *ranked it first* but scored 0.0 on `drain`/`drained` and `migration`/`migrations`, and the abstention case forbade a phrase that is the **user's own wording in the truthful entry**. Fixing them saturated the ruler at ~1.00 across the board, which is why four harder categories were added before any D mechanism was built. |
| Harder case set | `recall_hit` 0.9394 -> 0.9231. A *lower* number from a *better* instrument — the four new categories are cases the system genuinely fails, which is the only kind worth adding. |
| D4 retrieval-summary lane | **Mechanism verified, prompt not.** A hand-written summary carrying vocabulary the entry lacks takes `vocabulary_gap` 0.33 -> 1.00 and dereferences back to verbatim text. What that demonstrates is the *lane* — a summary hit reaching an entry the question could not — not the quality of the prompt's output, which needs a real model. |
| D2 two-stage distillation | **+0.0000, and correct.** With no key configured the lane is inert, so an unchanged eval is the expected result and confirms the stage gates off cleanly rather than perturbing recall. |
| D1 supersession | **Not shipped.** Only one case (`pd-01`) fails, and its subject ("region") appears in neither entry, so deterministic linking cannot reach it. A wrong supersession link hides a true fact, so no link beats a guessed one. |
| D3 confidence + usability gates | **Not shipped.** Cheap and deterministic, but nothing in the system emits helpful/harmful signals, so the fields would have no producer — recreating precisely the write-only trap this whole exercise was organised around avoiding. |

Three metric bugs were also caught by inspecting cases rather than trusting
scores: substring anchors reported hits the system never made ("Wen" inside
"when"); comparing character offsets failed a correctly ordered context where
the current entry mentions the superseded holder first ("Ravi handed on-call
over to Mira"); and `run_std` went 0.0000 -> 0.0192 after the recency change,
exposing real nondeterminism from same-millisecond entry ids.

And two new cases initially **passed for the wrong reason**, which is the
failure mode hardest to notice: one had both facts inside the two-turn rewrite
window, so the rewrite query contained the answer verbatim and retrieved it by
self-similarity; the other used values ("us-east-1" / "eu-central-1") sharing
the token "1". Fixing both made `pd-01` fail honestly — which is how it came to
be the case that D1 is measured against.

`run_std` is `0.0000` because the substrate path is fully deterministic, so any
delta there is signal. That stops being true the moment an LLM or a semantic
index enters the path — which is exactly when the number starts earning its
keep.
