# hindsight-unified

One memory mechanism for the Hermes agent, fusing the strengths of six memory
systems behind a single supervised sidecar.

| Contributor | Layer it owns | How it's used |
|---|---|---|
| **Hindsight** (`hindsight-api-slim`) | Retrieval/storage brain: L1 fact/entity/temporal-graph extraction (`retain_async`), 4-strategy RRF recall + rerank + MMR (`recall_async`), mental models, reflect (L3) | in-process `MemoryEngine`, optional extra |
| **EverOS** | Markdown-as-truth durability: crash-recovery reconciliation of the verbatim substrate's index from md logs | adapter (lean mode needs no dep) |
| **mempalace** | Verbatim / 100%-recall discipline, AAAK index-triage cards, embedder-identity mismatch guard | adapter (native `Dialect` when installed) |
| **OpenViking** | Tiered L1/L2 token-budgeted context assembly + observable retrieval trajectory | adapter (pure-Python tiering built in) |
| **MemOS** | MemCube-style portable export envelope for backup/transfer | adapter (native `GeneralMemCube` when installed) |
| **tencentdb-agent-memory** | The L0→L3 pipeline shape, the sidecar + thin-provider integration pattern, and the reliability engineering (ported to the Hermes plugin) | design + ported code |

## Design invariant

**Markdown is the source of truth; the semantic index is a rebuildable
derivative.** Every turn is appended verbatim (never summarized) to per-day
markdown logs with fsync'd writes. The Hindsight brain indexes *from* those
turns; if the index (or the whole brain) is lost or unavailable, capture and
keyword recall keep working and the index can be rebuilt from md.

There is exactly **one** semantic index (Hindsight's). No second vector store
is ever created.

## Architecture

```
POST /capture ──► L0 verbatim md substrate (truth, always on)
                    └─► L1 Hindsight retain_async (facts/entities/graph)
POST /recall  ──► adapters recall_enrich (brain N×4-way + substrate keyword)
                    └─► RRF fuse ─► AAAK triage cards ─► tiered budgeted payload
POST /session/end ─► L2 consolidate (EverOS reconcile, brain mental models)
POST /reflect ──► L3 persona synthesis (brain reflect; substrate fallback)
```

The HTTP layer is stdlib `http.server` — the sidecar itself has **zero
dependencies**. Heavy engines load lazily and degrade gracefully; `/health`
reports `ok` (brain up) or `degraded` (substrate-only), both usable.

## Run

```bash
python -m hindsight_unified.server           # zero-dep, substrate-only
pip install -e ".[hindsight]"                # + the Hindsight brain
pip install -e ".[all]"                      # + all six engines
```

Env: `UNIFIED_MEMORY_GATEWAY_HOST/PORT` (default `127.0.0.1:8766`),
`UNIFIED_MEMORY_HOME` (default `$HERMES_HOME/unified`),
`UNIFIED_MEMORY_ENABLE_{HINDSIGHT,EVEROS,MEMPALACE,OPENVIKING,MEMOS}`.

## Endpoints

`GET /health` · `POST /recall` · `POST /capture` · `POST /search/memories` ·
`POST /search/conversations` · `POST /session/end` · `POST /reflect` ·
`POST /seed` · `POST /forget/preview` · `POST /forget` · `POST /forget/restore`

Two fields are easy to conflate and mean different things. `bank` is the
tenant — one per Hermes identity, so memory accumulates across conversations.
`session_key` is the conversation a turn belongs to; recall uses it to fold
*this* conversation's last turns into a follow-up query. (A request without
`bank` falls back to `session_key` for the tenant, for older clients.)

`/capture` accepts an `idempotency_key`. A retry carrying the key of a turn
already written returns the original `entry_id` with `duplicate: true` and is
not fanned out to the semantic brain again, so an at-least-once client gets
exactly-once storage.

Forgetting is a recall filter, not erasure. `/forget/preview` takes a query and
returns candidates without changing anything; `/forget` takes explicit
`entry_ids` only, never a query; `/forget/restore` undoes it. A forgotten entry
is excluded from every recall path — search, the follow-up lane, triggers, and
any lesson drawn from it — and survives `rederive`. Its text stays in the
markdown log (`still_on_disk` in every response), and the semantic brain's own
extracted facts are not touched (`semantic_brain_unaffected`).

The surface mirrors the tencentdb Gateway so the Hermes thin provider
(`hermes-agent/plugins/memory/unified/`) is a near-copy of that proven plugin.

## Consume from Hermes

Set `memory.provider: unified` in the Hermes config. The provider supervises
the sidecar (auto-launch, health checks, circuit breaker, watchdog resurrect)
and exposes `unified_memory_search`, `unified_conversation_search`,
`unified_memory_reflect` and `unified_memory_forget` tools to the model.

## Tests

```bash
uv run --no-project --with pytest -- python -m pytest tests/ -q
```
