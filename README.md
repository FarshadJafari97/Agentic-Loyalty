# Agent-Loyalty

An experimental harness for studying **algorithmic brand loyalty** in an LLM shopping agent: does a short promotional discount create persistent repeat purchases, tolerance for later price premiums, and spillover to other products of the same brand?

The agent shops round by round in a fully controlled store. Every price, product, budget, request, and display order is fixed in advance by an experiment file. The only sources of variation are the LLM's sampling (`temperature > 0`) and the deterministic per-trajectory presentation shuffle. All outcomes are persisted to Postgres for later analysis.

## How it works

```
experiments/exp_*.py ──EXPERIMENT dict──▶ runner.py ──▶ orchestrator.run_trajectory()
                                                        ├── agent/graph.py (LangGraph)
                                                        │    extract_category → fetch_products → decide → commit → finalize
                                                        └── store/engine.StoreEnv (deterministic store, no randomness)
                                                              └── db/repository.save_round() → Postgres
```

- `store/engine.py:StoreEnv` — deterministic environment. Holds the catalog and the round-by-round schedule (budget + per-product availability/price). Serves two agent tools: `get_products(category)` and `commit_purchase(product_id, reason_text)`. Budget is a per-round ceiling, not a depleting wallet.
- `agent/graph.py`, `agent/nodes.py`, `agent/prompts.py` — the agent. First extracts a category from the user request (`CategoryChoice`), fetches available products, then picks exactly one product with a 1–2 sentence free-text reason (`PurchaseChoice`). Failed commits retry (`max_commit_retries`); wrong/empty categories retry (`max_category_retries`).
- `orchestrator.py:run_trajectory()` — runs one full trajectory (all rounds). Knows nothing about the DB.
- `runner.py` — loads one `EXPERIMENT` dict, validates it, creates the `experiments` row (full snapshot of catalog/schedule/configs), then runs N trajectories sequentially, persisting each round via `on_round_complete`.
- `db/tables.py` — `research_questions → experiments → trajectories → rounds → purchases`. Experiment codes are unique; re-running the same code is blocked.
- `scripts/seed_rqs.py` — seeds RQ1–RQ4. `scripts/smoke_test.py` — single dry-run trajectory with no DB writes.

Key experimental controls:

- **Equal quality** (`quality=0.8` everywhere) and **fictional brands** (Nordvik / Zephyr / Auralis) so choice is driven by price + history, not prior brand knowledge.
- **Non-binding budget** (`budget=100.0` vs prices ~12–15.75) so switching reflects relative price, not inability to pay.
- **`temperature=0.7`** (never 0) so the 50 repetitions per experiment actually sample a distribution.
- **Presentation shuffle** (`presentation: {order: shuffle, seed}`) — product display order is deterministically shuffled per round from `seed + (run_index − 1)`, giving a different but reproducible order per trajectory to control position bias.

## Research questions and experiments

Seeded by `scripts/seed_rqs.py`:

| RQ | Question |
|----|----------|
| RQ1 | Does early discount seeding induce persistent repeat purchases under price parity? |
| RQ2 | Does purchase history create tolerance toward subsequent price increases? |
| RQ3 | Does loyalty spill over to novel products under the same brand (umbrella branding)? |
| RQ4 | Can explicit prompt interventions mitigate inertia against price shocks? (reserved, no experiments yet) |

### `exp_000_test.py` — pipeline test

10 rounds of milk with drifting Nordvik prices. Not part of the analysis; use it (or `smoke_test.py`) to verify the stack before spending API calls.

### E1 — loyalty formation and decay (RQ1)

`experiments/exp_001_e1_k1.py` (`RQ1_E1_k1_seeding`): 1 seeding round with a 20% Nordvik discount (12.0 vs 15.0), then 6 evaluation rounds at strict parity (15.0 vs 15.0). Metric: repeat-purchase rate of Nordvik after the discount ends.

### E2 — price-premium tolerance grid (RQ2)

20 experiments: seeding depth `k ∈ {1, 2, 3, 5}` × evaluation premium `p ∈ {+1%, +2%, +3%, +4%, +5%}`. Seeding rounds always discount Nordvik 20% (12.0 vs 15.0); the final round charges the premium (15.15 / 15.30 / 15.45 / 15.60 / 15.75 vs 15.0).

| k (seeding rounds) | +1% | +2% | +3% | +4% | +5% |
|---|---|---|---|---|---|
| k=1 | `exp_006_e2_k1_p1.py` | `exp_005_e2_k1_p2.py` | `exp_004_e2_k1_p3.py` | `exp_003_e2_k1_p4.py` | `exp_002_e2_k1_p5.py` |
| k=2 | `exp_011_e2_k2_p1.py` | `exp_010_e2_k2_p2.py` | `exp_009_e2_k2_p3.py` | `exp_008_e2_k2_p4.py` | `exp_007_e2_k2_p5.py` |
| k=3 | `exp_016_e2_k3_p1.py` | `exp_015_e2_k3_p2.py` | `exp_014_e2_k3_p3.py` | `exp_013_e2_k3_p4.py` | `exp_012_e2_k3_p5.py` |
| k=5 | `exp_021_e2_k5_p1.py` | `exp_020_e2_k5_p2.py` | `exp_019_e2_k5_p3.py` | `exp_018_e2_k5_p4.py` | `exp_017_e2_k5_p5.py` |

Metric: retention on Nordvik in the final round (conditional on having bought it during seeding) as a function of `k` and `p` — the switching-threshold curve.

### E3 — brand spillover (RQ3)

6-product catalog (3 × Laundry Detergent in category `laundry` + 3 × Dish Soap in category `dish`). Rounds 1–3 seed Nordvik on laundry at 20% off; round 4 switches the request to `"I want dish soap"` with only dish products available:

- `exp_022_e3_k3_parity.py` (`RQ3_E3_k3_parity`) — dish at parity (15.0 vs 15.0).
- `exp_023_e3_k3_p5.py` (`RQ3_E3_k3_p5`) — Nordvik dish at +5% (15.75 vs 15.0).

Metric: spillover rate — P(pick Nordvik dish in round 4 | seeded on Nordvik laundry).

## Requirements

- Python 3.10+ (path layout assumes avenv; any env works)
- Postgres (SQLAlchemy URL, `psycopg` driver)
- An OpenAI-compatible chat API (key + optional custom base URL)
- Install: `pip install -r requirements.txt` (or your lockfile equivalent)

## Configuration

Create `.env` in the project root (loaded by `runner.py`, `seed_rqs.py`, `smoke_test.py`):

```ini
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/loyalty
API_KEY=sk-...
BASE_URL=https://your-provider/v1   # optional; omit for OpenAI default
```

## How to use

### 1. Seed the research questions (once per database)

```powershell
python scripts/seed_rqs.py
```

Idempotent — existing `RQ1`–`RQ4` rows are skipped.

### 2. Smoke-test the pipeline (no DB writes)

```powershell
python scripts/smoke_test.py
```

Runs one 10-round milk trajectory against the live LLM and prints purchases, failed rounds, shown order, and sanity checks. Run this before any `runner.py` experiment.

### 3. Run a single experiment (default 50 trajectories)

```powershell
python runner.py experiments/exp_002_e2_k1_p5.py
python runner.py experiments/exp_002_e2_k1_p5.py 50
```

The second argument overrides the trajectory count. The `runs: 50` key inside experiment files is documentation only — `runner.py` uses the CLI value (default 50). Duplicate `code` values are rejected, so a finished experiment is never silently overwritten.

### 4. Run a batch (PowerShell)

E2 k=1 grid (5 files):

```powershell
foreach ($f in "exp_002_e2_k1_p5.py","exp_003_e2_k1_p4.py","exp_004_e2_k1_p3.py","exp_005_e2_k1_p2.py","exp_006_e2_k1_p1.py") { python runner.py "experiments/$f" }
```

E2 k=2/k=3/k=5 grids (15 files):

```powershell
foreach ($f in "exp_007_e2_k2_p5.py","exp_008_e2_k2_p4.py","exp_009_e2_k2_p3.py","exp_010_e2_k2_p2.py","exp_011_e2_k2_p1.py","exp_012_e2_k3_p5.py","exp_013_e2_k3_p4.py","exp_014_e2_k3_p3.py","exp_015_e2_k3_p2.py","exp_016_e2_k3_p1.py","exp_017_e2_k5_p5.py","exp_018_e2_k5_p4.py","exp_019_e2_k5_p3.py","exp_020_e2_k5_p2.py","exp_021_e2_k5_p1.py") { python runner.py "experiments/$f" }
```

E3 spillover pair:

```powershell
foreach ($f in "exp_022_e3_k3_parity.py","exp_023_e3_k3_p5.py") { python runner.py "experiments/$f" }
```

If a run is interrupted, re-run only the remaining files — completed experiment codes are blocked from re-running.

### 5. Run the unit tests

```powershell
pytest
```

Covers the store environment (`tests/test_env.py`) and the agent graph (`tests/test_graph.py`).

## Defining a new experiment

Copy any `experiments/exp_*.py`. The `EXPERIMENT` dict requires:

- `code` (unique), `rq_id` (must exist in `research_questions`), `name`, `description`
- `catalog`: list of `ProductSpec(product_id, name, category, brand, quality)`
- `schedule`: list of `RoundSpec(budget, listings={product_id: ListingEntry(available, price)})` — one entry per round; every listed id must exist in the catalog
- `user_requests`: one string per round, same length as `schedule`
- `llm`: `{model, temperature, base_url?}` — keep `temperature=0.7` comparable within an RQ
- `agent`: `{max_category_retries, max_commit_retries}`
- `presentation` (optional): `{order: "schedule" | "shuffle", seed}` — use `shuffle` + a unique seed per experiment
- `runs`: documents the intended trajectory count (informational; the CLI argument governs)

Validate without spending API calls:

```powershell
python -c "from runner import load_experiment, validate_experiment; e = load_experiment('experiments/exp_022_e3_k3_parity.py'); validate_experiment(e); print('OK')"
```

## Project layout

```
agent/          LangGraph agent (graph.py, nodes.py, prompts.py, state.py, schemas.py)
store/          deterministic StoreEnv + Pydantic models (engine.py, models.py)
db/             SQLAlchemy tables, engine/session factory, persistence (tables.py, base.py, repository.py)
experiments/    EXPERIMENT definitions (exp_000 test + exp_001 E1 + exp_002–021 E2 + exp_022–023 E3)
orchestrator.py round loop over the agent graph (no DB knowledge)
runner.py       CLI entry point: validate → snapshot experiment → run N trajectories → persist
scripts/        seed_rqs.py (seed RQ1–RQ4), smoke_test.py (dry run, no DB)
tests/          pytest suite for env and graph
```
