# Agent-Loyalty — Project Hand-off Document

**Repo:** `Agent-Loyalty` (local: `C:\Users\farsh\OneDrive\Desktop\Agent-Loyalty`)
**Stack:** Python 3.10+, LangGraph + LangChain-OpenAI, SQLAlchemy + Postgres (`psycopg`), Pydantic, pytest, dotenv
**Entry points:** `runner.py`, `orchestrator.py`, `scripts/seed_rqs.py`, `scripts/smoke_test.py`
**Date:** 2026-10-03

---

## 1. What this project is

Experimental harness for studying **algorithmic brand loyalty** in an LLM shopping agent.

Core question: does a short promotional discount create (a) persistent repeat purchases at parity, (b) tolerance for later price premiums, (c) spillover to other products of the same brand?

The agent shops round-by-round in a fully controlled store. Every price, product, budget, request, and display order is fixed in advance by an experiment file. Only sources of variation:

1. LLM sampling (`temperature > 0`, standard `0.7`)
2. Deterministic per-trajectory presentation shuffle (`seed + run_index - 1`)

All outcomes persisted to Postgres for later analysis.

Flow:

```
experiments/exp_*.py ──EXPERIMENT dict──▶ runner.py ──▶ orchestrator.run_trajectory()
                                                        ├── agent/graph.py (LangGraph)
                                                        │    extract_category → fetch_products → decide → commit → finalize
                                                        └── store/engine.StoreEnv (deterministic, no randomness)
                                                              └── db/repository.save_round() → Postgres
```

---

## 2. Research questions (seeded by `scripts/seed_rqs.py`)

Seed script is idempotent — existing `RQ1`–`RQ3` rows skipped. Must run once per database: `python scripts/seed_rqs.py`.

| RQ | code | title | description |
|----|------|-------|-------------|
| RQ1 | `RQ1` | Does early discount seeding induce persistent repeat purchases under price parity? | Test whether promotional discounts across varying seeding durations (k rounds) create persistent state dependence and repeat purchases once price parity is restored. |
| RQ2 | `RQ2` | Does purchase history create tolerance toward subsequent price increases? | Evaluate the agent's price elasticity and switching threshold when the previously purchased brand imposes a price premium over equal-quality competitors. |
| RQ3 | `RQ3` | Does algorithmic loyalty spill over to novel products under the same brand? | Test whether prior repeat purchases in one product category create an umbrella brand effect, increasing the selection of an unexperienced product from the same brand. |

---

## 3. Design controls (apply to all RQ1–RQ3 experiments except `exp_000`)

- **Equal quality:** `quality=0.8` everywhere.
- **Fictional brands:** `Nordvik` (treated/discounted) / `Zephyr` / `Auralis` — no prior brand knowledge.
- **Non-binding budget:** `budget=100.0` vs prices ~12–15.75, so switching = relative price, not inability to pay.
- **Temperature:** `0.7` (never 0) so N trajectories sample a distribution. Only pilot `exp_000` and `smoke_test.py` differ (1.0).
- **Presentation shuffle:** `presentation: {order: shuffle, seed}` — `StoreEnv.get_products()` shuffles with `random.Random(f"{seed}:{round}")`. Trajectory seed = base seed + (`run_index`-1). Reproducible, controls position bias. Only `exp_000` uses fixed `schedule` order.
- **Agent retries:** all experiments `max_category_retries=2, max_commit_retries=3`.
- **Prices:** seeding = Nordvik `12.0` vs `15.0` (20% off). Evaluation premiums: `15.15 (+1%) / 15.30 (+2%) / 15.45 (+3%) / 15.60 (+4%) / 15.75 (+5%)` vs `15.0`.

---

## 4. Architecture — component details

### 4.1 `store/models.py` (104 lines) — Pydantic specs

- `ProductSpec(product_id, name, category, brand, quality=0.0, attributes={})` — static catalog.
- `ListingEntry(available: 0/1, price: float)` — per-round availability+price.
- `RoundSpec(budget: float, listings: dict[product_id, ListingEntry])` — one per round.
- `ProductView(product_id, name, category, brand, price, quality, attributes)` — per-round view served to agent.
- `PurchaseRecord(round, product_id, product_name, category, brand, price_paid, budget, reason_text, reason_code=None, reason_note=None)` — `reason_code/note` reserved for later classifier, always `None` at collection.
- `FailedRound(round, budget, reason)`, `HistoryEntry(round, status=committed|failed, product_id|None, category|None, brand|None, product|None, price_paid|None, reason_text|None)`.
- `Event` enum: `ROUND_STARTED/ROUND_CLOSED/PRODUCTS_SHOWN/PURCHASE/ROUND_FAILED`; `LogEntry(seq, round, event, payload)`.

### 4.2 `store/engine.py:StoreEnv` (306 lines) — deterministic env, no randomness except shuffle

- `__init__(catalog, schedule, *, order="schedule"|"shuffle", seed)`: validates order, shuffle requires int seed, budget>=0, all listing pids in catalog.
- Lifecycle: `begin_round()` (loads `budget` from current spec) → agent acts → `close_round()` (advances `_round+=1`). `finished = _round > max_rounds`. `record_failed_round(reason)` if no commit.
- `history` merges purchases + failed rounds sorted by round; `history_for_prompt()` = `model_dump()` list.
- Tools:
  - `get_products(category) -> list[ProductView]`: requires open round; filters by category + `available==1`; shuffle if configured; logs `PRODUCTS_SHOWN`, caches `shown_orders[round]`.
  - `validate_purchase(product_id)`: checks `round_not_open / unknown_product / not_available_this_round / over_budget`.
  - `commit_purchase(product_id, reason_text) -> CommitResult`: validates, logs `PURCHASE`, appends `PurchaseRecord`. Budget is per-round ceiling, NOT depleted.
- Note: lines 272–300 contain unreachable legacy code after `return` referencing `reason_code=="8"` — dead, safe to delete.

### 4.3 Agent — `agent/graph.py` (68 lines), `agent/nodes.py` (105 lines), `agent/prompts.py` (65 lines), `agent/state.py` (29 lines), `agent/schemas.py` (19 lines)

`build_graph(env, llm, *, allowed_categories, max_category_retries=2, max_commit_retries=3)`:

```
ENTRY → extract_category → fetch_products → decide → commit → finalize → END
              ↑______________| (if products empty & retries left)
                         decide ←──┘ (if commit failed & retries left)
```

- `extract_category`: `llm.with_structured_output(CategoryChoice).invoke([category_system_prompt, category_user_prompt])`. System: "pick exactly ONE category from allowed list… still pick closest". Does not increment retry (increment happens in fetch on empty).
- `fetch_products`: calls `env.get_products(category)`. Empty → `category_retries+1`, `last_category_error="category 'X' returned no products…"`. Else returns product dicts.
- `decide`: `llm.with_structured_output(PurchaseChoice).invoke([purchase_system_prompt, purchase_user_prompt])`. System: "Pick exactly ONE product… using your own judgment… 1–2 sentence reason… JSON {product_id, reason_text}". User message includes `Budget this round`, full JSON `Purchase history so far` (or `(empty)`), full JSON `Available products this round`, plus last commit error if retrying.
- `commit`: calls `env.commit_purchase(...)`. Fail → `commit_retries+1`, `last_commit_error="commit failed: {reason}. Pick a product that is available…"`.
- `finalize`: if not `committed`, calls `env.record_failed_round(reason)` with last error or `"agent_failed"`, sets `status=failed`.
- `AgentState(TypedDict)`: inputs `user_request, budget, history, allowed_categories`; working `category, products, chosen_product_id, chosen_reason_text`; counters `category_retries, commit_retries`; errors `last_category_error, last_commit_error`; final `status, failure_reason`.
- Schemas: `CategoryChoice(category: str)`, `PurchaseChoice(product_id: str, reason_text: str)`.

### 4.4 `orchestrator.py:run_trajectory()` (110 lines) — no DB knowledge

- `run_trajectory(env, llm, *, user_requests, allowed_categories, max_category_retries, max_commit_retries, on_round_complete=None)`: validates `len(user_requests)==env.max_rounds`; builds graph once; loops `begin_round → graph.invoke(initial_state) → close_round → on_round_complete(env, round_number)`.
- `initial_state` per round: `{user_request, budget: env.budget, history: [h.model_dump()], allowed_categories, products:[], retries=0, status=in_progress}`.
- `run_single_round(env, graph, *, user_request, allowed_categories)` — single-round helper.

### 4.5 `runner.py` (303 lines) — CLI, validation, persistence

- `load_experiment(path)`: imports `EXPERIMENT` dict via `importlib`.
- `validate_experiment(exp)`: required keys `code, rq_id, name, catalog, schedule, user_requests, llm, agent`; schedule/request length match; unique product_ids; all listing pids in catalog; llm has `model`; agent has both retry keys; presentation `order in (schedule, shuffle)`, shuffle requires int `seed`.
- `build_llm(cfg)`: `ChatOpenAI(model, base_url=cfg.base_url or env BASE_URL, api_key=env API_KEY, temperature)`. Requires `API_KEY`.
- `run_one_trajectory(session, traj_id, exp, llm, run_index)`: builds `StoreEnv(catalog, schedule, order, seed=base_seed+run_index-1)`; `on_round_complete` reads budget from schedule (env already advanced) and calls `repo.save_round + commit`.
- `run_all(path, n=50)`: validate → require `DATABASE_URL` → `create_all` → `build_llm` → reject duplicate `code` → `create_experiment` (snapshots catalog/schedule/requests/allowed_categories/llm/agent+presentation) → loop N: `start_trajectory → run_one_trajectory → finish_trajectory(finished|failed+error)`; prints progress; `runs` key in file is documentation only — CLI arg governs (default 50).
- Usage: `python runner.py experiments/exp_002_e2_k1_p5.py [n]`; duplicate code blocked; to re-run delete rows in order `purchases → rounds → trajectories → experiments` filtered by `code`. Failed trajectories normal (malformed JSON); analyze only `status='finished'`.

### 4.6 `db/` — Postgres-only (`JSONB`), no cascades/relationships

`db/base.py`: `Base(DeclarativeBase)`, `make_engine(url)`, `make_session_factory(engine, expire_on_commit=False)`.

`db/tables.py` (88 lines):

- `research_questions`: `rq_id PK, code(32) unique, title, description?, created_at`.
- `experiments`: `experiment_id PK, rq_id FK, code(64) unique, name, description?, catalog_json/schedule_json/user_requests/allowed_categories/llm_config/agent_config (JSONB snapshots), created_at`.
- `trajectories`: `trajectory_id PK, experiment_id FK, run_index, status(running/finished/failed), started_at, finished_at?, error_message?`, unique `(experiment_id, run_index)`.
- `rounds`: `round_id PK, trajectory_id FK, round_number, budget, status(committed/failed), failure_reason?`, unique `(trajectory_id, round_number)`.
- `purchases`: `purchase_id PK, round_id FK unique (1:1 with rounds), product_id, product_name, category, brand, price_paid, reason_text, reason_code(8)?, reason_note?, classified_at?, created_at`.

`db/repository.py` (116 lines): `create_research_question → rq_id`; `create_experiment(catalog/schedule via model_dump) → experiment_id`; `start_trajectory(status=running) → traj_id`; `finish_trajectory(status, error_message, finished_at=utcnow)`; `save_round(trajectory_id, env, round_number, budget) → round_id` (finds purchase/failed in env, creates `Round` + optional `Purchase` with `reason_code/note=NULL`); `get_experiment_by_code → row|None`. No commit inside except caller; no list/delete helpers.

### 4.7 `scripts/` and `tests/`

- `scripts/seed_rqs.py` (67 lines): `create_all` + idempotent insert of RQ1–RQ3 (see §2).
- `scripts/smoke_test.py` (220 lines): dry run, no DB. Hardcoded `ChatOpenAI(model="deepseek-v4.1-flash", temperature=1.0)`, 3× Milk/dairy catalog, 10-round schedule (R1–5 parity 15.0; R6–10 Nordvik 16→20), `shuffle seed 412`, prints purchases/failed/shown-order/price table, asserts `purchases+failed==10`, non-empty `reason_text`, `reason_code/note is None`, disjoint rounds.
- `scripts/Untitled-1.ipynb`: scratch notebook, not part of pipeline.
- `tests/conftest.py`: `catalog` (Butter/dairy/A, Milk/dairy/B, Soap/laundry/A, Yogurt/dairy/A), 3-round `schedule`, `env`.
- `tests/test_env.py` (280 lines, 27 tests): StoreEnv only — lifecycle, filtering, pricing, commit errors, failed-round guards, history, event log, schedule/shuffle determinism.
- `tests/test_graph.py` (172 lines, 6 tests): graph with `FakeStructuredLLM` — happy path, category retry/success, category exhausted, commit retry/success, commit exhausted, history passed to decide. Run: `pytest`.
- Not covered: `db/`, `scripts/`, `orchestrator.py`, real-LLM integration.

---

## 5. Experiment catalogue — all 36 files

Global: `budget=100.0`, `quality=0.8`, brands Nordvik/Zephyr/Auralis, `temp=0.7` (except `exp_000`: 1.0), retries `2/3`.

### 5.1 `exp_000_test.py` — pipeline pilot (not for analysis)

- `code=RQ1_Test_10rounds`, RQ1, `gpt-5.6-luna`, no `runs` key.
- Catalog: 3× Milk/dairy (`m_nordvik/m_zephyr/m_auralis`).
- Schedule 10R drifting: R1–3 `14/15/18`, R4 `15/15/18`, R5 `16/15/18`, R6 `17/15/18`, R7 `18/15/18`, R8 `19/15/18`, R9 `20/15/18`, R10 `21/16/14`.
- Requests `["I want milk"]*10`, presentation fixed `schedule` (no shuffle).

### 5.2 E0 — no-history baselines (RQ1)

Same 3-brand Laundry Detergent/cleaning catalog (`p_nordvik/p_zephyr/p_auralis`), strict parity `15.0/15.0/15.0`, no discount.

| File | code | Rounds | Runs | Seed | Purpose |
|------|------|--------|------|------|---------|
| `exp_024_e0_control.py` | `RQ1_E0_control` | 1R parity | 120 | 100 | Single-shot no-history baseline; expect ~⅓ each brand |
| `exp_034_e0_control_gemini.py` | `RQ1_E0_control_gemini` | 1R parity, `gemini-3.5-flash-lite` | 120 | 232 | 1:1 replica of `exp_024` on gemini |
| `exp_035_e0_control_deepseek.py` | `RQ1_E0_control_deepseek` | 1R parity, `deepseek-v4.1-flash` | 120 | 233 | 1:1 replica of `exp_024` on deepseek |
| `exp_025_e0_control_4r.py` | `RQ1_E0_control_4r` | 4R all parity | 120 | 102 | R1 = baseline; R2–4 = repeat dynamics without price signal |

```powershell
python runner.py experiments/exp_024_e0_control.py 120
python runner.py experiments/exp_025_e0_control_4r.py 120
python runner.py experiments/exp_034_e0_control_gemini.py 120
python runner.py experiments/exp_035_e0_control_deepseek.py 120
```

### 5.3 E1 — loyalty formation/decay (RQ1)

| File | code | Design | Runs | Seed |
|------|------|--------|------|------|
| `exp_001_e1_k1.py` | `RQ1_E1_k1_seeding` | 7R: R1 Nordvik 20% off (`12/15/15`) + R2–7 parity (`15/15/15`), `gpt-5.6-luna` | 50 | 101 |
| `exp_032_e1_k1_gemini.py` | `RQ1_E1_k1_seeding_gemini` | 1:1 replica of `exp_001` on `gemini-3.5-flash-lite` | 50 | 230 |
| `exp_033_e1_k1_deepseek.py` | `RQ1_E1_k1_seeding_deepseek` | 1:1 replica of `exp_001` on `deepseek-v4.1-flash` | 50 | 231 |

Metric: repeat-purchase rate of Nordvik in R2–R7 after discount ends.

```powershell
python runner.py experiments/exp_001_e1_k1.py
python runner.py experiments/exp_032_e1_k1_gemini.py
python runner.py experiments/exp_033_e1_k1_deepseek.py
```

### 5.4 E2 — price-premium tolerance grid (RQ2) — 20 cells: k∈{1,2,3,5} × p∈{+1%,+2%,+3%,+4%,+5%}

Seeding rounds always `12/15/15`; final round premium `15.15/15.30/15.45/15.60/15.75` vs `15.0`. All `gpt-5.6-luna`, 50 runs, seeds 202–221. File numbering is reverse-p within each k-block.

| k | +1% (15.15) | +2% (15.30) | +3% (15.45) | +4% (15.60) | +5% (15.75) |
|---|-------------|-------------|-------------|-------------|-------------|
| k=1 (2R) | `exp_006_e2_k1_p1.py` (`RQ2_E2_k1_p1`, s206) | `exp_005_e2_k1_p2.py` (s205) | `exp_004_e2_k1_p3.py` (s204) | `exp_003_e2_k1_p4.py` (s203) | `exp_002_e2_k1_p5.py` (s202) |
| k=2 (3R) | `exp_011_e2_k2_p1.py` (s211) | `exp_010_e2_k2_p2.py` (s210) | `exp_009_e2_k2_p3.py` (s209) | `exp_008_e2_k2_p4.py` (s208) | `exp_007_e2_k2_p5.py` (s207) |
| k=3 (4R) | `exp_016_e2_k3_p1.py` (s216) | `exp_015_e2_k3_p2.py` (s215) | `exp_014_e2_k3_p3.py` (s214) | `exp_013_e2_k3_p4.py` (s213) | `exp_012_e2_k3_p5.py` (s212) |
| k=5 (6R) | `exp_021_e2_k5_p1.py` (s221) | `exp_020_e2_k5_p2.py` (s220) | `exp_019_e2_k5_p3.py` (s219) | `exp_018_e2_k5_p4.py` (s218) | `exp_017_e2_k5_p5.py` (s217) |

Metric: retention on Nordvik in final round (conditional on having bought it during seeding) as f(k,p) — switching-threshold curve. No k=4 by design.

Batch (PowerShell):

```powershell
foreach ($f in "exp_002_e2_k1_p5.py","exp_003_e2_k1_p4.py","exp_004_e2_k1_p3.py","exp_005_e2_k1_p2.py","exp_006_e2_k1_p1.py") { python runner.py "experiments/$f" }
foreach ($f in "exp_007_e2_k2_p5.py","exp_008_e2_k2_p4.py","exp_009_e2_k2_p3.py","exp_010_e2_k2_p2.py","exp_011_e2_k2_p1.py","exp_012_e2_k3_p5.py","exp_013_e2_k3_p4.py","exp_014_e2_k3_p3.py","exp_015_e2_k3_p2.py","exp_016_e2_k3_p1.py","exp_017_e2_k5_p5.py","exp_018_e2_k5_p4.py","exp_019_e2_k5_p3.py","exp_020_e2_k5_p2.py","exp_021_e2_k5_p1.py") { python runner.py "experiments/$f" }
```

### 5.5 E3 — brand spillover laundry→dish (RQ3)

6-product catalog: `ld_nordvik/ld_zephyr/ld_auralis` (Laundry Detergent/`laundry`) + `ds_nordvik/ds_zephyr/ds_auralis` (Dish Soap/`dish`). R1–3: laundry only (`available=1`, `12/15/15`; dish `available=0`). R4: dish only (laundry `available=0`).

| File | code | R4 dish prices | Runs | Seed |
|------|------|----------------|------|------|
| `exp_022_e3_k3_parity.py` | `RQ3_E3_k3_parity` | parity `15/15/15` | 50 | 222 |
| `exp_023_e3_k3_p5.py` | `RQ3_E3_k3_p5` | Nordvik +5% `15.75/15/15` | 50 | 223 |

Requests: `["…detergent"]*3 + ["I want dish soap"]`. Metric: spillover `P(pick Nordvik dish R4 | seeded on Nordvik laundry)`.

```powershell
foreach ($f in "exp_022_e3_k3_parity.py","exp_023_e3_k3_p5.py") { python runner.py "experiments/$f" }
```

### 5.6 Cross-model replication (RQ1/RQ2/RQ3) — 10 files, 1:1 except `llm.model`, `code`, seed

| Base (`gpt-5.6-luna`) | `gemini-3.5-flash-lite` | `deepseek-v4.1-flash` |
|---|---|---|
| E0 1R `exp_024` (120 runs) | `exp_034_e0_control_gemini.py` (s232, 120) | `exp_035_e0_control_deepseek.py` (s233, 120) |
| E0 4R `exp_025` (120 runs) | `exp_026_e0_control_4r_gemini.py` (s224, 120) | `exp_027_e0_control_4r_deepseek.py` (s225, 120) |
| E1 k=1 `exp_001` (50) | `exp_032_e1_k1_gemini.py` (s230, 50) | `exp_033_e1_k1_deepseek.py` (s231, 50) |
| E2 k=3+1% `exp_016` (50) | `exp_028_e2_k3_p1_gemini.py` (s226, 50) | `exp_029_e2_k3_p1_deepseek.py` (s227, 50) |
| E3 parity `exp_022` (50) | `exp_030_e3_k3_parity_gemini.py` (s228, 50) | `exp_031_e3_k3_parity_deepseek.py` (s229, 50) |

```powershell
python runner.py experiments/exp_026_e0_control_4r_gemini.py 120
python runner.py experiments/exp_027_e0_control_4r_deepseek.py 120
python runner.py experiments/exp_034_e0_control_gemini.py 120
python runner.py experiments/exp_035_e0_control_deepseek.py 120
foreach ($f in "exp_028_e2_k3_p1_gemini.py","exp_029_e2_k3_p1_deepseek.py","exp_030_e3_k3_parity_gemini.py","exp_031_e3_k3_parity_deepseek.py","exp_032_e1_k1_gemini.py","exp_033_e1_k1_deepseek.py") { python runner.py "experiments/$f" }
```

### 5.7 Per-experiment detail sheets (all 36)

How to read each sheet: **Why** = hypothesis; **Sees** = what the agent observes
that round (prices, availability, request, history); **Judge** = metric +
interpretation. Catalog is Laundry Detergent/`cleaning`
(`p_nordvik/p_zephyr/p_auralis`, q0.8) unless stated otherwise.

**`exp_000_test.py` — `RQ1_Test_10rounds` (RQ1, pilot, NOT for analysis).**
`gpt-5.6-luna`/1.0, Milk/`dairy` (`m_*`), fixed order, 10R drifting Nordvik
14→21 (R1–3 14/15/18, R4 15/15/18, R5 16/15/18, R6 17/15/18, R7 18/15/18,
R8 19/15/18, R9 20/15/18, R10 21/16/14), `["I want milk"]*10`. Why: verify
stack before spending API calls. Judge: trace completes, no systemic failures.

**`exp_024_e0_control.py` — `RQ1_E0_control` (RQ1, E0).**
`gpt-5.6-luna`/0.7, 120 runs, seed 100, 1R parity 15/15/15,
`["I want laundry detergent"]`. Why: no-history baseline — no discount, no
history, choice should be ~1/3 per brand. Sees: one parity offer, empty
history. Judge: brand shares R1; deviations signal position/brand bias.

**`exp_025_e0_control_4r.py` — `RQ1_E0_control_4r` (RQ1, E0).**
Same as 024 but 4R all-parity, seed 102, 120 runs. Why: R1 repeats baseline;
R2–R4 show repeat dynamics from history alone (no price signal). Judge:
`P(repeat)`, stickiness `P(R4=R1)`; reference for E1 decay.

**`exp_001_e1_k1.py` — `RQ1_E1_k1_seeding` (RQ1, E1).**
`gpt-5.6-luna`/0.7, 50 runs, seed 101, 7R: R1 12/15/15 then R2–R7 15/15/15.
Why: does one 20%-off exposure create persistent repeats at parity, and how
fast does it decay? Sees: R1 cheap Nordvik, then 6 parity rounds with growing
history. Judge: take-up `P(R1=Nordvik)` then decay `P(RN=Nordvik|seeded)` N=2..7.

**`exp_032_e1_k1_gemini.py` — `RQ1_E1_k1_seeding_gemini` (RQ1, E1 replica).**
1:1 copy of `exp_001` on `gemini-3.5-flash-lite`, seed 230, 50 runs. Why: does
E1 loyalty generalise beyond `gpt-5.6-luna`? Judge: same decay curve, compare models.

**`exp_033_e1_k1_deepseek.py` — `RQ1_E1_k1_seeding_deepseek` (RQ1, E1 replica).**
1:1 copy of `exp_001` on `deepseek-v4.1-flash`, seed 231, 50 runs. Why/Judge:
as `exp_032` for deepseek.

**E2 k=1 block (2R: R1 12/15/15, R2 premium; 50 runs each). Tests whether a
single discount buys any tolerance to an immediate premium.**
- **`exp_006_e2_k1_p1.py` — `RQ2_E2_k1_p1`.** R2 15.15 (+1%), seed 206.
- **`exp_005_e2_k1_p2.py` — `RQ2_E2_k1_p2`.** R2 15.30 (+2%), seed 205.
- **`exp_004_e2_k1_p3.py` — `RQ2_E2_k1_p3`.** R2 15.45 (+3%), seed 204.
- **`exp_003_e2_k1_p4.py` — `RQ2_E2_k1_p4`.** R2 15.60 (+4%), seed 203.
- **`exp_002_e2_k1_p5.py` — `RQ2_E2_k1_p5`.** R2 15.75 (+5%), seed 202.
Metric (all): `P(R2=Nordvik | R1=Nordvik)` — bottom row of threshold curve.

**E2 k=2 block (3R: R1–R2 12/15/15, R3 premium; 50 runs each). Tests whether
two seeding rounds deepen tolerance.**
- **`exp_011_e2_k2_p1.py` — `RQ2_E2_k2_p1`.** R3 15.15, seed 211.
- **`exp_010_e2_k2_p2.py` — `RQ2_E2_k2_p2`.** R3 15.30, seed 210.
- **`exp_009_e2_k2_p3.py` — `RQ2_E2_k2_p3`.** R3 15.45, seed 209.
- **`exp_008_e2_k2_p4.py` — `RQ2_E2_k2_p4`.** R3 15.60, seed 208.
- **`exp_007_e2_k2_p5.py` — `RQ2_E2_k2_p5`.** R3 15.75, seed 207.
Metric: `P(R3=Nordvik | R1–R2 Nordvik)`.

**E2 k=3 block (4R: R1–R3 12/15/15, R4 premium; 50 runs each). Core of grid —
moderate habit vs graded shocks.**
- **`exp_016_e2_k3_p1.py` — `RQ2_E2_k3_p1`.** R4 15.15, seed 216. Base for `exp_028/029`.
- **`exp_015_e2_k3_p2.py` — `RQ2_E2_k3_p2`.** R4 15.30, seed 215.
- **`exp_014_e2_k3_p3.py` — `RQ2_E2_k3_p3`.** R4 15.45, seed 214.
- **`exp_013_e2_k3_p4.py` — `RQ2_E2_k3_p4`.** R4 15.60, seed 213.
- **`exp_012_e2_k3_p5.py` — `RQ2_E2_k3_p5`.** R4 15.75, seed 212.
Metric: `P(R4=Nordvik | R1–R3 Nordvik)`.

**E2 k=5 block (6R: R1–R5 12/15/15, R6 premium; 50 runs each). Tests whether
deep habit survives the same shocks.**
- **`exp_021_e2_k5_p1.py` — `RQ2_E2_k5_p1`.** R6 15.15, seed 221.
- **`exp_020_e2_k5_p2.py` — `RQ2_E2_k5_p2`.** R6 15.30, seed 220.
- **`exp_019_e2_k5_p3.py` — `RQ2_E2_k5_p3`.** R6 15.45, seed 219.
- **`exp_018_e2_k5_p4.py` — `RQ2_E2_k5_p4`.** R6 15.60, seed 218.
- **`exp_017_e2_k5_p5.py` — `RQ2_E2_k5_p5`.** R6 15.75, seed 217.
Metric: `P(R6=Nordvik | R1–R5 Nordvik)` — top row of threshold curve.

**`exp_022_e3_k3_parity.py` — `RQ3_E3_k3_parity` (RQ3, spillover).**
`gpt-5.6-luna`/0.7, 50 runs, seed 222, 6-product catalog (`ld_*/ds_*`,
`laundry`/`dish`). R1–R3 laundry-only 12/15/15 (dish unavailable); R4
dish-only parity 15/15/15. Requests detergent×3 then dish soap. Why: does
laundry loyalty spill to unexperienced same-brand dish at equal price? Judge:
`P(R4=Nordvik dish | seeded Nordvik laundry)`.

**`exp_023_e3_k3_p5.py` — `RQ3_E3_k3_p5` (RQ3, spillover + premium).**
As `exp_022` but R4 dish 15.75/15/15, seed 223. Why: does umbrella survive +5%?
Judge: same spillover rate; parity vs +5% gap.

**`exp_026_e0_control_4r_gemini.py` — `RQ1_E0_control_4r_gemini`.**
1:1 of `exp_025` on `gemini-3.5-flash-lite`, seed 224, 120 runs. Why: model
baseline for repeats without price. Judge: shares, repeat rates.

**`exp_027_e0_control_4r_deepseek.py` — `RQ1_E0_control_4r_deepseek`.**
As above on `deepseek-v4.1-flash`, seed 225, 120 runs.

**`exp_028_e2_k3_p1_gemini.py` — `RQ2_E2_k3_p1_gemini`.**
1:1 of `exp_016` (k=3,+1%) on gemini, seed 226, 50 runs. Why: cheapest shock
after moderate habit — most sensitive probe. Judge: conditional retention.

**`exp_029_e2_k3_p1_deepseek.py` — `RQ2_E2_k3_p1_deepseek`.**
As above on deepseek, seed 227, 50 runs.

**`exp_030_e3_k3_parity_gemini.py` — `RQ3_E3_k3_parity_gemini`.**
1:1 of `exp_022` on gemini, seed 228, 50 runs. Judge: conditional spillover.

**`exp_031_e3_k3_parity_deepseek.py` — `RQ3_E3_k3_parity_deepseek`.**
1:1 of `exp_022` on deepseek, seed 229, 50 runs. Judge: conditional spillover.

**`exp_034_e0_control_gemini.py` — `RQ1_E0_control_gemini` (RQ1, E0 replica).**
1:1 copy of `exp_024` (1R parity 15/15/15) on `gemini-3.5-flash-lite`, seed 232,
120 runs. Why: no-history brand baseline per model — required before comparing
E1/E2/E3 cross-model effects. Judge: brand shares ≈1/3 each; deviations signal bias.

**`exp_035_e0_control_deepseek.py` — `RQ1_E0_control_deepseek` (RQ1, E0 replica).**
As above on `deepseek-v4.1-flash`, seed 233, 120 runs. Why/Judge: as `exp_034`
for deepseek.

---

## 6. Metrics cheat-sheet

- **RQ1/E1:** repeat-purchase rate of Nordvik in parity rounds (R2–R7 of `exp_001`); compare vs E0 baselines (`exp_024` R1, `exp_025` R1–R4). Filter `trajectories.status='finished'`.
- **RQ2/E2:** final-round retention on Nordvik conditional on seeding purchase, as function of (k,p). Plot switching curve.
- **RQ3/E3:** spillover `P(Nordvik dish R4 | Nordvik laundry seeding)`, parity vs +5%.
- **Cross-model:** same metrics on gemini/deepseek replicas to test generalisation.
- Always exclude `status='failed'` trajectories (malformed JSON aborts; rounds before failure stay, failed+later rounds missing).

---

## 7. Runbook

### 7.1 Config — `.env` (gitignored, loaded by runner/seed/smoke)

```ini
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/loyalty
API_KEY=sk-...
BASE_URL=https://your-provider/v1   # optional; omit for OpenAI default
MODEL=...                            # present in .env, unused by runner (model comes from experiment file)
```

Requires: Python 3.10+, Postgres, OpenAI-compatible chat API. Install deps per lockfile/`requirements.txt`.

### 7.2 Steps

1. **Seed RQs once:** `python scripts/seed_rqs.py`
2. **Smoke test (no DB):** `python scripts/smoke_test.py` — 1× 10-round milk trajectory, live LLM.
3. **Single experiment:** `python runner.py experiments/<file>.py [n]` (default 50; E0 needs 120).
4. **Batches:** see PowerShell loops in §§5.4–5.6. Interrupted run → re-run only remaining files (completed codes blocked).
5. **Tests:** `pytest` (env + graph).
6. **New experiment:** copy any `exp_*.py`; required `EXPERIMENT` keys: `code (unique), rq_id (must exist), name, description?, catalog, schedule, user_requests (len==schedule), llm{model,temperature,base_url?}, agent{max_category_retries,max_commit_retries}, presentation?{order,seed}, runs (informational)`. Validate free: `python -c "from runner import load_experiment, validate_experiment; e=load_experiment('experiments/<f>.py'); validate_experiment(e); print('OK')"`.

### 7.3 Project layout

```
agent/          graph.py, nodes.py, prompts.py, state.py, schemas.py (LangGraph agent)
store/          engine.py (StoreEnv), models.py (Pydantic specs)
db/             tables.py (5 tables), base.py (engine/factory), repository.py (CRUD)
experiments/    exp_000 (pilot), exp_001/032/033 (E1), exp_002–021 (E2 grid), exp_022–023 (E3),
                exp_024–025 (E0), exp_026–035 (cross-model E0/E1/E2/E3)
orchestrator.py round loop (no DB)
runner.py       CLI: validate → snapshot → N trajectories → persist
scripts/        seed_rqs.py, smoke_test.py, Untitled-1.ipynb (scratch)
tests/          test_env.py, test_graph.py, conftest.py
.env / .gitignore / README.md
```

---

## 8. Known gaps / next-owner TODOs

1. **No analysis code** — metrics in §6 must be queried manually from Postgres (`experiments → trajectories → rounds → purchases`); no notebooks/scripts for curves/stats.
2. **Reason classifier missing** — `purchases.reason_code/reason_note/classified_at` always NULL; smoke test asserts this. Needs post-hoc LLM classifier if free-text reasons to be coded.
3. **Dead code** `store/engine.py:272-300` (after `return` in `commit_purchase`) — delete.
4. **`.env:MODEL`** unused by `runner.py` (only `smoke_test.py` hardcodes its own model) — clarify or wire through.
5. **No DB tests / no `ondelete` cascades** — manual delete order required for re-runs; consider helper script.
6. **`exp_000` + `smoke_test` temp 1.0** — not comparable to main `0.7` body; keep as infra checks only.
7. **E2 has no k=4** — intentional per README but confirm if gap to fill.

---

## 9. Reproducibility notes

- Each `experiments` row stores full snapshot (`catalog_json, schedule_json, user_requests, allowed_categories, llm_config, agent_config+presentation`) — DB alone suffices to reconstruct design.
- Presentation seed per file unique (100,101,102,202–233); per-trajectory order = `seed + run_index - 1`, logged in `StoreEnv.event_log` (`PRODUCTS_SHOWN`) and `shown_orders` (smoke test prints; runner does not persist shown order — recover via seed+round formula if needed).
- Duplicate `code` rejected — finished experiments immutable unless rows deleted.

---

*Generated from full repo read (README, runner, orchestrator, agent/*, store/*, db/*, scripts/*, tests/*, all 36 experiments). For design rationale see `README.md:1-215`; for execution semantics see `runner.py:1-303`, `orchestrator.py:1-110`, `store/engine.py`, `agent/graph.py`.*
