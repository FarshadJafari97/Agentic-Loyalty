# Results Extraction Guide — Per-Experiment Output Tables

Companion to `HANDOFF.md`. Tells you how to turn Postgres rows into a
comprehensive result table for **each** experiment.

Schema chain: `experiments(code) → trajectories(run_index,status) → rounds(round_number,status) → purchases(brand,product_id,price_paid,reason_text 1:1, nullable)`.
`scripts/Untitled-1.ipynb` is unrelated scratch (Persian RAG demo) — ignore it.

## 0. Global rules (apply to every query)

1. **Filter trajectories:** `trajectories.status = 'finished'` only. `failed` = malformed LLM JSON; rounds before failure stay, failed + later rounds missing.
2. **LEFT JOIN purchases:** failed rounds have a `rounds` row (`status='failed'`, `failure_reason`) but no `purchases` row.
3. **Identify treated brand:** `brand = 'Nordvik'` everywhere except E3-R4 where `product_id` disambiguates (`ld_nordvik` vs `ds_nordvik`).
4. **Ground-truth prices** live in `experiments.schedule_json`; `purchases.price_paid` is what agent actually paid. Prefer `price_paid` for results, `schedule_json` only to label design (k, p).
5. **Run counts:** E0 = 120 trajectories, E1/E2/E3 = 50, replications same as base. Expect fewer `finished` rows.

### Canonical round-level export (start here)

```sql
-- Sample: replace 'RQ1_E1_k1_seeding' with any experiment code
SELECT
  e.code            AS experiment_code,
  e.llm_config->>'model' AS model,
  t.run_index,
  t.status          AS trajectory_status,
  r.round_number,
  r.status          AS round_status,
  r.failure_reason,
  p.product_id,
  p.brand,
  p.price_paid,
  p.reason_text
FROM experiments e
JOIN trajectories t ON t.experiment_id = e.experiment_id
JOIN rounds r       ON r.trajectory_id = t.trajectory_id
LEFT JOIN purchases p ON p.round_id = r.round_id
WHERE e.code = 'RQ1_E1_k1_seeding'
  AND t.status = 'finished'
ORDER BY t.run_index, r.round_number;
```

Python equivalent (pandas):

```python
import os, pandas as pd
from sqlalchemy import create_engine, text
eng = create_engine(os.environ["DATABASE_URL"])
# Sample: replace 'RQ2_E2_k3_p1' with any experiment code
q = open("query.sql").read().replace("sample_experiment_code", "RQ2_E2_k3_p1")
df = pd.read_sql(text(q), eng)
# df is round-grain: one row per (run_index, round_number)
```

Quality checks per experiment (run first):

```sql
-- Sample: replace 'RQ1_E1_k1_seeding' with any experiment code
SELECT
  COUNT(*) FILTER (WHERE t.status='finished') AS n_finished,
  COUNT(*) FILTER (WHERE t.status='failed')   AS n_failed,
  COUNT(*) AS n_total
FROM experiments e JOIN trajectories t USING (experiment_id)
WHERE e.code = 'RQ1_E1_k1_seeding';
-- n_finished + n_failed must = intended runs (50 or 120); missing rounds => failed traj
```

---

## 1. E0 controls — RQ1 baselines (no seeding, parity 15.0/15.0)

### 1a. `exp_024` / `exp_034` / `exp_035` — `RQ1_E0_control[_gemini|_deepseek]` (1 round, 120 runs)

One row per trajectory (= one purchase). Healthy: each brand ≈ 1/3.
Replicas `RQ1_E0_control_gemini` (`exp_034`, seed 232) and
`RQ1_E0_control_deepseek` (`exp_035`, seed 233) — same query, swap code.

```sql
-- Sample for 'RQ1_E0_control'.
-- For replicas use 'RQ1_E0_control_gemini' or 'RQ1_E0_control_deepseek'.
SELECT t.run_index, p.brand, p.product_id, p.price_paid, p.reason_text,
       r.status AS round_status, r.failure_reason
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id
JOIN rounds r ON r.trajectory_id=t.trajectory_id AND r.round_number=1
LEFT JOIN purchases p ON p.round_id=r.round_id
WHERE e.code='RQ1_E0_control' AND t.status='finished'
ORDER BY t.run_index;
```

**Output table `e0_1r` columns:** `experiment_code|model|run_index|round_status|brand|product_id|price_paid|reason_text|failure_reason`.
**Summary:** `share_nordvik = AVG(brand='Nordvik')`, same for Zephyr/Auralis + `n_finished`. Chi-square vs uniform optional.

### 1b. `exp_025` / `exp_026` / `exp_027` — `RQ1_E0_control_4r[_gemini|_deepseek]` (4 rounds, all parity)

Wide format — one row per trajectory, R1 = no-history baseline, R2–R4 = repeat without price signal.

```sql
-- Sample for 'RQ1_E0_control_4r'.
-- For replicas, replace with 'RQ1_E0_control_4r_gemini' or 'RQ1_E0_control_4r_deepseek'.
SELECT t.run_index,
  MAX(p.brand) FILTER (WHERE r.round_number=1) AS brand_r1,
  MAX(p.brand) FILTER (WHERE r.round_number=2) AS brand_r2,
  MAX(p.brand) FILTER (WHERE r.round_number=3) AS brand_r3,
  MAX(p.brand) FILTER (WHERE r.round_number=4) AS brand_r4,
  COUNT(p.purchase_id) AS n_committed
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id
JOIN rounds r ON r.trajectory_id=t.trajectory_id
LEFT JOIN purchases p ON p.round_id=r.round_id
WHERE e.code = 'RQ1_E0_control_4r' AND t.status='finished'
GROUP BY t.run_index ORDER BY t.run_index;
```

**Output table `e0_4r` columns:** `experiment_code|model|run_index|brand_r1|brand_r2|brand_r3|brand_r4|n_committed|repeat_r2 (=brand_r2=brand_r1)|repeat_r3|repeat_r4|stayed_all_4|switched_ever|reason_r1..r4 (optional, array_agg reason_text by round)`.
**Summaries:** per-round brand shares; `P(repeat_rN)`; stickiness `P(brand_r4=brand_r1)`; compare gemini/deepseek vs gpt via same query with `GROUP BY e.code`.

---

## 2. E1 — `exp_001` `RQ1_E1_k1_seeding` + replicas `exp_032/033` (RQ1, 7R: 1×12/15/15 + 6×parity)

Replicas: `RQ1_E1_k1_seeding_gemini` (`exp_032`, seed 230) and
`RQ1_E1_k1_seeding_deepseek` (`exp_033`, seed 231) — same query, swap code.

```sql
-- Long (round-grain): use canonical query with 'RQ1_E1_k1_seeding'
-- (replicas: replace with 'RQ1_E1_k1_seeding_gemini' or 'RQ1_E1_k1_seeding_deepseek')
-- Wide (trajectory-grain, sample for base; swap code for replicas):
SELECT t.run_index,
  MAX(p.brand) FILTER (WHERE r.round_number=1) AS seed_pick,
  MAX(p.brand) FILTER (WHERE r.round_number=2) AS brand_r2,
  MAX(p.brand) FILTER (WHERE r.round_number=3) AS brand_r3,
  MAX(p.brand) FILTER (WHERE r.round_number=4) AS brand_r4,
  MAX(p.brand) FILTER (WHERE r.round_number=5) AS brand_r5,
  MAX(p.brand) FILTER (WHERE r.round_number=6) AS brand_r6,
  MAX(p.brand) FILTER (WHERE r.round_number=7) AS brand_r7
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id
JOIN rounds r ON r.trajectory_id=t.trajectory_id
LEFT JOIN purchases p ON p.round_id=r.round_id
WHERE e.code='RQ1_E1_k1_seeding' AND t.status='finished'
GROUP BY t.run_index ORDER BY t.run_index;
```

**Output tables:**
- `e1_long` (comprehensive, one row/round): `experiment_code|model|run_index|round_number|phase (=seeding|parity)|round_status|brand|product_id|price_paid|reason_text|seed_pick (brand_r1 repeated)|is_nordvik|is_repeat_of_seed`.
- `e1_wide` (one row/trajectory): above + derived: `seeded_nordvik (=seed_pick='Nordvik')|n_nordvik_parity (count R2–R7 Nordvik)|streak_from_r2 (consecutive Nordvik from R2)|first_switch_round|never_switched`.
**Summaries:** seeding take-up `P(seed_pick=Nordvik)` (≈1 expected); decay curve `P(brand_rN=Nordvik | seeded_nordvik)` for N=2..7; compare each parity round vs E0 parity baseline; compare decay curves across `RQ1_E1_k1_seeding[_gemini|_deepseek]`.

---

## 3. E2 grid — RQ2 (k∈{1,2,3,5} × p∈{+1..+5%}), 20 cells + 2 replicas

Codes: `RQ2_E2_k{p}_p{p}` pattern — k1: `exp_002..006`, k2: `exp_007..011`, k3: `exp_012..016`, k5: `exp_017..021`; replicas `RQ2_E2_k3_p1_gemini|_deepseek` (`exp_028|029`). Final-round premium: p1=15.15, p2=15.30, p3=15.45, p4=15.60, p5=15.75. Rounds = k+1.

### 3a. Per-cell trajectory table (run for each of the 22 codes)

```sql
-- Sample cell: k=3, final round 4, code 'RQ2_E2_k3_p1'.
-- For another cell, replace k values and code together:
--   k=1 -> seed rounds <= 1, final round = 2 (e.g. 'RQ2_E2_k1_p5')
--   k=2 -> seed rounds <= 2, final round = 3 (e.g. 'RQ2_E2_k2_p5')
--   k=3 -> seed rounds <= 3, final round = 4 (e.g. 'RQ2_E2_k3_p1')
--   k=5 -> seed rounds <= 5, final round = 6 (e.g. 'RQ2_E2_k5_p1')
SELECT t.run_index,
  COUNT(*) FILTER (WHERE r.round_number <= 3 AND p.brand='Nordvik') AS n_seed_nordvik,
  COUNT(*) FILTER (WHERE r.round_number <= 3) AS n_seed_committed,
  MAX(p.brand) FILTER (WHERE r.round_number = 4) AS final_brand,
  MAX(p.product_id) FILTER (WHERE r.round_number = 4) AS final_product,
  MAX(p.price_paid) FILTER (WHERE r.round_number = 4) AS final_price,
  MAX(p.reason_text) FILTER (WHERE r.round_number = 4) AS final_reason
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id
JOIN rounds r ON r.trajectory_id=t.trajectory_id
LEFT JOIN purchases p ON p.round_id=r.round_id
WHERE e.code = 'RQ2_E2_k3_p1' AND t.status='finished'
GROUP BY t.run_index ORDER BY t.run_index;
```

**Output table `e2_cell` (one row/trajectory, comprehensive):** `experiment_code|model|k|premium_pct|premium_price|run_index|n_seed_nordvik|n_seed_committed|fully_seeded (=n_seed_nordvik=k)|final_brand|final_product|final_price|final_reason|retained (=final_brand='Nordvik')|retained_conditional (=retained WHERE fully_seeded)|switched_to (=final_brand when not retained)|round_status_final|failure_reason_final`.
Keep `final_reason` (free text) — primary evidence for *why* premium tolerated/rejected.

### 3b. Grid summary (the switching-threshold curve — run once across all E2)

```sql
SELECT e.code, e.llm_config->>'model' AS model,
  COUNT(DISTINCT t.trajectory_id) AS n_finished,
  AVG(CASE WHEN seed_n.brand='Nordvik' THEN 1.0 ELSE 0.0 END) AS seeding_takeover,
  AVG(CASE WHEN f.brand='Nordvik' THEN 1.0 ELSE 0.0 END) AS retention_uncond,
  AVG(CASE WHEN f.brand='Nordvik' THEN 1.0 ELSE 0.0 END)
    FILTER (WHERE seed_all.is_full) AS retention_cond
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id AND t.status='finished'
LEFT JOIN LATERAL (
  SELECT p.brand FROM rounds r LEFT JOIN purchases p USING (round_id)
  WHERE r.trajectory_id=t.trajectory_id ORDER BY r.round_number DESC LIMIT 1
) f ON true
LEFT JOIN LATERAL (
  SELECT p.brand FROM rounds r LEFT JOIN purchases p USING (round_id)
  WHERE r.trajectory_id=t.trajectory_id ORDER BY r.round_number ASC LIMIT 1
) seed_n ON true
LEFT JOIN LATERAL (
  SELECT COUNT(*) FILTER (WHERE p.brand='Nordvik') = COUNT(*) AND COUNT(*)>0 AS is_full
  FROM (SELECT r.round_number, p.brand,
               ROW_NUMBER() OVER (ORDER BY r.round_number) AS rn,
               COUNT(*) OVER () AS tot
        FROM rounds r LEFT JOIN purchases p USING (round_id)
        WHERE r.trajectory_id=t.trajectory_id) s
  WHERE rn <= tot - 1
) seed_all ON true
WHERE e.code IN ('RQ2_E2_k1_p1','RQ2_E2_k1_p2','RQ2_E2_k1_p3','RQ2_E2_k1_p4','RQ2_E2_k1_p5',
 'RQ2_E2_k2_p1','RQ2_E2_k2_p2','RQ2_E2_k2_p3','RQ2_E2_k2_p4','RQ2_E2_k2_p5',
 'RQ2_E2_k3_p1','RQ2_E2_k3_p2','RQ2_E2_k3_p3','RQ2_E2_k3_p4','RQ2_E2_k3_p5',
 'RQ2_E2_k5_p1','RQ2_E2_k5_p2','RQ2_E2_k5_p3','RQ2_E2_k5_p4','RQ2_E2_k5_p5')
GROUP BY e.code, model ORDER BY e.code;
```

Parse `k` and `p` from `code` suffix (`k{k}_p{p}`) in Python for plotting. **Primary metric:** `retention_cond` as f(k,p). Also report `retention_uncond` and `seeding_takeover` as diagnostics. Replicas (`_gemini|_deepseek` for k3p1) use §3a query, then append as extra rows.

---

## 4. E3 spillover — RQ3 (laundry R1–R3 → dish R4)

Codes: `RQ3_E3_k3_parity` (`exp_022`), `RQ3_E3_k3_p5` (`exp_023`), replicas `RQ3_E3_k3_parity_gemini|_deepseek` (`exp_030|031`). Products: `ld_*` (laundry) / `ds_*` (dish).

```sql
-- Sample for 'RQ3_E3_k3_parity'.
-- For premium test use 'RQ3_E3_k3_p5';
-- for replicas use 'RQ3_E3_k3_parity_gemini' or 'RQ3_E3_k3_parity_deepseek'.
SELECT t.run_index,
  MAX(p.brand) FILTER (WHERE r.round_number=1) AS seed_r1,
  MAX(p.brand) FILTER (WHERE r.round_number=2) AS seed_r2,
  MAX(p.brand) FILTER (WHERE r.round_number=3) AS seed_r3,
  COUNT(*) FILTER (WHERE r.round_number<=3 AND p.brand='Nordvik') AS n_seed_nordvik,
  MAX(p.brand) FILTER (WHERE r.round_number=4) AS spill_brand,
  MAX(p.product_id) FILTER (WHERE r.round_number=4) AS spill_product,
  MAX(p.price_paid) FILTER (WHERE r.round_number=4) AS spill_price,
  MAX(p.reason_text) FILTER (WHERE r.round_number=4) AS spill_reason,
  MAX(r.status) FILTER (WHERE r.round_number=4) AS spill_round_status
FROM experiments e
JOIN trajectories t ON t.experiment_id=e.experiment_id
JOIN rounds r ON r.trajectory_id=t.trajectory_id
LEFT JOIN purchases p ON p.round_id=r.round_id
WHERE e.code = 'RQ3_E3_k3_parity' AND t.status='finished'
GROUP BY t.run_index ORDER BY t.run_index;
```
```

**Output table `e3` (one row/trajectory, comprehensive):** `experiment_code|model|test (=parity|+5%)|run_index|seed_r1|seed_r2|seed_r3|n_seed_nordvik|fully_seeded (=n_seed_nordvik=3)|spill_brand|spill_product|spill_price|spill_reason|spill_round_status|spilled (=spill_brand='Nordvik')|spilled_conditional (=spilled WHERE fully_seeded)`.
**Summaries:** `spillover_uncond = AVG(spilled)`, `spillover_cond = AVG(spilled) WHERE fully_seeded` — report both, lead with conditional. Compare parity vs +5% (does premium kill umbrella effect?); compare models on parity.

---

## 5. Pilot `exp_000` — `RQ1_Test_10rounds` (not for analysis)

Drifting milk prices, temp 1.0, fixed order. Extract with canonical query only to verify pipeline (purchase trace R1–R10, no failed/commit errors systemic). Do not pool with RQ1–RQ3.

## 6. Cross-model comparison sheet (1 table to rule them all)

Run §1b/§2/§3a/§4 queries for each triple, union with `model` column, then:

| comparison | gpt-5.6-luna | gemini-3.5-flash-lite | deepseek-v4.1-flash |
|---|---|---|---|
| E0 1R baseline shares | `RQ1_E0_control` | `RQ1_E0_control_gemini` | `RQ1_E0_control_deepseek` |
| E0 repeat `P(repeat_r2..r4)` | `RQ1_E0_control_4r` | `…_gemini` | `…_deepseek` |
| E1 decay `P(brand_rN=Nordvik)` | `RQ1_E1_k1_seeding` | `RQ1_E1_k1_seeding_gemini` | `RQ1_E1_k1_seeding_deepseek` |
| E2 retention cond (k3p1) | `RQ2_E2_k3_p1` | `…_gemini` | `…_deepseek` |
| E3 spillover cond (parity) | `RQ3_E3_k3_parity` | `…_gemini` | `…_deepseek` |

Test generalisation: same direction + overlapping CIs (Wilson 95% for all rates, n=finished).

## 7. Minimal CSV export template

```python
# export_one.py — usage: python export_one.py RQ2_E2_k3_p1 out.csv
import os, sys, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
eng = create_engine(os.environ["DATABASE_URL"])
code = sys.argv[1]
# NOTE: raw SQL below uses a static sample code so it also runs in a GUI client.
# Python builds the final query by string substitution (codes are internal constants).
q = """SELECT e.code AS experiment_code, e.llm_config->>'model' AS model,
 t.run_index, r.round_number, r.status AS round_status, r.failure_reason,
 p.product_id, p.brand, p.price_paid, p.reason_text
 FROM experiments e JOIN trajectories t ON t.experiment_id=e.experiment_id
 JOIN rounds r ON r.trajectory_id=t.trajectory_id
 LEFT JOIN purchases p ON p.round_id=r.round_id
 WHERE e.code='sample_experiment_code' AND t.status='finished' ORDER BY 3,4""".replace(
    "sample_experiment_code", code)
pd.read_sql(text(q), eng).to_csv(sys.argv[2], index=False)
print("wrote", sys.argv[2])
```

Derive all wide/conditional tables from this long CSV (groupby run_index) — keeps SQL small and auditable.

## 8. Master flat table — ALL experiments × ALL trajectories × ALL rounds (nothing omitted)

One row per round (round-grain). Uses `LEFT JOIN` throughout so failed
trajectories, failed rounds, and unclassified purchases are kept — no
`WHERE` filter. Every column from all 5 tables is included; derived flags
are appended at the end (never replace raw columns). Copy-paste directly
into any GUI client.

```sql
SELECT
  -- research_questions (all columns)
  rq.rq_id,
  rq.code                AS rq_code,
  rq.title               AS rq_title,
  rq.description         AS rq_description,
  rq.created_at          AS rq_created_at,
  -- experiments (all columns, incl. full JSON snapshots)
  e.experiment_id,
  e.rq_id                AS experiment_rq_id,
  e.code                 AS experiment_code,
  e.name                 AS experiment_name,
  e.description          AS experiment_description,
  e.catalog_json,
  e.schedule_json,
  e.user_requests,
  e.allowed_categories,
  e.llm_config,
  e.llm_config->>'model' AS llm_model,
  e.llm_config->>'temperature' AS llm_temperature,
  e.agent_config,
  e.created_at           AS experiment_created_at,
  -- trajectories (all columns)
  t.trajectory_id,
  t.experiment_id        AS trajectory_experiment_id,
  t.run_index,
  t.status               AS trajectory_status,
  t.started_at           AS trajectory_started_at,
  t.finished_at          AS trajectory_finished_at,
  t.error_message        AS trajectory_error_message,
  -- rounds (all columns)
  r.round_id,
  r.trajectory_id        AS round_trajectory_id,
  r.round_number,
  r.budget               AS round_budget,
  r.status               AS round_status,
  r.failure_reason       AS round_failure_reason,
  -- purchases (all columns; NULL when round failed)
  p.purchase_id,
  p.round_id             AS purchase_round_id,
  p.product_id,
  p.product_name,
  p.category             AS purchase_category,
  p.brand                AS purchase_brand,
  p.price_paid,
  p.reason_text,
  p.reason_code,
  p.reason_note,
  p.classified_at        AS purchase_classified_at,
  p.created_at           AS purchase_created_at,
  -- derived convenience flags (do not replace raw columns)
  CASE WHEN p.brand = 'Nordvik' THEN 1 ELSE 0 END AS is_nordvik,
  CASE WHEN r.status = 'committed' THEN 1 ELSE 0 END AS is_committed,
  CASE WHEN t.status = 'finished' THEN 1 ELSE 0 END AS is_trajectory_finished
FROM research_questions rq
JOIN experiments e   ON e.rq_id = rq.rq_id
JOIN trajectories t  ON t.experiment_id = e.experiment_id
JOIN rounds r        ON r.trajectory_id = t.trajectory_id
LEFT JOIN purchases p ON p.round_id = r.round_id
ORDER BY e.code, t.run_index, r.round_number;
```

Column notes:

- Grain: `(experiment_code, run_index, round_number)` is unique per row
  (enforced by `UNIQUE(experiment_id,run_index)` + `UNIQUE(trajectory_id,round_number)`).
- `catalog_json / schedule_json / llm_config / agent_config`: full design
  snapshots (prices, availability, seeds, temperature). Presentation seed =
  `agent_config->'presentation'->>'seed'`, order =
  `agent_config->'presentation'->>'order'`.
- `user_requests`: array; element `round_number` = request for that row
  (Postgres arrays are 1-indexed: `user_requests[round_number]` with a proper cast).
- Failed rounds: `round_status='failed'`, purchase columns all NULL,
  reason in `round_failure_reason`. Failed trajectories:
  `trajectory_status='failed'`, `trajectory_error_message` set, later rounds absent.
- `reason_code / reason_note / purchase_classified_at`: always NULL at
  collection (reserved for post-hoc classifier) — kept so the export never
  silently drops a future column.
- Analysis pattern: `WHERE is_trajectory_finished = 1` reproduces every
  per-experiment query in §§0–4; omit the filter for pipeline debugging
  (failure audit, count reconciliation vs intended 50/120 runs).
```
