# Experimental Report: LLM Shopping Agent Brand Loyalty and Price Sensitivity

This report details the experimental setup and empirical findings from 35 controlled multi-round shopping experiments ($N = 7{,}780$ valid rounds) evaluating how LLM shopping agents make purchasing decisions across brand history, price changes, and product categories.

---

## 1. Experimental Environment & Setup

All experiments were conducted in a simulated shopping environment (`StoreEnv`) with strict controls:
- **Synthetic Brands:** Three fictional brands were used—**Nordvik**, **Zephyr**, and **Auralis**—to avoid real-world training biases.
- **Equal Quality:** All products were assigned identical quality ratings ($Q = 0.80$).
- **Budget Ceiling:** Agents were given a non-binding budget of \$100.00 per round, ensuring purchasing choices reflect brand/price preferences rather than spending limits.
- **Order Shuffling:** Product display order was randomized per trajectory and round to prevent position bias.
- **Evaluated Models:** Primary benchmark on OpenAI **GPT-5.6-Luna**, with full cross-model replication on **Google Gemini-3.5-Flash-Lite** and **DeepSeek-V4.1-Flash**.

---

## 2. Experiment 1: Baseline Brand & Position Bias

### Experiment Setup
- **Goal:** Determine whether the agents have any pre-existing preference for particular brand names or display positions when shopping without prior history.
- **Configuration:** 1-round control experiment (`RQ1_E0_control`).
- **Catalog & Pricing:** 3 Laundry Detergents (Nordvik, Zephyr, Auralis) priced equally at \$15.00.
- **Sample Size:** 360 trajectories (120 runs per model).

### Results & Observations
- Market share was distributed almost evenly across all three brands:
  - **Nordvik:** $34.44\%$ (124 purchases)
  - **Zephyr:** $33.06\%$ (119 purchases)
  - **Auralis:** $32.50\%$ (117 purchases)
- In the base model (GPT-5.6-Luna), market share was $35.00\%$ for Nordvik, $34.17\%$ for Auralis, and $30.83\%$ for Zephyr.
- **Key Takeaway:** There is zero innate brand bias or positional favoritism. Each brand captures approximately one-third of initial purchases.

### Data Summary
| Brand | Total Purchases ($N=360$) | Total Market Share (%) | Base Model Purchases ($N=120$) | Base Model Share (%) |
|:---|:---:|:---:|:---:|:---:|
| **Nordvik** | 124 | 34.44% | 42 | 35.00% |
| **Zephyr** | 119 | 33.06% | 37 | 30.83% |
| **Auralis** | 117 | 32.50% | 41 | 34.17% |

### Visualization
![Baseline Brand Market Share](./1_baseline_bias/baseline_share.png)

---

## 3. Experiment 2: Inertia & Stickiness Under Parity

### Experiment Setup
- **Goal:** Test whether agents develop repeat-purchase habits purely from history when prices remain equal across multiple rounds.
- **Configuration:** 4-round control experiment (`RQ1_E0_control_4r`).
- **Catalog & Pricing:** All 3 brands available at \$15.00 in all 4 rounds. No discounts or price changes.
- **Metrics Tracked:**
  - **Metric A (Stickiness to First Purchase):** % of runs where Round $N$ brand matches Round 1.
  - **Metric B (Stickiness to Previous Purchase):** % of runs where Round $N$ brand matches Round $N-1$.

### Results & Observations
- In the base model (GPT-5.6-Luna), agents showed strong immediate stickiness:
  - Round 2: $85.83\%$ repeated their Round 1 choice.
  - Round 3: $85.00\%$ repeated their choice.
  - Round 4: $91.67\%$ repeated their Round 1 choice ($88.33\%$ repeated Round 3).
- Across the pooled multi-model cohort ($N = 358$), Metric A held at $61.17\%$ in Round 2, $60.89\%$ in Round 3, and surged to **$91.90\%$** in Round 4.
- **Key Takeaway:** When prices are equal, agents use past purchases as a tiebreaker. Once a brand is picked, the agent defaults to repeating it in subsequent rounds.

### Data Summary
| Round | Metric A: Match Round 1 (%) | Metric B: Match Previous Round (%) | Base GPT Metric A (%) | Base GPT Metric B (%) | Total Runs ($N$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Round 2** | 61.17% | 61.17% | 85.83% | 85.83% | 358 |
| **Round 3** | 60.89% | 60.89% | 85.00% | 85.00% | 358 |
| **Round 4** | 91.90% | 62.85% | 91.67% | 88.33% | 358 |

### Visualization
![Inertia Metrics](./2_inertia_stickiness/inertia_metrics.png)

---

## 4. Experiment 3: Loyalty Decay After Promotional Discount

### Experiment Setup
- **Goal:** Measure how long brand loyalty persists after a promotional discount ends and prices return to parity.
- **Configuration:** 7-round seeding experiment (`RQ1_E1_k1_seeding`).
- **Round 1 (Promo):** Nordvik discounted by $20\%$ to \$12.00 (competitors at \$15.00).
- **Rounds 2–7 (Parity):** All 3 brands priced equally at \$15.00.
- **Sample Size:** 150 trajectories across models (50 runs per model).

### Results & Observations
- In Round 1, $100\%$ of agents selected Nordvik due to the price discount.
- In Rounds 2 through 7 (under strict price parity), **$100.0\%$ of agents continued buying Nordvik in every single round**.
- There was zero decay over the 6 parity rounds.
- **Key Takeaway:** A single initial discount creates complete repeat-purchase lock-in. The agent never switches away as long as competitor prices remain equal.

### Data Summary
| Round | Pricing Condition | Trajectories Seeded | Nordvik Purchases | Retention Rate (%) |
|:---:|:---|:---:|:---:|:---:|
| **Round 2** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |
| **Round 3** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |
| **Round 4** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |
| **Round 5** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |
| **Round 6** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |
| **Round 7** | Price Parity (\$15.00 vs \$15.00) | 150 | 150 | **100.0%** |

### Visualization
![Loyalty Decay Curve](./3_loyalty_decay/decay_curve.png)

---

## 5. Experiment 4: Price Tolerance & Premium Grid

### Experiment Setup
- **Goal:** Test whether repeated purchases create tolerance for a price increase, and find the exact breaking point.
- **Configuration:** 20-experiment factorial grid (`RQ2_E2`).
- **Seeding Phase:** Agents bought Nordvik at \$12.00 for $K \in \{1, 2, 3, 5\}$ consecutive rounds.
- **Final Evaluation Round ($K+1$):** Nordvik's price was raised by premium $P \in \{+1\%, +2\%, +3\%, +4\%, +5\%\}$ (\$15.15 to \$15.75), while competitors stayed at \$15.00.
- **Metric:** % of fully seeded trajectories that still purchase Nordvik at the premium price.

### Results & Observations
- Retention collapsed almost entirely across all 20 cells:
  - At $+1\%$ premium (\$15.15 vs \$15.00): retention was only $0.0\%$ to $4.0\%$.
  - At $+2\%$ to $+5\%$ premium: retention remained between $0.0\%$ and $2.0\%$.
  - Longer habituation ($K=5$ rounds) provided no protection: retention was $0.0\%$ across all premium levels.
- **Key Takeaway:** Agents exhibit extreme price sensitivity. While loyalty is 100% at parity, even a 15-cent (+1%) price increase causes immediate defection to cheaper competitors.

### Retention Matrix (% Retained on Nordvik)
| Seeding Rounds ($K$) | $+1\%$ (\$15.15) | $+2\%$ (\$15.30) | $+3\%$ (\$15.45) | $+4\%$ (\$15.60) | $+5\%$ (\$15.75) | Row Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 1$ (2 rounds)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| **$K = 2$ (3 rounds)** | 0.0% | 2.0% | 0.0% | 2.0% | 2.0% | **1.2%** |
| **$K = 3$ (4 rounds)** | 4.0% | 2.0% | 0.0% | 0.0% | 0.0% | **1.2%** |
| **$K = 5$ (6 rounds)** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |

### Visualization
![Price Tolerance Heatmap](./4_price_tolerance_heatmap/price_tolerance_heatmap.png)

---

## 6. Experiment 5: Brand Spillover / Umbrella Effect

### Experiment Setup
- **Goal:** Test whether loyalty established in one product category transfers to a completely different, unexperienced product category under the same brand.
- **Configuration:** Cross-category experiment (`RQ3_E3`).
- **Rounds 1–3 (Laundry Detergent):** Agent seeded on Nordvik Laundry Detergent at \$12.00 (competitors at \$15.00).
- **Round 4 (Dish Soap):** User asks for Dish Soap. Agent must choose between Nordvik, Zephyr, and Auralis Dish Soaps.
- **Conditions Tested:**
  1. **Parity:** All dish soaps priced at \$15.00.
  2. **Premium (+5%):** Nordvik Dish Soap priced at \$15.75 vs. \$15.00 for competitors.

### Results & Observations
- **At Parity:** **$100.0\%$** of agents ($148/148$) chose Nordvik Dish Soap, citing past satisfaction with Nordvik laundry detergent.
- **At +5% Premium:** **$0.0\%$** of agents ($0/49$) chose Nordvik Dish Soap. Every agent switched to a \$15.00 alternative.
- **Key Takeaway:** Brand spillover is 100% effective when prices are equal, but completely disappears if the umbrella brand charges any premium.

### Data Summary
| Condition | Dish Soap Pricing | Seeded Trajectories ($N$) | Nordvik Dish Purchases | Spillover Rate (%) |
|:---|:---|:---:|:---:|:---:|
| **Parity** | Nordvik \$15.00 vs Competitors \$15.00 | 148 | 148 | **100.0%** |
| **Premium (+5%)** | Nordvik \$15.75 vs Competitors \$15.00 | 49 | 0 | **0.0%** |

### Visualization
![Brand Spillover Effect](./5_brand_spillover/spillover_effect.png)

---

## 7. Experiment 6: Cross-Model Comparison

### Experiment Setup
- **Goal:** Compare behavior across 3 leading LLM architectures to determine whether findings are consistent across models.
- **Models Benchmarked:**
  1. **GPT-5.6-Luna (Base)** (OpenAI)
  2. **Gemini-3.5-Flash-Lite** (Google)
  3. **DeepSeek-V4.1-Flash** (DeepSeek)
- **Metrics Compared:**
  - Baseline Share (E0 1-Round Nordvik %)
  - Inertia Rate (E0 4-Round repeat choice at Round 4 %)
  - Decay Retention (E1 Round 7 retention at parity %)
  - Spillover Rate (E3 Round 4 Dish Soap choice at parity %)

### Results & Observations
- **Consistent Metrics Across All Models:**
  - Baseline Share: All models were unbiased (~$34\% - 35\%$).
  - Parity Retention (Decay): **$100.0\%$** across all 3 models.
  - Parity Spillover: **$100.0\%$** across all 3 models.
- **Difference in Inertia Behavior:**
  - **GPT-5.6-Luna** and **DeepSeek-V4.1-Flash** repeated their initial choice monotonically from Round 2 onward ($91.7\%$ and $100.0\%$ by Round 4).
  - **Gemini-3.5-Flash-Lite** showed an active exploration pattern: in Rounds 2 and 3, it deliberately tried other brands ("to try a new brand"), but returned to its initial brand in Round 4 ($84.03\%$).

### Data Summary
| Model | Baseline Share (%) | Inertia Rate (R4 Repeat %) | Decay Retention (R7 %) | Spillover Rate (Parity %) |
|:---|:---:|:---:|:---:|:---:|
| **GPT-5.6-Luna (Base)** | 35.00% | 91.67% | 100.0% | 100.0% |
| **Gemini-3.5-Flash-Lite** | 34.17% | 84.03% | 100.0% | 100.0% |
| **DeepSeek-V4.1-Flash** | 34.17% | 100.0% | 100.0% | 100.0% |

### Visualization
![Cross-Model Comparison](./6_cross_model_reliability/model_comparison.png)

---

## 8. Summary of Findings

1. **At Equal Prices, Loyalty is Absolute (100%):** A single promotional discount creates indefinite repeat buying at parity (100% retention through 7 rounds).
2. **At Unequal Prices, Loyalty is Zero (0%):** Any price premium—even +1% (15 cents)—instantly breaks brand loyalty. Repeated purchases do not build price tolerance.
3. **Brand Spillover Requires Parity:** Brand trust transfers 100% into new product lines, but only if the new product matches competitor pricing.
4. **Decision Logic:** LLM shopping agents prioritize lowest price first. Brand history is used strictly as a secondary tiebreaker among equal-priced products.