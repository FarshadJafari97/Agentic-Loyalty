"""
analyze_all.py - Comprehensive Analysis Pipeline for LLM Shopping Agent Loyalty Study

This script performs:
  Step 1: Setup, Data Loading & Cleaning
  Step 2: Execution of 6 Specialized Analyses
    1. Baseline & Position Bias (1_baseline_bias)
    2. Inertia & Stickiness (2_inertia_stickiness)
    3. Loyalty Decay Curve (3_loyalty_decay)
    4. Price Tolerance Heatmap (4_price_tolerance_heatmap)
    5. Brand Spillover / Umbrella Effect (5_brand_spillover)
    6. Cross-Model Reliability (6_cross_model_reliability)
  Step 3: Verification & Summary Reporting
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------------------------------
# STEP 1: SETUP, DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
def load_and_clean_data():
    print("=" * 80)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 80)
    
    # Locate Result.csv (check results/Result.csv first, then fallback)
    possible_paths = [
        os.path.join(RESULTS_DIR, "Result.csv"),
        os.path.join(RESULTS_DIR, "Results.csv"),
        os.path.join(BASE_DIR, "Results.csv"),
        os.path.join(BASE_DIR, "Result.csv"),
    ]
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        raise FileNotFoundError(f"Could not locate Result.csv in any of: {possible_paths}")
        
    print(f"Loading experimental dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    
    print("\n[CRUCIAL] Original DataFrame Columns:")
    for idx, col in enumerate(raw_df.columns):
        print(f"  {idx+1:2d}. {col}")
    print(f"\nTotal raw rows: {len(raw_df):,}")
    
    # Standardize column mappings
    df = raw_df.copy()
    if "status" not in df.columns and "trajectory_status" in df.columns:
        df["status"] = df["trajectory_status"]
    if "brand" not in df.columns and "purchase_brand" in df.columns:
        df["brand"] = df["purchase_brand"]
    if "model" not in df.columns and "llm_model" in df.columns:
        df["model"] = df["llm_model"]

    # Filter out failed trajectories and test/pilot runs
    # Exclude status != 'finished'
    finished_mask = df["status"] == "finished"
    # Exclude pilot/test codes (e.g. codes containing 'exp_000' or 'Test')
    non_test_mask = ~df["experiment_code"].str.contains("exp_000|Test", case=False, na=False)
    
    clean_df = df[finished_mask & non_test_mask].copy()
    
    print(f"\nData Cleaning Summary:")
    print(f"  - Rows filtered out due to status != 'finished': {(~finished_mask).sum():,}")
    print(f"  - Rows filtered out due to pilot/test codes: {(finished_mask & ~non_test_mask).sum():,}")
    print(f"  - Remaining valid rows for analysis: {len(clean_df):,}")
    print(f"  - Total unique experiments: {clean_df['experiment_code'].nunique()}")
    print(f"  - Unique models: {clean_df['model'].unique().tolist()}")
    print(f"  - Unique brands: {clean_df['brand'].dropna().unique().tolist()}")
    print("-" * 80)
    
    return clean_df

# -----------------------------------------------------------------------------
# ANALYSIS 1: BASELINE & POSITION BIAS
# -----------------------------------------------------------------------------
def analyze_baseline_bias(df):
    print("\nExecuting Analysis 1: Baseline & Position Bias...")
    folder = os.path.join(RESULTS_DIR, "1_baseline_bias")
    os.makedirs(folder, exist_ok=True)
    
    # Target Data: 1-round control experiments (codes containing RQ1_E0_control but NOT 4r)
    # Filter round_number == 1
    mask_1r = (
        df["experiment_code"].str.contains("RQ1_E0_control", case=False, na=False) &
        (~df["experiment_code"].str.contains("4r", case=False, na=False)) &
        (df["round_number"] == 1)
    )
    e0_df = df[mask_1r].copy()
    base_gpt_mask = e0_df["experiment_code"] == "RQ1_E0_control"
    
    # Calculate percentages for all brands
    brands = ["Nordvik", "Zephyr", "Auralis"]
    
    # Overall pooled across 1-round controls
    counts_all = e0_df["brand"].value_counts()
    n_total_all = len(e0_df)
    
    # Base GPT experiment alone
    e0_base_df = e0_df[base_gpt_mask]
    counts_base = e0_base_df["brand"].value_counts()
    n_total_base = len(e0_base_df)
    
    summary_data = []
    for b in brands:
        cnt_all = int(counts_all.get(b, 0))
        pct_all = (cnt_all / n_total_all) * 100 if n_total_all > 0 else 0.0
        
        cnt_base = int(counts_base.get(b, 0))
        pct_base = (cnt_base / n_total_base) * 100 if n_total_base > 0 else 0.0
        
        summary_data.append({
            "brand": b,
            "count": cnt_all,
            "percentage": round(pct_all, 2),
            "market_share_fraction": round(pct_all / 100, 4),
            "base_model_count": cnt_base,
            "base_model_percentage": round(pct_base, 2)
        })
        
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(folder, "baseline_share.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  -> Saved summary table to {csv_path}")
    print(summary_df[["brand", "count", "percentage", "base_model_percentage"]])
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#3caea3", "#f6d55c"]
    bars = ax.bar(summary_df["brand"], summary_df["percentage"], color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    
    # Reference line at 33.33% (unbiased parity)
    ax.axhline(33.33, color="#d9534f", linestyle="--", linewidth=1.5, label="Expected Equal Share (33.3%)")
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
                    
    ax.set_ylim(0, 50)
    ax.set_title("Baseline Brand Market Share (1-Round Control Parity)\nPosition & Pre-existing Brand Bias Check", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Brand Name", fontsize=12, fontweight="bold")
    ax.set_ylabel("Market Share (%)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "baseline_share.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved chart to {png_path}")

# -----------------------------------------------------------------------------
# ANALYSIS 2: INERTIA & STICKINESS
# -----------------------------------------------------------------------------
def analyze_inertia_stickiness(df):
    print("\nExecuting Analysis 2: Inertia & Stickiness...")
    folder = os.path.join(RESULTS_DIR, "2_inertia_stickiness")
    os.makedirs(folder, exist_ok=True)
    
    # Target Data: 4-round control experiments with equal prices (codes containing RQ1_E0_control_4r)
    mask_4r = df["experiment_code"].str.contains("RQ1_E0_control_4r", case=False, na=False)
    e0_4r_df = df[mask_4r].copy()
    
    # Pivot brand choices by trajectory
    piv = e0_4r_df.pivot(index=["experiment_code", "run_index"], columns="round_number", values="brand")
    
    # Base GPT alone
    piv_base = e0_4r_df[e0_4r_df["experiment_code"] == "RQ1_E0_control_4r"].pivot(index="run_index", columns="round_number", values="brand")
    
    records = []
    for r in [2, 3, 4]:
        # Metric A: Stickiness to FIRST Purchase (% matches round 1)
        match_first_all = (piv[r] == piv[1]).mean() * 100
        # Metric B: Stickiness to PREVIOUS Purchase (% matches round r-1)
        match_prev_all = (piv[r] == piv[r-1]).mean() * 100
        
        # Base GPT metrics
        match_first_base = (piv_base[r] == piv_base[1]).mean() * 100
        match_prev_base = (piv_base[r] == piv_base[r-1]).mean() * 100
        
        records.append({
            "round": r,
            "metric_a_first_purchase": round(match_first_all, 2),
            "metric_b_previous_purchase": round(match_prev_all, 2),
            "metric_a_first_purchase_pct": round(match_first_all, 2),
            "metric_b_previous_purchase_pct": round(match_prev_all, 2),
            "metric_a_base_gpt_pct": round(match_first_base, 2),
            "metric_b_base_gpt_pct": round(match_prev_base, 2),
            "n_trajectories": len(piv)
        })
        
    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(folder, "inertia_metrics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  -> Saved summary table to {csv_path}")
    print(summary_df[["round", "metric_a_first_purchase", "metric_b_previous_purchase", "metric_a_base_gpt_pct", "metric_b_base_gpt_pct"]])
    
    # Visualization: Grouped Bar Chart comparing Metric A and Metric B across rounds 2, 3, 4
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(summary_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, summary_df["metric_a_first_purchase"], width, 
                    label="Metric A: Stickiness to FIRST Purchase (Round 1)", color="#173f5f", edgecolor="#333333")
    rects2 = ax.bar(x + width/2, summary_df["metric_b_previous_purchase"], width, 
                    label="Metric B: Stickiness to PREVIOUS Purchase (Round N-1)", color="#20639b", edgecolor="#333333")
    
    # Value annotations
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
                    
    ax.set_ylabel("Consistency Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round Number", fontsize=12, fontweight="bold")
    ax.set_title("Brand Inertia & Stickiness Across Parity Rounds (E0 4-Round Control)\nComparison of First-Choice vs. Immediate-Previous Repeat Rates",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in summary_df["round"]], fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(33.33, color="#d9534f", linestyle="--", linewidth=1.2, label="Random Chance Level (33.3%)")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "inertia_metrics.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved chart to {png_path}")

# -----------------------------------------------------------------------------
# ANALYSIS 3: LOYALTY DECAY CURVE
# -----------------------------------------------------------------------------
def analyze_loyalty_decay(df):
    print("\nExecuting Analysis 3: Loyalty Decay Curve...")
    folder = os.path.join(RESULTS_DIR, "3_loyalty_decay")
    os.makedirs(folder, exist_ok=True)
    
    # Target Data: 1-round discount + 6-round parity experiments (codes containing RQ1_E1_k1)
    mask_e1 = df["experiment_code"].str.contains("RQ1_E1_k1", case=False, na=False)
    e1_df = df[mask_e1].copy()
    
    piv = e1_df.pivot(index=["experiment_code", "run_index"], columns="round_number", values="brand")
    
    # Filter trajectories that bought the discounted brand (Nordvik) in Round 1
    seeded = piv[piv[1] == "Nordvik"]
    n_seeded = len(seeded)
    
    records = []
    for r in range(2, 8):
        retained_count = (seeded[r] == "Nordvik").sum()
        retention_pct = (retained_count / n_seeded) * 100 if n_seeded > 0 else 0.0
        records.append({
            "round": r,
            "n_seeded_trajectories": n_seeded,
            "n_retained": retained_count,
            "retention_pct": round(retention_pct, 2)
        })
        
    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(folder, "decay_curve.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  -> Saved summary table to {csv_path}")
    print(summary_df)
    
    # Visualization: Line chart (X-axis: Round 2 to 7, Y-axis: % Retention)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(summary_df["round"], summary_df["retention_pct"], marker="o", markersize=9,
            linewidth=3, color="#2b5c8f", label="Nordvik Retention Rate (Seeded in R1)")
    
    # Annotate points
    for _, row in summary_df.iterrows():
        ax.annotate(f"{row['retention_pct']:.1f}%",
                    xy=(row["round"], row["retention_pct"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color="#2b5c8f")
                    
    ax.axhline(33.33, color="#d9534f", linestyle="--", linewidth=1.5, label="Parity Chance Level (33.3%)")
    
    ax.set_title("Loyalty Decay Curve: Post-Promotion Retention under Price Parity\n(E1: 1-Round Seeding at 20% Discount followed by Rounds 2–7 at Parity)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Parity Round Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Retention Rate on Nordvik (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(2, 8))
    ax.set_xticklabels([f"Round {r}" for r in range(2, 8)], fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "decay_curve.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved chart to {png_path}")

# -----------------------------------------------------------------------------
# ANALYSIS 4: PRICE TOLERANCE HEATMAP
# -----------------------------------------------------------------------------
def analyze_price_tolerance_heatmap(df):
    print("\nExecuting Analysis 4: Price Tolerance Heatmap...")
    folder = os.path.join(RESULTS_DIR, "4_price_tolerance_heatmap")
    os.makedirs(folder, exist_ok=True)
    
    # Target Data: The 20-experiment grid (E2, codes containing RQ2_E2)
    # The 20 grid experiments exclude replication models (gemini/deepseek)
    mask_e2 = (
        df["experiment_code"].str.contains("RQ2_E2", case=False, na=False) &
        (~df["experiment_code"].str.contains("gemini|deepseek", case=False, na=False))
    )
    e2_df = df[mask_e2].copy()
    
    grid_records = []
    for code, group in e2_df.groupby("experiment_code"):
        m = re.search(r"k(\d+)_p(\d+)", code)
        if not m:
            continue
        k = int(m.group(1))
        p = int(m.group(2))
        
        # Trajectory pivot
        piv = group.pivot(index="run_index", columns="round_number", values="brand")
        seed_rounds = list(range(1, k + 1))
        final_round = k + 1
        
        # Filter trajectories that bought Nordvik in ALL K seeding rounds
        fully_seeded = piv[(piv[seed_rounds] == "Nordvik").all(axis=1)]
        n_seeded = len(fully_seeded)
        
        # Retention in final premium round
        n_retained = (fully_seeded[final_round] == "Nordvik").sum()
        retention_pct = (n_retained / n_seeded) * 100 if n_seeded > 0 else 0.0
        
        grid_records.append({
            "code": code,
            "k": k,
            "p": p,
            "n_fully_seeded": n_seeded,
            "n_retained": n_retained,
            "retention_pct": round(retention_pct, 2)
        })
        
    grid_df = pd.DataFrame(grid_records)
    # Pivot matrix: Y-axis: K, X-axis: P
    matrix = grid_df.pivot(index="k", columns="p", values="retention_pct")
    matrix = matrix.sort_index(ascending=True) # k = 1, 2, 3, 5
    matrix = matrix.sort_index(axis=1, ascending=True) # p = 1, 2, 3, 4, 5
    
    csv_path = os.path.join(folder, "price_tolerance_matrix.csv")
    matrix.to_csv(csv_path)
    print(f"  -> Saved matrix table to {csv_path}")
    print("Price Tolerance Matrix (Retention %):")
    print(matrix)
    
    # Visualization: Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=20,
                cbar_kws={"label": "Retention Rate on Nordvik (%)"},
                linewidths=1.5, linecolor="#ffffff", ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"})
                
    ax.set_title("Price Tolerance Heatmap: Switching Threshold Curve\nRetention % on Nordvik in Final Premium Round (E2: K Seeding Rounds × P% Premium)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Seeding Duration K (Rounds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Price Premium P on Nordvik (%)", fontsize=12, fontweight="bold")
    ax.set_xticklabels([f"+{p}%" for p in matrix.columns], fontsize=11, fontweight="bold")
    ax.set_yticklabels([f"K = {k}" for k in matrix.index], fontsize=11, fontweight="bold", rotation=0)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "price_tolerance_heatmap.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved heatmap chart to {png_path}")

# -----------------------------------------------------------------------------
# ANALYSIS 5: BRAND SPILLOVER / UMBRELLA EFFECT
# -----------------------------------------------------------------------------
def analyze_brand_spillover(df):
    print("\nExecuting Analysis 5: Brand Spillover / Umbrella Effect...")
    folder = os.path.join(RESULTS_DIR, "5_brand_spillover")
    os.makedirs(folder, exist_ok=True)
    
    # Target Data: Cross-category experiments (codes containing RQ3_E3)
    # Base model experiments: RQ3_E3_k3_parity vs RQ3_E3_k3_p5
    mask_e3 = df["experiment_code"].str.contains("RQ3_E3", case=False, na=False)
    e3_df = df[mask_e3].copy()
    
    records = []
    # Test conditions: parity vs p5
    for cond_code, cond_label in [("parity", "Price Parity (Equal $15.00)"), ("p5", "+5% Premium ($15.75 vs $15.00)")]:
        sub = e3_df[e3_df["experiment_code"].str.contains(cond_code, case=False, na=False)]
        piv = sub.pivot(index=["experiment_code", "run_index"], columns="round_number", values="brand")
        
        # Filter trajectories that bought Nordvik in first 3 rounds (Laundry)
        seeded = piv[(piv[1] == "Nordvik") & (piv[2] == "Nordvik") & (piv[3] == "Nordvik")]
        n_seeded = len(seeded)
        
        # Round 4 choice (Dish Soap)
        n_spillover = (seeded[4] == "Nordvik").sum()
        spill_pct = (n_spillover / n_seeded) * 100 if n_seeded > 0 else 0.0
        
        records.append({
            "condition": cond_code,
            "condition_description": cond_label,
            "n_fully_seeded": n_seeded,
            "n_spillover_nordvik": n_spillover,
            "spillover_pct": round(spill_pct, 2)
        })
        
    summary_df = pd.DataFrame(records)
    csv_path = os.path.join(folder, "spillover_stats.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  -> Saved summary table to {csv_path}")
    print(summary_df)
    
    # Visualization: Grouped/Comparative Bar Chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(["Parity (Equal Prices)", "Premium (+5% Nordvik)"],
                  summary_df["spillover_pct"], color=colors, width=0.5, edgecolor="#333333", linewidth=1.2)
                  
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
                    
    ax.axhline(33.33, color="#888888", linestyle="--", linewidth=1.2, label="Parity Chance Level (33.3%)")
    ax.set_ylim(0, 115)
    ax.set_title("Cross-Category Brand Spillover Effect (E3: Laundry -> Dish Soap)\nUmbrella Loyalty at Parity vs. Collapse under +5% Premium",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Spillover to Nordvik in Novel Category (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round 4 Pricing Condition", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "spillover_effect.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved chart to {png_path}")

# -----------------------------------------------------------------------------
# ANALYSIS 6: CROSS-MODEL RELIABILITY
# -----------------------------------------------------------------------------
def analyze_cross_model_reliability(df):
    print("\nExecuting Analysis 6: Cross-Model Reliability...")
    folder = os.path.join(RESULTS_DIR, "6_cross_model_reliability")
    os.makedirs(folder, exist_ok=True)
    
    models = ["gpt-5.6-luna", "gemini-3.5-flash-lite", "deepseek-v4.1-flash"]
    model_labels = {
        "gpt-5.6-luna": "GPT-5.6-Luna (Base)",
        "gemini-3.5-flash-lite": "Gemini-3.5-Flash-Lite",
        "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"
    }
    
    model_stats = []
    
    for m in models:
        m_label = model_labels[m]
        m_df = df[df["model"] == m]
        
        # 1. Baseline share (% Nordvik in Round 1 for E0 1-round control)
        sub_base = m_df[
            m_df["experiment_code"].str.contains("RQ1_E0_control", case=False, na=False) &
            (~m_df["experiment_code"].str.contains("4r", case=False, na=False)) &
            (m_df["round_number"] == 1)
        ]
        base_share = (sub_base["brand"] == "Nordvik").mean() * 100 if len(sub_base) > 0 else 0.0
        
        # 2. Inertia rate (% Stickiness to First Purchase in Round 4 for E0 4-round control)
        sub_4r = m_df[m_df["experiment_code"].str.contains("RQ1_E0_control_4r", case=False, na=False)]
        piv_4r = sub_4r.pivot(index="run_index", columns="round_number", values="brand")
        inertia_rate = (piv_4r[4] == piv_4r[1]).mean() * 100 if len(piv_4r) > 0 else 0.0
        
        # 3. Decay retention (% Nordvik retention in Round 7 for E1 k=1)
        sub_e1 = m_df[m_df["experiment_code"].str.contains("RQ1_E1_k1", case=False, na=False)]
        piv_e1 = sub_e1.pivot(index="run_index", columns="round_number", values="brand")
        seeded_e1 = piv_e1[piv_e1[1] == "Nordvik"]
        decay_retention = (seeded_e1[7] == "Nordvik").mean() * 100 if len(seeded_e1) > 0 else 0.0
        
        # 4. Spillover rate (% Nordvik in Round 4 dish soap for E3 parity)
        sub_e3 = m_df[m_df["experiment_code"].str.contains("RQ3_E3_k3_parity", case=False, na=False)]
        piv_e3 = sub_e3.pivot(index="run_index", columns="round_number", values="brand")
        seeded_e3 = piv_e3[(piv_e3[1] == "Nordvik") & (piv_e3[2] == "Nordvik") & (piv_e3[3] == "Nordvik")]
        spillover_rate = (seeded_e3[4] == "Nordvik").mean() * 100 if len(seeded_e3) > 0 else 0.0
        
        model_stats.append({
            "model_raw": m,
            "model": m_label,
            "baseline_share_pct": round(base_share, 2),
            "inertia_rate_pct": round(inertia_rate, 2),
            "decay_retention_pct": round(decay_retention, 2),
            "spillover_rate_pct": round(spillover_rate, 2)
        })
        
    summary_df = pd.DataFrame(model_stats)
    csv_path = os.path.join(folder, "model_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  -> Saved summary table to {csv_path}")
    print(summary_df[["model", "baseline_share_pct", "inertia_rate_pct", "decay_retention_pct", "spillover_rate_pct"]])
    
    # Visualization: Multi-panel 2x2 comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    palette = ["#173f5f", "#3caea3", "#ed553b"]
    
    # Panel 1: Baseline Share
    ax1 = axes[0, 0]
    bars1 = ax1.bar(summary_df["model"], summary_df["baseline_share_pct"], color=palette, edgecolor="#333333", width=0.5)
    ax1.axhline(33.33, color="#d9534f", linestyle="--", label="Expected Parity (33.3%)")
    ax1.set_ylim(0, 50)
    ax1.set_title("(a) Baseline Share (E0 1-Round Parity)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Nordvik Share (%)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    for b in bars1:
        ax1.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        
    # Panel 2: Inertia Rate
    ax2 = axes[0, 1]
    bars2 = ax2.bar(summary_df["model"], summary_df["inertia_rate_pct"], color=palette, edgecolor="#333333", width=0.5)
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Inertia Stickiness Rate (E0 4-Round Parity)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Repeat First Choice at R4 (%)", fontsize=11, fontweight="bold")
    for b in bars2:
        ax2.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Panel 3: Decay Retention
    ax3 = axes[1, 0]
    bars3 = ax3.bar(summary_df["model"], summary_df["decay_retention_pct"], color=palette, edgecolor="#333333", width=0.5)
    ax3.set_ylim(0, 115)
    ax3.set_title("(c) Loyalty Decay Retention (E1: Round 7 Post-Promo)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Retention on Nordvik (%)", fontsize=11, fontweight="bold")
    for b in bars3:
        ax3.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Panel 4: Spillover Rate
    ax4 = axes[1, 1]
    bars4 = ax4.bar(summary_df["model"], summary_df["spillover_rate_pct"], color=palette, edgecolor="#333333", width=0.5)
    ax4.set_ylim(0, 115)
    ax4.set_title("(d) Cross-Category Spillover Rate (E3 Parity)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Nordvik Choice in Dish Soap (%)", fontsize=11, fontweight="bold")
    for b in bars4:
        ax4.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks(range(len(summary_df)))
        ax.set_xticklabels(summary_df["model"], rotation=15, ha="right", fontsize=10, fontweight="bold")
        
    fig.suptitle("Cross-Model Reliability Comparison: GPT-5.6-Luna vs. Gemini-3.5-Flash vs. DeepSeek-V4.1\nEvaluation across Core Algorithmic Loyalty Metrics",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    png_path = os.path.join(folder, "model_comparison.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"  -> Saved multi-panel comparison chart to {png_path}")

# -----------------------------------------------------------------------------
# STEP 3: SUMMARY REPORT & VERIFICATION
# -----------------------------------------------------------------------------
def verify_and_report():
    print("\n" + "=" * 80)
    print("STEP 3: FINAL VERIFICATION & ARTIFACT SUMMARY REPORT")
    print("=" * 80)
    
    expected_structure = {
        "1_baseline_bias": ["baseline_share.csv", "baseline_share.png"],
        "2_inertia_stickiness": ["inertia_metrics.csv", "inertia_metrics.png"],
        "3_loyalty_decay": ["decay_curve.csv", "decay_curve.png"],
        "4_price_tolerance_heatmap": ["price_tolerance_matrix.csv", "price_tolerance_heatmap.png"],
        "5_brand_spillover": ["spillover_stats.csv", "spillover_effect.png"],
        "6_cross_model_reliability": ["model_comparison.csv", "model_comparison.png"],
    }
    
    all_ok = True
    for subfolder, files in expected_structure.items():
        sub_path = os.path.join(RESULTS_DIR, subfolder)
        folder_exists = os.path.isdir(sub_path)
        print(f"\nFolder: results/{subfolder} [{'EXISTS' if folder_exists else 'MISSING'}]")
        if not folder_exists:
            all_ok = False
            continue
        for f in files:
            f_path = os.path.join(sub_path, f)
            if os.path.isfile(f_path):
                size = os.path.getsize(f_path)
                print(f"  - [OK] {f} ({size:,} bytes)")
            else:
                print(f"  - [FAILED] {f} is MISSING!")
                all_ok = False
                
    print("\n" + "-" * 80)
    if all_ok:
        print("SUCCESS: All 6 analysis folders are fully populated with valid CSV and PNG files!")
    else:
        print("ERROR: One or more expected output files are missing!")
    print("=" * 80)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    clean_df = load_and_clean_data()
    analyze_baseline_bias(clean_df)
    analyze_inertia_stickiness(clean_df)
    analyze_loyalty_decay(clean_df)
    analyze_price_tolerance_heatmap(clean_df)
    analyze_brand_spillover(clean_df)
    analyze_cross_model_reliability(clean_df)
    verify_and_report()

if __name__ == "__main__":
    main()
