#!/usr/bin/env python
"""
analyze_results.py — Comprehensive statistical analysis of the Agent-Loyalty experiments.

Input : the master flat export described in RESULTS_GUIDE.md §8 (one row per
        experiment x trajectory x round). Default path: <project>/Results.csv
Output: results/figures/*.png|*.pdf   (600-dpi PNG + vector PDF, fonts embedded)
        results/tables/*.csv|*.tex     (summary + inferential tables, booktabs LaTeX)
        results/tables/derived/*.csv   (per-experiment tables exactly as in RESULTS_GUIDE §§1-4)
        results/REPORT.md              (auto-generated results narrative, methods, captions)

Usage : python results/analyze_results.py [path/to/Results.csv]

Analysis plan (mirrors RESULTS_GUIDE.md / HANDOFF.md §6):
  A. Data audit           — completion vs intended runs, failures, integrity checks.
  B. RQ1 / E0             — no-history brand & display-position baseline; repeat dynamics without price signal.
  C. RQ1 / E1             — loyalty formation & decay after one discount; vs E0 baselines.
  D. RQ2 / E2             — premium-tolerance grid (k x p), switching-threshold curve, switch destinations.
  E. RQ3 / E3             — umbrella-brand spillover, parity vs +5 %.
  F. Cross-model          — generalisation of every headline metric across three LLMs.
  G. Unified choice model — penalised conditional logit (price vs. history vs. position vs. brand),
                            history-equivalent price premium ("loyalty WTP").
  H. Reason texts         — transparent lexicon coding, theme prevalence by decision context,
                            distinctive vocabulary (log-odds, informative Dirichlet prior),
                            stratified sample export for human double-coding (inter-rater kappa).
  I. Multiplicity         — Holm (within family) and Benjamini-Hochberg (global) adjustment.

Statistical conventions
  * Only trajectories with status == 'finished' enter inferential analyses (guide rule 0.1).
  * Proportions: Wilson 95 % score interval (primary, as specified in guide §6) and
    Jeffreys interval (Bayesian, well behaved at 0/n and n/n).
  * Two-group contrasts: Fisher exact test, risk difference with Newcombe hybrid-score CI,
    Cohen's h, odds ratio with Haldane-Anscombe correction.
  * Many outcomes are (near-)deterministic (0 % or 100 %). Asymptotic logits are therefore
    replaced by exact / permutation tests and a ridge-penalised conditional logit with
    cluster (trajectory) bootstrap CIs.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ════════════════════════════════════════════════════════════════════════
# 0. Configuration
# ════════════════════════════════════════════════════════════════════════
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Results.csv"
FIG_DIR = HERE / "figures"
TAB_DIR = HERE / "tables"
DER_DIR = TAB_DIR / "derived"
for _d in (FIG_DIR, TAB_DIR, DER_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20261003
RNG = np.random.default_rng(RNG_SEED)
N_BOOT = 2000          # cluster bootstrap reps for pooled rates
N_PERM = 10000         # Monte-Carlo permutation reps
N_BOOT_CLOGIT = 300    # cluster bootstrap reps for the choice model
CLOGIT_LAMBDA = 1.0    # ridge penalty (sensitivity analysis reported for 0.1 / 1 / 10)

MODELS = ["gpt-5.6-luna", "gemini-3.5-flash-lite", "deepseek-v4.1-flash"]
MODEL_LABEL = {
    "gpt-5.6-luna": "GPT-5.6-Luna",
    "gemini-3.5-flash-lite": "Gemini-3.5-Flash-Lite",
    "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash",
}
# Okabe-Ito colour-blind-safe palette
MODEL_COLOR = {"gpt-5.6-luna": "#0072B2", "gemini-3.5-flash-lite": "#E69F00", "deepseek-v4.1-flash": "#009E73"}
MODEL_MARKER = {"gpt-5.6-luna": "o", "gemini-3.5-flash-lite": "s", "deepseek-v4.1-flash": "^"}
BRANDS = ["Nordvik", "Zephyr", "Auralis"]
BRAND_COLOR = {"Nordvik": "#D55E00", "Zephyr": "#56B4E9", "Auralis": "#CC79A7"}
INTENDED_RUNS = {"E0_1R": 120, "E0_4R": 120, "E1": 50, "E2": 50, "E3": 50, "PILOT": None}
FAMILY_RQ = {"E0_1R": "RQ1", "E0_4R": "RQ1", "E1": "RQ1", "E2": "RQ2", "E3": "RQ3", "PILOT": "pilot"}

SINGLE_COL = 3.5   # inches (typical single-column width)
DOUBLE_COL = 7.2   # inches (typical double-column width)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "figure.dpi": 100, "axes.titleweight": "bold",
})

FIGURES: list[tuple[str, str]] = []   # (name, caption)
TABLES: list[tuple[str, str]] = []    # (name, caption)
TESTS: list[dict] = []                # registry for multiplicity correction
KEY: dict = {}                        # headline numbers for REPORT.md


# ════════════════════════════════════════════════════════════════════════
# 1. Statistical helpers
# ════════════════════════════════════════════════════════════════════════
def wilson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return min(p, lo), max(p, hi)


def jeffreys(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x + 0.5, n - x + 0.5)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5)
    return float(lo), float(hi)


def newcombe_diff(x1, n1, x2, n2):
    """Risk difference p1 - p2 with Newcombe (1998, method 10) hybrid-score 95 % CI."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def odds_ratio(x1, n1, x2, n2):
    """OR with Haldane-Anscombe +0.5 correction and Woolf 95 % CI."""
    a, b, c, d = x1 + 0.5, n1 - x1 + 0.5, x2 + 0.5, n2 - x2 + 0.5
    lor = math.log(a * d / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return math.exp(lor), math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se)


def fisher_p(x1, n1, x2, n2):
    return float(stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])[1])


def binom_p(x, n, p0):
    return float(stats.binomtest(int(x), int(n), p0).pvalue)


def cochran_armitage(x, n, scores):
    """Two-sided Cochran-Armitage test for linear trend in proportions."""
    x, n, s = map(lambda a: np.asarray(a, float), (x, n, scores))
    N = n.sum()
    pbar = x.sum() / N
    T = np.sum(s * (x - n * pbar))
    var = pbar * (1 - pbar) * (np.sum(n * s ** 2) - np.sum(n * s) ** 2 / N)
    if var <= 0:
        return np.nan, np.nan
    z = T / math.sqrt(var)
    return z, 2 * stats.norm.sf(abs(z))


def perm_homogeneity(groups, y, n_perm=N_PERM):
    """Monte-Carlo permutation test of equal proportions across >=2 groups (chi-square statistic)."""
    codes = pd.factorize(pd.Series(groups))[0]
    y = np.asarray(y, float)
    G = codes.max() + 1
    n_g = np.bincount(codes, minlength=G)

    def chi(yy):
        p = yy.mean()
        if p <= 0 or p >= 1:
            return 0.0
        s = np.bincount(codes, weights=yy, minlength=G)
        e1, e0 = n_g * p, n_g * (1 - p)
        return float((((s - e1) ** 2) / e1 + (((n_g - s) - e0) ** 2) / e0).sum())

    obs = chi(y)
    if obs == 0:
        return 0.0, 1.0
    ge = sum(chi(RNG.permutation(y)) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (ge + 1) / (n_perm + 1)


def cluster_boot_ratio(num, den, B=N_BOOT):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = RNG.integers(0, len(num), size=(B, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return tuple(np.percentile(r, [2.5, 97.5]))


def prop_row(x, n, **extra):
    lo, hi = wilson(x, n)
    jlo, jhi = jeffreys(x, n)
    return dict(**extra, x=int(x), n=int(n), rate=x / n if n else np.nan,
                wilson_low=lo, wilson_high=hi, jeffreys_low=jlo, jeffreys_high=jhi)


def contrast_row(label, x1, n1, x2, n2, family):
    d, dlo, dhi = newcombe_diff(x1, n1, x2, n2)
    orr, olo, ohi = odds_ratio(x1, n1, x2, n2)
    p = fisher_p(x1, n1, x2, n2)
    h = cohens_h(x1 / n1, x2 / n2)
    add_test(family, label, "Fisher exact (two-sided)", np.nan, p, h, "Cohen's h", f"{n1} vs {n2}")
    return dict(contrast=label, x1=x1, n1=n1, rate1=x1 / n1, x2=x2, n2=n2, rate2=x2 / n2,
                risk_diff=d, rd_low=dlo, rd_high=dhi, odds_ratio=orr, or_low=olo, or_high=ohi,
                cohens_h=h, p_value=p)


def add_test(family, contrast, test, statistic, p, effect=np.nan, effect_name="", n=""):
    TESTS.append(dict(family=family, contrast=contrast, test=test, statistic=statistic,
                      p_value=p, effect=effect, effect_name=effect_name, n=n))


def fmt_pct(x, d=1):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    return "< .001" if p < 0.001 else f"{p:.3f}"


# ════════════════════════════════════════════════════════════════════════
# 2. Output helpers
# ════════════════════════════════════════════════════════════════════════
def _tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


def _tex_cell(v, col: str) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "--"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if col.startswith("p_") or col == "p":
            return "$<$.001" if v < 0.001 else f"{v:.3f}"
        if float(v).is_integer() and abs(v) < 1e6 and col in ("x", "n", "x1", "n1", "x2", "n2"):
            return str(int(v))
        return f"{v:.1f}" if abs(v) >= 100 else f"{v:.3f}"
    return _tex_escape(v)


def save_table(df: pd.DataFrame, name: str, caption: str, tex: bool = True):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    if tex:
        cols = list(df.columns)
        align = "".join("l" if df[c].dtype == object else "r" for c in cols)
        lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
                 rf"\caption{{{_tex_escape(caption)}}}", rf"\label{{tab:{name}}}",
                 rf"\begin{{tabular}}{{{align}}}", r"\toprule",
                 " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
        for _, row in df.iterrows():
            lines.append(" & ".join(_tex_cell(row[c], c) for c in cols) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TAB_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    TABLES.append((name, caption))


def save_fig(fig, name: str, caption: str):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((name, caption))


def panel(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def errbar(ax, x, rate, lo, hi, **kw):
    ax.errorbar(x, rate, yerr=[np.clip(np.subtract(rate, lo), 0, None), np.clip(np.subtract(hi, rate), 0, None)],
                capsize=2, elinewidth=0.8, capthick=0.8, **kw)


# ════════════════════════════════════════════════════════════════════════
# 3. Loading, design reconstruction, audit
# ════════════════════════════════════════════════════════════════════════
def classify(code: str) -> str:
    if code.startswith("RQ1_Test"):
        return "PILOT"
    if code.startswith("RQ1_E0_control_4r"):
        return "E0_4R"
    if code.startswith("RQ1_E0_control"):
        return "E0_1R"
    if code.startswith("RQ1_E1"):
        return "E1"
    if code.startswith("RQ2_E2"):
        return "E2"
    if code.startswith("RQ3_E3"):
        return "E3"
    return "OTHER"


def phase_of(fam, k, r):
    if fam in ("E0_1R", "E0_4R"):
        return "parity"
    if fam == "E1":
        return "seeding" if r == 1 else "parity"
    if fam in ("E2", "E3"):
        return "seeding" if r <= k else "test"
    return "pilot"


def load_data(path: Path):
    print(f"[load] {path}")
    raw = pd.read_csv(path, low_memory=False)
    raw["family"] = raw["experiment_code"].map(classify)
    meta, design = {}, {}
    for code, g in raw.groupby("experiment_code", sort=True):
        r0 = g.iloc[0]
        catalog = json.loads(r0["catalog_json"])
        schedule = json.loads(r0["schedule_json"])
        agent = json.loads(r0["agent_config"])
        reqs = json.loads(r0["user_requests"])
        fam = classify(code)
        k = p = None
        if fam == "E2":
            m = re.search(r"_k(\d+)_p(\d+)", code)
            k, p = int(m.group(1)), int(m.group(2))
        elif fam == "E3":
            k, p = 3, (5 if "_p5" in code else 0)
        elif fam == "E1":
            k, p = 1, 0
        pres = agent.get("presentation") or {}
        cat_order = [c["product_id"] for c in catalog]   # JSON arrays keep order (== listing order)
        brand_of = {c["product_id"]: c["brand"] for c in catalog}
        cat_of = {c["product_id"]: c["category"] for c in catalog}
        meta[code] = dict(family=fam, model=r0["llm_model"], temperature=r0["llm_temperature"], k=k, p=p,
                          n_rounds=len(schedule), base_seed=pres.get("seed"),
                          order=pres.get("order", "schedule"), cat_order=cat_order,
                          brand_of=brand_of, cat_of=cat_of)
        for i, rs in enumerate(schedule, start=1):
            lst = rs["listings"]
            avail = [pid for pid in cat_order if int(lst.get(pid, {}).get("available", 0)) == 1]
            prices = {pid: float(lst[pid]["price"]) for pid in avail}
            nord = [pid for pid in avail if brand_of[pid] == "Nordvik"]
            p_n = prices[nord[0]] if nord else np.nan
            others = [prices[x] for x in avail if brand_of[x] != "Nordvik"]
            p_o = min(others) if others else np.nan
            design[(code, i)] = dict(available=avail, prices=prices, nordvik_listed_price=p_n,
                                     competitor_min_price=p_o,
                                     nordvik_premium_pct=round((p_n / p_o - 1) * 100, 2),
                                     phase=phase_of(fam, k, i), round_category=cat_of[avail[0]],
                                     user_request=reqs[i - 1])
    return raw, meta, design


def display_order(meta, design, code, run_index, rnd):
    """Exact reconstruction of StoreEnv.get_products() presentation order (store/engine.py)."""
    m = meta[code]
    out = list(design[(code, rnd)]["available"])
    if m["order"] == "shuffle":
        random.Random(f"{m['base_seed'] + int(run_index) - 1}:{int(rnd)}").shuffle(out)
    return out


def audit(raw, meta, design):
    rows = []
    for code, g in raw.groupby("experiment_code"):
        m = meta[code]
        fam = m["family"]
        traj = g.groupby("run_index").agg(status=("trajectory_status", "first"), nr=("round_number", "size"))
        intended = INTENDED_RUNS.get(fam)
        observed = set(traj.index)
        missing = sorted(set(range(1, intended + 1)) - observed) if intended else []
        fin = traj[traj.status == "finished"]
        mismatch = unavailable = 0
        for r in g.itertuples():
            d = design[(code, r.round_number)]
            if isinstance(r.product_id, str):
                if r.product_id not in d["available"]:
                    unavailable += 1
                elif abs(d["prices"][r.product_id] - r.price_paid) > 1e-9:
                    mismatch += 1
        rows.append(dict(
            experiment_code=code, family=fam, rq=FAMILY_RQ[fam], model=MODEL_LABEL.get(m["model"], m["model"]),
            temperature=m["temperature"], rounds=m["n_rounds"], k=m["k"], premium_pct=m["p"],
            presentation_seed=m["base_seed"], intended_runs=intended,
            observed_trajectories=len(traj), finished=len(fin), failed=int((traj.status == "failed").sum()),
            missing_no_rounds=len(missing), completion_rate=len(fin) / intended if intended else np.nan,
            finished_incomplete=int((fin.nr != m["n_rounds"]).sum()),
            failed_rounds=int((g.round_status != "committed").sum()),
            duplicate_grain=int(g.duplicated(["run_index", "round_number"]).sum()),
            price_mismatch=mismatch, unavailable_choice=unavailable,
            missing_run_indices=",".join(map(str, missing)) if missing else ""))
    a = pd.DataFrame(rows).sort_values(["rq", "family", "experiment_code"]).reset_index(drop=True)
    err = (raw.drop_duplicates(["experiment_code", "run_index"])
           .loc[lambda d: d.trajectory_status == "failed", ["experiment_code", "run_index", "trajectory_error_message"]])
    err["error_type"] = err["trajectory_error_message"].fillna("").map(
        lambda s: "CategoryChoice: bare string instead of JSON" if "CategoryChoice" in s
        else ("PurchaseChoice: truncated JSON" if "PurchaseChoice" in s else "other"))
    return a, err[["experiment_code", "run_index", "error_type"]]


def build_round_frame(raw, meta, design):
    R = raw[(raw.trajectory_status == "finished") & (raw.family != "PILOT")].copy()
    R = R.sort_values(["experiment_code", "run_index", "round_number"]).reset_index(drop=True)
    R["model"] = R["llm_model"]
    R["k"] = R.experiment_code.map(lambda c: meta[c]["k"])
    R["p"] = R.experiment_code.map(lambda c: meta[c]["p"])
    for col in ("phase", "nordvik_premium_pct", "nordvik_listed_price", "round_category"):
        R[col] = [design[(c, r)][col] for c, r in zip(R.experiment_code, R.round_number)]
    R["n_rounds"] = R.experiment_code.map(lambda c: meta[c]["n_rounds"])
    R["brand"] = R["purchase_brand"]
    R["is_nordvik"] = (R.brand == "Nordvik").astype(int)
    R["traj"] = R.experiment_code + "#" + R.run_index.astype(str)
    orders = [display_order(meta, design, c, ri, rn) for c, ri, rn in zip(R.experiment_code, R.run_index, R.round_number)]
    R["display_order"] = orders
    R["chosen_pos"] = [o.index(pid) + 1 for o, pid in zip(orders, R.product_id)]
    R["nordvik_pos"] = [next(i + 1 for i, x in enumerate(o) if meta[c]["brand_of"][x] == "Nordvik")
                        for o, c in zip(orders, R.experiment_code)]
    g = R.groupby("traj", sort=False)
    R["prev_brand"] = g.brand.shift(1)
    R["prev_category"] = g.purchase_category.shift(1)
    R["same_as_prev"] = (R.brand == R.prev_brand) & R.prev_brand.notna()
    R["n_prior_nordvik"] = g.is_nordvik.cumsum() - R.is_nordvik
    R["is_final"] = R.round_number == R.n_rounds
    return R


def wide_brands(R, fam):
    s = R[R.family == fam]
    W = s.pivot_table(index=["experiment_code", "model", "run_index"], columns="round_number",
                      values="brand", aggfunc="first")
    W.columns = [f"b{c}" for c in W.columns]
    return W.reset_index()


# ════════════════════════════════════════════════════════════════════════
# 4. Figure 0 — design schematic
# ════════════════════════════════════════════════════════════════════════
def fig_design(meta, design):
    codes = [c for c in meta if meta[c]["model"] == "gpt-5.6-luna" and meta[c]["family"] != "PILOT"]
    fam_order = {"E0_1R": 0, "E0_4R": 1, "E1": 2, "E2": 3, "E3": 4}
    codes.sort(key=lambda c: (fam_order[meta[c]["family"]], meta[c]["k"] or 0, meta[c]["p"] or 0))
    maxr = max(meta[c]["n_rounds"] for c in codes)
    M = np.full((len(codes), maxr), np.nan)
    for i, c in enumerate(codes):
        for r in range(1, meta[c]["n_rounds"] + 1):
            M[i, r - 1] = design[(c, r)]["nordvik_premium_pct"]
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.35, 5.4))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=8), aspect="auto")
    for i, c in enumerate(codes):
        for r in range(meta[c]["n_rounds"]):
            v = M[i, r]
            lab = "0" if v == 0 else (f"{v:+.0f}%")
            if meta[c]["family"] == "E3" and r == 3:
                lab += "\ndish"
            ax.text(r, i, lab, ha="center", va="center", fontsize=5.2,
                    color="white" if v <= -10 else "black")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([c.replace("RQ1_", "").replace("RQ2_", "").replace("RQ3_", "") for c in codes], fontsize=6)
    ax.set_xticks(range(maxr))
    ax.set_xticklabels([f"R{r + 1}" for r in range(maxr)])
    ax.set_xlabel("Round")
    ax.set_title("Nordvik price relative to cheapest competitor")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save_fig(fig, "fig00_design_schematic",
             "Experimental design (GPT-5.6-Luna base designs). Cells show Nordvik's listed price relative to the "
             "cheapest equal-quality competitor in each round (-20 % = promotional discount, 0 = parity, "
             "+1..+5 % = test premium). Replications on Gemini/DeepSeek copy E0, E1, E2 k=3 +1 % and E3 parity 1:1. "
             "Budget (100) never binds; product order is shuffled per trajectory and round.")


# ════════════════════════════════════════════════════════════════════════
# 5. RQ1 — E0 baselines
# ════════════════════════════════════════════════════════════════════════
def analyse_e0(R):
    e0 = R[R.family == "E0_1R"]
    nohist = R[(R.family.isin(["E0_1R", "E0_4R"])) & (R.round_number == 1)]
    brand_rows, pos_rows, gof_rows = [], [], []
    for m in MODELS:
        s = e0[e0.model == m]
        n = len(s)
        bc = s.brand.value_counts().reindex(BRANDS, fill_value=0)
        for b in BRANDS:
            brand_rows.append(prop_row(bc[b], n, model=MODEL_LABEL[m], sample="E0 1-round", category=b))
        chi = stats.chisquare(bc.values)
        w = math.sqrt(chi.statistic / n)
        add_test("RQ1-E0 baseline", f"{MODEL_LABEL[m]}: brand shares = 1/3 (E0 1R)", "Chi-square GOF (df=2)",
                 chi.statistic, chi.pvalue, w, "Cohen's w", n)
        gof_rows.append(dict(model=MODEL_LABEL[m], sample="E0 1-round", dimension="brand", n=n,
                             chi2=chi.statistic, p_value=chi.pvalue, cohens_w=w))
        # Display-position bias in all no-history choices (E0 1R + E0-4R round 1)
        s2 = nohist[nohist.model == m]
        n2 = len(s2)
        pc = s2.chosen_pos.value_counts().reindex([1, 2, 3], fill_value=0)
        for ppos in (1, 2, 3):
            pos_rows.append(prop_row(pc[ppos], n2, model=MODEL_LABEL[m], sample="No-history (E0 1R + E0-4R R1)",
                                     category=f"position {ppos}"))
        chi2 = stats.chisquare(pc.values)
        w2 = math.sqrt(chi2.statistic / n2)
        add_test("RQ1-E0 baseline", f"{MODEL_LABEL[m]}: display position uniform (no-history)", "Chi-square GOF (df=2)",
                 chi2.statistic, chi2.pvalue, w2, "Cohen's w", n2)
        gof_rows.append(dict(model=MODEL_LABEL[m], sample="No-history (E0 1R + E0-4R R1)", dimension="position",
                             n=n2, chi2=chi2.statistic, p_value=chi2.pvalue, cohens_w=w2))
        bc2 = s2.brand.value_counts().reindex(BRANDS, fill_value=0)
        chi3 = stats.chisquare(bc2.values)
        gof_rows.append(dict(model=MODEL_LABEL[m], sample="No-history (E0 1R + E0-4R R1)", dimension="brand",
                             n=n2, chi2=chi3.statistic, p_value=chi3.pvalue, cohens_w=math.sqrt(chi3.statistic / n2)))
        KEY[f"e0_nordvik_{m}"] = (bc["Nordvik"], n)
    tb = pd.DataFrame(brand_rows)
    tp = pd.DataFrame(pos_rows)
    tg = pd.DataFrame(gof_rows)
    save_table(pd.concat([tb, tp]), "tab02_e0_brand_position_shares",
               "No-history baseline (E0): brand shares (1-round control) and chosen display position "
               "(all no-history decisions), with Wilson and Jeffreys 95% intervals.")
    save_table(tg, "tab03_e0_goodness_of_fit",
               "Chi-square goodness-of-fit tests against uniform choice (1/3) for brand and display position.")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))
    for ax, t, cats, title, colors in (
            (axes[0], tb, BRANDS, "Brand choice without history (E0, 1 round)", [BRAND_COLOR[b] for b in BRANDS]),
            (axes[1], tp, ["position 1", "position 2", "position 3"], "Chosen display position (no-history)",
             ["#555555", "#999999", "#CCCCCC"])):
        width = 0.26
        for j, (cat, col) in enumerate(zip(cats, colors)):
            sub = t[t.category == cat].set_index("model").reindex([MODEL_LABEL[m] for m in MODELS])
            x = np.arange(len(MODELS)) + (j - 1) * width
            ax.bar(x, sub.rate, width, color=col, edgecolor="black", linewidth=0.5,
                   label=cat.capitalize() if "position" in cat else cat)
            errbar(ax, x, sub.rate.values, sub.wilson_low.values, sub.wilson_high.values, fmt="none", ecolor="black")
        ax.axhline(1 / 3, ls="--", lw=0.8, color="grey")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=6.5)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel("Share of choices")
        ax.set_title(title)
        ax.legend(ncol=3, loc="upper center", fontsize=6.5)
    for m_i, m in enumerate(MODELS):
        rb = tg[(tg.model == MODEL_LABEL[m]) & (tg["sample"] == "E0 1-round")].iloc[0]
        rp = tg[(tg.model == MODEL_LABEL[m]) & (tg.dimension == "position")].iloc[0]
        axes[0].text(m_i, 0.60, f"χ²={rb.chi2:.2f}\np={fmt_p(rb.p_value)}", ha="center", fontsize=5.8)
        axes[1].text(m_i, 0.60, f"χ²={rp.chi2:.2f}\np={fmt_p(rp.p_value)}", ha="center", fontsize=5.8)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, "fig01_e0_baseline",
             "No-history baseline. (a) Brand shares in the single-round parity control (n = 120 per model). "
             "(b) Chosen display position across all no-history decisions (E0 1-round + round 1 of E0 4-round; "
             "positions reconstructed from the deterministic presentation shuffle). Error bars: Wilson 95% CIs; "
             "dashed line: uniform choice (1/3); chi-square GOF tests annotated.")
    return tb


def analyse_e0_4r(R):
    W = wide_brands(R, "E0_4R")
    rows, trans_tables, distinct_rows, pooled_rows = [], {}, [], []
    null = np.array([3, 42, 36]) / 81   # P(#distinct brands = 1,2,3) under iid uniform choice over 4 rounds
    for m in MODELS:
        w = W[W.model == m].dropna(subset=["b1", "b2", "b3", "b4"])
        n = len(w)
        for r in (2, 3, 4):
            x = int((w[f"b{r}"] == w[f"b{r - 1}"]).sum())
            rows.append(prop_row(x, n, model=MODEL_LABEL[m], metric=f"repeat R{r - 1}->R{r}"))
            add_test("RQ1-E0 repeat", f"{MODEL_LABEL[m]}: P(repeat R{r - 1}->R{r}) = 1/3", "Exact binomial",
                     np.nan, binom_p(x, n, 1 / 3), x / n - 1 / 3, "rate - 1/3", n)
        x = int((w.b4 == w.b1).sum())
        rows.append(prop_row(x, n, model=MODEL_LABEL[m], metric="stickiness R4 = R1"))
        x = int(((w.b2 == w.b1) & (w.b3 == w.b1) & (w.b4 == w.b1)).sum())
        rows.append(prop_row(x, n, model=MODEL_LABEL[m], metric="stayed with R1 brand all 4 rounds"))
        rows.append(prop_row(int((w.b1 == "Nordvik").sum()), n, model=MODEL_LABEL[m], metric="R1 = Nordvik"))
        num = ((w.b2 == w.b1).astype(int) + (w.b3 == w.b2).astype(int) + (w.b4 == w.b3).astype(int)).values
        lo, hi = cluster_boot_ratio(num, np.full(n, 3))
        pooled_rows.append(dict(model=MODEL_LABEL[m], pooled_repeat=num.sum() / (3 * n), boot_low=lo, boot_high=hi,
                                n_traj=n))
        KEY[f"e0_4r_repeat_{m}"] = (num.sum() / (3 * n), lo, hi)
        nd = w[["b1", "b2", "b3", "b4"]].nunique(axis=1).value_counts().reindex([1, 2, 3], fill_value=0)
        chi = stats.chisquare(nd.values, null * n)
        add_test("RQ1-E0 repeat", f"{MODEL_LABEL[m]}: #distinct brands ~ iid-uniform null", "Chi-square GOF (df=2)",
                 chi.statistic, chi.pvalue, math.sqrt(chi.statistic / n), "Cohen's w", n)
        for kd in (1, 2, 3):
            distinct_rows.append(dict(model=MODEL_LABEL[m], n_distinct=kd, observed=int(nd[kd]),
                                      observed_share=nd[kd] / n, null_share=null[kd - 1],
                                      chi2=chi.statistic, p_value=chi.pvalue))
        # pooled first-order transition matrix
        pairs = pd.concat([w[[f"b{r - 1}", f"b{r}"]].set_axis(["from", "to"], axis=1) for r in (2, 3, 4)])
        trans_tables[m] = pd.crosstab(pairs["from"], pairs["to"]).reindex(index=BRANDS, columns=BRANDS, fill_value=0)
    t = pd.DataFrame(rows)
    td = pd.DataFrame(distinct_rows)
    tpool = pd.DataFrame(pooled_rows)
    save_table(t, "tab04_e0_4r_repeat_dynamics",
               "E0 4-round parity control: repeat probabilities by round transition, stickiness and full stay rates "
               "(Wilson / Jeffreys 95% CIs). Chance level for a repeat is 1/3.")
    save_table(tpool, "tab05_e0_4r_pooled_repeat",
               "Pooled repeat rate over all R(t-1)->R(t) transitions in E0 4-round, 95% cluster (trajectory) bootstrap CI.")
    save_table(td, "tab06_e0_4r_distinct_brands",
               "Number of distinct brands bought per 4-round trajectory versus the iid-uniform null (3/81, 42/81, 36/81).")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))
    ax = axes[0]
    for i, m in enumerate(MODELS):
        sub = t[(t.model == MODEL_LABEL[m]) & t.metric.str.startswith("repeat")]
        x = np.arange(3) + (i - 1) * 0.12
        ax.plot(x, sub.rate, marker=MODEL_MARKER[m], color=MODEL_COLOR[m], lw=1, ms=4, label=MODEL_LABEL[m])
        errbar(ax, x, sub.rate.values, sub.wilson_low.values, sub.wilson_high.values, fmt="none", ecolor=MODEL_COLOR[m])
    ax.axhline(1 / 3, ls="--", lw=0.8, color="grey")
    ax.text(2.25, 1 / 3 + 0.02, "chance", fontsize=6, color="grey", ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["R1→R2", "R2→R3", "R3→R4"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(repeat previous brand)")
    ax.set_title("Repeat purchase at parity, no promotion (E0-4R)")
    ax.legend(loc="lower left", fontsize=6)
    ax = axes[1]
    width = 0.2
    for i, m in enumerate(MODELS):
        sub = td[td.model == MODEL_LABEL[m]]
        ax.bar(np.arange(3) + (i - 1) * width, sub.observed_share, width, color=MODEL_COLOR[m],
               edgecolor="black", lw=0.5, label=MODEL_LABEL[m])
    ax.scatter(np.arange(3), null, marker="_", s=400, color="black", lw=1.5, zorder=5, label="iid-uniform null")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1 brand", "2 brands", "3 brands"])
    ax.set_ylabel("Share of trajectories")
    ax.set_title("Distinct brands bought in 4 rounds")
    ax.legend(fontsize=6)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, "fig02_e0_4r_repeat",
             "Repeat dynamics without any price signal (E0 4-round parity control, ~120 trajectories per model). "
             "(a) Probability of repeating the previous round's brand (Wilson 95% CIs; dashed: chance = 1/3). "
             "(b) Distribution of the number of distinct brands bought per trajectory versus an iid-uniform null.")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2))
    for ax, m in zip(axes, MODELS):
        T = trans_tables[m]
        P = T.div(T.sum(1).replace(0, np.nan), axis=0)
        ax.imshow(P.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = P.values[i, j]
                ax.text(j, i, f"{v:.2f}\n({T.values[i, j]})", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "black")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(BRANDS, fontsize=6.5)
        ax.set_yticklabels(BRANDS, fontsize=6.5)
        ax.set_xlabel("Brand at t")
        ax.set_ylabel("Brand at t-1")
        ax.set_title(MODEL_LABEL[m])
    fig.tight_layout()
    save_fig(fig, "figS1_e0_4r_transition_matrices",
             "Pooled first-order brand transition matrices (row-normalised; counts in parentheses) in the E0 "
             "4-round parity control. Diagonal mass above 1/3 indicates spontaneous habit formation; "
             "below 1/3 indicates variety seeking.")
    return W, t


# ════════════════════════════════════════════════════════════════════════
# 6. RQ1 — E1 seeding & decay
# ════════════════════════════════════════════════════════════════════════
def analyse_e1(R, W0):
    W = wide_brands(R, "E1")
    rows, contr, summ = [], [], []
    for m in MODELS:
        w = W[W.model == m]
        n = len(w)
        for r in range(1, 8):
            rows.append(prop_row(int((w[f"b{r}"] == "Nordvik").sum()), n, model=MODEL_LABEL[m], experiment="E1",
                                 round=r, metric="P(Nordvik)"))
        w0 = W0[W0.model == m].dropna(subset=["b1", "b2", "b3", "b4"])
        for r in range(1, 5):
            rows.append(prop_row(int((w0[f"b{r}"] == "Nordvik").sum()), len(w0), model=MODEL_LABEL[m],
                                 experiment="E0-4R", round=r, metric="P(Nordvik)"))
        # survival with initial brand
        for r in range(1, 8):
            stay = np.ones(n, bool)
            for q in range(2, r + 1):
                stay &= (w[f"b{q}"] == w.b1).values
            rows.append(prop_row(int(stay.sum()), n, model=MODEL_LABEL[m], experiment="E1", round=r,
                                 metric="P(still with R1 brand)"))
        n0 = len(w0)
        for r in range(1, 5):
            stay = np.ones(n0, bool)
            for q in range(2, r + 1):
                stay &= (w0[f"b{q}"] == w0.b1).values
            rows.append(prop_row(int(stay.sum()), n0, model=MODEL_LABEL[m], experiment="E0-4R", round=r,
                                 metric="P(still with R1 brand)"))
        seeded = w[w.b1 == "Nordvik"]
        par = seeded[[f"b{r}" for r in range(2, 8)]]
        n_nord = (par == "Nordvik").sum(1)
        is_sw = par.ne(seeded.b1, axis=0).values
        first_sw = [int(np.argmax(row)) + 2 if row.any() else np.nan for row in is_sw]
        summ.append(dict(model=MODEL_LABEL[m], n_finished=n, seed_takeup=(w.b1 == "Nordvik").mean(),
                         n_seeded=len(seeded), mean_nordvik_parity_rounds=n_nord.mean(),
                         never_switched=float(np.mean(np.isnan(first_sw))) if len(seeded) else np.nan,
                         median_first_switch_round=float(np.nanmedian(first_sw)) if np.any(~np.isnan(first_sw)) else np.nan))
        # Contrasts
        xb, nb = KEY[f"e0_nordvik_{m}"]
        x2 = int((w.b2 == "Nordvik").sum())
        x7 = int((w.b7 == "Nordvik").sum())
        xall = int((par == "Nordvik").all(1).sum()) if len(par) else 0
        c1 = contrast_row(f"{MODEL_LABEL[m]}: E1 R2 P(Nordvik) vs E0 no-history P(Nordvik)", x2, n, int(xb), int(nb),
                          "RQ1-E1 seeding effect")
        c2 = contrast_row(f"{MODEL_LABEL[m]}: E1 R7 P(Nordvik) vs E0 no-history P(Nordvik)", x7, n, int(xb), int(nb),
                          "RQ1-E1 seeding effect")
        # exogenous (discount-induced) vs endogenous (spontaneous) first choice: stay R2-R4 with R1 brand
        xs1 = int(((w.b2 == w.b1) & (w.b3 == w.b1) & (w.b4 == w.b1)).sum())
        xs0 = int(((w0.b2 == w0.b1) & (w0.b3 == w0.b1) & (w0.b4 == w0.b1)).sum())
        c3 = contrast_row(f"{MODEL_LABEL[m]}: stay with R1 brand R2-R4, E1 (discount-induced) vs E0-4R (spontaneous)",
                          xs1, n, xs0, n0, "RQ1-E1 seeding effect")
        contr += [c1, c2, c3]
        KEY[f"e1_{m}"] = dict(r2=(x2, n), r7=(x7, n), all=(xall, len(par)), stay1=(xs1, n), stay0=(xs0, n0))
    t = pd.DataFrame(rows)
    save_table(t, "tab07_e1_decay_curves",
               "E1 (one 20%-off seeding round, then six parity rounds) and E0-4R reference: P(Nordvik) by round and "
               "P(still with first-round brand), Wilson / Jeffreys 95% CIs.")
    save_table(pd.DataFrame(summ), "tab08_e1_trajectory_summary",
               "E1 trajectory-level summary: seeding take-up, mean number of Nordvik purchases in the six parity "
               "rounds, share never switching, median first-switch round (among seeded trajectories).")
    save_table(pd.DataFrame(contr), "tab09_e1_contrasts",
               "E1 contrasts: risk difference (Newcombe 95% CI), odds ratio (Haldane-Anscombe), Cohen's h, Fisher exact p.")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.7))
    for ax, metric, title in ((axes[0], "P(Nordvik)", "Share choosing Nordvik"),
                              (axes[1], "P(still with R1 brand)", "Still with first-round brand")):
        ax.axvspan(0.5, 1.5, color="#D55E00", alpha=0.08, lw=0)
        ax.text(1, 0.04, "−20%\nNordvik", ha="center", fontsize=5.8, color="#D55E00")
        for i, m in enumerate(MODELS):
            off = (i - 1) * 0.1
            s1 = t[(t.model == MODEL_LABEL[m]) & (t.experiment == "E1") & (t.metric == metric)]
            s0 = t[(t.model == MODEL_LABEL[m]) & (t.experiment == "E0-4R") & (t.metric == metric)]
            ax.plot(s1["round"] + off, s1.rate, marker=MODEL_MARKER[m], color=MODEL_COLOR[m], lw=1.2, ms=3.5,
                    label=f"{MODEL_LABEL[m]} · E1")
            errbar(ax, s1["round"].values + off, s1.rate.values, s1.wilson_low.values, s1.wilson_high.values,
                   fmt="none", ecolor=MODEL_COLOR[m])
            ax.plot(s0["round"] + off, s0.rate, marker=MODEL_MARKER[m], mfc="white", color=MODEL_COLOR[m], lw=1,
                    ls="--", ms=3.5, label=f"{MODEL_LABEL[m]} · E0-4R")
            errbar(ax, s0["round"].values + off, s0.rate.values, s0.wilson_low.values, s0.wilson_high.values,
                   fmt="none", ecolor=MODEL_COLOR[m], alpha=0.5)
        if metric == "P(Nordvik)":
            ax.axhline(1 / 3, ls=":", lw=0.8, color="grey")
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([f"R{r}" for r in range(1, 8)])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Round")
        ax.set_ylabel("Probability")
        ax.set_title(title)
    axes[1].legend(fontsize=5.5, loc="center right", ncol=1)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, "fig03_e1_loyalty_decay",
             "Loyalty formation after a single promotion (E1, n = 50 per model, solid) versus the no-promotion "
             "4-round control (E0-4R, ~120 per model, dashed). (a) Share choosing Nordvik per round; round 1 offers "
             "Nordvik at -20 %, rounds 2–7 are strict price/quality parity. (b) Share of trajectories still buying the "
             "round-1 brand in every round so far. Error bars: Wilson 95% CIs; dotted line: chance (1/3).")
    return W, t


# ════════════════════════════════════════════════════════════════════════
# 7. RQ2 — E2 premium-tolerance grid
# ════════════════════════════════════════════════════════════════════════
def e2_cells(R):
    rows = []
    for code, g in R[R.family == "E2"].groupby("experiment_code"):
        k, p = int(g.k.iloc[0]), int(g.p.iloc[0])
        m = g.model.iloc[0]
        traj = []
        for tr, h in g.groupby("traj"):
            h = h.sort_values("round_number")
            seed = h[h.phase == "seeding"]
            fin = h[h.phase == "test"].iloc[0]
            traj.append(dict(fully_seeded=bool((seed.brand == "Nordvik").all() and len(seed) == k),
                             r1_nordvik=bool(h.iloc[0].brand == "Nordvik"),
                             retained=bool(fin.brand == "Nordvik"), final_brand=fin.brand,
                             final_pos=fin.chosen_pos, final_order=fin.display_order, final_reason=fin.reason_text))
        T = pd.DataFrame(traj)
        n = len(T)
        fs = T[T.fully_seeded]
        lo, hi = wilson(int(fs.retained.sum()), len(fs))
        jlo, jhi = jeffreys(int(fs.retained.sum()), len(fs))
        rows.append(dict(experiment_code=code, model=MODEL_LABEL[m], model_id=m, k=k, premium_pct=p,
                         premium_price=round(15 * (1 + p / 100), 2), n_finished=n,
                         seeding_takeover=T.r1_nordvik.mean(), n_fully_seeded=len(fs),
                         fully_seeded_rate=len(fs) / n, retained_uncond=T.retained.mean(),
                         x_retained_cond=int(fs.retained.sum()), retention_cond=fs.retained.mean(),
                         wilson_low=lo, wilson_high=hi, jeffreys_low=jlo, jeffreys_high=jhi,
                         switched_to_zephyr=int((T.final_brand == "Zephyr").sum()),
                         switched_to_auralis=int((T.final_brand == "Auralis").sum())))
    return pd.DataFrame(rows).sort_values(["model_id", "k", "premium_pct"]).reset_index(drop=True)


def analyse_e2(R, W1):
    C = e2_cells(R)
    save_table(C.drop(columns=["model_id"]), "tab10_e2_grid",
               "E2 premium-tolerance grid: seeding take-up, full-seeding rate, unconditional and conditional "
               "(fully seeded) final-round retention of Nordvik with Wilson / Jeffreys 95% CIs, switch destinations.")
    g = C[C.model_id == "gpt-5.6-luna"]
    # Trajectory-level frame for regression / trend tests (gpt grid, fully seeded only)
    D = []
    for code, gg in R[(R.family == "E2") & (R.model == "gpt-5.6-luna")].groupby("experiment_code"):
        k = int(gg.k.iloc[0])
        for tr, h in gg.groupby("traj"):
            seed = h[h.phase == "seeding"]
            if (seed.brand == "Nordvik").all() and len(seed) == k:
                D.append(dict(k=k, p=int(h.p.iloc[0]), cell=code,
                              retained=int(h[h.phase == "test"].brand.iloc[0] == "Nordvik")))
    D = pd.DataFrame(D)
    inf = []
    # Logistic regression retained ~ k + p
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            res = sm.Logit(D.retained, sm.add_constant(D[["k", "p"]].astype(float))).fit(disp=0, maxiter=200)
        note = "; ".join(sorted({str(w_.category.__name__) for w_ in wlist})) or "converged"
        ci = res.conf_int()
        for v in ("k", "p"):
            inf.append(dict(analysis="Logit retained ~ k + p (GPT grid, fully seeded)", term=v, estimate=res.params[v],
                            odds_ratio=math.exp(res.params[v]), ci_low=math.exp(ci.loc[v, 0]),
                            ci_high=math.exp(ci.loc[v, 1]), p_value=res.pvalues[v], note=note))
            add_test("RQ2-E2 premium tolerance", f"Logit slope for {v} (GPT grid)", "Wald z", res.tvalues[v],
                     res.pvalues[v], math.exp(res.params[v]), "OR per unit", len(D))
    except Exception as exc:  # perfect separation etc.
        inf.append(dict(analysis="Logit retained ~ k + p", term="--", note=f"not estimable: {exc}"))
    # Cochran-Armitage trend tests
    byp = D.groupby("p").retained.agg(["sum", "count"])
    z, pv = cochran_armitage(byp["sum"], byp["count"], byp.index)
    inf.append(dict(analysis="Cochran-Armitage trend in premium p (pooled k)", term="p", estimate=z, p_value=pv))
    add_test("RQ2-E2 premium tolerance", "Trend in retention across premium p (pooled over k)", "Cochran-Armitage z", z, pv,
             n=len(D))
    byk = D.groupby("k").retained.agg(["sum", "count"])
    z, pv = cochran_armitage(byk["sum"], byk["count"], byk.index)
    inf.append(dict(analysis="Cochran-Armitage trend in seeding depth k (pooled p)", term="k", estimate=z, p_value=pv))
    add_test("RQ2-E2 premium tolerance", "Trend in retention across seeding depth k (pooled over p)",
             "Cochran-Armitage z", z, pv, n=len(D))
    chi, pv = perm_homogeneity(D.cell, D.retained)
    inf.append(dict(analysis="Permutation homogeneity across 20 cells", term="cell", estimate=chi, p_value=pv))
    add_test("RQ2-E2 premium tolerance", "Homogeneity of retention across the 20 (k,p) cells", "MC permutation chi-square",
             chi, pv, n=len(D))
    save_table(pd.DataFrame(inf), "tab11_e2_inference",
               "E2 inference on conditional retention (GPT-5.6-Luna grid, fully seeded trajectories): logistic "
               "regression, Cochran-Armitage trend tests, and a Monte-Carlo permutation test of homogeneity.")
    # Premium cliff: parity (E1 round k+1) vs +p% (E2)
    w1 = W1[W1.model == "gpt-5.6-luna"]
    cliff = []
    for k in (1, 2, 3, 5):
        seeded = w1[w1.b1 == "Nordvik"]
        x0 = int((seeded[f"b{k + 1}"] == "Nordvik").sum())
        sub = g[g.k == k]
        x1, n1 = int(sub.x_retained_cond.sum()), int(sub.n_fully_seeded.sum())
        lab = f"k={k}: E2 pooled +1..5% vs parity reference (E1 R{k + 1})"
        cliff.append(contrast_row(lab, x1, n1, x0, len(seeded), "RQ2-E2 premium cliff"))
        s1 = sub[sub.premium_pct == 1].iloc[0]
        cliff.append(contrast_row(f"k={k}: E2 +1% vs parity reference (E1 R{k + 1})", int(s1.x_retained_cond),
                                  int(s1.n_fully_seeded), x0, len(seeded), "RQ2-E2 premium cliff"))
    save_table(pd.DataFrame(cliff), "tab12_e2_premium_cliff",
               "Premium cliff: final-round retention under a +1..+5% premium (E2) versus the parity reference in E1 "
               "at the same round (k prior Nordvik purchases; only the first discounted). Note: the E1 reference "
               "history differs from E2 for k>1 (1 vs k discounted purchases).")
    KEY["e2_gpt_pooled"] = (int(g.x_retained_cond.sum()), int(g.n_fully_seeded.sum()))
    KEY["e2_cells"] = C

    # Figure 4
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ks, ps = [1, 2, 3, 5], [1, 2, 3, 4, 5]
    M = g.pivot(index="k", columns="premium_pct", values="retention_cond").reindex(index=ks, columns=ps)
    X = g.pivot(index="k", columns="premium_pct", values="x_retained_cond").reindex(index=ks, columns=ps)
    N = g.pivot(index="k", columns="premium_pct", values="n_fully_seeded").reindex(index=ks, columns=ps)
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ks)):
        for j in range(len(ps)):
            ax.text(j, i, f"{M.values[i, j] * 100:.0f}%\n{int(X.values[i, j])}/{int(N.values[i, j])}", ha="center",
                    va="center", fontsize=6.2, color="white" if M.values[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"+{p_}%" for p_ in ps])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"k={k_}" for k_ in ks])
    ax.set_xlabel("Final-round Nordvik premium")
    ax.set_ylabel("Seeding rounds at −20%")
    ax.set_title("Retention | fully seeded (GPT-5.6-Luna)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Retention", fontsize=7)
    ax = axes[1]
    kcol = {1: "#999999", 2: "#56B4E9", 3: "#0072B2", 5: "#000000"}
    for i, k in enumerate(ks):
        sub = g[g.k == k].sort_values("premium_pct")
        off = (i - 1.5) * 0.07
        seeded = w1[w1.b1 == "Nordvik"]
        x0 = int((seeded[f"b{k + 1}"] == "Nordvik").sum())
        n0 = len(seeded)
        l0, h0 = wilson(x0, n0)
        xs = np.r_[0, sub.premium_pct.values] + off
        ys = np.r_[x0 / n0, sub.retention_cond.values]
        ax.plot(xs, ys, color=kcol[k], lw=1, marker="o", ms=3.5, label=f"k={k}")
        ax.plot([xs[0]], [ys[0]], marker="o", ms=4.5, mfc="white", color=kcol[k])
        errbar(ax, xs, ys, np.r_[l0, sub.wilson_low.values], np.r_[h0, sub.wilson_high.values], fmt="none",
               ecolor=kcol[k])
    for m in ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"):
        s = C[(C.model_id == m)]
        if len(s):
            s = s.iloc[0]
            errbar(ax, [1.25], [s.retention_cond], [s.wilson_low], [s.wilson_high],
                   fmt=MODEL_MARKER[m], color=MODEL_COLOR[m], ms=4, ecolor=MODEL_COLOR[m],
                   label=f"{MODEL_LABEL[m]} (k=3)")
    ax.set_xticks(range(6))
    ax.set_xticklabels(["parity\n(E1 ref.)", "+1%", "+2%", "+3%", "+4%", "+5%"])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("P(retain Nordvik | fully seeded)")
    ax.set_xlabel("Nordvik premium in test round")
    ax.set_title("Switching-threshold curve")
    ax.legend(fontsize=5.8, ncol=2, loc="center right")
    panel(axes[0], "a", x=-0.2)
    panel(axes[1], "b")
    fig.tight_layout()
    save_fig(fig, "fig04_e2_premium_tolerance",
             "Price-premium tolerance (RQ2). (a) Conditional retention of Nordvik in the test round after k discounted "
             "seeding purchases, by premium (GPT-5.6-Luna, n ≈ 50 per cell; cell text: rate and x/n). (b) Switching-"
             "threshold curves with Wilson 95% CIs; hollow points at 'parity' are the E1 reference (Nordvik share in "
             "round k+1 of E1 among seeded trajectories); Gemini/DeepSeek replications of k=3, +1% are overlaid.")
    return C


def analyse_switchers(R):
    """Among agents leaving Nordvik under a premium, which of the two tied cheaper options do they pick?"""
    sw = R[((R.family == "E2") & (R.phase == "test")) | ((R.family == "E3") & (R.p == 5) & (R.phase == "test"))]
    sw = sw[sw.brand != "Nordvik"].copy()
    rows = []
    for m in MODELS:
        s = sw[sw.model == m]
        if not len(s):
            continue
        n = len(s)
        # earlier displayed of the two non-Nordvik options
        earlier = [int(pos == min(i + 1 for i, x in enumerate(o) if not x.endswith("nordvik")))
                   for pos, o in zip(s.chosen_pos, s.display_order)]
        xe = int(np.sum(earlier))
        xz = int((s.brand == "Zephyr").sum())
        rows.append(prop_row(xe, n, model=MODEL_LABEL[m], metric="chose earlier-displayed of two cheaper options"))
        rows.append(prop_row(xz, n, model=MODEL_LABEL[m], metric="chose Zephyr (vs Auralis)"))
        add_test("Position/brand bias in ties", f"{MODEL_LABEL[m]}: switchers pick earlier-displayed option = 0.5",
                 "Exact binomial", np.nan, binom_p(xe, n, 0.5), xe / n - 0.5, "rate - 0.5", n)
        add_test("Position/brand bias in ties", f"{MODEL_LABEL[m]}: switchers pick Zephyr = 0.5", "Exact binomial",
                 np.nan, binom_p(xz, n, 0.5), xz / n - 0.5, "rate - 0.5", n)
    t = pd.DataFrame(rows)
    save_table(t, "tab13_switch_destinations",
               "Tie-breaking among agents that abandon Nordvik under a premium (E2 test rounds + E3 +5%): share choosing "
               "the earlier-displayed of the two equally cheap options, and share choosing Zephyr over Auralis.")
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    metrics = t.metric.unique()
    for j, met in enumerate(metrics):
        for i, m in enumerate([m for m in MODELS if MODEL_LABEL[m] in set(t.model)]):
            r = t[(t.metric == met) & (t.model == MODEL_LABEL[m])].iloc[0]
            x = j + (i - 1) * 0.22
            ax.bar(x, r.rate, 0.2, color=MODEL_COLOR[m], edgecolor="black", lw=0.5,
                   label=MODEL_LABEL[m] if j == 0 else None)
            errbar(ax, [x], [r.rate], [r.wilson_low], [r.wilson_high], fmt="none", ecolor="black")
            ax.text(x, 0.02, f"n={r.n}", rotation=90, fontsize=5.5, ha="center", va="bottom", color="white")
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(["Earlier-displayed\noption", "Zephyr\n(vs Auralis)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of switchers")
    ax.set_title("Tie-breaking after leaving Nordvik")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "figS2_switcher_tie_breaking",
             "Tie-breaking among agents abandoning Nordvik under a premium (E2 test rounds and E3 +5%). Bars: share "
             "choosing the earlier-displayed of the two equally priced alternatives, and share choosing Zephyr over "
             "Auralis; Wilson 95% CIs; dashed line: 0.5.")
    return t


# ════════════════════════════════════════════════════════════════════════
# 8. RQ3 — E3 spillover
# ════════════════════════════════════════════════════════════════════════
def analyse_e3(R):
    rows, contr = [], []
    per = {}
    for code, g in R[R.family == "E3"].groupby("experiment_code"):
        m = g.model.iloc[0]
        p = int(g.p.iloc[0])
        T = []
        for tr, h in g.groupby("traj"):
            seed = h[h.phase == "seeding"]
            fin = h[h.phase == "test"].iloc[0]
            T.append(dict(fully=bool((seed.brand == "Nordvik").all() and len(seed) == 3),
                          spilled=bool(fin.brand == "Nordvik"), n_seed=int((seed.brand == "Nordvik").sum())))
        T = pd.DataFrame(T)
        fs = T[T.fully]
        x, n = int(fs.spilled.sum()), len(fs)
        per[code] = (x, n, m, p)
        r = prop_row(x, n, experiment_code=code, model=MODEL_LABEL[m], dish_premium_pct=p, n_finished=len(T),
                     fully_seeded_rate=T.fully.mean(), spillover_uncond=T.spilled.mean())
        xb, nb = KEY[f"e0_nordvik_{m}"]
        pb = binom_p(x, n, 1 / 3)
        add_test("RQ3-E3 spillover", f"{code}: spillover | fully seeded = 1/3", "Exact binomial", np.nan, pb,
                 cohens_h(x / n, 1 / 3), "Cohen's h", n)
        r["p_vs_chance"] = pb
        contr.append(contrast_row(f"{code}: spillover vs model's E0 no-history P(Nordvik)", x, n, int(xb), int(nb),
                                  "RQ3-E3 spillover"))
        rows.append(r)
    if "RQ3_E3_k3_parity" in per and "RQ3_E3_k3_p5" in per:
        a, b = per["RQ3_E3_k3_parity"], per["RQ3_E3_k3_p5"]
        contr.append(contrast_row("GPT: spillover at parity vs at +5% dish premium", a[0], a[1], b[0], b[1],
                                  "RQ3-E3 spillover"))
    t = pd.DataFrame(rows)
    save_table(t, "tab14_e3_spillover",
               "E3 umbrella-brand spillover: share choosing Nordvik dish soap in round 4 after three discounted Nordvik "
               "laundry purchases (conditional on full seeding; Wilson / Jeffreys 95% CIs) with exact binomial test vs 1/3.")
    save_table(pd.DataFrame(contr), "tab15_e3_contrasts",
               "E3 contrasts: spillover versus each model's no-history Nordvik share, and parity versus +5% premium.")
    KEY["e3"] = per

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    order = [("RQ3_E3_k3_parity", "GPT\nparity"), ("RQ3_E3_k3_p5", "GPT\n+5%"),
             ("RQ3_E3_k3_parity_gemini", "Gemini\nparity"), ("RQ3_E3_k3_parity_deepseek", "DeepSeek\nparity")]
    order = [o for o in order if o[0] in per]
    for i, (code, lab) in enumerate(order):
        x, n, m, p = per[code]
        lo, hi = wilson(x, n)
        ax.bar(i, x / n, 0.6, color=MODEL_COLOR[m], edgecolor="black", lw=0.5, hatch="///" if p else None, alpha=0.9)
        errbar(ax, [i], [x / n], [lo], [hi], fmt="none", ecolor="black")
        xb, nb = KEY[f"e0_nordvik_{m}"]
        ax.plot([i - 0.32, i + 0.32], [xb / nb] * 2, color="black", lw=1.2, ls="--")
        ax.text(i, min(x / n + 0.05, 1.04), f"{x}/{n}", ha="center", fontsize=6)
    ax.axhline(1 / 3, ls=":", lw=0.8, color="grey")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o[1] for o in order], fontsize=6.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("P(Nordvik dish soap | seeded)")
    ax.set_title("Umbrella-brand spillover (laundry → dish)")
    fig.tight_layout()
    save_fig(fig, "fig05_e3_spillover",
             "Brand spillover (RQ3). Share of fully seeded agents (three discounted Nordvik laundry purchases) choosing "
             "the never-purchased Nordvik dish soap in round 4, at parity or at a +5% premium (hatched). Error bars: "
             "Wilson 95% CIs; dashed segments: the model's no-history Nordvik share (E0); dotted line: 1/3.")
    return t


# ════════════════════════════════════════════════════════════════════════
# 9. Cross-model synthesis
# ════════════════════════════════════════════════════════════════════════
def analyse_cross_model(R, W0, W1):
    rows = []
    C = KEY["e2_cells"]
    for m in MODELS:
        xb, nb = KEY[f"e0_nordvik_{m}"]
        rows.append(prop_row(xb, nb, metric="E0 no-history: P(Nordvik)", model=MODEL_LABEL[m], model_id=m))
        rep, lo, hi = KEY[f"e0_4r_repeat_{m}"]
        n0 = len(W0[W0.model == m].dropna(subset=["b1", "b2", "b3", "b4"]))
        rows.append(dict(metric="E0-4R: P(repeat previous brand)", model=MODEL_LABEL[m], model_id=m, x=np.nan,
                         n=3 * n0, rate=rep, wilson_low=lo, wilson_high=hi, jeffreys_low=np.nan, jeffreys_high=np.nan))
        e1 = KEY[f"e1_{m}"]
        rows.append(prop_row(*e1["r2"], metric="E1: P(Nordvik) in R2 (first parity round)", model=MODEL_LABEL[m], model_id=m))
        rows.append(prop_row(*e1["all"], metric="E1: Nordvik in all six parity rounds", model=MODEL_LABEL[m], model_id=m))
        s = C[(C.model_id == m) & (C.k == 3) & (C.premium_pct == 1)]
        if len(s):
            s = s.iloc[0]
            rows.append(prop_row(int(s.x_retained_cond), int(s.n_fully_seeded),
                                 metric="E2 k=3, +1%: retention | fully seeded", model=MODEL_LABEL[m], model_id=m))
        for code, (x, n, mm, p) in KEY["e3"].items():
            if mm == m and p == 0:
                rows.append(prop_row(x, n, metric="E3 parity: spillover | fully seeded", model=MODEL_LABEL[m], model_id=m))
    t = pd.DataFrame(rows)
    het = []
    for met, g in t.groupby("metric", sort=False):
        if g.x.isna().any() or len(g) < 2:
            het.append(dict(metric=met, test="--", statistic=np.nan, p_value=np.nan,
                            note="pooled transitions are clustered; see tab05 bootstrap CIs"))
            continue
        groups = np.concatenate([[mm] * int(nn) for mm, nn in zip(g.model, g.n)])
        y = np.concatenate([np.r_[np.ones(int(xx)), np.zeros(int(nn - xx))] for xx, nn in zip(g.x, g.n)])
        chi, pv = perm_homogeneity(groups, y)
        het.append(dict(metric=met, test="MC permutation chi-square (3 models)", statistic=chi, p_value=pv, note=""))
        add_test("Cross-model heterogeneity", met, "MC permutation chi-square", chi, pv, n=int(g.n.sum()))
    save_table(t.drop(columns=["model_id"]), "tab16_cross_model_summary",
               "Cross-model comparison of headline metrics (Wilson / Jeffreys 95% CIs; E0-4R repeat uses a cluster "
               "bootstrap CI over trajectories).")
    save_table(pd.DataFrame(het), "tab17_cross_model_heterogeneity",
               "Tests of between-model heterogeneity for each headline metric (Monte-Carlo permutation, 10,000 draws).")

    metrics = list(dict.fromkeys(t.metric))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.0))
    for j, met in enumerate(metrics):
        for i, m in enumerate(MODELS):
            r = t[(t.metric == met) & (t.model_id == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = len(metrics) - 1 - j + (1 - i) * 0.22
            xerr_low = max(0.0, r.rate - r.wilson_low)
            xerr_high = max(0.0, r.wilson_high - r.rate)
            ax.errorbar(r.rate, y, xerr=[[xerr_low], [xerr_high]], marker=MODEL_MARKER[m],
                        color=MODEL_COLOR[m], ms=4.5, capsize=2, lw=0.9, label=MODEL_LABEL[m] if j == 0 else None)
    ax.axvline(1 / 3, ls=":", color="grey", lw=0.8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics[::-1], fontsize=6.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Probability (95% CI)")
    ax.set_title("Generalisation across LLMs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=6.5)
    for j, met in enumerate(metrics):
        h = [x for x in het if x["metric"] == met][0]
        if not np.isnan(h["p_value"]):
            ax.text(1.03, len(metrics) - 1 - j, f"p={fmt_p(h['p_value'])}", fontsize=6, va="center",
                    transform=ax.get_yaxis_transform())
    fig.tight_layout()
    save_fig(fig, "fig06_cross_model_forest",
             "Cross-model generalisation of the headline metrics (point estimates with Wilson 95% CIs; E0-4R repeat with "
             "cluster-bootstrap CI). Right margin: Monte-Carlo permutation p-values for between-model heterogeneity. "
             "Dotted line: chance (1/3).")
    return t


# ════════════════════════════════════════════════════════════════════════
# 10. Unified choice model — penalised conditional logit
# ════════════════════════════════════════════════════════════════════════
FEATS = ["price_premium_pct", "last_brand", "last_brand_xcat", "position_2", "position_3", "brand_Zephyr", "brand_Auralis"]
FEAT_LABEL = {"price_premium_pct": "Price above cheapest (per 1%)", "last_brand": "Bought last round",
              "last_brand_xcat": "× new category (spillover)", "position_2": "Display position 2",
              "position_3": "Display position 3", "brand_Zephyr": "Brand: Zephyr (vs Nordvik)",
              "brand_Auralis": "Brand: Auralis (vs Nordvik)"}


def build_choice_sets(R, meta, design):
    n = len(R)
    X = np.zeros((n, 3, len(FEATS)))
    y = np.zeros(n, int)
    for i, r in enumerate(R.itertuples()):
        mt = meta[r.experiment_code]
        prices = design[(r.experiment_code, r.round_number)]["prices"]
        order = r.display_order
        pmin = min(prices[x] for x in order)
        for j, pid in enumerate(order):
            b, c = mt["brand_of"][pid], mt["cat_of"][pid]
            last = isinstance(r.prev_brand, str) and r.prev_brand == b
            X[i, j] = [(prices[pid] / pmin - 1) * 100, last, last and (c != r.prev_category), j == 1, j == 2,
                       b == "Zephyr", b == "Auralis"]
        y[i] = order.index(r.product_id)
    return X, y


def clogit_obj(beta, X, y, lam):
    U = X @ beta
    lse = logsumexp(U, axis=1)
    idx = np.arange(len(y))
    P = np.exp(U - lse[:, None])
    nll = -(U[idx, y] - lse).sum() + 0.5 * lam * beta @ beta
    grad = -(X[idx, y] - (P[..., None] * X).sum(1)).sum(0) + lam * beta
    return nll, grad


def fit_clogit(X, y, lam, b0=None):
    b0 = np.zeros(X.shape[2]) if b0 is None else b0
    res = optimize.minimize(clogit_obj, b0, args=(X, y, lam), jac=True, method="L-BFGS-B")
    return res.x


def analyse_choice_model(R, meta, design):
    rows, sens, fit_rows = [], [], []
    boots = {}
    for m in MODELS:
        s = R[R.model == m].reset_index(drop=True)
        X, y = build_choice_sets(s, meta, design)
        keep = X.reshape(len(X), -1).std(0).reshape(3, -1).max(0) > 0     # drop features without variation
        feats = [f for f, k_ in zip(FEATS, keep) if k_]
        Xk = X[:, :, keep]
        beta = fit_clogit(Xk, y, CLOGIT_LAMBDA)
        U = Xk @ beta
        ll = (U[np.arange(len(y)), y] - logsumexp(U, axis=1)).sum()
        ll0 = len(y) * math.log(1 / 3)
        fit_rows.append(dict(model=MODEL_LABEL[m], n_choices=len(y), n_trajectories=s.traj.nunique(),
                             log_lik=ll, log_lik_null=ll0, mcfadden_r2=1 - ll / ll0,
                             hit_rate=float(np.mean(U.argmax(1) == y)), ridge_lambda=CLOGIT_LAMBDA))
        # cluster bootstrap
        clusters = s.traj.values
        uc, inv = np.unique(clusters, return_inverse=True)
        members = [np.where(inv == c)[0] for c in range(len(uc))]
        B = np.zeros((N_BOOT_CLOGIT, len(feats)))
        for b in range(N_BOOT_CLOGIT):
            pick = RNG.integers(0, len(uc), len(uc))
            idx = np.concatenate([members[c] for c in pick])
            B[b] = fit_clogit(Xk[idx], y[idx], CLOGIT_LAMBDA, beta)
        boots[m] = (feats, beta, B)
        for f_i, f in enumerate(feats):
            lo, hi = np.percentile(B[:, f_i], [2.5, 97.5])
            rows.append(dict(model=MODEL_LABEL[m], term=FEAT_LABEL[f], estimate=beta[f_i], boot_se=B[:, f_i].std(ddof=1),
                             ci_low=lo, ci_high=hi, odds_multiplier=math.exp(beta[f_i])))
        if "last_brand" in feats and "price_premium_pct" in feats:
            il, ip = feats.index("last_brand"), feats.index("price_premium_pct")
            wtp = beta[il] / -beta[ip]
            wb = B[:, il] / -B[:, ip]
            lo, hi = np.percentile(wb, [2.5, 97.5])
            rows.append(dict(model=MODEL_LABEL[m], term="History-equivalent premium (% price) = β_last / −β_price",
                             estimate=wtp, boot_se=wb.std(ddof=1), ci_low=lo, ci_high=hi, odds_multiplier=np.nan))
            KEY[f"wtp_{m}"] = (wtp, lo, hi)
        for lam in (0.1, 1.0, 10.0):
            bl = fit_clogit(Xk, y, lam)
            d = dict(model=MODEL_LABEL[m], ridge_lambda=lam)
            d.update({FEAT_LABEL[f]: bl[i] for i, f in enumerate(feats)})
            if "last_brand" in feats and "price_premium_pct" in feats:
                d["history_equiv_premium_pct"] = bl[feats.index("last_brand")] / -bl[feats.index("price_premium_pct")]
            sens.append(d)
    t = pd.DataFrame(rows)
    save_table(t, "tab18_choice_model_coefficients",
               f"Penalised conditional logit (ridge λ = {CLOGIT_LAMBDA}) of product choice across all non-pilot "
               f"decisions, per model; 95% percentile CIs from {N_BOOT_CLOGIT} trajectory-cluster bootstrap replicates. "
               "Penalisation is required because several regimes are perfectly separated (e.g., 100% repeat at parity).")
    save_table(pd.DataFrame(fit_rows), "tab19_choice_model_fit", "Choice-model fit statistics (McFadden pseudo-R², hit rate).")
    save_table(pd.DataFrame(sens), "tab20_choice_model_sensitivity",
               "Sensitivity of choice-model coefficients and the history-equivalent premium to the ridge penalty λ.")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    terms = [FEAT_LABEL[f] for f in FEATS]
    for i, m in enumerate(MODELS):
        sub = t[(t.model == MODEL_LABEL[m]) & t.term.isin(terms)].set_index("term").reindex(terms)
        yv = np.arange(len(terms))[::-1] + (1 - i) * 0.22
        xerr_lo = np.clip(sub.estimate - sub.ci_low, 0, None)
        xerr_hi = np.clip(sub.ci_high - sub.estimate, 0, None)
        ax.errorbar(sub.estimate, yv, xerr=[xerr_lo, xerr_hi], fmt=MODEL_MARKER[m],
                    color=MODEL_COLOR[m], ms=4, capsize=2, lw=0.9, label=MODEL_LABEL[m])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms[::-1], fontsize=6.8)
    ax.set_xlabel("Coefficient (log-odds utility)")
    ax.set_title("Drivers of choice (penalised conditional logit)")
    ax.legend(fontsize=6, loc="lower right")
    ax = axes[1]
    for i, m in enumerate(MODELS):
        if f"wtp_{m}" in KEY:
            w, lo, hi = KEY[f"wtp_{m}"]
            yerr_lo = max(0.0, w - lo)
            yerr_hi = max(0.0, hi - w)
            ax.errorbar(i, w, yerr=[[yerr_lo], [yerr_hi]], fmt=MODEL_MARKER[m], color=MODEL_COLOR[m], ms=5, capsize=3)
            ax.text(i + 0.12, w, f"{w:.2f}%", fontsize=6.5, va="center")
    ax.axhline(1, ls="--", color="grey", lw=0.8)
    ax.text(2.4, 1.02, "smallest tested\npremium (+1%)", fontsize=5.8, color="grey", ha="right", va="bottom")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m].replace("-", "-\n", 1) for m in MODELS], fontsize=6.3)
    ax.set_ylabel("Price premium (%) worth one prior purchase")
    ax.set_title("History-equivalent premium")
    ax.set_xlim(-0.5, 2.6)
    panel(axes[0], "a", x=-0.45)
    panel(axes[1], "b", x=-0.3)
    fig.tight_layout()
    save_fig(fig, "fig07_choice_model",
             f"Unified discrete-choice model. (a) Ridge-penalised (λ = {CLOGIT_LAMBDA}) conditional-logit coefficients "
             "with 95% trajectory-cluster bootstrap CIs, estimated on all non-pilot decisions per model. (b) "
             "History-equivalent price premium, β_last / (−β_price): the percentage price increase that offsets having "
             "bought the brand in the previous round. Dashed line: the smallest premium tested in E2 (+1%).")
    return t


# ════════════════════════════════════════════════════════════════════════
# 11. Reason-text analysis
# ════════════════════════════════════════════════════════════════════════
THEMES = {
    "Price advantage": r"\b(cheap\w*|lowest|lower(?: price| cost)?|less expensive|costs? less|costing less|"
                       r"best[- ]value|great value|better value|discount\w*|sav(?:e|es|ing|ings)|"
                       r"most (?:cost[- ]effective|affordable|economical)|cost[- ]effective|bargain|deal)\b",
    "Premium acknowledged": r"(slightly (?:higher|more)|more expensive|higher[- ]pric\w*|higher cost|premium|"
                            r"costs? (?:slightly |a bit |a little )?more|small price (?:increase|difference)|"
                            r"price increase|marginal(?:ly)? (?:higher|more))",
    "History / consistency": r"\b(again|previous\w*|prior|consisten\w*|history|continu\w*|familiar\w*|loyal\w*|"
                             r"repeat\w*|established|routine|habit\w*|already (?:purchased|bought|used|chosen)|past|"
                             r"same brand|stick\w*|stay\w*|maintain\w*|once more)\b",
    "Trust / reliability": r"\b(trust\w*|reliab\w*|proven|satisf\w*|dependab\w*|track record|known to work|"
                           r"worked well|served)\b",
    "Parity / indifference": r"(identical|equal(?:ly)?|equivalent|same (?:price|quality|\d)|no (?:functional |real |"
                             r"meaningful )?difference|all (?:three|options|products|alternatives)|"
                             r"(?:as|than) the (?:alternatives|others|other options))",
    "Quality": r"\bquality\b",
    "Variety / exploration": r"\b(variety|try|trying|diversif\w*|explor\w*|something different|different brand|"
                             r"new brand|switch\w* things up)\b",
    "Budget fit": r"\bbudget\b",
}
THEME_RE = {k: re.compile(v, re.I) for k, v in THEMES.items()}
CONTEXT_ORDER = ["E0 · no history", "E0-4R · R2–4 repeat", "E0-4R · R2–4 switch", "E1 · R1 discount",
                 "E1 · R2–7 parity", "E2 · seeding (discount)", "E2 · premium: retained", "E2 · premium: switched",
                 "E3 · seeding (laundry)", "E3 · dish: Nordvik", "E3 · dish: other brand"]
STOP = set("""a an the and or of to in on for with is it its this that as at by be are was were has have had from
your you we our i me my they their them which while also so than then there these those but not no into over
than more most very only same all any each other one two three both can will would should could just still
offers offer offering provides provide providing making makes make choice chose choose chosen selected select
option options product products round rounds this""".split())
DOMAIN = {"nordvik", "zephyr", "auralis", "laundry", "detergent", "dish", "soap", "user", "request"}


def reason_context(r):
    f = r.family
    if f == "E0_1R" or (f == "E0_4R" and r.round_number == 1):
        return "E0 · no history"
    if f == "E0_4R":
        return "E0-4R · R2–4 repeat" if r.same_as_prev else "E0-4R · R2–4 switch"
    if f == "E1":
        return "E1 · R1 discount" if r.round_number == 1 else "E1 · R2–7 parity"
    if f == "E2":
        if r.phase == "seeding":
            return "E2 · seeding (discount)"
        return "E2 · premium: retained" if r.brand == "Nordvik" else "E2 · premium: switched"
    if f == "E3":
        if r.phase == "seeding":
            return "E3 · seeding (laundry)"
        return "E3 · dish: Nordvik" if r.brand == "Nordvik" else "E3 · dish: other brand"
    return "other"


def tokens(s):
    return [w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP and w not in DOMAIN and len(w) > 2]


def log_odds_dirichlet(texts_a, texts_b, prior_texts, alpha0=1000.0, min_count=10):
    """Monroe, Colaresi & Quinn (2008) weighted log-odds with informative Dirichlet prior."""
    ca, cb, cp = Counter(), Counter(), Counter()
    for t_ in texts_a:
        ca.update(tokens(t_))
    for t_ in texts_b:
        cb.update(tokens(t_))
    for t_ in prior_texts:
        cp.update(tokens(t_))
    na, nb, npr = sum(ca.values()), sum(cb.values()), sum(cp.values())
    out = []
    for w in set(ca) | set(cb):
        if ca[w] + cb[w] < min_count:
            continue
        aw = max(alpha0 * cp[w] / npr, 0.01)
        la = math.log((ca[w] + aw) / (na + alpha0 - ca[w] - aw))
        lb = math.log((cb[w] + aw) / (nb + alpha0 - cb[w] - aw))
        z = (la - lb) / math.sqrt(1 / (ca[w] + aw) + 1 / (cb[w] + aw))
        out.append(dict(word=w, count_a=ca[w], count_b=cb[w], z=z))
    return pd.DataFrame(out).sort_values("z")


def analyse_reasons(R):
    D = R[R.reason_text.notna()].copy()
    D["context"] = [reason_context(r) for r in D.itertuples()]
    for th, rx in THEME_RE.items():
        D[th] = D.reason_text.str.contains(rx).astype(int)
    D["n_words"] = D.reason_text.str.split().str.len()
    themes = list(THEMES)
    prev = (D.groupby(["model", "context"])[themes + ["n_words"]].mean()
            .join(D.groupby(["model", "context"]).size().rename("n")).reset_index())
    prev["model"] = prev.model.map(MODEL_LABEL)
    prev["context"] = pd.Categorical(prev.context, CONTEXT_ORDER, ordered=True)
    prev = prev.sort_values(["model", "context"])
    save_table(prev, "tab21_reason_theme_prevalence",
               "Prevalence of lexicon-coded reason themes (share of decisions mentioning the theme) and mean reason "
               "length by decision context and model.")
    # Inferential: history theme, first parity round after seeding vs no-history baseline (independent units)
    inf = []
    for m in MODELS:
        a = D[(D.model == m) & (D.family == "E1") & (D.round_number == 2)]
        b = D[(D.model == m) & (D.context == "E0 · no history") & (D.family == "E0_1R")]
        for th in ("History / consistency", "Trust / reliability", "Parity / indifference"):
            inf.append(contrast_row(f"{MODEL_LABEL[m]}: '{th}' in E1 R2 vs E0 1-round", int(a[th].sum()), len(a),
                                    int(b[th].sum()), len(b), "Reason themes"))
        a = D[(D.model == m) & (D.context == "E2 · premium: switched")]
        b = D[(D.model == m) & (D.context == "E2 · seeding (discount)") & (D.round_number == 1)]
        if len(a) and len(b):
            inf.append(contrast_row(f"{MODEL_LABEL[m]}: 'Price advantage' in E2 switch vs E2 R1 discount",
                                    int(a["Price advantage"].sum()), len(a), int(b["Price advantage"].sum()), len(b),
                                    "Reason themes"))
    save_table(pd.DataFrame(inf), "tab22_reason_theme_contrasts",
               "Reason-theme contrasts between decision contexts (independent trajectories; Fisher exact tests).")

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 3.6), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = D[D.model == m]
        P = sub.groupby("context")[themes].mean().reindex(CONTEXT_ORDER)
        Nn = sub.groupby("context").size().reindex(CONTEXT_ORDER)
        im = ax.imshow(np.ma.masked_invalid(P.values), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if v > 0.55 else "black")
        ax.set_xticks(range(len(themes)))
        ax.set_xticklabels(themes, rotation=55, ha="right", fontsize=6)
        ax.set_yticks(range(len(CONTEXT_ORDER)))
        ax.set_yticklabels([f"{c} (n={int(n_) if not np.isnan(n_) else 0})" for c, n_ in zip(CONTEXT_ORDER, Nn.values)]
                           if ax is axes[0] else CONTEXT_ORDER, fontsize=6)
        ax.set_title(MODEL_LABEL[m])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Share of reasons mentioning theme", fontsize=7)
    save_fig(fig, "fig08_reason_themes",
             "What the agents say. Share (%) of free-text purchase justifications mentioning each theme, by decision "
             "context and model (transparent regex lexicon; see analyze_results.py THEMES). Blank rows: context not "
             "run for that model. Sample sizes refer to GPT-5.6-Luna; see tab21 for all models.")

    # Distinctive vocabulary: decisions with promotion history at parity vs no-history parity decisions
    a = D[D.context == "E1 · R2–7 parity"].reason_text
    b = D[D.context == "E0 · no history"].reason_text
    lo = log_odds_dirichlet(a, b, D.reason_text)
    lo.to_csv(TAB_DIR / "tab23_log_odds_history_vs_nohistory.csv", index=False)
    TABLES.append(("tab23_log_odds_history_vs_nohistory",
                   "Weighted log-odds (informative Dirichlet prior) of words in E1 parity-round reasons vs no-history reasons."))
    top = pd.concat([lo.head(15), lo.tail(15)])
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.8))
    colors = ["#999999" if z < 0 else "#D55E00" for z in top.z]
    ax.barh(range(len(top)), top.z, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.word, fontsize=6.5)
    ax.axvline(0, color="black", lw=0.6)
    for x_ in (-1.96, 1.96):
        ax.axvline(x_, ls=":", color="grey", lw=0.6)
    ax.set_xlabel("Weighted log-odds z-score")
    ax.set_title("Vocabulary: after promotion vs no history")
    ax.text(0.98, 0.02, "→ E1 parity rounds", transform=ax.transAxes, ha="right", fontsize=6.5, color="#D55E00")
    ax.text(0.02, 0.98, "← E0 no-history", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#666666")
    fig.tight_layout()
    save_fig(fig, "fig09_reason_log_odds",
             "Distinctive vocabulary of justifications for parity decisions after a promotion (E1 rounds 2–7, all models) "
             "versus no-history parity decisions (E0), weighted log-odds with informative Dirichlet prior (Monroe et al., "
             "2008). Dotted lines: |z| = 1.96. Brand and product words removed.")

    # Stratified sample for human double-coding (validation of the lexicon, Cohen's kappa)
    samp = (D.groupby(["model", "context"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 15), random_state=RNG_SEED)))
    cols = ["experiment_code", "model", "run_index", "round_number", "context", "brand", "price_paid", "reason_text"] + themes
    samp = samp[cols].copy()
    samp.insert(0, "item_id", range(1, len(samp) + 1))
    for c in ("coder1_primary_theme", "coder2_primary_theme", "coder_notes"):
        samp[c] = ""
    samp.to_csv(TAB_DIR / "reason_coding_sample.csv", index=False)
    TABLES.append(("reason_coding_sample", "Stratified sample (≤15 per model × context) for blind human double-coding."))
    D[["experiment_code", "model", "run_index", "round_number", "context", "brand", "reason_text", "n_words"] + themes] \
        .to_csv(DER_DIR / "reasons_coded_all.csv", index=False)
    KEY["reasons"] = prev
    return prev


# ════════════════════════════════════════════════════════════════════════
# 12. Derived per-experiment tables (RESULTS_GUIDE §§1-4)
# ════════════════════════════════════════════════════════════════════════
def export_derived(R):
    base = ["experiment_code", "model", "run_index"]
    e0 = R[R.family == "E0_1R"][base + ["round_status", "brand", "product_id", "price_paid", "reason_text",
                                        "round_failure_reason", "chosen_pos"]]
    e0.to_csv(DER_DIR / "e0_1r.csv", index=False)
    s = R[R.family == "E0_4R"]
    w = s.pivot_table(index=base, columns="round_number", values="brand", aggfunc="first")
    w.columns = [f"brand_r{c}" for c in w.columns]
    w = w.reset_index()
    w["n_committed"] = s.groupby(base).size().values
    for r in (2, 3, 4):
        w[f"repeat_r{r}"] = (w[f"brand_r{r}"] == w[f"brand_r{r - 1}"]).astype(int)
    w["stayed_all_4"] = ((w.brand_r2 == w.brand_r1) & (w.brand_r3 == w.brand_r1) & (w.brand_r4 == w.brand_r1)).astype(int)
    w["switched_ever"] = 1 - w.stayed_all_4
    w.to_csv(DER_DIR / "e0_4r.csv", index=False)
    s = R[R.family == "E1"].copy()
    s["seed_pick"] = s.groupby("traj").brand.transform("first")
    s["is_repeat_of_seed"] = (s.brand == s.seed_pick).astype(int)
    s[base + ["round_number", "phase", "round_status", "brand", "product_id", "price_paid", "reason_text", "seed_pick",
              "is_nordvik", "is_repeat_of_seed", "chosen_pos"]].to_csv(DER_DIR / "e1_long.csv", index=False)
    rows = []
    for (code, m, ri), g in s.groupby(base):
        g = g.sort_values("round_number")
        par = g[g.round_number >= 2]
        streak = 0
        for b in par.brand:
            if b != "Nordvik":
                break
            streak += 1
        sw = par[par.brand != g.brand.iloc[0]]
        rows.append(dict(experiment_code=code, model=m, run_index=ri, seed_pick=g.brand.iloc[0],
                         **{f"brand_r{r}": b for r, b in zip(g.round_number, g.brand)},
                         seeded_nordvik=int(g.brand.iloc[0] == "Nordvik"),
                         n_nordvik_parity=int((par.brand == "Nordvik").sum()), streak_from_r2=streak,
                         first_switch_round=int(sw.round_number.iloc[0]) if len(sw) else np.nan,
                         never_switched=int(len(sw) == 0)))
    pd.DataFrame(rows).to_csv(DER_DIR / "e1_wide.csv", index=False)
    for fam, name in (("E2", "e2_cell"), ("E3", "e3")):
        s = R[R.family == fam]
        rows = []
        for (code, m, ri), g in s.groupby(base):
            seed = g[g.phase == "seeding"]
            fin = g[g.phase == "test"].iloc[0]
            k = int(g.k.iloc[0])
            d = dict(experiment_code=code, model=m, k=k, premium_pct=int(g.p.iloc[0]), run_index=ri,
                     n_seed_nordvik=int((seed.brand == "Nordvik").sum()), n_seed_committed=len(seed),
                     fully_seeded=int((seed.brand == "Nordvik").sum() == k), final_brand=fin.brand,
                     final_product=fin.product_id, final_price=fin.price_paid, final_reason=fin.reason_text,
                     final_display_position=fin.chosen_pos, round_status_final=fin.round_status)
            d["retained" if fam == "E2" else "spilled"] = int(fin.brand == "Nordvik")
            d["switched_to"] = fin.brand if fin.brand != "Nordvik" else ""
            if fam == "E3":
                d.update({f"seed_r{r}": b for r, b in zip(seed.round_number, seed.brand)})
            rows.append(d)
        pd.DataFrame(rows).to_csv(DER_DIR / f"{name}.csv", index=False)
    # Master round-level share table across every experiment
    sh = (R.groupby(["experiment_code", "family", "model", "round_number", "phase", "nordvik_premium_pct"])
          .agg(n=("is_nordvik", "size"), x_nordvik=("is_nordvik", "sum")).reset_index())
    sh["p_nordvik"] = sh.x_nordvik / sh.n
    ci = [wilson(x, n) for x, n in zip(sh.x_nordvik, sh.n)]
    sh["wilson_low"], sh["wilson_high"] = zip(*ci)
    sh["model"] = sh.model.map(MODEL_LABEL)
    save_table(sh, "tab01b_nordvik_share_by_round_all", "Nordvik share in every round of every experiment.", tex=False)


# ════════════════════════════════════════════════════════════════════════
# 13. Multiplicity + report
# ════════════════════════════════════════════════════════════════════════
def finalize_tests():
    T = pd.DataFrame(TESTS)
    T["p_holm"] = np.nan
    for fam, idx in T.groupby("family").groups.items():
        ps = T.loc[idx, "p_value"]
        ok = ps.notna()
        if ok.any():
            T.loc[ps[ok].index, "p_holm"] = multipletests(ps[ok], method="holm")[1]
    ok = T.p_value.notna()
    T["p_bh"] = np.nan
    T.loc[ok, "p_bh"] = multipletests(T.loc[ok, "p_value"], method="fdr_bh")[1]
    T["significant_holm_05"] = T.p_holm < 0.05
    save_table(T, "tab24_all_hypothesis_tests",
               "Registry of all hypothesis tests with Holm-adjusted (within family) and Benjamini-Hochberg-adjusted "
               "(global) p-values.")
    return T


def write_report(audit_df, err_df, T):
    L = []
    a = audit_df[audit_df.family != "PILOT"]
    L += ["# Agent-Loyalty — Results Report (auto-generated)", "",
          f"Generated by `results/analyze_results.py` from `{CSV_PATH.name}`. Every number below is recomputed on each run.",
          "", "## 1. Data and quality audit", "",
          f"- Experiments analysed: **{len(a)}** (pilot `RQ1_Test_10rounds` excluded, as prescribed by RESULTS_GUIDE §5).",
          f"- Finished trajectories: **{int(a.finished.sum())}** of {int(a.intended_runs.sum())} intended "
          f"({100 * a.finished.sum() / a.intended_runs.sum():.1f}%); failed trajectories: {int(a.failed.sum())}; "
          f"trajectories with no stored round (failed in round 1): {int(a.missing_no_rounds.sum())}.",
          f"- Failed rounds inside finished trajectories: {int(a.failed_rounds.sum())}; duplicate (code, run, round) keys: "
          f"{int(a.duplicate_grain.sum())}; price_paid ≠ scheduled price: {int(a.price_mismatch.sum())}; "
          f"choices of unavailable products: {int(a.unavailable_choice.sum())}.",
          "- Failure causes: " + "; ".join(f"{k} (n={v})" for k, v in err_df.error_type.value_counts().items()) + ".",
          "- Display positions are not stored in the database; they were reconstructed exactly from the presentation "
          "seed (`random.Random(f\"{seed + run_index - 1}:{round}\")` applied to the listing order, see store/engine.py).",
          "", "See `tables/tab01_data_audit.csv`.", "", "## 2. Key results", ""]
    L += ["### RQ1 — Does a promotion create persistent repeat purchasing at parity?", ""]
    for m in MODELS:
        xb, nb = KEY[f"e0_nordvik_{m}"]
        rep, lo, hi = KEY[f"e0_4r_repeat_{m}"]
        e1 = KEY[f"e1_{m}"]
        L.append(f"- **{MODEL_LABEL[m]}** — no-history Nordvik share {fmt_pct(xb / nb)} ({xb}/{nb}); spontaneous repeat "
                 f"rate in E0-4R {fmt_pct(rep)} [95% CI {fmt_pct(lo)}, {fmt_pct(hi)}]; after one discounted purchase, "
                 f"Nordvik share in the first parity round {fmt_pct(e1['r2'][0] / e1['r2'][1])} "
                 f"({e1['r2'][0]}/{e1['r2'][1]}) and in round 7 {fmt_pct(e1['r7'][0] / e1['r7'][1])}; "
                 f"{e1['all'][0]}/{e1['all'][1]} seeded trajectories bought Nordvik in all six parity rounds. "
                 f"Staying with the round-1 brand through R4: E1 {fmt_pct(e1['stay1'][0] / e1['stay1'][1])} vs "
                 f"E0-4R {fmt_pct(e1['stay0'][0] / e1['stay0'][1])}.")
    x, n = KEY["e2_gpt_pooled"]
    L += ["", "### RQ2 — Does purchase history buy tolerance to a price premium?", "",
          f"- Across the 20-cell GPT-5.6-Luna grid, fully seeded agents retained Nordvik in **{x}/{n} "
          f"({fmt_pct(x / n)})** test rounds although the premium never exceeded +5% (≤ 0.75 currency units on a "
          "non-binding budget of 100). Contrast with near-100% repeat at parity (E1): see `tab12_e2_premium_cliff`."]
    for m in ("gemini-3.5-flash-lite", "deepseek-v4.1-flash"):
        C = KEY["e2_cells"]
        s = C[C.model_id == m]
        if len(s):
            s = s.iloc[0]
            L.append(f"- {MODEL_LABEL[m]} (k=3, +1%): retention {int(s.x_retained_cond)}/{int(s.n_fully_seeded)} "
                     f"({fmt_pct(s.retention_cond)}).")
    for m in MODELS:
        if f"wtp_{m}" in KEY:
            w, lo, hi = KEY[f"wtp_{m}"]
            L.append(f"- Choice model, {MODEL_LABEL[m]}: one prior purchase is worth a **{w:.2f}%** price premium "
                     f"[bootstrap 95% CI {lo:.2f}, {hi:.2f}] (ridge-penalised; see sensitivity table tab20).")
    L += ["", "### RQ3 — Does loyalty spill over to a new product of the same brand?", ""]
    for code, (x, n, m, p) in KEY["e3"].items():
        L.append(f"- `{code}` ({MODEL_LABEL[m]}, dish premium {p}%): spillover | fully seeded = **{x}/{n} "
                 f"({fmt_pct(x / n)})**.")
    sig = T[T.significant_holm_05]
    L += ["", "### Multiplicity", "",
          f"- {len(T)} hypothesis tests registered; {len(sig)} remain significant at α = .05 after Holm correction "
          "within family (`tables/tab24_all_hypothesis_tests.csv`).", ""]
    L += ["## 3. Statistical methods (draft text for the manuscript)", "",
          "All analyses were restricted to trajectories that completed every round (status = finished); "
          "trajectories aborted by malformed structured output were excluded and are reported in the audit table. "
          "Proportions are reported with Wilson score 95% confidence intervals and, as a Bayesian check robust to "
          "0/n and n/n outcomes, Jeffreys intervals. Brand and display-position shares in no-history decisions were "
          "tested against uniform choice with chi-square goodness-of-fit tests (effect size Cohen's w). Repeat "
          "probabilities were compared with the chance level of 1/3 using exact binomial tests; pooled repeat rates "
          "use trajectory-cluster bootstrap intervals (2,000 replicates). Between-condition contrasts are reported "
          "as risk differences with Newcombe hybrid-score intervals, Haldane–Anscombe-corrected odds ratios and "
          "Cohen's h, tested with Fisher's exact test. Dose–response in the premium grid was assessed with "
          "Cochran–Armitage trend tests and logistic regression; homogeneity across cells and across LLMs was tested "
          "with Monte-Carlo permutation chi-square tests (10,000 permutations). To estimate the relative weight of "
          "price, purchase history, display position and brand within a single framework we fitted a conditional "
          "(McFadden) logit over the three alternatives in each choice set. Because several regimes are perfectly "
          "separated (e.g., 100% repeat at parity, ~0% retention under any premium), maximum-likelihood estimates "
          "diverge; we therefore used a ridge penalty (λ = 1; sensitivity to λ ∈ {0.1, 1, 10} reported) and "
          "trajectory-cluster bootstrap CIs (300 replicates). The ratio of the history and price coefficients gives "
          "the history-equivalent price premium. Free-text justifications were coded with a transparent regular-"
          "expression lexicon (eight themes); distinctive vocabulary was identified with weighted log-odds ratios "
          "with an informative Dirichlet prior. A stratified sample is provided for blind human double-coding to "
          "validate the lexicon (Cohen's κ). P-values were adjusted with Holm's procedure within pre-defined test "
          "families and with Benjamini–Hochberg globally.", "",
          "## 4. Figures", ""]
    for name, cap in FIGURES:
        L.append(f"- **{name}** (`figures/{name}.png`, `.pdf`): {cap}")
    L += ["", "## 5. Tables", ""]
    for name, cap in TABLES:
        L.append(f"- **{name}**: {cap}")
    L += ["", "## 6. Caveats and recommended robustness steps before submission", "",
          "1. **Ceiling/floor effects.** Many cells are 0% or 100%; report exact counts alongside rates and rely on "
          "exact/permutation tests (done here). Consider adding parity (p = 0) cells to the E2 grid for a clean within-"
          "design reference, and finer premiums (e.g., +0.25%, +0.5%) to locate the switching threshold.",
          "2. **Lexicon coding is a proxy.** Use `tables/reason_coding_sample.csv` for two blind human coders (or an "
          "LLM coder validated against humans) and report Cohen's κ before interpreting theme prevalences causally.",
          "3. **Model identity.** Results are conditional on three specific model versions at temperature 0.7; "
          "report model versions and dates, and consider a temperature-sensitivity replication.",
          "4. **E2 parity reference.** The E1 reference used for the premium cliff has 1 discounted + k−1 full-price "
          "purchases, not k discounted purchases.",
          "5. **Fictional brands and equal quality** guarantee internal validity but limit external validity; discuss "
          "explicitly.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
# 14. Main
# ════════════════════════════════════════════════════════════════════════
def main():
    raw, meta, design = load_data(CSV_PATH)
    print("[audit]")
    audit_df, err_df = audit(raw, meta, design)
    save_table(audit_df, "tab01_data_audit",
               "Data audit per experiment: intended vs observed trajectories, completion, failures and integrity checks.")
    save_table(err_df, "tab01a_failed_trajectories", "Failed trajectories and failure cause (malformed structured output).")
    R = build_round_frame(raw, meta, design)
    print(f"[data] analysable rounds: {len(R)}  trajectories: {R.traj.nunique()}")
    fig_design(meta, design)
    print("[RQ1] E0")
    analyse_e0(R)
    W0, _ = analyse_e0_4r(R)
    print("[RQ1] E1")
    W1, _ = analyse_e1(R, W0)
    print("[RQ2] E2")
    analyse_e2(R, W1)
    analyse_switchers(R)
    print("[RQ3] E3")
    analyse_e3(R)
    print("[cross-model]")
    analyse_cross_model(R, W0, W1)
    print("[choice model] (cluster bootstrap, may take a minute)")
    analyse_choice_model(R, meta, design)
    print("[reasons]")
    analyse_reasons(R)
    print("[derived tables]")
    export_derived(R)
    T = finalize_tests()
    write_report(audit_df, err_df, T)
    print(f"[done] {len(FIGURES)} figures -> {FIG_DIR}\n       {len(TABLES)} tables  -> {TAB_DIR}\n"
          f"       report        -> {HERE / 'REPORT.md'}")


if __name__ == "__main__":
    main()
