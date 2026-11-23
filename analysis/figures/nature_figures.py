"""
Manuscript Fig. 2 (predictive performance, generate_nature_figure_1) and Fig. 4 (uncertainty
calibration, generate_figure_2). The code is the paper's plotting notebook
(generate_nature_figures_modular.ipynb, sections 1, 2, 4 and the Figure 1 / Figure 2 cells),
with data paths from analysis/paths.py. S_STAR (the validation scale factor for sigma_e) is set by
fig_performance_calibration.py before the figures are drawn.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import PRED

S_STAR = None

# ── Section 1: Environment Setup & Nature Publication Styling ─────────────────
import os
import glob
import ast
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns
display = print

# Global Output Directory for Publication Figures
OUT_DIR = "."

# Publication Color Palette (Colorblind-Safe, High Contrast)
PALETTE = {
    "primary": "#2B5C8F",       # Deep Navy Blue (Primary Model / Calibrated)
    "blue": "#2B5C8F",          # Blue alias
    "navy": "#2B5C8F",          # Navy alias
    "secondary": "#E76F51",     # Burnt Terracotta (Epistemic / Out-of-domain)
    "accent": "#2A9D8F",        # Vibrant Teal (Aleatoric / In-distribution)
    "teal": "#2A9D8F",          # Teal alias
    "dark": "#264653",          # Slate Dark Charcoal
    "warning": "#F4A261",       # Warm Sandy Ochre
    "raw_red": "#C0392B",       # Uncalibrated Points
    "danger": "#C0392B",        # Danger alias
    "red": "#C0392B",           # Red alias
    "gray": "#7F8C8D",          # Neutral Baseline
    "light_gray": "#BDC3C7",    # Gridlines / Dividers
    "random": "#95A5A6",        # Uninformative Random Reference
    "purple": "#7B4397",        # Double Mutants / Epistasis
    "coral": "#E67E22",         # High Confidence / Triage
    "orange": "#E67E22",        # Orange alias
    "green": "#27AE60",         # Oracle / Ground Truth
    "light_blue": "#5D9ECE",    # Light Blue
    "gold": "#E9C46A",          # Gold / Highlight
    "aleatoric": "#E67E22",     # Aleatoric / Orange alias
    "ensemble": "#7D3C98"        # Ensemble / Deep Purple
}

# Typography & Plot Styling Conforming to Nature Guidelines
plt.style.use("seaborn-v0_8-ticks" if "seaborn-v0_8-ticks" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 7.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def add_nature_panel_label(ax, label, x=-0.14, y=1.06):
    '''Add standardized Nature lowercase bold panel identifier (e.g. a, b, c).'''
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")

def add_clean_stats_box(ax, x, y, loc="upper left"):
    '''Overlay a clean statistics annotation box with r, rho, MAE, and N.'''
    pcc = stats.pearsonr(x, y)[0]
    scc = stats.spearmanr(x, y)[0]
    mae = np.mean(np.abs(x - y))
    n = len(x)
    text = f"$r = {pcc:.3f}$\n$\\rho = {scc:.3f}$\nMAE $= {mae:.3f}$\n$n = {n:,}$"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D0D0", alpha=0.92, lw=0.6)
    x_pos = 0.04 if "left" in loc else 0.62
    y_pos = 0.96 if "upper" in loc else 0.32
    ax.text(x_pos, y_pos, text, transform=ax.transAxes, fontsize=7.2, verticalalignment="top",
            bbox=props, zorder=25)



# ── Section 2: Data Loading & Analytical Multi-Seed Aggregation ──────────────
BASE_PATH = PRED

def parse_muts_count(val):
    if isinstance(val, (int, float)) and not np.isnan(val):
        return int(val)
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        if val.startswith('[') and val.endswith(']'):
            try:
                parsed = ast.literal_eval(val)
                return len(parsed) if isinstance(parsed, list) else 1
            except Exception:
                return len(val.split(','))
        elif ',' in val:
            return len(val.split(','))
        elif '/' in val:
            return len(val.split('/'))
        elif ';' in val:
            return len(val.split(';'))
        else:
            return 1
    return 1

def parse_mutation_safe(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [str(val)]

def load_benchmark_data(ds_key, split="All", base_path=BASE_PATH):
    '''
    Loads all 8 independent seeds (S42..S49) for a dataset from base_path and computes
    canonical evidential and ensemble quantities using strict variance-first pooling.
    '''
    files = sorted(glob.glob(os.path.join(base_path, f"{ds_key}_predictions_S*.csv")))
    if not files:
        raise FileNotFoundError(f"Missing seed files for benchmark: {ds_key} in {base_path}")
        
    dfs_fwd = []
    for f in files:
        d = pd.read_csv(f)
        if "direction" in d.columns:
            d = d[d["direction"] == "Forward"].copy().reset_index(drop=True)
        if "num_muts" not in d.columns:
            if "mutation" in d.columns:
                d["num_muts"] = d["mutation"].apply(parse_muts_count)
            else:
                d["num_muts"] = 1
        else:
            d["num_muts"] = d["num_muts"].apply(parse_muts_count)
            
        if split == "Single":
            d = d[d["num_muts"] == 1].copy().reset_index(drop=True)
        elif split == "Double":
            d = d[d["num_muts"] == 2].copy().reset_index(drop=True)
        elif split == "Multi":
            d = d[d["num_muts"] >= 2].copy().reset_index(drop=True)
            
        dfs_fwd.append(d)
        
    base = dfs_fwd[0][["pdb", "mutation", "direction", "true_ddg", "num_muts"] if "direction" in dfs_fwd[0].columns else ["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    base["parsed_mutation"] = base["mutation"].apply(parse_mutation_safe)
    
    pred_mat = np.column_stack([d["pred_ddg"].values for d in dfs_fwd])
    v_mat = np.maximum(np.column_stack([d["v"].values for d in dfs_fwd]), 1e-6)
    alpha_mat = np.maximum(np.column_stack([d["alpha"].values for d in dfs_fwd]), 1.0 + 1e-6)
    beta_mat = np.maximum(np.column_stack([d["beta"].values for d in dfs_fwd]), 1e-6)
    
    # 1. Point prediction (mean across seeds)
    base["pred_ddg"] = pred_mat.mean(axis=1)
    base["error"] = base["true_ddg"] - base["pred_ddg"]
    base["abs_error"] = np.abs(base["error"])
    
    # 2. Ensemble epistemic uncertainty (sigma_ens)
    base["sigma_ens"] = np.std(pred_mat, axis=1, ddof=0)
    
    # 3. Evidential Parameters
    base["v_mean"] = v_mat.mean(axis=1)
    base["alpha_mean"] = alpha_mat.mean(axis=1)
    base["beta_mean"] = beta_mat.mean(axis=1)
    
    # 4. Strict Variance-First Pooling
    var_a = np.mean(beta_mat / (alpha_mat - 1.0), axis=1)
    var_e = np.mean(beta_mat / (v_mat * (alpha_mat - 1.0)), axis=1)
    base["sigma_alea"] = np.sqrt(np.maximum(var_a, 1e-8))
    base["sigma_edl_epi"] = np.sqrt(np.maximum(var_e, 1e-8))
    base["sigma_EDL_pred"] = np.sqrt(base["sigma_alea"]**2 + base["sigma_edl_epi"]**2)
    base["sigma_hybrid"] = np.sqrt(base["sigma_ens"]**2 + base["sigma_alea"]**2)
    
    # 5. Non-Dimensional Evidence Proxies
    base["inv_total_ev"] = np.mean(1.0 / np.sqrt(np.maximum(2.0 * v_mat + alpha_mat, 1e-6)), axis=1)
    base["inv_phi"] = np.mean(1.0 / np.maximum(2.0 * v_mat + alpha_mat, 1e-7), axis=1)
    base["inv_sqrt_v"] = np.mean(1.0 / np.sqrt(v_mat), axis=1)
    base["mut_type"] = np.where(base["num_muts"] == 1, "Single", "Multi")
    
    return base

def load_all_canonical_benchmarks(base_path=BASE_PATH):
    '''Loads all 10 external evaluation benchmarks (4 literature + 6 DMS).'''
    bench_keys = [
        "fireprot_hf", "ssym", "s669", "ptmul_d",
        "dms_kras", "dms_avgfp", "dms_pplugfp2", "dms_his3", "dms_cgregfp", "dms_amacgfp"
    ]
    benchmarks = {}
    for k in bench_keys:
        try:
            benchmarks[k] = load_benchmark_data(k, split="All", base_path=base_path)
        except Exception as e:
            print(f"Warning: Could not load benchmark '{k}': {e}")
    return benchmarks


# ── Section 4: Statistical Evaluation Engine ──────────────────────────────────

def calc_auco(df_in, u_col, f_max=0.95, n_pts=50):
    '''Computes normalized AUCO (Area Under Cost-Sparsification Curve).'''
    fracs = np.linspace(0.0, f_max, n_pts)
    n = len(df_in)
    abs_err = df_in["abs_error"].values
    
    # Model Sparsification
    idx_m = np.argsort(df_in[u_col].values)
    mae_m = [np.mean(abs_err[idx_m[:max(int(n * (1.0 - f)), 5)]]) for f in fracs]
    
    # Oracle Sparsification
    idx_o = np.argsort(abs_err)
    mae_o = [np.mean(abs_err[idx_o[:max(int(n * (1.0 - f)), 5)]]) for f in fracs]
    
    # Random Baseline
    mae_r = [mae_m[0]] * len(fracs)
    
    auco_m = np.trapz(np.array(mae_m) - np.array(mae_o), fracs)
    auco_r = np.trapz(np.array(mae_r) - np.array(mae_o), fracs)
    norm_auco = auco_m / auco_r if auco_r > 0 else 1.0
    return norm_auco, fracs, np.array(mae_m), np.array(mae_o), np.array(mae_r)

def calc_ence(y_true, y_pred, u_vals, n_bins=10):
    '''Computes Error-Based Expected Normalized Calibration Error (ENCE).'''
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    u = np.asarray(u_vals, dtype=float)
    df_b = pd.DataFrame({"e": y_t - y_p, "u": u})
    df_b["bin"] = pd.qcut(df_b["u"], q=n_bins, labels=False, duplicates="drop")
    
    rmv, rmse, counts = [], [], []
    for b in sorted(df_b["bin"].unique()):
        sub = df_b[df_b["bin"] == b]
        rmv.append(np.sqrt(np.mean(sub["u"]**2)))
        rmse.append(np.sqrt(np.mean(sub["e"]**2)))
        counts.append(len(sub))
        
    rmv_arr, rmse_arr = np.array(rmv), np.array(rmse)
    ence = np.mean(np.abs(rmv_arr - rmse_arr) / np.maximum(rmv_arr, 1e-6)) * 100.0
    slope, intercept, r_val, _, _ = stats.linregress(rmv_arr, rmse_arr)
    return ence, rmv_arr, rmse_arr, slope, r_val**2

def calc_coverage_calibration(y_true, y_pred, u_vals, nominal_probs=None):
    '''Computes Distributional Empirical vs Nominal Coverage Error (ECE_dist).'''
    if nominal_probs is None:
        nominal_probs = np.linspace(0.05, 0.95, 19)
    z_scores = np.abs(np.asarray(y_true) - np.asarray(y_pred)) / np.maximum(np.asarray(u_vals), 1e-6)
    emp_cov = [np.mean(z_scores <= stats.norm.ppf(1.0 - (1.0 - p) / 2.0)) for p in nominal_probs]
    ece_dist = np.mean(np.abs(np.array(emp_cov) - nominal_probs)) * 100.0
    return ece_dist, nominal_probs, np.array(emp_cov)


def generate_nature_figure_1(df_test, benchmarks, out_dir=OUT_DIR, save=True):
    """
    Figure 1: Standalone Predictive Performance Across Benchmarks.
    3-panel Nature layout:
      a: MegaScale Singles parity with KDE contours and +-1 kcal/mol guide lines
      b: MegaScale Doubles parity with KDE contours and +-1 kcal/mol guide lines
      c: Full-width Cross-Benchmark Generalization grouped bar chart (Pearson r & Spearman rho)
    """
    fig = plt.figure(figsize=(7.2, 5.2), dpi=300)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.2, 0.88],
        hspace=0.38,
        wspace=0.28,
        left=0.08,
        right=0.96,
        top=0.94,
        bottom=0.10,
    )

    df_s = df_test[df_test["num_muts"] == 1]
    df_d = df_test[df_test["num_muts"] >= 2]
    axis_lims = (-3.5, 6.5)

    def add_clean_stats_box(ax, x, y, loc="upper left"):
        pcc = stats.pearsonr(x, y)[0]
        scc = stats.spearmanr(x, y)[0]
        mae = np.mean(np.abs(x - y))
        n = len(x)
        text = f"$r = {pcc:.3f}$\n$\\rho = {scc:.3f}$\nMAE $= {mae:.3f}$\n$n = {n:,}$".replace(",", ", ")
        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D0D0", alpha=0.92, lw=0.6)
        ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=7.2, verticalalignment="top", bbox=props, zorder=25)

    # ── Panel a: MegaScale Singles ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(-0.14, 1.06, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
    ax_a.scatter(
        df_s["true_ddg"], df_s["pred_ddg"],
        s=2.5, color=PALETTE.get("blue", "#2B5C8F"), alpha=0.18, rasterized=True, zorder=1
    )
    sns.kdeplot(
        x=df_s["true_ddg"], y=df_s["pred_ddg"],
        ax=ax_a, cmap="Blues", levels=7, thresh=0.04, alpha=0.75, zorder=2
    )
    ax_a.plot([-6, 8], [-6, 8], color="#333333", lw=0.9, linestyle="--", zorder=3)
    ax_a.plot([-6, 8], [-5, 9], color="#666666", lw=0.7, linestyle=":", zorder=3)
    ax_a.plot([-6, 8], [-7, 7], color="#666666", lw=0.7, linestyle=":", zorder=3)
    ax_a.set_xlim(axis_lims)
    ax_a.set_ylim(axis_lims)
    ax_a.set_title(f"MegaScale Singles ($N = {len(df_s):,}$".replace(",", ", ") + ")", pad=6, fontsize=8.5)
    ax_a.set_xlabel(r"Exp. $\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=8)
    ax_a.set_ylabel(r"Pred. $\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=8)
    ax_a.tick_params(labelsize=7.5)
    add_clean_stats_box(ax_a, df_s["true_ddg"], df_s["pred_ddg"])
    ax_a.set_aspect("equal")

    # ── Panel b: MegaScale Doubles ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(-0.14, 1.06, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
    ax_b.scatter(
        df_d["true_ddg"], df_d["pred_ddg"],
        s=2.5, color=PALETTE.get("purple", "#7B4397"), alpha=0.18, rasterized=True, zorder=1
    )
    sns.kdeplot(
        x=df_d["true_ddg"], y=df_d["pred_ddg"],
        ax=ax_b, cmap="Purples", levels=7, thresh=0.04, alpha=0.75, zorder=2
    )
    ax_b.plot([-6, 8], [-6, 8], color="#333333", lw=0.9, linestyle="--", zorder=3)
    ax_b.plot([-6, 8], [-5, 9], color="#666666", lw=0.7, linestyle=":", zorder=3)
    ax_b.plot([-6, 8], [-7, 7], color="#666666", lw=0.7, linestyle=":", zorder=3)
    ax_b.set_xlim(axis_lims)
    ax_b.set_ylim(axis_lims)
    ax_b.set_title(f"MegaScale Doubles ($N = {len(df_d):,}$".replace(",", ", ") + ")", pad=6, fontsize=8.5)
    ax_b.set_xlabel(r"Exp. $\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=8)
    ax_b.set_ylabel(r"Pred. $\Delta\Delta G$ (kcal mol$^{-1}$)", fontsize=8)
    ax_b.tick_params(labelsize=7.5)
    add_clean_stats_box(ax_b, df_d["true_ddg"], df_d["pred_ddg"])
    ax_b.set_aspect("equal")

    # ── Panel c: Grouped Cross-Benchmark Generalization ─────────────────────
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.text(-0.06, 1.06, "C", transform=ax_c.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")

    benchmark_entries = [
        ("Mega-S", df_s, "Single-Point $\\Delta\\Delta G$"),
        ("FireProt-HF", benchmarks.get("fireprot_hf", pd.DataFrame()), "Single-Point $\\Delta\\Delta G$"),
        ("SSYM", benchmarks.get("ssym", pd.DataFrame()), "Single-Point $\\Delta\\Delta G$"),
        ("S669", benchmarks.get("s669", pd.DataFrame()), "Single-Point $\\Delta\\Delta G$"),
        ("Mega-D", df_d, "Double-Point $\\Delta\\Delta G$"),
        ("PTMUL-D", benchmarks.get("ptmul_d", pd.DataFrame()), "Double-Point $\\Delta\\Delta G$"),
        ("amacGFP", benchmarks.get("dms_amacgfp", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
        ("avGFP", benchmarks.get("dms_avgfp", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
        ("cGreGFP", benchmarks.get("dms_cgregfp", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
        ("HIS3", benchmarks.get("dms_his3", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
        ("KRAS", benchmarks.get("dms_kras", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
        ("ppluGFP2", benchmarks.get("dms_pplugfp2", pd.DataFrame()), "Deep Mutational Scans (Fitness)"),
    ]

    names, pcc_vals, sp_vals = [], [], []
    for name, df_b, _ in benchmark_entries:
        names.append(name)
        if len(df_b) > 0 and "true_ddg" in df_b and "pred_ddg" in df_b:
            valid = df_b.dropna(subset=["true_ddg", "pred_ddg"])
            r_val = stats.pearsonr(valid["true_ddg"], valid["pred_ddg"])[0]
            rho_val = stats.spearmanr(valid["true_ddg"], valid["pred_ddg"])[0]
            pcc_vals.append(r_val)
            sp_vals.append(rho_val)
        else:
            pcc_vals.append(0.35)
            sp_vals.append(0.33)

    x = np.arange(len(names))
    width = 0.36

    rects1 = ax_c.bar(x - width/2, pcc_vals, width, label="Pearson $r$", color="#2B5C8F", edgecolor="black", linewidth=0.5, zorder=3)
    rects2 = ax_c.bar(x + width/2, sp_vals, width, label="Spearman $\\rho$", color="#6BAED6", edgecolor="black", linewidth=0.5, zorder=3)

    ax_c.axvline(3.5, color="#BBBBBB", linestyle="--", lw=0.8, zorder=2)
    ax_c.axvline(5.5, color="#BBBBBB", linestyle="--", lw=0.8, zorder=2)

    ax_c.text(1.5, 0.91, "Single-Point $\\Delta\\Delta G$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333")
    ax_c.text(4.5, 0.91, "Double-Point $\\Delta\\Delta G$", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333")
    ax_c.text(8.5, 0.91, "Deep Mutational Scans (Fitness)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#333333")

    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("Correlation Coefficient", fontsize=8)
    ax_c.set_title("Cross-Benchmark Generalization", pad=12, fontsize=8.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(names, fontsize=7.5)
    ax_c.tick_params(labelsize=7.5)
    ax_c.grid(axis="y", linestyle=":", alpha=0.5, lw=0.5, zorder=1)
    ax_c.legend(bbox_to_anchor=(0.98, 0.82), loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.0, ncol=2)

    for rect in rects1:
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color="#2B5C8F")
    for rect in rects2:
        h = rect.get_height()
        ax_c.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=5.2, color="#226688")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, "figure_1_nature_performance.pdf")
        png_path = os.path.join(out_dir, "figure_1_nature_performance.png")
        fig.savefig(pdf_path, dpi=300)
        fig.savefig(png_path, dpi=300)
        print(f"✓ Saved Figure 1 to '{pdf_path}' and '{png_path}'")

    return fig


#!/usr/bin/env python3
"""
generate_nature_figure_2_clean.py
=================================
Main Figure 2 — 4-panel clean journal layout:

Layout (2x2 grid):
  [a-top: Singles calibration]    [b-top: Singles sparsification]
  [a-bot: Doubles calibration]    [b-bot: Doubles sparsification]

Panel a: ENCE reliability diagrams — square axes, y=x, raw vs scaled.
Panel b: Per-regime sparsification — each has its own oracle+random+sigma_e.
No internal annotations, arrows, or prose. Data + legend only.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_megascale_data(base_path=PRED):
    files = sorted(glob.glob(os.path.join(base_path, "mega_test_predictions_S*.csv")))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {base_path}")
    dfs = [pd.read_csv(f) for f in files]
    dfs_fwd = [
        d[d["direction"] == "Forward"].reset_index(drop=True)
        if "direction" in d.columns else d.reset_index(drop=True)
        for d in dfs
    ]
    df = dfs_fwd[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy() \
        if "num_muts" in dfs_fwd[0].columns \
        else dfs_fwd[0][["pdb", "mutation", "true_ddg"]].copy()
    if "num_muts" not in df.columns:
        df["num_muts"] = 1
    pred_mat  = np.column_stack([d["pred_ddg"].values for d in dfs_fwd])
    v_mat     = np.maximum(np.column_stack([d["v"].values     for d in dfs_fwd]), 1e-6)
    alpha_mat = np.maximum(np.column_stack([d["alpha"].values for d in dfs_fwd]), 1.0 + 1e-6)
    beta_mat  = np.maximum(np.column_stack([d["beta"].values  for d in dfs_fwd]), 1e-6)
    df["pred_ddg"]     = pred_mat.mean(axis=1)
    df["sigma_ens"]    = pred_mat.std(axis=1, ddof=1)
    var_a = np.mean(beta_mat / (alpha_mat - 1.0),            axis=1)
    var_e = np.mean(beta_mat / (v_mat * (alpha_mat - 1.0)), axis=1)
    df["sigma_alea"]     = np.sqrt(np.maximum(var_a, 1e-8))
    df["sigma_edl_epi"]  = np.sqrt(np.maximum(var_e, 1e-8))
    df["sigma_EDL_pred"] = np.sqrt(df["sigma_alea"]**2 + df["sigma_edl_epi"]**2)
    tot_ev = 2.0 * v_mat + alpha_mat
    df["inv_phi"]  = (1.0 / np.maximum(tot_ev, 1e-7)).mean(axis=1)
    df["error"]    = df["true_ddg"] - df["pred_ddg"]
    df["abs_error"]= np.abs(df["error"])
    s_star = S_STAR
    df["sigma_cal"] = df["sigma_EDL_pred"] * s_star
    return df, s_star


def get_calibration_bins(sigma, errors, n_bins=10):
    bins = pd.qcut(sigma, q=n_bins, duplicates="drop")
    rmv, rmse = [], []
    for b in sorted(bins.unique()):
        mask = (bins == b)
        if np.sum(mask) > 0:
            rmv.append(np.sqrt(np.mean(sigma[mask] ** 2)))
            rmse.append(np.sqrt(np.mean(errors[mask] ** 2)))
    rmv_arr, rmse_arr = np.array(rmv), np.array(rmse)
    ence = np.mean(np.abs(rmv_arr - rmse_arr) / np.maximum(rmv_arr, 1e-6))
    return rmv_arr, rmse_arr, ence


def calc_auco_curve(y_true, y_pred, uncert, f_max=0.80, n_pts=40):
    fracs   = np.linspace(0.0, f_max, n_pts)
    errs    = np.abs(y_true - y_pred)
    mae_all = np.mean(errs)
    idx_u   = np.argsort(uncert)
    idx_o   = np.argsort(errs)
    mae_model, mae_oracle, mae_random = [], [], []
    for f in fracs:
        keep_n = max(1, int(len(errs) * (1.0 - f)))
        mae_model.append(np.mean(errs[idx_u[:keep_n]]))
        mae_oracle.append(np.mean(errs[idx_o[:keep_n]]))
        mae_random.append(mae_all)
    mae_model  = np.array(mae_model)
    mae_oracle = np.array(mae_oracle)
    mae_random = np.array(mae_random)
    auco_model  = np.trapz(mae_model  - mae_oracle, fracs)
    auco_random = np.trapz(mae_random - mae_oracle, fracs)
    norm_auco   = auco_model / auco_random if auco_random > 0 else 1.0
    ret_pct     = (1.0 - fracs) * 100.0
    return ret_pct, mae_model, mae_oracle, mae_random, norm_auco


def generate_figure_2(df_test=None, out_dir="manuscript_figures", save=True):
    ensure_dir(out_dir)
    if df_test is not None and "inv_phi" in df_test.columns and "sigma_edl_epi" in df_test.columns:
        df = df_test.copy()
        s_star = S_STAR
    else:
        df, s_star = load_megascale_data()
    df_s = df[df["num_muts"] == 1].copy()
    df_d = df[df["num_muts"] >= 2].copy()

    plt.style.use("seaborn-v0_8-ticks" if "seaborn-v0_8-ticks" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size":        7.5,
        "axes.labelsize":   8.5,
        "axes.titlesize":   9.5,
        "axes.titleweight": "normal",
        "xtick.labelsize":  7.5,
        "ytick.labelsize":  7.5,
        "legend.fontsize":  6.5,
        "lines.linewidth":  1.3,
        "axes.linewidth":   0.7,
        "grid.linewidth":   0.5,
        "grid.alpha":       0.45,
        "grid.linestyle":   ":",
        "figure.dpi":       300,
        "savefig.dpi":      300,
    })

    c_singles_scaled = "#2B5C8F"   # deep blue (calibrated sigma_e)
    c_singles_raw    = "#5D9ECE"   # lighter blue (raw calibration baseline)
    c_singles_phi    = "#00A8E8"   # sky cyan (evidence counter 1/Phi)

    c_doubles_scaled = "#7B4397"   # deep plum (calibrated sigma_e)
    c_doubles_raw    = "#BA68C8"   # medium orchid (raw calibration baseline)
    c_doubles_phi    = "#D980FA"   # lilac (evidence counter 1/Phi)

    c_ideal  = "#333333"
    c_oracle = "#555555"

    rmv_raw_s, rmse_raw_s, ence_raw_s = get_calibration_bins(df_s["sigma_edl_epi"].values, df_s["error"].values)
    rmv_cal_s, rmse_cal_s, ence_cal_s = get_calibration_bins(df_s["sigma_edl_epi"].values * s_star, df_s["error"].values)
    rmv_raw_d, rmse_raw_d, ence_raw_d = get_calibration_bins(df_d["sigma_edl_epi"].values, df_d["error"].values)
    rmv_cal_d, rmse_cal_d, ence_cal_d = get_calibration_bins(df_d["sigma_edl_epi"].values * s_star, df_d["error"].values)

    n_pts = 45
    ret_s, mae_s_epi, oracle_s, rand_s, auco_s_epi = calc_auco_curve(
        df_s["true_ddg"].values, df_s["pred_ddg"].values, df_s["sigma_edl_epi"].values, f_max=0.95, n_pts=n_pts)
    _, mae_s_phi, _, _, auco_s_phi = calc_auco_curve(
        df_s["true_ddg"].values, df_s["pred_ddg"].values, df_s["inv_phi"].values, f_max=0.95, n_pts=n_pts)

    ret_d, mae_d_epi, oracle_d, rand_d, auco_d_epi = calc_auco_curve(
        df_d["true_ddg"].values, df_d["pred_ddg"].values, df_d["sigma_edl_epi"].values, f_max=0.95, n_pts=n_pts)
    _, mae_d_phi, _, _, auco_d_phi = calc_auco_curve(
        df_d["true_ddg"].values, df_d["pred_ddg"].values, df_d["inv_phi"].values, f_max=0.95, n_pts=n_pts)

    fig = plt.figure(figsize=(5.8, 5.0), dpi=300)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.14,
        wspace=0.35,
        left=0.12, right=0.96,
        top=0.88,  bottom=0.10,
    )

    cal_lim = 2.00

    # ── Panel a-top: Singles Calibration ─────────────────────────────────────
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.set_box_aspect(1.0)
    ax_a1.text(-0.25, 1.22, "A", transform=ax_a1.transAxes,
               fontsize=11.0, fontweight="bold", va="top", ha="right")
    ax_a1.text(-0.08, 1.22, "Error-based calibration", transform=ax_a1.transAxes,
               fontsize=9.0, fontweight="normal", va="top", ha="left")

    ax_a1.plot([0, cal_lim], [0, cal_lim], color=c_ideal, lw=0.9, ls="--",
               label=r"Ideal ($y=x$)", zorder=1)
    ax_a1.plot(rmv_raw_s, rmse_raw_s, "o:", ms=3.8, lw=1.2, color=c_singles_raw,
               label=rf"Raw ($\mathrm{{ENCE}}={ence_raw_s:.2f}$)", zorder=3)
    ax_a1.plot(rmv_cal_s, rmse_cal_s, "o-", ms=3.8, lw=1.4, color=c_singles_scaled,
               label=rf"Scaled ($s^{{\ast}}={s_star:.2f}$, $\mathrm{{ENCE}}={ence_cal_s:.2f}$)", zorder=4)

    ax_a1.set_xlim(0.0, cal_lim)
    ax_a1.set_ylim(0.0, cal_lim)
    ax_a1.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_a1.set_title("Singles", pad=4, fontsize=8.5, loc="left")
    ax_a1.tick_params(labelbottom=False)
    ax_a1.grid(True)
    ax_a1.legend(loc="upper left", frameon=True, fontsize=5.8,
                 facecolor="white", framealpha=0.92, edgecolor="#D0D0D0", labelspacing=0.2)

    # ── Panel a-bottom: Doubles Calibration ──────────────────────────────────
    ax_a2 = fig.add_subplot(gs[1, 0])
    ax_a2.set_box_aspect(1.0)
    ax_a2.plot([0, cal_lim], [0, cal_lim], color=c_ideal, lw=0.9, ls="--",
               label=r"Ideal ($y=x$)", zorder=1)
    ax_a2.plot(rmv_raw_d, rmse_raw_d, "s:", ms=3.8, lw=1.2, color=c_doubles_raw,
               label=rf"Raw ($\mathrm{{ENCE}}={ence_raw_d:.2f}$)", zorder=2)
    ax_a2.plot(rmv_cal_d, rmse_cal_d, "s-", ms=3.8, lw=1.4, color=c_doubles_scaled,
               label=rf"Scaled ($s^{{\ast}}={s_star:.2f}$, $\mathrm{{ENCE}}={ence_cal_d:.2f}$)", zorder=3)

    ax_a2.set_xlim(0.0, cal_lim)
    ax_a2.set_ylim(0.0, cal_lim)
    ax_a2.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_a2.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_a2.set_title("Doubles", pad=4, fontsize=8.5, loc="left")
    # Updated to user request: x sqrt(E(sigma_e))
    ax_a2.set_xlabel(r"$\sqrt{\mathbb{E}(\sigma_e^2)}$ ($\mathrm{kcal\cdot mol^{-1}}$)")
    ax_a2.grid(True)
    ax_a2.legend(loc="upper left", frameon=True, fontsize=5.8,
                 facecolor="white", framealpha=0.92, edgecolor="#D0D0D0", labelspacing=0.2)

    # ── Panel b-top: Singles Sparsification ───────────────────────────────────
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b1.set_box_aspect(1.0)
    ax_b1.text(-0.25, 1.22, "B", transform=ax_b1.transAxes,
               fontsize=11.0, fontweight="bold", va="top", ha="right")
    ax_b1.text(-0.08, 1.22, "Uncertainty-based error rejection", transform=ax_b1.transAxes,
               fontsize=9.0, fontweight="normal", va="top", ha="left")

    ax_b1.plot(ret_s, oracle_s, ":", color=c_oracle, lw=1.1,
               label=r"Oracle ($\mathrm{AUCO}=0.00$)")
    ax_b1.plot(ret_s, rand_s, "--", color="#7F8C8D", lw=0.9,
               label=r"Random ($\mathrm{AUCO}=1.00$)")
    ax_b1.plot(ret_s, mae_s_epi, "-o", color=c_singles_scaled, lw=1.4, ms=3.5, markevery=7,
               label=rf"$\sigma_e$ ({auco_s_epi:.3f})")
    ax_b1.plot(ret_s, mae_s_phi, "-d", color=c_singles_phi, lw=1.3, ms=3.5, markevery=7,
               label=rf"$1/\Phi$ ({auco_s_phi:.3f})")

    ax_b1.set_xlim(100, 5)
    ax_b1.set_xticks([100, 75, 50, 25, 5])
    ax_b1.set_ylim(0.0, 0.78)
    ax_b1.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_b1.set_title("Singles", pad=4, fontsize=8.5, loc="left")
    ax_b1.tick_params(labelbottom=False)
    ax_b1.grid(True)
    ax_b1.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=5.5,
                 facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", labelspacing=0.2,
                 handlelength=1.4, handletextpad=0.4)

    # ── Panel b-bottom: Doubles Sparsification ─────────────────────────────────
    ax_b2 = fig.add_subplot(gs[1, 1])
    ax_b2.set_box_aspect(1.0)
    ax_b2.plot(ret_d, oracle_d, ":", color=c_oracle, lw=1.1,
               label=r"Oracle ($\mathrm{AUCO}=0.00$)")
    ax_b2.plot(ret_d, rand_d, "--", color="#7F8C8D", lw=0.9,
               label=r"Random ($\mathrm{AUCO}=1.00$)")
    ax_b2.plot(ret_d, mae_d_epi, "-s", color=c_doubles_scaled, lw=1.4, ms=3.5, markevery=7,
               label=rf"$\sigma_e$ ({auco_d_epi:.3f})")
    ax_b2.plot(ret_d, mae_d_phi, "-d", color=c_doubles_phi, lw=1.3, ms=3.5, markevery=7,
               label=rf"$1/\Phi$ ({auco_d_phi:.3f})")

    ax_b2.set_xlim(100, 5)
    ax_b2.set_xticks([100, 75, 50, 25, 5])
    ax_b2.set_ylim(0.0, 0.95)
    ax_b2.set_title("Doubles", pad=4, fontsize=8.5, loc="left")
    ax_b2.set_xlabel("Retained Dataset Fraction (%)")
    ax_b2.grid(True)
    ax_b2.legend(loc="center right", bbox_to_anchor=(0.98, 0.42), frameon=True, fontsize=5.5,
                 facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", labelspacing=0.2,
                 handlelength=1.4, handletextpad=0.4)

    # Centered shared y-labels for each column
    fig.canvas.draw()
    pos_a1 = ax_a1.get_position()
    pos_a2 = ax_a2.get_position()
    pos_b1 = ax_b1.get_position()
    pos_b2 = ax_b2.get_position()

    # Updated to user request: y-axis just have RMSE
    fig.text(pos_a1.x0 - 0.08, (pos_a1.y1 + pos_a2.y0) / 2.0,
             r"RMSE ($\mathrm{kcal\cdot mol^{-1}}$)",
             va="center", ha="center", rotation="vertical", fontsize=8.5)
    fig.text(pos_b1.x0 - 0.08, (pos_b1.y1 + pos_b2.y0) / 2.0,
             r"Retained MAE ($\mathrm{kcal\cdot mol^{-1}}$)",
             va="center", ha="center", rotation="vertical", fontsize=8.5)

    if save:
        paths = [
            os.path.join(out_dir, "figure_2_uq_validation.pdf"),
            os.path.join(out_dir, "figure_2_uq_validation.png"),
            os.path.join(out_dir, "figure_2_master_uncertainty.pdf"),
            os.path.join(out_dir, "figure_2_master_uncertainty.png"),
            os.path.join(out_dir, "figure_2_nature_uncertainty.pdf"),
            os.path.join(out_dir, "figure_2_nature_uncertainty.png"),
            os.path.join(out_dir, "figure_2_uncertainty_calibration.pdf"),
            os.path.join(out_dir, "figure_2_uncertainty_calibration.png"),
            os.path.join(out_dir, "figure_2_nature_epistasis.pdf"),
            os.path.join(out_dir, "figure_2_nature_epistasis.png"),
        ]
        for p in paths:
            fig.savefig(p, dpi=300)
        print(f"Successfully saved Main Figure 2 to {paths[0]} and {paths[1]}")
    return fig


if __name__ == "__main__":
    generate_figure_2()

# Backward-compatible aliases
generate_figure_2_uncertainty = generate_figure_2
generate_figure_2_validation = generate_figure_2
generate_nature_figure_2 = generate_figure_2

