"""Fig. S4 - selective prediction within bins of predicted ddG. Style matches figure_2_uncertainty_calibration
(generate_nature_figures_modular.ipynb). Inputs: results/audits/binned_selective_prediction/ (no new analysis)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import AUDITS_OUT
import ast, numpy as np, pandas as pd, matplotlib.pyplot as plt
A = str(AUDITS_OUT / "binned_selective_prediction") + "/"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 7.5, "legend.fontsize": 6.0, "lines.linewidth": 1.3, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
    "grid.alpha": 0.45, "grid.linestyle": ":", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"singles": ("#2B5C8F", "#5D9ECE"), "doubles": ("#7B4397", "#BA68C8")}
C_PRED, C_RAND, C_OR = "#C0762B", "#7F8C8D", "#555555"
LEG = dict(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", labelspacing=0.2, fontsize=5.6)
cc = pd.read_csv(A + "coverage_curves.csv"); pb = pd.read_csv(A + "per_bin.csv")
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.5))
for r, (setn, mk) in enumerate([("singles", "o"), ("doubles", "s")]):
    deep, light = COL[setn]; c = cc[cc.set == setn]; get = lambda rule, col: c[c.rule == rule].sort_values("coverage")
    lab = setn.capitalize()
    # MAE vs coverage
    ax = axs[r, 0]
    for rule, kw in [("oracle", dict(ls=":", c=C_OR, lw=1.1, label="Oracle")), ("random", dict(ls="--", c=C_RAND, lw=0.9, label="Random")),
                     ("sigma_within_bin", dict(ls="--", c=deep, lw=1.3, marker=mk, ms=3.4, markevery=3, mfc="white", mew=1.0, label=r"$\sigma_e$, within bins")),
                     ("sigma_global", dict(ls="-", c=deep, lw=1.4, marker=mk, ms=3.2, markevery=3, label=r"$\sigma_e$, global"))]:
        g = get(rule, "mae"); ax.plot(g.coverage * 100, g.mae, **kw)
    ax.set_xlim(100, 10); ax.grid(True); ax.set_ylabel(r"Retained MAE (kcal$\cdot$mol$^{-1}$)"); ax.set_ylim(0, None)
    ax.set_title(lab, pad=4, fontsize=8.5, loc="left")
    # correlation vs coverage
    ax = axs[r, 1]
    for rule, kw in [("sigma_within_bin", dict(ls="--", c=deep, lw=1.3, marker=mk, ms=3.4, markevery=3, mfc="white", mew=1.0, label=r"$\sigma_e$, within bins")),
                     ("sigma_global", dict(ls="-", c=deep, lw=1.4, marker=mk, ms=3.2, markevery=3, label=r"$\sigma_e$, global"))]:
        g = get(rule, "pearson"); ax.plot(g.coverage * 100, g.pearson, **kw)
    full = get("random", "pearson").pearson.iloc[-1] if "pearson" in c else None
    ax.set_xlim(100, 10); ax.grid(True); ax.set_ylabel("Pearson $r$ of retained set"); ax.set_ylim(0, 1)
    ax.set_title(lab, pad=4, fontsize=8.5, loc="left")
    # per-bin reduction at 50% within bins
    ax = axs[r, 2]; p = pb[pb.set == setn].reset_index(drop=True); x = np.arange(len(p)); w = 0.6
    for off, col, ci, colr, name in [(0, "reldrop50_sigma", "reldrop50_sigma_ci", deep, r"$\sigma_e$, within bins")]:
        v = p[col].values * 100; lo_hi = np.array([ast.literal_eval(s) for s in p[ci]]) * 100
        ax.bar(x + off, v, w, color=colr, edgecolor="black", lw=0.4, label=name)
        ax.errorbar(x + off, v, yerr=[v - lo_hi[:, 0], lo_hi[:, 1] - v], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
    ax.axhline(0, c="black", lw=0.6); ax.grid(True, axis="y")
    names = {"<=-0.5": "≤ −0.5", "-0.5..0.5": "±0.5", "0.5..1.5": "0.5–1.5", "1.5..3": "1.5–3", ">3": "> 3", "<=0.5": "≤ 0.5"}
    ax.set_xticks(x, [names.get(b, b) + f"\n{n:,}" for b, n in zip(p.bin, p.n)], fontsize=5.6)
    ax.set_ylabel("MAE reduction at 50% retained (%)"); ax.set_title(lab, pad=4, fontsize=8.5, loc="left")
    if r == 1:
        axs[r, 0].set_xlabel("Retained Dataset Fraction (%)"); axs[r, 1].set_xlabel("Retained Dataset Fraction (%)")
        ax.set_xlabel(r"Predicted $\Delta\Delta G$ bin (kcal$\cdot$mol$^{-1}$)")
for j, (letter, title) in enumerate([("a", "Error rejection"), ("b", "Ranking of retained set"), ("c", "Within-bin filtering per bin")]):
    ax = axs[0, j]; ax.text(-0.25, 1.24, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
    ax.text(-0.08, 1.24, title, transform=ax.transAxes, fontsize=9, va="top", ha="left")
from matplotlib.lines import Line2D
H = [Line2D([], [], c="#333333", ls="-", marker="o", ms=3.6, lw=1.4, label=r"$\sigma_e$, global"),
     Line2D([], [], c="#333333", ls="--", marker="o", ms=3.6, mfc="white", mew=1.0, lw=1.3, label=r"$\sigma_e$, within bins"),
     Line2D([], [], c=C_RAND, ls="--", lw=0.9, label="Random"), Line2D([], [], c=C_OR, ls=":", lw=1.1, label="Oracle")]
fig.legend(handles=H, loc="lower center", ncol=4, bbox_to_anchor=(0.40, -0.035), **{k: v for k, v in LEG.items() if k != "fontsize"}, fontsize=6.2, handlelength=3.6, markerscale=1.0)
for ext in ("pdf", "png"): fig.savefig(f"figS4_selective_prediction.{ext}", bbox_inches="tight", pad_inches=0.05)
print("ok")
