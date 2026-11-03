"""figure_selection (bar version): wrong picks with and without sigma_e at the two settings used in the text
(neutral: 20% highest-ranked calls; stabilizing: top 5 per protein). Paired differences and their CIs are given in the caption
(rule == "difference" rows of figure_selection_numbers.csv). Input: figure_selection_numbers.csv."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
IN, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 7.5, "axes.linewidth": 0.7, "grid.linewidth": 0.5, "grid.alpha": 0.45, "grid.linestyle": ":",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
C_MU, C_SIG = "#B9BEC4", "#2B5C8F"
r = pd.read_csv(IN)
a = r[(r.panel == "a") & (r.x == 0.2)]; b = r[(r.panel == "b") & (r.metric == "destab") & (r.x == 5)]
PANELS = [("a", "Selection of predicted neutral variants", "Top 20% of variants predicted neutral (|μ| < 0.5)",
           "not neutral in experiment", a, "smallest |μ|", "P(neutral)"),
          ("b", "Selection of predicted stabilizing variants", "Top 5 candidates per protein for stabilization",
           r"destabilizing in experiment ($\Delta\Delta G$ ≥ 0.5)", b, "lowest μ", "P(ΔΔG < 0)")]
LMU, LSIG = "Ranked by the prediction (μ) only", r"Ranked using the uncertainty ($\sigma_e$)"
pts = lambda v: f"{v * 100:+.0f}".replace("-", "−")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw=dict(wspace=0.5))
for ax, (letter, title, sub, ylab, d, rmu, rsig) in zip(axs, PANELS):
    sets = ["MS-S", "FireProt-HF"]; x = np.arange(2); w = 0.36
    g0, g1 = (d[d.rule == q].set_index("set").loc[sets] for q in (rmu, rsig))
    for off, g, col, lab in [(-w / 2, g0, C_MU, LMU), (w / 2, g1, C_SIG, LSIG)]:
        v = g.value.values * 100
        ax.bar(x + off, v, w, color=col, edgecolor="black", lw=0.4, label=lab, zorder=2)
        for xi, vi in zip(x + off, v): ax.text(xi, vi + 0.8, f"{vi:.0f}%", ha="center", va="bottom", fontsize=6.6, color="#222222")
    ax.set_xticks(x, ["MegaScale-S", "FireProt-HF"]); ax.set_ylabel("Wrong selections (%)", labelpad=16); ax.text(-0.155, 0.5, ylab, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.2, color="#555555"); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, 57); ax.set_yticks(np.arange(0, 51, 10))   # same scale in both panels
    ax.text(0.02, 0.98, sub, transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color="#555555")
    ax.text(-0.2, 1.10, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
    ax.text(-0.06, 1.10, title, transform=ax.transAxes, fontsize=9, va="top", ha="left")
axs[0].legend(loc="upper center", bbox_to_anchor=(1.12, -0.13), ncol=2, frameon=False, fontsize=7.0, handlelength=1.4)
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/figure_selection.{ext}", bbox_inches="tight", pad_inches=0.05)
print(r[r.rule == "difference"].query("(panel=='a' and x==0.2) or (panel=='b' and x==5 and metric=='destab')")[["panel", "set", "value", "lo", "hi"]].round(4).values.tolist())
