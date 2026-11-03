"""figure_selection: using sigma_e to choose which predictions to act on. Style matches figure_selective_prediction
(figS4_selective_prediction.py). Input: figure_selection_numbers.csv from fig_selection_data.py."""
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
IN, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 7.5, "legend.fontsize": 6.0, "lines.linewidth": 1.3, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
    "grid.alpha": 0.45, "grid.linestyle": ":", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"MS-S": "#2B5C8F", "FireProt-HF": "#4DAF4A"}; MK = {"MS-S": "o", "FireProt-HF": "^"}
LAB = {"MS-S": "MegaScale-S", "FireProt-HF": "FireProt-HF"}; C_BASE = "#7F8C8D"
LEG = dict(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", labelspacing=0.25, fontsize=6.0)
r = pd.read_csv(IN)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw=dict(hspace=0.48, wspace=0.32))
for i, s in enumerate(["MS-S", "FireProt-HF"]):
    col, mk = COL[s], MK[s]
    # a: predicted-neutral calls
    ax = axs[i, 0]; a = r[(r.panel == "a") & (r.set == s)]
    ax.axhline(a.all_called.iloc[0] * 100, c="#555555", ls=":", lw=1.0)
    for rule, kw in [("smallest |μ|", dict(c=C_BASE, ls="--", mfc="white", mec=C_BASE)), ("P(neutral)", dict(c=col, ls="-", mfc=col, mec=col))]:
        g = a[a.rule == rule].sort_values("x")
        ax.fill_between(g.x * 100, g.lo * 100, g.hi * 100, color=kw["c"], alpha=0.15, lw=0)
        ax.plot(g.x * 100, g.value * 100, marker=mk, ms=3.6, mew=1.0, lw=1.4, **kw)
    ax.set_xlim(8, 52); ax.set_xticks([10, 20, 30, 40, 50]); ax.set_ylim(0, None); ax.grid(True)
    ax.set_ylabel("Not neutral in experiment (%)"); ax.set_title(f"{LAB[s]} (n = {int(a.n.iloc[0]):,} predicted neutral)", pad=4, fontsize=8.0, loc="left")
    # b: predicted-stabilizing picks per protein
    ax = axs[i, 1]; b = r[(r.panel == "b") & (r.set == s) & (r.metric == "destab")]
    ks = sorted(b.x.unique()); xp = np.arange(len(ks))
    for off, (rule, kw) in zip((-0.12, 0.12), [("lowest μ", dict(c=C_BASE, ls="--", mfc="white", mec=C_BASE)), ("P(ΔΔG < 0)", dict(c=col, ls="-", mfc=col, mec=col))]):
        g = b[b.rule == rule].set_index("x").loc[ks]
        ax.errorbar(xp + off, g.value * 100, yerr=[(g.value - g.lo) * 100, (g.hi - g.value) * 100], fmt="none", ecolor=kw["c"], elinewidth=0.8, capsize=1.8)
        ax.plot(xp + off, g.value * 100, marker=mk, ms=3.8, mew=1.0, lw=1.2, **kw)
    nu = b[b.rule == "lowest μ"].set_index("x").loc[ks].n_units.astype(int)
    ax.set_xticks(xp, [f"{int(k)}\n{n}" for k, n in zip(ks, nu)], fontsize=6.3)
    ax.set_xlim(-0.5, len(ks) - 0.5); ax.set_ylim(0, None); ax.grid(True)
    ax.set_ylabel("Destabilizing picks (%)"); ax.set_title(LAB[s], pad=4, fontsize=8.0, loc="left")
    ax.text(-0.02, -0.075, "k\nproteins", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color="#555555")
axs[1, 0].set_xlabel("Predicted-neutral calls kept (%)"); axs[1, 1].set_xlabel("Picks per protein (top k)", labelpad=2)
for j, (letter, title) in enumerate([("a", "Selecting stability-neutral variants"), ("b", "Choosing predicted-stabilizing mutations to test")]):
    ax = axs[0, j]; ax.text(-0.22, 1.30, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
    ax.text(-0.08, 1.30, title, transform=ax.transAxes, fontsize=9, va="top", ha="left")
H = [Line2D([], [], c="#333333", ls="-", marker="o", ms=3.6, lw=1.4, label=r"Ranked by $\sigma_e$-based probability: P(neutral) in a, P($\Delta\Delta G$ < 0) in b"),
     Line2D([], [], c=C_BASE, ls="--", marker="o", ms=3.6, mfc="white", lw=1.4, label=r"Ranked by $\mu$ alone: smallest |$\mu$| in a, lowest $\mu$ in b"),
     Line2D([], [], c="#555555", ls=":", lw=1.0, label="All predicted-neutral calls")]
fig.legend(handles=H, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), **{k: v for k, v in LEG.items() if k != "fontsize"}, fontsize=6.2, handlelength=3.0)
for ext in ("pdf", "png"): fig.savefig(f"{OUT}/figure_selection.{ext}", bbox_inches="tight", pad_inches=0.05)
print("ok")
