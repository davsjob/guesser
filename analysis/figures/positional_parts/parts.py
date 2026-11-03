"""Separate panels of the positional figure for assembly in PowerPoint (PNG 600 dpi + SVG + PDF)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from scipy.stats import pearsonr, spearmanr
D = "../data/"  # run from results/figures/positional_parts
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5, "lines.linewidth": 1.2, "axes.linewidth": 0.7, "grid.linewidth": 0.5, "grid.alpha": 0.45, "grid.linestyle": ":",
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
C_SIG, C_ERR, C_PAR = "#B2182B", "#2B5C8F", "#8C8C8C"
LEG = dict(frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", labelspacing=0.25, fontsize=6.2)
def save(fig, name):
    for ext in ("png", "svg", "pdf"): fig.savefig(f"parts_{name}.{ext}", bbox_inches="tight", pad_inches=0.03, transparent=(ext != "pdf"))
    plt.close(fig)
pp = pd.read_csv(D + "positional_intra_megascale_v2.csv").sort_values("spearman").reset_index(drop=True)
k = pd.read_csv(D + "positional_2k5h_intra.csv")
# panel a: see parts_a.py
# b colorbar
fig = plt.figure(figsize=(0.55, 1.8)); cax = fig.add_axes([0.1, 0.05, 0.25, 0.9])
cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colors.LinearSegmentedColormap.from_list("bwr_pymol", ["#0000FF", "#FFFFFF", "#FF0000"]))
cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=6, length=0); cb.set_label("relative value (0 to 1)", fontsize=6)
save(fig, "b_colorbar")
# c (style of the original figure: red uncertainty, blue MAE, shaded gap; display = 3-residue rolling mean, stats on raw values)
fig, ax2 = plt.subplots(figsize=(4.4, 2.3)); mm = lambda x: (x - x.min()) / (x.max() - x.min())
sm = lambda x: x.rolling(3, center=True, min_periods=1).mean()
ys, ye = mm(sm(k.sigma)), mm(sm(k.mae))
ax2.fill_between(k.pos, ys, ye, color="#E4E0F5", lw=0, zorder=0)
ax2.plot(k.pos, ys, "-o", c="#C0272D", ms=2.6, lw=1.2, label=r"Positional epistemic uncertainty $\bar{\sigma}_e$")
ax2.plot(k.pos, ye, "-s", c="#3B5BDB", ms=2.4, lw=1.2, label="Prediction error MAE")
r, rho = pearsonr(k.sigma, k.mae)[0], spearmanr(k.sigma, k.mae)[0]
ax2.text(0.02, 0.96, f"Pearson r = {r:+.2f}\nSpearman ρ = {rho:+.2f}", transform=ax2.transAxes, fontsize=6.2, va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D0D0D0", lw=0.6))
ax2.set_title("2K5H", fontsize=8.5, fontweight="bold", loc="left", pad=4)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax2.set(xlim=(0.5, k.pos.max() + 0.5), ylim=(-0.03, 1.27), xlabel="Sequence position", ylabel="Relative magnitude (0 to 1)")
ax2.grid(True); ax2.legend(loc="upper right", **{**LEG, "fontsize": 5.8})
save(fig, "c_trace")
print("ok")
