"""Manuscript figure: learned epistasis (2x2), style matched to figure_1_nature_performance / figure_2_uncertainty_calibration.
Numbers: run fig_epistasis_3panel.py first (writes megascale_test/tables/epistasis_fig_key_numbers.json + per-protein table).
Output: results/figures/figure_epistasis_nature.{pdf,png}
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde, spearmanr
import config as cfg

R = cfg.OUT_DIR / "megascale_test"
K = json.load(open(R / "tables/epistasis_fig_key_numbers.json"))
P = pd.read_csv(R / "tables/epistasis_fig_per_protein_ranking.csv")
d = pd.read_csv(R / "tables/epistasis_per_variant.csv").dropna(subset=["E_exp", "E_pred"])
from paths import FIGURES_OUT
OUT = FIGURES_OUT

plt.style.use("seaborn-v0_8-ticks" if "seaborn-v0_8-ticks" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3,
    "grid.linewidth": 0.4, "grid.alpha": 0.5, "grid.linestyle": ":", "grid.color": "#B0B0B0",
    "figure.dpi": 300, "savefig.dpi": 300, "mathtext.fontset": "custom", "mathtext.rm": "sans",
    "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold", "pdf.fonttype": 42, "ps.fonttype": 42,
})
PLUM, ORCHID, GREY, ORANGE, INK, EXPC = "#7B4397", "#BA68C8", "#A7ADB4", "#D9822B", "#2B2B2B", "#4A5568"
UNIT = r"(kcal mol$^{-1}$)"
LEG = dict(frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D5D5", labelspacing=0.3,
           handlelength=1.0, handletextpad=0.5, borderpad=0.45, borderaxespad=0.5)

def header(ax, letter, title):
    ax.text(-0.25, 1.045, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    ax.text(0.0, 1.045, title, transform=ax.transAxes, fontsize=8.3, va="bottom", ha="left")

def statbox(ax, lines, x=0.04, y=0.96, ha="left"):
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, va="top", ha=ha, fontsize=6.6, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D5D5D5", lw=0.6, alpha=0.95), zorder=10)

def density_panel(ax, x, y, lim_x, lim_y, col, light):
    ax.scatter(x, y, s=1.6, color=col, alpha=0.10, lw=0, rasterized=True, zorder=1)
    xx, yy = np.mgrid[lim_x[0]:lim_x[1]:200j, lim_y[0]:lim_y[1]:200j]
    zz = gaussian_kde(np.vstack([x, y]), bw_method=0.2)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    lv = np.quantile(zz[zz > zz.max() * 0.03], [0.25, 0.55, 0.8, 0.94])
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("c", [light, col])
    ax.contour(xx, yy, zz, levels=lv, colors="white", linewidths=1.5, alpha=0.95, zorder=2)
    ax.contour(xx, yy, zz, levels=lv, cmap=cm, linewidths=0.75, zorder=3)
    ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)

fig = plt.figure(figsize=(5.9, 5.7))
gs = fig.add_gridspec(2, 2, wspace=0.40, hspace=0.46, left=0.1, right=0.98, top=0.93, bottom=0.09)
ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))
for ax in (ax0, ax1, ax2, ax3):
    ax.set_box_aspect(1); ax.grid(True); ax.set_axisbelow(True)

# ---- a: experimental additivity
density_panel(ax0, d.add_exp.values, d.true_ddg.values, (-2.5, 9.0), (-2.5, 7.0), EXPC, "#D5DAE1")
ax0.plot([-2.5, 7.0], [-2.5, 7.0], ls="--", color=INK, lw=0.8, zorder=4)
ax0.set_xticks(range(-2, 10, 2)); ax0.set_yticks(range(-2, 8, 2))
ax0.set_xlabel(r"$\Delta\Delta G_{A,\mathrm{exp}} + \Delta\Delta G_{B,\mathrm{exp}}$ " + UNIT); ax0.set_ylabel(r"$\Delta\Delta G_{AB,\mathrm{exp}}$ " + UNIT)
statbox(ax0, [rf"$\rho = {spearmanr(d.true_ddg, d.add_exp)[0]:.2f}$"])
header(ax0, "a", "Experimental additivity")

# ---- b: learned non-additivity
density_panel(ax1, d.E_exp.values, d.E_pred.values, (-5.0, 2.5), (-4.0, 2.5), PLUM, "#E4D2EE")
ax1.axhline(0, color="#8A8F96", lw=0.5, zorder=0); ax1.axvline(0, color="#8A8F96", lw=0.5, zorder=0)
ax1.plot([-4, 2.5], [-4, 2.5], ls="--", color=INK, lw=0.8, zorder=4)
ax1.set_xticks(range(-4, 3, 2)); ax1.set_yticks(range(-4, 3, 2))
ax1.set_xlabel(r"$E_\mathrm{exp}$ " + UNIT); ax1.set_ylabel(r"$E_\mathrm{calc}$ " + UNIT)
statbox(ax1, [rf"$\rho = {K['rho_all']:.2f}$", rf"$n = {K['n']:,}$".replace(",", "{,}")])
header(ax1, "b", "Learned non-additivity")

# ---- c: distance dependence
bins = list(K["bins"].keys()); x = np.arange(len(bins)); off = 0.15
ax2.axvspan(-0.5, 0.5, color="#F3EAF7", zorder=0, lw=0)
y = np.array([K["bins"][bb]["rho"] for bb in bins]); ci = np.array([K["bins"][bb]["rho_ci"] for bb in bins])
ax2.vlines(x, ci[:, 0], ci[:, 1], color=PLUM, lw=1.4, zorder=2)
ax2.plot(x, y, "o", color=PLUM, ms=5.5, mec="white", mew=0.8, zorder=3)
ax2.axhline(0, color=INK, lw=0.6, zorder=1)
ax2.set_xticks(x); ax2.set_xticklabels([f"{b.replace(' Å', '')} Å\n$n$ = {K['bins'][b]['n']:,}" for b in bins])
ax2.set_xlim(-0.5, 2.5); ax2.set_ylim(-0.1, 0.8); ax2.set_yticks(np.arange(0, 0.81, 0.2))
ax2.grid(False, axis="x")
ax2.set_xlabel(r"C$\alpha$–C$\alpha$ distance"); ax2.set_ylabel(r"$\rho$, $E_\mathrm{calc}$ vs. $E_\mathrm{exp}$")
header(ax2, "c", "Distance dependence")

# ---- d: ranking double mutants
ax3.plot([0, 1], [0, 1], ls="--", color=INK, lw=0.8, zorder=1)
up = (P.full > P.addsum).values
ax3.scatter(P.addsum[up], P.full[up], s=18, color=PLUM, edgecolor="white", lw=0.6, zorder=3)
ax3.scatter(P.addsum[~up], P.full[~up], s=18, color=GREY, edgecolor="white", lw=0.6, zorder=3)
kf, ka = K["rank_full"], K["rank_add_pred"]; cf, ca = K["rank_full_ci"], K["rank_add_pred_ci"]
ax3.set_xlim(-0.03, 1.0); ax3.set_ylim(-0.03, 1.0)
ax3.set_xticks(np.arange(0, 1.01, 0.2)); ax3.set_yticks(np.arange(0, 1.01, 0.2))
ax3.set_xlabel(r"$\rho$, sum of predicted singles"); ax3.set_ylabel(r"$\rho$, double-mutant prediction")
statbox(ax3, [f"{K['per_protein_full_gt_add']}/{K['n_proteins']} proteins above $y = x$",
              rf"All doubles: ${kf:.2f}$ vs. ${ka:.2f}$"])
ax3.legend(handles=[Line2D([], [], marker="o", ls="", color=PLUM, mec="white", ms=5, label="Protein, model better"),
                    Line2D([], [], marker="o", ls="", color=GREY, mec="white", ms=5, label="Protein, additive ≥ model")],
           loc="lower right", **LEG)
header(ax3, "d", "Ranking double mutants")

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_epistasis_nature.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved", OUT / "figure_epistasis_nature.png")
