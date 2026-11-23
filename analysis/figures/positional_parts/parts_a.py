"""Panel a, matching David's original lollipop layout (viridis by value, value labels, mean/median lines, legend lower right).
EDL-only sigma_e; MegaScale singles, forward. Variant *_with_partial adds the effect-size-controlled partial rho (open grey)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "font.size": 7.5,
    "axes.linewidth": 0.7, "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none"})
pp = pd.read_csv("../data/positional_intra_megascale_v2.csv").sort_values("spearman").reset_index(drop=True)
mean, med = pp.spearman.mean(), pp.spearman.median()
cmap = mpl.colormaps["viridis"]; norm = mpl.colors.Normalize(0.1, 0.85)
for partial in (False, True):
    fig, ax = plt.subplots(figsize=(4.4, 4.9)); y = np.arange(len(pp))
    for yi, v, pv in zip(y, pp.spearman, pp.partial_spearman):
        c = "#D62728" if v < 0 else cmap(norm(v))
        ax.hlines(yi, 0, v, color=c, lw=1.2, zorder=1); ax.scatter(v, yi, s=26, color=c, zorder=3)
        xl = v + 0.018 if v >= 0 else (min(v, pv) if partial else v) - 0.025
        ax.text(xl, yi, f"{v:+.2f}".replace("-", "−"), va="center", ha="left" if v >= 0 else "right",
                fontsize=5.8, color="#D62728" if v < 0 else "#333333")
    h = [ax.axvline(mean, c="#2E8B57", ls="--", lw=0.9, label=f"Mean ρ = {mean:.2f}"),
         ax.axvline(med, c="#3B5BDB", ls=":", lw=1.0, label=f"Median ρ = {med:.2f}")]
    if partial:
        h.append(ax.scatter(pp.partial_spearman, y, s=16, facecolors="white", edgecolors="#8C8C8C", lw=0.9, zorder=2,
                            label=f"Partial ρ (mean {pp.partial_spearman.mean():.2f})"))
    ax.axvline(0, c="#333333", lw=0.6)
    ax.set_yticks(y, list(pp.pdb), fontsize=6); ax.set_ylim(-0.7, len(pp) - 0.3)
    ax.set_xlim(-0.25 if partial else -0.16, 0.95); ax.grid(axis="x", ls=":", alpha=0.45, lw=0.5)
    ax.set_xlabel(r"Spatial Spearman correlation $\rho(\bar{\sigma}_e$, MAE)")
    ax.set_title(f"Positional Uncertainty Tracks Prediction Error Across {len(pp)} PDBs", fontsize=8, pad=6, loc="left")
    ax.legend(handles=h, loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D0D0D0", fontsize=6.2)
    name = "parts_a_per_protein" + ("_with_partial" if partial else "")
    for ext in ("png", "svg", "pdf"): fig.savefig(f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, transparent=(ext != "pdf"))
    plt.close(fig)
print(round(mean, 3), round(med, 3))
