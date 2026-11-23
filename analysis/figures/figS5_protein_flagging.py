"""Fig. S5 - protein-level flagging by EDL epistemic uncertainty (sigma_e = sigma_intra), proteins with >= 15 variants,
forward mutations. Style matches figS4_selective_prediction.py. Input: deepdive_protein_flagging/per_protein_N15.csv.
v2 (2026-10-01): 3 panels, common axes, no tercile bars / pooled panel (pooled rho reported in text). v1 in _to_delete/."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import ATTRIBUTION_OUT
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr, rankdata
A3 = str(ATTRIBUTION_OUT) + "/"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 7.5, "lines.linewidth": 1.3, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
    "grid.alpha": 0.45, "grid.linestyle": ":", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
U = "sigma_intra"
SETS = [("MegaScale test singles", "MegaScale-S", "#2B5C8F", "o"), ("FireProt-HF singles", "FireProt-HF", "#4DAF4A", "^"),
        ("MegaScale test doubles", "MegaScale-D", "#7B4397", "s")]
pp = pd.read_csv(A3 + "deepdive_protein_flagging/per_protein_N15.csv")

def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); Z = np.column_stack([rz, np.ones(len(x))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)[0]

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True, gridspec_kw=dict(wspace=0.12))
for j, (sname, lab, col, mk) in enumerate(SETS):
    ax = axs[j]; g = pp[pp.set == sname]; x, y = g[U].values, g.r.values
    r, p = pearsonr(x, y); pr = partial(x, y, g.true_sd.values)
    ax.scatter(x, y, s=20, c=col, marker=mk, edgecolor="white", lw=0.5, zorder=3)
    b = np.polyfit(x, y, 1); xx = np.array([x.min(), x.max()]); ax.plot(xx, np.polyval(b, xx), c=col, lw=1.1, ls="--", zorder=2)
    ax.text(0.96, 0.96 if j == 1 else 0.05, ((f"r = {r:.2f}, p = {p:.3f}" if p >= 0.001 else f"r = {r:.2f}, p < 0.001") + f"\n{len(g)} proteins").replace("-", "−"),
            transform=ax.transAxes, fontsize=6.2, va="top" if j == 1 else "bottom", ha="right", color="#333333")
    ax.set_xlim(0.15, 1.0); ax.set_xticks(np.arange(0.2, 1.01, 0.2)); ax.set_ylim(-0.2, 1.0); ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax.grid(True); ax.set_title(lab, pad=4, fontsize=8.5, loc="left")
    ax.set_xlabel(r"Mean $\sigma_e$ per protein (kcal$\cdot$mol$^{-1}$)", fontsize=7)
    ax.text(-0.06 if j else -0.2, 1.13, "ABC"[j], transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="right")
axs[0].set_ylabel("Per-protein Pearson $r$")
for ext in ("pdf", "png"): fig.savefig(f"figS5_protein_flagging.{ext}", bbox_inches="tight", pad_inches=0.05)
print("ok"); [print(lab, pearsonr(pp[pp.set == s][U], pp[pp.set == s].r)) for s, lab, _, _ in SETS]
