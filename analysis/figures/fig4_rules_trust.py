"""Fig. 4 - what the model learned (mu) and where it trusts it (sigma). Singles, mutation IG, ensemble means.
Inputs: results/attribution deep-dive tables (no new analysis)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import ATTRIBUTION_OUT, AUDITS_OUT
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import paper_style as S
S.setup()
A3 = str(ATTRIBUTION_OUT) + "/"
rc = pd.read_csv(A3 + "deepdive_uncertainty_attribution/tables/response_curves_sigma.csv"); rc = rc[rc.subset == "singles"]
pk = pd.read_csv(A3 + "deepdive_uncertainty_attribution/tables/packing_curves_mu_sigma.csv")
kn = json.load(open(A3 + "deepdive_uncertainty_attribution/key_numbers.json"))["B_where_vs_what"]
cm = pd.read_csv(A3 + "deepdive_attribution_mechanisms/class_mean_attr_mu.csv").set_index("cls")
cs = pd.read_csv(A3 + "deepdive_attribution_mechanisms/class_mean_attr_log_sigma_epi.csv").set_index("cls")
csum = pd.read_csv(A3 + "deepdive_attribution_mechanisms/class_summary.csv").set_index("cls")
BUR = [("core", S.CORE), ("boundary", S.BOUNDARY), ("surface", S.SURFACE)]
rules = [("Hydrophobicity", "Δ hydrophobicity (KD)", "Buried hydrophobic loss"),
         ("pack", "Δ volume (Å³)", "Packing (Volume + clashes + VdW)"),
         ("Polar Solvation", "Δ polar solvation", "Burying polar groups"),
         ("Electrostatics", "Δ electrostatic energy", "Electrostatics")]
fig = plt.figure(figsize=(S.FULL_W, 6.1))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.42)
for j, (feat, xl, title) in enumerate(rules):
    for i, (col, ylab) in enumerate([("mu", "attribution to μ\n(kcal mol$^{-1}$)"), ("sig", "attribution to log σ")]):
        ax = fig.add_subplot(gs[i, j])
        for b, c in BUR:
            if feat == "pack":
                g = pk[pk.bur == b].sort_values("dx"); y = g[f"{col}_pack"]
            else:
                g = rc[(rc.bur == b) & (rc.feature == feat)].sort_values("dx"); y = g["mu_attr" if col == "mu" else "sig_attr"]
            ax.plot(g.dx, y, "-o", c=c, ms=2.2, lw=1.1, label=b)
        ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c=S.LIGHT, lw=0.5, zorder=0)
        if i == 0: ax.set_title(title, fontsize=6.8, pad=3)
        if j == 0: ax.set_ylabel(ylab); S.panel(ax, "AB"[i], -0.42, 1.08)
        if i == 1: ax.set_xlabel(xl)
        if i == 0 and j == 0: ax.legend(loc="upper right", handlelength=1.2)
        if i == 1: ax.set_ylim(-0.32, 0.6)
# C where vs what
sub = gs[2, :].subgridspec(1, 4, width_ratios=[0.55, 1.05, 2.3, 0.85], wspace=0.12)
ax = fig.add_subplot(sub[0]); S.panel(ax, "C", -0.75, 1.04)
sb = [kn["singles"]["share_var_g_site_baseline"], kn["doubles"]["share_var_g_site_baseline"]]
ax.bar([0, 1], sb, 0.6, color=S.GREY, label="site (burial)")
ax.bar([0, 1], [1 - x for x in sb], 0.6, bottom=sb, color=S.SIGMA, label="substitution")
for x, v in zip([0, 1], sb): ax.text(x, v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=6.5); ax.text(x, v + (1 - v) / 2, f"{1-v:.0%}", ha="center", va="center", color="white", fontsize=6.5)
ax.set(xticks=[0, 1], xticklabels=["singles", "doubles"], ylabel="share of var(log σ)", ylim=(0, 1))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, handlelength=1)
# D class signatures mu | sigma
order = ["to Pro", "hydrophobic→polar", "charge introduced", "to Gly", "from Gly", "hydrophobic→hydrophobic", "charge reversal", "charge removed"]
order = [o for o in order if o in cm.index]
feats = ["Hydrophobicity", "Polar Solvation", "Volume", "VdW Clashes", "VdW Energy", "Electrostatics", "Hydrogen Bonds", "Cyclic", "Backbone Entropy"]
short = ["Hydrophob.", "Polar solv.", "Volume", "Clashes", "VdW en.", "Electrost.", "H-bonds", "Cyclic", "BB entropy"]
axD = fig.add_subplot(sub[2]); axE = fig.add_subplot(sub[3])
X = cm.loc[order, feats].values; Y = cs.loc[order, feats].values
im = axD.imshow(X, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
axD.set_yticks(range(len(order)), [f"{o} ({int(csum.loc[o,'n']):,})" for o in order], fontsize=5.8); axD.set_xticks(range(len(feats)), short, rotation=45, ha="right", fontsize=5.8)
for (r, c), v in np.ndenumerate(X):
    if abs(v) >= 0.1: axD.text(c, r, f"{v:+.2f}".replace("+", ""), ha="center", va="center", fontsize=4.8, color="white" if abs(v) > 0.4 else "k")
cax = axD.inset_axes([0.62, 1.03, 0.38, 0.035]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cax.xaxis.set_ticks_position("top"); cb.set_label("attribution to μ (kcal mol$^{-1}$)", fontsize=5.5, labelpad=2); cax.xaxis.set_label_position("top"); cb.ax.tick_params(labelsize=5)
S.panel(axD, "D", -0.5, 1.04)
# E: uncertainty added by the substitution itself = class mean of g - g0 (all 23 features; completeness), with 95% protein-bootstrap CI.
#    Source: audits/verify_attribution_section/results_part_b.json -> class_sigma (2026-10-02; replaces the 12-feature sum).
cls_sig = json.load(open(AUDITS_OUT / "verify_attribution_section" / "results_part_b.json"))["class_sigma"]
tot = pd.Series({o: cls_sig[o]["mean_gm_all23"] for o in order}); lo = np.array([cls_sig[o]["mean_gm_ci"][0] for o in order]); hi = np.array([cls_sig[o]["mean_gm_ci"][1] for o in order])
axE.barh(range(len(order)), tot.values, color=[S.SIGMA if v > 0 else S.LIGHT for v in tot.values], height=0.65)
axE.errorbar(tot.values, range(len(order)), xerr=[tot.values - lo, hi - tot.values], fmt="none", ecolor="k", elinewidth=0.5, capsize=1.2)
for r, o in enumerate(order):
    top = cs.loc[o, feats].astype(float).idxmax(); axE.text(max(hi[r], 0) + 0.02, r, short[feats.index(top)], va="center", fontsize=5.2, color=S.GREY)
axE.axvline(0, c="k", lw=0.5); axE.set_yticks(range(len(order)), []); axE.set(xlabel="added log σ (mutation part)")
axE.set_ylim(len(order) - 0.5, -0.5); S.panel(axE, "E", -0.05, 1.04); axE.tick_params(axis="y", length=0); print(tot.round(2).to_dict())
S.save(fig, "fig4_rules_trust"); print("ok", sb)
