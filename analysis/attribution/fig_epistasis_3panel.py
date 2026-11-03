"""Manuscript figure: learned epistasis (3 panels).
A  E_pred vs E_exp hexbin (16,923 matched MegaScale doubles)
B  Spearman(E_pred, E_exp) by Calpha distance bin, raw and partial (| experimental additive sum); 95% CI = site-pair bootstrap
C  Ranking of double-mutant ddG: full model vs sum of the model's own single predictions, per protein (+ pooled)
"""
import ast, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import config as cfg
import figures  # noqa: F401  (applies the shared rcParams)
from figures import _panel_label, _save
C = cfg.C
R = cfg.OUT_DIR
d = pd.read_csv(R / "megascale_test/tables/epistasis_per_variant.csv").dropna(subset=["E_exp", "E_pred"]).reset_index(drop=True)
d["pair"] = [p + ":" + "-".join(f"{c}{n}" for c, n in sorted((c, n) for c, _, n, _ in ast.literal_eval(m)))
             for p, m in zip(d.pdb, d.mutation)]
BINS = ["< 6 Å", "6–9 Å", "> 9 Å"]
d["bin"] = pd.cut(d.d12, [0, 6, 9, np.inf], right=False, labels=BINS).astype(str)
rng = np.random.default_rng(0)
B = 2000

def rk(a): return pd.Series(a).rank().values
def partial(x, y, *zs):
    X, Y = rk(x), rk(y); A = np.column_stack([np.ones(len(X))] + [rk(z) for z in zs])
    ex = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]; ey = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]
def boot(g, fn):
    pairs = g.pair.unique(); idx = {p: np.where(g.pair.values == p)[0] for p in pairs}
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[p] for p in rng.choice(pairs, len(pairs))]); vals.append(fn(g.iloc[s]))
    return np.nanpercentile(vals, [2.5, 97.5])
rho = lambda g: spearmanr(g.E_exp, g.E_pred)[0]
prho = lambda g: partial(g.E_exp, g.E_pred, g.add_exp)
prho_pred = lambda g: partial(g.E_exp, g.E_pred, g.add_pred)           # controls for the model's own predicted singles sum only
prho2 = lambda g: partial(g.E_exp, g.E_pred, g.add_exp, g.add_pred)   # controls for experimental AND predicted singles sum
base = lambda g: spearmanr(g.E_exp, -g.add_pred)[0]                     # singles-only baseline (model's own singles, no pair info)

K = {"partial_pred_all": partial(d.E_exp, d.E_pred, d.add_pred), "partial_both_all": partial(d.E_exp, d.E_pred, d.add_exp, d.add_pred), "baseline_all": spearmanr(d.E_exp, -d.add_pred)[0], "n": len(d), "n_site_pairs": int(d.pair.nunique()), "rho_all": rho(d), "rho_all_ci": list(boot(d, rho)),
     "rho_E_exp_vs_add_exp": spearmanr(d.E_exp, d.add_exp)[0], "bins": {}}
for b in BINS:
    g = d[d.bin == b]
    K["bins"][b] = dict(n=len(g), n_pairs=int(g.pair.nunique()), rho=rho(g), rho_ci=list(boot(g, rho)),
                        partial=prho(g), partial_ci=list(boot(g, prho)),
                        partial_both=prho2(g), partial_both_ci=list(boot(g, prho2)),
                        baseline=base(g), baseline_ci=list(boot(g, base)),
                        partial_pred=prho_pred(g), partial_pred_ci=list(boot(g, prho_pred)),
                        rho_minus_baseline_ci=list(boot(g, lambda x: rho(x) - base(x))))
full = lambda g: spearmanr(g.true_ddg, g.pred_mu)[0]
addp = lambda g: spearmanr(g.true_ddg, g.add_pred)[0]
K["rank_full"], K["rank_full_ci"] = full(d), list(boot(d, full))
K["rank_add_pred"], K["rank_add_pred_ci"] = addp(d), list(boot(d, addp))
K["rank_diff_ci"] = list(boot(d, lambda g: full(g) - addp(g)))
K["rank_add_exp_reference"] = spearmanr(d.true_ddg, d.add_exp)[0]
P = pd.DataFrame([dict(pdb=p.replace(".pdb", ""), n=len(g), full=full(g), addsum=addp(g)) for p, g in d.groupby("pdb")])
K["per_protein_full_gt_add"] = int((P.full > P.addsum).sum()); K["n_proteins"] = len(P)
out = R / "megascale_test"; (out / "tables").mkdir(exist_ok=True)
P.round(4).to_csv(out / "tables/epistasis_fig_per_protein_ranking.csv", index=False)
json.dump(K, open(out / "tables/epistasis_fig_key_numbers.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=0.42))
# A
ax = axes[0]
hb = ax.hexbin(d.E_exp, d.E_pred, gridsize=40, cmap=matplotlib.colors.ListedColormap(cfg.SEQ_BLUE), mincnt=1,
               bins="log", linewidths=0.15, edgecolors="white", extent=(-5, 2.5, -4, 2))
ax.axhline(0, color=C["neutral"], lw=0.5, zorder=0); ax.axvline(0, color=C["neutral"], lw=0.5, zorder=0)
ax.plot([-4, 2], [-4, 2], color=C["ink2"], lw=0.7, ls="--")
ax.set_xlim(-5, 2.5); ax.set_ylim(-4, 2); ax.grid(False)
ax.set_xlabel(r"Experimental $E_\mathrm{exp}$ (kcal/mol)"); ax.set_ylabel(r"Predicted $E_\mathrm{pred}$ (kcal/mol)")
ax.text(0.03, 0.97, f"ρ = {K['rho_all']:.2f}\nn = {K['n']:,}", transform=ax.transAxes, va="top", fontsize=7)
cax = ax.inset_axes([0.60, 0.13, 0.34, 0.045])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=6, length=2, pad=1)
cb.outline.set_linewidth(0.4); cax.set_title("Variants per bin", fontsize=6.3, pad=2)
_panel_label(ax, "A")
# B
ax = axes[1]
x = np.arange(len(BINS)); off = 0.13
for k, (key, cik, lab, col, mk) in enumerate([("rho", "rho_ci", "Spearman ρ", C["blue"], "o"),
                                               ("partial", "partial_ci", "Partial ρ | additive sum", C["orange"], "s")]):
    y = np.array([K["bins"][b][key] for b in BINS]); ci = np.array([K["bins"][b][cik] for b in BINS])
    xx = x + (k - 0.5) * 2 * off
    ax.errorbar(xx, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="none", ecolor=col, elinewidth=1.2, capsize=0)
    ax.plot(xx, y, mk, color=col, ms=5.5, mec="white", mew=0.8, label=lab, zorder=3)
ax.axhline(0, color=C["ink2"], lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n({K['bins'][b]['n_pairs']})" for b in BINS], fontsize=6.8)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.2, 0.7)
ax.set_xlabel("Cα–Cα distance (site pairs)"); ax.set_ylabel(r"Correlation, $E_\mathrm{pred}$ vs $E_\mathrm{exp}$")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=6.3, handletextpad=0.3, borderaxespad=0.2)
_panel_label(ax, "B")
# C
ax = axes[2]
ax.plot([0, 1], [0, 1], color=C["ink2"], lw=0.7, ls="--", zorder=1)
above = P.full > P.addsum
ax.scatter(P.addsum[above], P.full[above], s=22, color=C["blue"], edgecolor="white", lw=0.6, zorder=3,
           label="Protein, model better")
ax.scatter(P.addsum[~above], P.full[~above], s=22, color=C["neutral"], edgecolor="white", lw=0.6, zorder=3,
           label="Protein, additive ≥ model")
ax.errorbar([K["rank_add_pred"]], [K["rank_full"]],
            xerr=[[K["rank_add_pred"] - K["rank_add_pred_ci"][0]], [K["rank_add_pred_ci"][1] - K["rank_add_pred"]]],
            yerr=[[K["rank_full"] - K["rank_full_ci"][0]], [K["rank_full_ci"][1] - K["rank_full"]]],
            fmt="D", color=C["orange"], ms=6.5, mec="white", mew=0.8, elinewidth=1.1, capsize=0, zorder=4, label="All doubles")
ax.set_xlim(-0.05, 1); ax.set_ylim(-0.05, 1)
ax.set_xlabel("ρ, sum of predicted singles"); ax.set_ylabel("ρ, double-mutant prediction")
ax.text(0.03, 0.97, f"{K['per_protein_full_gt_add']}/{K['n_proteins']} proteins above diagonal\n"
        f"All doubles: {K['rank_full']:.2f} vs {K['rank_add_pred']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.2, markerscale=0.9)
_panel_label(ax, "C")
_save(fig, R / "megascale_test/figures", "fig_epistasis_3panel")
print(json.dumps(K, indent=1, default=lambda v: round(float(v), 3)))
print(P.sort_values("full").round(2).to_string())
