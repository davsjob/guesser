"""Deep dive: what do the attributions teach us on MegaScale-S / -D?

    python deepdive_attribution_mechanisms.py
Outputs -> <OUT_DIR>/deepdive_attribution_mechanisms/  (tables, key_numbers.json, fig10-fig13)

1. Exact, bin-free importance: share of prediction variance per feature. IG is complete, so
   sum_f cov(A_f, mu - mu_null) = var(mu - mu_null). Also each feature's share of cov(mu, ddG_exp).
   Done for the mutation IG on mu, the environment IG on mu, and the mutation IG on log sigma_epi
   ("what drives variation in uncertainty").
2. Mutation classes (singles): attribution profile of mu and of log sigma_epi per substitution class,
   plus a burial split.
3. Charge pairs (doubles): electrostatics attribution and non-additivity vs Calpha distance.
4. "Expansion": does attribution spread spatially as uncertainty rises? Uses the |attribution|-weighted
   mean Calpha distance and the normalised spatial entropy per variant, pooled / within protein /
   controlling for graph size, |mu| and burial. Also feature-share shifts with sigma inside |mu| strata.
Forward predictions only.
"""
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr

import config as cfg
import figures  # noqa: F401  (shared rcParams)

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_attribution_mechanisms"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
KEY_FEATS = ["Hydrophobicity", "Polar Solvation", "Hydrophobic Solvation", "Volume", "VdW Clashes", "VdW Energy",
             "Electrostatics", "Charge", "Hydrogen Bonds", "Cyclic", "Backbone Entropy", "Entropy"]
CHARGE = {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1}
HYD, POL = set("AVILMFWC"), set("STNQYH")
K = {}


def partial(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([rankdata(c) for c in Z] + [np.ones(len(x))])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return pearsonr(ex, ey)


def load():
    v = pd.read_csv(cfg.CACHE_DIR / DS / "variants.csv")
    A = {k: np.load(cfg.CACHE_DIR / DS / f"attr_{k}.npz")["F_tot"] for k in cfg.ATTR_KEYS}
    return v, A


def substitution_class(w3, m3):
    w, m = cfg.THREE_TO_ONE.get(w3, "X"), cfg.THREE_TO_ONE.get(m3, "X")
    if m == "P":
        return "to Pro"
    if w == "P":
        return "from Pro"
    if m == "G":
        return "to Gly"
    if w == "G":
        return "from Gly"
    cw, cm = CHARGE.get(w3, 0), CHARGE.get(m3, 0)
    if cw * cm < 0:
        return "charge reversal"
    if cw == 0 and cm != 0:
        return "charge introduced"
    if cw != 0 and cm == 0:
        return "charge removed"
    if cw != 0 and cm != 0:
        return "charge kept"
    if w in HYD and m in POL:
        return "hydrophobic→polar"
    if w in POL and m in HYD:
        return "polar→hydrophobic"
    if w in HYD and m in HYD:
        return "hydrophobic→hydrophobic"
    return "polar→polar"


# ---------------------------------------------------------------------------- 1. variance decomposition
def variance_decomposition(v, A):
    rows = []
    for k, sub in ((1, "singles"), (2, "doubles")):
        m = (v.n_mutated == k).values
        s = v[m]
        y = s.true_ddg.values
        targets = {
            "mut_mu": (A["mut_mu"][m], (s.pred_mu - s.mu_null).values),
            "ctx_mu": (A["ctx_mu"][m], (s.f_ctx_mu - s.f0_ctx_mu).values),
            "mut_epistemic": (A["mut_epistemic"][m], (s.f_mut_epistemic - s.f0_mut_epistemic).values),
        }
        for fam, (X, t) in targets.items():
            vt = np.var(t, ddof=1)
            cy = np.cov(t, y)[0, 1]
            for j, f in enumerate(F):
                rows.append(dict(subset=sub, family=fam, feature=f,
                                 var_share=np.cov(X[:, j], t)[0, 1] / vt,
                                 cov_truth_share=(np.cov(X[:, j], y)[0, 1] / cy) if fam != "mut_epistemic" else np.nan,
                                 mean_abs=np.abs(X[:, j]).mean(), mean_signed=X[:, j].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "variance_decomposition.csv", index=False, float_format="%.4g")
    K["variance_decomposition"] = {
        f"{sub}_{fam}": [(r.feature, round(r.var_share, 3)) for r in
                         t[(t.subset == sub) & (t.family == fam)].sort_values("var_share", ascending=False).head(6).itertuples()]
        for sub in ("singles", "doubles") for fam in ("mut_mu", "ctx_mu", "mut_epistemic")}
    return t


# ---------------------------------------------------------------------------- 2. mutation classes
def mutation_classes(v, A):
    m = (v.n_mutated == 1).values
    s = v[m].copy()
    s["cls"] = [substitution_class(w, mt) for w, mt in zip(s.wt1, s.mt1)]
    ki = [F.index(f) for f in KEY_FEATS]
    stats = s.groupby("cls").agg(n=("true_ddg", "size"), mean_true=("true_ddg", "mean"), mean_pred=("pred_mu", "mean"),
                                 mae=("abs_err", "mean"), mean_sigma_e=("epistemic_std", "mean"),
                                 frac_core=("rsa_class", lambda x: (x == "core").mean()))
    stats["rho_pred_true"] = [spearmanr(g.pred_mu, g.true_ddg)[0] for _, g in s.groupby("cls")]
    stats["rho_err_sigma"] = [spearmanr(g.abs_err, g.epistemic_std)[0] for _, g in s.groupby("cls")]
    stats = stats.sort_values("mean_true")
    order = stats.index.tolist()
    mats = {}
    for fam, lab in (("mut_mu", "mu"), ("mut_epistemic", "log_sigma_epi"), ("ctx_epistemic", "log_sigma_epi_ctx")):
        X = pd.DataFrame(A[fam][m][:, ki], columns=KEY_FEATS)
        X["cls"] = s.cls.values
        mats[lab] = X.groupby("cls").mean().loc[order]
        mats[lab].to_csv(OUT / f"class_mean_attr_{lab}.csv", float_format="%.4g")
    # within-class variance shares for mu (which features explain variation *inside* each class)
    rows = []
    for c, g in s.groupby("cls"):
        idx = s.cls.values == c
        X = A["mut_mu"][m][idx]
        t = (g.pred_mu - g.mu_null).values
        vt = np.var(t, ddof=1)
        for j in ki:
            rows.append(dict(cls=c, feature=F[j], var_share=np.cov(X[:, j], t)[0, 1] / vt,
                             rho_attr_true=spearmanr(X[:, j], g.true_ddg)[0] if np.ptp(X[:, j]) > 0 else np.nan))
    within = pd.DataFrame(rows)
    within.to_csv(OUT / "class_within_variance_shares.csv", index=False, float_format="%.4g")
    burial = s.groupby(["cls", "rsa_class"], observed=True).agg(
        n=("true_ddg", "size"), mean_true=("true_ddg", "mean"), mean_pred=("pred_mu", "mean"), mae=("abs_err", "mean"),
        mean_sigma_e=("epistemic_std", "mean")).reset_index()
    burial.to_csv(OUT / "class_by_burial.csv", index=False, float_format="%.4g")
    stats.to_csv(OUT / "class_summary.csv", float_format="%.4g")
    top = lambda M, c, n=3: [(f, round(float(M.loc[c, f]), 3)) for f in M.loc[c].abs().sort_values(ascending=False).index[:n]]
    K["mutation_classes"] = {c: dict(n=int(stats.loc[c, "n"]), mean_true=round(stats.loc[c, "mean_true"], 2),
                                     mean_pred=round(stats.loc[c, "mean_pred"], 2), mae=round(stats.loc[c, "mae"], 3),
                                     sigma_e=round(stats.loc[c, "mean_sigma_e"], 3),
                                     rho_err_sigma=round(stats.loc[c, "rho_err_sigma"], 3),
                                     top_mu_attr=top(mats["mu"], c), top_sigma_attr=top(mats["log_sigma_epi"], c))
                             for c in order}
    ci = burial[burial.cls == "charge introduced"].set_index("rsa_class")
    K["charge_introduced_by_burial"] = {r: dict(n=int(ci.loc[r, "n"]), true=round(ci.loc[r, "mean_true"], 2),
                                                pred=round(ci.loc[r, "mean_pred"], 2)) for r in ci.index}
    return stats, mats


# ---------------------------------------------------------------------------- 3. charge pairs (doubles)
def charge_pairs(v, A):
    m = (v.n_mutated == 2).values
    d = v[m].copy()
    ep = pd.read_csv(cfg.OUT_DIR / DS / "tables" / "epistasis_per_variant.csv")
    d["E_exp"], d["E_pred"] = ep.E_exp.values, ep.E_pred.values
    w1, w2 = d.wt1.map(CHARGE).fillna(0).values, d.wt2.map(CHARGE).fillna(0).values
    m1, m2 = d.mt1.map(CHARGE).fillna(0).values, d.mt2.map(CHARGE).fillna(0).values
    wt_op, mt_op, wt_same, mt_same = w1 * w2 < 0, m1 * m2 < 0, w1 * w2 > 0, m1 * m2 > 0
    d["pair"] = np.select([mt_op & ~wt_op, wt_op & ~mt_op, mt_same & ~wt_same, wt_same & ~mt_same],
                          ["opposite charges introduced", "opposite-charge pair removed", "like charges introduced",
                           "like-charge pair removed"], "no charge-pair change")
    d["dbin"] = pd.cut(d.d12, [0, 6, 9, 999], labels=["<6 Å", "6–9 Å", ">9 Å"])
    d["elec"] = A["mut_mu"][m][:, F.index("Electrostatics")]
    d["elec_sigma"] = A["mut_epistemic"][m][:, F.index("Electrostatics")]
    rows = []
    for (p, b), g in d.groupby(["pair", "dbin"], observed=True):
        bs = [g.elec.values[RNG.integers(0, len(g), len(g))].mean() for _ in range(1000)]
        rows.append(dict(pair=p, d12=b, n=len(g), n_matched=int(g.E_exp.notna().sum()), mean_elec_attr=g.elec.mean(),
                         elec_ci_lo=np.percentile(bs, 2.5), elec_ci_hi=np.percentile(bs, 97.5),
                         mean_E_exp=g.E_exp.mean(), mean_E_pred=g.E_pred.mean(), mean_gamma_epi=g.pred_gamma_epi.mean(),
                         mean_true=g.true_ddg.mean(), mean_pred=g.pred_mu.mean(), mae=g.abs_err.mean(),
                         mean_sigma_e=g.epistemic_std.mean(), mean_elec_sigma_attr=g.elec_sigma.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "charge_pairs_by_distance.csv", index=False, float_format="%.4g")
    res = {}
    for p in ("like charges introduced", "opposite charges introduced", "opposite-charge pair removed"):
        a, b = d[(d.pair == p) & (d.dbin == "<6 Å")], d[(d.pair == p) & (d.dbin == ">9 Å")]
        res[p] = dict(elec_close=round(a.elec.mean(), 3), elec_far=round(b.elec.mean(), 3),
                      p_elec=float(f"{mannwhitneyu(a.elec, b.elec).pvalue:.2g}"),
                      E_exp_close=round(a.E_exp.mean(), 3), E_exp_far=round(b.E_exp.mean(), 3),
                      p_E_exp=float(f"{mannwhitneyu(a.E_exp.dropna(), b.E_exp.dropna()).pvalue:.2g}"),
                      n_close=len(a), n_far=len(b))
    cp = d[d.pair.isin(["like charges introduced", "opposite charges introduced"]) & d.E_exp.notna()]
    res["rho_E_pred_E_exp_charge_pairs"] = round(spearmanr(cp.E_pred, cp.E_exp)[0], 3)
    K["charge_pairs"] = res
    return t


# ---------------------------------------------------------------------------- 4. expansion
def expansion(v, A):
    g = np.load(cfg.CACHE_DIR / DS / "geom.npz", allow_pickle=True)
    dist = g["node_dist"].astype(float)
    with h5py.File(cfg.EXPL_DIR / cfg.DATASETS[DS], "r") as f:
        no = f["mutation/node_offset"][:]
        N = len(no) - 1
        var = np.repeat(np.arange(N), np.diff(no))
        out = {}
        for key in ("signed_attr_mu_wt", "signed_attr_epistemic_wt", "signed_attr_epistemic", "signed_attr_mu"):
            mass = np.abs(f["node"][key][:].astype(np.float64)).sum(1)
            tot = np.bincount(var, weights=mass, minlength=N) + 1e-12
            wd = np.bincount(var, weights=mass * dist, minlength=N) / tot
            p = mass / tot[var]
            ent = -np.bincount(var, weights=np.where(p > 0, p * np.log(p), 0), minlength=N) / np.log(np.diff(no))
            out[key] = (wd, ent)
    rows, K["expansion"] = [], {}
    for k, sub in ((1, "singles"), (2, "doubles")):
        m = (v.n_mutated == k).values
        s = v[m].copy()
        s["sig_pct"] = s.groupby("pdb").epistemic_std.rank(pct=True)
        for key, lab in (("signed_attr_mu_wt", "env IG on μ"), ("signed_attr_epistemic_wt", "env IG on log σ_epi"),
                         ("signed_attr_epistemic", "mut IG on log σ_epi")):
            wd, ent = out[key][0][m], out[key][1][m]
            s["wd"], s["ent"] = wd, ent
            s["wd_pct"] = s.groupby("pdb").wd.rank(pct=True)
            r = dict(subset=sub, family=lab,
                     rho_sigma_wdist=spearmanr(s.epistemic_std, wd)[0],
                     rho_sigma_wdist_within_protein=spearmanr(s.sig_pct, s.wd_pct)[0],
                     partial_sigma_wdist=partial(s.epistemic_std.values, wd,
                                                 [s.n_nodes, s.pred_mu.abs(), s.rsa_mean])[0],
                     rho_sigma_entropy=spearmanr(s.epistemic_std, ent)[0],
                     partial_sigma_entropy=partial(s.epistemic_std.values, ent,
                                                   [s.n_nodes, s.pred_mu.abs(), s.rsa_mean])[0],
                     rho_nnodes_wdist=spearmanr(s.n_nodes, wd)[0], rho_nnodes_sigma=spearmanr(s.n_nodes, s.epistemic_std)[0])
            q = pd.qcut(s.sig_pct, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            for qq in ["Q1", "Q2", "Q3", "Q4"]:
                r[f"wdist_{qq}"] = s.wd[q == qq].mean()
                r[f"entropy_{qq}"] = s.ent[q == qq].mean()
            rows.append(r)
            K["expansion"][f"{sub}|{lab}"] = {kk: round(float(vv), 3) for kk, vv in r.items() if kk not in ("subset", "family")}
        # feature-share shift with sigma, inside |mu| deciles (removes the magnitude confound)
        X = np.abs(A["mut_mu"][m])
        sh = X / np.maximum(X.sum(1, keepdims=True), 1e-12)
        mud = pd.qcut(s.pred_mu.abs(), 10, labels=False, duplicates="drop").values
        hi = np.zeros(len(s), bool)
        lo = np.zeros(len(s), bool)
        for b in np.unique(mud):
            mb = mud == b
            qs = s.epistemic_std.values[mb]
            hi[np.where(mb)[0][qs >= np.quantile(qs, .75)]] = True
            lo[np.where(mb)[0][qs <= np.quantile(qs, .25)]] = True
        shift = pd.DataFrame({"feature": F, "share_low_sigma": sh[lo].mean(0), "share_high_sigma": sh[hi].mean(0)})
        shift["delta_pp"] = (shift.share_high_sigma - shift.share_low_sigma) * 100
        shift.sort_values("delta_pp").to_csv(OUT / f"feature_share_shift_with_sigma_{sub}.csv", index=False, float_format="%.4g")
        K["expansion"][f"{sub}|feature_shift_top"] = [(r.feature, round(r.delta_pp, 2)) for r in
                                                      shift.reindex(shift.delta_pp.abs().sort_values(ascending=False).index).head(6).itertuples()]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "expansion.csv", index=False, float_format="%.4g")
    return t, out


# ---------------------------------------------------------------------------- figures
def div_cmap():
    return matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])


def fig_classes(stats, mats):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(width_ratios=[1.0, 4.2, 4.2], wspace=0.08))
    order = stats.index.tolist()
    y = np.arange(len(order))
    ax = axes[0]
    ax.barh(y, stats.mean_true, height=0.6, color=C["neutral"], label="experimental")
    ax.plot(stats.mean_pred, y, "o", color=C["orange"], ms=4, label="predicted")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={int(n):,})" for c, n in zip(order, stats.n)])
    ax.invert_yaxis()
    ax.set_xlabel("mean ΔΔG")
    ax.legend(fontsize=6, loc="center right")
    ax.grid(axis="y", visible=False)
    for ax, lab, title, vmax in ((axes[1], "mu", "Mutation IG on μ (kcal/mol)", 0.6),
                                 (axes[2], "log_sigma_epi", "Mutation IG on log σ_epi (+ = raises uncertainty)", 0.4)):
        M = mats[lab]
        im = ax.imshow(M.values, cmap=div_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.values[i, j]
                if abs(val) >= 0.05:
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if abs(val) > 0.6 * vmax else C["ink"])
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig10_mutation_class_attribution.{ext}")
    plt.close(fig)


def fig_charge(t):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw=dict(wspace=0.4))
    pairs = [("like charges introduced", C["red"]), ("opposite charges introduced", C["blue"]),
             ("opposite-charge pair removed", C["violet"])]
    bins = ["<6 Å", "6–9 Å", ">9 Å"]
    for k, (p, col) in enumerate(pairs):
        q = t[t.pair == p].set_index("d12").reindex(bins)
        x = np.arange(3) + (k - 1) * 0.18
        axes[0].errorbar(x, q.mean_elec_attr, yerr=[q.mean_elec_attr - q.elec_ci_lo, q.elec_ci_hi - q.mean_elec_attr],
                         fmt="o", color=col, ms=5, capsize=2, lw=1.2, label=p)
        axes[1].plot(np.arange(3), q.mean_E_exp, "-o", color=col, lw=1.6, ms=4.5, label=f"{p}: E_exp")
        axes[1].plot(np.arange(3), q.mean_E_pred, "--s", color=col, lw=1.0, ms=3.5, alpha=0.8)
    axes[0].axhline(0, color=C["ink2"], lw=0.6)
    axes[0].set_ylabel("Electrostatics attribution to μ (kcal/mol)")
    axes[0].set_title("Model electrostatics by site distance", loc="left")
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    axes[1].set_ylabel("non-additivity (kcal/mol)")
    axes[1].set_title("E_exp (solid) vs E_pred (dashed)", loc="left")
    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(bins)
        ax.set_xlabel("Cα–Cα distance between the two sites")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig11_charge_pairs.{ext}")
    plt.close(fig)


def fig_expansion(t):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9), gridspec_kw=dict(wspace=0.4))
    for k, sub in enumerate(("singles", "doubles")):
        ax = axes[k]
        q = t[t.subset == sub]
        for j, r in enumerate(q.itertuples()):
            ax.plot(range(4), [getattr(r, f"wdist_Q{i}") for i in range(1, 5)], "-o", color=cfg.SERIES[j], lw=1.8,
                    ms=4.5, label=f"{r.family} (partial ρ={r.partial_sigma_wdist:+.2f})")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Q1\nlow σ", "Q2", "Q3", "Q4\nhigh σ"])
        ax.set_xlabel("σ_e quartile (within protein)")
        ax.set_ylabel("|attr|-weighted mean Cα distance (Å)")
        ax.set_title(f"MegaScale {sub}", loc="left")
        ax.legend(fontsize=5.8, loc="best")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig12_expansion.{ext}")
    plt.close(fig)


def fig_variance(vd):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(wspace=0.7))
    for ax, fam, title in zip(axes, ("mut_mu", "ctx_mu", "mut_epistemic"),
                              ("Mutation IG → μ", "Environment IG → μ", "Mutation IG → log σ_epi")):
        q = vd[vd.family == fam]
        top = q[q.subset == "singles"].sort_values("var_share", ascending=False).head(8).feature.tolist()[::-1]
        y = np.arange(len(top))
        for k, sub in enumerate(("singles", "doubles")):
            qq = q[q.subset == sub].set_index("feature").loc[top]
            ax.barh(y + (k - 0.5) * 0.38, qq.var_share * 100, height=0.36, color=cfg.SERIES[k], label=sub)
        ax.axvline(0, color=C["ink2"], lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(top, fontsize=7)
        ax.set_xlabel("share of variance (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].legend(fontsize=6.5, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig13_variance_decomposition.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    v, A = load()
    vd = variance_decomposition(v, A)
    stats, mats = mutation_classes(v, A)
    cp = charge_pairs(v, A)
    ex, _ = expansion(v, A)
    fig_variance(vd)
    fig_classes(stats, mats)
    fig_charge(cp)
    fig_expansion(ex)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, default=str)
    print(json.dumps(K, indent=1, default=str))
