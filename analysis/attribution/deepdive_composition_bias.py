"""Deep dive: is the global feature-importance table a property of the model, or of the dataset's mutation mix?

    python deepdive_composition_bias.py
Outputs -> <OUT_DIR>/deepdive_composition_bias/  (tables, key_numbers.json, fig14-fig16)

The pooled variance share  s_f = cov(A_f, t) / var(t),  t = mu - mu_null,  weights every variant equally, so it is a
mixture over the dataset's substitution mix, burial mix and ddG variance. Because IG is complete
(sum_f A_f = t), the shares sum to 1 under ANY weighting, which lets us re-weight exactly.

1. Composition of MegaScale-S / -D (classes, WT residues, burial). MegaScale is saturation mutagenesis: the mutant
   residue is ~uniform, so any skew comes from the WT sequence composition and from site selection (doubles).
2. Law-of-total-covariance split of each pooled share into BETWEEN-group (explained by group means) and
   WITHIN-group parts, for groups = substitution class, substitution pair (wt->mt), burial, pair x burial.
3. Re-weighted shares: class-balanced, class x burial-balanced, substitution-pair-balanced (each of the ~380
   substitutions weighted equally), and the mean WITHIN-class share (each class equal and normalised by its own
   variance). Protein-cluster bootstrap CIs.
4. Identity dependence per feature: eta^2 of A_f on the substitution pair (how much of the attribution is fixed by
   which residue goes to which) vs on pair x burial.
5. Leave-one-class-out rank stability of the pooled table.
6. Doubles: pooled vs burial-balanced vs pair-of-classes-balanced shares.
7. Profiles: class x burial mean signed attribution, within-class shares, WT- and mutant-residue profiles.
Forward predictions only.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

import config as cfg
import figures  # noqa: F401  (shared rcParams)
from deepdive_attribution_mechanisms import KEY_FEATS, substitution_class

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_composition_bias"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
N_BOOT = 500
MIN_GROUP = 30
K = {}
CLS_ORDER = ["charge removed", "hydrophobic→hydrophobic", "charge introduced", "hydrophobic→polar",
             "polar→hydrophobic", "from Gly", "polar→polar", "to Gly", "to Pro", "from Pro", "charge reversal",
             "charge kept"]
BUR = ["core", "boundary", "surface"]
AA = "GAVLIMFWYPSTNQHCDEKR"


# ---------------------------------------------------------------------------- helpers
def wshares(X, t, w=None, y=None):
    """Weighted cov(X_f, target)/cov(t, target); target = t (prediction) or y (experiment)."""
    w = np.ones(len(t)) if w is None else w
    w = w / w.sum()
    Xc, tc = X - w @ X, t - w @ t
    tgt = tc if y is None else y - w @ y
    return (Xc * (w * tgt)[:, None]).sum(0) / (tc * w * tgt).sum()


def inv_count(g):
    g = pd.Series(g)
    return (1.0 / g.map(g.value_counts())).values


def mean_within(X, t, g, min_n=50):
    g = np.asarray(g)
    out = []
    for c in pd.unique(g):
        i = g == c
        if i.sum() >= min_n and np.var(t[i]) > 0:
            out.append(wshares(X[i], t[i]))
    return np.mean(out, axis=0), len(out)


def between_within(X, t, g):
    """Split cov(X_f, t)/var(t) into between-group and within-group parts (population moments)."""
    df = pd.DataFrame(X, columns=F)
    df["_t"], df["_g"] = t, g
    mu = df.groupby("_g").mean()
    p = df.groupby("_g").size() / len(df)
    vt = np.var(t)
    between = ((mu[F] - X.mean(0)).mul((mu["_t"] - t.mean()) * p, axis=0)).sum(0).values / vt
    total = ((X - X.mean(0)) * (t - t.mean())[:, None]).mean(0) / vt
    return total, between, total - between


def eta2(x, g):
    df = pd.DataFrame({"x": x, "g": g})
    m = df.groupby("g").x.transform("mean")
    return np.var(m) / np.var(x)


def load():
    v = pd.read_csv(cfg.CACHE_DIR / DS / "variants.csv")
    A = {k: np.load(cfg.CACHE_DIR / DS / f"attr_{k}.npz")["F_tot"] for k in ("mut_mu", "mut_epistemic")}
    return v, A


# ---------------------------------------------------------------------------- 1. composition
def composition(s, d):
    K["composition"] = dict(
        n_singles=len(s), n_sites=int(s.groupby(["pdb", "pos1"]).ngroups),
        median_subs_per_site=float(s.groupby(["pdb", "pos1"]).size().median()),
        class_frac=s.cls.value_counts(normalize=True).round(3).to_dict(),
        wt_frac=s.wt1.map(cfg.THREE_TO_ONE).value_counts(normalize=True).round(3).to_dict(),
        mt_frac_range=[round(x, 3) for x in s.mt1.value_counts(normalize=True).agg(["min", "max"])],
        burial_singles=s.rsa_class.value_counts(normalize=True).round(3).to_dict(),
        burial_doubles=d.rsa_class.value_counts(normalize=True).round(3).to_dict(),
        class_mean_true=s.groupby("cls").true_ddg.mean().round(2).to_dict(),
        class_var_true=s.groupby("cls").true_ddg.var().round(2).to_dict())
    # share of the prediction / experimental variance explained by identity and burial alone
    t, y = (s.pred_mu - s.mu_null).values, s.true_ddg.values
    K["variance_explained_by_groups"] = {
        g: dict(pred=round(eta2(t, s[g]), 3), exp=round(eta2(y, s[g]), 3)) for g in ("cls", "sub", "rsa_class", "cls_bur", "sub_bur")}
    # how many variants and how much ddG covariance each class contributes to the pooled table
    tc = t - t.mean()
    contrib = pd.Series(tc ** 2).groupby(s.cls.values).sum() / (tc ** 2).sum()
    K["class_share_of_pred_variance"] = contrib.round(3).to_dict()


# ---------------------------------------------------------------------------- 2-5. singles
def singles_tables(s, X, t, y, fam):
    schemes = {
        "pooled": None,
        "class-balanced": inv_count(s.cls),
        "class×burial-balanced": inv_count(s.cls_bur),
        "substitution-balanced": inv_count(s["sub"]),
        "burial-balanced": inv_count(s.rsa_class),
    }
    rows = {k: wshares(X, t, w) for k, w in schemes.items()}
    mw_cls, ncls = mean_within(X, t, s.cls.values)
    mw_sub, nsub = mean_within(X, t, s["sub"].values, min_n=MIN_GROUP)
    rows["mean within-class"] = mw_cls
    rows["mean within-substitution"] = mw_sub
    T = pd.DataFrame(rows, index=F)
    if y is not None:
        T["exp_pooled"] = wshares(X, t, None, y)
        T["exp_substitution-balanced"] = wshares(X, t, inv_count(s["sub"]), y)
    # between / within splits
    for g in ("cls", "sub", "rsa_class", "sub_bur"):
        tot, bet, wit = between_within(X, t, s[g].values)
        T[f"between_{g}"], T[f"within_{g}"] = bet, wit
    T["eta2_sub"] = [eta2(X[:, j], s["sub"].values) for j in range(len(F))]
    T["eta2_sub_bur"] = [eta2(X[:, j], s["sub_bur"].values) for j in range(len(F))]
    # protein-cluster bootstrap
    pdbs = s.pdb.unique()
    idx_by = {p: np.where(s.pdb.values == p)[0] for p in pdbs}
    bs = {k: [] for k in ("pooled", "class-balanced", "substitution-balanced", "mean within-class")}
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[p] for p in RNG.choice(pdbs, len(pdbs))])
        ss, Xb, tb = s.iloc[ii], X[ii], t[ii]
        bs["pooled"].append(wshares(Xb, tb))
        bs["class-balanced"].append(wshares(Xb, tb, inv_count(ss.cls)))
        bs["substitution-balanced"].append(wshares(Xb, tb, inv_count(ss["sub"])))
        bs["mean within-class"].append(mean_within(Xb, tb, ss.cls.values)[0])
    for k, b in bs.items():
        b = np.array(b)
        T[f"{k}_lo"], T[f"{k}_hi"] = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
        T[f"{k}_rank_top1_freq"] = (np.argsort(-b, 1)[:, 0][:, None] == np.arange(len(F))).mean(0)
    # leave-one-class-out
    loco = {}
    for c in CLS_ORDER:
        keep = s.cls.values != c
        sh = pd.Series(wshares(X[keep], t[keep]), index=F).sort_values(ascending=False)
        loco[c] = [(f, round(v, 3)) for f, v in sh.head(4).items()]
    T.to_csv(OUT / f"singles_{fam}_shares_by_scheme.csv", float_format="%.4g")
    main = ["pooled", "class-balanced", "class×burial-balanced", "substitution-balanced", "burial-balanced",
            "mean within-class", "mean within-substitution"]
    K[f"singles_{fam}"] = {
        "top6": {m: [(f, round(v, 3)) for f, v in T[m].sort_values(ascending=False).head(6).items()] for m in main},
        "sum_check": {m: round(T[m].sum(), 4) for m in main},
        "rank_corr_vs_pooled": {m: round(spearmanr(T["pooled"], T[m])[0], 3) for m in main[1:]},
        "n_groups": dict(classes=ncls, substitutions=nsub),
        "leave_one_class_out_top4": loco,
        "between_sub_frac_of_pooled": {f: round(T.loc[f, "between_sub"] / T.loc[f, "pooled"], 3)
                                       for f in T.pooled.sort_values(ascending=False).index[:8]},
        "between_sub_bur_frac_of_pooled": {f: round(T.loc[f, "between_sub_bur"] / T.loc[f, "pooled"], 3)
                                           for f in T.pooled.sort_values(ascending=False).index[:8]},
        "eta2_sub": {f: round(T.loc[f, "eta2_sub"], 3) for f in T.pooled.sort_values(ascending=False).index[:8]},
    }
    return T


# ---------------------------------------------------------------------------- 6. doubles
def doubles_tables(d, X, t, y, fam):
    schemes = {"pooled": None, "burial-balanced": inv_count(d.rsa_class),
               "class-pair-balanced": inv_count(d.pair_cls), "pair×burial-balanced": inv_count(d.pair_bur)}
    T = pd.DataFrame({k: wshares(X, t, w) for k, w in schemes.items()}, index=F)
    T["mean within-class-pair"] = mean_within(X, t, d.pair_cls.values)[0]
    if y is not None:
        T["exp_pooled"] = wshares(X, t, None, y)
    for g in ("pair_cls", "rsa_class"):
        tot, bet, wit = between_within(X, t, d[g].values)
        T[f"between_{g}"], T[f"within_{g}"] = bet, wit
    pdbs = d.pdb.unique()
    idx_by = {p: np.where(d.pdb.values == p)[0] for p in pdbs}
    b1, b2 = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[p] for p in RNG.choice(pdbs, len(pdbs))])
        b1.append(wshares(X[ii], t[ii]))
        b2.append(wshares(X[ii], t[ii], inv_count(d.pair_cls.values[ii])))
    for k, b in (("pooled", b1), ("class-pair-balanced", b2)):
        T[f"{k}_lo"], T[f"{k}_hi"] = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
    T.to_csv(OUT / f"doubles_{fam}_shares_by_scheme.csv", float_format="%.4g")
    cols = list(schemes) + ["mean within-class-pair"]
    K[f"doubles_{fam}"] = {"top6": {m: [(f, round(v, 3)) for f, v in T[m].sort_values(ascending=False).head(6).items()]
                                    for m in cols},
                           "sum_check": {m: round(T[m].sum(), 4) for m in cols},
                           "n_class_pairs": int(d.pair_cls.nunique())}
    return T


# ---------------------------------------------------------------------------- 7. profiles
def profiles(s, X):
    df = pd.DataFrame(X, columns=F)
    df["cls"], df["bur"] = s.cls.values, s.rsa_class.values
    df["wt"], df["mt"] = s.wt1.map(cfg.THREE_TO_ONE).values, s.mt1.map(cfg.THREE_TO_ONE).values
    df["true"], df["pred"] = s.true_ddg.values, s.pred_mu.values
    cb = df.groupby(["cls", "bur"])[KEY_FEATS + ["true", "pred"]].mean()
    cb["n"] = df.groupby(["cls", "bur"]).size()
    cb.to_csv(OUT / "profile_class_by_burial.csv", float_format="%.4g")
    wt = df.groupby("wt")[F + ["true", "pred"]].mean()
    mt = df.groupby("mt")[F + ["true", "pred"]].mean()
    wt.to_csv(OUT / "profile_wt_residue.csv", float_format="%.4g")
    mt.to_csv(OUT / "profile_mut_residue.csv", float_format="%.4g")
    # within-class shares (each class normalised by its own variance)
    t = (s.pred_mu - s.mu_null).values
    W = pd.DataFrame({c: wshares(X[s.cls.values == c], t[s.cls.values == c]) for c in CLS_ORDER}, index=F).T
    W.to_csv(OUT / "profile_within_class_shares_all_features.csv", float_format="%.4g")
    # does the burial dependence survive inside each class? (core minus surface, key features)
    dcs = {}
    for c in CLS_ORDER:
        if ("core" in cb.loc[c].index) and ("surface" in cb.loc[c].index):
            dd = cb.loc[(c, "core"), KEY_FEATS] - cb.loc[(c, "surface"), KEY_FEATS]
            dcs[c] = [(f, round(dd[f], 3)) for f in dd.abs().sort_values(ascending=False).index[:3]]
    K["core_minus_surface_top3_by_class"] = dcs
    K["class_by_burial_true_pred"] = {f"{c} | {b}": dict(n=int(r.n), true=round(r.true, 2), pred=round(r.pred, 2))
                                      for (c, b), r in cb.iterrows()}
    return cb, wt, mt, W


# ---------------------------------------------------------------------------- figures
def div_cmap():
    return LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f08a89", "#b8292a"])


def fig_bias(s, T, Td):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), gridspec_kw=dict(wspace=0.75, hspace=0.45))
    # a. composition
    ax = axes[0, 0]
    fr = s.cls.value_counts(normalize=True).loc[CLS_ORDER[::-1]]
    ax.barh(range(len(fr)), fr.values * 100, color=C["blue"], height=0.7)
    ax.set_yticks(range(len(fr)), fr.index)
    ax.set_xlabel("% of MegaScale-S test variants")
    ax.set_title("a  singles by substitution class", loc="left")
    ax.grid(axis="y", visible=False)
    # b. schemes (singles)
    ax = axes[0, 1]
    order = T.pooled.sort_values(ascending=False).index[:9][::-1]
    sch = [("pooled", C["ink2"], "o"), ("class-balanced", C["blue"], "s"),
           ("substitution-balanced", C["orange"], "D"), ("mean within-class", C["aqua"], "^")]
    y = np.arange(len(order))
    for k, (m, col, mk) in enumerate(sch):
        off = (k - 1.5) * 0.18
        ax.hlines(y + off, T.loc[order, f"{m}_lo"] * 100, T.loc[order, f"{m}_hi"] * 100, color=col, lw=1)
        ax.plot(T.loc[order, m] * 100, y + off, mk, ms=4.5, color=col, mec="white", mew=0.5, label=m)
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(y, order)
    ax.set_xlabel("share of prediction variance (%)")
    ax.set_title("b  singles: importance under re-weighting", loc="left")
    ax.legend(fontsize=6.3, loc="lower right")
    # c. between (substitution identity) vs within
    ax = axes[1, 0]
    b, w = T.loc[order, "between_sub"] * 100, T.loc[order, "within_sub"] * 100
    ax.barh(y, b, color=C["blue"], height=0.65, label="between substitutions (identity)")
    ax.barh(y, w, left=np.where(w >= 0, b.clip(lower=0), 0), color=C["orange"], height=0.65,
            label="within substitution (context)")
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(y, order)
    ax.set_xlabel("pooled share (%), split")
    ax.set_title("c  identity vs context", loc="left")
    ax.legend(fontsize=6.3, loc="lower right")
    # d. doubles
    ax = axes[1, 1]
    od = Td.pooled.sort_values(ascending=False).index[:9][::-1]
    yd = np.arange(len(od))
    for k, (m, col, mk) in enumerate([("pooled", C["ink2"], "o"), ("burial-balanced", C["violet"], "s"),
                                      ("class-pair-balanced", C["orange"], "D"),
                                      ("mean within-class-pair", C["aqua"], "^")]):
        off = (k - 1.5) * 0.18
        if f"{m}_lo" in Td:
            ax.hlines(yd + off, Td.loc[od, f"{m}_lo"] * 100, Td.loc[od, f"{m}_hi"] * 100, color=col, lw=1)
        ax.plot(Td.loc[od, m] * 100, yd + off, mk, ms=4.5, color=col, mec="white", mew=0.5, label=m)
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(yd, od)
    ax.set_xlabel("share of prediction variance (%)")
    ax.set_title("d  doubles: importance under re-weighting", loc="left")
    ax.legend(fontsize=6.3, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig14_composition_bias.{ext}")
    plt.close(fig)


def fig_profiles(cb, W):
    feats = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Cyclic", "Backbone Entropy",
             "Hydrogen Bonds", "Electrostatics", "Charge", "VdW Energy", "Entropy"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.6), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=1.35))
    ax = axes[0]
    M = W.loc[CLS_ORDER, feats] * 100
    im = ax.imshow(M.values, cmap=div_cmap(), vmin=-60, vmax=60, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
    ax.set_yticks(range(len(CLS_ORDER)), CLS_ORDER)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M.values[i, j]) >= 10:
                ax.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(M.values[i, j]) > 40 else C["ink"])
    ax.set_title("a  within-class variance share (%)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    ax = axes[1]
    rows = [(c, b) for c in CLS_ORDER for b in BUR if (c, b) in cb.index]
    M = cb.loc[rows, feats]
    vmax = np.nanpercentile(np.abs(M.values), 99)
    im = ax.imshow(M.values, cmap=div_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
    ax.set_yticks(range(len(rows)), [f"{c} · {b} ({cb.loc[(c, b), 'true']:.1f}/{cb.loc[(c, b), 'pred']:.1f})"
                                     for c, b in rows], fontsize=5.3)
    ax.grid(False)
    for k in range(1, len(CLS_ORDER)):
        ax.axhline(3 * k - 0.5, color="white", lw=1.2)
    ax.set_title("b  mean signed attribution (kcal/mol)\n    by class × burial  (exp/pred ΔΔG)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig15_class_profiles.{ext}")
    plt.close(fig)


def fig_residues(wt, mt):
    feats = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Cyclic", "Backbone Entropy",
             "Hydrogen Bonds", "Electrostatics", "Charge", "Aromatic", "VdW Energy"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw=dict(wspace=0.35))
    for ax, M, lab in ((axes[0], wt, "a  by wild-type residue"), (axes[1], mt, "b  by mutant residue")):
        rows = [a for a in AA if a in M.index]
        Z = M.loc[rows, feats]
        vmax = 0.8
        im = ax.imshow(Z.values, cmap=div_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
        ax.set_yticks(range(len(rows)), [f"{a}  ({M.loc[a, 'true']:.2f}/{M.loc[a, 'pred']:.2f})" for a in rows],
                      fontsize=6)
        ax.grid(False)
        ax.set_title(f"{lab}\n    mean signed attribution (exp/pred ΔΔG)", loc="left", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02, label="kcal/mol")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig16_residue_profiles.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main():
    v, A = load()
    m1, m2 = (v.n_mutated == 1).values, (v.n_mutated == 2).values
    s, d = v[m1].copy(), v[m2].copy()
    s["cls"] = [substitution_class(w, mt) for w, mt in zip(s.wt1, s.mt1)]
    s["sub"] = s.wt1.map(cfg.THREE_TO_ONE) + ">" + s.mt1.map(cfg.THREE_TO_ONE)
    s["cls_bur"] = s.cls + "|" + s.rsa_class
    s["sub_bur"] = s["sub"] + "|" + s.rsa_class
    c1 = [substitution_class(w, mt) for w, mt in zip(d.wt1, d.mt1)]
    c2 = [substitution_class(w, mt) for w, mt in zip(d.wt2, d.mt2)]
    d["pair_cls"] = [" + ".join(sorted(p)) for p in zip(c1, c2)]
    vc = d.pair_cls.value_counts()
    d.loc[d.pair_cls.map(vc) < 50, "pair_cls"] = "other"
    d["pair_bur"] = d.pair_cls + "|" + d.rsa_class
    composition(s, d)
    t1, y1 = (s.pred_mu - s.mu_null).values, s.true_ddg.values
    T = singles_tables(s, A["mut_mu"][m1], t1, y1, "mu")
    ts = (s.f_mut_epistemic - s.f0_mut_epistemic).values
    singles_tables(s, A["mut_epistemic"][m1], ts, None, "log_sigma_epi")
    t2, y2 = (d.pred_mu - d.mu_null).values, d.true_ddg.values
    Td = doubles_tables(d, A["mut_mu"][m2], t2, y2, "mu")
    cb, wt, mt, W = profiles(s, A["mut_mu"][m1])
    fig_bias(s, T, Td)
    fig_profiles(cb, W)
    fig_residues(wt, mt)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(K, indent=1, ensure_ascii=False, default=str)[:12000])


if __name__ == "__main__":
    main()
