"""Stage 2: analysis modules. Each writes CSV tables and returns a dict of key numbers.

    python analyses.py --dataset megascale_test          # one dataset
    python analyses.py --all                              # all five

Outputs: results/attribution/<dataset>/tables/*.csv and key_numbers.json
Conventions: ddG > 0 = destabilizing; "mut" = mutation IG (x_diff, WT fixed; sums to mu - mu_null);
"ctx" = environment IG (x_wt vs per-graph mean; sums to f - f0_ctx). Protein-level (PDB) cluster
bootstrap for confidence intervals.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr

import config as cfg

F = cfg.FEATURES
NF = len(F)


# ---------------------------------------------------------------------------- helpers
def load(ds):
    d = cfg.CACHE_DIR / ds
    v = pd.read_csv(d / "variants.csv")
    A = {k: dict(np.load(d / f"attr_{k}.npz")) for k in cfg.ATTR_KEYS}
    g = dict(np.load(d / "geom.npz", allow_pickle=True))
    return v, A, g


def subsets(v):
    out = []
    if (v.n_mutated == 1).any():
        out.append(("singles", (v.n_mutated == 1).values))
    if (v.n_mutated == 2).any():
        out.append(("doubles", (v.n_mutated == 2).values))
    return out


def rho_cols(X, y):
    """Spearman rho of every column of X with y (NaN-safe on y; constant columns -> NaN)."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    ry = rankdata(y)
    ry = (ry - ry.mean()) / ry.std()
    out = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) > 0:
            rx = rankdata(X[:, j])
            out[j] = np.mean((rx - rx.mean()) / rx.std() * ry)
    return out


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = pearsonr(ex, ey)
    return r, p


def boot_diff(X, groups, ma, mb, n_boot=cfg.N_BOOT, seed=cfg.SEED):
    """mean(X[ma]) - mean(X[mb]) per column, with protein-cluster bootstrap 95% CI."""
    ug, gi = np.unique(groups, return_inverse=True)
    G = len(ug)
    Sa = np.zeros((G, X.shape[1]))
    Sb = np.zeros((G, X.shape[1]))
    np.add.at(Sa, gi[ma], X[ma])
    np.add.at(Sb, gi[mb], X[mb])
    na = np.bincount(gi[ma], minlength=G).astype(float)
    nb = np.bincount(gi[mb], minlength=G).astype(float)
    point = Sa.sum(0) / max(na.sum(), 1) - Sb.sum(0) / max(nb.sum(), 1)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.ones(G) / G, size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ Sa) / (W @ na)[:, None] - (W @ Sb) / (W @ nb)[:, None]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return point, lo, hi


def tiers(x):
    return x < cfg.TIER_STAB, (x >= cfg.TIER_STAB) & (x <= cfg.TIER_NEUT_HI), x > cfg.TIER_DESTAB


def perf(y, p):
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(y - p))), rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]))


def _r(x, n=3):
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), n)
    return x


# ---------------------------------------------------------------------------- A0 sanity
def a0_sanity(v, A, g, T, K):
    rows = []
    for name, m in subsets(v):
        s = v[m]
        r = dict(subset=name, **perf(s.true_ddg.values, s.pred_mu.values))
        r.update(
            mu_null_median_abs=s.mu_null.abs().median(), mu_null_p95_abs=s.mu_null.abs().quantile(.95),
            mu_null_mean=s.mu_null.mean(),
            compl_err_mut_mu_max=s.completeness_err_mut_mu.abs().max(),
            compl_err_mut_mu_median=s.completeness_err_mut_mu.abs().median(),
            compl_err_ctx_mu_max=s.completeness_err_ctx_mu.abs().max(),
            compl_err_mut_epistemic_max=s.completeness_err_mut_epistemic.abs().max(),
            mapping_ok_frac=s.mapping_ok.mean(), wt_identity_frac=(s.wt_identity_match == 1).mean(),
            frac_pred_stab=(s.pred_mu < 0).mean(), frac_true_stab=(s.true_ddg < 0).mean(),
            precision_pred_stab=((s.pred_mu < 0) & (s.true_ddg < 0)).sum() / max((s.pred_mu < 0).sum(), 1),
            rsa_core_frac=(s.rsa_class == "core").mean(), rsa_surface_frac=(s.rsa_class == "surface").mean(),
            mean_nodes=s.n_nodes.mean(),
        )
        rows.append(r)
    t = pd.DataFrame(rows)
    T["sanity"] = t
    K["sanity"] = {r["subset"]: {k: _r(val, 4) for k, val in r.items() if k != "subset"} for r in rows}


# ---------------------------------------------------------------------------- A1 driving forces
def a1_driving(v, A, g, T, K):
    X = A["mut_mu"]["F_tot"]
    pdb = v.pdb.values
    K["driving"] = {}
    for name, m in subsets(v):
        Xs, s = X[m], v[m]
        y, p = s.true_ddg.values, s.pred_mu.values
        ps, pn, pd_ = tiers(p)
        ts, tn, td = tiers(y)
        sp, splo, sphi = boot_diff(Xs, pdb[m], pd_, ps)
        st, stlo, sthi = boot_diff(Xs, pdb[m], td, ts)
        absX = np.abs(Xs)
        t = pd.DataFrame({
            "feature": F, "group": [cfg.FEATURE_GROUP[f] for f in F],
            "mean_abs": absX.mean(0), "share_abs": absX.mean(0) / absX.mean(0).sum(),
            "mean_signed": Xs.mean(0),
            "predtier_stab": Xs[ps].mean(0), "predtier_neutral": Xs[pn].mean(0), "predtier_destab": Xs[pd_].mean(0),
            "shift_pred": sp, "shift_pred_lo": splo, "shift_pred_hi": sphi,
            "truetier_stab": Xs[ts].mean(0), "truetier_neutral": Xs[tn].mean(0), "truetier_destab": Xs[td].mean(0),
            "shift_true": st, "shift_true_lo": stlo, "shift_true_hi": sthi,
            "rho_true": rho_cols(Xs, y), "rho_pred": rho_cols(Xs, p),
        })
        t["n_stab_pred"], t["n_destab_pred"] = ps.sum(), pd_.sum()
        t["n_stab_true"], t["n_destab_true"] = ts.sum(), td.sum()
        t = t.reindex(t.shift_true.abs().sort_values(ascending=False).index).reset_index(drop=True)
        T[f"driving_{name}"] = t
        # burial strata
        rows = []
        for cls in ["core", "boundary", "surface"]:
            mc = (s.rsa_class == cls).values
            if mc.sum() < 30:
                continue
            ts2, _, td2 = tiers(y[mc])
            if ts2.sum() >= 5 and td2.sum() >= 5:
                sh, lo, hi = boot_diff(Xs[mc], pdb[m][mc], td2, ts2)
            else:
                sh = lo = hi = np.full(NF, np.nan)
            rr = rho_cols(Xs[mc], y[mc])
            for j, f in enumerate(F):
                rows.append(dict(rsa_class=cls, n=int(mc.sum()), feature=f, mean_abs=np.abs(Xs[mc][:, j]).mean(),
                                 mean_signed=Xs[mc][:, j].mean(), shift_true=sh[j], shift_true_lo=lo[j],
                                 shift_true_hi=hi[j], rho_true=rr[j]))
        tb = pd.DataFrame(rows)
        T[f"burial_{name}"] = tb
        # volume bidirectionality (self dX of Volume)
        jv = F.index("Volume")
        dv = g["dx_self"][m][:, jv]
        bins = [-np.inf, -40, -10, 10, 40, np.inf]
        lab = ["< -40 (large cavity)", "-40 to -10", "-10 to 10", "10 to 40", "> 40 (overpacking)"]
        b = pd.cut(dv, bins, labels=lab)
        tv = pd.DataFrame({"dV_bin": b, "vol_attr": Xs[:, jv], "true": y, "pred": p, "abs_err": np.abs(y - p)}) \
            .groupby("dV_bin", observed=True).agg(n=("true", "size"), mean_volume_attr=("vol_attr", "mean"),
                                                   mean_true_ddg=("true", "mean"), mean_pred_mu=("pred", "mean"),
                                                   mae=("abs_err", "mean")).reset_index()
        T[f"volume_bidirectional_{name}"] = tv
        top = t.head(6)
        K["driving"][name] = dict(
            top_by_shift_true=[(f, _r(a), _r(lo), _r(hi)) for f, a, lo, hi in
                               zip(top.feature, top.shift_true, top.shift_true_lo, top.shift_true_hi)],
            top_by_rho_true=[(f, _r(r)) for f, r in t.reindex(t.rho_true.abs().sort_values(ascending=False).index)
                             [["feature", "rho_true"]].head(6).values],
            # rank correlations on features that carry >= 2 % of |attribution| (avoids tiny-magnitude artefacts)
            top_by_rho_true_material=[(f, _r(r), _r(sh)) for f, r, sh in
                                      t[t.share_abs >= 0.02].reindex(t[t.share_abs >= 0.02].rho_true.abs()
                                                                     .sort_values(ascending=False).index)
                                      [["feature", "rho_true", "share_abs"]].head(6).values],
            rank_agreement_shift_pred_vs_true=_r(spearmanr(t.shift_pred, t.shift_true, nan_policy="omit")[0]),
            n_pred_stab=int(ps.sum()), n_true_stab=int(ts.sum()),
        )


# ---------------------------------------------------------------------------- A2 environment
def a2_environment(v, A, g, T, K):
    X = A["ctx_mu"]["F_tot"]
    Xm = A["mut_mu"]["F_tot"]
    K["environment"] = {}
    for name, m in subsets(v):
        Xs, s = X[m], v[m]
        y, p, rsa = s.true_ddg.values, s.pred_mu.values, s.rsa_mean.values
        absX = np.abs(Xs)
        ctx_total = (s.f_ctx_mu - s.f0_ctx_mu).values
        t = pd.DataFrame({
            "feature": F, "group": [cfg.FEATURE_GROUP[f] for f in F],
            "mean_abs": absX.mean(0), "share_abs": absX.mean(0) / absX.mean(0).sum(),
            "mean_signed": Xs.mean(0), "rho_true": rho_cols(Xs, y), "rho_pred": rho_cols(Xs, p),
            "rho_rsa": rho_cols(Xs, rsa),
            "self_share_abs": np.abs(A["ctx_mu"]["F_self"][m]).sum(0) / np.maximum(absX.sum(0), 1e-12),
        }).sort_values("mean_abs", ascending=False).reset_index(drop=True)
        T[f"environment_{name}"] = t
        tb = s.assign(ctx_total=ctx_total, sasa_ctx=Xs[:, cfg.SASA_IDX]).groupby("rsa_class", observed=True).agg(
            n=("true_ddg", "size"), mean_true=("true_ddg", "mean"), mean_pred=("pred_mu", "mean"),
            mean_ctx_total=("ctx_total", "mean"), mean_sasa_ctx=("sasa_ctx", "mean"), mae=("abs_err", "mean"),
            mean_epistemic_std=("epistemic_std", "mean")).reset_index()
        T[f"environment_burial_{name}"] = tb
        mass_ratio = absX.sum(1) / np.maximum(np.abs(Xm[m]).sum(1), 1e-12)
        K["environment"][name] = dict(
            top_features_mean_abs=[(f, _r(a)) for f, a in t[["feature", "mean_abs"]].head(5).values],
            sasa_rank=int(t.index[t.feature == "SASA"][0]) + 1,
            ctx_to_mut_mass_ratio_median=_r(np.median(mass_ratio)),
            rho_ctx_total_vs_rsa=_r(spearmanr(ctx_total, rsa)[0]),
            rho_sasa_ctx_vs_rsa=_r(spearmanr(Xs[:, cfg.SASA_IDX], rsa)[0]),
            rho_ctx_total_vs_true=_r(spearmanr(ctx_total, y)[0]),
            rho_ctx_total_vs_pred=_r(spearmanr(ctx_total, p)[0]),
        )


# ---------------------------------------------------------------------------- A3 localisation
FAMILIES = ["mut_mu", "ctx_mu", "mut_epistemic", "ctx_epistemic"]
SHELL_NAMES = [s[0] for s in cfg.SHELLS]


def _shell_rows(A, m, fam, extra):
    M, n = A[fam]["mass_shell"][m], A[fam]["n_shell"][m]
    share = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    rows = []
    for j, sh in enumerate(SHELL_NAMES):
        has = n[:, j] > 0
        dens = share[has, j] / n[has, j]
        rows.append(dict(**extra, family=fam, shell=sh, mass_share=share[:, j].mean(),
                         nodes_per_variant=n[:, j].mean(), per_node_share=dens.mean() if has.any() else np.nan))
    return rows


def a3_localisation(v, A, g, T, K):
    K["localisation"] = {}
    for name, m in subsets(v):
        s = v[m]
        rows = []
        for fam in FAMILIES:
            rows += _shell_rows(A, m, fam, dict(subset=name, stratum="all"))
        T[f"localisation_{name}"] = pd.DataFrame(rows)
        # by uncertainty quartile and error quartile
        rows = []
        for col, lab in [("epistemic_std", "epistemic_std_Q"), ("abs_err", "abs_err_Q")]:
            q = pd.qcut(s[col], 4, labels=["Q1", "Q2", "Q3", "Q4"]).values
            for qq in ["Q1", "Q2", "Q3", "Q4"]:
                mm = np.zeros(len(v), bool)
                mm[np.where(m)[0][q == qq]] = True
                for fam in ["ctx_mu", "ctx_epistemic"]:
                    rows += _shell_rows(A, mm, fam, dict(subset=name, stratum=f"{lab}{qq[1]}"))
        T[f"localisation_by_quartile_{name}"] = pd.DataFrame(rows)
        # neighbour dX share vs neighbour attribution share (mutation IG)
        adn, ads = g["adx_neigh"][m], g["adx_self"][m]
        an, at = A["mut_mu"]["absF_neigh"][m], A["mut_mu"]["absF_tot"][m]
        rows = []
        for j, f in enumerate(F):
            ok = (adn[:, j] + ads[:, j]) > 1e-9
            if ok.sum() < 20:
                continue
            rows.append(dict(feature=f, group=cfg.FEATURE_GROUP[f], n_variants_changed=int(ok.sum()),
                             dx_share_neighbours=(adn[ok, j] / (adn[ok, j] + ads[ok, j])).mean(),
                             attr_share_neighbours=(an[ok, j] / np.maximum(at[ok, j], 1e-12)).mean()))
        tn = pd.DataFrame(rows)
        T[f"neighbour_dx_vs_attr_{name}"] = tn
        loc = T[f"localisation_{name}"]
        get = lambda fam, sh, c: float(loc[(loc.family == fam) & (loc.shell == sh)][c].iloc[0])
        mad = tn[tn.group == "MadraX energy"]
        K["localisation"][name] = dict(
            mut_mu_self_share=_r(get("mut_mu", "self", "mass_share")),
            mut_mu_top_node_is_mutated=_r(A["mut_mu"]["top_node_is_mut"][m].mean()),
            ctx_mu_self_share=_r(get("ctx_mu", "self", "mass_share")),
            ctx_mu_per_node_self_vs_gt9=_r(get("ctx_mu", "self", "per_node_share") /
                                           max(get("ctx_mu", ">9 A", "per_node_share"), 1e-12), 2),
            ctx_mu_per_node=[(sh, _r(get("ctx_mu", sh, "per_node_share"), 4)) for sh in SHELL_NAMES],
            mean_changed_neighbours=_r(s.n_changed_neigh.mean(), 2),
            madrax_dx_share_neighbours_mean=_r(mad.dx_share_neighbours.mean()) if len(mad) else None,
            madrax_attr_share_neighbours_mean=_r(mad.attr_share_neighbours.mean()) if len(mad) else None,
        )


# ---------------------------------------------------------------------------- A4 epistasis
def _d12_bin(d):
    return np.where(np.isnan(d), ">9 A", np.where(d < 6, "<6 A", np.where(d <= 9, "6-9 A", ">9 A")))


def a4_epistasis(v, A, g, T, K, ds):
    m = (v.n_mutated == 2).values
    if not m.any():
        return
    s = v[m].copy()
    s["d12_bin"] = _d12_bin(s.d12.values)
    s["sep_bin"] = np.where(s.seq_sep <= 4, "|i-j|<=4", "|i-j|>4")
    s["mu_no_epi"] = s.pred_mu - s.pred_gamma_epi
    res = {"n_doubles": int(len(s)),
           "mae_full": _r(s.abs_err.mean()), "mae_without_gamma_epi": _r((s.mu_no_epi - s.true_ddg).abs().mean()),
           "mean_gamma_epi": _r(s.pred_gamma_epi.mean()),
           "rho_gamma_epi_vs_residual_of_no_epi": _r(spearmanr(s.pred_gamma_epi, s.true_ddg - s.mu_no_epi)[0])}
    # experimental and predicted non-additivity (needs singles for both sites)
    exp_single, pred_single = {}, {}
    if ds == "megascale_test" and cfg.SINGLES_LOOKUP.exists():
        exp_single = json.load(open(cfg.SINGLES_LOOKUP))
    one = cfg.THREE_TO_ONE
    sv = v[v.n_mutated == 1]
    for pdb_, w, pos, mt, mu, mn in zip(sv.pdb, sv.wt1, sv.pos1, sv.mt1, sv.pred_mu, sv.mu_null):
        pred_single[f"{pdb_}::{one[w]}{pos}{one[mt]}"] = (mu, mn)
    k1 = [f"{a}::{one[w]}{p}{one[t]}" for a, w, p, t in zip(s.pdb, s.wt1, s.pos1, s.mt1)]
    k2 = [f"{a}::{one[w]}{p}{one[t]}" for a, w, p, t in zip(s.pdb, s.wt2, s.pos2, s.mt2)]
    s["add_exp"] = [exp_single[a] + exp_single[b] if a in exp_single and b in exp_single else np.nan
                    for a, b in zip(k1, k2)]
    s["add_pred"] = [pred_single[a][0] + pred_single[b][0] if a in pred_single and b in pred_single else np.nan
                     for a, b in zip(k1, k2)]
    s["add_pred_null"] = [pred_single[a][0] - pred_single[a][1] + pred_single[b][0] - pred_single[b][1]
                          if a in pred_single and b in pred_single else np.nan for a, b in zip(k1, k2)]
    s["E_exp"] = s.true_ddg - s.add_exp
    s["E_pred"] = s.pred_mu - s.add_pred
    s["E_pred_null"] = (s.pred_mu - s.mu_null) - s.add_pred_null
    rows = []
    groups = [("all", np.ones(len(s), bool))] + [(f"d12 {b}", (s.d12_bin == b).values) for b in ["<6 A", "6-9 A", ">9 A"]] \
        + [(b, (s.sep_bin == b).values) for b in ["|i-j|<=4", "|i-j|>4"]]
    have_E = s.E_exp.notna()
    for lab, mm in groups:
        r = dict(group=lab, n=int(mm.sum()), mean_gamma_epi=s.pred_gamma_epi[mm].mean(), mae=s.abs_err[mm].mean(),
                 mae_without_gamma_epi=(s.mu_no_epi - s.true_ddg)[mm].abs().mean())
        mE = mm & have_E.values
        r["n_matched"] = int(mE.sum())
        if mE.sum() >= 30:
            q = s[mE]
            r.update(mean_E_exp=q.E_exp.mean(), mean_E_pred=q.E_pred.mean(),
                     rho_E_exp_vs_additive=spearmanr(q.E_exp, q.add_exp)[0],
                     rho_E_pred_vs_E_exp=spearmanr(q.E_pred, q.E_exp)[0],
                     partial_rho_E_pred_vs_E_exp_given_add=partial_spearman(q.E_pred, q.E_exp, q.add_exp)[0],
                     rho_gamma_epi_vs_E_exp=spearmanr(q.pred_gamma_epi, q.E_exp)[0],
                     rho_gamma_epi_vs_additive=spearmanr(q.pred_gamma_epi, q.add_exp)[0],
                     partial_rho_gamma_epi_vs_E_exp_given_add=partial_spearman(q.pred_gamma_epi, q.E_exp, q.add_exp)[0])
        rows.append(r)
    te = pd.DataFrame(rows)
    T["epistasis"] = te
    s[["pdb", "mutation", "true_ddg", "pred_mu", "pred_gamma_epi", "mu_null", "d12", "seq_sep", "add_exp",
       "add_pred", "E_exp", "E_pred", "E_pred_null"]].to_csv(cfg.OUT_DIR / ds / "tables" / "epistasis_per_variant.csv",
                                                             index=False)
    # what drives gamma_epi (mutation IG on the epistasis head)
    Xe = A["mut_epi"]["F_tot"][m]
    T["epi_head_features"] = pd.DataFrame({
        "feature": F, "mean_abs": np.abs(Xe).mean(0), "mean_signed": Xe.mean(0),
        "rho_with_gamma_epi": rho_cols(Xe, s.pred_gamma_epi.values),
        "rho_with_E_exp": rho_cols(Xe, s.E_exp.values) if have_E.any() else np.nan,
    }).sort_values("mean_abs", ascending=False).reset_index(drop=True)
    res["groups"] = {r["group"]: {k: _r(val) for k, val in r.items() if k != "group"} for r in rows}
    res["n_matched_singles"] = int(have_E.sum())
    K["epistasis"] = res


# ---------------------------------------------------------------------------- A5 uncertainty
def _within_bin_rho(err, u, key, nb=10):
    b = pd.qcut(key, nb, labels=False, duplicates="drop")
    rs = [spearmanr(err[b == i], u[b == i])[0] for i in np.unique(b) if (b == i).sum() > 30]
    return float(np.nanmean(rs)) if rs else np.nan


def a5_uncertainty(v, A, g, T, K):
    K["uncertainty"] = {}
    rows = []
    for name, m in subsets(v):
        s = v[m]
        err, mu = s.abs_err.values, s.pred_mu.values
        r = dict(subset=name, n=int(m.sum()), mean_aleatoric=s.aleatoric_std.mean(),
                 mean_intra=s.intra_epistemic_std.mean(), mean_inter=s.inter_epistemic_std.mean(),
                 mean_epistemic=s.epistemic_std.mean(), median_v=s.pred_v.median(), rmse=np.sqrt((err ** 2).mean()),
                 aleatoric_var_fraction=(s.aleatoric_std ** 2 / s.total_std ** 2).mean())
        for col in ["epistemic_std", "intra_epistemic_std", "inter_epistemic_std", "aleatoric_std", "total_std"]:
            r[f"rho_err_{col}"] = spearmanr(err, s[col])[0]
            r[f"rho_err_{col}_within_absmu_deciles"] = _within_bin_rho(err, s[col].values, np.abs(mu))
        q = pd.qcut(s.epistemic_std, 4, labels=False)
        r["mae_ratio_Q4_vs_Q1_epistemic"] = err[q == 3].mean() / err[q == 0].mean()
        r["rho_absmu_epistemic"] = spearmanr(np.abs(mu), s.epistemic_std)[0]
        rows.append(r)
        Xm, Xc = A["mut_epistemic"]["F_tot"][m], A["ctx_epistemic"]["F_tot"][m]
        t = pd.DataFrame({
            "feature": F, "group": [cfg.FEATURE_GROUP[f] for f in F],
            "mut_mean_abs": np.abs(Xm).mean(0), "mut_mean_signed": Xm.mean(0),
            "ctx_mean_abs": np.abs(Xc).mean(0), "ctx_mean_signed": Xc.mean(0),
            "rho_mut_vs_abs_err": rho_cols(Xm, err), "rho_ctx_vs_abs_err": rho_cols(Xc, err),
            "rho_ctx_vs_rsa": rho_cols(Xc, s.rsa_mean.values),
        }).sort_values("ctx_mean_abs", ascending=False).reset_index(drop=True)
        T[f"uncertainty_attribution_{name}"] = t
        cm = np.abs(Xc).sum(1)
        mm_ = np.abs(Xm).sum(1)
        K["uncertainty"][name] = dict(
            {k: _r(val) for k, val in r.items() if k != "subset"},
            ctx_share_of_epistemic_attr_median=_r(np.median(cm / np.maximum(cm + mm_, 1e-12))),
            top_ctx_epistemic_features=[(f, _r(a)) for f, a in t[["feature", "ctx_mean_abs"]].head(5).values],
            top_mut_epistemic_features=[(f, _r(a)) for f, a in
                                        t.sort_values("mut_mean_abs", ascending=False)[["feature", "mut_mean_abs"]].head(5).values],
        )
    T["uncertainty_summary"] = pd.DataFrame(rows)


# ---------------------------------------------------------------------------- A6 per-protein uncertainty
def a6_uq_per_protein(v, A, g, T, K):
    K["uq_per_protein"] = {}
    for name, m in subsets(v):
        s = v[m].assign(serr=lambda d: d.pred_mu - d.true_ddg)
        rows = []
        for p, q in s.groupby("pdb"):
            if len(q) < 15:
                continue
            rows.append(dict(pdb=p, n=len(q), pearson=pearsonr(q.pred_mu, q.true_ddg)[0],
                             spearman=spearmanr(q.pred_mu, q.true_ddg)[0], mae=q.abs_err.mean(),
                             mean_signed_err=q.serr.mean(), mean_epistemic=q.epistemic_std.mean(),
                             rho_err_epistemic=spearmanr(q.abs_err, q.epistemic_std)[0],
                             rho_err_intra=spearmanr(q.abs_err, q.intra_epistemic_std)[0],
                             rho_err_inter=spearmanr(q.abs_err, q.inter_epistemic_std)[0]))
        t = pd.DataFrame(rows).sort_values("rho_err_epistemic", ascending=False)
        T[f"uq_per_protein_{name}"] = t
        ret = {}
        for frac in (0.8, 0.5, 0.2):
            pooled = s.epistemic_std <= s.epistemic_std.quantile(frac)
            within = s.groupby("pdb").epistemic_std.rank(pct=True) <= frac
            ret[f"keep_{int(frac * 100)}"] = dict(pooled_mae=_r(s.abs_err[pooled].mean()),
                                                  within_protein_mae=_r(s.abs_err[within].mean()))
        big = t[t.n >= 40]
        K["uq_per_protein"][name] = dict(
            n_proteins=int(len(big)), mean_within_rho=_r(big.rho_err_epistemic.mean()),
            sd_within_rho=_r(big.rho_err_epistemic.std()),
            n_positive=int((big.rho_err_epistemic > 0).sum()), n_above_0p2=int((big.rho_err_epistemic > 0.2).sum()),
            best=[(p, _r(r)) for p, r in big[["pdb", "rho_err_epistemic"]].head(4).values],
            worst=[(p, _r(r)) for p, r in big[["pdb", "rho_err_epistemic"]].tail(3).values],
            protein_r_sigma_vs_pearson=_r(pearsonr(t.mean_epistemic, t.pearson)[0]),
            protein_p_sigma_vs_pearson=_r(pearsonr(t.mean_epistemic, t.pearson)[1], 4),
            protein_r_sigma_vs_mae=_r(pearsonr(t.mean_epistemic, t.mae)[0]),
            share_err_var_from_protein_bias=_r(s.groupby("pdb").serr.transform("mean").var() / s.serr.var()),
            retention=ret, mean_epistemic=_r(s.epistemic_std.mean()),
        )


# ---------------------------------------------------------------------------- A7 Rashomon (seed agreement)
def _partial_multi(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([rankdata(c) for c in Z] + [np.ones(len(x))])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return pearsonr(ex, ey)[0]


def a7_rashomon(v, A, g, T, K, ds):
    rows, K["rashomon"] = [], {}
    for key in ("mut_mu", "ctx_mu"):
        p = cfg.CACHE_DIR / ds / f"seeds_{key}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        for name, m in subsets(v):
            s = v[m]
            Fs = z["F_seed"][:, m]
            M = Fs.shape[0]
            imp = np.abs(Fs).mean(1)
            y = s.true_ddg.values
            st = Fs[:, y > 1].mean(1) - Fs[:, y < 0].mean(1)
            agree_imp = np.mean([spearmanr(imp[a], imp[b])[0] for a in range(M) for b in range(a + 1, M)])
            agree_shift = np.mean([spearmanr(st[a], st[b])[0] for a in range(M) for b in range(a + 1, M)])
            for metric in ("cos_full", "cos_feat", "rank_feat", "cos_node"):
                x = z[metric][m]
                dis = 1 - x
                wp = [spearmanr(1 - x[(s.pdb == q).values], s.abs_err[s.pdb == q])[0]
                      for q in s.pdb.unique() if (s.pdb == q).sum() >= 40]
                rows.append(dict(
                    family=key, subset=name, metric=metric, median=np.median(x),
                    q25=np.quantile(x, .25), q75=np.quantile(x, .75),
                    rho_disagreement_vs_abs_err=spearmanr(dis, s.abs_err)[0],
                    rho_disagreement_vs_abs_mu=spearmanr(dis, s.pred_mu.abs())[0],
                    rho_disagreement_vs_sigma_inter=spearmanr(dis, s.inter_epistemic_std)[0],
                    partial_rho_vs_abs_err_given_sigma_and_mu=_partial_multi(
                        dis, s.abs_err.values, [s.epistemic_std, s.inter_epistemic_std, s.pred_mu.abs()]),
                    within_protein_median_rho=np.median(wp) if wp else np.nan,
                    dataset_level_importance_rank_agreement=agree_imp,
                    dataset_level_shift_rank_agreement=agree_shift))
    if rows:
        t = pd.DataFrame(rows)
        T["rashomon"] = t
        for (fam, sub), q in t[t.metric == "rank_feat"].groupby(["family", "subset"]):
            r = q.iloc[0]
            K["rashomon"][f"{fam}_{sub}"] = {c: _r(r[c]) for c in t.columns if c not in ("family", "subset", "metric")}


# ---------------------------------------------------------------------------- driver
def run(ds):
    out = cfg.OUT_DIR / ds / "tables"
    out.mkdir(parents=True, exist_ok=True)
    v, A, g = load(ds)
    T, K = {}, {"dataset": ds, "label": cfg.DATASET_LABEL[ds], "n": int(len(v)),
                "distance_source": str(g["d_source"])}
    a0_sanity(v, A, g, T, K)
    a1_driving(v, A, g, T, K)
    a2_environment(v, A, g, T, K)
    a3_localisation(v, A, g, T, K)
    a4_epistasis(v, A, g, T, K, ds)
    a5_uncertainty(v, A, g, T, K)
    a6_uq_per_protein(v, A, g, T, K)
    a7_rashomon(v, A, g, T, K, ds)
    for k, t in T.items():
        t.to_csv(out / f"{k}.csv", index=False, float_format="%.5g")
    with open(cfg.OUT_DIR / ds / "key_numbers.json", "w") as fh:
        json.dump(K, fh, indent=1, default=lambda o: _r(o) if isinstance(o, (float, np.floating)) else str(o))
    print(f"[analyses] {ds}: {len(T)} tables -> {out}")
    return K


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(cfg.DATASETS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for ds in (cfg.DATASETS if a.all else [a.dataset]):
        run(ds)
