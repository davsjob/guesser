"""Deep dive: what does the model attribute its EPISTEMIC uncertainty to? (MegaScale-S / -D, forward only)

    PYTHONPATH=. python deepdive_uncertainty_attribution.py            # analyses + figures (~60-90 s)
    PYTHONPATH=. python deepdive_uncertainty_attribution.py --figs     # figures only, from _fig_data.pkl
Needs cache/megascale_test/seed_scalars.npz and seeds_mut_epistemic.npz; both are (re)built here if missing
(seeds via `python extract.py --dataset megascale_test --stage seeds --key mut_epistemic`, ~30 s).
Outputs -> <OUT_DIR>/deepdive_uncertainty_attribution/ (tables/*.csv, key_numbers.json, fig18-fig22).

Target of the uncertainty IG: per seed s, g_s = log sigma_s = 0.5 log beta_s / (v_s (alpha_s - 1)); the ensemble
explanation is the seed mean, g = mean_s g_s (verified here). sigma_intra = sqrt(mean_s sigma_s^2), so g tracks
log sigma_intra (rho 0.99) but NOT sigma_inter (the seed spread of mu), which IG does not attribute.
Mutation IG is complete: sum_f A_f = g - g0, with g0 = the model's log sigma for the identity mutation at the same
site ("site baseline"). Environment IG: sum_f A_f = g - g0_ctx, g0_ctx = the same mutation in a featureless
(node-mean) environment.

A. What does the IG explain?  target vs sigma_intra / sigma_e / sigma_inter.
B. Where vs what: exact split var(g) = cov(g0, g) + cov(g - g0, g)  (site baseline vs mutation part), plus the
   environment view; error ranking of each sigma component at four scopes (pooled, within class x burial stratum,
   within site / site pair, between sites / site pairs).
C. Response curves of the sigma attributions (Delta feature vs attribution, split core / boundary / surface).
D. mu vs sigma attributions: variance shares, per-feature and per-variant sign agreement.
E. Useful vs useless uncertainty: does any attribution pattern flag errors beyond sigma itself?
F. Doubles: site pairs, distance d12, charge pairs, protein / pair-level bias.
G. The sigma_inter gap: exact per-variant decomposition of the seed variance of mu into features
   (per-seed IG is complete: mu_s = mu_null_s + sum_f A_sf  =>  var_s(mu_s) = cov_s(mu_null_s, mu_s) + sum_f
   cov_s(A_sf, mu_s)), and seed stability of the sigma explanations.
Attributions are model-internal, not causal.
"""
import json
import pickle
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, rankdata, spearmanr

import config as cfg
import deepdive_feature_direction as fd
import figures  # noqa: F401  (shared rcParams)
from deepdive_attribution_mechanisms import CHARGE, substitution_class
from deepdive_composition_bias import inv_count, wshares

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_uncertainty_attribution"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
N_BOOT = 300
BUR = ["core", "boundary", "surface"]
BCOL = {"core": C["red"], "boundary": C["yellow"], "surface": C["blue"]}
COL_MU, COL_INTRA, COL_INTER = C["blue"], C["orange"], C["violet"]
COL_SITE, COL_MUT = C["aqua"], C["orange"]
SHORT = {"Hydrophobicity": "Hydrophobicity", "Volume": "Volume", "VdW Clashes": "VdW clashes",
         "Polar Solvation": "Polar solv.", "Hydrophobic Solvation": "Hydrophobic solv.", "VdW Energy": "VdW energy",
         "Electrostatics": "Electrostatics", "Charge": "Charge", "Hydrogen Bonds": "H-bonds (energy)",
         "Cyclic": "Cyclic (Pro)", "Backbone Entropy": "Backbone entropy", "Entropy": "Side-chain entropy (aa)",
         "Aromatic": "Aromatic", "H_Bond_Donors": "H-bond donors", "H_Bond_Acceptors": "H-bond acceptors",
         "Rotamer Violation": "Rotamer viol.", "Branched": "Branched", "Sulfur": "Sulfur", "SASA": "SASA",
         "Polarizability": "Polarizability", "Side Chain Entropy": "Side-chain entropy (E)",
         "Peptide Bond Violations": "Peptide-bond viol.", "Disulfide bonds": "Disulfide"}
CURVE_FEATS = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Hydrophobic Solvation", "VdW Energy",
               "Electrostatics", "Charge", "Hydrogen Bonds", "Cyclic", "Backbone Entropy", "Entropy", "Aromatic"]
K = {}
FD = {}  # data for figures


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


# ---------------------------------------------------------------------------- data
def ensure_cache():
    cd = cfg.CACHE_DIR / DS
    if not (cd / "seeds_mut_epistemic.npz").exists():
        subprocess.run([sys.executable, "extract.py", "--dataset", DS, "--stage", "seeds", "--key", "mut_epistemic"],
                       check=True, cwd=cfg.PKG_DIR)
    if not (cd / "seed_scalars.npz").exists():
        import h5py
        f = h5py.File(cfg.EXPL_DIR / cfg.DATASETS[DS], "r")
        d = {k: f["prediction"][k][:] for k in ("per_member_mu", "per_member_v", "per_member_alpha", "per_member_beta")}
        for k in ("f_mut_mu", "f0_mut_mu", "f_mut_epistemic", "f0_mut_epistemic", "f_ctx_mu", "f0_ctx_mu",
                  "f_ctx_epistemic", "f0_ctx_epistemic", "completeness_err_mut_mu"):
            d["seed_" + k] = f["ig_per_seed"][k][:]
        np.savez_compressed(cd / "seed_scalars.npz", **d)


def load():
    cd = cfg.CACHE_DIR / DS
    v = pd.read_csv(cd / "variants.csv")
    A = {k: np.load(cd / f"attr_{k}.npz")["F_tot"] for k in ("mut_mu", "ctx_mu", "mut_epistemic", "ctx_epistemic")}
    DX = np.load(cd / "geom.npz")["dx_all"]
    S = dict(np.load(cd / "seed_scalars.npz"))
    SE = {k: np.load(cd / f"seeds_{k}.npz") for k in ("mut_mu", "ctx_mu", "mut_epistemic")}
    v["cls"] = [substitution_class(w, m) for w, m in zip(v.wt1, v.mt1)]
    v["cls2"] = [substitution_class(w, m) if isinstance(w, str) else "" for w, m in zip(v.wt2, v.mt2)]
    v["g"] = v.f_mut_epistemic                       # ensemble IG target: mean_s log sigma_s
    v["g0"] = v.f0_mut_epistemic                     # site baseline (identity mutation)
    v["gm"] = v.g - v.g0                             # mutation part (= sum of mutation-IG attributions)
    v["tmu"] = v.pred_mu - v.mu_null
    v["serr"] = v.pred_mu - v.true_ddg
    v["site"] = v.pdb + "_" + v.pos1.astype(str) + np.where(v.n_mutated == 2, "_" + v.pos2.astype(str), "")
    v["stratum"] = np.where(v.n_mutated == 1, v.cls, v.cls + "+" + v.cls2) + "|" + v.rsa_class
    v["ctx_sum"] = A["ctx_epistemic"].sum(1)
    v["ctx_sasa"] = A["ctx_epistemic"][:, cfg.SASA_IDX]
    return v, A, DX, S, SE


def subsets(v):
    return (("singles", (v.n_mutated == 1).values), ("doubles", (v.n_mutated == 2).values))


def boot_idx(pdb, n=N_BOOT):
    pdb = np.asarray(pdb)
    u = np.unique(pdb)
    by = {p: np.where(pdb == p)[0] for p in u}
    for _ in range(n):
        yield np.concatenate([by[p] for p in RNG.choice(u, len(u))])


def ci(vals):
    vals = np.asarray(vals, float)
    return [r3(np.nanpercentile(vals, 2.5)), r3(np.nanpercentile(vals, 97.5))]


def srho(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def strat_rank(x, st):
    """Global ranks, centred within stratum."""
    r = rankdata(x).astype(float)
    return r - pd.Series(r).groupby(np.asarray(st)).transform("mean").values


def strat_rho(a, b, st):
    return pearsonr(strat_rank(a, st), strat_rank(b, st))[0]


def strat_partial(a, b, Z, st):
    ea, eb = strat_rank(a, st), strat_rank(b, st)
    Zr = np.column_stack([strat_rank(z, st) for z in Z])
    ea = ea - Zr @ np.linalg.lstsq(Zr, ea, rcond=None)[0]
    eb = eb - Zr @ np.linalg.lstsq(Zr, eb, rcond=None)[0]
    return pearsonr(ea, eb)[0]


def within_group_rho(a, b, g, min_n):
    df = pd.DataFrame(dict(a=np.asarray(a), b=np.asarray(b), g=np.asarray(g)))
    r = []
    for _, q in df.groupby("g"):
        if len(q) >= min_n and q.a.std() > 0 and q.b.std() > 1e-6:
            r.append(srho(q.a, q.b))
    r = np.array(r)
    return (np.nan, np.nan, 0) if len(r) == 0 else (float(r.mean()), float((r > 0).mean()), len(r))


# ---------------------------------------------------------------------------- A. what the IG explains
def target_check(v, S):
    ls = 0.5 * np.log(S["per_member_beta"] / (S["per_member_v"] * (S["per_member_alpha"] - 1)))
    out = dict(max_abs_diff_target_vs_mean_seed_log_sigma=float(np.abs(ls.mean(0) - v.g).max()),
               max_abs_diff_sigma_intra_vs_rms_seed_sigma=float(np.abs(np.sqrt(np.exp(2 * ls).mean(0))
                                                                     - v.intra_epistemic_std).max()))
    for sub, m in subsets(v):
        x = v[m]
        out[sub] = dict(
            rho_target_log_sigma_intra=r3(srho(x.g, x.intra_epistemic_std)),
            rho_target_log_sigma_e=r3(srho(x.g, x.epistemic_std)),
            rho_target_sigma_inter=r3(srho(x.g, x.inter_epistemic_std)),
            inter_share_of_sigma_e2=r3((x.inter_epistemic_std ** 2).sum() / (x.epistemic_std ** 2).sum()),
            max_completeness_err=r3(np.abs(x.completeness_err_mut_epistemic).max()))
    K["A_target"] = out


# ---------------------------------------------------------------------------- B. where vs what
COMPONENTS = {"σ_e (total)": "epistemic_std", "site baseline g0": "g0", "mutation part g−g0": "gm",
              "σ_inter (ensemble)": "inter_epistemic_std", "σ_intra (NIG)": "intra_epistemic_std",
              "environment IG sum": "ctx_sum", "environment IG, SASA": "ctx_sasa"}


def where_vs_what(v, A):
    res, rows = {}, []
    for sub, m in subsets(v):
        x = v[m].reset_index(drop=True)
        g, g0, gm = x.g.values, x.g0.values, x.gm.values
        site = np.cov(g0, g, bias=True)[0, 1] / g.var()
        bs = [np.cov(g0[i], g[i], bias=True)[0, 1] / g[i].var() for i in boot_idx(x.pdb, 500)]
        Ac = A["ctx_epistemic"][m]
        ctx_sh = pd.Series(((Ac - Ac.mean(0)) * (g - g.mean())[:, None]).mean(0) / g.var(), index=F)
        f0c = x.f0_ctx_epistemic.values
        res[sub] = dict(
            share_var_g_site_baseline=r3(site), share_var_g_site_baseline_ci=ci(bs),
            share_var_g_mutation_part=r3(1 - site),
            eta2_g_by_site=r3(np.var(pd.Series(g).groupby(x.site).transform("mean")) / g.var()),
            eta2_g_by_protein=r3(np.var(pd.Series(g).groupby(x.pdb).transform("mean")) / g.var()),
            env_view_share_featureless_baseline=r3(np.cov(f0c, g, bias=True)[0, 1] / g.var()),
            env_view_top=[(f, r3(s)) for f, s in ctx_sh.sort_values(ascending=False).head(4).items()],
            rho_g0_rsa=r3(srho(g0, x.rsa_mean)), rho_gm_rsa=r3(srho(gm, x.rsa_mean)),
            rho_g0_absmu=r3(srho(g0, x.pred_mu.abs())), rho_gm_absmu=r3(srho(gm, x.pred_mu.abs())),
            rho_g_absmu=r3(srho(g, x.pred_mu.abs())),
            mean_gm=r3(gm.mean()), mean_sigma_by_burial=x.groupby("rsa_class").epistemic_std.mean().round(3).to_dict(),
            mean_g0_by_burial=x.groupby("rsa_class").g0.mean().round(3).to_dict(),
            mean_gm_by_burial=x.groupby("rsa_class").gm.mean().round(3).to_dict())
        # error ranking of each component at four scopes
        e = x.abs_err.values
        min_n = 5 if sub == "singles" else 20
        for lab, col in COMPONENTS.items():
            c = x[col].values
            bst = [strat_rho(e[i], c[i], x.stratum.values[i]) for i in boot_idx(x.pdb, 150)]
            wg = within_group_rho(e, c, x.site, min_n)
            site_means = x.groupby("site").agg(e=("abs_err", "mean"), c=(col, "mean"))
            wp = within_group_rho(e, c, x.pdb, 20)
            rows.append(dict(subset=sub, component=lab, pooled=srho(e, c),
                             within_stratum=strat_rho(e, c, x.stratum.values),
                             within_stratum_lo=np.nanpercentile(bst, 2.5), within_stratum_hi=np.nanpercentile(bst, 97.5),
                             partial_within_stratum_given_sigma_absmu=np.nan if col == "epistemic_std" else
                             strat_partial(e, c, [x.epistemic_std.values, x.pred_mu.abs().values], x.stratum.values),
                             within_site=wg[0], within_site_frac_pos=wg[1], n_sites_used=wg[2],
                             between_sites=srho(site_means.e, site_means.c), n_sites=len(site_means),
                             within_protein=wp[0], within_protein_frac_pos=wp[1]))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "error_ranking_by_component.csv", index=False, float_format="%.4g")
    K["B_where_vs_what"] = res
    K["B_error_ranking"] = {f"{r.subset} | {r.component}": dict(
        pooled=r3(r.pooled), within_stratum=r3(r.within_stratum), ci=[r3(r.within_stratum_lo), r3(r.within_stratum_hi)],
        partial_given_sigma_absmu=r3(r.partial_within_stratum_given_sigma_absmu),
        within_site=r3(r.within_site), within_site_frac_pos=r3(r.within_site_frac_pos), n_sites_used=int(r.n_sites_used),
        between_sites=r3(r.between_sites), n_sites=int(r.n_sites), within_protein=r3(r.within_protein))
        for r in T.itertuples()}
    # singles: within a site, does the mutation part add to |mu|?
    s = v[v.n_mutated == 1]
    part = []
    for _, q in s.groupby("site"):
        if len(q) >= 8 and q.gm.std() > 0:
            ea, eb, z = rankdata(q.abs_err), rankdata(q.gm), rankdata(q.pred_mu.abs())
            Z = np.column_stack([z, np.ones(len(q))])
            ra = ea - Z @ np.linalg.lstsq(Z, ea, rcond=None)[0]
            rb = eb - Z @ np.linalg.lstsq(Z, eb, rcond=None)[0]
            if ra.std() > 0 and rb.std() > 0:
                part.append(np.corrcoef(ra, rb)[0, 1])
    K["B_error_ranking"]["singles | within-site partial rho(|err|, mutation part | |mu|)"] = dict(
        mean=r3(np.mean(part)), frac_pos=r3(np.mean(np.array(part) > 0)), n_sites=len(part))
    FD["B"] = dict(T=T, res=res)


# ---------------------------------------------------------------------------- C. response curves
def shape_sigma(dn, up, thr=0.03):
    if np.isnan(dn) or np.isnan(up):
        x = up if np.isnan(dn) else dn
        if np.isnan(x):
            return "n/a"
        return "one-sided: raises σ" if x > thr else "one-sided: lowers σ" if x < -thr else "weak"
    if dn > thr and up > thr:
        return "any change raises σ (V)"
    if dn < -thr and up < -thr:
        return "any change lowers σ"
    if (dn > thr and up < -thr / 2) or (dn < -thr / 2 and up > thr):
        return "directional"
    if max(dn, up) > thr:
        return "one-sided: raises σ"
    if min(dn, up) < -thr:
        return "one-sided: lowers σ"
    return "weak"


def response_curves(v, A, DX):
    curves, summ = [], []
    for sub, m in subsets(v):
        x = v[m].reset_index(drop=True)
        As, Am, D = A["mut_epistemic"][m], A["mut_mu"][m], DX[m]
        pdb = x.pdb.values
        for f in CURVE_FEATS:
            j, eps = F.index(f), fd.EPS[f]
            dx = D[:, j]
            lab = fd.assign(dx, eps, fd.bins_for(dx, eps))
            df = pd.DataFrame(dict(x=dx, s=As[:, j], mu=Am[:, j], lab=lab, bur=x.rsa_class, pdb=pdb))
            gq = df.groupby(["bur", "lab"]).agg(n=("s", "size"), dx=("x", "mean"), sig_attr=("s", "mean"),
                                                mu_attr=("mu", "mean"), n_prot=("pdb", "nunique"))
            gq = gq[gq.n >= fd.MIN_N].reset_index()
            gq["feature"], gq["subset"] = f, sub
            curves.append(gq)
            df["side"] = np.select([df.x < -eps, df.x > eps], ["decrease", "increase"], "none")
            for b in BUR + ["all"]:
                d = df if b == "all" else df[df.bur == b]
                row = dict(subset=sub, feature=f, burial=b)
                for side in ("decrease", "increase"):
                    q = d[d.side == side]
                    ok = len(q) >= fd.MIN_N and q.pdb.nunique() >= 3
                    row[f"n_{side}"] = len(q)
                    row[f"sig_{side}"] = q.s.mean() if ok else np.nan
                    row[f"mu_{side}"] = q.mu.mean() if ok else np.nan
                    if ok:  # protein bootstrap from per-protein sums
                        agg = q.groupby("pdb").s.agg(["sum", "size"])
                        P = len(agg)
                        idx = RNG.integers(0, P, (400, P))
                        bm = agg["sum"].values[idx].sum(1) / agg["size"].values[idx].sum(1)
                        row[f"sig_{side}_lo"], row[f"sig_{side}_hi"] = np.percentile(bm, [2.5, 97.5])
                    else:
                        row[f"sig_{side}_lo"] = row[f"sig_{side}_hi"] = np.nan
                row["shape_sigma"] = shape_sigma(row["sig_decrease"], row["sig_increase"])
                summ.append(row)
    cur = pd.concat(curves, ignore_index=True)
    sm = pd.DataFrame(summ)
    cur.to_csv(TAB / "response_curves_sigma.csv", index=False, float_format="%.4g")
    sm.to_csv(TAB / "direction_summary_sigma.csv", index=False, float_format="%.4g")
    K["C_direction"] = {f"{r.subset} | {r.feature} | {r.burial}": dict(
        loss=r3(r.sig_decrease), loss_ci=[r3(r.sig_decrease_lo), r3(r.sig_decrease_hi)],
        gain=r3(r.sig_increase), gain_ci=[r3(r.sig_increase_lo), r3(r.sig_increase_hi)],
        n_loss=int(r.n_decrease), n_gain=int(r.n_increase), shape=r.shape_sigma,
        mu_loss=r3(r.mu_decrease), mu_gain=r3(r.mu_increase))
        for r in sm.itertuples() if r.feature in CURVE_FEATS}
    K["C_shape_counts"] = {sub: sm[(sm.subset == sub) & (sm.burial != "all")].shape_sigma.value_counts().to_dict()
                           for sub in ("singles", "doubles")}
    FD["C"] = dict(cur=cur, sm=sm)


# ---------------------------------------------------------------------------- D. mu vs sigma
def mu_vs_sigma(v, A, DX):
    out, share_tabs, rho_rows = {}, {}, []
    for sub, m in subsets(v):
        x = v[m].reset_index(drop=True)
        Am, As, D = A["mut_mu"][m], A["mut_epistemic"][m], DX[m]
        grp = x.cls.values if sub == "singles" else (x.cls + "+" + x.cls2).values
        T = pd.DataFrame({
            "mu_pooled": wshares(Am, x.tmu.values), "mu_class_bal": wshares(Am, x.tmu.values, inv_count(grp)),
            "sigma_pooled": wshares(As, x.gm.values), "sigma_class_bal": wshares(As, x.gm.values, inv_count(grp)),
            "sigma_share_of_total_var_g": ((As - As.mean(0)) * (x.g.values - x.g.mean())[:, None]).mean(0) / x.g.var(),
            "mu_abs_mass": np.abs(Am).mean(0) / np.abs(Am).sum(1).mean(),
            "sigma_abs_mass": np.abs(As).mean(0) / np.abs(As).sum(1).mean()}, index=F)
        for j, f in enumerate(F):  # sign agreement per feature (variants where the feature changes)
            ch = np.abs(D[:, j]) > fd.EPS[f]
            if ch.sum() < 200:
                continue
            within = []
            for c in np.unique(grp):
                i = ch & (grp == c)
                if i.sum() >= 100 and Am[i, j].std() > 1e-6 and As[i, j].std() > 1e-6:
                    within.append(srho(Am[i, j], As[i, j]))
            bs = []
            for i in boot_idx(x.pdb.values[ch], 150):
                bs.append(srho(Am[ch][i, j], As[ch][i, j]))
            rho_rows.append(dict(subset=sub, feature=f, n=int(ch.sum()), rho_pooled=srho(Am[ch, j], As[ch, j]),
                                 rho_lo=np.nanpercentile(bs, 2.5), rho_hi=np.nanpercentile(bs, 97.5),
                                 rho_mean_within_class=np.mean(within) if within else np.nan, n_classes=len(within),
                                 mu_share=T.loc[f, "mu_class_bal"], sigma_share=T.loc[f, "sigma_class_bal"]))
        share_tabs[sub] = T
        T.to_csv(TAB / f"shares_mu_vs_sigma_{sub}.csv", float_format="%.4g")
        cos = (Am * As).sum(1) / (np.linalg.norm(Am, axis=1) * np.linalg.norm(As, axis=1) + 1e-12)
        cabs = (np.abs(Am) * np.abs(As)).sum(1) / (np.linalg.norm(Am, axis=1) * np.linalg.norm(As, axis=1) + 1e-12)
        tm, ts = np.abs(Am).argmax(1), np.abs(As).argmax(1)
        chance = np.mean([np.mean([(tm[i] == ts[i][RNG.permutation(i.sum())]).mean() for i in
                                   [grp == c for c in np.unique(grp)]]) for _ in range(3)])
        chance_w = sum((tm[grp == c] == ts[grp == c][RNG.permutation((grp == c).sum())]).sum() for c in np.unique(grp)) / len(grp)
        out[sub] = dict(
            rank_corr_pooled_shares=r3(spearmanr(T.mu_pooled, T.sigma_pooled)[0]),
            rank_corr_class_bal_shares=r3(spearmanr(T.mu_class_bal, T.sigma_class_bal)[0]),
            rank_corr_abs_mass=r3(spearmanr(T.mu_abs_mass, T.sigma_abs_mass)[0]),
            per_variant_signed_cos_median=r3(np.median(cos)), frac_signed_cos_pos=r3((cos > 0).mean()),
            per_variant_abs_cos_median=r3(np.median(cabs)),
            top_feature_agreement=r3((tm == ts).mean()), top_feature_agreement_chance_within_class=r3(chance_w),
            top_mu=pd.Series(np.array(F)[tm]).value_counts(normalize=True).head(4).round(3).to_dict(),
            top_sigma=pd.Series(np.array(F)[ts]).value_counts(normalize=True).head(4).round(3).to_dict(),
            rho_gm_tmu=r3(srho(x.gm, x.tmu)), rho_g_mu=r3(srho(x.g, x.pred_mu)),
            sigma_top6_pooled=[(f, r3(s)) for f, s in T.sigma_pooled.sort_values(ascending=False).head(6).items()],
            sigma_top6_class_bal=[(f, r3(s)) for f, s in T.sigma_class_bal.sort_values(ascending=False).head(6).items()],
            mu_top6_class_bal=[(f, r3(s)) for f, s in T.mu_class_bal.sort_values(ascending=False).head(6).items()])
    R = pd.DataFrame(rho_rows)
    R.to_csv(TAB / "mu_sigma_sign_agreement.csv", index=False, float_format="%.4g")
    K["D_mu_vs_sigma"] = out
    K["D_sign_agreement"] = {f"{r.subset} | {r.feature}": dict(
        rho=r3(r.rho_pooled), ci=[r3(r.rho_lo), r3(r.rho_hi)], rho_within_class=r3(r.rho_mean_within_class),
        n=int(r.n)) for r in R.itertuples()}
    FD["D"] = dict(R=R, shares=share_tabs)


# ---------------------------------------------------------------------------- E. useful vs useless
def useful(v, A):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out, feat_rows, grp_rows = {}, [], []
    for sub, m in subsets(v):
        x = v[m].reset_index(drop=True)
        e, se, st = x.abs_err.values, x.serr.values, x.stratum.values
        Z = [x.epistemic_std.values, x.pred_mu.abs().values]
        for blk in ("mut_epistemic", "ctx_epistemic"):
            for j, f in enumerate(F):
                a = A[blk][m][:, j]
                if a.std() < 1e-6:
                    continue
                feat_rows.append(dict(subset=sub, family="mutation IG" if blk.startswith("mut") else "environment IG",
                                      feature=f, rho_within_stratum=strat_rho(e, a, st),
                                      partial_abs_err=strat_partial(e, a, Z, st),
                                      partial_signed_err=strat_partial(se, a, Z, st)))
        # high-sigma (top quartile within protein): wrong (|err|>1) vs right (|err|<0.5)
        hi = (x.groupby("pdb").epistemic_std.rank(pct=True) > 0.75).values
        lab = np.select([e > 1.0, e < 0.5], [1, 0], -1)
        sel = hi & (lab >= 0)
        h = x[sel].assign(y=lab[sel], inter_frac=lambda q: q.inter_epistemic_std ** 2 / q.epistemic_std ** 2,
                          absmu=lambda q: q.pred_mu.abs())
        for yv, name in ((1, "wrong (|err|>1)"), (0, "right (|err|<0.5)")):
            q = h[h.y == yv]
            grp_rows.append(dict(subset=sub, group=name, n=len(q), sigma_e=q.epistemic_std.mean(),
                                 sigma_inter=q.inter_epistemic_std.mean(), sigma_intra=q.intra_epistemic_std.mean(),
                                 inter_frac=q.inter_frac.mean(), site_g0=q.g0.mean(), mutation_gm=q.gm.mean(),
                                 abs_mu=q.absmu.mean(), pred=q.pred_mu.mean(), true=q.true_ddg.mean(),
                                 frac_underpredicted=(q.serr < 0).mean(), core_frac=(q.rsa_class == "core").mean()))
        y = lab[sel]
        auc_raw = {c: r3(roc_auc_score(y, h[c])) for c in ("epistemic_std", "inter_epistemic_std",
                                                           "intra_epistemic_std", "g0", "gm", "ctx_sasa")}
        base = np.column_stack([np.log(h.epistemic_std), np.log(h.inter_epistemic_std), h.pred_mu, h.rsa_mean,
                                h.g0, h.gm])
        oh = pd.get_dummies(h.stratum).values.astype(float)
        sets = {"σ_e only": np.log(h.epistemic_std.values)[:, None],
                "+ σ parts, μ, burial": base,
                "+ class×burial": np.column_stack([base, oh]),
                "+ σ attributions (46)": np.column_stack([base, oh, A["mut_epistemic"][m][sel], A["ctx_epistemic"][m][sel]]),
                "+ μ attributions (23)": np.column_stack([base, oh, A["mut_epistemic"][m][sel], A["ctx_epistemic"][m][sel],
                                                          A["mut_mu"][m][sel]])}
        aucs = {}
        for nm, X in sets.items():
            p = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, h.pdb.values):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
                p[te] = mdl.predict_proba(X[te])[:, 1]
            aucs[nm] = r3(roc_auc_score(y, p))
        out[sub] = dict(n_high_sigma=int(hi.sum()), n_wrong=int((y == 1).sum()), n_right=int((y == 0).sum()),
                        auc_single_quantity_in_high_sigma=auc_raw, auc_protein_grouped_cv=aucs)
    Fr = pd.DataFrame(feat_rows)
    Fr.to_csv(TAB / "useful_uncertainty_feature_partials.csv", index=False, float_format="%.4g")
    G = pd.DataFrame(grp_rows)
    G.to_csv(TAB / "high_sigma_wrong_vs_right.csv", index=False, float_format="%.4g")
    for sub in ("singles", "doubles"):
        q = Fr[Fr.subset == sub].assign(a=lambda d: d.partial_abs_err.abs()).sort_values("a", ascending=False)
        out[sub]["max_abs_partial"] = r3(q.a.iloc[0])
        out[sub]["top_partials"] = [(f"{r.family}: {r.feature}", r3(r.partial_abs_err), r3(r.partial_signed_err))
                                    for r in q.head(6).itertuples()]
        out[sub]["groups"] = G[G.subset == sub].drop(columns="subset").round(3).to_dict("records")
    K["E_useful"] = out
    FD["E"] = dict(Fr=Fr, G=G, out=out)


# ---------------------------------------------------------------------------- F. doubles
def doubles(v, A):
    d = v[v.n_mutated == 2].reset_index(drop=True)
    m = (v.n_mutated == 2).values
    s = v[v.n_mutated == 1]
    out = dict(n_site_pairs=int(d.site.nunique()), median_variants_per_pair=float(d.groupby("site").size().median()),
               pairs_per_protein=d.groupby("pdb").site.nunique().describe()[["min", "50%", "max"]].to_dict(),
               n_sites_singles=int(s.site.nunique()))
    eta = []
    for lab, col in [("signed error", "serr"), ("|error|", "abs_err"), ("experimental ΔΔG", "true_ddg"),
                     ("σ_e", "epistemic_std"), ("σ_inter", "inter_epistemic_std"), ("σ_intra", "intra_epistemic_std"),
                     ("mutation part g−g0", "gm")]:
        for sub, x in (("doubles", d), ("singles", s)):
            y = x[col]
            eta.append(dict(quantity=lab, subset=sub,
                            by_site=np.var(y.groupby(x.site).transform("mean")) / np.var(y),
                            by_protein=np.var(y.groupby(x.pdb).transform("mean")) / np.var(y)))
    E = pd.DataFrame(eta)
    E.to_csv(TAB / "eta2_site_vs_protein.csv", index=False, float_format="%.4g")
    out["eta2"] = {f"{r.subset} | {r.quantity}": dict(by_site=r3(r.by_site), by_protein=r3(r.by_protein))
                   for r in E.itertuples()}
    P = d.groupby("site").agg(pdb=("pdb", "first"), d12=("d12", "first"), n=("d12", "size"),
                              sigma_e=("epistemic_std", "mean"), sigma_inter=("inter_epistemic_std", "mean"),
                              sigma_intra=("intra_epistemic_std", "mean"), g0=("g0", "mean"), gm=("gm", "mean"),
                              mae=("abs_err", "mean"), bias=("serr", "mean"), true=("true_ddg", "mean"),
                              rsa=("rsa_mean", "mean"))
    P.to_csv(TAB / "doubles_site_pairs.csv", float_format="%.4g")
    out["pair_level"] = {f"rho(d12, {c})": r3(srho(P.d12, P[c])) for c in ("sigma_e", "sigma_inter", "sigma_intra", "gm", "mae", "bias")}
    for c in ("sigma_e", "sigma_inter", "sigma_intra", "g0", "gm"):
        out["pair_level"][f"rho(MAE, {c})"] = r3(srho(P.mae, P[c]))
        out["pair_level"][f"rho(|bias|, {c})"] = r3(srho(P.bias.abs(), P[c]))
    wp = {c: within_group_rho(d.abs_err, d[c], d.site, 20) for c in ("epistemic_std", "inter_epistemic_std", "gm")}
    out["within_pair_rho_abs_err"] = {c: dict(mean=r3(a), frac_pos=r3(b), n=n) for c, (a, b, n) in wp.items()}
    d["serr_c"] = d.serr - d.groupby("site").serr.transform("mean")
    out["mae"] = r3(d.abs_err.mean())
    out["mae_after_removing_pair_bias"] = r3(d.serr_c.abs().mean())
    out["mae_after_removing_protein_bias"] = r3((d.serr - d.groupby("pdb").serr.transform("mean")).abs().mean())
    # regimes
    reg = []
    for lo, hi, name in ((-99, 1, "ΔΔG_exp < 1"), (1, 3, "1–3"), (3, 99, "> 3")):
        q = d[(d.true_ddg >= lo) & (d.true_ddg < hi)]
        reg.append(dict(regime=name, n=len(q), mean_signed_err=q.serr.mean(), sigma_e=q.epistemic_std.mean(),
                        sigma_inter=q.inter_epistemic_std.mean(), gm=q.gm.mean()))
    out["regimes"] = pd.DataFrame(reg).round(3).to_dict("records")
    # distance / packing / charge pairs
    d["dbin"] = pd.cut(d.d12, [0, 6, 9, 999], labels=["<6 Å", "6–9 Å", ">9 Å"])
    w1, w2 = d.wt1.map(CHARGE).fillna(0).values, d.wt2.map(CHARGE).fillna(0).values
    m1, m2 = d.mt1.map(CHARGE).fillna(0).values, d.mt2.map(CHARGE).fillna(0).values
    wt_op, mt_op, wt_same, mt_same = w1 * w2 < 0, m1 * m2 < 0, w1 * w2 > 0, m1 * m2 > 0
    d["cpair"] = np.select([mt_op & ~wt_op, wt_op & ~mt_op, mt_same & ~wt_same, wt_same & ~mt_same],
                           ["opposite charges introduced", "opposite-charge pair removed", "like charges introduced",
                            "like-charge pair removed"], "no charge-pair change")
    d["elec_sig"] = A["mut_epistemic"][m][:, F.index("Electrostatics")]
    d["elec_mu"] = A["mut_mu"][m][:, F.index("Electrostatics")]
    cp = d.groupby(["cpair", "dbin"], observed=True).agg(
        n=("serr", "size"), n_pairs=("site", "nunique"), sigma_e=("epistemic_std", "mean"),
        sigma_inter=("inter_epistemic_std", "mean"), gm=("gm", "mean"), elec_sigma_attr=("elec_sig", "mean"),
        elec_mu_attr=("elec_mu", "mean"), mae=("abs_err", "mean")).reset_index()
    cp.to_csv(TAB / "doubles_charge_pairs_sigma.csv", index=False, float_format="%.4g")
    # within-pair comparison (removes the site-pair confound): charge-pair type vs "no change" in the same pair
    wp_rows = []
    for t in ("opposite-charge pair removed", "like-charge pair removed", "opposite charges introduced",
              "like charges introduced"):
        diffs = []
        for _, q in d.groupby("site"):
            a, b = q[q.cpair == t], q[q.cpair == "no charge-pair change"]
            if len(a) >= 5 and len(b) >= 5:
                diffs.append((a.epistemic_std.mean() - b.epistemic_std.mean(), a.elec_sig.mean() - b.elec_sig.mean(),
                              len(a)))
        if diffs:
            dd = np.array(diffs)
            wp_rows.append(dict(type=t, n_pairs=len(dd), n_variants=int(dd[:, 2].sum()),
                                dsigma_e=np.average(dd[:, 0], weights=dd[:, 2]), frac_pairs_lower=(dd[:, 0] < 0).mean(),
                                d_elec_sigma_attr=np.average(dd[:, 1], weights=dd[:, 2])))
    out["charge_pairs_within_pair"] = pd.DataFrame(wp_rows).round(3).to_dict("records")
    out["charge_pairs"] = cp.round(3).to_dict("records")
    ex = d.groupby("dbin", observed=True).agg(n=("serr", "size"), n_pairs=("site", "nunique"),
                                              sigma_e=("epistemic_std", "mean"), sigma_inter=("inter_epistemic_std", "mean"),
                                              gm=("gm", "mean"), mae=("abs_err", "mean"))
    out["by_d12_bin"] = ex.round(3).reset_index().astype({"dbin": str}).to_dict("records")
    # protein level
    Pp = d.groupby("pdb").agg(bias=("serr", "mean"), mae=("abs_err", "mean"), sigma_e=("epistemic_std", "mean"),
                              sigma_inter=("inter_epistemic_std", "mean"))
    out["protein_level"] = {f"rho(|bias|, {c})": r3(srho(Pp.bias.abs(), Pp[c])) for c in ("sigma_e", "sigma_inter")}
    out["protein_level"].update({f"rho(MAE, {c})": r3(srho(Pp.mae, Pp[c])) for c in ("sigma_e", "sigma_inter")})
    K["F_doubles"] = out
    FD["F"] = dict(P=P, E=E, cp=cp, d=d[["site", "pdb", "serr", "epistemic_std", "inter_epistemic_std", "true_ddg",
                                          "pred_mu", "abs_err"]])


# ---------------------------------------------------------------------------- G. sigma_inter
def sigma_inter(v, A, S, SE):
    mu_s = S["per_member_mu"]
    null_s, f0c_s = S["seed_f0_mut_mu"], S["seed_f0_ctx_mu"]
    Fm, Fc = SE["mut_mu"]["F_seed"].astype(float), SE["ctx_mu"]["F_seed"].astype(float)
    M = mu_s.shape[0]

    def cov_s(a, b):
        return ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (M - 1)
    var_mu = mu_s.var(0, ddof=1)
    Cm = np.stack([cov_s(Fm[:, :, j], mu_s) for j in range(len(F))], 1)
    Cn = cov_s(null_s, mu_s)
    Cc = np.stack([cov_s(Fc[:, :, j], mu_s) for j in range(len(F))], 1)
    Cf = cov_s(f0c_s, mu_s)
    comp_m = np.abs(Fm.sum(2) + null_s - mu_s)
    out = dict(sigma_inter_is_seed_sd_ddof1=bool(np.allclose(np.sqrt(var_mu), v.inter_epistemic_std)),
               per_seed_completeness_mut_mu=dict(median=float(np.median(comp_m)), p99=float(np.percentile(comp_m, 99))),
               decomposition_residual_max=float(np.abs(Cm.sum(1) + Cn - var_mu).max()))
    rows = []
    Fe = SE["mut_epistemic"]["F_seed"].astype(float)
    t_s = S["seed_f_mut_epistemic"] - S["seed_f0_mut_epistemic"]
    for sub, m in subsets(v):
        x = v[m]
        grp = x.cls.values if sub == "singles" else (x.cls + "+" + x.cls2).values
        res = {}
        for wn, w in (("pooled", np.ones(m.sum())), ("class_bal", inv_count(grp))):
            w = w / w.sum()
            tot = (w * var_mu[m]).sum()
            shm = (w[:, None] * Cm[m]).sum(0) / tot
            shc = (w[:, None] * Cc[m]).sum(0) / tot
            bsm = []
            for i in boot_idx(x.pdb.values, 300):
                wi = w[i] / w[i].sum()
                bsm.append((wi[:, None] * Cm[m][i]).sum(0) / (wi * var_mu[m][i]).sum())
            bsm = np.array(bsm)
            for j, f in enumerate(F):
                rows.append(dict(subset=sub, weighting=wn, feature=f, share_mut_view=shm[j],
                                 lo=np.percentile(bsm[:, j], 2.5), hi=np.percentile(bsm[:, j], 97.5),
                                 share_env_view=shc[j]))
            res[wn] = dict(null_share=r3((w * Cn[m]).sum() / tot), featureless_env_share=r3((w * Cf[m]).sum() / tot),
                           top_mut=[(F[j], r3(shm[j])) for j in np.argsort(-shm)[:6]],
                           top_env=[(F[j], r3(shc[j])) for j in np.argsort(-shc)[:4]])
        spread = Fm[:, m, :].std(0, ddof=1).sum(1)
        res["rho_sigma_inter_vs_seed_spread_of_mu_attr"] = r3(srho(x.inter_epistemic_std, spread))
        res["rho_sigma_inter_vs_1_minus_seed_cos_mu_attr"] = r3(srho(x.inter_epistemic_std, 1 - SE["mut_mu"]["cos_feat"][m]))
        ls = S["seed_f_mut_epistemic"][:, m]
        dev = np.abs(mu_s[:, m] - mu_s[:, m].mean(0))
        res["per_seed_rho(|mu_s - ensemble|, own log sigma_s)"] = r3(np.mean([srho(dev[k], ls[k]) for k in range(M)]))
        res["rho(sigma_inter, mean_s log sigma_s)"] = r3(srho(x.inter_epistemic_std, ls.mean(0)))
        e = x.abs_err.values
        res["rho_abs_err"] = {c: r3(srho(e, x[c])) for c in ("inter_epistemic_std", "intra_epistemic_std", "epistemic_std")}
        res["partial_within_stratum_abs_err_sigma_inter_given_intra_absmu"] = r3(strat_partial(
            e, x.inter_epistemic_std.values, [x.intra_epistemic_std.values, x.pred_mu.abs().values], x.stratum.values))
        # seed stability of the sigma explanations
        Sh = np.array([wshares(Fe[k][m], t_s[k][m], inv_count(grp)) for k in range(M)])
        Rr = np.corrcoef(np.array([rankdata(r) for r in Sh]))
        iu = np.triu_indices(M, 1)
        res["sigma_attr_seed_stability"] = dict(
            dataset_share_rank_agreement_mean=r3(Rr[iu].mean()), min=r3(Rr[iu].min()),
            volume_top_in_n_seeds=int((Sh.argmax(1) == F.index("Volume")).sum()),
            per_variant_feature_rank_agreement_median_sigma=r3(np.median(SE["mut_epistemic"]["rank_feat"][m])),
            per_variant_feature_rank_agreement_median_mu=r3(np.median(SE["mut_mu"]["rank_feat"][m])),
            per_seed_site_baseline_share=[r3(np.cov(S["seed_f0_mut_epistemic"][k][m], S["seed_f_mut_epistemic"][k][m],
                                                    bias=True)[0, 1] / S["seed_f_mut_epistemic"][k][m].var()) for k in range(M)])
        if sub == "singles":
            pro, gly = (x.cls == "to Pro").values, (x.cls == "to Gly").values
            res["sigma_attr_seed_stability"]["to_Pro_Cyclic_per_seed"] = [r3(a) for a in Fe[:, m][:, pro, F.index("Cyclic")].mean(1)]
            res["sigma_attr_seed_stability"]["to_Gly_Volume_per_seed"] = [r3(a) for a in Fe[:, m][:, gly, F.index("Volume")].mean(1)]
        FD.setdefault("G_seedshares", {})[sub] = Sh
        FD.setdefault("G_rank", {})[sub] = (SE["mut_epistemic"]["rank_feat"][m], SE["mut_mu"]["rank_feat"][m])
        K.setdefault("G_sigma_inter", {})[sub] = res
    K["G_sigma_inter"]["checks"] = out
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "sigma_inter_decomposition.csv", index=False, float_format="%.4g")
    FD["G"] = T


# ---------------------------------------------------------------------------- figures
def _pl(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def div_color(val, lim):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#d9d8d4", "#f08a89", "#b8292a"])
    return cm(0.5 + 0.5 * np.clip(val / lim, -1, 1))


def curve_panel(ax, cur, sub, f, title=True, ylab=False, legend=False):
    c = cur[(cur.subset == sub) & (cur.feature == f)]
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.axvline(0, color=C["neutral"], lw=0.6, ls=":")
    for b in BUR:
        q = c[c.bur == b].sort_values("dx")
        if len(q):
            ax.plot(q.dx, q.sig_attr, "-o", color=BCOL[b], lw=1.5, ms=3.2, mec="white", mew=0.5, label=b)
    if title:
        ax.set_title(SHORT.get(f, f), loc="left", fontsize=8.5)
    ax.set_xlabel(fd.UNITS.get(f, "Δ " + f), fontsize=7)
    if ylab:
        ax.set_ylabel("attribution to log σ\n(+ = more uncertain)", fontsize=7.5)
    if legend:
        ax.legend(fontsize=6.5, loc="best", handlelength=1.2)


def fig_main():
    B, Cc, D, G = FD["B"], FD["C"], FD["D"], FD["G"]
    fig = plt.figure(figsize=(7.4, 7.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.62)
    top = gs[0].subgridspec(1, 3, width_ratios=[0.55, 1.0, 1.15], wspace=0.62)
    bot = gs[1].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.42)
    # a: where vs what
    ax = fig.add_subplot(top[0])
    res = B["res"]
    for i, sub in enumerate(("singles", "doubles")):
        sshare = res[sub]["share_var_g_site_baseline"]
        ax.bar(i, sshare, color=COL_SITE, width=0.62, label="site baseline" if i == 0 else None)
        ax.bar(i, 1 - sshare, bottom=sshare, color=COL_MUT, width=0.62, label="mutation part" if i == 0 else None)
        ax.text(i, sshare / 2, f"{sshare:.0%}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        ax.text(i, sshare + (1 - sshare) / 2, f"{1 - sshare:.0%}", ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["MS-S", "MS-D"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of var(log σ$_{epi}$) across variants")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, handlelength=1)
    ax.set_title("Where vs what", loc="left", fontsize=8.5)
    _pl(ax, "a", x=-0.45, y=1.12)
    # b: error ranking of each component (within class x burial stratum)
    ax = fig.add_subplot(top[1])
    T = B["T"]
    comps = ["σ_e (total)", "site baseline g0", "mutation part g−g0", "σ_inter (ensemble)"]
    lab = ["σ$_e$ total", "site baseline", "mutation part", "σ$_{inter}$"]
    cols = [C["ink2"], COL_SITE, COL_MUT, COL_INTER]
    y = np.arange(len(comps))
    for k, (sub, off, mk) in enumerate((("singles", 0.17, "o"), ("doubles", -0.17, "s"))):
        t = T[T.subset == sub].set_index("component").loc[comps]
        ax.hlines(y + off, t.within_stratum_lo, t.within_stratum_hi, color=cols, lw=1.3)
        ax.scatter(t.within_stratum, y + off, c=cols, marker=mk, s=26, edgecolor="white", lw=0.6, zorder=3)
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(lab)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ρ(|error|, component)\nwithin class × burial stratum")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["ink2"], label="singles"),
                       Line2D([], [], marker="s", ls="", color=C["ink2"], label="doubles")],
              fontsize=6.5, loc="upper left", handletextpad=0.2)
    ax.set_title("Which part flags errors", loc="left", fontsize=8.5)
    _pl(ax, "b", x=-0.5, y=1.12)
    # c: shares mu vs sigma_intra (mutation part) vs sigma_inter, singles, class-balanced
    ax = fig.add_subplot(top[2])
    sh = D["shares"]["singles"]
    gi = G[(G.subset == "singles") & (G.weighting == "class_bal")].set_index("feature")
    feats = ["Volume", "Hydrophobicity", "VdW Clashes", "Polar Solvation", "Cyclic", "VdW Energy",
             "Hydrophobic Solvation"]
    y = np.arange(len(feats))
    h = 0.26
    ax.barh(y - h, sh.loc[feats, "mu_class_bal"], h, color=COL_MU, label="μ")
    ax.barh(y, sh.loc[feats, "sigma_class_bal"], h, color=COL_INTRA, label="σ$_{intra}$ (IG, mutation part)")
    ax.barh(y + h, gi.loc[feats, "share_mut_view"], h, color=COL_INTER, label="σ$_{inter}^2$ (seed decomposition)")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[f] for f in feats])
    ax.invert_yaxis()
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("variance share, class-balanced (singles)")
    ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.35, -0.22), handlelength=1, ncol=1)
    ax.set_title("Which features", loc="left", fontsize=8.5)
    _pl(ax, "c", x=-0.52, y=1.12)
    # d: response curves (2 x 2)
    dd = bot[0].subgridspec(2, 2, wspace=0.38, hspace=0.85)
    for k, f in enumerate(["Hydrophobicity", "Volume", "VdW Clashes", "Cyclic"]):
        ax = fig.add_subplot(dd[k // 2, k % 2])
        curve_panel(ax, Cc["cur"], "singles", f, ylab=(k % 2 == 0), legend=(k == 0))
        if k == 0:
            _pl(ax, "d", x=-0.42, y=1.12)
    # e: sign agreement of mu and sigma attributions per feature
    ax = fig.add_subplot(bot[1])
    R = D["R"]
    feats = ["Hydrophobicity", "VdW Energy", "Volume", "Polar Solvation", "Hydrophobic Solvation", "Electrostatics",
             "Hydrogen Bonds", "Cyclic", "VdW Clashes"]
    feats = list(R[R.subset == "singles"].set_index("feature").loc[feats].rho_mean_within_class
                 .sort_values(ascending=False).index)
    y = np.arange(len(feats))
    for sub, off, mk in (("singles", 0.17, "o"), ("doubles", -0.17, "s")):
        t = R[R.subset == sub].set_index("feature").reindex(feats)
        colr = [div_color(val, 0.5) for val in t.rho_mean_within_class.fillna(0)]
        ax.scatter(t.rho_pooled, y + off, marker="|", s=40, color=C["ink2"], lw=1.0, zorder=2)
        ax.hlines(y + off, np.minimum(t.rho_pooled, t.rho_mean_within_class),
                  np.maximum(t.rho_pooled, t.rho_mean_within_class), color=C["neutral"], lw=0.9, zorder=1)
        ax.scatter(t.rho_mean_within_class, y + off, c=colr, marker=mk, s=30, edgecolor=C["ink2"], lw=0.4, zorder=3)
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[f] for f in feats])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.75, 0.55)
    ax.set_ylim(len(feats) + 0.35, -0.6)
    ax.set_xlabel("ρ(A$_μ$, A$_σ$), mean within class\n● singles  ■ doubles  | pooled")
    ax.text(0.53, len(feats) - 0.05, "destabilizing\n⇒ more uncertain", ha="right", va="center",
            fontsize=6.3, color="#b8292a")
    ax.text(-0.73, len(feats) - 0.05, "destabilizing\n⇒ more confident", ha="left", va="center",
            fontsize=6.3, color="#184f95")
    ax.set_title("Same features, often opposite sign", loc="left", fontsize=8.5)
    _pl(ax, "e", x=-0.5, y=1.07)
    fig.text(0.01, 0.005, "MegaScale test, forward only. log σ$_{epi}$ = seed mean of 0.5·log β/(v(α−1)). "
                          "Bars/CI: 95% protein bootstrap. d: mutation IG on log σ, singles; + raises uncertainty.",
             fontsize=6.2, color=C["ink2"])
    _save(fig, "fig18_uncertainty_attribution_main")


def fig_curves(sub, name):
    cur = FD["C"]["cur"]
    feats = ["Hydrophobicity", "Volume", "VdW Clashes", "VdW Energy", "Hydrophobic Solvation", "Polar Solvation",
             "Electrostatics", "Cyclic"]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.9), gridspec_kw=dict(wspace=0.4, hspace=0.62))
    for k, (ax, f) in enumerate(zip(axes.flat, feats)):
        curve_panel(ax, cur, sub, f, ylab=(k % 4 == 0), legend=(k == 0))
    fig.suptitle(f"MegaScale {sub}: response of the log σ$_{{epi}}$ attribution to the feature change "
                 f"(mutation IG; + = more uncertain)", x=0.02, ha="left", fontsize=8.5, y=1.0)
    _save(fig, name)


def fig_useful():
    B, E = FD["B"], FD["E"]
    fig = plt.figure(figsize=(11.2, 3.7))
    gsu = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.05, 0.9], wspace=0.75)
    T = B["T"]
    comps = ["σ_e (total)", "site baseline g0", "mutation part g−g0", "σ_inter (ensemble)", "environment IG, SASA"]
    scopes = [("pooled", "pooled"), ("within_stratum", "within\nstratum"), ("within_site", "within\nsite/pair"),
              ("between_sites", "between\nsites/pairs")]
    for i, sub in enumerate(("singles", "doubles")):
        ax = fig.add_subplot(gsu[i])
        t = T[T.subset == sub].set_index("component").loc[comps]
        M_ = t[[c for c, _ in scopes]].values.astype(float)
        ax.imshow(np.nan_to_num(M_), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "div", ["#184f95", "#6da7ec", "#d9d8d4", "#f08a89", "#b8292a"]), vmin=-0.5, vmax=0.5, aspect="auto")
        for a_ in range(M_.shape[0]):
            for b_ in range(M_.shape[1]):
                val = M_[a_, b_]
                ax.text(b_, a_, "–" if np.isnan(val) else f"{val:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.35 else "black")
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels([sc[1] for sc in scopes], fontsize=6.2, rotation=45, ha="right")
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels(["σ$_e$ total", "site baseline", "mutation part", "σ$_{inter}$", "env. IG: SASA"]
                           if i == 0 else [])
        ax.grid(False)
        ax.set_title(f"{sub}: ρ(|error|, ·)", loc="left", fontsize=8)
        if i == 0:
            _pl(ax, "a", x=-0.05, y=1.1)
    # b: per-feature partial correlations beyond sigma and |mu|
    ax = fig.add_subplot(gsu[2])
    Fr = E["Fr"]
    top = (Fr.assign(a=Fr.partial_abs_err.abs()).sort_values("a", ascending=False).drop_duplicates(["family", "feature"])
           .head(10))
    keys = list(zip(top.family, top.feature))
    y = np.arange(len(keys))
    for sub, off, mk in (("singles", 0.17, "o"), ("doubles", -0.17, "s")):
        t = Fr[Fr.subset == sub].set_index(["family", "feature"]).reindex(keys)
        ax.scatter(t.partial_abs_err, y + off, marker=mk, s=24, c=[div_color(v_, 0.2) for v_ in t.partial_abs_err.fillna(0)],
                   edgecolor=C["ink2"], lw=0.4, label=sub, zorder=3)
    ax.axvline(0, color=C["ink2"], lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([("env: " if fam.startswith("env") else "mut: ") + SHORT.get(f, f) for fam, f in keys], fontsize=6.8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("partial ρ(|error|, attribution to log σ)\n| σ$_e$, |μ|, within stratum")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.2)
    ax.set_title("No feature signature beyond σ", loc="left", fontsize=8)
    _pl(ax, "b", x=-0.05, y=1.1)
    # c: AUC ladder
    ax = fig.add_subplot(gsu[3])
    for sub, col, mk in (("singles", C["blue"], "o"), ("doubles", C["orange"], "s")):
        a = E["out"][sub]["auc_protein_grouped_cv"]
        ax.plot(range(len(a)), list(a.values()), "-" + mk, color=col, ms=4, lw=1.3, label=sub)
    ax.axhline(0.5, color=C["neutral"], lw=0.8, ls=":")
    ax.set_xticks(range(len(a)))
    ax.set_xticklabels(list(a.keys()), rotation=40, ha="right", fontsize=6.3)
    ax.set_ylabel("AUC, wrong vs right\n(top-quartile σ; protein-grouped CV)")
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("Wrong vs right among high σ", loc="left", fontsize=8)
    _pl(ax, "c", x=-0.05, y=1.1)
    _save(fig, "fig20_useful_uncertainty")


def fig_doubles():
    Fd = FD["F"]
    P, E, cp = Fd["P"], Fd["E"], Fd["cp"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), gridspec_kw=dict(wspace=0.75, width_ratios=[1.0, 1.05, 1.0]))
    ax = axes[0]
    q = E.pivot(index="quantity", columns="subset", values="by_site")
    order = ["signed error", "|error|", "experimental ΔΔG", "σ_e", "σ_intra", "σ_inter", "mutation part g−g0"]
    q = q.loc[order]
    y = np.arange(len(order))
    ax.barh(y - 0.19, q.doubles, 0.38, color=C["orange"], label="doubles (66 site pairs)")
    ax.barh(y + 0.19, q.singles, 0.38, color=C["blue"], label="singles (1,562 sites)")
    ax.set_yticks(y)
    ax.set_yticklabels(["signed error", "|error|", "exp. ΔΔG", "σ$_e$", "σ$_{intra}$", "σ$_{inter}$", "mutation part"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("η² explained by site (singles) / site pair (doubles)")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
    _pl(ax, "a", x=-0.05, y=1.06)
    ax = axes[1]
    sc = ax.scatter(P.sigma_e, P.bias, s=np.clip(P.n / 12, 6, 90), c=P.sigma_inter, cmap="viridis", edgecolor="white",
                    lw=0.4)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.set_xlabel("pair-mean σ$_e$")
    ax.set_ylabel("pair-mean signed error (pred − exp)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("pair-mean σ$_{inter}$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    r1 = srho(P.sigma_e, P.bias.abs())
    ax.set_title(f"{len(P)} site pairs: ρ(σ$_e$, |bias|) = {r1:.2f}", loc="left", fontsize=8)
    _pl(ax, "b", x=-0.05, y=1.06)
    ax = axes[2]
    types = ["opposite-charge pair removed", "like-charge pair removed", "no charge-pair change",
             "opposite charges introduced", "like charges introduced"]
    bins = ["<6 Å", "6–9 Å", ">9 Å"]
    bc = {"<6 Å": C["red"], "6–9 Å": C["yellow"], ">9 Å": C["blue"]}
    y = np.arange(len(types))
    for k, b in enumerate(bins):
        t = cp[cp.dbin.astype(str) == b].set_index("cpair").reindex(types)
        ax.barh(y + (k - 1) * 0.26, t.sigma_e, 0.26, color=bc[b], label=f"d12 {b}")
    ax.set_yticks(y)
    ax.set_yticklabels(["opp. pair\nremoved", "like pair\nremoved", "no change", "opp.\nintroduced", "like\nintroduced"],
                       fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 0.72)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean σ$_e$")
    ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("Charge-pair change vs σ$_e$", loc="left", fontsize=8)
    _pl(ax, "c", x=-0.05, y=1.06)
    _save(fig, "fig21_doubles_uncertainty")


def fig_inter():
    G = FD["G"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), gridspec_kw=dict(wspace=0.75, width_ratios=[1.1, 1.1, 0.9]))
    for ax, view, lab, feats in ((axes[0], "share_mut_view", "a", ["Volume", "Hydrophobicity", "VdW Clashes", "Polar Solvation",
                                                                    "Cyclic", "Aromatic", "H_Bond_Acceptors", "Entropy"]),
                                 (axes[1], "share_env_view", "b", ["SASA", "Volume", "Hydrophobicity", "H_Bond_Donors",
                                                                   "H_Bond_Acceptors", "Polar Solvation"])):
        y = np.arange(len(feats))
        for sub, off, col in (("singles", -0.19, C["blue"]), ("doubles", 0.19, C["orange"])):
            t = G[(G.subset == sub) & (G.weighting == "class_bal")].set_index("feature").loc[feats]
            ax.barh(y + off, t[view], 0.38, color=col, label=sub)
            if view == "share_mut_view":
                ax.hlines(y + off, t.lo, t.hi, color=C["ink2"], lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[f] for f in feats])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=C["ink2"], lw=0.6)
        ax.set_xlabel("share of σ$_{inter}^2$ (class-balanced)")
        nul = K["G_sigma_inter"]["singles"]["class_bal"]["null_share" if view == "share_mut_view" else "featureless_env_share"]
        ax.set_title(("mutation view\n(remainder: null μ, %.0f%%)" if view == "share_mut_view" else
                      "environment view\n(remainder: featureless env., %.0f%%)") % (100 * nul), loc="left", fontsize=7.8)
        ax.legend(fontsize=6.5, loc="lower right")
        _pl(ax, lab, x=-0.05, y=1.14)
    ax = axes[2]
    for sub, col in (("singles", C["blue"]), ("doubles", C["orange"])):
        rs, rm = FD["G_rank"][sub]
        ax.hist(rm, bins=np.linspace(-0.6, 1, 33), histtype="step", color=col, lw=1.3, ls="-", density=True,
                label=f"μ, {sub}")
        ax.hist(rs, bins=np.linspace(-0.6, 1, 33), histtype="step", color=col, lw=1.3, ls="--", density=True,
                label=f"log σ, {sub}")
    ax.set_xlabel("feature-rank agreement\nacross 8 seeds (per variant)")
    ax.set_ylabel("density")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("seed agreement of\nper-variant explanations", loc="left", fontsize=7.8)
    _pl(ax, "c", x=-0.05, y=1.14)
    _save(fig, "fig22_sigma_inter_and_seeds")


# ---------------------------------------------------------------------------- H. mu rules vs sigma (packing split)
def packing_curves(v, A, DX):
    """Net packing attribution (Volume + VdW Clashes + VdW Energy) vs Delta volume, singles, for mu and log sigma."""
    m = (v.n_mutated == 1).values
    x = v[m].reset_index(drop=True)
    iV, iC, iE = F.index("Volume"), F.index("VdW Clashes"), F.index("VdW Energy")
    dv = DX[m][:, iV]
    lab = fd.assign(dv, fd.EPS["Volume"], fd.bins_for(dv, fd.EPS["Volume"]))
    df = pd.DataFrame(dict(dx=dv, lab=lab, bur=x.rsa_class))
    for nm, key in (("mu", "mut_mu"), ("sig", "mut_epistemic")):
        Aa = A[key][m]
        df[nm + "_V"], df[nm + "_C"], df[nm + "_E"] = Aa[:, iV], Aa[:, iC], Aa[:, iE]
        df[nm + "_pack"] = Aa[:, [iV, iC, iE]].sum(1)
    cur = df.groupby(["bur", "lab"]).agg(n=("dx", "size"), dx=("dx", "mean"),
                                         **{c: (c, "mean") for c in df.columns if c[:3] in ("mu_", "sig")})
    cur = cur[cur.n >= fd.MIN_N].reset_index()
    cur.to_csv(TAB / "packing_curves_mu_sigma.csv", index=False, float_format="%.4g")
    df["side"] = np.select([dv < -20, dv > 20], ["loss >20 Å³", "gain >20 Å³"], "small")
    summ = df.groupby(["bur", "side"])[[c for c in df.columns if c[:3] in ("mu_", "sig")]].mean().round(3)
    summ.to_csv(TAB / "packing_summary_mu_sigma.csv")
    K["H_packing"] = {f"{b} | {sd}": r.to_dict() for (b, sd), r in summ.iterrows()}
    # rule strength vs added uncertainty, per feature x burial x direction cell
    sm = FD["C"]["sm"]
    q = []
    for side in ("decrease", "increase"):
        t = sm[(sm.subset == "singles") & (sm.burial != "all")][["feature", "burial", f"mu_{side}", f"sig_{side}"]].dropna()
        t.columns = ["feature", "burial", "mu", "sig"]
        q.append(t)
    q = pd.concat(q)
    K["H_rule_strength_vs_sigma"] = {b: dict(mean_abs_mu=r3(q[q.burial == b].mu.abs().mean()),
                                             mean_sig=r3(q[q.burial == b].sig.mean()))
                                     for b in BUR}
    K["H_rule_strength_vs_sigma"]["rho_abs_mu_sig_cells"] = r3(spearmanr(q.mu.abs(), q.sig)[0])
    FD["H"] = cur


def fig_rules():
    cur, pk = FD["C"]["cur"], FD["H"]
    cols = [("Hydrophobicity", "Hydrophobicity"), ("packing", "Packing (net)"),
            ("Polar Solvation", "Polar solvation"), ("Electrostatics", "Electrostatics")]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4), sharey="row", gridspec_kw=dict(wspace=0.18, hspace=0.5))
    for r, (tgt, ylab) in enumerate((("mu", "attribution to μ (kcal/mol)\n+ = destabilizing"),
                                     ("sig", "attribution to log σ$_{epi}$\n+ = more uncertain"))):
        for c, (f, title) in enumerate(cols):
            ax = axes[r, c]
            ax.axhline(0, color=C["ink2"], lw=0.6)
            ax.axvline(0, color=C["neutral"], lw=0.6, ls=":")
            if f == "packing":
                for b in BUR:
                    q = pk[pk.bur == b].sort_values("dx")
                    ax.plot(q.dx, q[f"{tgt}_pack"], "-o", color=BCOL[b], lw=1.6, ms=3.2, mec="white", mew=0.5, label=b)
                q = pk[pk.bur == "core"].sort_values("dx")
                for comp, ls, nm in (("V", "--", "Volume"), ("C", ":", "Clashes"), ("E", "-.", "VdW energy")):
                    ax.plot(q.dx, q[f"{tgt}_{comp}"], ls, color=C["ink2"], lw=1.0, label=f"core: {nm}")
                ax.set_xlabel("Δ volume (Å³)", fontsize=7)
            else:
                q = cur[(cur.subset == "singles") & (cur.feature == f)]
                for b in BUR:
                    qq = q[q.bur == b].sort_values("dx")
                    ax.plot(qq.dx, qq[f"{tgt}_attr"], "-o", color=BCOL[b], lw=1.6, ms=3.2, mec="white", mew=0.5,
                            label=b)
                ax.set_xlabel(fd.UNITS[f], fontsize=7)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=7.5)
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    axes[0, 2].legend(handles=[Line2D([], [], ls=ls_, color=C["ink2"], lw=1.0, label=f"packing, core: {nm}")
                               for ls_, nm in (("--", "Volume"), (":", "VdW clashes"), ("-.", "VdW energy"))],
                      fontsize=6.2, loc="upper left", handlelength=2)
    axes[0, 0].text(-0.42, 1.12, "a  prediction (μ)", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=9.5)
    axes[1, 0].text(-0.42, 1.08, "b  uncertainty (log σ)", transform=axes[1, 0].transAxes, fontweight="bold",
                    fontsize=9.5)
    fig.text(0.01, -0.02, "MegaScale singles, mutation IG. Packing: net of Volume, VdW Clashes and VdW Energy per burial class; "
                          "thin grey lines = core components. Core sets the μ rules; σ is added where the rule is weak.",
             fontsize=6.3, color=C["ink2"])
    _save(fig, "fig23_rules_mu_vs_sigma")


def all_figs():
    fig_main()
    fig_curves("singles", "fig19a_sigma_response_curves_singles")
    fig_curves("doubles", "fig19b_sigma_response_curves_doubles")
    fig_useful()
    fig_doubles()
    fig_inter()
    fig_rules()


def main():
    if "--figs" in sys.argv:
        FD.update(pickle.load(open(OUT / "_fig_data.pkl", "rb")))
        K.update(json.load(open(OUT / "key_numbers.json")))
        all_figs()
        return
    ensure_cache()
    v, A, DX, S, SE = load()
    target_check(v, S)
    where_vs_what(v, A)
    response_curves(v, A, DX)
    mu_vs_sigma(v, A, DX)
    useful(v, A)
    doubles(v, A)
    sigma_inter(v, A, S, SE)
    packing_curves(v, A, DX)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, ensure_ascii=False, default=str)
    pickle.dump(FD, open(OUT / "_fig_data.pkl", "wb"))
    all_figs()
    print(json.dumps(K, indent=1, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
