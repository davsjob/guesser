"""Steps 5-6: doubles and limits. sigma_e = sigma_intra throughout."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import FI, F, OUTV3, PB, ci, cfg, load_A, load_v, r3, save
from deepdive_composition_bias import inv_count, wshares

v = load_v()
A = load_A(("mut_mu", "mut_epistemic", "ctx_epistemic"))
R = {}

# ---------------------------------------------------------------- Step 5
vs = {}
for sub, nm in ((1, "singles"), (2, "doubles")):
    m = (v.n_mutated == sub).values
    x = v[m].reset_index(drop=True)
    Am = A["mut_mu"][m]
    pt = wshares(Am, x.tmu.values)
    out = dict(volume_share=r3(pt[FI["Volume"]]),
               top4=[(F[j], r3(pt[j])) for j in np.argsort(-pt)[:4]])
    for unit, col in (("protein", "pdb"), ("site", "site")):
        if nm == "singles" and unit == "site":
            continue
        u = x[col].values
        uu = np.unique(u)
        by = {p: np.where(u == p)[0] for p in uu}
        rng = np.random.default_rng(7)
        bs = []
        for _ in range(500):
            i = np.concatenate([by[p] for p in rng.choice(uu, len(uu))])
            bs.append(wshares(Am[i], x.tmu.values[i])[FI["Volume"]])
        out[f"volume_share_ci_{unit}"] = ci(bs)
    vs[nm] = out
R["step5_volume_share_of_mu_variance"] = vs

d = v[v.n_mutated == 2].reset_index(drop=True)
R["step5_gamma_epi"] = dict(mean_all_doubles=r3(d.pred_gamma_epi.mean()), sd=r3(d.pred_gamma_epi.std()),
                            text_says="+0.78 (doubles section)")
ek = __import__("json").load(open(OUTV3 / "deepdive_epistasis_route" / "key_numbers.json"))
R["step5_epistasis_route"] = dict(
    head_var_share=ek["shares"]["all|[head: gamma_epi]"]["var"], head_mean=ek["shares"]["all|[head: gamma_epi]"]["mean"],
    head_var_share_ci_protein=ek["contact_ci"]["protein"]["[head: gamma_epi]"]["var_share_ci"],
    head_var_share_ci_sitepair=ek["contact_ci"]["site_pair"]["[head: gamma_epi]"]["var_share_ci"],
    base_phys_mass_ratio_by_bin={k.split("|")[0]: r3(val["ratio_mass"]) for k, val in ek["consistency"].items()
                                 if k.endswith("mut_rest")})
pbp, pbr = PB(d.site), PB(d.pdb)
p, dp = pbp.mean(d.gm.values)
_, dr = pbr.mean(d.gm.values)
pair_means = d.groupby("site").gm.mean()
R["step5_doubles_mutation_part"] = dict(mean_gm=r3(p), ci_sitepair=ci(dp), ci_protein=ci(dr),
                                        mean_of_pair_means=r3(pair_means.mean()),
                                        frac_pairs_negative=r3((pair_means < 0).mean()), n_pairs=len(pair_means))
# boundary vs surface for singles mutation part (paired)
s = v[v.n_mutated == 1].reset_index(drop=True)
pb = PB(s.pdb)
_, db = pb.mean(s.gm.values, (s.rsa_class == "boundary").values)
_, dsf = pb.mean(s.gm.values, (s.rsa_class == "surface").values)
R["step4_extra_gm_boundary_minus_surface_singles"] = ci(db - dsf)

# ---------------------------------------------------------------- Step 6a: seed stability, same metrics for mu and sigma
Ss = np.load(cfg.CACHE_DIR / "megascale_test" / "seed_scalars.npz")
seeds = {"mu": np.load(cfg.CACHE_DIR / "megascale_test" / "seeds_mut_mu.npz"),
         "sigma": np.load(cfg.CACHE_DIR / "megascale_test" / "seeds_mut_epistemic.npz")}
tgt = {"mu": Ss["seed_f_mut_mu"] - Ss["seed_f0_mut_mu"],
       "sigma": Ss["seed_f_mut_epistemic"] - Ss["seed_f0_mut_epistemic"]}
st = {}
for sub, nm in ((1, "singles"), (2, "doubles")):
    m = (v.n_mutated == sub).values
    x = v[m]
    grp = x.cls.values if nm == "singles" else (x.cls + "+" + x.cls2).values
    for k in ("mu", "sigma"):
        Fs = seeds[k]["F_seed"][:, m].astype(float)
        M = Fs.shape[0]
        iu = np.triu_indices(M, 1)
        imp = np.abs(Fs).mean(1)  # metric 1: mean |attr| per feature (used for mu in v3 report)
        r_imp = np.corrcoef(np.array([rankdata(r) for r in imp]))[iu]
        Sh = np.array([wshares(Fs[s_], tgt[k][s_][m], inv_count(grp)) for s_ in range(M)])  # metric 2 (used for sigma)
        r_sh = np.corrcoef(np.array([rankdata(r) for r in Sh]))[iu]
        st[f"{nm} | {k}"] = dict(mean_abs_rank_agreement=r3(r_imp.mean()), mean_abs_min=r3(r_imp.min()),
                                 classbal_share_rank_agreement=r3(r_sh.mean()), classbal_share_min=r3(r_sh.min()),
                                 per_variant_rank_feat_median=r3(np.median(seeds[k]["rank_feat"][m])))
R["step6_seed_stability"] = st

# ---------------------------------------------------------------- Step 6b: do sigma attributions flag errors beyond sigma?
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

auc = {}
for sub, nm in ((1, "singles"), (2, "doubles")):
    m = (v.n_mutated == sub).values
    x = v[m].reset_index(drop=True)
    x["stratum"] = (x.cls if nm == "singles" else x.cls + "+" + x.cls2) + "|" + x.rsa_class
    e = x.abs_err.values
    res = {}
    for sig_col, tag in (("intra_epistemic_std", "sigma_intra (EDL, now sigma_e)"), ("epistemic_std", "old total (as in deep dive)")):
        hi = (x.groupby("pdb")[sig_col].rank(pct=True) > 0.75).values
        lab = np.select([e > 1.0, e < 0.5], [1, 0], -1)
        sel = hi & (lab >= 0)
        h = x[sel]
        y = lab[sel]
        base_edl = np.column_stack([np.log(h[sig_col]), h.pred_mu, h.rsa_mean, h.g0, h.gm])
        oh = pd.get_dummies(h.stratum).values.astype(float)
        sets = {"sigma only": np.log(h[sig_col].values)[:, None],
                "+ sigma parts, mu, burial": base_edl,
                "+ class x burial": np.column_stack([base_edl, oh]),
                "+ sigma attributions (46)": np.column_stack([base_edl, oh, A["mut_epistemic"][m][sel],
                                                              A["ctx_epistemic"][m][sel]]),
                "+ mu attributions (23)": np.column_stack([base_edl, oh, A["mut_epistemic"][m][sel],
                                                           A["ctx_epistemic"][m][sel], A["mut_mu"][m][sel]])}
        if tag.startswith("old"):  # exact replication of the deep-dive base (includes log sigma_inter)
            base_old = np.column_stack([np.log(h.epistemic_std), np.log(h.inter_epistemic_std), h.pred_mu, h.rsa_mean,
                                        h.g0, h.gm])
            sets["+ sigma parts, mu, burial"] = base_old
            sets["+ class x burial"] = np.column_stack([base_old, oh])
            sets["+ sigma attributions (46)"] = np.column_stack([base_old, oh, A["mut_epistemic"][m][sel],
                                                                 A["ctx_epistemic"][m][sel]])
            sets.pop("+ mu attributions (23)")
        aucs = {}
        for k, X in sets.items():
            p = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, h.pdb.values):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
                p[te] = mdl.predict_proba(X[te])[:, 1]
            aucs[k] = r3(roc_auc_score(y, p))
        res[tag] = dict(n_high=int(hi.sum()), n_wrong=int((y == 1).sum()), n_right=int((y == 0).sum()), auc=aucs,
                        raw_auc_sigma=r3(roc_auc_score(y, h[sig_col])))
    auc[nm] = res
R["step6_useful_uncertainty_auc"] = auc
save("results_part_c", R)
