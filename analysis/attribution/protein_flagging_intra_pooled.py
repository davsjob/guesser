"""Pooled protein flagging and partial rho with sigma_e = EDL-only (intra) std (2026-10-04).
Re-runs the pooled within-set-rank test and the spread-controlled partial rho of deepdive_protein_flagging.py,
which used the old total sigma (epistemic_std). N >= 15 variants per protein, forward predictions."""
import json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
import config as cfg
RNG = np.random.default_rng(20261004)
def partial(t, x, y, z):
    rx, ry = rankdata(t[x]), rankdata(t[y]); Z = np.column_stack([rankdata(t[c]) for c in z] + [np.ones(len(t))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)
tables, out = {}, {}
for ds in cfg.DATASETS:
    v = pd.read_csv(cfg.CACHE_DIR / ds / "variants.csv")
    for k, lab in ((1, "singles"), (2, "doubles")):
        s = v[v.n_mutated == k]
        if not len(s): continue
        rows = [dict(pdb=p, n=len(g), r=pearsonr(g.pred_mu, g.true_ddg)[0], sig=g.intra_epistemic_std.mean(),
                     sig_old=g.epistemic_std.mean(), true_sd=g.true_ddg.std())
                for p, g in s.groupby("pdb") if len(g) >= 15 and g.true_ddg.std() > 0 and g.pred_mu.std() > 0]
        if len(rows) < 4: continue
        t = pd.DataFrame(rows); name = f"{cfg.DATASET_LABEL[ds]} {lab}"; t["set"] = name; tables[name] = t
        pc = partial(t, "sig", "r", ["true_sd"])
        out[name] = dict(n_prot=len(t), pearson=round(pearsonr(t.sig, t.r)[0], 3), spearman=round(spearmanr(t.sig, t.r)[0], 3),
                         partial_rho_given_spread=round(pc[0], 3), partial_p=round(pc[1], 4))
def pooled(names, col):
    a = pd.concat([tables[n] for n in names], ignore_index=True)
    a["s_pct"] = a.groupby("set")[col].rank(pct=True); a["r_pct"] = a.groupby("set").r.rank(pct=True)
    obs = spearmanr(a.s_pct, a.r_pct)[0]; null = []
    for _ in range(5000):
        null.append(spearmanr(a.groupby("set").s_pct.transform(lambda x: RNG.permutation(x.values)), a.r_pct)[0])
    k = int(np.sum(np.array(null) <= obs))
    return dict(n_proteins=len(a), sets=names, rho=round(obs, 3), n_perm=5000, n_null_as_extreme=k, perm_p_one_sided=round((k + 1) / 5001, 5))
allsets = list(tables)
noss = [n for n in allsets if not n.startswith("Ssym")]
out["pooled_intra_all"] = pooled(allsets, "sig"); out["pooled_intra_noSsym"] = pooled(noss, "sig")
out["pooled_old_noSsym"] = pooled(noss, "sig_old")
json.dump(out, open(cfg.OUT_DIR / "deepdive_protein_flagging" / "intra_pooled.json", "w"), indent=1)
print(json.dumps(out, indent=1))
