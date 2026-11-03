"""All sigma-dependent manuscript numbers with the EDL-only uncertainty sigma = sigma_intra (NIG epistemic, ensemble-averaged)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import ATTRIBUTION_OUT
import ast, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
A3 = str(ATTRIBUTION_OUT) + "/"
t = pd.read_csv("data/mega_test_uq.csv"); v = pd.read_csv("data/mega_val_uq.csv")
t["pos"] = t.mutation.map(lambda s: ast.literal_eval(s)[0][2]); U = "sigma_intra"
sg, db = t[t.num_muts == 1], t[t.num_muts == 2]; N = {}
N["s_star"] = float(np.sqrt((v.err**2).mean() / (v[U]**2).mean()))
for lab, d in [("singles", sg), ("doubles", db)]:
    N[f"{lab}_pooled_rho"] = spearmanr(d[U], d.abs_err)[0]
    w = [spearmanr(g[U], g.abs_err)[0] for _, g in d.groupby("pdb") if len(g) >= 15]
    N[f"{lab}_within_rho"] = (float(np.mean(w)), int(np.sum(np.array(w) > 0)), len(w), int(np.sum(np.array(w) > 0.2)))
    N[f"{lab}_mae_all"] = d.abs_err.mean(); N[f"{lab}_mae_keep20"] = d[d[U] <= d[U].quantile(0.2)].abs_err.mean()
    N[f"{lab}_decile_mae"] = d.groupby(pd.qcut(d[U], 10, labels=False)).abs_err.mean().round(3).tolist()
    N[f"{lab}_mean_sigma"] = d[U].mean()
    g = d.groupby(pd.cut(d.true_ddg, [-10, 0, 1, 2, 3, 20]), observed=True); N[f"{lab}_sigma_by_exp"] = g[U].mean().round(3).tolist()
# protein flagging
pp = pd.read_csv(A3 + "deepdive_protein_flagging/per_protein_N15.csv"); pp = pp[pp.set.str.contains("MegaScale|FireProt|S669|PTMUL")]
def partial(x, y, z):
    rx, ry, rz = (rankdata(a) for a in (x, y, z)); res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return np.corrcoef(res(rx), res(ry))[0, 1]
for s, g in pp.groupby("set"):
    r, p = pearsonr(g[U], g.r); N[f"flag|{s}"] = dict(n=len(g), pearson=round(r, 3), p=round(p, 4), spearman=round(spearmanr(g[U], g.r)[0], 3), partial_spread=round(partial(g[U].values, g.r.values, g.true_sd.values), 3), rho_vs_mae=round(spearmanr(g[U], g.mae)[0], 3))
pp["rs"] = pp.groupby("set")[U].transform(lambda x: rankdata(x) / len(x)); pp["rr"] = pp.groupby("set").r.transform(lambda x: rankdata(x) / len(x))
rho = spearmanr(pp.rs, pp.rr)[0]; rng = np.random.default_rng(0)
perm = np.array([spearmanr(pp.rs, pp.groupby("set").rr.transform(lambda x: rng.permutation(x.values)))[0] for _ in range(5000)])
N["flag_pooled"] = (round(rho, 3), float((np.sum(perm <= rho) + 1) / 5001), len(pp))
# positional (MS-S from cache; FireProt from table)
pr = []
for p, g in sg.groupby("pdb"):
    k = g.groupby("pos").agg(s=(U, "mean"), e=("abs_err", "mean"), m=("true_ddg", lambda x: x.abs().mean()))
    r = pearsonr(k.s, k.e)[0]; X = np.column_stack([np.ones(len(k)), k.m]); rs = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    pr.append(dict(pdb=p, r=r, rho=spearmanr(k.s, k.e)[0], partial=np.corrcoef(rs(k.s.values), rs(k.e.values))[0, 1]))
pr = pd.DataFrame(pr); pr.to_csv("data/positional_intra_megascale.csv", index=False)
N["positional"] = dict(mean_r=pr.r.mean(), mean_rho=pr.rho.mean(), pos=int((pr.r > 0).sum()), n=len(pr), partial=pr.partial.mean(), r_2k5h=float(pr[pr.pdb == "2K5H.pdb"].r.iloc[0]), rho_2k5h=float(pr[pr.pdb == "2K5H.pdb"].rho.iloc[0]))
N["positional_fphf_mean_r_intra"] = None  # came from the retired deepdive_positional_uq; not used in the paper
# doubles site pairs
sp = pd.read_csv(A3 + "deepdive_uncertainty_attribution/tables/doubles_site_pairs.csv")
N["pair_rho_intra_absbias"] = spearmanr(sp[U], sp.bias.abs())[0]; N["pair_rho_intra_mae"] = spearmanr(sp[U], sp.mae)[0]; N["pair_rho_inter_mae"] = spearmanr(sp.sigma_inter, sp.mae)[0]
N["sigma_over_mae"] = (sg[U].mean() / sg.abs_err.mean(), db[U].mean() / db.abs_err.mean())
json.dump(N, open("uq_numbers_intra.json", "w"), indent=1, default=float); print(json.dumps(N, indent=1, default=lambda x: round(float(x), 3)))
