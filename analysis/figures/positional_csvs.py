"""Rebuild data/positional_intra_megascale_v2.csv and data/positional_2k5h_intra.csv (inputs of positional_parts/)
from data/mega_test_uq.csv (sigma_intra = sigma_e). Run from results/figures/. Arg: output suffix ('' to overwrite)."""
import ast, sys, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
suf = sys.argv[1] if len(sys.argv) > 1 else ""
t = pd.read_csv("data/mega_test_uq.csv"); t = t[t.num_muts == 1].copy()
m = t.mutation.map(lambda s: ast.literal_eval(s)[0]); t["pos"] = m.map(lambda x: int(x[2])); t["wt"] = m.map(lambda x: x[1])
t["pdb"] = t.pdb.str.replace(".pdb", "", regex=False); t["absexp"] = t.true_ddg.abs()
def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); Z = np.column_stack([rz, np.ones(len(x))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)[0]
rows, per = [], {}
for pdb, g in t.groupby("pdb"):
    k = g.groupby(["pos", "wt"]).agg(sigma=("sigma_intra", "mean"), mae=("abs_err", "mean"), mean_abs_exp=("absexp", "mean"), n=("abs_err", "size")).reset_index()
    per[pdb] = k
    rows.append(dict(pdb=pdb, npos=len(k), pearson=pearsonr(k.sigma, k.mae)[0], spearman=spearmanr(k.sigma, k.mae)[0],
                     partial_spearman=partial(k.sigma, k.mae, k.mean_abs_exp), rho_sigma_absexp=spearmanr(k.sigma, k.mean_abs_exp)[0],
                     rho_err_absexp=spearmanr(k.mae, k.mean_abs_exp)[0]))
pp = pd.DataFrame(rows).sort_values("spearman", ascending=False)
pp.to_csv(f"data/positional_intra_megascale_v2{suf}.csv", index=False)
k = per["2K5H"].copy(); k["resi_pdb"] = k.pos + 34
old = pd.read_csv("data/positional_2k5h_intra.csv") if suf else None
if old is not None and "rsa" in old: k = k.merge(old[["pos", "rsa"]], on="pos", how="left")
k[[c for c in ["pos", "wt", "sigma", "mae", "mean_abs_exp", "n", "rsa", "resi_pdb"] if c in k]].to_csv(f"data/positional_2k5h_intra{suf}.csv", index=False)
print(pp.spearman.mean(), pp.spearman.median(), (pp.spearman > 0).sum(), len(pp))
