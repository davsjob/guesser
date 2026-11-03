"""Per-variant ensemble quantities from the 8 per-seed CSVs (forward only).
sigma_intra = sqrt(mean_s beta/(v(alpha-1)));  sigma_inter = std_s(mu) (ddof 1, as in the explanation files);  sigma_e = sqrt(intra^2 + inter^2)
inv_phi = mean_s 1/(2v+alpha)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import PRED
import glob, numpy as np, pandas as pd
P = PRED
def build(ds):
    fs = sorted(glob.glob(P + f"{ds}_predictions_S*.csv")); dfs = [pd.read_csv(f) for f in fs]
    dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    for d in dfs[1:]: assert (d.mutation.values == dfs[0].mutation.values).all()
    mu = np.column_stack([d.pred_ddg for d in dfs]); v = np.column_stack([d.v for d in dfs]).clip(1e-6)
    a = np.column_stack([d.alpha for d in dfs]).clip(1 + 1e-6); b = np.column_stack([d.beta for d in dfs]).clip(1e-6)
    out = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    out["mu"] = mu.mean(1); out["err"] = out.mu - out.true_ddg; out["abs_err"] = out.err.abs()
    out["sigma_intra"] = np.sqrt((b / (v * (a - 1))).mean(1)); out["sigma_inter"] = mu.std(1, ddof=1)
    out["sigma_e"] = np.sqrt(out.sigma_intra**2 + out.sigma_inter**2)
    out["sigma_alea"] = np.sqrt((b / (a - 1)).mean(1)); out["inv_phi"] = (1 / (2 * v + a)).mean(1)
    out["nseeds"] = len(fs); return out
for ds in ["mega_test", "mega_val"]:
    o = build(ds); o.to_csv(f"data/{ds}_uq.csv", index=False)
    print(ds, len(o), o.groupby("num_muts")[["abs_err", "sigma_intra", "sigma_inter", "sigma_e"]].mean().round(3).to_dict())
