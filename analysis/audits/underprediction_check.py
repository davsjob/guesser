"""Is the under-prediction of large destabilizations a model mechanism or a binning effect?
Signed error (pred - exp), MegaScale test, forward, 8-seed mean. Binned by experimental vs by predicted ddG.
Val set used as a proxy for the training distribution. (2026-10-02)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import AUDITS_OUT, PRED
import glob, json, numpy as np, pandas as pd
B = PRED
def load(tag):
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(B + f"{tag}_predictions_S4*.csv"))]
    dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    df = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    df["pred"] = np.column_stack([d.pred_ddg.values for d in dfs]).mean(1)
    df["err"] = df.pred - df.true_ddg
    return df
te, va = load("mega_test"), load("mega_val")
bins = [-np.inf, -0.5, 0.5, 1.5, 3, np.inf]; out = {}
for name, m in [("singles", 1), ("doubles", 2)]:
    t, v = te[te.num_muts == m], va[va.num_muts == m]
    out[name] = dict(
        n=len(t), r=round(float(np.corrcoef(t.true_ddg, t.pred)[0, 1]), 3),
        slope_pred_on_exp=round(float(np.polyfit(t.true_ddg, t.pred, 1)[0]), 3),
        frac_exp_gt3_test=round(float(np.mean(t.true_ddg > 3)), 4), frac_exp_gt3_val=round(float(np.mean(v.true_ddg > 3)), 4),
        err_by_exp_bin={str(k): round(float(x), 2) for k, x in t.groupby(pd.cut(t.true_ddg, bins), observed=False).err.mean().items()},
        err_by_pred_bin={str(k): round(float(x), 2) for k, x in t.groupby(pd.cut(t.pred, bins), observed=False).err.mean().items()},
        n_by_pred_bin={str(k): int(x) for k, x in t.groupby(pd.cut(t.pred, bins), observed=False).size().items()})
json.dump(out, open(AUDITS_OUT / "underprediction_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
