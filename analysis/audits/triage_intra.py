"""Stabilizing-mutation triage with EDL-only sigma_e (and 1/Phi), mega_test forward, 8 seeds. 2026-10-02.
Pool = pred <= -0.5; hit = exp <= -0.5. Mirrors triage_robustness_audit.py but with sigma_e = sqrt(mean_s beta/(v(alpha-1)))."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import AUDITS_OUT, PRED
import glob, os, json, numpy as np, pandas as pd
from scipy.stats import hypergeom, spearmanr
HERE = str(AUDITS_OUT); B = PRED
rng = np.random.default_rng(0)
def load(tag):
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(B + f"{tag}_predictions_S4*.csv"))]
    dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    df = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy(); M = lambda c: np.column_stack([d[c].values for d in dfs])
    v = np.maximum(M("v"), 1e-6); a = np.maximum(M("alpha"), 1 + 1e-6); b = np.maximum(M("beta"), 1e-6)
    df["pred"] = M("pred_ddg").mean(1); df["se"] = np.sqrt(np.mean(b / (v * (a - 1)), 1)); df["iphi"] = np.mean(1 / (2 * v + a), 1)
    return df
t = load("mega_test"); out = {}
KS = [5, 10, 15, 20, 30, 50, 100]
for name, sel in [("singles", t.num_muts == 1), ("doubles", t.num_muts >= 2)]:
    d = t[sel].copy(); d["hit"] = d.true_ddg <= -0.5; pool = d[d.pred <= -0.5].copy()
    n, h = len(pool), int(pool.hit.sum()); r = dict(n_all=len(d), hits_all=int(d.hit.sum()), pool=n, pool_hits=h, ppv=h / n, n_prot_pool=int(pool.pdb.nunique()))
    def topk(score, df):
        o = df.sort_values(score, kind="stable"); return {k: int(o.hit.values[:k].sum()) for k in KS if k <= len(df)}
    r["sigma"] = topk("se", pool); r["iphi"] = topk("iphi", pool); r["greedy"] = topk("pred", pool)
    r["p_sigma"] = {k: float(hypergeom.sf(v - 1, n, h, k)) for k, v in r["sigma"].items()}
    top = pool.sort_values("se").head(20); r["top20_by_protein"] = top.groupby("pdb").hit.agg(["size", "sum"]).sort_values("size", ascending=False).head(4).to_dict()
    prot_hits = pool.groupby("pdb").hit.agg(["size", "sum"]).sort_values("sum", ascending=False); r["pool_hits_by_protein_top3"] = prot_hits.head(3).to_dict()
    big = prot_hits.index[0]; ex = pool[pool.pdb != big]
    r["excl_top_protein"] = dict(protein=big, pool=len(ex), hits=int(ex.hit.sum()), sigma=topk("se", ex), greedy=topk("pred", ex))
    null = {k: [] for k in (10, 20, 50)}
    for _ in range(2000):
        sh = pool.copy(); sh["se"] = sh.groupby("pdb").se.transform(lambda x: rng.permutation(x.values)); tk = topk("se", sh)
        for k in null: null[k].append(tk[k])
    r["within_protein_shuffle"] = {k: dict(obs=r["sigma"][k], null_mean=float(np.mean(v)), p=float((np.sum(np.array(v) >= r["sigma"][k]) + 1) / 2001)) for k, v in null.items()}
    bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, 3, np.inf]
    d["bin"] = pd.cut(d.pred, bins); r["median_sigma_by_pred"] = {str(k): round(v, 3) for k, v in d.groupby("bin", observed=True).se.median().items()}
    r["rho_pred_sigma_in_pool"] = float(spearmanr(pool.pred, pool.se)[0])
    out[name] = r
json.dump(out, open(os.path.join(HERE, "triage_intra.json"), "w"), indent=1, default=str)
for k, r in out.items():
    print("=====", k); [print(" ", kk, vv) for kk, vv in r.items()]
