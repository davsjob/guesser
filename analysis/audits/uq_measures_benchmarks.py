"""Benchmark-wide comparison of uncertainty measures from the EDL head (2026-10-03).
Forward only, 8 seeds. Per variant:
  sigma_e   = sqrt(mean_s beta/(v(alpha-1)))      EDL epistemic (paper's sigma_e)
  inv_phi   = mean_s 1/(2v+alpha)                  inverse total evidence
  sigma_a   = sqrt(mean_s beta/(alpha-1))          EDL aleatoric
  sigma_inter = std_s(mu)                          ensemble spread (reference only)
Metrics: AUCO (same function as figures/uq_metrics.py: oracle 0, random 1),
pooled Spearman rho(u, |err|), within-protein mean Spearman (proteins with >= 15 variants),
protein-bootstrap 95% CI (1000 draws) for AUCO and rho, and for the sigma_e - inv_phi differences."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import os
from paths import AUDITS_OUT, PRED
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
P = PRED
OUT = str(AUDITS_OUT / "uq_measures_benchmarks") + "/"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
MEAS = ["sigma_e", "inv_phi", "sigma_a", "sigma_inter"]

def build(ds):
    fs = sorted(glob.glob(P + f"{ds}_predictions_S4*.csv")); assert len(fs) == 8, (ds, len(fs))
    dfs = [pd.read_csv(f) for f in fs]; dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    for d in dfs[1:]: assert (d.mutation.values == dfs[0].mutation.values).all() and (d.pdb.values == dfs[0].pdb.values).all()
    M = lambda c: np.column_stack([d[c].values for d in dfs])
    mu = M("pred_ddg"); v = M("v").clip(1e-6); a = M("alpha").clip(1 + 1e-6); b = M("beta").clip(1e-6)
    o = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    o["mu"] = mu.mean(1); o["abs_err"] = (o.mu - o.true_ddg).abs()
    o["sigma_e"] = np.sqrt((b / (v * (a - 1))).mean(1)); o["inv_phi"] = (1 / (2 * v + a)).mean(1)
    o["sigma_a"] = np.sqrt((b / (a - 1)).mean(1)); o["sigma_inter"] = mu.std(1, ddof=1)
    return o

def curve(order, e, fr):
    n = len(e); return np.array([e[order[: max(1, int(round(f * n)))]].mean() for f in fr])
def auco(s, e):
    fr = np.linspace(1, 0.05, 20)
    c = curve(np.argsort(s), e, fr); o = curve(np.argsort(e), e, fr); r = np.full_like(fr, e.mean())
    return np.trapz(c - o, fr[::-1]) / np.trapz(r - o, fr[::-1])

def within(d, col, nmin=15):
    rs = [spearmanr(g[col], g.abs_err)[0] for _, g in d.groupby("pdb") if len(g) >= nmin]
    rs = [r for r in rs if np.isfinite(r)]
    return (float(np.mean(rs)), int(sum(r > 0 for r in rs)), len(rs)) if rs else (None, 0, 0)

def stats(d):
    e = d.abs_err.values; res = {"n": len(d), "proteins": int(d.pdb.nunique())}
    for c in MEAS:
        res[c] = {"auco": auco(d[c].values, e), "rho": spearmanr(d[c], e)[0], "within": within(d, c)}
    # protein bootstrap
    groups = {p: g.index.values for p, g in d.groupby("pdb")}; keys = list(groups)
    boot = {c: [] for c in MEAS}; bdiff_a, bdiff_r = [], []
    for _ in range(1000):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]); bd = d.loc[idx]; be = bd.abs_err.values
        vals = {}
        for c in ["sigma_e", "inv_phi"]:
            vals[c] = (auco(bd[c].values, be), spearmanr(bd[c], be)[0]); boot[c].append(vals[c])
        bdiff_a.append(vals["sigma_e"][0] - vals["inv_phi"][0]); bdiff_r.append(vals["sigma_e"][1] - vals["inv_phi"][1])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    for c in ["sigma_e", "inv_phi"]:
        arr = np.array(boot[c]); res[c]["auco_ci"] = ci(arr[:, 0]); res[c]["rho_ci"] = ci(arr[:, 1])
    res["diff_sigma_e_minus_inv_phi"] = {"auco": res["sigma_e"]["auco"] - res["inv_phi"]["auco"], "auco_ci": ci(bdiff_a),
                                         "rho": res["sigma_e"]["rho"] - res["inv_phi"]["rho"], "rho_ci": ci(bdiff_r)}
    res["rho_sigma_e_inv_phi"] = spearmanr(d.sigma_e, d.inv_phi)[0]
    return res

sets = [("MegaScale-S", "mega_test", 1), ("MegaScale-D", "mega_test", 2), ("FireProt-HF", "fireprot_hf", None),
        ("SSYM", "ssym", None), ("S669", "s669", None), ("PTMUL-D", "ptmul_d", None)]
out, cache = {}, {}
for name, ds, nm in sets:
    if ds not in cache: cache[ds] = build(ds)
    d = cache[ds] if nm is None else cache[ds][cache[ds].num_muts == nm]
    d = d.reset_index(drop=True); out[name] = stats(d); print(name, "done", flush=True)

def fl(x):
    if isinstance(x, dict): return {k: fl(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [fl(v) for v in x]
    if isinstance(x, (np.floating, float)): return round(float(x), 4)
    if isinstance(x, np.integer): return int(x)
    return x
out = fl(out); json.dump(out, open(OUT + "results.json", "w"), indent=1)
rows = []
for s, r in out.items():
    for c in MEAS:
        w = r[c]["within"]; rows.append(dict(set=s, n=r["n"], proteins=r["proteins"], measure=c, auco=r[c]["auco"], rho=r[c]["rho"],
                                             within_rho=w[0], within_pos=f"{w[1]}/{w[2]}"))
pd.DataFrame(rows).to_csv(OUT + "table.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print({s: (r["diff_sigma_e_minus_inv_phi"], r["rho_sigma_e_inv_phi"]) for s, r in out.items()})
