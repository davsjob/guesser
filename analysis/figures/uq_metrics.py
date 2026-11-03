"""AUCO (normalised: oracle 0, random 1) and ENCE for sigma_intra and sigma_e; s* fitted on validation (Gaussian NLL, closed form)."""
import numpy as np, pandas as pd, json
t = pd.read_csv("data/mega_test_uq.csv"); v = pd.read_csv("data/mega_val_uq.csv")
def curve(order, e, fr):
    n = len(e); return np.array([e[order[: max(1, int(round(f * n)))]].mean() for f in fr])
def auco(s, e):
    e = np.abs(e); fr = np.linspace(1, 0.05, 20)
    c = curve(np.argsort(s), e, fr); o = curve(np.argsort(e), e, fr); r = np.full_like(fr, e.mean())
    return np.trapz(c - o, fr[::-1]) / np.trapz(r - o, fr[::-1])
def ence(s, e, K=10):
    q = pd.qcut(s, K, labels=False, duplicates="drop"); return np.mean([abs(np.sqrt(np.mean(s[q == k]**2)) - np.sqrt(np.mean(e[q == k]**2))) / np.sqrt(np.mean(s[q == k]**2)) for k in np.unique(q)])
out = {}
for col in ["sigma_intra", "sigma_e", "inv_phi"]:
    sstar = float(np.sqrt(np.mean((v.err / v[col])**2))) if col != "inv_phi" else None
    for nm, d in [("singles", t[t.num_muts == 1]), ("doubles", t[t.num_muts == 2])]:
        s, e = d[col].values, d.err.values; r = {"auco": round(auco(s, e), 3)}
        if sstar: r.update(ence_raw=round(ence(s, e), 2), ence_scaled=round(ence(sstar * s, e), 2), s_star=round(sstar, 2))
        out[f"{col}|{nm}"] = r
json.dump(out, open("uq_metrics.json", "w"), indent=1); print(json.dumps(out, indent=0))
