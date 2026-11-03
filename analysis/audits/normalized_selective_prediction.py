"""Selective prediction with uncertainty normalised by predicted ddG, 2026-10-01.
Continuous counterpart of binned_selective_prediction.py.

Normalisations (all fitted on mega_val; k-NN window of k=200 val variants nearest in predicted ddG):
  *_pct : conditional percentile of the test score among the k val scores at similar predicted ddG
  *_lr  : log(score) - mean_k(log score)  (log-ratio to the typical score at that predicted ddG)
Scores: sigma_e and 1/Phi.  Reference rules: sigma_global, sigma_within_bin (v2 bins), predonly_global, random, oracle.
Ranking a normalised score globally keeps the predicted-ddG mix approximately fixed, so the fair comparator is random
(composition-preserving) and sigma_within_bin.  Design fixed before running.
mega_test, 8-seed ensemble, forward only; protein-cluster bootstrap (1000).
usage: python normalized_selective_prediction.py [singles|doubles|plot]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import AUDITS_OUT, PRED
import glob, os, sys, json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = str(AUDITS_OUT)
B = PRED
OUT = os.path.join(HERE, "normalized_selective_prediction"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); NBOOT = 1000; K = 200
COV = np.round(np.arange(0.10, 1.0001, 0.05), 2)
EDGES = {"singles": [-np.inf, -0.5, 0.5, 1.5, 3.0, np.inf], "doubles": [-np.inf, -0.5, 0.5, 1.5, 3.0, np.inf]}  # same edges as singles (2026-10-01)
SEL = {"singles": lambda x: x.num_muts == 1, "doubles": lambda x: x.num_muts >= 2}

def load(tag):
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(B + f"{tag}_predictions_S4*.csv"))]
    dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    df = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    M = lambda c: np.column_stack([d[c].values for d in dfs])
    v = np.maximum(M("v"), 1e-6); a = np.maximum(M("alpha"), 1 + 1e-6); b = np.maximum(M("beta"), 1e-6)
    df["pred"] = M("pred_ddg").mean(1)
    df["se"] = np.sqrt(np.mean(b / (v * (a - 1)), 1)); df["iphi"] = np.mean(1 / (2 * v + a), 1)
    df["abserr"] = (df.pred - df.true_ddg).abs()
    return df

def windows(val_pred_sorted, test_pred, k=K):
    n = len(val_pred_sorted); k = min(k, n); i = np.searchsorted(val_pred_sorted, test_pred)
    lo = np.clip(i - k // 2, 0, n - k); return lo, lo + k

def normalise(val, test):
    v = val.sort_values("pred").reset_index(drop=True); lo, hi = windows(v.pred.values, test.pred.values)
    out = {}
    ve = np.concatenate([[0], np.cumsum(v.abserr.values)]); out["eerr"] = (ve[hi] - ve[lo]) / (hi - lo)
    for col in ["se", "iphi"]:
        lv = np.log(v[col].values); cs = np.concatenate([[0], np.cumsum(lv)])
        out[f"{col}_lr"] = np.log(test[col].values) - (cs[hi] - cs[lo]) / (hi - lo)
        vv = v[col].values; t = test[col].values
        out[f"{col}_pct"] = np.array([(vv[l:h] < x).mean() + 0.5 * (vv[l:h] == x).mean() for l, h, x in zip(lo, hi, t)])
    return out

def mae_curve_global(score, e, covs):
    o = np.argsort(score, kind="stable"); cs = np.cumsum(e[o]); n = len(e)
    ks = np.ceil(np.asarray(covs) * n).astype(int); return cs[ks - 1] / ks

def mae_curve_within(score, e, groups, covs):
    tot = np.zeros(len(covs)); cnt = np.zeros(len(covs))
    for g in np.unique(groups):
        gi = groups == g; o = np.argsort(score[gi], kind="stable"); cs = np.cumsum(e[gi][o]); n = gi.sum()
        ks = np.ceil(np.asarray(covs) * n).astype(int); tot += cs[ks - 1]; cnt += ks
    return tot / cnt

def keep_mask(score, c):
    k = np.zeros(len(score), bool); k[np.argsort(score, kind="stable")[: int(np.ceil(c * len(score)))]] = True; return k

def spearman(x, y): return pd.Series(x).rank().corr(pd.Series(y).rank()) if len(x) > 2 else np.nan
def ci(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

GLOBAL_RULES = {"sigma_global": "se", "iphi_global": "iphi", "predonly_global": "eerr",
                "sigma_pct": "se_pct", "sigma_lr": "se_lr", "iphi_pct": "iphi_pct", "iphi_lr": "iphi_lr", "oracle": "abserr"}
WITHIN_RULES = {"sigma_within_bin": "se", "iphi_within_bin": "iphi"}

def all_curves(d, covs):
    e = d.abserr.values; c = {r: mae_curve_global(d[col].values, e, covs) for r, col in GLOBAL_RULES.items()}
    c.update({r: mae_curve_within(d[col].values, e, d.bin.values, covs) for r, col in WITHIN_RULES.items()})
    c["random"] = np.full(len(covs), e.mean()); return c

def analyse(name, d, val):
    d = d.reset_index(drop=True)
    d = d.assign(bin=np.digitize(d.pred.values, EDGES[name][1:-1], right=True), **normalise(val, d))
    labels = []
    for i in range(len(EDGES[name]) - 1):
        lo, hi = EDGES[name][i], EDGES[name][i + 1]
        labels.append(f"<={hi:g}" if np.isinf(lo) else (f">{lo:g}" if np.isinf(hi) else f"{lo:g}..{hi:g}"))
    e = d.abserr.values; full = e.mean()
    cur = all_curves(d, COV)
    rows = []
    for r, m in cur.items():
        for c, v in zip(COV, m):
            row = dict(set=name, rule=r, coverage=c, mae=v)
            if r in GLOBAL_RULES:
                s = d[keep_mask(d[GLOBAL_RULES[r]].values, c)]
                row.update(pearson=np.corrcoef(s.pred, s.true_ddg)[0, 1], frac_absgt15=(s.pred.abs() > 1.5).mean())
            rows.append(row)
    curves = pd.DataFrame(rows)
    per = []
    for i, lab in enumerate(labels):
        m = d.bin.values == i; s = d[m]; row = dict(set=name, bin=lab, n=int(m.sum()), mae_full=s.abserr.mean())
        for r in ["sigma_global", "iphi_global", "sigma_pct", "sigma_lr", "iphi_pct", "iphi_lr"]:
            k = keep_mask(d[GLOBAL_RULES[r]].values, 0.5)[m]
            row[f"kept_share_{r}"] = k.mean(); row[f"reldrop_kept_{r}"] = 1 - s.abserr.values[k].mean() / s.abserr.mean()
        for col in ["se_pct", "se_lr", "iphi_pct", "iphi_lr"]: row[f"rho_{col}_abserr"] = spearman(s[col].values, s.abserr.values)
        per.append(row)
    per = pd.DataFrame(per)
    covb = COV; bs = {}
    prots = d.pdb.unique(); idx = {p: np.where(d.pdb.values == p)[0] for p in prots}
    for _ in range(NBOOT):
        s = d.iloc[np.concatenate([idx[p] for p in rng.choice(prots, len(prots))])].reset_index(drop=True)
        cb = all_curves(s, covb); aurc = {r: m.mean() for r, m in cb.items()}; i50 = int(np.where(covb == 0.5)[0][0])
        for r, m in cb.items():
            bs.setdefault(f"aurc_{r}", []).append(aurc[r]); bs.setdefault(f"reldrop50_{r}", []).append(1 - m[i50] / s.abserr.mean())
        for x in ["sigma_pct", "sigma_lr", "iphi_pct", "iphi_lr"]:
            for y in ["random", "sigma_within_bin", "predonly_global", "sigma_global"]:
                bs.setdefault(f"d_aurc_{x}_minus_{y}", []).append(aurc[x] - aurc[y])
        bs.setdefault("d_aurc_iphi_lr_minus_sigma_lr", []).append(aurc["iphi_lr"] - aurc["sigma_lr"])
        bs.setdefault("d_aurc_sigma_pct_minus_sigma_lr", []).append(aurc["sigma_pct"] - aurc["sigma_lr"])
    summ = dict(n=len(d), n_proteins=int(d.pdb.nunique()), mae_full=full, bins=labels,
                aurc_point={r: float(m.mean()) for r, m in cur.items()},
                reldrop50_point={r: float(1 - m[list(COV).index(0.5)] / full) for r, m in cur.items()},
                full_pearson=float(np.corrcoef(d.pred, d.true_ddg)[0, 1]), full_frac_absgt15=float((d.pred.abs() > 1.5).mean()),
                rho_with_absPred={c: spearman(d[c].values, d.pred.abs().values) for c in ["se", "iphi", "se_pct", "se_lr", "iphi_pct", "iphi_lr"]},
                rho_with_pred={c: spearman(d[c].values, d.pred.values) for c in ["se", "iphi", "se_pct", "se_lr", "iphi_pct", "iphi_lr"]},
                rho_with_abserr={c: spearman(d[c].values, d.abserr.values) for c in ["se", "iphi", "eerr", "se_pct", "se_lr", "iphi_pct", "iphi_lr"]},
                boot_ci={k: dict(mean=float(np.nanmean(v)), ci95=ci(v)) for k, v in bs.items()})
    return curves, per, summ

mode = sys.argv[1] if len(sys.argv) > 1 else "plot"
if mode in SEL:
    val, test = load("mega_val"), load("mega_test")
    r = analyse(mode, test[SEL[mode](test)], val[SEL[mode](val)])
    pickle.dump(r, open(os.path.join(OUT, f"_res_{mode}.pkl"), "wb")); print("saved", mode); sys.exit(0)

res = {n: pickle.load(open(os.path.join(OUT, f"_res_{n}.pkl"), "rb")) for n in ["singles", "doubles"]}
pd.concat([r[0] for r in res.values()]).to_csv(os.path.join(OUT, "coverage_curves.csv"), index=False)
pd.concat([r[1] for r in res.values()]).to_csv(os.path.join(OUT, "per_bin.csv"), index=False)
json.dump({k: r[2] for k, r in res.items()}, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)

C = {"sigma_global": "#2a78d6", "sigma_within_bin": "#eb6834", "sigma_lr": "#1baf7a", "sigma_pct": "#eda100",
     "iphi_lr": "#e87ba4", "predonly_global": "#4a4a46"}
LAB = {"sigma_global": "σₑ, global", "sigma_within_bin": "σₑ, within bin", "sigma_lr": "σₑ, normalised (log-ratio)",
       "sigma_pct": "σₑ, normalised (percentile)", "iphi_lr": "1/Φ, normalised (log-ratio)", "predonly_global": "pred-only, global"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
for row, name in enumerate(["singles", "doubles"]):
    cv, pb, sm = res[name]
    a = ax[row, 0]
    for r in ["sigma_global", "predonly_global", "sigma_within_bin", "sigma_lr", "sigma_pct", "iphi_lr"]:
        s = cv[cv.rule == r]; a.plot(s.coverage, s.mae, color=C[r], lw=2, label=LAB[r])
    for r, ls in [("random", ":"), ("oracle", "--")]:
        s = cv[cv.rule == r]; a.plot(s.coverage, s.mae, color="#8a8a85", lw=1.5, ls=ls, label=r)
    a.invert_xaxis(); a.grid(alpha=0.25, lw=0.5); a.set_xlabel("coverage (fraction kept)"); a.set_ylabel("MAE (kcal/mol)")
    a.set_title(f"{name}: MAE vs coverage (n={sm['n']})", loc="left", fontsize=9)
    if row == 0: a.legend(frameon=False, fontsize=6.5)
    a = ax[row, 1]
    for r in ["sigma_global", "predonly_global", "sigma_within_bin", "sigma_lr", "iphi_lr"]:
        s = cv[(cv.rule == r)]; a.plot(s.coverage, s.pearson if r != "sigma_within_bin" else np.nan * s.coverage, color=C[r], lw=2)
    a.axhline(sm["full_pearson"], color="#8a8a85", ls=":", lw=1.2); a.invert_xaxis(); a.grid(alpha=0.25, lw=0.5)
    a.set_xlabel("coverage (fraction kept)"); a.set_ylabel("Pearson r on kept set")
    a.set_title("correlation on kept set (dotted: full set)", loc="left", fontsize=9)
    a = ax[row, 2]; x = np.arange(len(pb)); w = 0.27
    for k, r in enumerate(["sigma_global", "sigma_lr", "iphi_lr"]):
        v = pb[f"kept_share_{r}"].values * 100
        a.bar(x + (k - 1) * w, v, w * 0.9, color=C[r], label=LAB[r])
    a.axhline(50, color="#8a8a85", ls="--", lw=1)
    a.set_xticks(x); a.set_xticklabels([f"{l}\nn={n}" for l, n in zip(pb.bin, pb.n)], fontsize=7); a.set_ylim(0, 105)
    a.set_xlabel("predicted ΔΔG bin (kcal/mol)"); a.set_ylabel("% of bin kept at 50% coverage")
    a.set_title("composition of the kept set", loc="left", fontsize=9)
    if row == 0: a.legend(frameon=False, fontsize=6.5, loc="upper right")
for ext in ["png", "pdf"]: fig.savefig(os.path.join(OUT, f"fig_normalized_selective_prediction.{ext}"), dpi=200)
print("done")
