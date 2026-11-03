"""Binned (stratified) selective prediction control, 2026-09-30 (v2: adds 1/Phi).

Question: do the EDL uncertainty scores rank errors *within* a predicted-ddG regime,
or does global filtering lower MAE mainly by keeping near-zero predictions?

Design fixed before running (no tuning):
  bins on ensemble-mean predicted ddG
    singles: <=-0.5 | -0.5..0.5 | 0.5..1.5 | 1.5..3 | >3
    doubles: same edges as singles   (bins with n<40 merged into neighbour)
  scores (keep the lowest-score fraction c), each applied globally and within bin:
    sigma    : epistemic std sigma_e = sqrt(mean_seeds beta/(v(alpha-1)))
    iphi     : 1/Phi = mean_seeds 1/(2v+alpha)   (low = more evidence)
    predonly : E[|err| | pred] fitted on mega_val (k-NN in pred, k=200)
  plus random (expectation = full MAE) and oracle (true |err|).
  CIs: protein-cluster bootstrap (1000 reps). 8-seed ensemble, forward only, mega_test.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import AUDITS_OUT, PRED
import glob, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = str(AUDITS_OUT)
B = PRED
OUT = os.path.join(HERE, "binned_selective_prediction"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
NBOOT = 1000
COV = np.round(np.arange(0.10, 1.0001, 0.05), 2)
EDGES = {"singles": [-np.inf, -0.5, 0.5, 1.5, 3.0, np.inf], "doubles": [-np.inf, -0.5, 0.5, 1.5, 3.0, np.inf]}  # same edges as singles (2026-10-01)
SCORES = {"sigma": "se", "iphi": "iphi", "predonly": "eerr"}

def load(tag):
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(B + f"{tag}_predictions_S4*.csv"))]
    dfs = [d[d.direction == "Forward"].reset_index(drop=True) for d in dfs]
    df = dfs[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    M = lambda c: np.column_stack([d[c].values for d in dfs])
    v = np.maximum(M("v"), 1e-6); a = np.maximum(M("alpha"), 1 + 1e-6); b = np.maximum(M("beta"), 1e-6)
    df["pred"] = M("pred_ddg").mean(1)
    df["se"] = np.sqrt(np.mean(b / (v * (a - 1)), 1))
    df["iphi"] = np.mean(1 / (2 * v + a), 1)
    df["abserr"] = (df.pred - df.true_ddg).abs()
    return df, len(dfs)

def knn_expected_err(val, test_pred, k=200):
    v = val.sort_values("pred"); vp = v.pred.values; ve = v.abserr.values
    cs = np.concatenate([[0], np.cumsum(ve)]); n = len(vp); k = min(k, n)
    i = np.searchsorted(vp, test_pred)
    lo = np.clip(i - k // 2, 0, n - k); hi = lo + k
    return (cs[hi] - cs[lo]) / k

def assign_bins(pred, edges):
    b = np.digitize(pred, edges[1:-1], right=True)
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"<={hi:g}" if np.isinf(lo) else (f">{lo:g}" if np.isinf(hi) else f"{lo:g}..{hi:g}"))
    return b, labels

def merge_small(b, labels, nmin=40):
    b = b.copy(); labels = list(labels)
    while True:
        counts = np.bincount(b, minlength=len(labels))
        small = [i for i, c in enumerate(counts) if c < nmin]
        if not small or len(labels) == 1: return b, labels
        i = small[0]; j = i - 1 if i == len(labels) - 1 else i + 1
        lo, hi = min(i, j), max(i, j)
        labels[lo] = f"{labels[lo]}+{labels[hi]}"; del labels[hi]
        b[b == hi] = lo; b[b > hi] -= 1

def keep_mask(score, c, groups=None):
    n = len(score); keep = np.zeros(n, bool)
    if groups is None:
        keep[np.argsort(score, kind="stable")[: int(np.ceil(c * n))]] = True; return keep
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        keep[gi[np.argsort(score[gi], kind="stable")[: int(np.ceil(c * len(gi)))]]] = True
    return keep

def spearman(x, y):
    return pd.Series(x).rank().corr(pd.Series(y).rank()) if len(x) > 2 else np.nan

def rules(d):
    r = {}
    for s, col in SCORES.items():
        r[f"{s}_global"] = (d[col].values, None); r[f"{s}_within_bin"] = (d[col].values, d.bin.values)
    r["oracle"] = (d.abserr.values, None)
    return r

def curve_metrics(d, score, groups, c):
    s = d[keep_mask(score, c, groups)]
    return dict(mae=s.abserr.mean(), pearson=np.corrcoef(s.pred, s.true_ddg)[0, 1],
                spearman=spearman(s.pred.values, s.true_ddg.values), frac_absgt15=(s.pred.abs() > 1.5).mean())

def boot_samples(d):
    prots = d.pdb.unique(); idx_by = {p: np.where(d.pdb.values == p)[0] for p in prots}
    for _ in range(NBOOT):
        yield d.iloc[np.concatenate([idx_by[p] for p in rng.choice(prots, len(prots))])].reset_index(drop=True)

def ci(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

def analyse(name, d, val):
    b, labels = assign_bins(d.pred.values, EDGES[name]); b, labels = merge_small(b, labels)
    d = d.assign(bin=b, eerr=knn_expected_err(val, d.pred.values)).reset_index(drop=True)
    full_mae = d.abserr.mean()
    rows = []
    for r, (sc, gr) in rules(d).items():
        for c in COV: rows.append(dict(set=name, rule=r, coverage=c, **curve_metrics(d, sc, gr, c)))
    for c in COV: rows.append(dict(set=name, rule="random", coverage=c, mae=full_mae, pearson=np.nan, spearman=np.nan, frac_absgt15=np.nan))
    curves = pd.DataFrame(rows)
    aurc = curves.groupby("rule").mae.mean().to_dict()
    per = []
    for i, lab in enumerate(labels):
        s = d[d.bin == i]; e = s.abserr.values
        row = dict(set=name, bin=lab, n=len(s), n_proteins=s.pdb.nunique(), mean_pred=s.pred.mean(), mae_full=e.mean(),
                   reldrop50_oracle=1 - e[keep_mask(e, 0.5)].mean() / e.mean())
        for sname, col in SCORES.items():
            row[f"reldrop50_{sname}"] = 1 - e[keep_mask(s[col].values, 0.5)].mean() / e.mean()
            if sname != "predonly":
                row[f"rho_{sname}_abserr"] = spearman(s[col].values, e)
                wp = [spearman(g[col].values, g.abserr.values) for _, g in s.groupby("pdb") if len(g) >= 10]
                row[f"rho_{sname}_within_protein_mean"] = np.nanmean(wp) if wp else np.nan
                row[f"retained_by_global_{sname}50"] = keep_mask(d[col].values, 0.5)[d.bin.values == i].mean()
        per.append(row)
    per = pd.DataFrame(per)
    # bootstrap
    bs = {}; bb = {}
    cov_b = COV[::2]
    for s in boot_samples(d):
        m = s.abserr.mean(); a = {}
        for r, (sc, gr) in rules(s).items():
            if r == "oracle": continue
            a[r] = np.mean([s.abserr.values[keep_mask(sc, c, gr)].mean() for c in cov_b])
            bs.setdefault(f"aurc_{r}", []).append(a[r])
            bs.setdefault(f"reldrop50_{r}", []).append(1 - s.abserr.values[keep_mask(sc, 0.5, gr)].mean() / m)
        for mode in ["global", "within_bin"]:
            for x, y in [("sigma", "predonly"), ("iphi", "predonly"), ("iphi", "sigma")]:
                bs.setdefault(f"d_aurc_{mode}_{x}_minus_{y}", []).append(a[f"{x}_{mode}"] - a[f"{y}_{mode}"])
        for i in range(len(labels)):
            g = s[s.bin.values == i]
            if len(g) < 10: continue
            e = g.abserr.values; rd = {}
            for sname, col in SCORES.items():
                rd[sname] = 1 - e[keep_mask(g[col].values, 0.5)].mean() / e.mean()
                bb.setdefault((i, f"reldrop50_{sname}"), []).append(rd[sname])
                if sname != "predonly": bb.setdefault((i, f"rho_{sname}"), []).append(spearman(g[col].values, e))
            bb.setdefault((i, "sigma_minus_predonly"), []).append(rd["sigma"] - rd["predonly"])
            bb.setdefault((i, "iphi_minus_predonly"), []).append(rd["iphi"] - rd["predonly"])
            bb.setdefault((i, "iphi_minus_sigma"), []).append(rd["iphi"] - rd["sigma"])
    for key in ["reldrop50_sigma", "reldrop50_iphi", "reldrop50_predonly", "rho_sigma", "rho_iphi",
                "sigma_minus_predonly", "iphi_minus_predonly", "iphi_minus_sigma"]:
        per[f"{key}_ci"] = [ci(bb.get((i, key), [np.nan])) for i in range(len(labels))]
    summ = dict(n=len(d), n_proteins=int(d.pdb.nunique()), mae_full=full_mae, bins=labels, aurc_point_step005=aurc,
                reldrop50_point={r: 1 - curves[(curves.rule == r) & (curves.coverage == 0.5)].mae.iloc[0] / full_mae for r in rules(d)},
                corr_at50={r: curves[(curves.rule == r) & (curves.coverage == 0.5)][["pearson", "spearman", "frac_absgt15"]].iloc[0].to_dict()
                           for r in rules(d) if r != "oracle"},
                full_corr=dict(pearson=float(np.corrcoef(d.pred, d.true_ddg)[0, 1]), spearman=spearman(d.pred.values, d.true_ddg.values),
                               frac_absgt15=float((d.pred.abs() > 1.5).mean())),
                rho_with_abserr={s: spearman(d[c].values, d.abserr.values) for s, c in SCORES.items()},
                rho_with_absPred={s: spearman(d[c].values, d.pred.abs().values) for s, c in SCORES.items()},
                rho_sigma_iphi=spearman(d.se.values, d.iphi.values),
                boot_ci={k: dict(mean=float(np.nanmean(v)), ci95=ci(v)) for k, v in bs.items()})
    return d, curves, per, summ

import sys, pickle
# usage: python binned_selective_prediction.py [singles|doubles|all|plot]  (split runs keep each call short)
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
SEL = {"singles": lambda x: x.num_muts == 1, "doubles": lambda x: x.num_muts >= 2}
if mode in ("singles", "doubles", "all"):
    val_all, nseeds = load("mega_val"); test_all, _ = load("mega_test"); print("seeds:", nseeds)
    for name in (["singles", "doubles"] if mode == "all" else [mode]):
        r = analyse(name, test_all[SEL[name](test_all)], val_all[SEL[name](val_all)])
        pickle.dump(r[1:], open(os.path.join(OUT, f"_res_{name}.pkl"), "wb")); print("saved", name)
    if mode != "all": sys.exit(0)
res = {n: (None,) + tuple(pickle.load(open(os.path.join(OUT, f"_res_{n}.pkl"), "rb"))) for n in ["singles", "doubles"]}
curves = pd.concat([r[1] for r in res.values()]); per = pd.concat([r[2] for r in res.values()])
curves.to_csv(os.path.join(OUT, "coverage_curves.csv"), index=False)
per.to_csv(os.path.join(OUT, "per_bin.csv"), index=False)
json.dump({k: r[3] for k, r in res.items()}, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)

# ---------------- figure
C = {"sigma_global": "#2a78d6", "sigma_within_bin": "#eb6834", "predonly_global": "#1baf7a", "predonly_within_bin": "#eda100",
     "iphi_global": "#e87ba4", "iphi_within_bin": "#008300"}
LAB = {"sigma_global": "σₑ, global", "sigma_within_bin": "σₑ, within bin", "iphi_global": "1/Φ, global", "iphi_within_bin": "1/Φ, within bin",
       "predonly_global": "pred-only, global", "predonly_within_bin": "pred-only, within bin", "random": "random", "oracle": "oracle"}
BARC = {"sigma": "#eb6834", "iphi": "#008300", "predonly": "#eda100"}
BARL = {"sigma": "σₑ", "iphi": "1/Φ", "predonly": "pred-only"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#555", "ytick.color": "#555"})
fig, ax = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
for row, name in enumerate(["singles", "doubles"]):
    d, cv, pb, sm = res[name]
    a = ax[row, 0]
    for r in ["sigma_global", "iphi_global", "predonly_global", "sigma_within_bin", "iphi_within_bin", "predonly_within_bin"]:
        s = cv[cv.rule == r]; a.plot(s.coverage, s.mae, color=C[r], lw=2, ls="-" if "global" in r else (0, (4, 1.5)), label=LAB[r])
    for r, ls in [("random", ":"), ("oracle", "--")]:
        s = cv[cv.rule == r]; a.plot(s.coverage, s.mae, color="#8a8a85", lw=1.5, ls=ls, label=LAB[r])
    a.set_xlabel("coverage (fraction kept)"); a.set_ylabel("MAE (kcal/mol)")
    a.set_title(f"{name}: MAE vs coverage (n={sm['n']}, {sm['n_proteins']} proteins)", loc="left", fontsize=9)
    a.grid(alpha=0.25, lw=0.5); a.invert_xaxis()
    if row == 0: a.legend(frameon=False, fontsize=6.5, ncol=1)
    a = ax[row, 1]; x = np.arange(len(pb)); w = 0.27
    for k, sname in enumerate(["sigma", "iphi", "predonly"]):
        off = (k - 1) * w
        y = pb[f"reldrop50_{sname}"].values * 100
        lo = np.array([c[0] for c in pb[f"reldrop50_{sname}_ci"]]) * 100; hi = np.array([c[1] for c in pb[f"reldrop50_{sname}_ci"]]) * 100
        a.bar(x + off, y, w * 0.9, color=BARC[sname], label=BARL[sname])
        a.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333", lw=0.9, capsize=1.5)
    a.axhline(0, color="#888", lw=0.8)
    a.set_xticks(x); a.set_xticklabels([f"{l}\nn={n}" for l, n in zip(pb.bin, pb.n)], fontsize=7)
    a.set_xlabel("predicted ΔΔG bin (kcal/mol)"); a.set_ylabel("MAE reduction at 50% kept (%)")
    a.set_title("within-bin filter, 50% kept (protein-boot. 95% CI)", loc="left", fontsize=9)
    if row == 0: a.legend(frameon=False, fontsize=7, loc="upper right")
    a = ax[row, 2]; w = 0.38
    for k, (sname, col) in enumerate([("sigma", C["sigma_global"]), ("iphi", C["iphi_global"])]):
        v = pb[f"retained_by_global_{sname}50"].values * 100
        a.bar(x + (k - 0.5) * w, v, w * 0.9, color=col, label=BARL[sname] + ", global")
        for xi, vv in zip(x + (k - 0.5) * w, v): a.text(xi, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=6.5, color="#222")
    a.axhline(50, color="#8a8a85", ls="--", lw=1); a.text(len(x) - 0.5, 51, "within-bin rule", ha="right", va="bottom", fontsize=7, color="#555")
    a.set_xticks(x); a.set_xticklabels(pb.bin, fontsize=7); a.set_ylim(0, 105)
    a.set_xlabel("predicted ΔΔG bin (kcal/mol)"); a.set_ylabel("% of bin kept")
    a.set_title("global filter, 50% kept: share of each bin kept", loc="left", fontsize=9)
    if row == 0: a.legend(frameon=False, fontsize=7, loc="upper right")
for ext in ["png", "pdf"]: fig.savefig(os.path.join(OUT, f"fig_binned_selective_prediction.{ext}"), dpi=200)
print("done")
