"""Numbers for figure_selection: per-rule values with 95% protein-bootstrap CIs (B = 1000).
Same definitions as calls_threeclass.py (Addendum 5, neutral class) and stab_twostage.py (Addendum 8, per protein).
Released ensemble, forward singles, sigma_e = sigma_intra, s* = 1.708. Usage: python fig_selection_data.py [<pred_dir> <outdir>] (defaults: GUESSER_DATA/predictions, results/figures/data)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import FIGURES_OUT, PRED_DIR
import sys, glob, numpy as np, pandas as pd
from scipy.special import ndtr as Phi
PRED, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (str(PRED_DIR), str(FIGURES_OUT / "data")); S = 1.708; B = 1000
def load(name):
    seeds = [pd.read_csv(f) for f in sorted(glob.glob(f"{PRED}/{name}_predictions_S*.csv"))]
    seeds = [d[d.direction == "Forward"].reset_index(drop=True) for d in seeds]
    base = seeds[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    for s in seeds[1:]: assert (s.mutation.values == base.mutation.values).all()
    st = lambda c: np.stack([s[c].values for s in seeds])
    v, a, b = st("v"), st("alpha"), st("beta")
    base["mu"] = st("pred_ddg").mean(0); base["s_intra"] = np.sqrt((b / (v * (a - 1))).mean(0))
    base["protein"] = base.pdb.str.replace(".pdb", "", regex=False)
    d = base[base.num_muts == 1].reset_index(drop=True); assert len(seeds) == 8; return d
cls = lambda y: np.where(y <= -0.5, 0, np.where(y >= 0.5, 2, 1))
rng = np.random.default_rng(8); rows = []
for name, f in [("MS-S", "mega_test"), ("FireProt-HF", "fireprot_hf")]:
    d = load(f); sd = S * d.s_intra.values; mu = d.mu.values
    # ---- a: predicted-neutral calls, keep the most confident fraction, % not neutral in experiment
    x = d[cls(mu) == 1].reset_index(drop=True); m = x.mu.values; sx = S * x.s_intra.values
    margin = 0.5 - np.abs(m) + 1e-9
    P = Phi((0.5 - m) / sx) - Phi((-0.5 - m) / sx) + 1e-9 * margin
    wrong = (cls(x.true_ddg.values) != 1).astype(float)
    prots = x.protein.unique(); g = {q: np.where(x.protein.values == q)[0] for q in prots}
    boots = [np.concatenate([g[q] for q in rng.choice(prots, len(prots))]) for _ in range(B)]
    def top(score, idx, cov):
        s, w = score[idx], wrong[idx]; o = np.lexsort((np.arange(len(s)), -s)); k = max(1, int(round(cov * len(s)))); return w[o[:k]].mean()
    for cov in np.round(np.arange(0.10, 0.501, 0.05), 2):
        bsd, obd = {}, {}
        for rn, sc in [("smallest |μ|", margin), ("P(neutral)", P)]:
            bs = np.array([top(sc, ix, cov) for ix in boots]); bsd[rn] = bs; obd[rn] = top(sc, np.arange(len(x)), cov)
            rows.append(dict(panel="a", set=name, rule=rn, x=cov, n=len(x), n_units=len(prots), value=obd[rn],
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), all_called=wrong.mean()))
        dd = bsd["P(neutral)"] - bsd["smallest |μ|"]
        rows.append(dict(panel="a", set=name, rule="difference", x=cov, n=len(x), n_units=len(prots), value=obd["P(neutral)"] - obd["smallest |μ|"],
                         lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
    # ---- b: per protein, top k predicted-stabilizing picks; % destabilizing (>= 0.5) and % stabilizing (<= -0.5)
    P0 = Phi(-mu / sd); y = d.true_ddg.values
    pidx = pd.factorize(d.protein)[0]; groups = [np.where(pidx == i)[0] for i in range(pidx.max() + 1)]
    for k in [3, 5, 10, 20, 50]:
        keep = [gg for gg in groups if len(gg) >= 2 * k]
        M = np.stack([np.bincount(rng.choice(len(keep), len(keep)), minlength=len(keep)) for _ in range(B)]).astype(float)
        res = {}
        for rn, sc in [("lowest μ", -mu), ("P(ΔΔG < 0)", P0)]:
            per = []
            for ix in keep:
                o = ix[np.lexsort((ix, -sc[ix]))][:k]; per.append([(y[o] >= 0.5).mean(), (y[o] <= -0.5).mean()])
            per = np.array(per); obs = per.mean(0); bt = (M @ per) / M.sum(1)[:, None]; res[rn] = (obs, bt)
            for j, met in enumerate(["destab", "stab"]):
                rows.append(dict(panel="b", set=name, rule=rn, metric=met, x=k, n_units=len(keep), value=obs[j],
                                 lo=np.percentile(bt[:, j], 2.5), hi=np.percentile(bt[:, j], 97.5)))
        (o1, b1), (o0, b0) = res["P(ΔΔG < 0)"], res["lowest μ"]
        for j, met in enumerate(["destab", "stab"]):
            rows.append(dict(panel="b", set=name, rule="difference", metric=met, x=k, n_units=len(keep), value=o1[j] - o0[j],
                             lo=np.percentile(b1[:, j] - b0[:, j], 2.5), hi=np.percentile(b1[:, j] - b0[:, j], 97.5)))
    print(name, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/figure_selection_numbers.csv", index=False)
