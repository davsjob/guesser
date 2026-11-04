"""Is the doubles error distance-dependent (large error for close core pairs)? MegaScale doubles, GUESSER ensemble.
Site-pair bootstrap CIs; controls for experimental ddG (saturation-driven under-prediction) and burial."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import config as cfg
R = cfg.OUT_DIR; OUT = R / "deepdive_epistasis_distance"; (OUT / "tables").mkdir(parents=True, exist_ok=True)
v = pd.read_csv(cfg.CACHE_DIR / "megascale_test/variants.csv")
d = v[v.n_mutated == 2].copy()
d["pair"] = d.pdb + ":" + [f"{min(a,b)}-{max(a,b)}" for a, b in zip(d.pos1, d.pos2)]
d["err"] = d.pred_mu - d.true_ddg
d["dbin"] = pd.cut(d.d12, [0, 6, 9, np.inf], right=False, labels=["<6", "6-9", ">9"]).astype(str)
d["bur"] = pd.cut(d.rsa_mean, [-np.inf, cfg.RSA_CORE, cfg.RSA_SURF, np.inf], right=False, labels=["core", "boundary", "surface"]).astype(str)
d["tbin"] = pd.cut(d.true_ddg, [-np.inf, 1, 2, 3, np.inf], labels=["<1", "1-2", "2-3", ">3"]).astype(str)
Dsc = pd.read_csv(OUT / "tables/site_pair_distances.csv"); Dsc["pair"] = Dsc.pdb + ":" + [f"{a[1:]}-{b[1:]}" for a, b in zip(Dsc.site_a, Dsc.site_b)]
d = d.merge(Dsc[["pair", "d_sc"]], on="pair", how="left")
rng = np.random.default_rng(0)
def boot(g, f, B=2000):
    pr = g.pair.unique(); ix = {p: np.where(g.pair.values == p)[0] for p in pr}
    return np.nanpercentile([f(g.iloc[np.concatenate([ix[p] for p in rng.choice(pr, len(pr))])]) for _ in range(B)], [2.5, 97.5]).round(3).tolist()
mae = lambda g: g.err.abs().mean(); bias = lambda g: g.err.mean()
K = {"n": len(d), "n_pairs": int(d.pair.nunique()), "d_sc_missing": int(d.d_sc.isna().sum())}
def table(keys):
    rows = []
    for k, g in d.groupby(keys):
        if len(g) < 100 or g.pair.nunique() < 3: continue
        rows.append(dict(zip(keys if isinstance(keys, list) else [keys], k if isinstance(k, tuple) else [k]), n=len(g), pairs=g.pair.nunique(),
                         mae=round(mae(g), 3), mae_ci=boot(g, mae), bias=round(bias(g), 3), bias_ci=boot(g, bias),
                         mean_true=round(g.true_ddg.mean(), 2), mean_pred=round(g.pred_mu.mean(), 2)))
    return rows
K["by_distance"] = table("dbin"); K["by_burial"] = table("bur"); K["by_distance_x_burial"] = table(["dbin", "bur"])
K["by_true_ddG"] = table("tbin"); K["by_distance_x_true_ddG"] = table(["tbin", "dbin"])
# residual error after regressing on true ddG (saturation/shrinkage), per distance
c = np.polyfit(d.true_ddg, d.err, 1); d["err_res"] = d.err - np.polyval(c, d.true_ddg)
K["err_vs_true_fit"] = c.round(3).tolist()
K["residual_abs_err_by_distance"] = {b: dict(mean_abs_res=round(g.err_res.abs().mean(), 3), ci=boot(g, lambda x: x.err_res.abs().mean()), bias_res=round(g.err_res.mean(), 3))
                                     for b, g in d.groupby("dbin")}
P = d.groupby("pair").agg(d12=("d12", "first"), d_sc=("d_sc", "first"), rsa=("rsa_mean", "mean"), mae=("err", lambda x: x.abs().mean()),
                          bias=("err", "mean"), true=("true_ddg", "mean"), res=("err_res", lambda x: x.abs().mean()), n=("err", "size"))
S = {}
for col in ["mae", "bias", "res"]:
    for dm in ["d12", "d_sc", "rsa", "true"]:
        S[f"{col}~{dm}"] = round(spearmanr(P[dm], P[col], nan_policy="omit")[0], 3)
K["site_pair_spearman"] = S
P.round(3).to_csv(OUT / "tables/doubles_error_by_site_pair.csv")
json.dump(K, open(OUT / "doubles_error_key_numbers.json", "w"), indent=1, default=float)
print(json.dumps(K, indent=1, default=float))
