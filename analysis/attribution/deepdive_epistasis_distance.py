"""Does experimental (and predicted) non-additivity get stronger for closer residue pairs?
Input: megascale_test/tables/epistasis_per_variant.csv (matched doubles). d12 = Calpha-Calpha distance.
E = ddG_double - (ddG_A + ddG_B). Saturation removed per protein with a cubic fit of E on the additive sum."""
import ast, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import config as cfg
R = cfg.OUT_DIR
OUT = R / "deepdive_epistasis_distance"; (OUT / "tables").mkdir(parents=True, exist_ok=True)
d = pd.read_csv(R / "megascale_test/tables/epistasis_per_variant.csv").dropna(subset=["E_exp"])
def site(r):
    m = ast.literal_eval(r.mutation); return r.pdb + ":" + "-".join(f"{c}{p}" for c, p in sorted((c, p) for c, _, p, _ in m))
d["pair"] = d.apply(site, axis=1)
d["bin"] = pd.cut(d.d12, [0, 6, 9, np.inf], right=False, labels=["<6", "6-9", ">9"]).astype(str)

def resid(df, y, x, deg=3):
    out = pd.Series(np.nan, index=df.index)
    for p, g in df.groupby("pdb"):
        k = min(deg, max(1, len(g) // 20))
        out[g.index] = g[y] - np.polyval(np.polyfit(g[x], g[y], k), g[x])
    return out
for src, add in [("exp", "add_exp"), ("pred", "add_pred")]:
    d[f"r_{src}"] = resid(d, f"E_{src}", add)                     # per-protein saturation removed
    d[f"rg_{src}"] = d[f"E_{src}"] - np.polyval(np.polyfit(d[add], d[f"E_{src}"], 3), d[add])  # global fit
    d[f"rl_{src}"] = resid(d, f"E_{src}", add, deg=1)             # per-protein linear

agg = {"pdb": ("pdb", "first"), "d12": ("d12", "first"), "seq_sep": ("seq_sep", "first"), "n": ("E_exp", "size"),
       "mean_add_exp": ("add_exp", "mean")}
for c in ["E_exp", "E_pred", "r_exp", "r_pred", "rg_exp", "rg_pred", "rl_exp", "rl_pred"]:
    agg[f"mean_{c}"] = (c, "mean"); agg[f"medabs_{c}"] = (c, lambda x: np.median(np.abs(x))); agg[f"sd_{c}"] = (c, "std")
P = d.groupby("pair").agg(**agg).reset_index(); P = P[P.n >= 20]
P.to_csv(OUT / "tables/site_pair_strength.csv", index=False)

rng = np.random.default_rng(0)
def rho_ci(x, y, B=4000):
    x, y = np.asarray(x), np.asarray(y); r = spearmanr(x, y)[0]; n = len(x)
    bs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(B))]
    return [round(r, 3), round(np.nanpercentile(bs, 2.5), 3), round(np.nanpercentile(bs, 97.5), 3)]
def partial(x, y, z):  # rank partial: residualise ranks on z
    rk = lambda a: pd.Series(a).rank().values
    X, Y, Z = rk(x), rk(y), rk(z); A = np.c_[np.ones_like(Z), Z]
    ex = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]; ey = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    return round(np.corrcoef(ex, ey)[0, 1], 3)

K = {"n_variants": int(len(d)), "n_site_pairs": int(len(P)), "n_proteins": int(P.pdb.nunique()),
     "distance": "Calpha-Calpha", "site_pair_spearman_vs_d12 [rho, CI95 bootstrap over site pairs]": {}}
S = K["site_pair_spearman_vs_d12 [rho, CI95 bootstrap over site pairs]"]
for c in ["mean_E_exp", "medabs_E_exp", "sd_E_exp", "mean_r_exp", "medabs_r_exp", "sd_r_exp", "medabs_rg_exp", "medabs_rl_exp",
          "mean_E_pred", "medabs_E_pred", "sd_E_pred", "medabs_r_pred", "sd_r_pred", "medabs_rl_pred", "mean_add_exp"]:
    S[c] = rho_ci(P.d12, P[c])
K["partial_rho_vs_d12_given_mean_add_exp"] = {c: partial(P.d12, P[c], P.mean_add_exp) for c in
    ["mean_E_exp", "medabs_E_exp", "medabs_r_exp", "sd_r_exp", "medabs_r_pred", "sd_r_pred"]}
K["partial_rho_vs_d12_given_seq_sep"] = {c: partial(P.d12, P[c], P.seq_sep) for c in ["mean_E_exp", "medabs_r_exp", "sd_r_exp"]}
K["partial_rho_vs_d12_excluding_seq_sep_le2"] = {c: rho_ci(P[P.seq_sep > 2].d12, P[P.seq_sep > 2][c], 2000)
    for c in ["mean_E_exp", "medabs_r_exp", "sd_r_exp"]}
K["n_pairs_seq_sep_gt2"] = int((P.seq_sep > 2).sum())
# bin table (site-pair medians)
P["bin"] = pd.cut(P.d12, [0, 6, 9, np.inf], right=False, labels=["<6", "6-9", ">9"]).astype(str)
T = P.groupby("bin").agg(n_pairs=("pair", "size"), mean_E_exp=("mean_E_exp", "median"), medabs_E_exp=("medabs_E_exp", "median"),
    medabs_r_exp=("medabs_r_exp", "median"), sd_r_exp=("sd_r_exp", "median"), mean_add_exp=("mean_add_exp", "median"),
    medabs_r_pred=("medabs_r_pred", "median"), sd_r_pred=("sd_r_pred", "median")).round(3)
T.to_csv(OUT / "tables/bins_site_pair_medians.csv"); K["bins_site_pair_medians"] = T.reset_index().to_dict("records")
# variant level, per bin
V = d.groupby("bin").agg(n=("E_exp", "size"), mean_E_exp=("E_exp", "mean"), medabs_E_exp=("E_exp", lambda x: np.median(np.abs(x))),
    medabs_r_exp=("r_exp", lambda x: np.median(np.abs(x))), sd_r_exp=("r_exp", "std"),
    frac_absr_gt1=("r_exp", lambda x: np.mean(np.abs(x) > 1)), medabs_r_pred=("r_pred", lambda x: np.median(np.abs(x)))).round(3)
K["bins_variant_level"] = V.reset_index().to_dict("records")
K["model_vs_exp_strength_site_pair_rho"] = rho_ci(P.medabs_r_exp, P.medabs_r_pred)

# ---- side-chain-aware distances (epi_pair_distances.py) ----
D = pd.read_csv(OUT / "tables/site_pair_distances.csv")
P = P.merge(D[["pair", "d_ca", "d_cb", "d_heavy", "d_sc"]], on="pair", how="left")
K["distance_merge_missing"] = int(P.d_sc.isna().sum()); K["max_abs_d_ca_minus_d12"] = round(float((P.d_ca - P.d12).abs().max()), 3)
K["pairs_by_sidechain_distance"] = {"d_sc<4.5 (direct side-chain contact)": int((P.d_sc < 4.5).sum()),
    "4.5<=d_sc<6": int(((P.d_sc >= 4.5) & (P.d_sc < 6)).sum()), "d_sc>=6": int((P.d_sc >= 6).sum()),
    "d_heavy<4.5 (any heavy atom contact)": int((P.d_heavy < 4.5).sum())}
C = ["mean_E_exp", "medabs_E_exp", "medabs_r_exp", "sd_r_exp", "medabs_rg_exp", "medabs_r_pred", "sd_r_pred"]
K["site_pair_spearman_by_distance_metric"] = {dm: {c: rho_ci(P[dm], P[c], 2000) for c in C} for dm in ["d_cb", "d_heavy", "d_sc"]}
from scipy.stats import mannwhitneyu
cont = P.d_sc < 4.5
K["sidechain_contact_vs_not"] = {c: dict(contact_median=round(P.loc[cont, c].median(), 3), noncontact_median=round(P.loc[~cont, c].median(), 3),
    p_mannwhitney=round(mannwhitneyu(P.loc[cont, c], P.loc[~cont, c]).pvalue, 3)) for c in C}
P.to_csv(OUT / "tables/site_pair_strength.csv", index=False)
json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1); print(json.dumps(K, indent=1))
