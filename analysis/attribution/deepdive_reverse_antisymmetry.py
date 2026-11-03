"""Do the forward rules flip for the reverse mutation? (MegaScale-S/-D test, existing predictions only)

No model is re-run. Uses
  - forward attributions and per-seed scalars from cache/megascale_test (explanations_v2),
  - the stored reverse predictions (per-member prediction CSVs, Reverse rows)
    (ensemble CSV + the 8 per-seed CSVs, which carry v, alpha, beta per seed),
  - the Ssym ensemble CSV (forward + reverse).
A physics rule should be antisymmetric: if a forward feature contributes +x to mu, the reverse prediction
should carry -x. A learned "mutations destabilize" prior would not flip.

    python deepdive_reverse_antisymmetry.py      # ~20 s
Outputs -> <OUT_DIR>/deepdive_reverse_antisymmetry/ (key_numbers.json, tables/*.csv, fig26)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

import config as cfg
import figures  # noqa: F401  (shared rcParams)
from deepdive_attribution_mechanisms import substitution_class

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_reverse_antisymmetry"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
PRED = cfg.PRED_DIR
SEEDS = range(42, 50)
N_BOOT = 1000
RNG = np.random.default_rng(cfg.SEED)
GROUPS = {
    "hydrophobic / solvation": ["Hydrophobicity", "Polar Solvation", "Hydrophobic Solvation"],
    "packing": ["Volume", "VdW Clashes", "VdW Energy"],
    "backbone": ["Cyclic", "Backbone Entropy", "Hydrogen Bonds"],
    "electrostatics": ["Electrostatics", "Charge"],
}
GROUPS["other"] = [f for f in F if f not in sum(GROUPS.values(), [])]
EPS = {f: (1e-6 if i < 12 else 0.05) for i, f in enumerate(F)}
K = {}


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def srho(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def boot_idx(pdb, n=N_BOOT):
    pdb = np.asarray(pdb)
    u = np.unique(pdb)
    by = {p: np.where(pdb == p)[0] for p in u}
    for _ in range(n):
        yield np.concatenate([by[p] for p in RNG.choice(u, len(u))])


def ci(vals):
    vals = np.asarray(vals, float)
    return [r3(np.nanpercentile(vals, 2.5)), r3(np.nanpercentile(vals, 97.5))]


def sig_from_seeds(mu, v, a, b):
    """mu, v, alpha, beta: [M, N]. Same definitions as the explanations_v2 H5."""
    intra2 = b / (v * (a - 1))
    s_intra = np.sqrt(intra2.mean(0))
    s_inter = mu.std(0, ddof=1)
    g = (0.5 * np.log(intra2)).mean(0)  # the IG target: mean_s log sigma_s
    return s_intra, s_inter, np.sqrt(s_intra ** 2 + s_inter ** 2), g


def ensemble_predictions(name):
    """Ensemble of the per-member prediction CSVs: pdb, mutation, direction, true_ddg, pred_ddg (mean), epistemic_std (member SD)."""
    seeds = [pd.read_csv(f) for f in sorted(PRED.glob(f"{name}_predictions_S*.csv"))]
    e = seeds[0][["pdb", "mutation", "direction", "true_ddg"]].copy()
    for t in seeds[1:]:
        assert (t.mutation.values == e.mutation.values).all() and (t.direction.values == e.direction.values).all()
    mu = np.array([t.pred_ddg.values for t in seeds])
    e["pred_ddg"] = mu.mean(0)
    e["epistemic_std"] = mu.std(0, ddof=1)   # the old ensemble files stored the member spread under this name
    return e


# ---------------------------------------------------------------------------- load
def load():
    v = pd.read_csv(cfg.CACHE_DIR / DS / "variants.csv")
    A = np.load(cfg.CACHE_DIR / DS / "attr_mut_mu.npz")["F_tot"]
    AS = np.load(cfg.CACHE_DIR / DS / "attr_mut_epistemic.npz")["F_tot"]
    DX = np.load(cfg.CACHE_DIR / DS / "geom.npz")["dx_all"]
    ens = ensemble_predictions("mega_test")
    fw = ens[ens.direction == "Forward"].reset_index(drop=True)
    rv = ens[ens.direction == "Reverse"].reset_index(drop=True)
    assert (fw.pdb.values == v.pdb.values).all() and (fw.mutation.values == v.mutation.values).all()
    assert (rv.pdb.values == v.pdb.values).all() and (rv.mutation.values == v.mutation.values).all()
    assert np.abs(rv.true_ddg.values + v.true_ddg.values).max() < 1e-5
    P = {d: {k: [] for k in ("mu", "v", "a", "b")} for d in ("Forward", "Reverse")}
    for s in SEEDS:
        t = pd.read_csv(PRED / f"mega_test_predictions_S{s}.csv")
        for d in P:
            q = t[t.direction == d].reset_index(drop=True)
            assert (q.mutation.values == v.mutation.values).all() and (q.pdb.values == v.pdb.values).all()
            for k, c in (("mu", "pred_ddg"), ("v", "v"), ("a", "alpha"), ("b", "beta")):
                P[d][k].append(q[c].values)
    P = {d: {k: np.array(x) for k, x in P[d].items()} for d in P}
    fi, fe, fs, fg = sig_from_seeds(*(P["Forward"][k] for k in ("mu", "v", "a", "b")))
    ri, re, rs, rg = sig_from_seeds(*(P["Reverse"][k] for k in ("mu", "v", "a", "b")))
    K["check"] = dict(
        max_abs_fwd_mu_vs_h5=r3(np.abs(P["Forward"]["mu"].mean(0) - v.pred_mu).max()),
        max_abs_fwd_sigma_e_vs_h5=r3(np.abs(fs - v.epistemic_std).max()),
        max_abs_fwd_logsigma_target_vs_h5=r3(np.abs(fg - v.f_mut_epistemic).max()),
        max_abs_rev_ensemble_vs_seed_mean=r3(np.abs(P["Reverse"]["mu"].mean(0) - rv.pred_ddg).max()),
        note="prediction CSV column 'epistemic_std' equals sigma_inter, not sigma_e "
             f"(max diff to sigma_inter {np.abs(fw.epistemic_std - v.inter_epistemic_std).max():.2e})")
    v["mu_f"], v["mu_r"] = v.pred_mu, P["Reverse"]["mu"].mean(0)
    v["true_f"], v["true_r"] = v.true_ddg, -v.true_ddg
    v["s_asym"] = v.mu_f + v.mu_r                         # 0 for a perfectly antisymmetric model
    v["sig_f"], v["sig_r"] = fs, rs
    v["sintra_f"], v["sintra_r"], v["sinter_f"], v["sinter_r"] = fi, ri, fe, re
    v["g_f"], v["g_r"] = fg, rg
    v["g0_f"] = v.f0_mut_epistemic
    v["gm_f"] = v.g_f - v.g0_f
    v["cls"] = [substitution_class(w, m) for w, m in zip(v.wt1, v.mt1)]
    v["cls_rev"] = [substitution_class(m, w) for w, m in zip(v.wt1, v.mt1)]
    v["amu"] = pd.cut(v.mu_f.abs(), [0, .5, 1, 2, 3, np.inf], right=False,
                      labels=["<0.5", "0.5-1", "1-2", "2-3", ">3"]).astype(str)
    v["tbin"] = pd.cut(v.true_f, [-np.inf, 0, 1, 3, np.inf], labels=["<0", "0-1", "1-3", ">3"]).astype(str)
    v["site"] = v.pdb + "_" + v.pos1.astype(str) + np.where(v.n_mutated == 2, "_" + v.pos2.astype(str), "")
    G = {g: A[:, [F.index(f) for f in fs_]].sum(1) for g, fs_ in GROUPS.items()}
    GS = {g: AS[:, [F.index(f) for f in fs_]].sum(1) for g, fs_ in GROUPS.items()}
    return v, A, AS, DX, G, GS


def subsets(v):
    return (("singles", (v.n_mutated == 1).values), ("doubles", (v.n_mutated == 2).values))


# ---------------------------------------------------------------------------- 1. global antisymmetry
def global_antisym(v):
    rows = []
    for sub, m in subsets(v):
        s = v[m]
        slope = lambda q: np.polyfit(q.mu_f, -q.mu_r, 1)[0]
        bs = [(s.iloc[i].s_asym.mean(), slope(s.iloc[i])) for i in boot_idx(s.pdb)]
        r = dict(subset=sub, n=len(s),
                 mean_asym=s.s_asym.mean(), mean_asym_ci=ci([b[0] for b in bs]),
                 mae_asym=s.s_asym.abs().mean(), sd_asym=s.s_asym.std(),
                 pcc_fwd_vs_neg_rev=pearsonr(s.mu_f, -s.mu_r)[0], scc_fwd_vs_neg_rev=spearmanr(s.mu_f, -s.mu_r)[0],
                 slope_neg_rev_on_fwd=slope(s), slope_ci=ci([b[1] for b in bs]),
                 frac_sign_flipped=((np.sign(s.mu_f) != np.sign(s.mu_r)) | (s.mu_f == 0)).mean(),
                 mean_mu_f=s.mu_f.mean(), mean_mu_r=s.mu_r.mean(), mean_true_f=s.true_f.mean(),
                 mean_mu_null=s.mu_null.mean(), rho_asym_vs_mu_null=srho(s.s_asym, s.mu_null),
                 fwd_pcc=pearsonr(s.mu_f, s.true_f)[0], fwd_mae=(s.mu_f - s.true_f).abs().mean(),
                 rev_pcc=pearsonr(s.mu_r, s.true_r)[0], rev_mae=(s.mu_r - s.true_r).abs().mean(),
                 rev_signed_err=(s.mu_r - s.true_r).mean(),
                 eta2_asym_by_site=s.groupby("site").s_asym.transform("mean").var() / s.s_asym.var(),
                 eta2_asym_by_protein=s.groupby("pdb").s_asym.transform("mean").var() / s.s_asym.var())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "antisymmetry_global.csv", index=False, float_format="%.4g")
    K["global"] = {r["subset"]: {k: (r3(x) if isinstance(x, (float, np.floating)) else x) for k, x in r.items()}
                   for r in rows}
    return t


# ---------------------------------------------------------------------------- 2. where the asymmetry lives
def where_asym(v):
    rows = []
    for sub, m in subsets(v):
        s = v[m]
        for col in ("cls", "rsa_class", "amu", "tbin"):
            if sub == "doubles" and col == "cls":
                continue
            for lev, q in s.groupby(col):
                if len(q) < 50:
                    continue
                b = [q.iloc[i].s_asym.mean() for i in boot_idx(q.pdb, 300)]
                rows.append(dict(subset=sub, by=col, level=lev, n=len(q), mean_mu_f=q.mu_f.mean(),
                                 mean_neg_mu_r=(-q.mu_r).mean(), mean_true_f=q.true_f.mean(),
                                 mean_asym=q.s_asym.mean(), asym_lo=ci(b)[0], asym_hi=ci(b)[1],
                                 flip_ratio=(-q.mu_r).mean() / q.mu_f.mean() if abs(q.mu_f.mean()) > 0.05 else np.nan,
                                 slope=np.polyfit(q.mu_f, -q.mu_r, 1)[0] if q.mu_f.std() > 0 else np.nan,
                                 sig_f=q.sig_f.mean(), sig_r=q.sig_r.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "asymmetry_by_group.csv", index=False, float_format="%.4g")
    K["by_group"] = {f"{r.subset}|{r.by}|{r.level}": dict(n=int(r.n), mu_f=r3(r.mean_mu_f), neg_mu_r=r3(r.mean_neg_mu_r),
                                                          asym=r3(r.mean_asym), ci=[r3(r.asym_lo), r3(r.asym_hi)],
                                                          sig_f=r3(r.sig_f), sig_r=r3(r.sig_r))
                     for r in t.itertuples()}
    return t


# ---------------------------------------------------------------------------- 3. which forward contributions flip
def flip_regression(v, G):
    """-mu_r = b0 + b_null * mu_null + sum_g b_g * A_g(forward).  b_g = 1: that contribution flips fully."""
    rows = []
    names = ["mu_null"] + list(GROUPS)
    for sub, m in subsets(v):
        s = v[m].reset_index(drop=True)
        X = np.column_stack([s.mu_null.values] + [G[g][m] for g in GROUPS] + [np.ones(len(s))])
        y = -s.mu_r.values
        fit = lambda idx: np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        b = fit(np.arange(len(s)))
        bb = np.array([fit(i) for i in boot_idx(s.pdb, 500)])
        r2 = 1 - np.var(y - X @ b) / np.var(y)
        for j, nm in enumerate(names):
            share = np.cov(X[:, j], s.mu_f - s.mu_null)[0, 1] / np.var(s.mu_f - s.mu_null, ddof=1) if j else np.nan
            rows.append(dict(subset=sub, term=nm, coef=b[j], lo=np.percentile(bb[:, j], 2.5),
                             hi=np.percentile(bb[:, j], 97.5), fwd_variance_share=share))
        rows.append(dict(subset=sub, term="intercept", coef=b[-1], lo=np.percentile(bb[:, -1], 2.5),
                         hi=np.percentile(bb[:, -1], 97.5)))
        rows.append(dict(subset=sub, term="R2", coef=r2))
        # per-burial version (singles): does the flip depend on burial?
        if sub == "singles":
            for bur in ("core", "boundary", "surface"):
                mm = (s.rsa_class == bur).values
                bb2 = np.linalg.lstsq(X[mm], y[mm], rcond=None)[0]
                for j, nm in enumerate(names):
                    rows.append(dict(subset=f"singles|{bur}", term=nm, coef=bb2[j]))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "flip_regression.csv", index=False, float_format="%.4g")
    K["flip_regression"] = {f"{r.subset}|{r.term}": [r3(r.coef), r3(r.lo), r3(r.hi)] for r in t.itertuples()}
    return t


# ---------------------------------------------------------------------------- 4. the four rules, forward vs reverse
def rule_cells(v, DX, G, GS):
    s = v[v.n_mutated == 1]
    idx = s.index.values
    dx = lambda f: DX[idx, F.index(f)]
    core, bnd, surf = (s.rsa_class == "core").values, (s.rsa_class == "boundary").values, (s.rsa_class == "surface").values
    cells = {
        "1 core hydrophobic loss": core & (dx("Hydrophobicity") < -EPS["Hydrophobicity"]),
        "1 core hydrophobic gain": core & (dx("Hydrophobicity") > EPS["Hydrophobicity"]),
        "1 boundary hydrophobic loss": bnd & (dx("Hydrophobicity") < -EPS["Hydrophobicity"]),
        "1 core charge introduced": core & (s.cls == "charge introduced").values,
        "1 core charge removed": core & (s.cls == "charge removed").values,
        "2 core cavity (dV < -20)": core & (dx("Volume") < -20),
        "2 core overpacking (dV > 20)": core & (dx("Volume") > 20),
        "2 boundary size loss (dV < -20)": bnd & (dx("Volume") < -20),
        "2 boundary size gain (dV > 20)": bnd & (dx("Volume") > 20),
        "3 core electrostatics worse": core & (dx("Electrostatics") > EPS["Electrostatics"]),
        "3 core electrostatics better": core & (dx("Electrostatics") < -EPS["Electrostatics"]),
        "4 to Pro, core": core & (s.cls == "to Pro").values,
        "4 to Pro, exposed": (bnd | surf) & (s.cls == "to Pro").values,
        "4 from Pro, exposed": (bnd | surf) & (s.cls == "from Pro").values,
        "to Gly, core": core & (s.cls == "to Gly").values,
        "from Gly, core": core & (s.cls == "from Gly").values,
    }
    rows = []
    for nm, mm in cells.items():
        q = s[mm]
        if len(q) < 30:
            continue
        b = np.array([(q.iloc[i].mu_f.mean(), (-q.iloc[i].mu_r).mean()) for i in boot_idx(q.pdb, 300)])
        ratio = b[:, 1] / b[:, 0]
        rows.append(dict(rule=nm, n=len(q), true_f=q.true_f.mean(), mu_f=q.mu_f.mean(), neg_mu_r=(-q.mu_r).mean(),
                         flip_ratio=(-q.mu_r).mean() / q.mu_f.mean(), flip_lo=np.percentile(ratio, 2.5),
                         flip_hi=np.percentile(ratio, 97.5), asym=q.s_asym.mean(),
                         err_f=(q.mu_f - q.true_f).mean(), err_r=(q.mu_r - q.true_r).mean(),
                         sig_f=q.sig_f.mean(), sig_r=q.sig_r.mean(), logsig_ratio=(q.g_r - q.g_f).mean(),
                         sintra_f=q.sintra_f.mean(), sintra_r=q.sintra_r.mean(),
                         sinter_f=q.sinter_f.mean(), sinter_r=q.sinter_r.mean(),
                         **{f"A_{g}": G[g][q.index].mean() for g in GROUPS},
                         **{f"Asig_{g}": GS[g][q.index].mean() for g in GROUPS}))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "rule_cells_fwd_vs_rev.csv", index=False, float_format="%.4g")
    K["rule_cells"] = {r.rule: dict(n=int(r.n), true_f=r3(r.true_f), mu_f=r3(r.mu_f), neg_mu_r=r3(r.neg_mu_r),
                                    flip_ratio=r3(r.flip_ratio), flip_ci=[r3(r.flip_lo), r3(r.flip_hi)],
                                    sig_f=r3(r.sig_f), sig_r=r3(r.sig_r))
                       for r in t.itertuples()}
    return t


# ---------------------------------------------------------------------------- 5. is the uncertainty symmetric?
def sigma_sym(v):
    out = {}
    for sub, m in subsets(v):
        s = v[m].reset_index(drop=True)
        X = np.column_stack([s.g0_f, s.gm_f, np.ones(len(s))])
        b = np.linalg.lstsq(X, s.g_r, rcond=None)[0]
        bb = np.array([np.linalg.lstsq(X[i], s.g_r.values[i], rcond=None)[0] for i in boot_idx(s.pdb, 500)])
        out[sub] = dict(
            rho_sig_f_vs_sig_r=r3(srho(s.sig_f, s.sig_r)), rho_logsig_target=r3(srho(s.g_f, s.g_r)),
            mean_sig_f=r3(s.sig_f.mean()), mean_sig_r=r3(s.sig_r.mean()),
            mean_sintra_f=r3(s.sintra_f.mean()), mean_sintra_r=r3(s.sintra_r.mean()),
            mean_sinter_f=r3(s.sinter_f.mean()), mean_sinter_r=r3(s.sinter_r.mean()),
            rho_sinter_f_r=r3(srho(s.sinter_f, s.sinter_r)), rho_sintra_f_r=r3(srho(s.sintra_f, s.sintra_r)),
            coef_site_baseline=r3(b[0]), coef_site_baseline_ci=ci(bb[:, 0]),
            coef_mutation_part=r3(b[1]), coef_mutation_part_ci=ci(bb[:, 1]),
            rho_err_sig_fwd=r3(srho((s.mu_f - s.true_f).abs(), s.sig_f)),
            rho_err_sig_rev=r3(srho((s.mu_r - s.true_r).abs(), s.sig_r)),
            rho_absasym_vs_sig_f=r3(srho(s.s_asym.abs(), s.sig_f)),
            rho_absasym_vs_sinter_f=r3(srho(s.s_asym.abs(), s.sinter_f)),
            rho_absasym_vs_abs_err_f=r3(srho(s.s_asym.abs(), (s.mu_f - s.true_f).abs())))
    K["sigma_symmetry"] = out


# ---------------------------------------------------------------------------- 6. Ssym (curated reverse set)
def ssym():
    e = ensemble_predictions("ssym")
    f = e[e.direction == "Forward"].reset_index(drop=True)
    r = e[e.direction == "Reverse"].reset_index(drop=True)
    ok = (f.pdb.values == r.pdb.values).all() and (f.mutation.values == r.mutation.values).all()
    if not ok:
        K["ssym"] = "forward/reverse rows not aligned; skipped"
        return
    s = f.pred_ddg.values + r.pred_ddg.values
    K["ssym"] = dict(n=len(f), n_pdb=int(f.pdb.nunique()), mean_asym=r3(s.mean()), mae_asym=r3(np.abs(s).mean()),
                     pcc_fwd_vs_neg_rev=r3(pearsonr(f.pred_ddg, -r.pred_ddg)[0]),
                     slope=r3(np.polyfit(f.pred_ddg, -r.pred_ddg, 1)[0]),
                     fwd_pcc=r3(pearsonr(f.pred_ddg, f.true_ddg)[0]), rev_pcc=r3(pearsonr(r.pred_ddg, r.true_ddg)[0]),
                     fwd_mae=r3((f.pred_ddg - f.true_ddg).abs().mean()), rev_mae=r3((r.pred_ddg - r.true_ddg).abs().mean()),
                     note="Ssym reverse inputs: see the benchmark pipeline for whether mutant structures were used")


# ---------------------------------------------------------------------------- figure
def figure(v, gl, wh, fr, rc):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1, 1.5]))
    s = v[v.n_mutated == 1]
    a = ax[0]
    a.hexbin(s.mu_f, -s.mu_r, gridsize=60, mincnt=1, bins="log", cmap="Greys")
    lim = [-2, 5]
    a.plot(lim, lim, color=C["red"], lw=0.8)
    a.set(xlim=lim, ylim=lim, xlabel="forward μ (kcal/mol)", ylabel="− reverse μ",
          title="a  Singles: forward vs −reverse")
    g = gl.set_index("subset").loc["singles"]
    a.text(0.03, 0.97, f"slope {g.slope_neg_rev_on_fwd:.2f}\nmean μf+μr {g.mean_asym:+.2f}",
           transform=a.transAxes, va="top", fontsize=7)
    a = ax[1]
    q = wh[(wh.subset == "singles") & (wh.by == "cls")].sort_values("mean_asym")
    y = np.arange(len(q))
    a.barh(y, q.mean_asym, color=C["blue"], height=0.6)
    a.errorbar(q.mean_asym, y, xerr=[q.mean_asym - q.asym_lo, q.asym_hi - q.mean_asym], fmt="none", ecolor=C["ink2"], lw=0.7)
    a.set_yticks(y, q.level)
    a.axvline(0, color=C["ink2"], lw=0.6)
    a.set(xlabel="mean μ_fwd + μ_rev (0 = antisymmetric)", title="b  Asymmetry by class (singles)")
    a = ax[2]
    q = fr[(fr.subset == "singles") & fr.term.isin(["mu_null"] + list(GROUPS))]
    y = np.arange(len(q))
    a.errorbar(q.coef, y, xerr=[q.coef - q.lo, q.hi - q.coef], fmt="o", color=C["violet"], ms=4, lw=0.8)
    a.axvline(1, color=C["red"], lw=0.7, ls="--")
    a.axvline(0, color=C["ink2"], lw=0.6)
    a.set_yticks(y, q.term)
    a.set(xlabel="coefficient in −μ_rev (1 = flips fully)", title="c  Which forward terms flip")
    a = ax[3]
    q = rc.iloc[::-1]
    y = np.arange(len(q))
    a.barh(y + 0.2, q.mu_f, height=0.38, color=C["orange"], label="forward μ")
    a.barh(y - 0.2, q.neg_mu_r, height=0.38, color=C["aqua"], label="− reverse μ")
    a.scatter(q.true_f, y, marker="|", color=C["ink"], s=60, zorder=3, label="experiment (fwd)")
    a.set_yticks(y, q.rule, fontsize=6.5)
    a.axvline(0, color=C["ink2"], lw=0.6)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=6.5)
    a.set(xlabel="kcal/mol", title="d  Rules: forward vs −reverse")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig26_reverse_antisymmetry.{ext}")
    plt.close(fig)


def main():
    v, A, AS, DX, G, GS = load()
    gl = global_antisym(v)
    wh = where_asym(v)
    fr = flip_regression(v, G)
    rc = rule_cells(v, DX, G, GS)
    sigma_sym(v)
    ssym()
    figure(v, gl, wh, fr, rc)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: K[k] for k in ("check", "global", "sigma_symmetry", "ssym")}, indent=1, default=str))


if __name__ == "__main__":
    main()
