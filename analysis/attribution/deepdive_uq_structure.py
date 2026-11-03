"""Deep dive: (A) are doubles more uncertain than singles? (B) does aleatoric uncertainty track structure quality?

    python deepdive_uq_structure.py
Writes tables + fig8_uq_structure_deepdive.{png,pdf} to <OUT_DIR>/deepdive_uq_structure/.
Forward predictions only. MegaScale structures are AlphaFold2 models (B-factor = pLDDT); the four Rosetta
designs (HEEH_*, HHH_*) have B = 0 and are excluded from pLDDT analyses. Benchmark structures are genuine
X-ray / NMR entries (method + resolution from archive/outputs/pdb_metadata_cache.json).
"""
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, wilcoxon

import config as cfg
import figures  # noqa: F401  (applies the shared rcParams)

C = cfg.C
OUT = cfg.OUT_DIR / "deepdive_uq_structure"
OUT.mkdir(parents=True, exist_ok=True)
META = cfg.PDB_METADATA


def partial(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    Zr = np.column_stack([rankdata(c) for c in Z] + [np.ones(len(x))])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return pearsonr(ex, ey)


def load_mega():
    v = pd.read_csv(cfg.CACHE_DIR / "megascale_test" / "variants.csv")
    with h5py.File(cfg.EXPL_DIR / cfg.DATASETS["megascale_test"], "r") as f:
        v["inv_total_evidence"] = 1.0 / (2 * f["prediction/pred_v"][:] + f["prediction/pred_alpha"][:])
    pdir = cfg.pdb_dir("megascale_test")
    B = {}
    for p in v.pdb.unique():
        d = {}
        for line in open(pdir / p):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                d[int(line[22:26])] = float(line[60:66])
        B[p] = d
    v["plddt_site"] = [np.nanmean([B[p].get(a, np.nan)] + ([B[p].get(b, np.nan)] if b > 0 else []))
                       for p, a, b in zip(v.pdb, v.pos1, v.pos2)]
    v["plddt_protein"] = v.pdb.map({p: np.mean(list(d.values())) for p, d in B.items()})
    return v


# ---------------------------------------------------------------------------- A. doubles vs singles
MEASURES = [("inter_epistemic_std", "σ_inter (ensemble spread)"), ("inv_total_evidence", "1/Φ (inverse total evidence)"),
            ("epistemic_std", "σ_epi (total epistemic)"), ("intra_epistemic_std", "σ_intra (NIG)"),
            ("aleatoric_std", "σ_alea")]


def part_a(v, K):
    S, D = v[v.n_mutated == 1], v[v.n_mutated == 2]
    rows = []
    for c, lab in MEASURES:
        pp = pd.DataFrame([(p, S[S.pdb == p][c].mean(), g[c].mean()) for p, g in D.groupby("pdb")],
                          columns=["pdb", "s", "d"])
        rows.append(dict(measure=c, label=lab, singles_mean=S[c].mean(), doubles_mean=D[c].mean(),
                         pct_mean=(D[c].mean() / S[c].mean() - 1) * 100,
                         singles_median=S[c].median(), doubles_median=D[c].median(),
                         pct_median=(D[c].median() / S[c].median() - 1) * 100,
                         pct_paired_protein=((pp.d / pp.s).mean() - 1) * 100,
                         n_proteins_doubles_higher=int((pp.d > pp.s).sum()), n_proteins=len(pp),
                         wilcoxon_p=wilcoxon(pp.d, pp.s).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "A_doubles_vs_singles.csv", index=False, float_format="%.5g")
    # stratified by predicted magnitude
    rows = []
    for lo, hi in [(0, .5), (.5, 1), (1, 2), (2, 3), (3, 99)]:
        ms, md = S.pred_mu.abs().between(lo, hi, inclusive="left"), D.pred_mu.abs().between(lo, hi, inclusive="left")
        rows.append(dict(abs_mu_bin=f"{lo}-{hi}", n_singles=int(ms.sum()), n_doubles=int(md.sum()),
                         **{f"{c}_singles": S[c][ms].mean() for c, _ in MEASURES},
                         **{f"{c}_doubles": D[c][md].mean() for c, _ in MEASURES}))
    pd.DataFrame(rows).to_csv(OUT / "A_by_abs_mu.csv", index=False, float_format="%.5g")
    K["A"] = dict(
        table={r.measure: dict(pct_mean=round(r.pct_mean, 1), pct_median=round(r.pct_median, 1),
                               pct_paired=round(r.pct_paired_protein, 1),
                               proteins_higher=f"{r.n_proteins_doubles_higher}/{r.n_proteins}",
                               wilcoxon_p=float(f"{r.wilcoxon_p:.2g}")) for r in t.itertuples()},
        mae_singles=round(S.abs_err.mean(), 3), mae_doubles=round(D.abs_err.mean(), 3),
        sigma_epi_over_mae_singles=round(S.epistemic_std.mean() / S.abs_err.mean(), 3),
        sigma_epi_over_mae_doubles=round(D.epistemic_std.mean() / D.abs_err.mean(), 3))
    return t


# ---------------------------------------------------------------------------- B. aleatoric vs structure quality
def part_b(v, K):
    af = v[v.plddt_protein > 1].copy()  # AlphaFold2 models only
    K["B_plddt"] = {"n_proteins": int(af.pdb.nunique())}
    for k, lab in [(1, "singles"), (2, "doubles")]:
        s = af[af.n_mutated == k]
        wp = [spearmanr(g.plddt_site, g.aleatoric_std)[0] for _, g in s.groupby("pdb")
              if len(g) >= 40 and g.plddt_site.nunique() > 3]
        prot = s.groupby("pdb").agg(pl=("plddt_protein", "first"), al=("aleatoric_std", "mean"))
        rp = pearsonr(prot.pl, prot.al)
        K["B_plddt"][lab] = dict(
            variant_rho=round(spearmanr(s.plddt_site, s.aleatoric_std)[0], 3),
            partial_rho_given_rsa=round(partial(s.plddt_site, s.aleatoric_std, [s.rsa_mean])[0], 3),
            partial_rho_given_rsa_absmu_intra=round(partial(s.plddt_site, s.aleatoric_std,
                                                            [s.rsa_mean, s.pred_mu.abs(), s.intra_epistemic_std])[0], 3),
            within_protein_median_rho=round(float(np.nanmedian(wp)), 3),
            within_protein_n_claim_direction=f"{int(np.sum(np.array(wp) < 0))}/{len(wp)}",
            protein_level_r=round(rp[0], 3), protein_level_p=round(rp[1], 4), protein_level_n=len(prot))
    K["B_plddt"]["pooled_singles_doubles_rho"] = round(spearmanr(af.plddt_site, af.aleatoric_std)[0], 3)
    K["B_plddt"]["pooled_partial_given_n_mutated"] = round(partial(af.plddt_site, af.aleatoric_std, [af.n_mutated])[0], 3)
    # benchmarks: genuine experimental structures
    meta = json.load(open(META))
    get = lambda p, k: (meta.get(p) or meta.get(p.upper()) or meta.get(p.lower()) or {}).get(k)
    bm = pd.concat([pd.read_csv(cfg.CACHE_DIR / d / "variants.csv").assign(ds=d) for d in ("s669", "ssym", "fphf", "ptmul")],
                   ignore_index=True)
    bm["method"], bm["resolution"] = bm.pdb.map(lambda p: get(p, "method")), bm.pdb.map(lambda p: get(p, "resolution"))
    x = bm[(bm.method == "X-RAY DIFFRACTION") & bm.resolution.notna()].copy()
    x["resolution"] = x.resolution.astype(float)
    x["alea_centred"] = x.aleatoric_std - x.groupby("ds").aleatoric_std.transform("mean")
    prot = x.groupby(["ds", "pdb"]).agg(res=("resolution", "first"), al=("alea_centred", "mean"),
                                        n=("resolution", "size")).reset_index()
    prot = prot[prot.n >= 3]
    prot.to_csv(OUT / "B_resolution_protein_level.csv", index=False, float_format="%.5g")
    per_ds = {}
    for ds, g in prot.groupby("ds"):
        r = spearmanr(g.res, g.al)
        per_ds[ds] = dict(rho=round(r[0], 3), p=round(r[1], 3), n_pdbs=len(g))
    nmr = bm[bm.method.fillna("").str.contains("NMR")]
    xr = bm[(bm.method == "X-RAY DIFFRACTION") & bm.ds.isin(nmr.ds.unique())]
    K["B_resolution"] = dict(
        n_xray_variants=len(x), pooled_variant_rho=round(spearmanr(x.resolution, x.aleatoric_std)[0], 3),
        dataset_centred_variant_rho=round(spearmanr(x.resolution, x.alea_centred)[0], 3),
        protein_level_all=dict(rho=round(spearmanr(prot.res, prot.al)[0], 3), n_pdbs=len(prot)), per_dataset=per_ds,
        nmr_vs_xray=dict(n_nmr_variants=len(nmr), n_nmr_pdbs=int(nmr.pdb.nunique()),
                         alea_nmr=round(nmr.aleatoric_std.mean(), 4), alea_xray=round(xr.aleatoric_std.mean(), 4)))
    mag = {}
    for ds in cfg.DATASETS:
        w = pd.read_csv(cfg.CACHE_DIR / ds / "variants.csv")
        rmse = np.sqrt((w.abs_err ** 2).mean())
        mag[ds] = dict(mean_alea=round(w.aleatoric_std.mean(), 4), rmse=round(rmse, 3),
                       alea_pct_of_rmse=round(w.aleatoric_std.mean() / rmse * 100, 1),
                       rho_alea_intra=round(spearmanr(w.aleatoric_std, w.intra_epistemic_std)[0], 3),
                       rho_alea_abs_err=round(spearmanr(w.aleatoric_std, w.abs_err)[0], 3))
    K["B_magnitude"] = mag
    return af, prot


# ---------------------------------------------------------------------------- figure
def figure(tA, af, prot):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw=dict(wspace=0.55, width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    y = np.arange(len(tA))[::-1]
    ax.axvline(0, color=C["ink2"], lw=0.6)
    for col, lab, c, mk in [("pct_mean", "mean", C["blue"], "o"), ("pct_median", "median", C["orange"], "s"),
                            ("pct_paired_protein", "paired per protein", C["aqua"], "D")]:
        ax.plot(tA[col], y, mk, color=c, ms=5.5, mec="white", mew=0.7, label=lab, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels(tA.label)
    ax.set_xlabel("Doubles vs singles (% difference)")
    ax.set_title("Is the model less confident on doubles?", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.text(-0.02, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    ax = axes[1]
    bins = [40, 70, 80, 85, 90, 95, 100]
    for k, lab, c in [(1, "singles", C["blue"]), (2, "doubles", C["orange"])]:
        s = af[af.n_mutated == k]
        b = pd.cut(s.plddt_site, bins)
        m = s.groupby(b, observed=True).agg(x=("plddt_site", "mean"), y=("aleatoric_std", "mean"), n=("aleatoric_std", "size"))
        m = m[m.n >= 30]
        ax.plot(m.x, m.y * 100, "-o", color=c, lw=2, ms=4.5, label=lab)
    b = pd.cut(af.plddt_site, bins)
    m = af.groupby(b, observed=True).agg(x=("plddt_site", "mean"), y=("aleatoric_std", "mean"), n=("aleatoric_std", "size"))
    m = m[m.n >= 30]
    ax.plot(m.x, m.y * 100, "--", color=C["ink2"], lw=1.2, label="pooled (paper)")
    ax.set_xlabel("pLDDT at the mutated site (AF2 model)")
    ax.set_ylabel("Mean σ_alea (×100, kcal/mol)")
    ax.set_title("Aleatoric σ vs pLDDT (MegaScale)", loc="left")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.text(-0.02, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    ax = axes[2]
    for i, (ds, g) in enumerate(prot.groupby("ds")):
        ax.scatter(g.res, g.al * 100, s=np.clip(g.n, 10, 120), color=cfg.SERIES[i], alpha=0.75, edgecolor="white",
                   lw=0.5, label=cfg.DATASET_LABEL[ds])
    ax.axhline(0, color=C["ink2"], lw=0.6)
    r = spearmanr(prot.res, prot.al)[0]
    ax.set_xlabel("X-ray resolution (Å)")
    ax.set_ylabel("Protein mean σ_alea, dataset-centred (×100)")
    ax.set_title(f"Aleatoric σ vs resolution (ρ={r:+.2f}, {len(prot)} PDBs)", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.text(-0.02, 1.04, "c", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig8_uq_structure_deepdive.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    K = {}
    v = load_mega()
    tA = part_a(v, K)
    af, prot = part_b(v, K)
    figure(tA, af, prot)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, default=str)
    print(json.dumps(K, indent=1, default=str))
