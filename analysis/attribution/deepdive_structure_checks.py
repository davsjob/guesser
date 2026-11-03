"""Two follow-up checks on the uncertainty attributions (MegaScale singles, forward only).

    PYTHONPATH=. python deepdive_structure_checks.py
Outputs -> <OUT_DIR>/deepdive_uncertainty_attribution/ (tables/sscheck_*.csv, key_numbers_structure_checks.json,
fig24_pro_gly_secondary_structure, fig25_core_cavity_structure_quality).

1. Does the uncertainty for to-Pro (and to-Gly) at exposed sites split by secondary structure?
   Secondary structure is NOT a model input. We assign it with a self-contained numpy re-implementation of
   the DSSP hydrogen-bond criterion (Kabsch & Sander 1983: E = 0.084*332*(1/rON + 1/rCH - 1/rOH - 1/rCN)
   < -0.5 kcal/mol; H/G/I from consecutive n-turns, E from parallel / antiparallel bridges) on the wild-type
   structure the model saw. States: helix (H, G, I; with the first turn = the first 3 residues of a helix
   segment labelled "helix N-term"), strand (E), coil.
2. Does the core-cavity uncertainty (carried by the VdW Energy attribution) track structure quality?
   Proxies: (a) site pLDDT of the wild-type AF2 model (B-factor; Rosetta designs excluded);
   (b) the mutant-model violation terms (change in Rotamer Violation + Peptide Bond Violations);
   (c) "unexpected" lost contacts: the residual of the Delta VdW Energy after removing what the substitution and
       its burial explain (group mean by wt>mt substitution), i.e. how idiosyncratic the modelled repacking is.
   And whether that uncertainty is well-founded (tracks |error|).
Attributions are model-internal, not causal.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, rankdata, spearmanr

import config as cfg
import figures  # noqa: F401
from deepdive_attribution_mechanisms import substitution_class

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_uncertainty_attribution"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
K = {}
SS_ORDER = ["helix interior", "helix N-term", "strand", "coil"]
SS_COL = {"helix interior": C["red"], "helix N-term": C["orange"], "strand": C["blue"], "coil": C["neutral"]}


# ---------------------------------------------------------------------------- DSSP (numpy)
def read_backbone(path):
    res, order = {}, []
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        an, rn, ch, ri = ln[12:16].strip(), ln[17:20], ln[21], int(ln[22:26])
        key = (ch, ri)
        if key not in res:
            res[key] = dict(resn=rn, B=float(ln[60:66]))
            order.append(key)
        if an in ("N", "CA", "C", "O"):
            res[key][an] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    order = [k for k in order if all(a in res[k] for a in ("N", "CA", "C", "O"))]
    return order, res


def dssp(order, res):
    n = len(order)
    N = np.array([res[k]["N"] for k in order])
    CA = np.array([res[k]["CA"] for k in order])
    Cc = np.array([res[k]["C"] for k in order])
    O = np.array([res[k]["O"] for k in order])
    H = N.copy()
    v = Cc[:-1] - O[:-1]
    H[1:] = N[1:] + v / np.linalg.norm(v, axis=1, keepdims=True)  # amide H along C=O of i-1
    # chain breaks
    brk = np.r_[False, np.linalg.norm(N[1:] - Cc[:-1], axis=1) > 2.5]
    H[brk] = N[brk]

    def d(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2) + 1e-6
    # E[i, j]: C=O of i accepts from N-H of j
    E = 0.084 * 332 * (1 / d(O, N) + 1 / d(Cc, H) - 1 / d(O, H) - 1 / d(Cc, N))
    hb = (E < -0.5)
    idx = np.arange(n)
    hb[idx[:, None] == idx[None, :]] = False
    for k in (-1, 1, 2):
        m = (idx[None, :] - idx[:, None]) == k
        if k in (-1, 1):
            hb[m] = False
    hb[:, 0] = False  # first residue has no amide H
    ss = np.array(["C"] * n, dtype=object)

    def turn(nn):
        t = np.zeros(n, bool)
        t[: n - nn] = hb[idx[: n - nn], idx[: n - nn] + nn]
        return t
    t3, t4, t5 = turn(3), turn(4), turn(5)
    helix = np.zeros(n, bool)
    for i in range(1, n - 4):
        if t4[i - 1] and t4[i]:
            helix[i:i + 4] = True
    g = np.zeros(n, bool)
    for i in range(1, n - 3):
        if t3[i - 1] and t3[i]:
            g[i:i + 3] = True
    ii = np.zeros(n, bool)
    for i in range(1, n - 5):
        if t5[i - 1] and t5[i]:
            ii[i:i + 5] = True
    # bridges
    br = np.zeros(n, bool)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if abs(i - j) < 3:
                continue
            par = (hb[i - 1, j] and hb[j, i + 1]) or (hb[j - 1, i] and hb[i, j + 1])
            anti = (hb[i, j] and hb[j, i]) or (hb[i - 1, j + 1] and hb[j - 1, i + 1])
            if par or anti:
                br[i] = True
    ss[g] = "G"
    ss[ii] = "I"
    ss[br] = "E"
    ss[helix] = "H"
    return ss


def assign_ss():
    pdir = cfg.pdb_dir(DS)
    if pdir is None:
        raise FileNotFoundError("MegaScale PDBs not found (GUESSER_DATA/pdb/megascale)")
    v = pd.read_csv(cfg.CACHE_DIR / DS / "variants.csv", usecols=["pdb"])
    rows = []
    for p in v.pdb.unique():
        order, res = read_backbone(pdir / p)
        ss = dssp(order, res)
        three = np.where(np.isin(ss, ["H", "G", "I"]), "helix", np.where(ss == "E", "strand", "coil"))
        seg_pos = np.zeros(len(ss), int)
        for i in range(len(ss)):
            seg_pos[i] = seg_pos[i - 1] + 1 if i > 0 and three[i] == "helix" and three[i - 1] == "helix" else (
                1 if three[i] == "helix" else 0)
        for (ch, ri), s8, s3, sp in zip(order, ss, three, seg_pos):
            lab = s3
            if s3 == "helix":
                lab = "helix N-term" if sp <= 3 else "helix interior"
            rows.append(dict(pdb=p, pos=ri, resn=res[(ch, ri)]["resn"], plddt=res[(ch, ri)]["B"], dssp8=s8, ss3=s3,
                             ss=lab))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "sscheck_dssp_assignments.csv", index=False)
    K["dssp_fractions"] = T.ss3.value_counts(normalize=True).round(3).to_dict()
    K["dssp_per_protein_helix_frac"] = T.groupby("pdb").ss3.apply(lambda s: round((s == "helix").mean(), 2)).to_dict()
    return T


# ---------------------------------------------------------------------------- data
def load(T):
    cd = cfg.CACHE_DIR / DS
    v = pd.read_csv(cd / "variants.csv")
    m = (v.n_mutated == 1).values
    s = v[m].reset_index(drop=True)
    As = np.load(cd / "attr_mut_epistemic.npz")["F_tot"][m]
    Am = np.load(cd / "attr_mut_mu.npz")["F_tot"][m]
    DX = np.load(cd / "geom.npz")["dx_all"][m]
    DXs = np.load(cd / "geom.npz")["dx_self"][m]
    s["cls"] = [substitution_class(w, mt) for w, mt in zip(s.wt1, s.mt1)]
    s["gm"] = s.f_mut_epistemic - s.f0_mut_epistemic
    s["g0"] = s.f0_mut_epistemic
    s["serr"] = s.pred_mu - s.true_ddg
    s = s.merge(T.rename(columns={"pos": "pos1"}), on=["pdb", "pos1"], how="left", validate="many_to_one")
    three = s.resn.map(cfg.THREE_TO_ONE)
    K["dssp_merge"] = dict(n=len(s), matched=int(s.ss.notna().sum()),
                           wt_resname_agrees=float((s.resn == s.wt1).mean()))
    return s, As, Am, DX, DXs


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    b = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(1000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


# ---------------------------------------------------------------------------- 1. Pro / Gly by SS
def check_ss(s, As, Am):
    iC, iV = F.index("Cyclic"), F.index("Volume")
    rows, res = [], {}
    for cls, key in (("to Pro", iC), ("to Gly", iV)):
        q = s[s.cls == cls].copy()
        q["sig_attr"], q["mu_attr"] = As[s.cls == cls][:, key], Am[s.cls == cls][:, key]
        q["exposure"] = np.where(q.rsa_class == "core", "core", "exposed (boundary+surface)")
        for exp_, g in q.groupby("exposure"):
            for ss_, h in g.groupby("ss"):
                mt, lo, hi = mean_ci(h.true_ddg)
                ms, slo, shi = mean_ci(h.epistemic_std)
                rows.append(dict(cls=cls, exposure=exp_, ss=ss_, n=len(h), n_sites=h.pos1.nunique() if False else
                                 h.groupby(["pdb", "pos1"]).ngroups, true=mt, true_lo=lo, true_hi=hi,
                                 pred=h.pred_mu.mean(), serr=h.serr.mean(), mae=h.abs_err.mean(),
                                 sigma_e=ms, sigma_lo=slo, sigma_hi=shi, sigma_inter=h.inter_epistemic_std.mean(),
                                 g0=h.g0.mean(), gm=h.gm.mean(), sig_attr=h.sig_attr.mean(), mu_attr=h.mu_attr.mean(),
                                 rho_err_sigma=spearmanr(h.abs_err, h.epistemic_std)[0] if len(h) >= 20 else np.nan))
            gg = g[g.ss.isin(SS_ORDER)]
            groups = [gg[gg.ss == k] for k in SS_ORDER if (gg.ss == k).sum() >= 10]
            eta = lambda col: np.var(gg[col].groupby(gg.ss).transform("mean")) / np.var(gg[col])
            res[f"{cls} | {exp_}"] = dict(
                n=len(gg),
                kruskal_p_true=float(f"{kruskal(*[h.true_ddg for h in groups]).pvalue:.2g}"),
                kruskal_p_pred=float(f"{kruskal(*[h.pred_mu for h in groups]).pvalue:.2g}"),
                kruskal_p_sigma=float(f"{kruskal(*[h.epistemic_std for h in groups]).pvalue:.2g}"),
                eta2_ss_true=round(eta("true_ddg"), 3), eta2_ss_pred=round(eta("pred_mu"), 3),
                eta2_ss_sigma=round(eta("epistemic_std"), 3), eta2_ss_abs_err=round(eta("abs_err"), 3),
                eta2_ss_signed_err=round(eta("serr"), 3),
                rho_abs_err_sigma=round(spearmanr(gg.abs_err, gg.epistemic_std)[0], 3))
            # does sigma split by SS once burial (RSA) and |mu| are held fixed? rank-regression partial eta
            if len(gg) > 60:
                Z = np.column_stack([rankdata(gg.rsa_mean), rankdata(gg.pred_mu), np.ones(len(gg))])
                for col in ("epistemic_std", "true_ddg", "serr"):
                    r = rankdata(gg[col]).astype(float)
                    r = r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
                    res[f"{cls} | {exp_}"][f"partial_eta2_ss_{col}_given_rsa_mu"] = round(
                        np.var(pd.Series(r).groupby(gg.ss.values).transform("mean")) / np.var(r), 3)
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "sscheck_pro_gly_by_ss.csv", index=False, float_format="%.4g")
    K["ss_tests"] = res
    K["ss_table"] = R.round(3).to_dict("records")
    return R


# ---------------------------------------------------------------------------- 2. core cavity vs structure quality
def check_cavity(s, As, Am, DX, DXs):
    iV, iE = F.index("Volume"), F.index("VdW Energy")
    iR, iP = F.index("Rotamer Violation"), F.index("Peptide Bond Violations")
    s = s.copy()
    s["dV"], s["dE"] = DX[:, iV], DX[:, iE]
    s["viol"] = DX[:, iR] + DX[:, iP]
    s["sigE"], s["muV"], s["sigV"] = As[:, iE], Am[:, iV], As[:, iV]
    s["sub"] = s.wt1 + ">" + s.mt1
    # "unexpected lost contacts": residual of dE after the substitution x burial group mean
    s["dE_resid"] = s.dE - s.groupby(["sub", "rsa_class"]).dE.transform("mean")
    cav = s[(s.rsa_class == "core") & (s.dV < -20)].copy()
    af = cav[cav.plddt > 1]  # AF2 models only (Rosetta designs have B = 0)
    out = dict(n_core_cavity=len(cav), n_af2=len(af), n_proteins=int(cav.pdb.nunique()))

    def rho(a, b):
        return round(spearmanr(a, b)[0], 3)

    def part(a, b, Z):
        ra, rb = rankdata(a).astype(float), rankdata(b).astype(float)
        Zr = np.column_stack([rankdata(z) for z in Z] + [np.ones(len(a))])
        ra -= Zr @ np.linalg.lstsq(Zr, ra, rcond=None)[0]
        rb -= Zr @ np.linalg.lstsq(Zr, rb, rcond=None)[0]
        return round(pearsonr(ra, rb)[0], 3)
    for lab, x, d in (("site pLDDT", "plddt", af), ("Δ violations (rotamer + peptide)", "viol", cav),
                      ("Δ VdW energy (lost contacts)", "dE", cav),
                      ("unexpected Δ VdW energy (residual)", "dE_resid", cav)):
        within = [spearmanr(g[x], g.sigE)[0] for _, g in d.groupby("pdb") if len(g) >= 30 and g[x].std() > 0]
        out[lab] = dict(rho_sigma_attr_vdw=rho(d[x], d.sigE), rho_sigma_e=rho(d[x], d.epistemic_std),
                        rho_abs_err=rho(d[x], d.abs_err), rho_signed_err=rho(d[x], d.serr),
                        within_protein_rho_sigma_attr=round(float(np.mean(within)), 3) if within else None,
                        n_proteins_within=len(within),
                        partial_sigma_attr_given_dV=part(d[x], d.sigE, [d.dV]))
    # is the VdW-energy uncertainty well-founded?
    out["sigma_attr_vdw_vs_abs_err"] = dict(
        rho=rho(cav.sigE, cav.abs_err),
        partial_given_dV_absmu=part(cav.sigE, cav.abs_err, [cav.dV, cav.pred_mu.abs()]),
        rho_signed=rho(cav.sigE, cav.serr),
        within_protein=round(float(np.mean([spearmanr(g.sigE, g.abs_err)[0] for _, g in cav.groupby("pdb")
                                            if len(g) >= 30])), 3))
    out["sigma_e_vs_abs_err_core_cavity"] = rho(cav.epistemic_std, cav.abs_err)
    # quartiles of the unexpected-contact residual
    cav["q_resid"] = pd.qcut(cav.dE_resid, 4, labels=["Q1 (fewer lost)", "Q2", "Q3", "Q4 (more lost)"])
    qt = cav.groupby("q_resid", observed=True).agg(n=("dE", "size"), dE_resid=("dE_resid", "mean"), sigE=("sigE", "mean"),
                                                   sigma_e=("epistemic_std", "mean"), mae=("abs_err", "mean"),
                                                   serr=("serr", "mean"), true=("true_ddg", "mean"),
                                                   pred=("pred_mu", "mean"), viol=("viol", "mean"))
    qt.to_csv(TAB / "sscheck_core_cavity_by_resid_quartile.csv", float_format="%.4g")
    out["by_resid_quartile"] = qt.round(3).reset_index().astype({"q_resid": str}).to_dict("records")
    cav["q_pl"] = pd.qcut(cav.plddt.where(cav.plddt > 1), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    qp = cav.groupby("q_pl", observed=True).agg(n=("dE", "size"), plddt=("plddt", "mean"), sigE=("sigE", "mean"),
                                                sigma_e=("epistemic_std", "mean"), mae=("abs_err", "mean"),
                                                serr=("serr", "mean"))
    qp.to_csv(TAB / "sscheck_core_cavity_by_plddt_quartile.csv", float_format="%.4g")
    out["by_plddt_quartile"] = qp.round(3).reset_index().astype({"q_pl": str}).to_dict("records")
    K["core_cavity"] = out
    return cav, qt, qp


# ---------------------------------------------------------------------------- figures
def fig_ss(R):
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.2), gridspec_kw=dict(wspace=0.55))
    for k, (cls, exp_) in enumerate((("to Pro", "exposed (boundary+surface)"), ("to Pro", "core"),
                                     ("to Gly", "exposed (boundary+surface)"), ("to Gly", "core"))):
        ax = axes[k]
        t = R[(R.cls == cls) & (R.exposure == exp_)].set_index("ss").reindex(SS_ORDER).dropna(subset=["n"])
        y = np.arange(len(t))
        ax.errorbar(t.true, y + 0.15, xerr=[t.true - t.true_lo, t.true_hi - t.true], fmt="o", color=C["ink"],
                    ms=4.5, lw=1, label="experimental ΔΔG")
        ax.plot(t.pred, y + 0.15, "D", mfc="white", mec=C["blue"], ms=4.5, mew=1.1, label="predicted μ")
        ax2 = ax.twiny()
        ax2.errorbar(t.sigma_e, y - 0.18, xerr=[t.sigma_e - t.sigma_lo, t.sigma_hi - t.sigma_e], fmt="s",
                     color=C["orange"], ms=4.5, lw=1, label="σ$_e$")
        ax2.set_xlim(0.2, 1.4)
        ax2.tick_params(axis="x", colors=C["orange"], labelsize=6.5)
        ax2.spines["top"].set_visible(True)
        ax2.spines["top"].set_color(C["orange"])
        ax2.grid(False)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}\n(n={int(n)})" for i, n in zip(t.index, t.n)], fontsize=6.5)
        ax.set_ylim(len(t) - 0.5, -0.6)
        ax.set_xlim(-0.2, 3.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("ΔΔG (kcal/mol)", fontsize=7)
        if k == 0:
            ax2.set_xlabel("σ$_e$", fontsize=7, color=C["orange"])
        ax.set_title(f"{cls}, {exp_.split(' ')[0]}", loc="left", fontsize=8, y=1.17)
        ax.text(-0.02, 1.2, "abcd"[k], transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], marker="s", ls="", color=C["orange"])], l1 + ["σ$_e$ (top axis)"],
               loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig24_pro_gly_secondary_structure.{ext}")
    plt.close(fig)


def fig_cav(cav, qt, qp):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    ax.scatter(cav.dE, cav.sigE, s=4, alpha=0.25, color=C["ink2"], lw=0)
    b = pd.qcut(cav.dE, 10)
    g = cav.groupby(b, observed=True).agg(x=("dE", "mean"), y=("sigE", "mean"))
    ax.plot(g.x, g.y, "-o", color=C["red"], ms=3.5, lw=1.6)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.set_xlabel("Δ VdW energy (mt − wt; + = lost contacts)", fontsize=7)
    ax.set_ylabel("VdW-energy attribution to log σ", fontsize=7.5)
    ax.set_title("core cavities (ΔV < −20 Å³)", loc="left", fontsize=8)
    ax.text(-0.02, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    ax = axes[1]
    x = np.arange(len(qt))
    ax.bar(x - 0.2, qt.sigE, 0.4, color=C["orange"], label="VdW-energy attribution to log σ")
    ax.bar(x + 0.2, qt.mae, 0.4, color=C["blue"], label="MAE (kcal/mol)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\nfewer lost\nthan expected", "Q2", "Q3", "Q4\nmore lost\nthan expected"], fontsize=6.5)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=6.3, loc="upper left", ncol=1)
    ax.set_title("unexpected lost contacts (residual)", loc="left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.text(-0.02, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    ax = axes[2]
    x = np.arange(len(qp))
    ax.bar(x - 0.2, qp.sigE, 0.4, color=C["orange"], label="VdW-energy attribution to log σ")
    ax.bar(x + 0.2, qp.mae, 0.4, color=C["blue"], label="MAE (kcal/mol)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n{p_:.0f}" for q, p_ in zip(["Q1 low", "Q2", "Q3", "Q4 high"], qp.plddt)], fontsize=6.5)
    ax.set_xlabel("site pLDDT quartile (AF2 models)", fontsize=7)
    ax.set_title("wild-type model confidence", loc="left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.text(-0.02, 1.05, "c", transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig25_core_cavity_structure_quality.{ext}")
    plt.close(fig)


def main():
    T = assign_ss()
    s, As, Am, DX, DXs = load(T)
    R = check_ss(s, As, Am)
    cav, qt, qp = check_cavity(s, As, Am, DX, DXs)
    json.dump(K, open(OUT / "key_numbers_structure_checks.json", "w"), indent=1, ensure_ascii=False, default=str)
    fig_ss(R)
    fig_cav(cav, qt, qp)
    pd.set_option("display.width", 250)
    print(json.dumps({k: K[k] for k in ("dssp_fractions", "dssp_merge", "ss_tests", "core_cavity")}, indent=1,
                     ensure_ascii=False, default=str))
    print(R.drop(columns=["true_lo", "true_hi", "sigma_lo", "sigma_hi"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
