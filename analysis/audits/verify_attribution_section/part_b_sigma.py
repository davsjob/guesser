"""Step 4: uncertainty attribution, recomputed against sigma_e = sigma_intra (EDL only). MegaScale-S/-D forward test."""
import numpy as np
import pandas as pd

from common import FI, F, OUTV3, PB, RNG, boot_idx, cfg, ci, load_A, load_v, r3, save, srho
import deepdive_feature_direction as fd

v = load_v()
A = load_A(("mut_mu", "mut_epistemic", "ctx_epistemic"))
DX = np.load(cfg.CACHE_DIR / "megascale_test" / "geom.npz")["dx_all"]
R = {}


def wcov_share(pb, a, b):
    """cov(a, b)/var(b) with protein-bootstrap draws (weights)."""
    S = lambda x: pb.sums(x)[0]
    n = pb.sums(np.ones(len(a)))[0]
    sab, sa, sb, sbb = S(a * b), S(a), S(b), S(b * b)
    W = pb.W
    N = W @ n
    cov = (W @ sab) / N - (W @ sa) / N * (W @ sb) / N
    var = (W @ sbb) / N - ((W @ sb) / N) ** 2
    pt = (np.cov(a, b, bias=True)[0, 1]) / np.var(b)
    return pt, cov / var


# ---------------------------------------------------------------- 0. target check
for sub, nm in ((1, "singles"), (2, "doubles")):
    x = v[v.n_mutated == sub]
    R.setdefault("target", {})[nm] = dict(
        rho_g_log_sigma_intra=r3(srho(x.g, x.intra_epistemic_std)),
        rho_g_sigma_inter=r3(srho(x.g, x.inter_epistemic_std)),
        rho_g_old_total=r3(srho(x.g, x.epistemic_std)),
        mean_sigma_intra=r3(x.intra_epistemic_std.mean()), mean_old_total=r3(x.epistemic_std.mean()))

# ---------------------------------------------------------------- 1. site baseline share and coupling to |mu|
for sub, nm in ((1, "singles"), (2, "doubles")):
    m = (v.n_mutated == sub).values
    x = v[m].reset_index(drop=True)
    pb = PB(x.pdb)
    pt, d = wcov_share(pb, x.g0.values, x.g.values)
    out = dict(n=len(x), n_proteins=int(x.pdb.nunique()), site_share=r3(pt), site_share_ci=ci(d))
    if nm == "doubles":  # site-pair bootstrap as well
        pbp = PB(x.site)
        out["n_site_pairs"] = int(x.site.nunique())
        out["site_share_ci_sitepair"] = ci(wcov_share(pbp, x.g0.values, x.g.values)[1])
    bs = {k: [] for k in ("g0", "gm", "g", "sigma_intra")}
    am = x.pred_mu.abs().values
    for i in boot_idx(x.pdb.values, 300):
        for k in ("g0", "gm", "g"):
            bs[k].append(srho(x[k].values[i], am[i]))
        bs["sigma_intra"].append(srho(x.intra_epistemic_std.values[i], am[i]))
    for k in bs:
        col = x.intra_epistemic_std if k == "sigma_intra" else x[k]
        out[f"rho_{k}_absmu"] = r3(srho(col, am))
        out[f"rho_{k}_absmu_ci"] = ci(bs[k])
    # mutation part by burial
    for b in ("core", "boundary", "surface"):
        p, dd = pb.mean(x.gm.values, (x.rsa_class == b).values)
        out[f"mean_gm | {b}"] = dict(mean=r3(p), ci=ci(dd))
        out[f"_d_{b}"] = dd
    out["mean_gm | boundary - core"] = ci(out["_d_boundary"] - out["_d_core"])
    pa, da = pb.mean(x.gm.values)
    out["mean_gm_all"] = dict(mean=r3(pa), ci=ci(da))
    for b in ("core", "boundary", "surface"):
        out.pop(f"_d_{b}")
        mb = (x.rsa_class == b).values
        out[f"mean_sigma_intra | {b}"] = r3(x.intra_epistemic_std[mb].mean())
        out[f"mean_old_total | {b}"] = r3(x.epistemic_std[mb].mean())
    R[f"site_vs_mutation | {nm}"] = out

# ---------------------------------------------------------------- 2. "doubt added per rule" by burial (cells)
s_m = (v.n_mutated == 1).values
s = v[s_m].reset_index(drop=True)
As, Am, D = A["mut_epistemic"][s_m], A["mut_mu"][s_m], DX[s_m]
pb = PB(s.pdb)
CURVE = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Hydrophobic Solvation", "VdW Energy",
         "Electrostatics", "Charge", "Hydrogen Bonds", "Cyclic", "Backbone Entropy", "Entropy", "Aromatic"]
cells = []
for f in F:
    j, eps = FI[f], fd.EPS[f]
    for b in ("core", "boundary", "surface"):
        mb = (s.rsa_class == b).values
        for side, ms in (("decrease", D[:, j] < -eps), ("increase", D[:, j] > eps)):
            m = mb & ms
            if m.sum() >= fd.MIN_N and s.pdb[m].nunique() >= 3:
                p, d = pb.mean(As[:, j], m)
                pm, _ = pb.mean(Am[:, j], m)
                cells.append(dict(feature=f, burial=b, side=side, n=int(m.sum()), sig=p, mu=pm, draws=d,
                                  curve=f in CURVE))
C = pd.DataFrame(cells)
orig = pd.read_csv(OUTV3 / "deepdive_uncertainty_attribution" / "tables" / "direction_summary_sigma.csv")
rule = {"original_key_numbers": {"core": 0.006, "boundary": 0.048, "surface": 0.037}}


def summarise(Cs, weight=False):
    o = {}
    for b in ("core", "boundary", "surface"):
        q = Cs[Cs.burial == b]
        w = q.n.values.astype(float) if weight else np.ones(len(q))
        dr = np.average(np.vstack(q.draws.values), axis=0, weights=w)
        o[b] = dict(n_cells=len(q), mean=r3(np.average(q.sig, weights=w)), ci=ci(dr))
        o[f"_{b}"] = dr
    o["boundary - core"] = ci(o["_boundary"] - o["_core"])
    o["surface - core"] = ci(o["_surface"] - o["_core"])
    for b in ("core", "boundary", "surface"):
        o.pop(f"_{b}")
    return o


rule["13 curve features, cell mean (reproduces original)"] = summarise(C[C.curve])
rule["all features, cell mean"] = summarise(C)
rule["13 curve features, variant-weighted"] = summarise(C[C.curve], weight=True)
loo = {b: [] for b in ("core", "boundary", "surface")}
for f in CURVE:
    q = C[C.curve & (C.feature != f)]
    for b in loo:
        loo[b].append(q[q.burial == b].sig.mean())
rule["leave-one-feature-out range"] = {b: [r3(min(x)), r3(max(x))] for b, x in loo.items()}
rule["cells_boundary_gt_core_and_surface"] = None
piv = C[C.curve].pivot_table(index=["feature", "side"], columns="burial", values="sig")
piv = piv.dropna()
rule["n_feature_side_with_all_3_burials"] = int(len(piv))
rule["frac_feature_side_where_boundary_is_max"] = r3((piv.idxmax(1) == "boundary").mean())
rule["frac_feature_side_where_core_is_min"] = r3((piv.idxmin(1) == "core").mean())
rule["per_rule_table"] = [(i[0], i[1], r3(r.core), r3(r.boundary), r3(r.surface)) for i, r in piv.iterrows()]
R["doubt_per_rule"] = rule

# ---------------------------------------------------------------- 3. within-class rho(A_mu, A_sigma)
grp = s.cls.values


def within_class_rho(idx, j, f):
    ch = np.abs(D[idx, j]) > fd.EPS[f]
    a, b, g = Am[idx, j][ch], As[idx, j][ch], grp[idx][ch]
    r = []
    for c in np.unique(g):
        i = g == c
        if i.sum() >= 100 and a[i].std() > 1e-6 and b[i].std() > 1e-6:
            r.append(srho(a[i], b[i]))
    return np.mean(r) if r else np.nan, len(r)


wc = {}
for f in ("VdW Clashes", "Electrostatics", "Hydrogen Bonds", "Rotamer Violation", "Hydrophobicity", "Volume",
          "Cyclic", "Polar Solvation", "Hydrophobic Solvation", "VdW Energy"):
    j = FI[f]
    allidx = np.arange(len(s))
    pt, nc = within_class_rho(allidx, j, f)
    bs = [within_class_rho(i, j, f)[0] for i in boot_idx(s.pdb.values, 300)]
    wc[f] = dict(rho_within_class=r3(pt), n_classes=nc, ci=ci(bs), ci_excludes_0=bool(np.nanpercentile(bs, 97.5) < 0 or
                                                                                         np.nanpercentile(bs, 2.5) > 0))
R["within_class_rho_mu_sigma_attr | singles"] = wc

# ---------------------------------------------------------------- 4. Pro by secondary structure, with sigma_intra
ss = pd.read_csv(OUTV3 / "deepdive_uncertainty_attribution" / "tables" / "sscheck_dssp_assignments.csv")
s2 = s.merge(ss[["pdb", "pos", "ss"]], left_on=["pdb", "pos1"], right_on=["pdb", "pos"], how="left")
pss = {}
for exp_, me in (("exposed", (s2.rsa_class != "core").values), ("core", (s2.rsa_class == "core").values)):
    for st in ("coil", "helix N-term", "helix interior", "strand"):
        m = (s2.cls == "to Pro").values & me & (s2.ss == st).values
        a, da = pb.mean(s2.intra_epistemic_std.values, m)
        o, _ = pb.mean(s2.epistemic_std.values, m)
        e, de = pb.mean(s2.abs_err.values, m)
        pss[f"{exp_} | {st}"] = dict(n=int(m.sum()), sigma_intra=r3(a), sigma_intra_ci=ci(da), old_total=r3(o),
                                     mae=r3(e), mae_ci=ci(de), rho_err_sigma_intra=r3(srho(s2.abs_err[m], s2.intra_epistemic_std[m])))
R["pro_by_ss_sigma_intra"] = pss
ex = {k: v_ for k, v_ in pss.items() if k.startswith("exposed")}
R["pro_by_ss_sigma_intra"]["exposed: highest sigma_intra"] = max(ex, key=lambda k: ex[k]["sigma_intra"])
R["pro_by_ss_sigma_intra"]["exposed: highest MAE"] = max(ex, key=lambda k: ex[k]["mae"])

# ---------------------------------------------------------------- 5. class-level sigma with sigma_intra, and Fig. 4E check
cs = pd.read_csv(OUTV3 / "deepdive_attribution_mechanisms" / "class_mean_attr_log_sigma_epi.csv").set_index("cls")
cls_tab = {}
for c in sorted(s.cls.unique()):
    m = (s.cls == c).values
    p, d = pb.mean(s.intra_epistemic_std.values, m)
    pg, dg = pb.mean(s.gm.values, m)
    cls_tab[c] = dict(n=int(m.sum()), sigma_intra=r3(p), sigma_intra_ci=ci(d), old_total=r3(s.epistemic_std[m].mean()),
                      mean_gm_all23=r3(pg), mean_gm_ci=ci(dg),
                      fig4E_sum_over_12_features=r3(cs.loc[c].astype(float).sum()) if c in cs.index else None)
R["class_sigma"] = cls_tab
R["class_sigma_rank_by_sigma_intra"] = sorted(cls_tab, key=lambda c: -cls_tab[c]["sigma_intra"])[:3]

# ---------------------------------------------------------------- 6. environment localisation CI (table definition)
z = np.load(cfg.CACHE_DIR / "megascale_test" / "attr_ctx_mu.npz")
M, Nn = z["mass_shell"][s_m].astype(float), z["n_shell"][s_m].astype(float)
share = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
dens = np.where(Nn > 0, share / np.maximum(Nn, 1), np.nan)
h0, h3 = Nn[:, 0] > 0, Nn[:, 3] > 0
p0, d0 = pb.mean(np.nan_to_num(dens[:, 0]), h0)
p3, d3 = pb.mean(np.nan_to_num(dens[:, 3]), h3)
R["env_localisation_table_definition"] = dict(per_node_self=r3(p0), per_node_gt9=r3(p3), ratio=r3(p0 / p3), ci=ci(d0 / d3))
save("results_part_b", R)
