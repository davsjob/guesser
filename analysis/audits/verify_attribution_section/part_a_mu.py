"""Steps 1-3: driving forces, burial gating, class rules (mu attributions, MegaScale-S forward test)."""
import numpy as np
import pandas as pd

from common import CHARGE, FI, OUTV3, PB, ci, cfg, load_A, load_v, r3, save

v = load_v()
A = load_A(("mut_mu", "ctx_mu"))
s_m = (v.n_mutated == 1).values
s = v[s_m].reset_index(drop=True)
Am, Ac = A["mut_mu"][s_m], A["ctx_mu"][s_m]
pb = PB(s.pdb)
stab, dest = (s.true_ddg < 0).values, (s.true_ddg > 1).values
R = {"n": {"singles": int(s_m.sum()), "proteins": int(s.pdb.nunique()),
           "stab_true": int(stab.sum()), "destab_true": int(dest.sum())}}


def shift(f, mask=None):
    m = np.ones(len(s), bool) if mask is None else mask
    a = Am[:, FI[f]]
    pd_, bd = pb.mean(a, m & dest)
    ps, bs = pb.mean(a, m & stab)
    return pd_ - ps, bd - bs


# ---------------------------------------------------------------- Step 1: driving forces
top4 = ["Hydrophobicity", "VdW Clashes", "Volume", "Polar Solvation"]
dr = {}
draws = {}
for f in top4:
    p, d = shift(f)
    draws[f] = d
    dr[f] = dict(shift=r3(p), ci=ci(d), ci_excludes_0=bool(np.nanpercentile(d, 2.5) > 0))
for a, b in zip(top4[:-1], top4[1:]):
    dd = draws[a] - draws[b]
    dr[f"{a} - {b}"] = dict(diff=r3(dr[a]["shift"] - dr[b]["shift"]), ci=ci(dd), p_order_holds=r3(np.mean(dd > 0)))
order = np.argsort(-np.column_stack([draws[f] for f in top4]), axis=1)
dr["p_full_order_H>C>V>P"] = r3(np.mean((order == np.arange(4)).all(1)))
dr["p_H_first_and_P_fourth"] = r3(np.mean((order[:, 0] == 0) & (order[:, 3] == 3)))
R["step1_driving"] = dr

# localisation: per-node environment IG share, site vs > 9 A (6.7x claim)
z = np.load(cfg.CACHE_DIR / "megascale_test" / "attr_ctx_mu.npz")
Ms, Ns = z["mass_shell"][s_m].astype(float), z["n_shell"][s_m].astype(float)
tot = Ms.sum()
share = Ms.sum(0) / tot
pernode = share / Ns.mean(0)
sS, _ = pb.sums(Ms[:, 0]); sF, _ = pb.sums(Ms[:, 3]); nS, _ = pb.sums(Ns[:, 0]); nF, _ = pb.sums(Ns[:, 3])
ratio_b = ((pb.W @ sS) / (pb.W @ nS)) / ((pb.W @ sF) / (pb.W @ nF))
zm = np.load(cfg.CACHE_DIR / "megascale_test" / "attr_mut_mu.npz")["mass_shell"][s_m].astype(float)
R["step1_localisation"] = dict(
    env_mass_share_by_shell=[r3(x) for x in share], env_per_node_share=[r3(x) for x in pernode],
    env_site_over_periphery=r3(pernode[0] / pernode[3]), ci=ci(ratio_b),
    mutation_IG_self_share=r3(zm[:, 0].sum() / zm.sum()))

# environment term and its SASA part by burial
env = {}
for b in ("core", "boundary", "surface"):
    mb = (s.rsa_class == b).values
    p1, d1 = pb.mean(Ac.sum(1), mb)
    p2, d2 = pb.mean(Ac[:, FI["SASA"]], mb)
    env[b] = dict(n=int(mb.sum()), env_total=r3(p1), env_total_ci=ci(d1), env_SASA=r3(p2), env_SASA_ci=ci(d2))
R["step1_environment_by_burial"] = env
top_env = {}
for ds in cfg.DATASETS:
    for sub in ("singles", "doubles"):
        p = OUTV3 / ds / "tables" / f"environment_{sub}.csv"
        if p.exists():
            t = pd.read_csv(p).sort_values("mean_abs", ascending=False)
            top_env[f"{ds}|{sub}"] = [t.feature.iloc[0], r3(t.mean_abs.iloc[0]), t.feature.iloc[1], r3(t.mean_abs.iloc[1])]
R["step1_top_env_feature_all_datasets"] = top_env

# ---------------------------------------------------------------- Step 2: burial gating
bur = {}
for f in ("Hydrophobicity", "Polar Solvation", "Volume", "VdW Clashes", "Electrostatics", "Hydrophobic Solvation"):
    row = {}
    dd = {}
    for b in ("core", "boundary", "surface"):
        p, d = shift(f, (s.rsa_class == b).values)
        dd[b] = d
        row[b] = dict(shift=r3(p), ci=ci(d))
    rat = dd["core"] / dd["surface"]
    row["core/surface"] = dict(ratio=r3(row["core"]["shift"] / row["surface"]["shift"]) if row["surface"]["shift"] else None,
                               ci=ci(rat), p_surface_le_0=r3(np.mean(dd["surface"] <= 0)))
    row["core-surface"] = dict(diff=r3(row["core"]["shift"] - row["surface"]["shift"]), ci=ci(dd["core"] - dd["surface"]))
    bur[f] = row
R["step2_burial"] = bur

vb = pd.read_csv(OUTV3 / "megascale_test" / "tables" / "volume_bidirectional_singles.csv")
R["step2_volume_bins"] = dict(max_abs_pred_minus_true=r3((vb.mean_pred_mu - vb.mean_true_ddg).abs().max()),
                              per_bin=[(r.dV_bin, int(r.n), r3(r.mean_volume_attr), r3(r.mean_pred_mu - r.mean_true_ddg))
                                       for r in vb.itertuples()])

# electrostatics vs charge change
cw = s.wt1.map(lambda x: CHARGE.get(x, 0)).values
cm = s.mt1.map(lambda x: CHARGE.get(x, 0)).values
chg = cw != cm
ae = np.abs(Am[:, FI["Electrostatics"]])
p1, d1 = pb.mean(ae, chg)
p0, d0 = pb.mean(ae, ~chg)
el = dict(n_charge_changing=int(chg.sum()), n_charge_preserving=int((~chg).sum()),
          mean_abs_elec_changing=r3(p1), ci_changing=ci(d1), mean_abs_elec_preserving=r3(p0), ci_preserving=ci(d0),
          ratio=r3(p1 / p0), ratio_ci=ci(d1 / d0),
          frac_variants_changing=r3(chg.mean()), frac_abs_elec_mass_in_changing=r3(ae[chg].sum() / ae.sum()))
se = Am[:, FI["Electrostatics"]]
el["signed_mean_changing"], el["signed_mean_preserving"] = r3(se[chg].mean()), r3(se[~chg].mean())
for nm, mk in (("charge introduced", (cw == 0) & (cm != 0)), ("charge removed", (cw != 0) & (cm == 0)),
               ("charge reversal", cw * cm < 0), ("neutral->neutral", (cw == 0) & (cm == 0))):
    p, d = pb.mean(ae, mk)
    el[f"mean_abs_elec | {nm}"] = dict(n=int(mk.sum()), mean=r3(p), ci=ci(d))
# electrostatic *input* change as the alternative explanation
dx = np.load(cfg.CACHE_DIR / "megascale_test" / "geom.npz")["dx_all"][s_m][:, FI["Electrostatics"]]
el["mean_abs_dX_elec_changing"], el["mean_abs_dX_elec_preserving"] = r3(np.abs(dx[chg]).mean()), r3(np.abs(dx[~chg]).mean())
for b in ("core", "boundary", "surface"):
    mb = (s.rsa_class == b).values
    a1, b1 = pb.mean(ae, chg & mb)
    a0, b0 = pb.mean(ae, ~chg & mb)
    el[f"ratio | {b}"] = dict(ratio=r3(a1 / a0), ci=ci(b1 / b0))
R["step2_electrostatics_charge"] = el

# ---------------------------------------------------------------- Step 3: class rules
cls_rows = {}
for c, feats in (("to Pro", ["Cyclic", "Backbone Entropy", "Hydrogen Bonds"]), ("to Gly", ["Volume"]),
                 ("hydrophobic→polar", ["Hydrophobicity"]), ("charge introduced", ["Hydrophobicity", "Polar Solvation"])):
    mc = (s.cls == c).values
    row = dict(n=int(mc.sum()), n_proteins=int(s.pdb[mc].nunique()))
    for f in feats:
        p, d = pb.mean(Am[:, FI[f]], mc)
        row[f] = dict(mean=r3(p), ci=ci(d))
    cls_rows[c] = row
R["step3_classes"] = cls_rows

# Pro x secondary structure (exposed = boundary + surface)
ss = pd.read_csv(OUTV3 / "deepdive_uncertainty_attribution" / "tables" / "sscheck_dssp_assignments.csv")
s2 = s.merge(ss[["pdb", "pos", "ss"]], left_on=["pdb", "pos1"], right_on=["pdb", "pos"], how="left")
assert len(s2) == len(s) and s2.ss.notna().all()
pss = {}
for exp_, me in (("exposed", (s2.rsa_class != "core").values), ("core", (s2.rsa_class == "core").values)):
    for st in ("coil", "helix N-term", "helix interior", "strand"):
        m = (s2.cls == "to Pro").values & me & (s2.ss == st).values
        p, d = pb.mean(Am[:, FI["Cyclic"]], m)
        pt, dt = pb.mean(s2.true_ddg.values, m)
        pss[f"{exp_} | {st}"] = dict(n=int(m.sum()), n_proteins=int(s2.pdb[m].nunique()), cyclic_mu_attr=r3(p),
                                     ci=ci(d), true_ddg=r3(pt), true_ci=ci(dt))
        if exp_ == "exposed":
            pss[f"{exp_} | {st}"]["_draws"] = d
hx, co = pss["exposed | helix interior"].pop("_draws"), pss["exposed | coil"].pop("_draws")
for k in list(pss):
    pss[k].pop("_draws", None)
pss["exposed helix interior - coil"] = dict(diff=r3(pss["exposed | helix interior"]["cyclic_mu_attr"]
                                                    - pss["exposed | coil"]["cyclic_mu_attr"]), ci=ci(hx - co),
                                           ci_overlap=bool(pss["exposed | helix interior"]["ci"][0]
                                                           <= pss["exposed | coil"]["ci"][1]))
R["step3_pro_by_ss"] = pss
save("results_part_a", R)
