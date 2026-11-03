"""Driving forces and burial gating binned by the model's own prediction (2026-10-02, at David's request).
Tiers: predicted stabilizing mu < 0, predicted destabilizing mu > 1 kcal/mol (ensemble mean). Mutation IG on mu.
MegaScale-S forward test; 95% protein-bootstrap CIs (paired draws)."""
import json, sys, types
sys.modules.setdefault("h5py", types.ModuleType("h5py"))  # not needed here; avoids import error in this shell
import numpy as np
from common import FI, PB, ci, load_A, load_v, r3, HERE

v = load_v(); A = load_A(("mut_mu",))
s_m = (v.n_mutated == 1).values; s = v[s_m].reset_index(drop=True); Am = A["mut_mu"][s_m]
pb = PB(s.pdb)
stab, dest = (s.pred_mu < 0).values, (s.pred_mu > 1).values
R = {"n": dict(stab_pred=int(stab.sum()), destab_pred=int(dest.sum()))}
def mean(f, m): return pb.mean(Am[:, FI[f]], m)
def shift(f, mask=None):
    m = np.ones(len(s), bool) if mask is None else mask
    pdv, bd = mean(f, m & dest); ps, bs = mean(f, m & stab); return pdv - ps, bd - bs
feats = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Hydrophobic Solvation", "VdW Energy", "Electrostatics", "Cyclic", "Backbone Entropy"]
dr, draws = {}, {}
for f in feats:
    p, d = shift(f); draws[f] = d; dr[f] = dict(shift=r3(p), ci=ci(d))
for a, b in [("Hydrophobicity", "Volume"), ("Volume", "VdW Clashes"), ("VdW Clashes", "Polar Solvation")]:
    dd = draws[a] - draws[b]; dr[f"{a} - {b}"] = dict(diff=r3(np.nanmean(dd) * 0 + (dr[a]["shift"] - dr[b]["shift"])), ci=ci(dd), p_order=r3(np.mean(dd > 0)))
R["driving_pred"] = dr
bur = {}
for f in feats[:6] + ["Electrostatics"]:
    row, dd = {}, {}
    for b in ("core", "boundary", "surface"):
        p, d = shift(f, (s.rsa_class == b).values); dd[b] = d; row[b] = dict(shift=r3(p), ci=ci(d))
    row["core/surface"] = dict(ratio=r3(row["core"]["shift"] / row["surface"]["shift"]) if row["surface"]["shift"] else None, ci=ci(dd["core"] / dd["surface"]))
    row["core-surface"] = dict(diff=r3(row["core"]["shift"] - row["surface"]["shift"]), ci=ci(dd["core"] - dd["surface"]))
    bur[f] = row
R["burial_pred"] = bur
R["n_burial"] = {b: dict(n=int((s.rsa_class == b).sum()), stab=int(((s.rsa_class == b).values & stab).sum()), dest=int(((s.rsa_class == b).values & dest).sum())) for b in ("core", "boundary", "surface")}
# stabilizing signature: mean attribution among predicted-stabilizing, all and by burial; plus sum over all features
sig = {}
for f in feats:
    p, d = mean(f, stab); sig[f] = dict(all=dict(mean=r3(p), ci=ci(d)))
    for b in ("core", "surface"):
        p, d = mean(f, stab & (s.rsa_class == b).values); sig[f][b] = dict(mean=r3(p), ci=ci(d))
tot = Am.sum(1); p, d = pb.mean(tot, stab); q, e = pb.mean(tot, dest)
sig["SUM_all_features"] = dict(stab=dict(mean=r3(p), ci=ci(d)), destab=dict(mean=r3(q), ci=ci(e)))
R["stab_signature_pred"] = sig
json.dump(R, open(HERE / "results_part_d_predbins.json", "w"), indent=1)
print(json.dumps(R, indent=1))
