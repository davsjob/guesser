"""Doubles: stabilizing signature and driving shifts binned by the model's own prediction (2026-10-02).
Predicted stabilizing mu < 0, predicted destabilizing mu > 1 (ensemble mean). Mutation IG on mu, MegaScale-D forward test.
95% protein-bootstrap CIs (paired draws). Compared with singles on the same definitions."""
import json, sys, types
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
import numpy as np
from common import FI, F, PB, ci, load_A, load_v, r3, HERE

v = load_v(); A = load_A(("mut_mu",))["mut_mu"]
R = {}
for nm, k in (("singles", 1), ("doubles", 2)):
    m = (v.n_mutated == k).values; s = v[m].reset_index(drop=True); Am = A[m]
    pb = PB(s.pdb); stab, dest = (s.pred_mu < 0).values, (s.pred_mu > 1).values
    r = dict(n=int(len(s)), n_stab=int(stab.sum()), frac_stab=r3(stab.mean()), n_dest=int(dest.sum()),
             completeness_med_abs=r3(np.median(np.abs(Am.sum(1) - (s.pred_mu - s.mu_null)))))
    sig, sh, dr_sig, dr_sh = {}, {}, {}, {}
    for f in F:
        p, d = pb.mean(Am[:, FI[f]], stab); sig[f] = dict(mean=r3(p), ci=ci(d)); dr_sig[f] = d
        q, e = pb.mean(Am[:, FI[f]], dest); sh[f] = dict(shift=r3(q - p), ci=ci(e - d)); dr_sh[f] = e - d
    tot = Am.sum(1); p, d = pb.mean(tot, stab); q, e = pb.mean(tot, dest)
    r["sum_stab"] = dict(mean=r3(p), ci=ci(d)); r["sum_dest"] = dict(mean=r3(q), ci=ci(e))
    r["mu_stab"] = r3(s.pred_mu[stab].mean())
    r["stab_signature"] = dict(sorted(sig.items(), key=lambda x: x[1]["mean"])[:8])
    r["stab_signature_penalties"] = {f: sig[f] for f in ("VdW Clashes", "Polar Solvation", "Hydrophobic Solvation", "Electrostatics", "Backbone Entropy")}
    r["driving_shift"] = dict(sorted(sh.items(), key=lambda x: -x[1]["shift"])[:8])
    # volume vs hydrophobicity: difference and share of the two
    dv, dh = dr_sig["Volume"], dr_sig["Hydrophobicity"]
    r["stab_vol_minus_hyd"] = dict(diff=r3(sig["Volume"]["mean"] - sig["Hydrophobicity"]["mean"]), ci=ci(dv - dh), p_vol_more_negative=r3(np.mean(dv < dh)))
    r["stab_vol_share_of_two"] = dict(share=r3(sig["Volume"]["mean"] / (sig["Volume"]["mean"] + sig["Hydrophobicity"]["mean"])), ci=ci(dv / (dv + dh)))
    sv, shh = dr_sh["Volume"], dr_sh["Hydrophobicity"]
    r["shift_vol_minus_hyd"] = dict(diff=r3(sh["Volume"]["shift"] - sh["Hydrophobicity"]["shift"]), ci=ci(sv - shh), p_vol_larger=r3(np.mean(sv > shh)))
    r["shift_vol_share_of_two"] = dict(share=r3(sh["Volume"]["shift"] / (sh["Volume"]["shift"] + sh["Hydrophobicity"]["shift"])), ci=ci(sv / (sv + shh)))
    # burial of the predicted stabilizing (mean RSA class mix)
    r["stab_burial_mix"] = {b: r3(np.mean(s.rsa_class[stab] == b)) for b in ("core", "boundary", "surface")}
    R[nm] = r
json.dump(R, open(HERE / "results_part_f_doubles_predbins.json", "w"), indent=1)
print(json.dumps(R, indent=1))
