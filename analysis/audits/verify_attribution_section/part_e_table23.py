"""SI table: driving shift by burial for all 23 features, predicted bins (mu<0 vs mu>1), MegaScale-S forward. 2026-10-02."""
import json, sys, types
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
import numpy as np
from common import F, FI, PB, ci, load_A, load_v, r3, HERE
v = load_v(); A = load_A(("mut_mu",)); s_m = (v.n_mutated == 1).values; s = v[s_m].reset_index(drop=True); Am = A["mut_mu"][s_m]
pb = PB(s.pdb); stab, dest = (s.pred_mu < 0).values, (s.pred_mu > 1).values
rows = {}
for f in F:
    rows[f] = {}
    for b in ("all", "core", "boundary", "surface"):
        m = np.ones(len(s), bool) if b == "all" else (s.rsa_class == b).values
        a = Am[:, FI[f]]; p1, d1 = pb.mean(a, m & dest); p0, d0 = pb.mean(a, m & stab); lo, hi = ci(d1 - d0)
        rows[f][b] = dict(shift=r3(p1 - p0), lo=lo, hi=hi)
json.dump(rows, open(HERE / "results_part_e_table23.json", "w"), indent=1)
order = sorted(F, key=lambda f: -abs(rows[f]["all"]["shift"] or 0))
nice = lambda f: f.replace("_", " ").replace("H Bond", "H-bond")
star = lambda r: f"${r['shift']:+.2f}$" + ("$^{*}$" if r["lo"] is not None and (r["lo"] > 0 or r["hi"] < 0) else "")
for f in order:
    print(f"\t\t{nice(f)} & " + " & ".join(star(rows[f][b]) for b in ("all", "core", "boundary", "surface")) + " \\\\")
