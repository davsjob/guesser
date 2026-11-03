"""Side-chain-aware distances for MegaScale double-mutant site pairs (WT structures)."""
import ast, numpy as np, pandas as pd
from pathlib import Path
import config as cfg
R = cfg.OUT_DIR
PDB = cfg.pdb_dir("megascale_test")
BB = {"N", "CA", "C", "O", "OXT"}
d = pd.read_csv(R / "megascale_test/tables/epistasis_per_variant.csv")
def read(p):
    at = {}
    for l in open(p):
        if l.startswith(("ATOM", "HETATM")):
            name, el = l[12:16].strip(), (l[76:78].strip() or l[12:16].strip()[0])
            if el == "H" or name.startswith("H"): continue
            if l[16] not in " A": continue
            k = (l[21], int(l[22:26]), l[26].strip())
            at.setdefault(k, []).append((name, np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return at
cache, rows = {}, []
for (pdb, mut), g in d.groupby(["pdb", "mutation"], sort=False):
    m = ast.literal_eval(mut); key = (pdb, tuple(sorted((c, p) for c, _, p, _ in m)))
    if key in cache: continue
    S = read(PDB / pdb) if pdb not in cache else cache[pdb]
    res = []
    for c, p in key[1]:
        a = S.get((c, p, "")); res.append(a)
    if any(r is None for r in res):
        cache[key] = None; continue
    def xyz(r, sel):
        return np.array([x for n, x in r if sel(n)])
    A, B = res
    heavy = np.linalg.norm(xyz(A, lambda n: True)[:, None] - xyz(B, lambda n: True)[None], axis=-1).min()
    scA = xyz(A, lambda n: n not in BB); scB = xyz(B, lambda n: n not in BB)
    if len(scA) == 0: scA = xyz(A, lambda n: n == "CA")
    if len(scB) == 0: scB = xyz(B, lambda n: n == "CA")
    sc = np.linalg.norm(scA[:, None] - scB[None], axis=-1).min()
    cb = lambda r: (xyz(r, lambda n: n == "CB") if any(n == "CB" for n, _ in r) else xyz(r, lambda n: n == "CA"))[0]
    ca = lambda r: xyz(r, lambda n: n == "CA")[0]
    cache[key] = dict(pdb=pdb, site_a=f"{key[1][0][0]}{key[1][0][1]}", site_b=f"{key[1][1][0]}{key[1][1][1]}",
                      d_ca=np.linalg.norm(ca(A) - ca(B)), d_cb=np.linalg.norm(cb(A) - cb(B)), d_heavy=heavy, d_sc=sc)
    rows.append(cache[key])
D = pd.DataFrame(rows); D["pair"] = D.pdb + ":" + D.site_a + "-" + D.site_b
out = R / "deepdive_epistasis_distance/tables"; out.mkdir(parents=True, exist_ok=True)
D.round(3).to_csv(out / "site_pair_distances.csv", index=False)
print(len(D), "pairs;", sum(v is None for v in cache.values() if not isinstance(v, dict)), "missing")
print(D.describe().round(2))
