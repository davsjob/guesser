"""Step 3 sanity check of the numpy DSSP re-implementation used for the Pro x secondary-structure numbers.
(1) Three wwPDB entries (1PSE, 1TUC, 2K5H; GUESSER_DATA/reference/pdb_checks): compare with the wwPDB HELIX/SHEET records.
(2) Ideal alpha-helix and an isolated extended chain built from internal coordinates.
(3) Multi-model NMR files: read_backbone() keeps the LAST model's coordinates; compare with model 1."""
from pathlib import Path

import numpy as np

from common import REF_DIR, RESULTS, save
import deepdive_structure_checks as sc

PDB_CHECKS = REF_DIR / "pdb_checks"
TMP = RESULTS / "tmp"
TMP.mkdir(exist_ok=True)
R = {}


def model1(path, out):
    keep, seen = [], False
    for ln in open(path):
        if ln.startswith("MODEL"):
            if seen:
                break
            seen = True
        keep.append(ln)
    out.write_text("".join(keep))
    return out


def ref_labels(path):
    lab = {}
    for ln in open(path):
        if ln.startswith("HELIX"):
            ch, a, b = ln[19], int(ln[21:25]), int(ln[33:37])
            for r in range(a, b + 1):
                lab[(ch, r)] = "helix"
        if ln.startswith("SHEET"):
            ch, a, b = ln[21], int(ln[22:26]), int(ln[33:37])
            for r in range(a, b + 1):
                lab[(ch, r)] = "strand"
    return lab


def three(ss):
    return np.where(np.isin(ss, ["H", "G", "I"]), "helix", np.where(ss == "E", "strand", "coil"))


for name in ("1pse", "1tuc", "2k5h"):
    p = PDB_CHECKS / f"{name}.pdb"
    ref = ref_labels(p)
    res = {}
    for tag, path in (("model1", model1(p, TMP / f"{name}_m1.pdb")), ("as_read_by_pipeline", p)):
        order, rr = sc.read_backbone(path)
        ss3 = three(sc.dssp(order, rr))
        refl = np.array([ref.get(k, "coil") for k in order])
        res[tag] = dict(n=len(order), agreement_3state=round(float((ss3 == refl).mean()), 3),
                        strand_recall=round(float((ss3[refl == "strand"] == "strand").mean()), 3),
                        strand_precision=round(float((refl[ss3 == "strand"] == "strand").mean()), 3) if (ss3 == "strand").any() else None,
                        n_helix_pred=int((ss3 == "helix").sum()), n_helix_ref=int((refl == "helix").sum()),
                        n_strand_pred=int((ss3 == "strand").sum()), n_strand_ref=int((refl == "strand").sum()))
        res[tag]["_ss"] = ss3
    res["model1_vs_as_read_agreement"] = round(float((res["model1"].pop("_ss") == res["as_read_by_pipeline"].pop("_ss")).mean()), 3)
    R[name] = res


# ideal geometry
def place(a, b, c, bond, angle, tors):
    angle, tors = np.radians(angle), np.radians(tors)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.column_stack([bc, np.cross(n, bc), n])
    d2 = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(tors), bond * np.sin(angle) * np.sin(tors)])
    return c + m @ d2


def build(phi, psi, n=24):
    N, CA = np.array([0.0, 1.458, 0.0]), np.array([0.0, 0.0, 0.0])
    C = place(np.array([1.0, 1.458, 0.0]), N, CA, 1.525, 111.2, -60.0)
    rr, order = {}, []
    for i in range(n):
        Nn = place(N, CA, C, 1.329, 116.2, psi)
        O = C - 1.231 * ((CA - C) / np.linalg.norm(CA - C) + (Nn - C) / np.linalg.norm(Nn - C)) / np.linalg.norm(
            (CA - C) / np.linalg.norm(CA - C) + (Nn - C) / np.linalg.norm(Nn - C))
        rr[("A", i + 1)] = dict(N=N, CA=CA, C=C, O=O, resn="ALA", B=0.0)
        order.append(("A", i + 1))
        CAn = place(CA, C, Nn, 1.458, 121.7, 180.0)
        Cn = place(C, Nn, CAn, 1.525, 111.2, phi)
        N, CA, C = Nn, CAn, Cn
    return order, rr


for nm, (phi, psi) in (("ideal_alpha_helix", (-57.8, -47.0)), ("extended_single_chain", (-120.0, 130.0))):
    order, rr = build(phi, psi)
    ss = sc.dssp(order, rr)
    R[nm] = dict(dssp8="".join(ss), frac_H=round(float((ss == "H").mean()), 3), frac_E=round(float((ss == "E").mean()), 3))
save("results_part_a2_dssp", R)
print(R)
