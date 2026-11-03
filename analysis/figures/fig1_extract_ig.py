"""Fig. 1E input: integrated-gradient explanations for the 2K5H single mutants at sequence position 40 (L40X).
Reads the MegaScale-test explanation file (7 GB; needs h5py) and writes a small npz with, for every L40X variant:
per-node mutation-IG of mu and of log sigma_e ([nodes, 23 features], mean over the 8 seeds), node residue ids,
edges, IG scalars (f, f0, completeness error) and the ensemble predictions.

    python fig1_extract_ig.py        # ~1 min; writes fig1_parts/2k5h_pos40_ig.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import EXPL_DIR
import ast, json
import h5py, numpy as np

H5 = str(EXPL_DIR / "megascale_test_explanations.h5")
PDB, POS = "2K5H.pdb", 40
f = h5py.File(H5, "r")
pdb = f["mutation/pdb"][:].astype(str)
mut = f["mutation/mutation"][:].astype(str)
nm = f["mutation/n_mutated"][:]
idx = [i for i in np.where((pdb == PDB) & (nm == 1))[0] if ast.literal_eval(mut[i])[0][2] == POS]
no, eo = f["mutation/node_offset"][:], f["mutation/edge_offset"][:]
feat = json.loads(f["meta"].attrs["feature_names"])
out = dict(feature_names=np.array(feat), index=np.array(idx), mutation=mut[idx], true_ddg=f["mutation/true_ddg"][idx])
for k in ["pred_mu", "intra_epistemic_std", "pred_gamma_base", "pred_gamma_phys"]:
    out[k] = f["prediction/" + k][idx]
for fam in ["mut"]:
    for t in ["mu", "epistemic"]:
        for s in ["f", "f0", "completeness_err"]:
            out[f"{s}_{fam}_{t}"] = f[f"ig/{s}_{fam}_{t}"][idx]
for j, i in enumerate(idx):
    r, e = slice(no[i], no[i + 1]), slice(eo[i], eo[i + 1])
    out[f"v{j}_res_id"] = f["node/res_id"][r]
    out[f"v{j}_res_name"] = f["node/res_name_3"][r].astype(str)
    out[f"v{j}_is_mut"] = f["node/is_mutated"][r]
    out[f"v{j}_attr_mu"] = f["node/signed_attr_mu"][r]
    out[f"v{j}_attr_sig"] = f["node/signed_attr_epistemic"][r]
    out[f"v{j}_dx"] = f["node/raw_node_features_mt"][r] - f["node/raw_node_features_wt"][r]
    out[f"v{j}_edge_index"] = f["edge/edge_index"][e]
    out[f"v{j}_edge_dist"] = f["edge/edge_attr"][e][:, 1]
np.savez_compressed("fig1_parts/2k5h_pos40_ig.npz", **out)
print(len(idx), "variants:", [ast.literal_eval(m)[0][3] for m in out["mutation"]])
