"""Shared configuration for the attribution analysis (paths come from analysis/paths.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ATTRIBUTION_OUT, CACHE_DIR, EXPL_DIR, MEGASCALE_PDB_DIR, REF_DIR, PRED_DIR  # noqa: E402

PKG_DIR = Path(__file__).resolve().parent
OUT_DIR = ATTRIBUTION_OUT
SINGLES_LOOKUP = REF_DIR / "megascale_singles_lookup.json"
PDB_METADATA = REF_DIR / "pdb_metadata_cache.json"

# MegaScale structures give exact Calpha distances beyond the 9 A graph edges.
# Within 9 A the graph edges are exact (edge set == all Calpha pairs <= 9 A, verified).
PDB_DIRS = {"megascale_test": [MEGASCALE_PDB_DIR]}


def pdb_dir(ds):
    for p in PDB_DIRS.get(ds, []):
        if p.is_dir():
            return p
    return None

DATASETS = {
    "megascale_test": "megascale_test_explanations.h5",
    "s669": "s669_explanations.h5",
    "ssym": "ssym_explanations.h5",
    "ptmul": "ptmul_explanations.h5",
    "fphf": "fphf_explanations.h5",
}
DATASET_LABEL = {
    "megascale_test": "MegaScale test", "s669": "S669", "ssym": "Ssym",
    "ptmul": "PTMUL-D", "fphf": "FireProt-HF",
}

FEATURES = [
    "Hydrophobicity", "Volume", "Charge", "H_Bond_Donors", "H_Bond_Acceptors",
    "Entropy", "Polarizability", "Aromatic", "Sulfur", "Branched", "Cyclic",
    "SASA",
    "Disulfide bonds", "Hydrogen Bonds", "Electrostatics", "VdW Clashes",
    "Polar Solvation", "Hydrophobic Solvation", "VdW Energy", "Backbone Entropy",
    "Side Chain Entropy", "Peptide Bond Violations", "Rotamer Violation",
]
SASA_IDX = 11
MADRAX_IDX = list(range(12, 23))
FEATURE_GROUP = {f: ("Amino-acid property" if i < 11 else "Burial (SASA)" if i == 11 else "MadraX energy")
                 for i, f in enumerate(FEATURES)}

# Attribution arrays extracted from /node (family "mut" = mutation IG, "ctx" = environment IG).
ATTR_KEYS = {
    "mut_mu": "signed_attr_mu", "ctx_mu": "signed_attr_mu_wt",
    "mut_base": "signed_attr_base", "mut_epi": "signed_attr_epi",
    "mut_epistemic": "signed_attr_epistemic", "ctx_epistemic": "signed_attr_epistemic_wt",
    "mut_aleatoric": "signed_attr_aleatoric", "ctx_aleatoric": "signed_attr_aleatoric_wt",
}

# Stability tiers (kcal/mol, positive = destabilizing). Same cut-points as the v2 report for comparability.
TIER_STAB, TIER_NEUT_HI, TIER_DESTAB = 0.0, 0.5, 1.0
# Relative solvent accessibility strata (Tien et al. 2013 theoretical max ASA).
RSA_CORE, RSA_SURF = 0.20, 0.50
# Distance shells: Calpha distance to the nearest mutated residue.
SHELLS = [("self", None), ("0-6 A", 6.0), ("6-9 A", 9.0), (">9 A", float("inf"))]
N_BOOT = 1000
SEED = 20260924

TIEN_MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0, "GLN": 225.0,
    "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0, "THR": 172.0, "TRP": 285.0,
    "TYR": 263.0, "VAL": 174.0,
}
THREE_TO_ONE = {k: v for k, v in zip(
    ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
     "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"], "ARNDCQEGHILKMFPSTWYV")}

# Figure palette (validated reference palette, light mode).
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "neutral": "#b9b8b3",
    "surface": "#fcfcfb",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
