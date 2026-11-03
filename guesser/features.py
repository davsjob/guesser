"""
Node features (physchem_v1, 23 per residue):

    0-10   11 intrinsic amino-acid properties (data/amino_acid_properties.csv)
    11     SASA (Shrake-Rupley, absolute, from the wild-type structure)
    12-22  11 MadraX per-residue energy terms, in ForceField output order
"""
import os

import numpy as np
import pandas as pd

_PROPERTIES_CSV = os.path.join(os.path.dirname(__file__), 'data', 'amino_acid_properties.csv')
_df = pd.read_csv(_PROPERTIES_CSV, index_col='Residue')

INTRINSIC_NAMES = list(_df.columns)
AA_FEATURES = {residue: np.array(row, dtype=np.float32) for residue, row in _df.iterrows()}

MADRAX_NAMES = [
    "Disulfide bonds", "Hydrogen Bonds", "Electrostatics", "VdW Clashes",
    "Polar Solvation", "Hydrophobic Solvation", "VdW Energy", "Backbone Entropy",
    "Side Chain Entropy", "Peptide Bond Violations", "Rotamer Violation",
]
FEATURE_NAMES = INTRINSIC_NAMES + ["SASA"] + MADRAX_NAMES

NODE_DIM = len(FEATURE_NAMES)                       # 23
INTRINSIC_SLICE = slice(0, len(INTRINSIC_NAMES))    # 0-10
SASA_INDEX = len(INTRINSIC_NAMES)                   # 11
MADRAX_SLICE = slice(SASA_INDEX + 1, NODE_DIM)      # 12-22

# Feature-set ablations: columns are zeroed in both graphs (so x_diff is zero too),
# which keeps node_dim = 23 for the model, inference and the explainer.
FEATURE_SETS = ("all", "madrax_only", "no_madrax")


def apply_feature_set(graph_list, feature_set):
    """Zero the ablated columns in place for a list of {'graph_pair': (g_wt, g_mt)} items."""
    if feature_set == "all":
        return graph_list
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    cols = slice(0, SASA_INDEX + 1) if feature_set == "madrax_only" else MADRAX_SLICE
    for item in graph_list:
        for g in item['graph_pair']:
            if g.x.shape[1] != NODE_DIM:
                raise ValueError(f"feature_set expects {NODE_DIM}-dim physchem_v1 nodes, got {g.x.shape[1]}")
            g.x[:, cols] = 0.0
    return graph_list
