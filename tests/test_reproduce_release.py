"""The released checkpoints reproduce the paper's per-member and ensemble outputs on 41 bundled
graphs (29 Ssym singles, 12 PTmul multi-mutants). Reference values come from the paper's runs."""
import os

import numpy as np
import pandas as pd
import torch

from guesser.inference import predict
from guesser.model import GUESSER_MODEL_KWARGS, GUESSER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-4  # float32 on CPU vs GPU


def test_all_checkpoints_load_strict(weights):
    assert len(weights) == 8
    for w in weights:
        m = GUESSER(node_features=23, **GUESSER_MODEL_KWARGS)
        m.load_state_dict(torch.load(w, weights_only=True), strict=True)


def test_reproduces_reference(example_graphs, weights):
    ref = pd.read_csv(os.path.join(ROOT, "examples", "example_reference.csv"))
    ens, members = predict(example_graphs, weights, device="cpu", return_members=True)
    assert list(ens["mutation"].astype(str)) == list(ref["mutation"].astype(str))
    for m, w in enumerate(weights):
        seed = os.path.basename(w)[len("guesser_"):-3]
        fwd = members[m][members[m].direction == "Forward"].reset_index(drop=True)
        for col, key in [("pred_ddg", "mu"), ("v", "v"), ("alpha", "alpha"), ("beta", "beta"),
                         ("pred_phys", "gamma_phys"), ("pred_base", "gamma_base"), ("pred_epi", "gamma_epi")]:
            np.testing.assert_allclose(fwd[col], ref[f"{key}_{seed}"], atol=TOL, rtol=0)
    np.testing.assert_allclose(ens["pred_ddg"], ref["pred_mu"], atol=TOL, rtol=0)
    for col in ("aleatoric_std", "intra_epistemic_std", "inter_epistemic_std", "epistemic_std", "total_std"):
        np.testing.assert_allclose(ens[col], ref[col], atol=TOL, rtol=0)


def test_structure_of_outputs(example_graphs, weights):
    ens, members = predict(example_graphs, weights[:2], device="cpu", return_members=True)
    m = members[0]
    singles = m[(m.direction == "Forward") & (m.num_muts == 1)]
    assert (singles["pred_epi"] == 0).all(), "epistasis head must be exactly zero for single mutants"
    assert (m["alpha"] > 1).all() and (m["v"] > 0).all() and (m["beta"] > 0).all()
    fwd = m[m.direction == "Forward"].reset_index(drop=True)
    np.testing.assert_allclose(fwd["pred_ddg"], fwd["pred_phys"] + fwd["pred_base"] + fwd["pred_epi"], atol=1e-5)
