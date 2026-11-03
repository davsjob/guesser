"""Explainer checks (ported from the audit suite, CPU and small step counts):
inputs unchanged, fast mode == reference mode, batch invariance, completeness."""
import numpy as np
import torch
from torch_geometric.data import Batch

from guesser.explain.core import TARGET_KEYS, explain_batch, load_explain_model

FAST = dict(fast=True, max_nodes_per_pass=50_000)


def _batch(pairs, idx):
    return Batch.from_data_list([pairs[i][0] for i in idx]), Batch.from_data_list([pairs[i][1] for i in idx])


def _rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def _setup(example_graphs, weights):
    pairs = [d["graph_pair"] for d in example_graphs]
    model = load_explain_model(weights[0], 23, torch.device("cpu"), torch.float64)
    return pairs, model


def test_inputs_unchanged(example_graphs, weights):
    pairs, model = _setup(example_graphs, weights)
    bw, bm = _batch(pairs, range(4))
    xw0, xm0 = bw.x.clone(), bm.x.clone()
    explain_batch(model, bw, bm, 8, **FAST)
    assert torch.equal(bw.x, xw0) and torch.equal(bm.x, xm0)


def test_fast_equals_reference(example_graphs, weights):
    pairs, model = _setup(example_graphs, weights)
    single = [i for i, (w, _) in enumerate(pairs) if int(w.mut_mask.sum()) == 1][:3]
    multi = [i for i, (w, _) in enumerate(pairs) if int(w.mut_mask.sum()) > 1][:3]
    ids = single + multi
    ref, _, _ = explain_batch(model, *_batch(pairs, ids), 16)
    fst, _, _ = explain_batch(model, *_batch(pairs, ids), 16, **FAST)
    worst = max(_rel(f[f"signed_attr_{k}{s}"], r[f"signed_attr_{k}{s}"])
                for f, r in zip(fst, ref) for k in TARGET_KEYS for s in ("", "_wt"))
    assert worst < 1e-8, worst


def test_batch_invariance_and_completeness(example_graphs, weights):
    pairs, model = _setup(example_graphs, weights)
    idx = [0, 1, 2, 35, 36]
    big, _, endpoint_err = explain_batch(model, *_batch(pairs, idx), 32, **FAST)
    assert endpoint_err < 1e-8
    for j, d in zip(idx, big):
        one = explain_batch(model, *_batch(pairs, [j]), 32, **FAST)[0][0]
        for k in ("mu", "aleatoric", "epistemic"):
            assert _rel(d[f"signed_attr_{k}"], one[f"signed_attr_{k}"]) < 1e-9
        mass = d["attr_mass_mut_mu"]
        assert abs(d["completeness_err_mut_mu"]) <= 1e-2 * max(mass, 1e-9)
