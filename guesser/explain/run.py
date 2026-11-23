#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-seed Integrated-Gradients explanations for one ensemble member (format v2).

    python -m guesser.explain.run --dataset graphs.pt --weight_file weights/guesser_S42.pt --seed 42 \
        --pdb_dirs pdbs/ --out_prefix out/ssym_S42

v1 problems fixed in v2 (kept for the record):
  * gamma_phys IG crashed on every batch and was silently stored as zeros
  * node -> residue labels were wrong (graphs are local neighbourhoods, not whole PDBs)
  * WT baseline was the mean over the whole *batch* (batch-size dependent)
  * uncertainty wrapper detached z_epi, which the trained model does not do
  * float16 storage, no completeness check

Math and design: see guesser/explain/core.py. Residue mapping:
guesser/explain/residue_mapping.py. One run = one ensemble member; combine with
guesser.explain.merge.

Outputs:
  {out_prefix}_raw_data.pt   all per-mutation attribution matrices (float32) + checks
  {out_prefix}_summary.csv   per-mutation predictions, completeness residuals, mapping status
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..datasets import CoordinateNoiseDataset, GraphPairDataset, collate_pairs
from ..features import FEATURE_NAMES
from .core import (
    TARGETS, TARGET_KEYS, GUESSER_MODEL_KWARGS, explain_batch, file_md5, load_explain_model,
)
from .residue_mapping import ResidueMapper
PRED_KEYS = ['mu', 'gamma_phys', 'gamma_base', 'gamma_epi', 'v', 'alpha', 'beta']
CORE_FILES = ['explain/run.py', 'explain/core.py', 'explain/residue_mapping.py', 'model.py']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-seed IG explainer (v2)")
    p.add_argument("--dataset", required=True, help="Cached PyG dataset (.pt)")
    p.add_argument("--weight_file", required=True, help="One ensemble checkpoint, e.g. weights/guesser_S42.pt")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pdb_dirs", nargs='+', required=True,
                   help="Candidate PDB folders, tried in order until the graph verifies")
    p.add_argument("--cutoff", type=float, default=9.0,
                   help="Neighbourhood radius and edge cutoff the graphs were built with (default 9.0)")
    p.add_argument("--n_steps", type=int, default=64,
                   help="Gauss-Legendre IG nodes (64: worst completeness residual <0.3%% of "
                        "attribution mass in the audit tests)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Results do not depend on this (per-graph baselines; tested exactly in fp64)")
    p.add_argument("--max_nodes_per_pass", type=int, default=5_000,
                   help="Quadrature nodes are stacked into one forward pass up to this many graph "
                        "nodes (50000 ~ 37 GiB peak in fp64, used for the paper on a GH200). "
                        "Speed/memory only, no effect on results")
    p.add_argument("--precision", choices=["fp64", "fp32"], default="fp64",
                   help="fp64 evaluates the trained (fp32) weights exactly in double precision; "
                        "removes round-off from gradients (~25%% slower on GH200)")
    p.add_argument("--max_batches", type=int, default=None, help="Stop early (testing)")
    p.add_argument("--reference_csv", default=None,
                   help="Benchmark predictions CSV of the same seed (Forward rows) to cross-check")
    p.add_argument("--out_prefix", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Device: {device} | Gauss-Legendre nodes: {args.n_steps} | batch: {args.batch_size}")

    graphs = torch.load(args.dataset, weights_only=False)
    base_dataset = GraphPairDataset(graphs)
    loader = DataLoader(CoordinateNoiseDataset(base_dataset, use_noise=False),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    node_dim = base_dataset[0][0].x.shape[1]
    assert node_dim == len(FEATURE_NAMES) == 23, (node_dim, len(FEATURE_NAMES))
    print(f"Dataset: {args.dataset} ({len(base_dataset)} mutations, {node_dim} features)")

    dtype = torch.float64 if args.precision == 'fp64' else torch.float32
    model = load_explain_model(args.weight_file, node_dim, device, dtype)
    weight_md5 = file_md5(args.weight_file)
    print(f"Loaded (strict) {args.weight_file}  md5={weight_md5}  kwargs={GUESSER_MODEL_KWARGS}")

    mapper = ResidueMapper(args.pdb_dirs, cutoff=args.cutoff)

    per_mut = []          # list of per-graph dicts
    preds = {k: [] for k in PRED_KEYS}
    max_endpoint_err = 0.0
    n_done = 0
    for batch_idx, (batch_wt, batch_mt) in enumerate(tqdm(loader, desc="Explaining", mininterval=30)):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch_wt, batch_mt = batch_wt.to(device), batch_mt.to(device)
        per_graph, direct, endpoint_err = explain_batch(
            model, batch_wt, batch_mt, args.n_steps,
            max_nodes_per_pass=args.max_nodes_per_pass, fast=True)
        max_endpoint_err = max(max_endpoint_err, endpoint_err)
        for k in PRED_KEYS:
            preds[k].append(direct[k])

        for i, d in enumerate(per_graph):
            gidx = n_done + i
            item = base_dataset.data_list[gidx]
            g_wt, g_mt = base_dataset[gidx]
            assert d['signed_attr_mu'].shape[0] == g_wt.x.shape[0]
            pdb_name, mutation = item['pdb'], item['mutation']
            info, status, wt_ok = mapper.map_graph(pdb_name, mutation, g_wt)
            mut_idx = torch.where(g_wt.mut_mask)[0].numpy()
            d.update({
                'pdb': pdb_name, 'mutation': mutation,
                'true_ddg': float(g_mt.y.reshape(-1)[0]) if getattr(g_mt, 'y', None) is not None else np.nan,
                'n_nodes': int(g_wt.x.shape[0]),
                'mutated_node_indices': mut_idx,
                'residue_info': info, 'mapping_status': status, 'wt_identity_match': wt_ok,
                'raw_node_features_wt': g_wt.x.numpy().astype(np.float32),
                'raw_node_features_mt': g_mt.x.numpy().astype(np.float32),
                'edge_index': g_wt.edge_index.numpy(), 'edge_attr': g_wt.edge_attr.numpy(),
            })
            per_mut.append(d)
        n_done += len(per_graph)

    N = len(per_mut)
    preds = {k: np.concatenate(v)[:N] for k, v in preds.items()}

    # ---------------- checks ----------------
    def err_arr(fam, k):
        return np.array([d[f'completeness_err_{fam}_{k}'] for d in per_mut])

    def mass_arr(fam, k):
        return np.array([d[f'attr_mass_{fam}_{k}'] for d in per_mut])

    checks = {'max_endpoint_vs_forward_abs_err_mu': max_endpoint_err}
    print(f"\n=== Checks ({N} mutations) ===")
    print(f"IG path end vs model forward, max |d mu|: {max_endpoint_err:.2e}")
    for fam in ('mut', 'ctx'):
        for k in TARGET_KEYS:
            e = np.abs(err_arr(fam, k))
            m = mass_arr(fam, k)
            rel = np.where(m > 1e-9, e / np.maximum(m, 1e-12), 0.0)
            checks[f'completeness_{fam}_{k}_max_abs'] = float(e.max())
            checks[f'completeness_{fam}_{k}_max_rel_mass'] = float(rel.max())
            print(f"  completeness {fam:3s} {k:15s} max|err| {e.max():.2e}  "
                  f"err/sum|attr|: median {np.median(rel):.1e} max {rel.max():.1e}")
    # attr(mu) = attr(phys)+attr(base)+attr(epi) and attr(phys) = closed form hold by
    # construction in fast mode; fast == reference IG is verified by tests/test_explainer.py.
    # The completeness residual of mu above independently checks base + epi + phys together.
    n_ok = sum(d['mapping_status'].startswith('verified') for d in per_mut)
    n_wt = sum(bool(d['wt_identity_match']) for d in per_mut)
    checks.update({'mapping_verified': n_ok, 'wt_identity_match': n_wt})
    print(f"  residue mapping verified: {n_ok}/{N} | annotated WT == structure: {n_wt}/{n_ok}")
    for st in sorted({d['mapping_status'] for d in per_mut if not d['mapping_status'].startswith('verified')})[:10]:
        print(f"    unverified example: {st}")

    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv)
        ref = ref[ref['direction'] == 'Forward'].reset_index(drop=True).iloc[:N]
        same_ids = (list(ref['pdb']) == [d['pdb'] for d in per_mut]) and \
                   (list(ref['mutation'].astype(str)) == [str(d['mutation']) for d in per_mut])
        ref_err = {c: float(np.abs(ref[r].values - preds[c]).max())
                   for c, r in [('mu', 'pred_ddg'), ('v', 'v'), ('beta', 'beta'), ('alpha', 'alpha'),
                                ('gamma_phys', 'pred_phys'), ('gamma_base', 'pred_base'),
                                ('gamma_epi', 'pred_epi')]}
        checks['reference_same_order'] = bool(same_ids)
        checks.update({f'reference_max_abs_{c}': v for c, v in ref_err.items()})
        print(f"  reference CSV same pdb/mutation order: {same_ids}")
        print("  reference CSV max |diff|: " + ", ".join(f"{c} {v:.1e}" for c, v in ref_err.items()))

    # Hard failures: a wrong model, wrong data or broken IG path must not produce outputs.
    fatal = []
    if max_endpoint_err > (1e-8 if args.precision == 'fp64' else 1e-4):
        fatal.append(f"IG path end differs from model forward by {max_endpoint_err:.2e}")
    if args.reference_csv:
        if not checks['reference_same_order']:
            fatal.append("pdb/mutation order differs from reference CSV")
        if checks['reference_max_abs_mu'] > 1e-4:
            fatal.append(f"mu differs from reference CSV by {checks['reference_max_abs_mu']:.2e}")
    if fatal:
        raise SystemExit("FATAL: " + "; ".join(fatal))

    # ---------------- save ----------------
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    list_keys = [k for k in per_mut[0].keys()]
    raw = {
        'format_version': 'explain_s_std_a_v2',
        'feature_names': FEATURE_NAMES,
        'targets': TARGETS,
        'attribution_families': {
            'signed_attr_<t>': 'IG on x_diff = x_mt - x_wt; baseline x_diff = 0 (no mutation); '
                               'x_wt fixed. sum = f(wt,mt) - f(wt,wt)',
            'signed_attr_<t>_wt': 'IG on x_wt; baseline = per-graph mean of x_wt over nodes; '
                                  'x_diff fixed. sum = f(x_wt,x_diff) - f(mean_wt,x_diff)',
            'exact_phys_attr': 'closed form w_phys * x_diff[:, 12:23]',
        },
        'ig_method': f'Gauss-Legendre, {args.n_steps} nodes',
        'seed': args.seed, 'weight_file': os.path.abspath(args.weight_file), 'weight_md5': weight_md5,
        'model_kwargs': GUESSER_MODEL_KWARGS, 'dataset_path': os.path.abspath(args.dataset),
        'pdb_dirs': args.pdb_dirs, 'cutoff': args.cutoff, 'n_steps': args.n_steps,
        'batch_size': args.batch_size, 'n_mutations': N, 'precision': args.precision,
        'code_md5': {f: file_md5(os.path.join(here, f)) for f in CORE_FILES},
        'torch_version': torch.__version__, 'checks': checks,
        **{f'pred_{k}': preds[k] for k in PRED_KEYS},
        **{k: [d[k] for d in per_mut] for k in list_keys},
    }
    raw['true_ddg'] = np.array(raw['true_ddg'])
    raw['n_nodes'] = np.array(raw['n_nodes'])
    torch.save(raw, f"{args.out_prefix}_raw_data.pt")

    rows = []
    for i, d in enumerate(per_mut):
        r = {'pdb': d['pdb'], 'mutation': d['mutation'], 'true_ddg': d['true_ddg'],
             'n_nodes': d['n_nodes'], 'mapping_status': d['mapping_status'],
             'wt_identity_match': d['wt_identity_match'],
             'mutated_nodes': ", ".join(d['residue_info'][j]['label'] for j in d['mutated_node_indices'])
             if d['residue_info'] else ", ".join(f"Node_{j}" for j in d['mutated_node_indices'])}
        r.update({f'pred_{k}': float(preds[k][i]) for k in PRED_KEYS})
        for fam in ('mut', 'ctx'):
            for k in TARGET_KEYS:
                r[f'f0_{fam}_{k}'] = d[f'f0_{fam}_{k}']
                r[f'completeness_err_{fam}_{k}'] = d[f'completeness_err_{fam}_{k}']
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(f"\nSaved {args.out_prefix}_raw_data.pt and _summary.csv  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
