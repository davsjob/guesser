#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combine per-seed explanations (guesser.explain.run) into the ensemble.

    python -m guesser.explain.merge --seed_files "out/ssym_S*_raw_data.pt" --out_prefix out/ssym --save_h5

Consistency checks (hard errors): same format, same mutations in the same order, same
graphs, same IG settings and explainer code, distinct weight files.

Ensemble quantities
  pred_*                mean over members of the raw model outputs
  aleatoric_std         sqrt(mean_m beta/(alpha-1))
  intra_epistemic_std   sqrt(mean_m beta/(v(alpha-1)))
  inter_epistemic_std   sqrt(var_m(mu)), ddof=1 (as guesser.inference.ensemble)
  epistemic_std         sqrt(inter^2 + intra^2);  total_std = sqrt(aleatoric^2 + epistemic^2)
  signed_attr_*         mean over members. IG is linear in f, so this is exactly the IG of
                        the ensemble-mean mu / gamma_*; for the log-std targets it is the IG
                        of the members' mean log-std.
Completeness of the ensemble attributions holds with the ensemble means:
  sum(signed_attr_mu) = mean_m mu_m - mean_m mu_m(null mutation)

Summary CSV (one row per mutation; all contributions in the target's units):
  mu_null                      ensemble prediction for the null mutation (x_mt = x_wt)
  mu_contrib_<feature>         sum over nodes of signed_attr_mu[:, feature]; these 23 numbers
                               sum to pred_mu - mu_null (up to the quadrature residual)
  mu_contrib_mutsite / _environment   the same split by mutated nodes vs all other nodes
  ctx_mu_contrib_<feature>     WT-environment attribution vs the protein's mean environment
  aleatoric_contrib_<f>, epistemic_contrib_<f>   same for the log-std targets
  top residues by |sum over features|, focus ratios, member disagreement, residuals.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

try:
    import h5py
except ImportError:
    h5py = None

TARGET_KEYS = ['mu', 'phys', 'base', 'epi', 'aleatoric', 'epistemic', 'evidence', 'total_evidence']
ATTR_KEYS = [f'signed_attr_{k}' for k in TARGET_KEYS] + [f'signed_attr_{k}_wt' for k in TARGET_KEYS] \
            + ['exact_phys_attr']
PRED_KEYS = ['mu', 'gamma_phys', 'gamma_base', 'gamma_epi', 'v', 'alpha', 'beta']
SAME_META = ['feature_names', 'n_steps', 'precision', 'cutoff', 'model_kwargs', 'dataset_path',
             'n_mutations', 'code_md5']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Merge per-seed explanations into the ensemble")
    p.add_argument("--seed_files", nargs='+', required=True, help="Per-seed *_raw_data.pt (globs ok)")
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--expected_seeds", type=int, default=8)
    p.add_argument("--save_h5", action="store_true",
                   help="(kept for compatibility; the self-contained .h5 is always written)")
    p.add_argument("--keep_full", action="store_true",
                   help="(kept for compatibility; per-seed matrices are stored unless --no_per_seed)")
    p.add_argument("--no_per_seed", action="store_true",
                   help="Do not store per-seed attribution matrices (smaller file, not self-contained)")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args(argv)


def load_seeds(patterns, expected):
    files = sorted({f for pat in patterns for f in (glob.glob(pat) if any(c in pat for c in '*?[') else [pat])
                    if os.path.exists(f)})
    if not files:
        raise FileNotFoundError(f"No seed files matched {patterns}")
    seeds = [torch.load(f, weights_only=False) for f in files]
    s0 = seeds[0]
    for f, s in zip(files, seeds):
        if s.get('format_version') != 'explain_s_std_a_v2':
            raise ValueError(f"{f}: not a v2 explanation file")
        for k in SAME_META:
            if s[k] != s0[k]:
                raise ValueError(f"{f}: '{k}' differs from {files[0]}")
        if s['pdb'] != s0['pdb'] or [str(m) for m in s['mutation']] != [str(m) for m in s0['mutation']]:
            raise ValueError(f"{f}: mutation list/order differs")
        if not np.array_equal(s['n_nodes'], s0['n_nodes']):
            raise ValueError(f"{f}: graph sizes differ")
        if s['mapping_status'] != s0['mapping_status']:
            raise ValueError(f"{f}: residue mapping differs")
    md5s = [s['weight_md5'] for s in seeds]
    if len(set(md5s)) != len(md5s):
        raise ValueError(f"Duplicate weight files among seeds: {md5s}")
    if len(seeds) != expected:
        raise ValueError(f"Expected {expected} seeds, found {len(seeds)}: {files}")
    return files, seeds


def main(argv=None):
    args = parse_args(argv)
    files, seeds = load_seeds(args.seed_files, args.expected_seeds)
    s0 = seeds[0]
    M, N = len(seeds), s0['n_mutations']
    feat = s0['feature_names']
    F = len(feat)
    print(f"Merging {M} seeds ({[s['seed'] for s in seeds]}), {N} mutations, {F} features")

    # ---- predictions and EDL uncertainty ----
    pm = {k: np.stack([np.asarray(s[f'pred_{k}'], dtype=np.float64) for s in seeds]) for k in PRED_KEYS}
    ens = {k: pm[k].mean(0) for k in PRED_KEYS}
    am1 = pm['alpha'] - 1.0
    ale_var = (pm['beta'] / am1).mean(0)
    intra_var = (pm['beta'] / (pm['v'] * am1)).mean(0)
    inter_var = pm['mu'].var(0, ddof=1) if M > 1 else np.zeros(N)
    unc = {
        'aleatoric_std': np.sqrt(ale_var),
        'intra_epistemic_std': np.sqrt(intra_var),
        'inter_epistemic_std': np.sqrt(inter_var),
        'epistemic_std': np.sqrt(inter_var + intra_var),
        'total_std': np.sqrt(ale_var + inter_var + intra_var),
    }

    # ---- attributions ----
    mean_attr = {k: [] for k in ATTR_KEYS}
    keep_full = not args.no_per_seed
    full_attr = {k: [] for k in ATTR_KEYS} if keep_full else None
    node_dis, feat_dis = [], []
    for i in range(N):
        for k in ATTR_KEYS:
            st = np.stack([s[k][i] for s in seeds]).astype(np.float64)  # [M, n, F]
            mean_attr[k].append(st.mean(0).astype(np.float32))
            if keep_full:
                full_attr[k].append(np.stack([s[k][i] for s in seeds]).astype(np.float32))
            if k == 'signed_attr_mu':
                node_dis.append(st.sum(2).std(0).astype(np.float32))   # per-node signed contribution
                feat_dis.append(st.sum(1).std(0).astype(np.float32))   # per-feature signed contribution

    # per-mutation scalars were stored inside the per-graph dicts as lists; collect them
    scal, scal_seed = {}, {}
    for fam in ('mut', 'ctx'):
        for k in TARGET_KEYS:
            for pre in ('f', 'f0', 'completeness_err', 'attr_mass'):
                name = f'{pre}_{fam}_{k}'
                scal_seed[name] = np.stack([np.asarray(s[name], dtype=np.float64) for s in seeds])
                scal[name] = scal_seed[name].mean(0)

    # ---- summary ----
    res_info = s0['residue_info']
    rows = []
    for i in range(N):
        A = mean_attr['signed_attr_mu'][i].astype(np.float64)
        Actx = mean_attr['signed_attr_mu_wt'][i].astype(np.float64)
        Aal = mean_attr['signed_attr_aleatoric'][i].astype(np.float64)
        Aep = mean_attr['signed_attr_epistemic'][i].astype(np.float64)
        mut_idx = np.asarray(s0['mutated_node_indices'][i])
        info = res_info[i]
        lab = (lambda j: info[j]['label']) if info else (lambda j: f"Node_{j}")
        node_c = A.sum(1)
        mask = np.zeros(A.shape[0], bool)
        mask[mut_idx] = True

        def focus(M_):
            imp = np.abs(M_).sum(1)
            return float(imp[mask].mean() / imp.mean()) if imp.mean() > 0 else np.nan

        order = np.argsort(-np.abs(node_c))[:3]
        r = {
            'pdb': s0['pdb'][i], 'mutation': s0['mutation'][i], 'true_ddg': float(s0['true_ddg'][i]),
            'n_nodes': int(s0['n_nodes'][i]), 'n_mutated': int(len(mut_idx)),
            'mutated_nodes': ", ".join(lab(j) for j in mut_idx),
            'mapping_status': s0['mapping_status'][i], 'wt_identity_match': s0['wt_identity_match'][i],
            **{f'pred_{k}': float(ens[k][i]) for k in PRED_KEYS},
            **{k: float(v[i]) for k, v in unc.items()},
            'mu_null': float(scal['f0_mut_mu'][i]),
            'mu_contrib_total': float(A.sum()),
            'mu_contrib_mutsite': float(A[mask].sum()),
            'mu_contrib_environment': float(A[~mask].sum()),
            'completeness_err_mu': float(scal['completeness_err_mut_mu'][i]),
            'completeness_err_aleatoric': float(scal['completeness_err_mut_aleatoric'][i]),
            'completeness_err_epistemic': float(scal['completeness_err_mut_epistemic'][i]),
            'focus_ratio_mu': focus(A), 'focus_ratio_aleatoric': focus(Aal),
            'focus_ratio_epistemic': focus(Aep),
            'member_std_mu': float(pm['mu'][:, i].std(ddof=1)) if M > 1 else 0.0,
            'mean_node_disagreement_mu': float(node_dis[i].mean()),
            'top_disagree_feature_mu': feat[int(np.argmax(feat_dis[i]))],
        }
        for rank, j in enumerate(order, 1):
            r[f'top{rank}_residue_mu'] = lab(j)
            r[f'top{rank}_residue_mu_contrib'] = float(node_c[j])
        for f_i, fname in enumerate(feat):
            r[f'mu_contrib_{fname}'] = float(A[:, f_i].sum())
        for f_i, fname in enumerate(feat):
            r[f'ctx_mu_contrib_{fname}'] = float(Actx[:, f_i].sum())
        for f_i, fname in enumerate(feat):
            r[f'aleatoric_contrib_{fname}'] = float(Aal[:, f_i].sum())
        for f_i, fname in enumerate(feat):
            r[f'epistemic_contrib_{fname}'] = float(Aep[:, f_i].sum())
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    df.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    # ---- self-contained HDF5 ----
    import datetime
    import hashlib
    from .h5 import write_explanation_h5

    def md5(path):
        return hashlib.md5(open(path, 'rb').read()).hexdigest()

    here = os.path.dirname(os.path.abspath(__file__))
    n_nodes = np.asarray(s0['n_nodes'], dtype=np.int64)
    n_edges = np.array([np.asarray(e).shape[1] for e in s0['edge_index']], dtype=np.int64)
    node_offset = np.concatenate([[0], np.cumsum(n_nodes)])
    edge_offset = np.concatenate([[0], np.cumsum(n_edges)])
    V = int(node_offset[-1])

    is_mut = np.zeros(V, dtype=bool)
    local_idx = np.concatenate([np.arange(n) for n in n_nodes])
    res_chain = np.full(V, '', dtype=object); res_icode = np.full(V, '', dtype=object)
    res_name = np.full(V, '', dtype=object); res_label = np.full(V, '', dtype=object)
    res_id = np.full(V, -1, dtype=np.int64)
    for i in range(N):
        o = node_offset[i]
        is_mut[o + np.asarray(s0['mutated_node_indices'][i], dtype=np.int64)] = True
        info = res_info[i]
        if info:
            for j, r_ in enumerate(info):
                res_chain[o + j], res_id[o + j], res_icode[o + j] = r_['chain'], r_['res_id'], r_['icode']
                res_name[o + j], res_label[o + j] = r_['res_name_3'], r_['label']
        else:
            res_label[o:o + n_nodes[i]] = [f"Node_{j}" for j in range(n_nodes[i])]

    cat = lambda lst: np.concatenate([np.asarray(x) for x in lst], axis=0)
    node = {k: cat(mean_attr[k]).astype(np.float32) for k in ATTR_KEYS}
    node.update({
        'phys_attr_per_node': cat([np.asarray(x) for x in mean_attr['exact_phys_attr']]).sum(1).astype(np.float32),
        'raw_node_features_wt': cat(s0['raw_node_features_wt']).astype(np.float32),
        'raw_node_features_mt': cat(s0['raw_node_features_mt']).astype(np.float32),
        'is_mutated': is_mut, 'local_node_index': local_idx.astype(np.int32),
        'res_chain': res_chain, 'res_id': res_id, 'res_icode': res_icode,
        'res_name_3': res_name, 'res_label': res_label,
        'disagreement_node_mu': cat(node_dis).astype(np.float32),
    })
    node_per_seed = {}
    if keep_full:
        for k in ATTR_KEYS:
            node_per_seed[k] = np.concatenate(full_attr[k], axis=1)  # [M, V, F]
            full_attr[k] = None
    edge = {'edge_index': np.concatenate([np.asarray(e).T for e in s0['edge_index']]).astype(np.int32),
            'edge_attr': cat(s0['edge_attr']).astype(np.float32)}
    wt_match = np.array([-1 if w is None else int(bool(w)) for w in s0['wt_identity_match']], dtype=np.int8)
    per_mutation = {
        'pdb': np.array(s0['pdb'], dtype=object), 'mutation': np.array([str(x) for x in s0['mutation']], dtype=object),
        'true_ddg': np.asarray(s0['true_ddg'], dtype=np.float64), 'n_nodes': n_nodes, 'n_edges': n_edges,
        'n_mutated': np.array([len(x) for x in s0['mutated_node_indices']], dtype=np.int64),
        'node_offset': node_offset, 'edge_offset': edge_offset,
        'mapping_status': np.array(s0['mapping_status'], dtype=object), 'wt_identity_match': wt_match,
    }
    prediction = {**{f'pred_{k}': ens[k] for k in PRED_KEYS},
                  **{f'per_member_{k}': pm[k] for k in PRED_KEYS}, **unc}
    meta = {k: s0[k] for k in SAME_META + ['targets', 'attribution_families', 'ig_method',
                                          'pdb_dirs', 'batch_size', 'torch_version']}
    meta.update({
        'n_seeds': M, 'seeds': [int(s['seed']) for s in seeds],
        'weight_files': [s['weight_file'] for s in seeds], 'weight_md5': [s['weight_md5'] for s in seeds],
        'seed_files': [os.path.abspath(f) for f in files],
        'per_seed_checks': {str(s['seed']): s['checks'] for s in seeds},
        'per_seed_arrays_stored': keep_full,
        'merge_code_md5': {f: md5(os.path.join(here, f)) for f in
                           ('merge.py', 'h5.py')},
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    })
    h5_path = f"{args.out_prefix}_raw_data.h5"
    write_explanation_h5(
        h5_path, meta=meta, per_mutation=per_mutation, node=node, node_per_seed=node_per_seed,
        edge=edge, mutation_feature={'disagreement_feat_mu': np.stack(feat_dis).astype(np.float32)},
        prediction=prediction, ig=scal, ig_per_seed=scal_seed,
        summary_csv_text=df.to_csv(index=False))
    print(f"  -> {h5_path}  ({os.path.getsize(h5_path) / 1e6:.1f} MB)")
    print(f"  -> {args.out_prefix}_summary.csv")

    # ---- report ----
    ok = df['mapping_status'].str.startswith('verified').sum()
    rel = (df['completeness_err_mu'].abs() / df[[f'mu_contrib_{f}' for f in feat]].abs().sum(1).clip(lower=1e-9))
    print(f"\n  residue mapping verified {ok}/{N}; annotated WT matches structure "
          f"{int(df['wt_identity_match'].fillna(False).astype(bool).sum())}/{ok}")
    print(f"  ensemble mu completeness |err|: max {df['completeness_err_mu'].abs().max():.2e}, "
          f"median rel to sum|feature contrib| {rel.median():.1e}")
    print(f"  mean |pred_mu - mu_null - sum contrib| = {np.abs(df['pred_mu'] - df['mu_null'] - df['mu_contrib_total']).mean():.2e}")
    gm = pd.Series({f: df[f'mu_contrib_{f}'].abs().mean() for f in feat}).sort_values(ascending=False)
    print("  top features by mean |contribution to mu| (mutation attribution):")
    for fname, v in gm.head(6).items():
        print(f"     {fname:25s} {v:.4f}  (mean signed {df[f'mu_contrib_{fname}'].mean():+.4f})")

    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        panels = [('mu_contrib_', 'Prediction μ: mutation attribution'),
                  ('ctx_mu_contrib_', 'Prediction μ: WT-environment attribution'),
                  ('aleatoric_contrib_', 'log aleatoric std: mutation attribution'),
                  ('epistemic_contrib_', 'log epistemic std: mutation attribution')]
        fig, axes = plt.subplots(1, 4, figsize=(26, 8))
        for ax, (pre, title) in zip(axes, panels):
            cols = [f'{pre}{f}' for f in feat]
            mabs = df[cols].abs().mean().values
            msig = df[cols].mean().values
            o = np.argsort(mabs)
            ax.barh(range(F), mabs[o], color='#9ab', label='mean |contribution|')
            ax.barh(range(F), msig[o], color='#258', height=0.4, label='mean signed contribution')
            ax.set_yticks(range(F)); ax.set_yticklabels([feat[j] for j in o])
            ax.axvline(0, color='k', lw=0.5); ax.set_title(title); ax.legend(loc='lower right', fontsize=8)
        fig.suptitle(f"{os.path.basename(args.out_prefix)}: per-feature IG contributions "
                     f"(ensemble of {M}, {N} mutations)")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_feature_contributions.png", dpi=200)
        plt.close(fig)
        print(f"  -> {args.out_prefix}_feature_contributions.png")


if __name__ == "__main__":
    main()
