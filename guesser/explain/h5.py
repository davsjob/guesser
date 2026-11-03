# -*- coding: utf-8 -*-
"""
Self-contained HDF5 format for the GUESSER ensemble explanations. The format identifiers
keep the internal name "s_std_a" so that files written for the paper stay readable.

Writer: used by guesser.explain.merge.
Reader: this file has no package dependencies, so it can be copied next to a downloaded
.h5 on its own; needs numpy + h5py (pandas only for summary()).

    from guesser.explain.h5 import ExplanationFile
    ef = ExplanationFile("s_std_a_s669_raw_data.h5")
    print(ef.readme)                 # full description of every dataset
    df = ef.summary()                # the per-mutation summary table (pandas)
    m = ef.mutation(0)               # everything for mutation 0 as a dict
    A = ef.node_array("signed_attr_mu", 0)              # [n_nodes, 23] ensemble mean
    A8 = ef.node_array("signed_attr_mu", 0, per_seed=True)  # [8, n_nodes, 23]

Layout (N mutations, M seeds, V = total nodes, E = total edges, F = 23 features):
  /meta                     attrs: provenance, settings, JSON blobs, README
  /mutation/*               [N] per-mutation vectors; node_offset/edge_offset are [N+1]
  /prediction/*             [N] ensemble, [M, N] per seed
  /ig/<quantity>            [N] ensemble mean and /ig_per_seed/<quantity> [M, N]
  /node/*                   [V, ...] concatenated over mutations
  /node_per_seed/*          [M, V, F] per-seed attribution matrices
  /edge/*                   [E, ...] concatenated, edge_index local to its mutation
  /mutation_feature/*       [N, F]
  /summary_csv              gzip-compressed UTF-8 bytes of the summary CSV
Rows of mutation i in /node: node_offset[i]:node_offset[i+1]; in /edge likewise.
"""
import io
import json

import numpy as np

FORMAT_VERSION = 'explain_s_std_a_h5_v3'

README = """GUESSER ensemble explanations (Integrated Gradients), format explain_s_std_a_h5_v3.

MODEL: GUESSER (guesser/model.py; trained as SiameseEvidentialResGAT_Ultimate), 8 seeds (see /meta seeds,
weight_files, weight_md5), outputs mu = gamma_phys + gamma_base + gamma_epi and NIG (v, alpha, beta).
Predictions are the raw model(batch_wt, batch_mt) outputs (identical to the benchmark CSVs).

FEATURES (F = 23, /meta feature_names): 11 intrinsic amino-acid properties, SASA, 11 MadraX energy
terms (order of madrax ForceField output). Graph nodes are the residues within 9 A of the mutated
residue(s) (guesser/graphs.py), NOT the whole protein.

ATTRIBUTION TARGETS t (/meta targets):
  mu, phys (gamma_phys), base (gamma_base), epi (gamma_epi, 0 for single mutants),
  aleatoric = 0.5*log(beta/(alpha-1)), epistemic = 0.5*log(beta/(v(alpha-1))),
  evidence = v, total_evidence = 2v + alpha.

ATTRIBUTION FAMILIES (per node x feature, units of the target):
  signed_attr_<t>     "mutation" IG: input x_diff = x_mt - x_wt, baseline x_diff = 0 (null mutation),
                      WT features fixed. Sum over all nodes and features = f(wt,mt) - f(wt,wt).
  signed_attr_<t>_wt  "environment" IG: input x_wt, baseline = this graph's per-feature node mean,
                      x_diff fixed. Sum = f(x_wt, x_diff) - f(mean_wt, x_diff).
  exact_phys_attr     closed form of gamma_phys = w . sum_nodes x_diff[:, 12:23] (equals signed_attr_phys).
  Method: Gauss-Legendre quadrature (/meta ig_method), computed in float64, stored float32.
  /node/<key> is the mean over seeds (exactly the IG of the ensemble-mean mu/gamma_*; for the log-std
  targets the IG of the seeds' mean log-std). /node_per_seed/<key> holds each seed's matrix.

IG SCALARS (/ig: ensemble mean, /ig_per_seed: [M, N]):
  f_<fam>_<t>              target value at the input (fam = mut | ctx)
  f0_<fam>_<t>             target value at the baseline; f0_mut_mu = "mu_null", the prediction for
                           the null mutation (not 0: antisymmetry is only approximate)
  completeness_err_<fam>_<t>  sum(attr) - (f - f0): quadrature residual
  attr_mass_<fam>_<t>      sum |attr|

PREDICTIONS (/prediction): pred_<k> ensemble mean, per_member_<k> [M, N] for k in mu, gamma_phys,
  gamma_base, gamma_epi, v, alpha, beta. aleatoric_std = sqrt(mean beta/(alpha-1)),
  intra_epistemic_std = sqrt(mean beta/(v(alpha-1))), inter_epistemic_std = seed std of mu (ddof=1),
  epistemic_std = sqrt(inter^2 + intra^2), total_std = sqrt(aleatoric^2 + epistemic^2).

RESIDUES (/node res_*): node -> PDB residue, accepted only if the rebuilt graph reproduces the cached
  graph exactly (node count, mutated nodes, per-node amino-acid features, all edges and distances).
  /mutation mapping_status says 'verified:<pdb path>' or why not; wt_identity_match: 1 annotated WT
  equals the structure, 0 differs, -1 unmapped.

OTHER: /node raw_node_features_wt|mt [V, F], is_mutated [V], disagreement_node_mu [V] (seed std of
  the per-node signed mu contribution); /mutation_feature disagreement_feat_mu [N, F];
  /edge edge_index [E, 2] (local node indices), edge_attr [E, 2] (cos angle, CA distance A);
  /meta per_seed_checks: each seed run's verification results (JSON).
"""


def _h5():
    import h5py
    return h5py


def _str_dt():
    return _h5().string_dtype()


def _put(grp, name, arr, chunk_rows=None):
    arr = np.asarray(arr)
    if arr.dtype.kind in ('U', 'O'):
        grp.create_dataset(name, data=np.array([str(x) for x in arr.ravel()], dtype=object).reshape(arr.shape),
                           dtype=_str_dt())
        return
    kw = {}
    if arr.size > 1024:
        kw = dict(compression='gzip', compression_opts=4, shuffle=True)
        if chunk_rows is not None and arr.ndim >= 1:
            ch = list(arr.shape)
            ax = 1 if (arr.ndim == 3) else 0  # [M, V, F]: chunk along V
            ch[ax] = max(1, min(arr.shape[ax], chunk_rows))
            kw['chunks'] = tuple(ch)
    grp.create_dataset(name, data=arr, **kw)


def write_explanation_h5(path, *, meta, per_mutation, node, node_per_seed, edge,
                         mutation_feature, prediction, ig, ig_per_seed, summary_csv_text):
    """All arrays already concatenated/flattened by the caller (merge_explain_seeds.py)."""
    import gzip
    h5py = _h5()
    with h5py.File(path, 'w') as f:
        m = f.create_group('meta')
        m.attrs['format_version'] = FORMAT_VERSION
        m.attrs['README'] = README
        for k, v in meta.items():  # non-scalars as JSON; read back with ExplanationFile.meta(key)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, str):
                v = str(v)  # plain str (e.g. torch.__version__ is a str subclass h5py rejects)
            if isinstance(v, (dict, list, tuple)):
                m.attrs[k] = json.dumps(v, default=str)
            else:
                m.attrs[k] = v if v is not None else 'None'
        for gname, content in (('mutation', per_mutation), ('prediction', prediction), ('ig', ig),
                               ('ig_per_seed', ig_per_seed), ('mutation_feature', mutation_feature)):
            g = f.create_group(gname)
            for k, v in content.items():
                _put(g, k, v)
        for gname, content, rows in (('node', node, 65536), ('node_per_seed', node_per_seed, 4096),
                                     ('edge', edge, 65536)):
            g = f.create_group(gname)
            for k, v in content.items():
                _put(g, k, v, chunk_rows=rows)
        raw = gzip.compress(summary_csv_text.encode('utf-8'))
        f.create_dataset('summary_csv', data=np.frombuffer(raw, dtype=np.uint8))


class ExplanationFile:
    """Read-only access to an explain_s_std_a_h5_v3 file (numpy + h5py only)."""

    def __init__(self, path):
        h5py = _h5()
        self.f = h5py.File(path, 'r')
        v = self.f['meta'].attrs['format_version']
        if v != FORMAT_VERSION:
            raise ValueError(f"unsupported format {v}")
        self.readme = self.f['meta'].attrs['README']
        self.feature_names = self.meta('feature_names')
        self.seeds = self.meta('seeds')
        self.node_offset = self.f['mutation/node_offset'][:]
        self.edge_offset = self.f['mutation/edge_offset'][:]
        self.n_mutations = len(self.node_offset) - 1

    def meta(self, key):
        v = self.f['meta'].attrs[key]
        if isinstance(v, str) and v[:1] in '[{':
            try:
                return json.loads(v)
            except ValueError:
                pass
        return v

    def keys(self, group):
        return list(self.f[group].keys())

    def _rows(self, i):
        return slice(int(self.node_offset[i]), int(self.node_offset[i + 1]))

    def node_array(self, key, i, per_seed=False):
        if per_seed:
            return self.f['node_per_seed'][key][:, self._rows(i)]
        return self.f['node'][key][self._rows(i)]

    def mutation(self, i):
        """Everything stored for mutation i."""
        out = {}
        for k in self.f['mutation']:
            if k not in ('node_offset', 'edge_offset'):
                out[k] = _dec(self.f['mutation'][k][i])
        for g in ('prediction', 'ig'):
            for k in self.f[g]:
                d = self.f[g][k]
                out[k] = d[i] if d.ndim == 1 else d[:, i]
        for k in self.f['ig_per_seed']:
            out[f'per_seed_{k}'] = self.f['ig_per_seed'][k][:, i]
        for k in self.f['mutation_feature']:
            out[k] = self.f['mutation_feature'][k][i]
        r = self._rows(i)
        for k in self.f['node']:
            a = self.f['node'][k][r]
            out[k] = np.array([_dec(x) for x in a]) if a.dtype.kind == 'O' else a
        for k in self.f['node_per_seed']:
            out[f'{k}_per_seed'] = self.f['node_per_seed'][k][:, r]
        e = slice(int(self.edge_offset[i]), int(self.edge_offset[i + 1]))
        for k in self.f['edge']:
            out[k] = self.f['edge'][k][e]
        return out

    def summary(self):
        import gzip
        import pandas as pd
        txt = gzip.decompress(self.f['summary_csv'][:].tobytes()).decode('utf-8')
        return pd.read_csv(io.StringIO(txt))

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _dec(x):
    return x.decode('utf-8') if isinstance(x, bytes) else x
