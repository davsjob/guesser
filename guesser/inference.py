"""
Inference with one checkpoint or the 8-member ensemble.

Per member, every mutation is evaluated forward (WT -> MT) and reverse (MT -> WT), as in
the paper's benchmark files. Ensemble quantities (Forward rows), as in the manuscript (Eqs. 1-2):

    pred_ddg             mu = mean_m gamma_m
    intra_epistemic_std  sigma_e = sqrt(mean_m beta/(v(alpha-1)))   (epistemic uncertainty)
    calibrated_std       s* sigma_e, with s* = 1.71 fitted on the MegaScale validation set
    aleatoric_std        sigma_a = sqrt(mean_m beta/(alpha-1))
    inv_phi              1/Phi = mean_m 1/(2v + alpha)               (inverse total evidence; ranking only)
    inter_epistemic_std  std_m(mu), ddof=1                           (spread between members)
    epistemic_std        sqrt(intra^2 + inter^2)
    total_std            sqrt(aleatoric^2 + epistemic^2)
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from .datasets import CoordinateNoiseDataset, GraphPairDataset, ReverseAugmentedDataset, collate_pairs
from .features import apply_feature_set
from .model import load_model

# Validation scale factor for sigma_e (manuscript Eq. 2)
S_STAR = 1.708

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


def released_weights(weights_dir=DEFAULT_WEIGHTS_DIR):
    paths = sorted(glob.glob(os.path.join(weights_dir, "guesser_S*.pt")))
    if not paths:
        raise FileNotFoundError(f"No guesser_S*.pt checkpoints in {weights_dir}")
    return paths


def _loader(graphs, batch_size):
    ds = CoordinateNoiseDataset(ReverseAugmentedDataset(GraphPairDataset(graphs), augment=True), use_noise=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs, num_workers=0)


@torch.no_grad()
def predict_member(model, graphs, device="cpu", batch_size=64):
    """
    One model on a list of graph pairs. Returns a DataFrame with Forward then Reverse rows and
    the columns of the paper's per-seed prediction files.
    """
    model.eval()
    cols = {k: [] for k in ('mu', 'v', 'alpha', 'beta', 'pred_epi', 'pred_phys', 'pred_base', 'num_muts', 'y')}
    for batch_wt, batch_mt in _loader(graphs, batch_size):
        batch_wt, batch_mt = batch_wt.to(device), batch_mt.to(device)
        mu, v, alpha, beta, g_epi, g_phys, g_base = model(batch_wt, batch_mt)[:7]
        n_muts = global_add_pool(batch_wt.mut_mask.float(), batch_wt.batch)
        for k, t in zip(cols, (mu, v, alpha, beta, g_epi, g_phys, g_base, n_muts, batch_mt.y)):
            cols[k].append(t.reshape(-1).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in cols.items()}

    meta = pd.DataFrame([{
        'pdb': item.get('pdb', 'unknown'), 'mutation': item.get('mutation', 'unknown'),
        'ddg': item['graph_pair'][1].y.item() if getattr(item['graph_pair'][1], 'y', None) is not None else np.nan,
    } for item in graphs])

    parts = []
    for direction, sl, sign in (('Forward', slice(0, None, 2), 1.0), ('Reverse', slice(1, None, 2), -1.0)):
        df = meta.copy()
        df['direction'] = direction
        df['pred_ddg'] = res['mu'][sl]
        df['true_ddg'] = sign * res['y'][0::2]
        for c in ('v', 'alpha', 'beta', 'pred_epi', 'pred_phys', 'pred_base', 'num_muts'):
            df[c] = res[c][sl]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df['uncert_epistemic'] = np.sqrt(np.clip(df['beta'] / (df['v'] * (df['alpha'] - 1.0 + 1e-8)), 0, None))
    df['uncert_aleatoric'] = np.sqrt(np.clip(df['beta'] / (df['alpha'] - 1.0 + 1e-8), 0, None))
    df['uncert_total'] = np.sqrt(df['uncert_epistemic'] ** 2 + df['uncert_aleatoric'] ** 2)
    df['error'] = np.abs(df['pred_ddg'] - df['true_ddg'])
    return df


def ensemble(member_dfs, direction="Forward"):
    """Combine per-member predictions (same graphs, same order) into ensemble mean and uncertainties."""
    dfs = [d[d['direction'] == direction].reset_index(drop=True) for d in member_dfs]
    out = dfs[0][['pdb', 'mutation', 'true_ddg', 'num_muts']].copy()
    for d in dfs[1:]:
        if not (d['mutation'].values == out['mutation'].values).all():
            raise ValueError("member predictions are not aligned")
    mu = np.stack([d['pred_ddg'].values for d in dfs])
    am1 = np.stack([d['alpha'].values for d in dfs]) - 1.0
    beta = np.stack([d['beta'].values for d in dfs])
    v = np.stack([d['v'].values for d in dfs])
    ale_var = (beta / am1).mean(0)
    intra_var = (beta / (v * am1)).mean(0)
    inter_var = mu.var(0, ddof=1) if len(dfs) > 1 else np.zeros(mu.shape[1])
    out['pred_ddg'] = mu.mean(0)
    for name in ('pred_phys', 'pred_base', 'pred_epi'):
        out[name] = np.stack([d[name].values for d in dfs]).mean(0)
    out['intra_epistemic_std'] = np.sqrt(intra_var)
    out['calibrated_std'] = S_STAR * np.sqrt(intra_var)
    out['aleatoric_std'] = np.sqrt(ale_var)
    out['inv_phi'] = (1.0 / (2.0 * v + am1 + 1.0)).mean(0)
    out['inter_epistemic_std'] = np.sqrt(inter_var)
    out['epistemic_std'] = np.sqrt(inter_var + intra_var)
    out['total_std'] = np.sqrt(ale_var + inter_var + intra_var)
    out['n_members'] = len(dfs)
    return out


def predict(graphs, weight_paths=None, device=None, batch_size=64, feature_set="all", return_members=False):
    """Ensemble prediction for a list of graph pairs. Returns the Forward ensemble DataFrame
    (and the per-member DataFrames if return_members)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weight_paths = weight_paths or released_weights()
    apply_feature_set(graphs, feature_set)
    node_dim = graphs[0]['graph_pair'][0].x.shape[1]
    members = [predict_member(load_model(w, node_features=node_dim, device=device), graphs, device, batch_size)
               for w in weight_paths]
    ens = ensemble(members)
    return (ens, members) if return_members else ens


def metrics(df, pred="pred_ddg", true="true_ddg", sigma=None):
    """Pearson, Spearman, RMSE, MAE; with sigma also Spearman(|error|, sigma)."""
    from scipy.stats import pearsonr, spearmanr
    d = df.dropna(subset=[pred, true])
    out = {'N': len(d)}
    if len(d) < 3:
        return out
    e = d[pred] - d[true]
    out.update(PCC=float(pearsonr(d[true], d[pred])[0]), Spearman=float(spearmanr(d[true], d[pred])[0]),
               RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(e.abs().mean()))
    if sigma is not None:
        out['Spearman_err_sigma'] = float(spearmanr(e.abs(), d[sigma])[0])
    return out
