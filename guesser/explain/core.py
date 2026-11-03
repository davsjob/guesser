# -*- coding: utf-8 -*-
"""
Integrated-Gradients core for the released GUESSER ensemble (guesser.model.GUESSER).
Design decisions, all checked by
tests/test_explainer.py:

1. Fidelity: every attributed quantity is computed by calling the trained model's
   own forward(batch_wt, batch_mt). No re-implementation of the architecture.
   The model sees x_diff = batch_mt.x - batch_wt.x, so feeding
   (batch_wt.x = x_wt, batch_mt.x = x_wt + x_diff) reproduces it exactly.

2. Two IG attributions per target, each with its own completeness identity:
   - "mut" (stored as signed_attr_<t>):   input x_diff, baseline 0 (no mutation),
       x_wt fixed.  sum(attr) = f(wt, mt) - f(wt, wt)
   - "ctx" (stored as signed_attr_<t>_wt): input x_wt, baseline = that protein
       graph's own per-feature node mean, x_diff fixed.
       sum(attr) = f(x_wt, x_diff) - f(mean_wt, x_diff)
   Baselines are per graph, so results are invariant to batch size/composition.

3. Integration: Gauss-Legendre quadrature on [0, 1]. The per-graph completeness
   residual is returned for every target, so the approximation error is known.

4. Graphs in a PyG batch do not interact in this model (per-node LayerNorm,
   message passing only along intra-graph edges, per-graph pooling and
   attentional aggregation), so d(sum_g f_g)/dx_node = d f_{graph(node)}/dx_node.

5. Ensemble: IG is linear in f for a fixed input, baseline and path, so the mean
   of the member attributions equals the attribution of the ensemble-mean
   function (exact for mu, gamma_phys, gamma_base and gamma_epi).
"""
import copy
import hashlib

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from ..model import GUESSER, GUESSER_MODEL_KWARGS


# Target name -> meaning. Uncertainty formulas match guesser.inference
# (aleatoric var = beta/(alpha-1), epistemic var = beta/(v(alpha-1))), taken as log-std.
TARGETS = {
    'mu':             'predicted ddG, mu = gamma_phys + gamma_base + gamma_epi',
    'phys':           'gamma_phys: linear physics prior on summed MadraX energy differences',
    'base':           'gamma_base: structural GNN head',
    'epi':            'gamma_epi: epistasis head (identically 0 for single mutants)',
    'aleatoric':      'log aleatoric std = 0.5*log(beta/(alpha-1))',
    'epistemic':      'log epistemic std = 0.5*log(beta/(v*(alpha-1)))',
    'evidence':       'v (virtual evidence for mu)',
    'total_evidence': '2v + alpha',
}
TARGET_KEYS = list(TARGETS.keys())

MADRAX_SLICE = slice(12, 23)  # model uses x[:, -11:]; node_dim is asserted to be 23


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def load_explain_model(weight_path, node_dim, device, dtype=torch.float32):
    """Build the exact training architecture and load weights strictly.
    dtype=torch.float64 evaluates the same trained function in double precision
    (weights are upcast exactly), removing float32 round-off from the IG integrand."""
    if node_dim != 23:
        raise ValueError(f"GUESSER was trained on 23 physchem_v1 features, dataset has {node_dim}")
    model = GUESSER(node_features=node_dim, **GUESSER_MODEL_KWARGS)
    state = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)  # any architecture mismatch raises
    model.to(device=device, dtype=dtype).eval()
    return model


def _forward_targets(model, batch_wt, batch_mt, x_wt, x_diff):
    """Run the model's own forward with substituted node features; return {target: [G]}."""
    bw = copy.copy(batch_wt)
    bm = copy.copy(batch_mt)
    bw.x = x_wt
    bm.x = x_wt + x_diff
    out = model(bw, bm)
    mu, v, alpha, beta, g_epi, g_phys, g_base = out[:7]
    log_beta = torch.log(beta)
    log_am1 = torch.log(alpha - 1.0)  # alpha = softplus(.) + 1 + 1e-6  =>  alpha - 1 > 0
    t = {
        'mu': mu,
        'phys': g_phys,
        'base': g_base,
        'epi': g_epi,
        'aleatoric': 0.5 * (log_beta - log_am1),
        'epistemic': 0.5 * (log_beta - torch.log(v) - log_am1),
        'evidence': v,
        'total_evidence': 2.0 * v + alpha,
    }
    return {k: val.reshape(-1) for k, val in t.items()}


def direct_outputs(model, batch_wt, batch_mt):
    """Plain model(batch_wt, batch_mt), exactly as in guesser.inference."""
    with torch.no_grad():
        out = model(batch_wt, batch_mt)
    names = ['mu', 'v', 'alpha', 'beta', 'gamma_epi', 'gamma_phys', 'gamma_base']
    return {n: out[i].reshape(-1) for i, n in enumerate(names)}


def gauss_legendre(n_steps):
    nodes, weights = np.polynomial.legendre.leggauss(n_steps)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _replicate(batch, S):
    """S disjoint copies of a batch as one bigger batch (graphs never interact, see note 4)."""
    N, G = batch.x.shape[0], int(batch.batch.max().item()) + 1
    rep = copy.copy(batch)
    rep.edge_index = torch.cat([batch.edge_index + s * N for s in range(S)], dim=1)
    rep.edge_attr = batch.edge_attr.repeat(S, 1)
    rep.batch = torch.cat([batch.batch + s * G for s in range(S)])
    rep.mut_mask = batch.mut_mask.repeat(S)
    rep.x = batch.x.repeat(S, 1)
    return rep


def _integrated_gradients(model, batch_wt, family, inp, baseline, fixed, n_steps,
                          grad_keys, max_nodes_per_pass):
    """
    IG_i = (inp_i - base_i) * integral_0^1 d(sum_g f_g)/dx_i (base + a (inp - base)) da,
    Gauss-Legendre quadrature. Up to S = max_nodes_per_pass // n_nodes quadrature nodes are
    evaluated in one forward pass by stacking S disjoint copies of the batch.
    family 'mut': inp/baseline are x_diff, fixed is x_wt; 'ctx': inp/baseline are x_wt, fixed is x_diff.
    Returns ({key: attr}, f_input, f_baseline).
    """
    N = inp.shape[0]
    alphas, weights = gauss_legendre(n_steps)
    delta = (inp - baseline).detach()
    grad_int = {k: torch.zeros_like(inp) for k in grad_keys}
    S_max = max(1, min(n_steps, max_nodes_per_pass // max(N, 1)))
    rep_cache = {}
    for c0 in range(0, n_steps, S_max):
        a_c = alphas[c0:c0 + S_max]
        w_c = weights[c0:c0 + S_max]
        S = len(a_c)
        if S not in rep_cache:
            rep_cache[S] = _replicate(batch_wt, S)
        rep = rep_cache[S]
        a_t = torch.as_tensor(a_c, dtype=inp.dtype, device=inp.device).repeat_interleave(N).unsqueeze(-1)
        x = (baseline.repeat(S, 1) + a_t * delta.repeat(S, 1)).detach().requires_grad_(True)
        fixed_rep = fixed.repeat(S, 1)
        if family == 'mut':
            t = _forward_targets(model, rep, rep, fixed_rep, x)
        else:
            t = _forward_targets(model, rep, rep, x, fixed_rep)
        for k in grad_keys:
            if not t[k].requires_grad:
                continue  # target does not depend on this input: gradient is exactly 0
            g, = torch.autograd.grad(t[k].sum(), x, retain_graph=True, allow_unused=True)
            if g is not None:
                w_t = torch.as_tensor(w_c, dtype=inp.dtype, device=inp.device).view(S, 1, 1)
                grad_int[k] += (w_t * g.view(S, N, -1)).sum(dim=0)
        del t, x
    with torch.no_grad():
        if family == 'mut':
            f_in = _forward_targets(model, batch_wt, batch_wt, fixed, inp)
            f_base = _forward_targets(model, batch_wt, batch_wt, fixed, baseline)
        else:
            f_in = _forward_targets(model, batch_wt, batch_wt, inp, fixed)
            f_base = _forward_targets(model, batch_wt, batch_wt, baseline, fixed)
    attr = {k: (delta * grad_int[k]).detach() for k in grad_keys}
    return attr, f_in, f_base


def explain_batch(model, batch_wt, batch_mt, n_steps, max_nodes_per_pass=1, fast=False):
    """
    fast=False: reference mode, one quadrature node per pass, IG backprop for all 8 targets.
    fast=True : production mode, many quadrature nodes per pass, and two exact identities
                instead of redundant backward passes:
                  attr(phys) = closed form (mut) / 0 (ctx)   [gamma_phys linear in x_diff only]
                  attr(mu)   = attr(phys) + attr(base) + attr(epi)   [IG linear in f]
                attr(epi) is exactly 0 when every graph in the batch is a single mutant
                (hard gate). explain_audit/test_explainer.py checks fast == reference.
    Returns per-graph dicts with float32 numpy arrays:
      signed_attr_<t>      [n_nodes, 23]  mutation IG (input x_diff, baseline 0)
      signed_attr_<t>_wt   [n_nodes, 23]  context IG  (input x_wt, baseline graph mean)
      exact_phys_attr      [n_nodes, 23]  closed-form gamma_phys decomposition
      phys_attr_per_node   [n_nodes]
      f_<fam>_<t>, f0_<fam>_<t>, completeness_err_<fam>_<t>   scalars
    plus a batch-level dict of direct model outputs.
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    if batch_wt.x.dtype != dtype:
        batch_wt, batch_mt = copy.copy(batch_wt), copy.copy(batch_mt)
        batch_wt.x, batch_mt.x = batch_wt.x.to(dtype), batch_mt.x.to(dtype)
        batch_wt.edge_attr = batch_wt.edge_attr.to(dtype)
    x_wt = batch_wt.x.detach()
    x_diff = (batch_mt.x - batch_wt.x).detach()
    b = batch_wt.batch

    direct = direct_outputs(model, batch_wt, batch_mt)

    # Closed form: gamma_phys = w . sum_nodes x_diff[:, 12:23] (bias-free linear layer)
    w_phys = model.phys_linear.weight.detach().reshape(-1)
    exact_phys = torch.zeros_like(x_diff)
    exact_phys[:, MADRAX_SLICE] = x_diff[:, MADRAX_SLICE] * w_phys

    if fast:
        from torch_geometric.nn import global_add_pool
        any_multi = bool((global_add_pool(batch_wt.mut_mask.double(), b) > 1.5).any())
        grad_keys = ['base', 'aleatoric', 'epistemic', 'evidence', 'total_evidence'] + \
                    (['epi'] if any_multi else [])
        chunk = max_nodes_per_pass
    else:
        grad_keys, chunk = TARGET_KEYS, 1

    wt_base = global_mean_pool(x_wt, b)[b]
    # Family "mut": x_wt fixed, attribute x_diff against the no-mutation baseline.
    attr_mut, fin_mut, f0_mut = _integrated_gradients(
        model, batch_wt, 'mut', x_diff, torch.zeros_like(x_diff), x_wt, n_steps, grad_keys, chunk)
    # Family "ctx": x_diff fixed, attribute x_wt against this graph's mean environment.
    attr_ctx, fin_ctx, f0_ctx = _integrated_gradients(
        model, batch_wt, 'ctx', x_wt, wt_base, x_diff, n_steps, grad_keys, chunk)

    if fast:
        for attr in (attr_mut, attr_ctx):
            attr.setdefault('epi', torch.zeros_like(x_diff))
        attr_mut['phys'] = exact_phys.clone()
        attr_ctx['phys'] = torch.zeros_like(x_diff)
        for attr in (attr_mut, attr_ctx):
            attr['mu'] = attr['phys'] + attr['base'] + attr['epi']

    n_graphs = int(b.max().item()) + 1
    per_graph = []
    for g in range(n_graphs):
        m = (b == g)
        d = {}
        for k in TARGET_KEYS:
            a_mut = attr_mut[k][m]
            a_ctx = attr_ctx[k][m]
            d[f'signed_attr_{k}'] = a_mut.cpu().numpy().astype(np.float32)
            d[f'signed_attr_{k}_wt'] = a_ctx.cpu().numpy().astype(np.float32)
            for fam, a, fin, f0 in (('mut', a_mut, fin_mut, f0_mut), ('ctx', a_ctx, fin_ctx, f0_ctx)):
                fi, fb = float(fin[k][g]), float(f0[k][g])
                d[f'f_{fam}_{k}'] = fi
                d[f'f0_{fam}_{k}'] = fb
                d[f'completeness_err_{fam}_{k}'] = float(a.double().sum()) - (fi - fb)
                d[f'attr_mass_{fam}_{k}'] = float(a.double().abs().sum())
        d['exact_phys_attr'] = exact_phys[m].cpu().numpy().astype(np.float32)
        d['phys_attr_per_node'] = d['exact_phys_attr'].sum(axis=1)
        per_graph.append(d)

    # Endpoint sanity: the IG path end must reproduce the plain forward pass.
    endpoint_err = max(
        float((fin_mut['mu'] - direct['mu']).abs().max()),
        float((fin_ctx['mu'] - direct['mu']).abs().max()),
    )
    direct_np = {k: v.cpu().numpy().astype(np.float64) for k, v in direct.items()}
    return per_graph, direct_np, endpoint_err
