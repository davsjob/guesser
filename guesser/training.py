"""
Training of one ensemble member.

Loss per batch (targets in kcal/mol, no standardisation):

    L = EDL(fwd) + EDL(rev) + lambda_antisym * mean((mu_fwd + mu_rev)^2)

where EDL is the Normal-Inverse-Gamma NLL plus lambda_reg * |y - mu| * (2v + alpha)
(Amini et al. 2020), fwd = model(WT, MT) with target ddG and rev = model(MT, WT) with
target -ddG. lambda_reg ramps linearly from 0 over the first `edl_warmup_epochs` epochs.
The checkpoint with the best validation Pearson r is kept.

This is the evidential path of the original training loop with the options the released
model did not use (Bayesian heads, SAM, SWA, EMA, target scaling, quantile EDL, DropEdge)
removed. Operation order and random number use are unchanged, so a given seed and graph
set reproduce the original run.
"""
import csv
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from .datasets import CoordinateNoiseDataset, GraphPairDataset, ReverseAugmentedDataset, collate_pairs
from .features import apply_feature_set
from .model import GUESSER


@dataclass
class TrainConfig:
    """Defaults are the settings of the released ensemble (seeds 42-49)."""
    seed: int = 42
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 25
    hidden_dim: int = 256
    num_layers: int = 4
    alpha_mode: str = "unbounded"
    fix_mode: str = "A"
    edl_lambda_reg: float = 0.05
    edl_warmup_epochs: int = 30
    lambda_antisym: float = 1.0
    reverse_augmentation: bool = True
    coord_noise_sigma: float = 0.2      # no effect on graphs without .ca_coords (see README)
    double_weight: float = 1.0          # loss weight for multi-mutants; 1.0 = unweighted (released model)
    mutation_type: str = "all"          # all | single | double
    feature_set: str = "all"            # all | madrax_only | no_madrax
    use_physics: bool = True
    use_epistasis: bool = True
    use_masking: bool = True
    use_damping: bool = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Evidential loss
# ----------------------------------------------------------------------------
def nig_nll_loss(gamma, v, alpha, beta, target, weight=None):
    """Normal-Inverse-Gamma negative log-likelihood."""
    omega = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(math.pi / v)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((target - gamma) ** 2 * v + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    if weight is not None:
        return (nll * weight).mean()
    return nll.mean()


def nig_reg_loss(gamma, v, alpha, beta, target, weight=None):
    """Evidence regulariser |y - gamma| * (2v + alpha): penalises confident errors."""
    error = torch.abs(target - gamma)
    evidence = 2.0 * v + alpha
    if weight is not None:
        return (error * evidence * weight).mean()
    return (error * evidence).mean()


def evidential_loss(target, gamma, v, alpha, beta, lambda_reg=0.05, weight=None):
    return nig_nll_loss(gamma, v, alpha, beta, target, weight=weight) + \
        lambda_reg * nig_reg_loss(gamma, v, alpha, beta, target, weight=weight)


def batch_loss(model, batch_wt, batch_mt, raw_target, lambda_antisym, lambda_reg, double_weight=1.0):
    """Forward + reverse EDL loss and antisymmetry penalty. Returns (loss, fwd, rev, antisym, out_fwd)."""
    t_mean, t_std = 0.0, 1.0  # targets are not standardised
    target_fwd = (raw_target - t_mean) / (t_std + 1e-8)
    target_rev = (-raw_target - t_mean) / (t_std + 1e-8)

    weight = None
    if double_weight != 1.0:
        num_muts = global_add_pool(batch_wt.mut_mask.float(), batch_wt.batch)
        weight = torch.where(num_muts > 1.5, double_weight, 1.0).unsqueeze(-1)

    out_fwd = model(batch_wt, batch_mt)
    gamma_fwd, v_fwd, alpha_fwd, beta_fwd = out_fwd[:4]
    loss_fwd = evidential_loss(target_fwd, gamma_fwd, v_fwd, alpha_fwd, beta_fwd, lambda_reg, weight)

    if lambda_antisym > 0.0:
        out_rev = model(batch_mt, batch_wt)
        gamma_rev, v_rev, alpha_rev, beta_rev = out_rev[:4]
        loss_rev = evidential_loss(target_rev, gamma_rev, v_rev, alpha_rev, beta_rev, lambda_reg, weight)
        pred_real_fwd = (gamma_fwd * t_std) + t_mean
        pred_real_rev = (gamma_rev * t_std) + t_mean
        loss_antisym = torch.mean((pred_real_fwd + pred_real_rev) ** 2)
    else:
        loss_rev = torch.tensor(0.0, device=loss_fwd.device)
        loss_antisym = torch.tensor(0.0, device=loss_fwd.device)

    loss = loss_fwd + loss_rev + (lambda_antisym * loss_antisym)
    return loss, loss_fwd, loss_rev, loss_antisym, out_fwd


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, lambda_reg):
    """Returns (mse, loss, pearson_r, stats) on a loader (targets in kcal/mol)."""
    model.eval()
    total_mse, total_nll = 0.0, 0.0
    preds, targets, alphas, vs, betas, n_muts = [], [], [], [], [], []
    for batch_wt, batch_mt in loader:
        batch_wt, batch_mt = batch_wt.to(device), batch_mt.to(device)
        target = batch_mt.y.float().view(-1, 1)
        target_scaled = (target - 0.0) / (1.0 + 1e-8)
        gamma, v, alpha, beta = model(batch_wt, batch_mt)[:4]
        total_nll += evidential_loss(target_scaled, gamma, v, alpha, beta, lambda_reg=lambda_reg).item()
        pred = (gamma * 1.0) + 0.0
        total_mse += F.mse_loss(pred, target).item()
        preds.extend(pred.cpu().numpy())
        alphas.extend(alpha.cpu().numpy())
        vs.extend(v.cpu().numpy())
        betas.extend(beta.cpu().numpy())
        targets.extend(target.cpu().numpy())
        n_muts.extend(global_add_pool(batch_wt.mut_mask.float(), batch_wt.batch).cpu().numpy())

    nb = len(loader)
    preds, targets = np.array(preds).flatten(), np.array(targets).flatten()
    corr = np.corrcoef(targets, preds)[0, 1] if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0 else 0.0

    a = np.array(alphas).flatten()
    v_ = np.array(vs).flatten()
    b = np.array(betas).flatten()
    a_safe, v_safe = np.maximum(a, 1.0 + 1e-8), np.maximum(v_, 1e-8)
    aleatoric = b / (a_safe - 1.0)
    epistemic = b / (v_safe * (a_safe - 1.0))
    stats = {
        'alpha_mean': float(a.mean()), 'alpha_min': float(a.min()), 'alpha_max': float(a.max()),
        'v_mean': float(v_.mean()), 'v_min': float(v_.min()), 'v_max': float(v_.max()),
        'beta_mean': float(b.mean()), 'beta_min': float(b.min()), 'beta_max': float(b.max()),
        'aleatoric_mean': float(aleatoric.mean()), 'epistemic_mean': float(epistemic.mean()),
    }
    nm = np.array(n_muts).flatten()
    single, double = nm <= 1.5, nm > 1.5
    stats['alpha_mean_singles'] = float(a[single].mean()) if single.any() else 0.0
    stats['alpha_mean_doubles'] = float(a[double].mean()) if double.any() else 0.0
    err = np.abs(targets - preds)
    for name, sel in (('singles', single), ('doubles', double)):
        rho = 0.0
        if sel.sum() > 1 and np.std(err[sel]) > 0 and np.std(epistemic[sel]) > 0:
            rho, _ = spearmanr(err[sel], epistemic[sel])
        stats[f'spearman_{name}'] = float(rho)
    if stats['alpha_min'] <= 1.0:
        print(f"  WARNING: alpha_min={stats['alpha_min']:.6f} <= 1.0", flush=True)
    return total_mse / nb, total_nll / nb, corr, stats


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
LOG_FIELDS = [
    'epoch', 'lr', 'train_nll', 'train_antisym', 'val_nll', 'val_mse', 'val_pcc',
    'train_alpha_mean', 'train_alpha_min', 'train_alpha_max',
    'train_v_mean', 'train_v_min', 'train_v_max',
    'train_beta_mean', 'train_beta_min', 'train_beta_max',
    'val_alpha_mean', 'val_alpha_min', 'val_alpha_max',
    'val_v_mean', 'val_v_min', 'val_v_max',
    'val_beta_mean', 'val_beta_min', 'val_beta_max',
    'val_aleatoric_mean', 'val_epistemic_mean',
    'train_alpha_mean_singles', 'train_alpha_mean_doubles',
    'val_alpha_mean_singles', 'val_alpha_mean_doubles',
    'val_spearman_singles', 'val_spearman_doubles',
    'lambda_eff', 'patience_counter',
]


def _filter_mutation_type(graphs, mutation_type):
    if mutation_type == "single":
        return [g for g in graphs if g['graph_pair'][0].mut_mask.sum().item() == 1]
    if mutation_type == "double":
        return [g for g in graphs if g['graph_pair'][0].mut_mask.sum().item() >= 2]
    return graphs


def train(train_graphs, val_graphs, cfg: TrainConfig, out_dir=".", name="model", device=None):
    """
    Train one ensemble member on lists of graph pairs. Writes to out_dir:
      {name}_best_weights.pt  (best validation Pearson r; this is what was released)
      {name}_final.pt         (last epoch)
      {name}_epoch_log.csv
    Returns the path of the best checkpoint.
    """
    set_seed(cfg.seed)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, name)

    if cfg.feature_set != "all":
        apply_feature_set(train_graphs, cfg.feature_set)
        apply_feature_set(val_graphs, cfg.feature_set)
    train_graphs = _filter_mutation_type(train_graphs, cfg.mutation_type)
    val_graphs = _filter_mutation_type(val_graphs, cfg.mutation_type)

    train_ds = CoordinateNoiseDataset(
        ReverseAugmentedDataset(GraphPairDataset(train_graphs), augment=cfg.reverse_augmentation),
        sigma=cfg.coord_noise_sigma, use_noise=True)
    val_ds = CoordinateNoiseDataset(
        ReverseAugmentedDataset(GraphPairDataset(val_graphs), augment=cfg.reverse_augmentation),
        use_noise=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_pairs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_pairs, num_workers=0)

    node_dim = train_ds[0][0].x.shape[1]  # (the original also drew this sample before building the model)
    model = GUESSER(
        node_features=node_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, edge_dim=2,
        use_physics=cfg.use_physics, use_epistasis=cfg.use_epistasis, use_masking=cfg.use_masking,
        use_damping=cfg.use_damping, alpha_mode=cfg.alpha_mode, fix_mode=cfg.fix_mode,
    ).to(device)
    print(f"{name}: {len(train_graphs)} train / {len(val_graphs)} val mutations, node_dim {node_dim}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, device {device}", flush=True)
    print(f"  config: {asdict(cfg)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    with open(f"{prefix}_epoch_log.csv", 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    best_val_corr, patience_counter = -np.inf, 0
    best_path = f"{prefix}_best_weights.pt"
    n_batches = len(train_loader)
    warmup = max(1, cfg.edl_warmup_epochs)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_nll = epoch_antisym = 0.0
        all_alpha, all_v, all_beta, all_num_muts = [], [], [], []
        lambda_eff = cfg.edl_lambda_reg * min(1.0, epoch / warmup)

        for batch_wt, batch_mt in train_loader:
            batch_wt, batch_mt = batch_wt.to(device), batch_mt.to(device)
            raw_target = batch_mt.y.float().view(-1, 1)

            optimizer.zero_grad()
            loss, loss_fwd, loss_rev, loss_antisym, out_fwd = batch_loss(
                model, batch_wt, batch_mt, raw_target, cfg.lambda_antisym, lambda_eff, cfg.double_weight)
            if not torch.isfinite(loss):
                torch.save(model.state_dict(), f"{prefix}_panic_model.pt")
                raise ValueError("loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            v_fwd, alpha_fwd, beta_fwd = out_fwd[1].detach(), out_fwd[2].detach(), out_fwd[3].detach()
            if not (torch.isfinite(v_fwd).all() and torch.isfinite(alpha_fwd).all() and torch.isfinite(beta_fwd).all()):
                torch.save(model.state_dict(), f"{prefix}_panic_model.pt")
                raise ValueError("Non-finite output tensors")
            all_v.append(v_fwd)
            all_alpha.append(alpha_fwd)
            all_beta.append(beta_fwd)
            all_num_muts.append(global_add_pool(batch_wt.mut_mask.float(), batch_wt.batch).detach())

            epoch_nll += (loss_fwd.item() + loss_rev.item()) / 2.0
            epoch_antisym += loss_antisym.item()

        a = torch.cat(all_alpha).cpu().numpy().flatten()
        v_ = torch.cat(all_v).cpu().numpy().flatten()
        b = torch.cat(all_beta).cpu().numpy().flatten()
        nm = torch.cat(all_num_muts).cpu().numpy().flatten()
        train_stats = {
            'alpha_mean': float(a.mean()), 'alpha_min': float(a.min()), 'alpha_max': float(a.max()),
            'v_mean': float(v_.mean()), 'v_min': float(v_.min()), 'v_max': float(v_.max()),
            'beta_mean': float(b.mean()), 'beta_min': float(b.min()), 'beta_max': float(b.max()),
            'alpha_mean_singles': float(a[nm <= 1.5].mean()) if (nm <= 1.5).any() else 0.0,
            'alpha_mean_doubles': float(a[nm > 1.5].mean()) if (nm > 1.5).any() else 0.0,
        }
        avg_nll, avg_antisym = epoch_nll / n_batches, epoch_antisym / n_batches

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        val_mse, val_nll, val_corr, val_stats = validate(model, val_loader, device, cfg.edl_lambda_reg)

        print(f"Epoch {epoch + 1:03d}/{cfg.epochs} | LR {lr:.2e} | AntiSym {avg_antisym:.4f} | "
              f"Train NLL {avg_nll:.4f} | Val NLL {val_nll:.4f} | Val MSE {val_mse:.4f} | Val PCC {val_corr:.4f} | "
              f"a_mean {val_stats['alpha_mean']:.2f} v_mean {val_stats['v_mean']:.2f}", flush=True)

        row = {'epoch': epoch + 1, 'lr': lr, 'train_nll': avg_nll, 'train_antisym': avg_antisym,
               'val_nll': val_nll, 'val_mse': val_mse, 'val_pcc': val_corr,
               'lambda_eff': lambda_eff, 'patience_counter': patience_counter}
        row.update({f'train_{k}': x for k, x in train_stats.items()})
        row.update({f'val_{k}': x for k, x in val_stats.items()})
        with open(f"{prefix}_epoch_log.csv", 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction='ignore').writerow(row)

        if val_corr > best_val_corr:
            best_val_corr, patience_counter = val_corr, 0
            torch.save(model.state_dict(), best_path)
            print(f"  <-- best val PCC {val_corr:.4f}, saved {best_path}", flush=True)
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {cfg.patience} epochs).", flush=True)
            break

    torch.save(model.state_dict(), f"{prefix}_final.pt")
    return best_path
