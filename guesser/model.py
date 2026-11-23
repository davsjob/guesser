"""
GUESSER: evidential graph neural network for mutation ddG with uncertainty.

The wild-type and mutant graphs share nodes and edges and differ only in node features.
For each residue, the wild-type features and the mutation-induced change (MT - WT) are
concatenated and encoded in one pass by four GATv2 layers. The network predicts

    mu = gamma_phys + gamma_base + gamma_epi

and the Normal-Inverse-Gamma parameters (v, alpha, beta) of an evidential head:

    aleatoric variance  = beta / (alpha - 1)
    epistemic variance  = beta / (v * (alpha - 1))

* gamma_phys: linear prior on the summed MadraX energy differences (11 terms)
* gamma_base: structural head on a damped, pre-norm residual GATv2 + RBF backbone
* gamma_epi:  epistasis head, attentional pooling over mutated nodes; hard-zeroed for singles

Module and attribute names match the training code, so the released checkpoints load
with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

NUM_MADRAX_TERMS = 11

# Architecture of the released GUESSER ensemble.
GUESSER_MODEL_KWARGS = dict(hidden_dim=256, num_layers=4, edge_dim=2, fix_mode="A", alpha_mode="unbounded")


class RBFExpansion(nn.Module):
    """Gaussian radial basis expansion of CA-CA distances (learnable widths)."""

    def __init__(self, num_kernels=32, cutoff=10.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_kernels)
        self.register_buffer('centers', centers)
        self.scales = nn.Parameter(torch.ones(num_kernels) * (num_kernels / cutoff))

    def forward(self, dists):
        if dists.dim() == 1:
            dists = dists.unsqueeze(-1)
        diff = dists - self.centers.unsqueeze(0)
        return torch.exp(-self.scales.unsqueeze(0) * diff**2)


class DampedResidualGATBlock(nn.Module):
    """Pre-norm GATv2 + distance-weighted message branch, added to the residual with a
    learnable damping factor (stabilises variance across layers)."""

    def __init__(self, dim, edge_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(dim, dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.rbf = RBFExpansion(num_kernels=32, cutoff=10.0)
        self.rbf_proj = nn.Linear(32, dim)
        self.rbf_linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.damping = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr, edge_dists):
        residual = x
        x_normed = self.norm(x)

        x_attn = F.gelu(self.gat(x_normed, edge_index, edge_attr))

        rbf_w = self.rbf_proj(self.rbf(edge_dists))
        x_j = x_normed[edge_index[1]]
        x_rbf = torch.zeros_like(x)
        x_rbf.index_add_(0, edge_index[0], rbf_w * self.rbf_linear(x_j))

        out = self.dropout(x_attn + x_rbf)
        return residual + self.damping * out


class ResidualGATBlock(nn.Module):
    """Post-norm variant without damping; only used by the use_damping=False ablation."""

    def __init__(self, dim, edge_dim, heads=4, dropout=0.35):
        super().__init__()
        self.gat = GATv2Conv(dim, dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.rbf_layer = RBFExpansion(num_kernels=32, cutoff=10.0)
        self.rbf_proj = nn.Linear(32, dim)
        self.rbf_linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_dists):
        residual = x
        x_attn = F.gelu(self.gat(x, edge_index, edge_attr))
        rbf_weights = self.rbf_proj(self.rbf_layer(edge_dists))
        x_j = x[edge_index[1]]
        x_rbf = torch.zeros_like(x)
        x_rbf.index_add_(0, edge_index[0], rbf_weights * self.rbf_linear(x_j))
        return self.norm(self.dropout(x_attn + x_rbf + residual))


class EpistasisPooling(nn.Module):
    """Attentional aggregation over the embeddings of the mutated nodes of each graph."""

    def __init__(self, hidden_dim):
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pooler = AttentionalAggregation(gate_nn=gate_nn, nn=nn_transform)

    def forward(self, node_embeddings, mut_mask, batch_index):
        mut_nodes = node_embeddings[mut_mask]
        mut_batch = batch_index[mut_mask]
        if mut_nodes.size(0) == 0:
            return torch.zeros((batch_index.max().item() + 1, node_embeddings.size(-1)),
                               device=node_embeddings.device)
        return self.pooler(mut_nodes, mut_batch)


class GUESSER(nn.Module):
    """
    Args:
        node_features: node feature dimension (23 for physchem_v1: 11 intrinsic, SASA, 11 MadraX)
        fix_mode: 'A' (released model) normalises and projects the three latents before the
            uncertainty head; any other value concatenates them raw (earlier variant)
        alpha_mode: 'unbounded' (released model), 'cap100' or 'cap50'
        use_physics / use_epistasis / use_masking / use_damping: architecture ablations
    """

    def __init__(self, node_features, hidden_dim=128, num_layers=4, edge_dim=2, dropout=0.2,
                 use_physics=True, use_epistasis=True, use_masking=True, use_damping=True,
                 alpha_mode='cap50', fix_mode='C'):
        super().__init__()
        self.fix_mode = fix_mode
        self.use_physics = use_physics
        self.use_epistasis = use_epistasis
        self.use_masking = use_masking
        self.use_damping = use_damping
        self.alpha_mode = alpha_mode
        self.hidden_dim = hidden_dim

        if node_features <= 15:
            raise ValueError("GUESSER requires the MadraX energy features.")

        # Submodules are created in the training code's order, so a given seed
        # initialises identical weights.
        self.node_enc = nn.Sequential(
            nn.Linear(node_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        block = DampedResidualGATBlock if use_damping else ResidualGATBlock
        self.backbone = nn.ModuleList([
            block(dim=hidden_dim, edge_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])

        self.phys_linear = nn.Linear(NUM_MADRAX_TERMS, 1, bias=False)
        self.phys_mlp = nn.Sequential(
            nn.Linear(NUM_MADRAX_TERMS, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.epistasis_decoder = EpistasisPooling(hidden_dim=hidden_dim)

        # z_struct = [global mean pool, sum pool over mutated nodes] -> hidden_dim * 2
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.epi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.base_head[-1].weight)
        nn.init.zeros_(self.base_head[-1].bias)
        nn.init.zeros_(self.epi_head[-1].weight)
        nn.init.zeros_(self.epi_head[-1].bias)

        if self.fix_mode in ["A", "C"]:
            self.struct_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.ln_struct = nn.LayerNorm(hidden_dim)
            self.ln_epi = nn.LayerNorm(hidden_dim)
            self.ln_phys = nn.LayerNorm(hidden_dim)
            uncert_in = hidden_dim * 3
        else:
            uncert_in = hidden_dim * 4
        self.uncert_head = nn.Sequential(
            nn.Linear(uncert_in, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)  # v, alpha, beta
        )

    def forward(self, batch_wt, batch_mt):
        """Returns (mu, v, alpha, beta, gamma_epi, gamma_phys, gamma_base, raw_alpha), each [G, 1]."""
        x_diff = batch_mt.x - batch_wt.x
        madrax_sum = global_add_pool(x_diff[:, -NUM_MADRAX_TERMS:], batch_wt.batch)
        n_graphs = batch_wt.batch.max() + 1

        if self.use_physics:
            gamma_phys = self.phys_linear(madrax_sum)
            z_phys = self.phys_mlp(madrax_sum)
        else:
            gamma_phys = torch.zeros(n_graphs, 1, device=x_diff.device)
            z_phys = torch.zeros(n_graphs, self.hidden_dim, device=x_diff.device)

        x_ctx = torch.cat([batch_wt.x, x_diff], dim=-1)
        h = F.gelu(self.node_enc(x_ctx))
        e_feat = F.gelu(self.edge_enc(batch_wt.edge_attr))
        edge_dists = batch_wt.edge_attr[:, 1:2]
        for layer in self.backbone:
            h = layer(h, batch_wt.edge_index, e_feat, edge_dists)

        g_pool = global_mean_pool(h, batch_wt.batch)
        l_pool = global_add_pool(h * batch_wt.mut_mask.unsqueeze(-1), batch_wt.batch)
        z_struct = torch.cat([g_pool, l_pool], dim=-1)

        if self.use_epistasis:
            z_epi = self.epistasis_decoder(h, batch_wt.mut_mask, batch_wt.batch)
        else:
            z_epi = torch.zeros(n_graphs, self.hidden_dim, device=h.device)

        # Single mutants get exactly zero epistasis
        if self.use_masking:
            num_muts = global_add_pool(batch_wt.mut_mask.float(), batch_wt.batch)
            hard_gate = (num_muts > 1.5).float().unsqueeze(-1)
            z_epi = z_epi * hard_gate
        else:
            hard_gate = torch.ones(n_graphs, 1, device=h.device)

        gamma_base = self.base_head(z_struct)
        gamma_epi = self.epi_head(z_epi) * hard_gate
        mu = gamma_phys + gamma_base + gamma_epi

        if self.fix_mode in ["A", "C"]:
            z_joint = torch.cat([self.ln_struct(self.struct_proj(z_struct)),
                                 self.ln_epi(z_epi),
                                 self.ln_phys(z_phys)], dim=-1)
        else:
            z_joint = torch.cat([z_struct, z_epi, z_phys], dim=-1)
        uncert_out = self.uncert_head(z_joint)

        v = F.softplus(uncert_out[:, 0:1]) + 1e-6
        raw_alpha = uncert_out[:, 1:2]
        alpha = F.softplus(raw_alpha) + 1.0 + 1e-6
        if self.alpha_mode == 'cap100':
            alpha = torch.clamp(alpha, min=1.0, max=100.0)
        elif self.alpha_mode != 'unbounded':
            alpha = torch.clamp(alpha, min=1.0, max=50.0)
        beta = F.softplus(uncert_out[:, 2:3]) + 1e-6

        return mu, v, alpha, beta, gamma_epi, gamma_phys, gamma_base, raw_alpha


def load_model(weight_path, node_features=23, device="cpu", dtype=torch.float32, **overrides):
    """Build the released architecture and load one checkpoint strictly."""
    kwargs = {**GUESSER_MODEL_KWARGS, **overrides}
    model = GUESSER(node_features=node_features, **kwargs)
    state = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval()
