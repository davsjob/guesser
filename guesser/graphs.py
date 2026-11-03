"""
Build (wild-type, mutant) graph pairs from a structure and a mutation table.

For each mutation:
  * nodes: every residue with any atom within `radius` A of a mutated residue's CA
    (union over mutated residues), sorted by (chain, resnum)
  * edges: the global CA graph restricted to those nodes (CA-CA <= edge_cutoff, symmetric,
    no self loops), features [cos(CA->CB_i, CA->CB_j), CA-CA distance]
  * node features (23): intrinsic properties (mutant residue swapped in the MT graph), SASA
    (WT structure, same in both graphs), and 11 MadraX per-residue energies
  * MadraX is run in "isolated" mode: WT energies from one WT-only call per PDB, MT energies
    from a call containing just that mutant, so every feature depends only on
    (structure, mutation) and not on batching or sharding

The released model was trained on graphs built with radius = edge_cutoff = 9 A.

Mutation tables need columns `pdb`, `mut_list` and (for training/evaluation) `ddg`, where
mut_list is a list of (chain, wt, position, mt) tuples, e.g. [('A', 'LEU', 45, 'PHE')].
Rows are processed grouped by PDB (sorted), in table order within a PDB.
"""
import ast
import gc
import os
import re
import warnings

import Bio.PDB
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm

from .features import AA_FEATURES, INTRINSIC_SLICE
from .structure import (
    ONE_TO_THREE, ca_graph_edges, compute_sasa, global_graph, madrax_wt_energies,
    require_madrax, residue_neighbourhoods, run_madrax,
)

THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}


def make_graph_pair(local_graph, mut_indices, mut_res_names, wt_energy_vecs=None, mt_energy_vecs=None):
    """WT and MT graphs sharing topology; MT gets the mutant's intrinsic features at mutated
    nodes, and each graph gets its own MadraX energies appended."""
    g_wt, g_mt = local_graph.clone(), local_graph.clone()
    base_x = local_graph.x.clone()
    x_wt, x_mt = base_x.clone(), base_x.clone()

    for m_idx, mut_name in zip(mut_indices, mut_res_names):
        feat_mt = AA_FEATURES.get(mut_name, AA_FEATURES['ALA'])
        x_mt[m_idx, INTRINSIC_SLICE] = torch.tensor(feat_mt, dtype=torch.float)

    if wt_energy_vecs is not None:
        x_wt = torch.cat([x_wt, torch.tensor(wt_energy_vecs, dtype=torch.float)], dim=1)
        x_mt = torch.cat([x_mt, torch.tensor(mt_energy_vecs, dtype=torch.float)], dim=1)
    g_wt.x, g_mt.x = x_wt, x_mt

    mask = torch.zeros(g_wt.num_nodes, dtype=torch.bool)
    mask[mut_indices] = True
    g_wt.mut_mask = g_mt.mut_mask = mask
    g_wt.feature_representation = "physchem_v1"
    g_mt.feature_representation = "physchem_v1"
    return g_wt, g_mt


def _energies_for(union_list, mx_energy_tensor, mx_res_map, wt_tensor):
    """Per-node (WT, MT) MadraX energy vectors; zeros where a residue is missing from MadraX."""
    local_i = 0  # isolated mode: one mutant per MadraX call, at conformer index 1
    wt_v, mt_v = [], []
    for res in union_list:
        c_id = res.get_parent().id if res.get_parent().id.strip() else 'A'
        key = (c_id, res.id[1], res.id[2])
        if key in mx_res_map:
            res_idx = mx_res_map[key]
            try:
                if (local_i + 1) < mx_energy_tensor.shape[3]:
                    wt_v.append(wt_tensor[0, res_idx[0], res_idx[1], 0, :].detach().numpy())
                    mt_v.append(mx_energy_tensor[0, res_idx[0], res_idx[1], local_i + 1, :].detach().numpy())
                else:
                    wt_v.append(np.zeros(11)); mt_v.append(np.zeros(11))
            except IndexError:
                wt_v.append(np.zeros(11)); mt_v.append(np.zeros(11))
        else:
            wt_v.append(np.zeros(11)); mt_v.append(np.zeros(11))
    return np.array(wt_v), np.array(mt_v)


def _as_mut_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def build_graphs(df, pdb_dir, radii=(9.0,), edge_cutoff=9.0, device=None, progress_callback=None):
    """
    Build graph pairs for every row of `df` at each neighbourhood radius.

    Returns {radius: [{'pdb', 'mutation', 'row', 'graph_pair': (g_wt, g_mt)}, ...]}, where row is the
    df index label. A mutation that fails at any radius is dropped at all radii. g_mt.y holds the
    row's ddg (0.0 if absent).
    """
    require_madrax()
    from madrax import utils as mx_utils

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    radii = sorted(float(r) for r in radii)
    parser = Bio.PDB.PDBParser(QUIET=True)
    if "ddg" not in df.columns:
        df = df.assign(ddg=0.0)

    structures = {}
    for pdb_id in tqdm(df['pdb'].unique(), desc="Loading PDB files"):
        file_path = os.path.join(pdb_dir, pdb_id if pdb_id.endswith('.pdb') else pdb_id + '.pdb')
        try:
            model = parser.get_structure(pdb_id, file_path)[0]
            for chain in model:
                if chain.id.strip() == "":
                    chain.id = "A"
            structures[pdb_id] = model
        except Exception as e:
            warnings.warn(f"Failed {pdb_id}: {e}")
    structures = compute_sasa(structures)

    out = {r: [] for r in radii}
    processed = failures = 0
    total_mutations = len(df)

    for pdb_id, group in df.groupby('pdb'):
        if pdb_id not in structures:
            continue
        model = structures[pdb_id]

        edge_list, edge_features, res_to_idx, _, ca_coords, _ = ca_graph_edges(model, edge_thresh=edge_cutoff)
        nb_dicts = {r: residue_neighbourhoods(model, threshold=r) for r in radii}
        whole = global_graph(list(res_to_idx.keys()), edge_list, AA_FEATURES, edge_features, ca_coords)

        coords = atnames = wt_tensor = None
        fname = pdb_id if pdb_id.endswith('.pdb') else pdb_id + '.pdb'
        try:
            mx_data = mx_utils.parsePDB(os.path.join(pdb_dir, fname))
            if len(mx_data) >= 2:
                coords, atnames = mx_data[0], mx_data[1]
                if len(atnames) > 0 and isinstance(atnames[0], str):
                    atnames = [atnames]
        except Exception as e:
            print(f"MadraX load failed for {pdb_id}: {e}")
            coords = None
        if coords is not None:
            try:
                wt_tensor = madrax_wt_energies(coords, atnames, device)
            except Exception as e:
                print(f"MadraX WT calc failed for {pdb_id}: {e}")
                coords = None

        for row_id, row in group.iterrows():
            mx_energy_tensor = mx_res_map = None
            mut_list = _as_mut_list(row['mut_list'])
            if coords is not None:
                try:
                    mx_energy_tensor, mx_res_map = run_madrax(coords, atnames, [mut_list], device=device)
                except Exception as e:
                    print(f"MadraX calc failed for {pdb_id} {mut_list}: {e}")
            if mx_energy_tensor is not None and mx_energy_tensor.device.type != 'cpu':
                mx_energy_tensor = mx_energy_tensor.cpu()

            try:
                items = {}
                for r in radii:
                    union_residues, mut_node_indices = set(), []
                    for item in mut_list:
                        if len(item) == 4:
                            chain_id, _, pos, mut_n = item
                        elif len(item) == 3:
                            _, pos, mut_n = item
                            chain_id = 'A'
                        else:
                            continue
                        for res in res_to_idx.keys():
                            if res.id[1] == pos and res.get_parent().id == chain_id:
                                union_residues.update(nb_dicts[r][res])
                                mut_node_indices.append((chain_id, mut_n, res))
                                break
                    if not union_residues:
                        raise ValueError(f"Empty neighbourhood for mutation {mut_list}")

                    union_list = sorted(list(union_residues), key=lambda q: (q.get_parent().id, q.id[1]))
                    local_indices = torch.tensor([res_to_idx[q] for q in union_list], dtype=torch.long)
                    edge_idx, edge_att = subgraph(local_indices, whole.edge_index, whole.edge_attr, relabel_nodes=True)
                    local_graph = Data(x=whole.x[local_indices].clone(), edge_index=edge_idx, edge_attr=edge_att)

                    mut_idxs, mt_names = [], []
                    for _, mut_n, center_res in mut_node_indices:
                        mut_idxs.append(union_list.index(center_res))
                        mt_names.append(mut_n.upper() if len(mut_n) == 3 else ONE_TO_THREE.get(mut_n, mut_n))

                    wt_e = mt_e = None
                    if mx_energy_tensor is not None and mx_res_map is not None:
                        wt_e, mt_e = _energies_for(union_list, mx_energy_tensor, mx_res_map, wt_tensor)

                    g_wt, g_mt = make_graph_pair(local_graph, mut_idxs, mt_names, wt_e, mt_e)
                    g_mt.y = torch.tensor([row["ddg"]], dtype=torch.float)
                    items[r] = {'pdb': pdb_id, 'mutation': str(mut_list), 'row': int(row_id), 'graph_pair': (g_wt, g_mt)}
                for r in radii:
                    out[r].append(items[r])
                processed += 1
            except Exception:
                failures += 1
            if progress_callback:
                progress_callback(processed, failures, total_mutations)

            del mx_energy_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return out


def check_graphs(graphs, node_dim=23):
    """Raise on malformed node features (shape or NaN), as the training pipeline did."""
    for item in graphs:
        g_wt, g_mt = item["graph_pair"]
        if g_wt.x.shape != (g_wt.num_nodes, node_dim) or g_wt.x.shape != g_mt.x.shape:
            raise ValueError(f"bad node features for {item['pdb']} {item['mutation']}: {tuple(g_wt.x.shape)}")
        if torch.isnan(g_wt.x).any() or torch.isnan(g_mt.x).any():
            raise ValueError(f"NaN features for {item['pdb']} {item['mutation']}")


_MUT_RE = re.compile(r"^(?:([A-Za-z0-9]):)?([A-Za-z]{1,3}?)(-?\d+)([A-Za-z]{1,3})$")


def parse_mutation(text, default_chain="A"):
    """
    Parse one variant, e.g. "A:L45F", "L45F", "A:LEU45PHE" or "L45F+G50A" (multi-mutant),
    into a mut_list [(chain, WT3, pos, MT3), ...].
    """
    out = []
    for part in re.split(r"[+;,/ ]+", text.strip()):
        if not part:
            continue
        m = _MUT_RE.match(part)
        if not m:
            raise ValueError(f"Cannot parse mutation {part!r}; use e.g. A:L45F or L45F+G50A")
        chain, wt, pos, mt = m.groups()
        wt3 = wt.upper() if len(wt) == 3 else ONE_TO_THREE[wt.upper()]
        mt3 = mt.upper() if len(mt) == 3 else ONE_TO_THREE[mt.upper()]
        out.append((chain or default_chain, wt3, int(pos), mt3))
    return out


def mutation_table(pdb, mutations, default_chain="A"):
    """DataFrame for build_graphs from a PDB name and a list of mutation strings."""
    rows = [{"pdb": pdb, "mut_list": parse_mutation(m, default_chain), "ddg": 0.0, "input": m} for m in mutations]
    return pd.DataFrame(rows)


def load_table(paths):
    """Concatenate CSV/XLSX tables in the order given (training used S then D for Megascale)."""
    if isinstance(paths, str):
        paths = [paths]
    dfs = [pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)
