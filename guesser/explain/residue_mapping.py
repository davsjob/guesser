# -*- coding: utf-8 -*-
"""
Node -> PDB residue mapping for cached mutation graphs.

Graph nodes are NOT the residues of the whole protein. guesser.graphs builds
each graph from the union of cutoff-Å neighbourhoods (residue_neighbourhoods) of the
mutated residues, sorted by (chain, resnum), with edges taken from the global CA
graph (ca_graph_edges) via subgraph(relabel_nodes=True). This module rebuilds that
node list with the same functions and accepts a mapping only if it
reproduces the cached graph:
  * same number of nodes
  * mutated node indices == torch.where(mut_mask)
  * WT intrinsic features x_wt[:, 0:11] == AA_FEATURES[resname] for every node
  * identical edge_index and edge_attr (cos angle, CA distance) within tolerance
Several PDB folders exist per benchmark (original / fixed / re-downloaded), so
candidate folders are tried in order until one verifies.
"""
import ast
import itertools
import math
import os
import warnings

import numpy as np
import torch
import Bio.PDB
from torch_geometric.utils import subgraph

from ..features import AA_FEATURES, INTRINSIC_NAMES
from ..structure import ca_graph_edges as precompute_edges
from ..structure import residue_neighbourhoods as precompute_res_NB_dicts


INTRINSIC_SLICE = slice(0, len(INTRINSIC_NAMES))

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'MSE': 'M',
}


class _Structure:
    """Per-PDB precomputation, identical to guesser.graphs.build_graphs."""

    def __init__(self, path, cutoff):
        parser = Bio.PDB.PDBParser(QUIET=True)
        model = parser.get_structure('s', path)[0]
        for chain in model:
            if chain.id.strip() == "":
                chain.id = "A"
        edge_list, edge_features, res_to_idx, _, _, _ = precompute_edges(model, edge_thresh=cutoff)
        self.res_to_idx = res_to_idx
        self.nb_dict = precompute_res_NB_dicts(model, threshold=cutoff)
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_features, dtype=torch.float)


class ResidueMapper:
    def __init__(self, pdb_dirs, cutoff=9.0, atol=1e-3):
        self.pdb_dirs = [d for d in pdb_dirs if os.path.isdir(d)]
        missing = [d for d in pdb_dirs if not os.path.isdir(d)]
        if missing:
            warnings.warn(f"PDB dirs not found (skipped): {missing}")
        self.cutoff = cutoff
        self.atol = atol
        self._cache = {}

    def _structure(self, pdb_dir, pdb_name):
        key = (pdb_dir, pdb_name)
        if key not in self._cache:
            fname = pdb_name if pdb_name.endswith('.pdb') else pdb_name + '.pdb'
            stem = fname[:-4]
            path = None
            for cand in (fname, stem.upper() + '.pdb', stem.lower() + '.pdb'):
                p = os.path.join(pdb_dir, cand)
                if os.path.isfile(p):
                    path = p
                    break
            s = None
            if path is not None:
                try:
                    s = _Structure(path, self.cutoff)
                    s.path = path
                except Exception as e:  # unreadable PDB: treat as not available
                    warnings.warn(f"Failed to parse {path}: {e}")
            self._cache[key] = s
        return self._cache[key]

    @staticmethod
    def _parse_mutation(mutation):
        mut_list = ast.literal_eval(mutation) if isinstance(mutation, str) else mutation
        parsed = []
        for item in mut_list:
            if len(item) == 4:
                chain_id, wt, pos, mt = item
            elif len(item) == 3:
                wt, pos, mt = item
                chain_id = 'A'
            else:
                continue
            parsed.append((chain_id, wt, int(pos), mt))
        return parsed

    def _candidate_orders(self, union):
        """Sort exactly as guesser.graphs; enumerate orders of (chain, resnum) ties (insertion codes)."""
        base = sorted(union, key=lambda r: (r.get_parent().id, r.id[1]))
        groups = [list(g) for _, g in itertools.groupby(base, key=lambda r: (r.get_parent().id, r.id[1]))]
        tied = [g for g in groups if len(g) > 1]
        if not tied:
            yield base
            return
        n_perm = int(np.prod([math.factorial(len(g)) for g in tied]))
        if n_perm > 120:
            yield base
            return
        for combo in itertools.product(*[list(itertools.permutations(g)) for g in groups]):
            yield [r for grp in combo for r in grp]

    def _try(self, s, parsed, g_wt):
        union, centers = set(), []
        for chain_id, _, pos, _ in parsed:
            for res in s.res_to_idx.keys():
                if res.id[1] == pos and res.get_parent().id == chain_id:
                    union.update(s.nb_dict[res])
                    centers.append(res)
                    break
        if not union:
            return None, 'mutated residue not found in PDB'
        n_nodes = g_wt.x.shape[0]
        if len(union) != n_nodes:
            return None, f'node count {len(union)} != graph {n_nodes}'
        graph_mut = sorted(torch.where(g_wt.mut_mask.cpu())[0].tolist())
        x_intr = g_wt.x[:, INTRINSIC_SLICE].cpu().numpy()
        reason = 'no candidate order matched'
        for order in self._candidate_orders(union):
            if sorted(order.index(c) for c in centers) != graph_mut:
                reason = 'mutated node indices differ'
                continue
            feats = np.stack([AA_FEATURES.get(r.get_resname(), AA_FEATURES['ALA']) for r in order])
            if not np.allclose(feats, x_intr, atol=1e-4):
                reason = 'per-node amino-acid features differ'
                continue
            local = torch.tensor([s.res_to_idx[r] for r in order], dtype=torch.long)
            ei, ea = subgraph(local, s.edge_index, s.edge_attr, relabel_nodes=True,
                              num_nodes=len(s.res_to_idx))
            if ei.shape != g_wt.edge_index.shape or not torch.equal(ei, g_wt.edge_index.cpu()):
                reason = 'edge topology differs'
                continue
            if not torch.allclose(ea, g_wt.edge_attr.cpu().float(), atol=self.atol):
                reason = 'edge geometry (CA distance / angle) differs'
                continue
            return order, 'ok'
        return None, reason

    def map_graph(self, pdb_name, mutation, g_wt):
        """Returns (residue_info list | None, status string, wt_identity_ok bool | None)."""
        parsed = self._parse_mutation(mutation)
        reasons = []
        for d in self.pdb_dirs:
            s = self._structure(d, pdb_name)
            if s is None:
                continue
            order, reason = self._try(s, parsed, g_wt)
            if order is None:
                reasons.append(f'{d}: {reason}')
                continue
            info = []
            for i, r in enumerate(order):
                chain, resnum, icode = r.get_parent().id, r.id[1], r.id[2].strip()
                name3 = r.get_resname()
                name1 = THREE_TO_ONE.get(name3, 'X')
                info.append({
                    'node_idx': i, 'chain': chain, 'res_id': int(resnum), 'icode': icode,
                    'res_name_3': name3, 'res_name_1': name1,
                    'label': f"{name3} {resnum}{icode}({chain})",
                    'short_label': f"{name1}{resnum}{icode}",
                })
            # Does the annotated WT residue agree with the structure? (dataset quality flag)
            wt_ok = True
            for chain_id, wt, pos, _ in parsed:
                hits = [x for x in info if x['chain'] == chain_id and x['res_id'] == pos]
                wt3 = wt.upper() if len(wt) == 3 else None
                if not hits or (wt3 is not None and hits[0]['res_name_3'] != wt3) or \
                        (wt3 is None and hits[0]['res_name_1'] != wt.upper()):
                    wt_ok = False
            return info, f'verified:{s.path}', wt_ok
        if not reasons:
            return None, f'unverified: {pdb_name} not found in {self.pdb_dirs}', None
        return None, 'unverified: ' + ' | '.join(reasons), None
