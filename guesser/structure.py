"""
Per-structure precomputation: residue neighbourhoods, the global CA graph, SASA, and
MadraX per-residue energies. Logic is unchanged from the code that built the training
graphs; only names and comments were tidied.
"""
import warnings

import numpy as np
import torch
from Bio.PDB import NeighborSearch, Selection
from torch_geometric.data import Data

try:
    import madrax
    from madrax import dataStructures
    from madrax.ForceField import ForceField
    from madrax.mutate.mutatingEngine import mutate

    if not hasattr(madrax.ForceField, 'SolventAccessibility'):
        try:
            from madrax.energyTerms.SolventAccessibility import SolventAccessibility
            madrax.ForceField.SolventAccessibility = SolventAccessibility
        except ImportError:
            pass
    MADRAX_AVAILABLE = True
except ImportError:
    MADRAX_AVAILABLE = False

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
    'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
    'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

# Standard residues kept in the MadraX residue map (drops HOH, ligands, caps)
VALID_AAS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'MSE', 'CYX'
}


def require_madrax():
    if not MADRAX_AVAILABLE:
        raise ImportError(
            "MadraX is required to build graphs from structures: "
            "pip install 'git+https://github.com/davsjob/madrax.git' (Linux x86-64, Python 3.7-3.12).")


def run_madrax(coords, atnames, mutation_batch_list, device='cuda'):
    """
    MadraX energies for the WT (conformer 0) and each mutant in mutation_batch_list
    (conformers 1..n) of one pre-loaded structure.

    Returns (energy tensor [1, chains, residues, 1 + n, 11], {(chain, resnum, icode): (chain_idx, res_idx)}),
    or (None, None) on failure.
    """
    if not MADRAX_AVAILABLE:
        return None, None

    res_map = {}
    flat_atnames = atnames[0] if isinstance(atnames[0], list) else atnames

    # 1. Chains. Atom names come as Index_ResNum_ResName_Chain_Atom or ResName_ResNum_Atom_Chain.
    all_chains = set()
    for atom_str in flat_atnames:
        parts = atom_str.split('_')
        if len(parts) >= 4:
            if parts[0].isdigit() and (len(parts[3]) == 1 or parts[3] == ''):
                chain = parts[3]
            elif parts[1].isdigit():
                if len(parts) > 3:
                    chain = parts[3]
                else:
                    continue
            else:
                continue
            if chain == '':
                chain = 'A'
            all_chains.add(chain)
    sorted_chains = sorted(list(all_chains))

    # 2. Residue map
    for atom_str in flat_atnames:
        parts = atom_str.split('_')
        if len(parts) < 4:
            continue
        r_str, res_name, chain = None, None, None
        if parts[0].isdigit():
            if len(parts) >= 4:
                r_str = parts[1]
                res_name = parts[2]
                chain = parts[3]
        elif parts[1].isdigit():
            r_str = parts[1]
            res_name = parts[0]
            if len(parts) > 3:
                chain = parts[3]
        if r_str is None or res_name is None:
            continue
        if chain == '':
            chain = 'A'
        if res_name not in VALID_AAS:
            continue
        if r_str[-1].isalpha():
            try:
                r_num, ins_code = int(r_str[:-1]), r_str[-1]
            except ValueError:
                continue
        else:
            try:
                r_num, ins_code = int(r_str), ' '
            except ValueError:
                continue
        key = (chain, r_num, ins_code)
        if chain in sorted_chains:
            res_map[key] = (sorted_chains.index(chain), r_num)

    # 3. Mutations as "<pos>_<chain>_<MT3>"
    formatted_variants = []
    for row_muts in mutation_batch_list:
        variant_strings = []
        for item in row_muts:
            if len(item) == 4:
                chain, _, pos, mt = item
            elif len(item) == 3:
                _, pos, mt = item
                chain = 'A'
            else:
                continue
            mt_3 = mt.upper() if len(mt) == 3 else ONE_TO_THREE.get(mt, mt)
            variant_strings.append(f"{pos}_{chain}_{mt_3}")
        formatted_variants.append(variant_strings)

    try:
        with torch.no_grad():
            new_coords, new_atnames = mutate(coords, atnames, [formatted_variants])
            ff = ForceField(device=device)
            new_info = dataStructures.create_info_tensors(new_atnames, device=device)
            energy_tensor = ff(new_coords.to(device), new_info)
        del ff
        return energy_tensor, res_map
    except Exception as e:
        print(f"   [MadraX] Calculation Error: {e}")
        return None, None


def madrax_wt_energies(coords, atnames, device):
    """WT-only MadraX energies, shape (1, chains, residues, 1, 11).
    mutate() with an empty mutant list only strips padding, so coordinates and atom names have
    exactly the format of the WT part of a mutant call (as in run_madrax)."""
    with torch.no_grad():
        wt_coords, wt_atnames = mutate(coords, atnames, [[]])
        ff = ForceField(device=device)
        info = dataStructures.create_info_tensors(wt_atnames, device=device)
        return ff(wt_coords.to(device), info).cpu()


def residue_neighbourhoods(entity, threshold=8.0):
    """{residue: residues with any atom within `threshold` A of its CA} (standard residues with a CA)."""
    all_residues = [res for res in Selection.unfold_entities(entity, 'R') if res.id[0] == " " and 'CA' in res]
    all_residues.sort(key=lambda r: (r.get_parent().id, r.id[1]))
    ns = NeighborSearch(Selection.unfold_entities(entity, 'A'))
    return {res: [r for r in ns.search(res['CA'].coord, threshold, level='R')
                  if r.id[0] == " " and r.get_resname() != "HOH" and 'CA' in r]
            for res in all_residues}


def ca_graph_edges(entity, edge_thresh=5.0):
    """
    Global CA graph: symmetric edges for CA-CA <= edge_thresh, no self loops,
    edge features [cos(CA->CB_i, CA->CB_j), CA-CA distance] (pseudo-CB for Gly).
    Residues are sorted by (chain, resnum).
    """
    def _ca_cb_unit_vector(residue):
        try:
            ca = residue["CA"].get_coord()
            if residue.get_resname() == "GLY" or not residue.has_id('CB'):
                n, c = residue["N"].get_coord(), residue["C"].get_coord()
                normal = np.cross(c - ca, n - ca)
                if np.linalg.norm(normal) < 1e-6:
                    return np.zeros(3)
                pseudo_cb = ca - 1.5 * normal / np.linalg.norm(normal)
                return (pseudo_cb - ca) / np.linalg.norm(pseudo_cb - ca)
            cb = residue["CB"].get_coord()
            return (cb - ca) / np.linalg.norm(cb - ca)
        except Exception:
            return np.zeros(3)

    valid_residues = [res for res in Selection.unfold_entities(entity, 'R') if res.id[0] == " " and 'CA' in res]
    valid_residues.sort(key=lambda r: (r.get_parent().id, r.id[1]))
    res_to_idx = {res: i for i, res in enumerate(valid_residues)}
    ca_coords = np.array([res['CA'].get_coord() for res in valid_residues])
    ca_cb_vecs = np.stack([_ca_cb_unit_vector(res) for res in valid_residues])

    edge_list, edge_features = [], []
    for i in range(len(valid_residues)):
        for j in range(i + 1, len(valid_residues)):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if dist <= edge_thresh:
                edge_list.extend([[i, j], [j, i]])
                cos_angle = np.dot(ca_cb_vecs[i], ca_cb_vecs[j])
                edge_features.extend([[cos_angle, dist], [cos_angle, dist]])
    return edge_list, edge_features, res_to_idx, {i: r for r, i in res_to_idx.items()}, ca_coords, ca_cb_vecs


def global_graph(residues, edges, node_features, edge_features, ca_coords=None):
    """Whole-protein graph with [intrinsic features, SASA] per residue."""
    node_feats = [np.append(node_features.get(res.get_resname(), node_features['ALA']), getattr(res, 'sasa', 0.0))
                  for res in residues]
    node_feats = torch.tensor(np.array(node_feats), dtype=torch.float)
    data = Data(x=node_feats, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float))
    if ca_coords is not None:
        data.ca_coords = torch.tensor(ca_coords, dtype=torch.float)
    return data


def compute_sasa(structures, n_points=960):
    """Absolute per-residue SASA (Shrake-Rupley), stored as residue.sasa."""
    from Bio.PDB import SASA
    sr = SASA.ShrakeRupley(n_points=n_points)
    for pdb_id, structure in structures.items():
        try:
            sr.compute(structure, level="R")
        except Exception as e:
            warnings.warn(f"SASA failed for {pdb_id}: {e}")
    return structures
