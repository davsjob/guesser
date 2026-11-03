"""Fig. 1A structure render (needs pymol-open-source). 2K5H model 1, MegaScale construct (resi 35-96).
Mutated site L74 = sequence position 40 (PDB resi = position + 34) in red sticks. Graph nodes = the model's own
graph for the L40X variants (27 residues, from fig1_parts/2k5h_pos40_ig.npz, written by fig1_extract_ig.py).
Writes fig1_parts/2k5h_L74.png and fig1_parts/2k5h_L74_nodes.csv (node CA coordinates projected with the same view,
so the graph drawings in fig1_overview.py have the same orientation as the render)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import REF_DIR
import numpy as np, pandas as pd
from pymol import cmd
SITE, OFFSET = 74, 34
IG = np.load("fig1_parts/2k5h_pos40_ig.npz", allow_pickle=True)
cmd.load(str(REF_DIR / 'pdb_checks' / '2k5h.pdb'), 'p')
cmd.split_states('p'); cmd.delete('p'); cmd.set_name('p_0001', 'base')
for o in cmd.get_names('objects'):
    if o != 'base': cmd.delete(o)
cmd.remove('base and not resi 35-96'); cmd.remove('solvent or hydro')
ca = {}
cmd.iterate_state(1, 'base and name CA', 'ca[int(resi)] = (resn, (x, y, z))', space={'ca': ca})
c0 = np.array(ca[SITE][1])
nodes = [int(r) + OFFSET for r in IG["v0_res_id"]]                       # model graph order (local node index)
sel_nodes = 'base and resi ' + '+'.join(map(str, nodes))
cmd.hide('everything'); cmd.show('cartoon')
for s, v in [('cartoon_fancy_helices', 1), ('cartoon_smooth_loops', 1), ('cartoon_oval_length', 1.3), ('cartoon_oval_width', 0.35),
             ('cartoon_rect_length', 1.4), ('cartoon_rect_width', 0.45), ('orthoscopic', 1), ('ray_shadows', 0), ('antialias', 2),
             ('ambient', 0.45), ('direct', 0.55), ('reflect', 0.2), ('ray_opaque_background', 0), ('cartoon_transparency', 0.0),
             ('stick_radius', 0.36), ('sphere_scale', 0.42), ('sphere_transparency', 0.0)]:
    cmd.set(s, v)
cmd.set_color('cgrey', [0.80, 0.80, 0.80]); cmd.set_color('cnode', [0.17, 0.36, 0.56]); cmd.set_color('cmut', [0.75, 0.10, 0.12])
cmd.set_color('cnode_l', [0.62, 0.74, 0.86])
cmd.color('cgrey', 'base'); cmd.color('cnode_l', sel_nodes)
cmd.show('spheres', f'({sel_nodes}) and name CA and not resi {SITE}'); cmd.color('cnode', f'({sel_nodes}) and name CA')
cmd.show('sticks', f'base and resi {SITE} and not name N+C+O'); cmd.color('cmut', f'base and resi {SITE}')
cmd.set('cartoon_side_chain_helper', 1)
cmd.bg_color('white'); cmd.set('specular', 0.15); cmd.orient('base')
cmd.turn('y', -35); cmd.turn('x', 15)
cmd.zoom('base', 1)
cmd.png('fig1_parts/2k5h_L74.png', width=2000, height=1700, dpi=600, ray=1)
v = cmd.get_view(); R = np.array(v[:9]).reshape(3, 3); org = np.array(v[12:15])
xyz = np.array([ca[r][1] for r in nodes]); cam = (xyz - org) @ R
pd.DataFrame(dict(resi=nodes, pos=[r - OFFSET for r in nodes], resn=[ca[r][0] for r in nodes], x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                  sx=cam[:, 0], sy=cam[:, 1], sz=cam[:, 2])).to_csv('fig1_parts/2k5h_L74_nodes.csv', index=False)
print('nodes', len(nodes), nodes)
