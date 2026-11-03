"""PyMOL renders of 2K5H for fig_positional_mapping (needs pymol-open-source). Colour = blue_white_red on min-max scaled 3-residue rolling mean (as panel c); stats use raw values.
EDL-only sigma_e per position, MegaScale singles forward; residue = MegaScale position + 34 (WT identity verified for all 62)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import REF_DIR
import pandas as pd, numpy as np
from pymol import cmd
k = pd.read_csv('data/positional_2k5h_intra.csv')
mm = lambda x: (x - x.min()) / (x.max() - x.min()) * 100
sm = lambda x: x.rolling(3, center=True, min_periods=1).mean()   # same 3-residue display smoothing as panel c
k['s100'] = mm(sm(k.sigma)); k['e100'] = mm(sm(k.mae))
cmd.load(str(REF_DIR / 'pdb_checks' / '2k5h.pdb'), 'p')
cmd.split_states('p'); cmd.delete('p'); cmd.set_name('p_0001', 'base')
for o in cmd.get_names('objects'):
    if o != 'base': cmd.delete(o)
lo, hi = int(k.resi_pdb.min()), int(k.resi_pdb.max())
cmd.remove(f'base and not resi {lo}-{hi}'); cmd.remove('solvent or hydro')
cmd.hide('everything'); cmd.show('cartoon')
for s, v in [('cartoon_fancy_helices', 1), ('cartoon_smooth_loops', 1), ('cartoon_oval_length', 1.3), ('cartoon_oval_width', 0.35),
             ('cartoon_rect_length', 1.4), ('cartoon_rect_width', 0.45), ('orthoscopic', 1), ('ray_shadows', 0), ('antialias', 2),
             ('ambient', 0.45), ('direct', 0.55), ('reflect', 0.25), ('ray_opaque_background', 0)]:
    cmd.set(s, v)
cmd.bg_color('white'); cmd.orient('base'); cmd.zoom('base', 2)
for col, name in [('s100', 'sigma'), ('e100', 'mae')]:
    cmd.alter('base', 'b=0')
    for r in k.itertuples(): cmd.alter(f'base and resi {r.resi_pdb}', f'b={getattr(r, col):.3f}')
    cmd.spectrum('b', 'blue_white_red', 'base', minimum=0, maximum=100)
    cmd.png(f'fig_positional_render_{name}.png', width=2800, height=2400, dpi=600, ray=1)
print('rendered', lo, hi)
