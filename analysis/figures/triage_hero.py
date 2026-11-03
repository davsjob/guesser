#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import PRED
import glob, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load data
base_path = PRED
files = sorted(glob.glob(os.path.join(base_path, 'mega_test_predictions_S*.csv')))
dfs = [pd.read_csv(f) for f in files]
dfs_fwd = [d[d['direction'] == 'Forward'].reset_index(drop=True) for d in dfs]

df = dfs_fwd[0][['pdb', 'mutation', 'true_ddg', 'num_muts']].copy()
pred_mat = np.column_stack([d['pred_ddg'].values for d in dfs_fwd])
v_mat = np.maximum(np.column_stack([d['v'].values for d in dfs_fwd]), 1e-6)
alpha_mat = np.maximum(np.column_stack([d['alpha'].values for d in dfs_fwd]), 1.0 + 1e-6)
beta_mat = np.maximum(np.column_stack([d['beta'].values for d in dfs_fwd]), 1e-6)

df['pred_ddg'] = pred_mat.mean(axis=1)
tot_ev = 2.0 * v_mat + alpha_mat
df['inv_phi'] = np.mean(1.0 / np.maximum(tot_ev, 1e-7), axis=1)
var_e = np.mean(beta_mat / (v_mat * (alpha_mat - 1.0)), axis=1)
df['sigma_edl_epi'] = np.sqrt(np.maximum(var_e, 1e-8))

df_s = df[df['num_muts'] == 1].copy()
df_d = df[df['num_muts'] >= 2].copy()

cand_s = df_s[df_s['pred_ddg'] <= -0.5].copy()
cand_d = df_d[df_d['pred_ddg'] <= -0.5].copy()

cand_s['is_tp'] = (cand_s['true_ddg'] <= -0.5).astype(int)
cand_d['is_tp'] = (cand_d['true_ddg'] <= -0.5).astype(int)

K_vals = np.array([5, 10, 15, 20, 30, 50])

def calc_curve(df_c, col, asc=True):
    s = df_c.sort_values(col, ascending=asc)
    return np.array([s.iloc[:k]['is_tp'].mean() * 100 for k in K_vals])

detpr_s_naive   = calc_curve(cand_s, 'pred_ddg', True)
detpr_s_sigma_e = calc_curve(cand_s, 'sigma_edl_epi', True)
detpr_s_inv_phi = calc_curve(cand_s, 'inv_phi', True)

detpr_d_naive   = calc_curve(cand_d, 'pred_ddg', True)
detpr_d_sigma_e = calc_curve(cand_d, 'sigma_edl_epi', True)
detpr_d_inv_phi = calc_curve(cand_d, 'inv_phi', True)

baseline_s = cand_s['is_tp'].mean() * 100.0
baseline_d = cand_d['is_tp'].mean() * 100.0

# 2. Style configuration (Nature compliant, clean)
plt.style.use('seaborn-v0_8-ticks' if 'seaborn-v0_8-ticks' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.0,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.0,
    'axes.linewidth': 0.7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})

def add_nature_panel_label(ax, label, x=-0.13, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={'wspace': 0.28})

c_navy = '#2B5C8F'
c_teal = '#009688'
c_gray = '#7F8C8D'
c_base = '#A0AAB2'

# Panel A: Singles
add_nature_panel_label(ax_a, 'a')
ax_a.axhline(baseline_s, color=c_base, linestyle='--', lw=1.0, zorder=1, label=f'Baseline ({baseline_s:.1f}%)')
ax_a.plot(K_vals, detpr_s_naive, 's--', color=c_gray, lw=1.3, ms=4.5, markeredgecolor='white', markeredgewidth=0.7, zorder=2, label=r'Naïve $\hat{\mu}$')
ax_a.plot(K_vals, detpr_s_sigma_e, 'o-', color=c_navy, lw=1.5, ms=5.0, markeredgecolor='white', markeredgewidth=0.7, zorder=3, label=r'$\sigma_e$')
ax_a.plot(K_vals, detpr_s_inv_phi, '^-', color=c_teal, lw=1.5, ms=5.2, markeredgecolor='white', markeredgewidth=0.7, zorder=4, label=r'$1/\Phi$')

ax_a.set_xlabel(r'Top $N$ Candidates Retained')
ax_a.set_ylabel('PPV (%)')
ax_a.set_title(f'MegaScale Singles\n($N_{{\\mathrm{{cand}}}}={len(cand_s)}$, $N_{{\\mathrm{{TP}}}}={int(cand_s["is_tp"].sum())}$)', pad=6)
ax_a.set_xticks(K_vals)
ax_a.set_ylim(-2, 105)
ax_a.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#E0E0E0')
ax_a.grid(axis='y', linestyle=':', alpha=0.4, lw=0.5)

# Panel B: Doubles
add_nature_panel_label(ax_b, 'b')
ax_b.axhline(baseline_d, color=c_base, linestyle='--', lw=1.0, zorder=1, label=f'Baseline ({baseline_d:.1f}%)')
ax_b.plot(K_vals, detpr_d_naive, 's--', color=c_gray, lw=1.3, ms=4.5, markeredgecolor='white', markeredgewidth=0.7, zorder=2, label=r'Naïve $\hat{\mu}$')
ax_b.plot(K_vals, detpr_d_sigma_e, 'o-', color=c_navy, lw=1.5, ms=5.0, markeredgecolor='white', markeredgewidth=0.7, zorder=3, label=r'$\sigma_e$')
ax_b.plot(K_vals, detpr_d_inv_phi, '^-', color=c_teal, lw=1.5, ms=5.2, markeredgecolor='white', markeredgewidth=0.7, zorder=4, label=r'$1/\Phi$')

ax_b.set_xlabel(r'Top $N$ Candidates Retained')
ax_b.set_ylabel('PPV (%)')
ax_b.set_title(f'MegaScale Doubles\n($N_{{\\mathrm{{cand}}}}={len(cand_d)}$, $N_{{\\mathrm{{TP}}}}={int(cand_d["is_tp"].sum())}$)', pad=6)
ax_b.set_xticks(K_vals)
ax_b.set_ylim(-2, 105)
ax_b.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='#E0E0E0')
ax_b.grid(axis='y', linestyle=':', alpha=0.4, lw=0.5)

fig.tight_layout()

# Save
out_png = 'figure_3_PPV_improvement.png'
out_pdf = 'figure_3_PPV_improvement.pdf'
fig.savefig(out_png)
fig.savefig(out_pdf)
plt.close(fig)
print('✓ Successfully generated clean Figure 7!')
