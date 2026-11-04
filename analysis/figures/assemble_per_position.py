"""Assemble figure_per_position.png from positional_parts/ (a, c) and the PyMOL renders (b), in the manuscript's layout
(previously assembled in PowerPoint). Run from results/figures/."""
import numpy as np, matplotlib.pyplot as plt, matplotlib.image as mpimg
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
def crop(im):
    a = im[..., 3] if im.shape[-1] == 4 else 1 - im[..., :3].min(-1)
    ys, xs = np.where(a > 0.02); return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
A = mpimg.imread("positional_parts/parts_a_per_protein.png"); C = mpimg.imread("positional_parts/parts_c_trace.png")
S = crop(mpimg.imread("fig_positional_render_sigma.png")); M = crop(mpimg.imread("fig_positional_render_mae.png"))
CB = mpimg.imread("positional_parts/parts_b_colorbar.png")
fig = plt.figure(figsize=(13.33, 7.5))
def place(im, rect):
    ax = fig.add_axes(rect); ax.imshow(im); ax.axis("off"); return ax
place(A, [0.01, 0.10, 0.44, 0.80])
place(S, [0.48, 0.55, 0.22, 0.37]); place(M, [0.71, 0.55, 0.22, 0.37]); place(CB, [0.935, 0.58, 0.05, 0.30])
place(C, [0.53, 0.08, 0.43, 0.42])
fig.text(0.59, 0.52, r"Epistemic Uncertainty ($\sigma_e$)", ha="center", fontsize=12)
fig.text(0.82, 0.52, "Prediction Error (MAE)", ha="center", fontsize=12)
for x, y, l in [(0.03, 0.92, "a"), (0.47, 0.92, "b"), (0.47, 0.47, "c")]:
    fig.text(x, y, l, fontsize=16, fontweight="bold", va="top", fontfamily="Arial")
for ext in ("png", "pdf"): fig.savefig(f"figure_per_position.{ext}", dpi=300, facecolor="white")
