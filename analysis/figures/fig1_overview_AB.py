"""Fig. 1 (panels A/B; C-F were removed from the final figure) - overview (simplified): model schematic (A), how single and double mutants become graphs (B),
MegaScale-S / -D test performance (C, D), prediction and uncertainty on 2K5H (E), explanation of one prediction (F).

Inputs
  data/mega_test_uq.csv                 per-variant ensemble mu and sigma (build_uq_cache.py)
  fig1_parts/2k5h_L74.png               PyMOL render (fig1_render_structure.py; L74 in PDB numbering = sequence position 40)
  fig1_parts/2k5h_pos40_ig.npz          IG of mu and log sigma_e for the 19 L40X variants (fig1_extract_ig.py, from the
                                        explanations_v2 H5: mutation IG, ensemble mean over 8 seeds)
sigma_e = EDL epistemic uncertainty = sigma_intra column (as Fig. 3), uncalibrated (s* = 1.68 not applied).
Residue numbering = MegaScale sequence position (1-62), as in the positional figure. Forward predictions only.
Graph cartoons in A and B are schematic (redrawn from the "Graphs" slide). Writes fig1_overview.pdf/png, fig1_numbers.json.
"""
import ast, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr, t as student_t
import seaborn as sns
import paper_style as S

S.setup()
U = "sigma_intra"                       # labelled sigma_e in the paper
SITE, WIN, EX_MT = 40, (33, 46), "D"    # site in A, window in E, variant explained in F (L40D, boxed in E)
INK, MUT = "#3A3A3A", "#C0392B"
NODE_FC, NODE_EC = "#D6E4F5", "#8FAED6"  # graph node (within 9 A)
AA = list("ACDEFGHIKLMNPQRSTVWY")
T3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
          MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
SIG_CMAP = LinearSegmentedColormap.from_list("sig", ["#FFF5EB", "#FDBE85", "#F0802B", S.SIGMA, "#7F2704"])
nums = {}
t = pd.read_csv("data/mega_test_uq.csv")
IG = np.load("fig1_parts/2k5h_pos40_ig.npz", allow_pickle=True)

R3 = 1.95                               # bottom row (E, F): y 0 .. R3
R2 = R3 + 0.35; R2_H = 2.05             # middle row (B, C, D): y R2 .. R2 + R2_H
A_H = 2.50
FS = 1.25                               # font scale for panel A (2026-10-31: larger, readable schematic)
A_EXT = 0.42                            # extra band under A for the explanation path (2026-10-31)
FIG_W, FIG_H = S.FULL_W, R2 + R2_H + 0.10 + A_H + A_EXT
fig = plt.figure(figsize=(FIG_W, FIG_H))
inch = lambda x0, y0, w, h: [x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H]   # axes rectangle in inches
letter = lambda x, y, s: fig.text(x / FIG_W, y / FIG_H, s, fontsize=9, fontweight="bold", va="bottom")

# --- cartoon graphs (pixel coordinates from the "Graphs" slide; y down). in-graph = red/blue, white = outside the cutoff
CART = {
    "single": dict(c=(334, 450), red=[(312, 450)], blue=[(258, 404), (348, 404), (266, 503), (340, 513)],
                   white=[(340, 323), (403, 368), (457, 341), (430, 423), (231, 567), (376, 567)],
                   edges=[((258, 404), (312, 450)), ((348, 404), (312, 450)), ((312, 450), (266, 503)), ((312, 450), (340, 513)),
                          ((266, 503), (340, 513)), ((348, 404), (403, 368)), ((340, 323), (403, 368)), ((403, 368), (457, 341)),
                          ((403, 368), (430, 423)), ((266, 503), (231, 567)), ((340, 513), (376, 567))]),
    "joint": dict(c=(810, 446), red=[(733, 418), (878, 382)],
                  blue=[(815, 337), (932, 355), (823, 418), (905, 436), (787, 463), (742, 517)],
                  white=[(815, 527), (706, 581), (851, 581)],
                  edges=[((815, 337), (878, 382)), ((878, 382), (932, 355)), ((878, 382), (905, 436)), ((878, 382), (823, 418)),
                         ((823, 418), (787, 463)), ((733, 418), (787, 463)), ((787, 463), (742, 517)), ((742, 517), (815, 527)),
                         ((787, 463), (815, 527)), ((742, 517), (706, 581)), ((815, 527), (851, 581))]),
    "disjoint": dict(c=(1304, 446), red=[(1422, 331), (1196, 557)], blue=[(1368, 358), (1395, 412), (1232, 494)],
                     white=[(1304, 313), (1223, 395), (1313, 394), (1277, 439), (1304, 503), (1341, 557)],
                     edges=[((1304, 313), (1368, 358)), ((1368, 358), (1422, 331)), ((1368, 358), (1395, 412)),
                            ((1368, 358), (1313, 394)), ((1313, 394), (1277, 439)), ((1223, 395), (1277, 439)),
                            ((1277, 439), (1232, 494)), ((1277, 439), (1304, 503)), ((1232, 494), (1304, 503)),
                            ((1232, 494), (1196, 557)), ((1304, 503), (1341, 557))]),
}

def cartoon(axx, key, cx, cy, size, show_outside=True, show_cut=True, wt=False, node_r=None, lws=1.0):
    """Draw a cartoon graph centred at (cx, cy) in inch coordinates; size = inches per 440 slide pixels."""
    G = CART[key]; s = size / 440.0
    tr = lambda p: (cx + (p[0] - G["c"][0]) * s, cy - (p[1] - G["c"][1]) * s)
    ingraph = set(G["red"]) | set(G["blue"])
    rr = node_r or 18 * s
    if show_cut:
        for p in G["red"]:
            axx.add_patch(Circle(tr(p), 96 * s, fc="none", ec="#8A8A8A", lw=0.5 * lws, ls=(0, (2, 1.5)), zorder=1))
    for p, q in G["edges"]:
        inside = p in ingraph and q in ingraph
        if not show_outside and not inside:
            continue
        axx.plot(*zip(tr(p), tr(q)), c=INK if inside else "#C4C4C4", lw=(0.6 if inside else 0.45) * lws, zorder=2)
    if show_outside:
        for p in G["white"]:
            axx.add_patch(Circle(tr(p), rr, fc="white", ec="#9A9A9A", lw=0.5 * lws, zorder=3))
    for p in G["blue"]:
        axx.add_patch(Circle(tr(p), rr, fc=NODE_FC, ec=NODE_EC, lw=0.5 * lws, zorder=3))
    for p in G["red"]:
        axx.add_patch(Circle(tr(p), rr * 1.1, fc="white" if wt else MUT, ec=MUT, lw=0.9 * lws, zorder=4))

# =====================================================================================================
# A  schematic. One axes in inch coordinates: x 0..7.2, y 0..A_H
# =====================================================================================================
ax = fig.add_axes(inch(0, FIG_H - A_H - A_EXT, FIG_W, A_H + A_EXT)); ax.set_xlim(0, FIG_W); ax.set_ylim(-A_EXT, A_H); ax.set_aspect("equal"); ax.axis("off")
ax.text(0.0, A_H - 0.02, "A", fontsize=9 * FS, fontweight="bold", va="top")

def box(x0, y0, w, h, fc, ec, title, sub=None, ls="-", fs=6.5 * FS, sub_fs=5.5 * FS):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=0.7, ls=ls))
    if sub is None:
        ax.text(x0 + w / 2, y0 + h / 2, title, ha="center", va="center", fontsize=fs)
    else:
        ax.text(x0 + w / 2, y0 + h / 2 + 0.085, title, ha="center", va="center", fontsize=fs)
        ax.text(x0 + w / 2, y0 + h / 2 - 0.10, sub, ha="center", va="center", fontsize=sub_fs, color="#555555")

def arrow(xy0, xy1, c=INK, lw=0.7, ls="-", style="-|>", conn=None):
    ax.annotate("", xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle=style, lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0,
                                                         mutation_scale=6, connectionstyle=conn or "arc3"))

# --- structure render (auto-cropped), placed with fixed width
im = mpimg.imread("fig1_parts/2k5h_L74.png")
m = im[..., 3] > 0.05 if im.shape[-1] == 4 else im[..., :3].min(-1) < 0.98
ys, xs = np.where(m); im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
w_r = 1.42; h_r = w_r * im.shape[0] / im.shape[1]; cy = 1.42
ax.imshow(im, extent=(0.02, 0.02 + w_r, cy - h_r / 2, cy + h_r / 2), zorder=1, interpolation="lanczos")
ax.text(0.02 + w_r / 2, 2.38, "Structure (2K5H)", ha="center", fontsize=6.5 * FS)
ax.text(0.02 + w_r / 2, 2.24, f"mutated site L{SITE}, 9 Å environment", ha="center", fontsize=5.5 * FS, color="#555555")

# --- WT / MT residue graphs (cartoon: mutated residue + neighbours within 9 A)
GX = 2.02
for gy, wt, lab in [(1.97, True, "WT"), (1.10, False, "MT")]:
    cartoon(ax, "single", GX, gy, 0.95, show_outside=False, show_cut=True, wt=wt)
    ax.text(GX - 0.42, gy + 0.30, lab, fontsize=6.5 * FS, fontweight="bold", ha="left", va="center")
arrow((1.47, 1.62), (1.66, 1.88), conn="arc3,rad=-0.15"); arrow((1.47, 1.30), (1.66, 1.12), conn="arc3,rad=0.15")
ax.text(GX, 0.66, "residues ≤ 9 Å\n23 features per node", ha="center", va="top", fontsize=5 * FS, color="#555555", linespacing=1.15)

# --- 8-seed ensemble enclosure
ax.add_patch(FancyBboxPatch((3.10, 0.50), 3.02, 1.95, boxstyle="round,pad=0,rounding_size=0.08", fc="none", ec="#9A9A9A", lw=0.6, ls=(0, (3, 2))))
ax.text(6.08, 2.37, "× 8 seeds", ha="right", va="center", fontsize=5.5 * FS, color="#555555")

# --- one encoder (2026-11-06): both graphs share the WT topology; per residue, the WT features and the
#     mutation-induced change are concatenated and encoded by a single four-layer GATv2
GAT_FC, GAT_EC = "#DCEBD8", "#6E9A64"
CY = 1.54
# explicit concatenation box: per residue, WT descriptors next to the mutation-induced change (2026-11-23)
KX0, KW, KH = 2.47, 0.60, 0.92
ax.add_patch(FancyBboxPatch((KX0, CY - KH / 2), KW, KH, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=INK, lw=0.7))
ax.text(KX0 + KW / 2, CY + KH / 2 - 0.08, "Concatenate", ha="center", va="top", fontsize=5.5 * FS)
for k, (lab, fc, ec) in enumerate([(r"$x_\mathrm{WT}$", NODE_FC, NODE_EC), (r"$x_\mathrm{MT} - x_\mathrm{WT}$", "#F6D3CF", MUT)]):
    yk = CY + 0.06 - k * 0.27
    ax.add_patch(FancyBboxPatch((KX0 + 0.06, yk - 0.10), KW - 0.12, 0.20, boxstyle="round,pad=0,rounding_size=0.03", fc=fc, ec=ec, lw=0.5))
    ax.text(KX0 + KW / 2, yk, lab, ha="center", va="center", fontsize=4.8 * FS)
arrow((2.33, 1.96), (KX0, CY + 0.26), conn="arc3,rad=-0.15"); arrow((2.33, 1.12), (KX0, CY - 0.26), conn="arc3,rad=0.15")
GX0, GW = 3.20, 0.60
box(GX0, CY - 0.30, GW, 0.60, GAT_FC, GAT_EC, "GATv2", "4 layers")
arrow((KX0 + KW, CY), (GX0, CY))
ax.text(GX0 + GW / 2, CY - 0.36, "one pass on\nthe WT graph", ha="center", va="top", fontsize=5 * FS, color="#555555", linespacing=1.15)
GXR = GX0 + GW

# --- three output streams
SX, SW, SH = 3.92, 1.10, 0.44
box(SX, 1.85, SW, SH, "#E7E2F1", "#8E81B8", "Global head", "all nodes + mutated sites")
box(SX, 1.29, SW, SH, "#DDE7F4", "#6F8FBF", "Multi-mutation head", "mutated sites, ≥ 2 only", ls=(0, (3, 1.5)))
box(SX, 0.54, SW, SH, "#F4E6CC", "#B98F45", "Physics prior", "linear in Σ ΔE (MadraX)")
arrow((GXR, CY + 0.18), (SX, 2.07), conn="arc3,rad=-0.2"); arrow((GXR, 1.51), (SX, 1.51))
# MadraX bypasses the GNN: from the graphs, under the GAT
ax.annotate("", xy=(SX, 0.76), xytext=(2.32, 0.76), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#B98F45", shrinkA=0, shrinkB=0, mutation_scale=6))

# --- evidential head
HX, HW = 5.30, 0.74
box(HX, 0.54, HW, 1.75, "#F7DCD8", "#C47E72", "", None)
ax.text(HX + HW / 2, 1.83, "Evidential\nhead", ha="center", va="center", fontsize=6.5 * FS, linespacing=1.1)
ax.text(HX + HW / 2, 1.38, "μ = Σ γ", ha="center", va="center", fontsize=6 * FS)
ax.text(HX + HW / 2, 1.13, "ν, α, β", ha="center", va="center", fontsize=6 * FS)
ax.text(HX + HW / 2, 0.80, "NIG", ha="center", va="center", fontsize=5.5 * FS, color="#555555")
for y, lab in [(2.07, r"$\gamma_\mathrm{base}$"), (1.51, r"$\gamma_\mathrm{epi}$"), (0.76, r"$\gamma_\mathrm{phys}$")]:
    arrow((SX + SW, y), (HX, y)); ax.text((SX + SW + HX) / 2, y + 0.035, lab, ha="center", va="bottom", fontsize=5.5 * FS)

# --- output: predictive distribution
OX0, OX1, OYb = 6.28, 7.16, 1.05
xx = np.linspace(-3.2, 3.2, 300); yy = student_t.pdf(xx, df=5); yy = yy / yy.max()
X = OX0 + (xx + 3.2) / 6.4 * (OX1 - OX0); Y = OYb + yy * 0.62
sd = (OX1 - OX0) / 6.4                                                   # one unit of the curve
band = (xx > -1) & (xx < 1)
ax.fill_between(X[band], OYb, Y[band], color=S.SIGMA, alpha=0.28, lw=0)
ax.plot(X, Y, c=INK, lw=0.9); ax.plot([OX0, OX1], [OYb, OYb], c=INK, lw=0.5)
xm = (OX0 + OX1) / 2; ax.plot([xm, xm], [OYb, OYb + 0.62], c=INK, lw=0.6, ls="--")
ax.text(xm, OYb + 0.70, "μ  (ΔΔG)", ha="center", va="bottom", fontsize=6 * FS)
ax.annotate("", xy=(xm + sd, OYb + 0.10), xytext=(xm, OYb + 0.10), arrowprops=dict(arrowstyle="<->", lw=0.6, color=S.SIGMA, shrinkA=0, shrinkB=0, mutation_scale=5))
ax.text(xm + sd / 2, OYb + 0.15, r"$\sigma_e$", ha="center", va="bottom", fontsize=6.5 * FS, color="#9A4100")
ax.text(xm, OYb - 0.07, r"$\sigma_e^2 = \langle \beta / \nu(\alpha-1) \rangle_\mathrm{seeds}$", ha="center", va="top", fontsize=5.5 * FS)
ax.text(xm, 2.38, "Prediction", ha="center", fontsize=6.5 * FS)
arrow((HX + HW, 1.45), (OX0 - 0.04, 1.45))

# --- explanation path (integrated gradients), back to the inputs, with schematic glyphs of what it returns (no data)
IGC = "#6B6B6B"; yb = -0.10
ax.plot([xm, xm], [OYb - 0.30, yb], c=IGC, lw=0.8, ls=(0, (3, 2)))
ax.plot([xm, 0.73], [yb, yb], c=IGC, lw=0.8, ls=(0, (3, 2)))
ax.annotate("", xy=(0.73, cy - h_r / 2 + 0.06), xytext=(0.73, yb), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=IGC, ls=(0, (3, 2)), shrinkA=0, shrinkB=0, mutation_scale=6))
ax.text(2.55, yb + 0.07, "Integrated gradients:\nμ and log $\\sigma_e$ attributed to residues and features",
        ha="center", va="bottom", fontsize=6 * FS, color="#444444", linespacing=1.25, bbox=dict(fc="white", ec="none", pad=0.6))
# glyph 1: residues coloured by attribution
gx = 4.75
ax.add_patch(FancyBboxPatch((gx - 0.36, yb - 0.27), 0.72, 0.54, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec="#BDBDBD", lw=0.6, zorder=4))
G = CART["single"]; sc = 0.78 / 440.0; trg = lambda p: (gx + (p[0] - G["c"][0]) * sc, yb + 0.06 - (p[1] - G["c"][1]) * sc)
for p, q in G["edges"]:
    if p in G["blue"] + G["red"] and q in G["blue"] + G["red"]: ax.plot(*zip(trg(p), trg(q)), c=INK, lw=0.5, zorder=5)
for p, c in zip(G["blue"], ["#F4A582", "#92C5DE", "#D6604D", "#FDDBC7"]):
    ax.add_patch(Circle(trg(p), 20 * sc, fc=c, ec="#777777", lw=0.4, zorder=6))
ax.add_patch(Circle(trg(G["red"][0]), 22 * sc, fc="#B2182B", ec="#777777", lw=0.4, zorder=6))
ax.add_patch(Circle(trg(G["red"][0]), 96 * sc, fc="none", ec="#8A8A8A", lw=0.5, ls=(0, (2, 1.5)), zorder=5))
ax.text(gx, yb - 0.24, "per residue", ha="center", va="bottom", fontsize=4.6 * FS, color="#555555", zorder=6)
# glyph 2: per-feature bars (schematic, unlabelled)
fx = 5.70
ax.add_patch(FancyBboxPatch((fx - 0.36, yb - 0.27), 0.72, 0.54, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec="#BDBDBD", lw=0.6, zorder=4))
for k, v in enumerate([0.26, 0.18, 0.12, -0.10, 0.07]):
    y = yb + 0.16 - k * 0.065
    ax.add_patch(Rectangle((fx, y - 0.025), v, 0.05, fc="#B2182B" if v > 0 else "#2166AC", lw=0, zorder=5))
ax.plot([fx, fx], [yb - 0.12, yb + 0.20], c=INK, lw=0.5, zorder=6)
ax.text(fx, yb - 0.24, "per feature", ha="center", va="bottom", fontsize=4.6 * FS, color="#555555", zorder=6)

# =====================================================================================================
# B  how variants become graphs (cartoons)
# =====================================================================================================
# Row 2 (2026-10-31): B (how variants become graphs), centred under A
BW, GAP, BH = 1.40, 0.22, 2.25                      # box size, gap between boxes, row height (inches)
BTOP = R2 + R2_H                                    # top of the row, directly under A
bx = fig.add_axes(inch(0, BTOP - BH, FIG_W, BH)); bx.set_xlim(0, FIG_W); bx.set_ylim(0, BH); bx.set_aspect("equal"); bx.axis("off")
letter(0.0, BTOP - 0.12, "B")
X0 = (FIG_W - 3 * BW - 2 * GAP) / 2
for i, (key, title, sub) in enumerate([("single", "Single mutant", "9 Å around the site"),
                                       ("joint", "Double, one graph", "neighbourhoods overlap"),
                                       ("disjoint", "Double, two subgraphs", "neighbourhoods separate")]):
    x0 = X0 + i * (BW + GAP); y0 = 0.38
    bx.add_patch(FancyBboxPatch((x0, y0), BW, BW, boxstyle="round,pad=0,rounding_size=0.09", fc="white", ec="#BDBDBD", lw=0.65))
    cartoon(bx, key, x0 + BW / 2, y0 + BW / 2, BW * 0.97, lws=1.3)
    bx.text(x0 + BW / 2, y0 + BW + 0.22, title, ha="center", va="bottom", fontsize=7.5)
    bx.text(x0 + BW / 2, y0 + BW + 0.08, sub, ha="center", va="bottom", fontsize=6.3, color="#555555")
ly = 0.15
for k, (fc, ec, lab) in enumerate([(MUT, MUT, "mutated residue"), (NODE_FC, NODE_EC, "graph node"), ("white", "#9A9A9A", "outside the cutoff")]):
    lx = X0 + 0.18 + k * (BW + GAP)
    bx.add_patch(Circle((lx, ly), 0.05, fc=fc, ec=ec, lw=0.65)); bx.text(lx + 0.10, ly, lab, va="center", fontsize=6.3)

S.save(fig, "fig1_overview_AB")
