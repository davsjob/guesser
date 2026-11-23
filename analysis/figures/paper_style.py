"""Shared style for all manuscript figures (v2). Import this in every figure script."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette (fixed across the paper) ----
SINGLES = "#2B5C8F"   # deep blue (as Fig. 1)
DOUBLES = "#7B4397"   # plum (as Fig. 1)
CORE, BOUNDARY, SURFACE = "#B2182B", "#E08E0B", "#1B8A8A"
MU, SIGMA = "#3A3A3A", "#D95F02"
GREY, LIGHT = "#8C8C8C", "#D9D9D9"

FULL_W = 7.2   # in, full page width (~18.3 cm)
HALF_W = 4.7   # in, two-column width (~12 cm)

def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        
    })

def panel(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
