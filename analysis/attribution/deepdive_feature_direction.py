"""Does the DIRECTION of a feature change matter, and where? (MegaScale-S singles, mutation IG on mu)

    python deepdive_feature_direction.py
Outputs -> <OUT_DIR>/deepdive_feature_direction/  (response_curves.csv, direction_summary.csv, key_numbers.json, fig17)

Collapses "mutation class x feature x sign x burial" into one response curve per feature:
x = change in the feature (mt - wt, summed over the graph; raw units), y = that feature's attribution
(kcal/mol, + = destabilizing), one line per burial class. The curve shape is the answer:
  monotonic through 0  -> directional (loss and gain push opposite ways)
  V / U shape          -> any change costs (e.g. packing)
  flat on one side     -> only one direction matters
Feature units: Hydrophobicity = Kyte-Doolittle, Volume = A^3, Charge = e, Cyclic = Pro flag;
MadraX terms = kcal/mol-like energies where + = less favourable.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import figures  # noqa: F401  (shared rcParams)

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_feature_direction"
OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["Hydrophobicity", "Volume", "VdW Clashes", "Polar Solvation", "Hydrophobic Solvation", "VdW Energy",
         "Electrostatics", "Charge", "Hydrogen Bonds", "Cyclic", "Backbone Entropy"]
UNITS = {"Hydrophobicity": "Δ hydrophobicity (KD)", "Volume": "Δ volume (Å³)", "Charge": "Δ charge (e)",
         "Cyclic": "Δ Pro flag", "Hydrogen Bonds": "Δ H-bond energy", "VdW Clashes": "Δ clash energy",
         "Polar Solvation": "Δ polar solvation energy", "Hydrophobic Solvation": "Δ hydrophobic solvation energy",
         "VdW Energy": "Δ vdW energy", "Electrostatics": "Δ electrostatic energy",
         "Backbone Entropy": "Δ backbone entropy"}
BUR = ["core", "boundary", "surface"]
BCOL = {"core": C["red"], "boundary": C["yellow"], "surface": C["blue"]}
EPS = {f: (1e-6 if i < 12 else 0.05) for i, f in enumerate(F)}
NBIN, MIN_N = 5, 30
K = {}


def bins_for(x, eps):
    u = np.unique(np.round(x, 6))
    if len(u) <= 9:
        return None  # discrete: use values
    neg, pos = x[x < -eps], x[x > eps]
    en = np.unique(np.quantile(neg, np.linspace(0, 1, NBIN + 1))) if len(neg) > NBIN * MIN_N else np.array([])
    ep = np.unique(np.quantile(pos, np.linspace(0, 1, NBIN + 1))) if len(pos) > NBIN * MIN_N else np.array([])
    return en, ep


def assign(x, eps, b):
    if b is None:
        return np.round(x, 3).astype(str)
    en, ep = b
    lab = np.full(len(x), "0", dtype=object)
    if len(en):
        k = np.clip(np.searchsorted(en, x, side="right") - 1, 0, len(en) - 2)
        lab[x < -eps] = [f"n{j}" for j in k[x < -eps]]
    if len(ep):
        k = np.clip(np.searchsorted(ep, x, side="right") - 1, 0, len(ep) - 2)
        lab[x > eps] = [f"p{j}" for j in k[x > eps]]
    return lab


def main():
    v = pd.read_csv(cfg.CACHE_DIR / DS / "variants.csv")
    m = (v.n_mutated == 1).values
    s = v[m].reset_index(drop=True)
    A = np.load(cfg.CACHE_DIR / DS / "attr_mut_mu.npz")["F_tot"][m]
    DX = np.load(cfg.CACHE_DIR / DS / "geom.npz")["dx_all"][m]
    curves, summ = [], []
    for f in FEATS:
        j = F.index(f)
        x, a, eps = DX[:, j], A[:, j], EPS[f]
        lab = assign(x, eps, bins_for(x, eps))
        df = pd.DataFrame(dict(x=x, a=a, lab=lab, bur=s.rsa_class, true=s.true_ddg, pred=s.pred_mu))
        g = df.groupby(["bur", "lab"]).agg(n=("a", "size"), dx=("x", "mean"), attr=("a", "mean"),
                                           attr_sd=("a", "std"), true=("true", "mean"), pred=("pred", "mean"))
        g = g[g.n >= MIN_N].reset_index()
        g["feature"] = f
        curves.append(g)
        df["side"] = np.select([df.x < -eps, df.x > eps], ["decrease", "increase"], "no change")
        for b in BUR + ["all"]:
            d = df if b == "all" else df[df.bur == b]
            row = dict(feature=f, burial=b, n=len(d))
            for side in ("decrease", "no change", "increase"):
                q = d[d.side == side]
                row[f"n_{side}"] = len(q)
                row[f"attr_{side}"] = q.a.mean() if len(q) >= MIN_N else np.nan
                row[f"frac_pos_{side}"] = (q.a > 0).mean() if len(q) >= MIN_N else np.nan
            summ.append(row)
    cur = pd.concat(curves, ignore_index=True)
    sm = pd.DataFrame(summ)
    cur.to_csv(OUT / "response_curves.csv", index=False, float_format="%.4g")
    sm.to_csv(OUT / "direction_summary.csv", index=False, float_format="%.4g")

    def shape(r):
        dn, up = r.attr_decrease, r.attr_increase
        if np.isnan(dn) or np.isnan(up):
            return "one-sided"
        if dn > 0.05 and up > 0.05:
            return "any change costs"
        if (dn > 0.05 and up < -0.02) or (dn < -0.02 and up > 0.05):
            return "directional"
        return "weak"
    sm["shape"] = [shape(r) for r in sm.itertuples()]
    K["summary"] = {f"{r.feature} | {r.burial}": dict(
        loss=None if np.isnan(r.attr_decrease) else round(r.attr_decrease, 3),
        gain=None if np.isnan(r.attr_increase) else round(r.attr_increase, 3),
        n_loss=int(r.n_decrease), n_gain=int(r.n_increase), shape=r.shape) for r in sm.itertuples()}
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, ensure_ascii=False)
    sm.to_csv(OUT / "direction_summary.csv", index=False, float_format="%.4g")
    figure(cur)
    pd.set_option("display.width", 200)
    print(sm[["feature", "burial", "n_decrease", "attr_decrease", "n_no change", "attr_no change", "n_increase",
              "attr_increase", "shape"]].round(3).to_string(index=False))


def figure(cur):
    feats = ["Hydrophobicity", "Volume", "Polar Solvation", "VdW Clashes", "Hydrophobic Solvation", "VdW Energy",
             "Electrostatics", "Cyclic"]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.8), gridspec_kw=dict(wspace=0.38, hspace=0.62))
    for ax, f in zip(axes.flat, feats):
        c = cur[cur.feature == f]
        ax.axhline(0, color=C["ink2"], lw=0.6)
        ax.axvline(0, color=C["neutral"], lw=0.6, ls=":")
        for b in BUR:
            q = c[c.bur == b].sort_values("dx")
            ax.plot(q.dx, q.attr, "-o", color=BCOL[b], lw=1.6, ms=3.5, mec="white", mew=0.5, label=b)
        ax.set_title(f, loc="left", fontsize=8.5)
        ax.set_xlabel(UNITS[f], fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("attribution (kcal/mol)\n+ destabilizing")
    axes[0, 0].legend(fontsize=6.8, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig17_feature_direction.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
