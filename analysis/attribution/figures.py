"""Stage 3: figures from the analysis tables (PNG 300 dpi + PDF).

    python figures.py --all
Per-dataset figures go to <OUT_DIR>/<dataset>/figures, the cross-dataset figure to <OUT_DIR>/cross_dataset.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

C = cfg.C
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "text.color": C["ink"], "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def _save(fig, d, name):
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)


def _tab(ds, name):
    p = cfg.OUT_DIR / ds / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


# ---------------------------------------------------------------------------- 1. driving forces
def fig_driving(ds, fd):
    subs = [s for s in ("singles", "doubles") if _tab(ds, f"driving_{s}") is not None]
    ref = _tab(ds, f"driving_{subs[0]}")
    order = ref.head(10).feature.tolist()[::-1]
    fig, axes = plt.subplots(1, len(subs), figsize=(3.4 * len(subs), 3.4), sharey=True, squeeze=False)
    for ax, sub, lab in zip(axes[0], subs, "ab"):
        t = _tab(ds, f"driving_{sub}").set_index("feature").loc[order]
        y = np.arange(len(order))
        ax.axvline(0, color=C["ink2"], lw=0.6)
        ax.hlines(y + 0.14, t.shift_true_lo, t.shift_true_hi, color=C["blue"], lw=1.2)
        ax.plot(t.shift_true, y + 0.14, "o", ms=5, color=C["blue"], mec="white", mew=0.8,
                label="binned by experimental ΔΔG")
        ax.hlines(y - 0.14, t.shift_pred_lo, t.shift_pred_hi, color=C["neutral"], lw=1.2)
        ax.plot(t.shift_pred, y - 0.14, "o", ms=5, mfc="white", mec=C["ink2"], mew=0.9,
                label="binned by predicted μ")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        n = int(t.n_stab_true.iloc[0]), int(t.n_destab_true.iloc[0])
        ax.set_title(f"{cfg.DATASET_LABEL[ds]} {sub}\nstabilizing n={n[0]:,} · destabilizing n={n[1]:,}", loc="left")
        ax.grid(axis="y", visible=False)
        _panel_label(ax, lab)
    axes[0][0].legend(loc="lower right")
    fig.supxlabel("Attribution shift, destabilizing − stabilizing (kcal/mol)", fontsize=8, y=0.02)
    fig.subplots_adjust(wspace=0.28)
    fig.text(0.01, -0.05, "Mutation IG (sums to μ − μ_null). Error bars: 95% protein-cluster bootstrap. "
                          "Tiers: stabilizing ΔΔG<0, destabilizing ΔΔG>1 kcal/mol.", fontsize=6.5, color=C["ink2"])
    _save(fig, fd, "fig1_driving_forces")


# ---------------------------------------------------------------------------- 2. burial dependence
def fig_burial(ds, fd):
    subs = [s for s in ("singles", "doubles") if _tab(ds, f"burial_{s}") is not None]
    ref = _tab(ds, f"driving_{subs[0]}")
    feats = ref.head(6).feature.tolist()[::-1]
    cls_col = {"core": C["blue"], "boundary": C["aqua"], "surface": C["orange"]}
    fig, axes = plt.subplots(1, len(subs), figsize=(3.4 * len(subs), 3.0), sharey=True, squeeze=False)
    for ax, sub, lab in zip(axes[0], subs, "ab"):
        t = _tab(ds, f"burial_{sub}")
        y = np.arange(len(feats))
        ax.axvline(0, color=C["ink2"], lw=0.6)
        for k, (cls, col) in enumerate(cls_col.items()):
            q = t[t.rsa_class == cls].set_index("feature")
            if q.empty or q.shift_true.isna().all():
                continue
            q = q.loc[feats]
            off = (k - 1) * 0.22
            ax.hlines(y + off, q.shift_true_lo, q.shift_true_hi, color=col, lw=1.1)
            ax.plot(q.shift_true, y + off, "o", ms=4.5, color=col, mec="white", mew=0.7,
                    label=f"{cls} (n={int(q.n.iloc[0]):,})")
        ax.set_yticks(y)
        ax.set_yticklabels(feats)
        ax.set_title(f"{sub}: burial dependence (relative SASA)", loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        _panel_label(ax, lab)
    fig.supxlabel("Attribution shift, destabilizing − stabilizing, by experimental ΔΔG tier (kcal/mol)", fontsize=8, y=0.0)
    fig.subplots_adjust(wspace=0.28)
    fig.text(0.01, -0.07, "Points without error bars: too few stabilizing variants in that stratum for a stable protein bootstrap.",
             fontsize=6.5, color=C["ink2"])
    _save(fig, fd, "fig2_burial_dependence")


# ---------------------------------------------------------------------------- 3. volume
def fig_volume(ds, fd):
    t = _tab(ds, "volume_bidirectional_singles")
    if t is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    x = np.arange(len(t))
    short = ["< −40", "−40 to −10", "−10 to 10", "10 to 40", "> 40"]
    ax = axes[0]
    ax.bar(x, t.mean_volume_attr, width=0.62, color=C["blue"])
    ax.set_ylabel("Mean Volume attribution (kcal/mol)")
    ax.set_title("Model: Volume contribution", loc="left")
    ax = axes[1]
    ax.plot(x, t.mean_true_ddg, "-o", color=C["ink"], ms=5, lw=2, label="experimental ΔΔG")
    ax.plot(x, t.mean_pred_mu, "-o", color=C["orange"], ms=5, lw=2, label="predicted μ")
    ax.set_ylabel("Mean ΔΔG (kcal/mol)")
    ax.set_title("Data vs prediction", loc="left")
    ax.legend(loc="upper center")
    for ax, lab in zip(axes, "ab"):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}\nn={n:,}" for a, n in zip(short[:len(t)], t.n)], fontsize=6.5)
        ax.set_xlabel("Side-chain volume change ΔV (Å³)")
        ax.grid(axis="x", visible=False)
        _panel_label(ax, lab)
    fig.subplots_adjust(wspace=0.3)
    _save(fig, fd, "fig3_volume_bidirectional")


# ---------------------------------------------------------------------------- 4. environment + localisation
def fig_environment(ds, fd):
    subs = [s for s in ("singles", "doubles") if _tab(ds, f"environment_{s}") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    ref = _tab(ds, f"environment_{subs[0]}")
    feats = ref.head(8).feature.tolist()[::-1]
    y = np.arange(len(feats))
    h = 0.8 / len(subs)
    for k, sub in enumerate(subs):
        t = _tab(ds, f"environment_{sub}").set_index("feature").loc[feats]
        ax.barh(y + (k - (len(subs) - 1) / 2) * h, t.mean_abs, height=h * 0.9, color=cfg.SERIES[k], label=sub)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Mean |environment attribution| (kcal/mol)")
    ax.set_title("Environment IG: which WT features matter", loc="left")
    ax.grid(axis="y", visible=False)
    if len(subs) > 1:
        ax.legend(loc="lower right")
    _panel_label(ax, "a")
    ax = axes[1]
    loc = _tab(ds, f"localisation_{subs[0]}")
    shells = [s[0] for s in cfg.SHELLS]
    for k, (fam, lab) in enumerate([("ctx_mu", "environment IG on μ"), ("ctx_epistemic", "environment IG on log σ_epi")]):
        q = loc[loc.family == fam].set_index("shell").loc[shells]
        ax.plot(range(len(shells)), q.per_node_share, "-o", color=cfg.SERIES[k], lw=2, ms=5, label=lab)
    ax.set_xticks(range(len(shells)))
    ax.set_xticklabels(["mutated\nresidue", "0–6 Å", "6–9 Å", ">9 Å"])
    ax.set_xlabel("Cα distance to mutated residue")
    ax.set_ylabel("Share of |attribution| per residue")
    ax.set_title(f"Per-residue attribution decays with distance ({subs[0]})", loc="left")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    _panel_label(ax, "b")
    _save(fig, fd, "fig4_environment_localisation")


# ---------------------------------------------------------------------------- 5. epistasis
def fig_epistasis(ds, fd):
    t = _tab(ds, "epistasis")
    p = cfg.OUT_DIR / ds / "tables" / "epistasis_per_variant.csv"
    if t is None or not p.exists():
        return
    pv = pd.read_csv(p).dropna(subset=["E_exp", "E_pred"])
    if len(pv) < 50:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw=dict(wspace=0.55))
    ax = axes[0]
    hb = ax.hexbin(pv.E_exp, pv.E_pred, gridsize=45, cmap=matplotlib.colors.ListedColormap(cfg.SEQ_BLUE),
                   mincnt=1, bins="log", linewidths=0.2, edgecolors="white", extent=(-6, 3, -4, 3))
    ax.axhline(0, color=C["ink2"], lw=0.5)
    ax.axvline(0, color=C["ink2"], lw=0.5)
    ax.plot([-4, 3], [-4, 3], color=C["ink2"], lw=0.8, ls="--")
    ax.set_xlabel("Experimental non-additivity E_exp (kcal/mol)")
    ax.set_ylabel("Model non-additivity E_pred (kcal/mol)")
    r = t[t.group == "all"].iloc[0]
    ax.set_title(f"n={int(r.n_matched):,}, Spearman ρ={r.rho_E_pred_vs_E_exp:.2f}", loc="left")
    fig.colorbar(hb, ax=ax, label="variants (log)", shrink=0.8)
    _panel_label(ax, "a")
    ax = axes[1]
    g = t[t.group.str.startswith("d12")].set_index("group")
    x = np.arange(len(g))
    w = 0.26
    series = [("rho_E_pred_vs_E_exp", "E_pred vs E_exp", C["blue"]),
              ("partial_rho_E_pred_vs_E_exp_given_add", "E_pred vs E_exp | additive sum", C["aqua"]),
              ("partial_rho_gamma_epi_vs_E_exp_given_add", "γ_epi vs E_exp | additive sum", C["orange"])]
    for k, (col, lab, c) in enumerate(series):
        ax.bar(x + (k - 1) * w, g[col], width=w * 0.92, color=c, label=lab)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i.replace('d12 ', '')}\n(n={int(n):,})" for i, n in zip(g.index, g.n_matched)])
    ax.set_xlabel("Cα–Cα distance between the two sites")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Non-additivity tracking by site distance", loc="left")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    _panel_label(ax, "b")
    _save(fig, fd, "fig5_epistasis")


# ---------------------------------------------------------------------------- 6. uncertainty
def fig_uncertainty(ds, fd):
    v = pd.read_csv(cfg.CACHE_DIR / ds / "variants.csv")
    subs = [(s, v[v.n_mutated == k]) for s, k in (("singles", 1), ("doubles", 2)) if (v.n_mutated == k).any()]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(wspace=0.55))
    ax = axes[0]
    for k, (sub, s) in enumerate(subs):
        q = pd.qcut(s.epistemic_std, 10, labels=False)
        m = s.groupby(q).abs_err.mean()
        ax.plot(np.arange(1, 11), m.values, "-o", color=cfg.SERIES[k], lw=2, ms=4.5, label=sub)
    ax.set_xlabel("Epistemic σ decile (1 = most confident)")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Does uncertainty rank errors?", loc="left")
    ax.set_xticks(range(1, 11))
    ax.legend(loc="upper left")
    _panel_label(ax, "a")
    ax = axes[1]
    t = _tab(ds, f"uncertainty_attribution_{subs[0][0]}")
    feats = t.head(8).feature.tolist()[::-1]
    tt = t.set_index("feature").loc[feats]
    y = np.arange(len(feats))
    ax.barh(y + 0.2, tt.ctx_mean_abs, height=0.38, color=C["blue"], label="environment IG")
    ax.barh(y - 0.2, tt.mut_mean_abs, height=0.38, color=C["orange"], label="mutation IG")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Mean |attribution| to log σ_epi")
    ax.set_title(f"What drives epistemic uncertainty ({subs[0][0]})", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _panel_label(ax, "b")
    _save(fig, fd, "fig6_uncertainty")


# ---------------------------------------------------------------------------- 7. cross-dataset
def fig_cross():
    rows = []
    for ds in cfg.DATASETS:
        for sub in ("singles", "doubles"):
            t = _tab(ds, f"driving_{sub}")
            if t is None:
                continue
            for f, r, sh in zip(t.feature, t.rho_true, t.share_abs):
                rows.append(dict(col=f"{cfg.DATASET_LABEL[ds]}\n{sub}", feature=f, rho=r, share=sh))
    df = pd.DataFrame(rows)
    feats = (df[df.col.str.startswith("MegaScale test\nsingles")].sort_values("share", ascending=False)
             .head(12).feature.tolist())
    cols = list(dict.fromkeys(df.col))
    M = df.pivot(index="feature", columns="col", values="rho").loc[feats, cols]
    S = df.pivot(index="feature", columns="col", values="share").loc[feats, cols]
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.2, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8322f"])
    im = ax.imshow(M.values, cmap=cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.values[i, j]
            if np.isfinite(val):
                faint = S.values[i, j] < 0.02
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6.5,
                        color=C["ink2"] if faint else C["ink"], fontstyle="italic" if faint else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.grid(False)
    ax.set_title("Spearman ρ between per-feature mutation attribution and experimental ΔΔG", loc="left")
    fig.colorbar(im, ax=ax, label="ρ", shrink=0.7)
    fig.text(0.01, -0.02, "Rows: the 12 features with the largest attribution share on MegaScale singles. "
                          "Italic grey = feature carries <2% of |attribution| in that set.", fontsize=6.5,
             color=C["ink2"])
    _save(fig, cfg.OUT_DIR / "cross_dataset", "fig7_cross_dataset_feature_rho")


def run(ds):
    fd = cfg.OUT_DIR / ds / "figures"
    for fn in (fig_driving, fig_burial, fig_volume, fig_environment, fig_epistasis, fig_uncertainty):
        try:
            fn(ds, fd)
        except Exception as e:  # keep going; report which figure failed
            print(f"  [!] {fn.__name__} failed for {ds}: {e}")
    print(f"[figures] {ds} -> {fd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(cfg.DATASETS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for ds in (cfg.DATASETS if a.all else [a.dataset]):
        run(ds)
    if a.all:
        fig_cross()
