"""Deep dive: does mean epistemic uncertainty (sigma_e) flag hard-to-predict proteins?

    python deepdive_protein_flagging.py
Writes tables, key_numbers.json and fig9_protein_flagging.{png,pdf} to <OUT_DIR>/deepdive_protein_flagging/.

Per protein (forward predictions only): Pearson r, Spearman rho, MAE, RMSE / sd(ddG_exp) ("nRMSE"), spread of
experimental ddG, and mean sigma_e (plus its sigma_intra / sigma_inter parts).
Tests:
  1. Spearman rho(mean sigma_e, per-protein r) per dataset and minimum-N threshold, with 95% protein bootstrap CI;
  2. partial rho controlling for the ddG spread (range restriction), mean |mu| and N;
  3. leave-one-protein-out range;
  4. AUROC of mean sigma_e for flagging the hardest third of proteins (lowest r);
  5. all datasets pooled on within-dataset ranks, with a within-dataset permutation test.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

import config as cfg
import figures  # noqa: F401  (shared rcParams)

C = cfg.C
OUT = cfg.OUT_DIR / "deepdive_protein_flagging"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
THRESHOLDS = [5, 10, 15, 20, 30]
MAIN_NMIN = 15


def subsets():
    for ds in cfg.DATASETS:
        v = pd.read_csv(cfg.CACHE_DIR / ds / "variants.csv")
        for k, lab in ((1, "singles"), (2, "doubles")):
            s = v[v.n_mutated == k]
            if len(s):
                yield f"{cfg.DATASET_LABEL[ds]} {lab}", s


def protein_table(s, nmin):
    rows = []
    for p, g in s.groupby("pdb"):
        if len(g) < nmin or g.true_ddg.std() == 0 or g.pred_mu.std() == 0:
            continue
        rmse = np.sqrt(((g.pred_mu - g.true_ddg) ** 2).mean())
        rows.append(dict(pdb=p, n=len(g), r=pearsonr(g.pred_mu, g.true_ddg)[0], rho=spearmanr(g.pred_mu, g.true_ddg)[0],
                         mae=g.abs_err.mean(), rmse=rmse, nrmse=rmse / g.true_ddg.std(), true_sd=g.true_ddg.std(),
                         mean_abs_mu=g.pred_mu.abs().mean(), sigma_e=g.epistemic_std.mean(),
                         sigma_intra=g.intra_epistemic_std.mean(), sigma_inter=g.inter_epistemic_std.mean()))
    return pd.DataFrame(rows)


def partial(t, x, y, z):
    rx, ry = rankdata(t[x]), rankdata(t[y])
    Z = np.column_stack([rankdata(t[c]) for c in z] + [np.ones(len(t))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)


def boot_rho(x, y, n=2000):
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    bs = [spearmanr(x[i], y[i])[0] for i in idx]
    return np.nanpercentile(bs, [2.5, 97.5])


def auroc(score, positive):
    """P(score of a random positive > score of a random negative)."""
    r = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    K, rows_thr, rows_main, tables = {}, [], [], {}
    for name, s in subsets():
        for nmin in THRESHOLDS:
            t = protein_table(s, nmin)
            if len(t) < 6:
                continue
            r = dict(set=name, nmin=nmin, n_proteins=len(t))
            for y in ("r", "rho", "mae", "nrmse"):
                sr = spearmanr(t.sigma_e, t[y])
                r[f"rho_sigma_vs_{y}"], r[f"p_{y}"] = sr[0], sr[1]
            r["pearson_sigma_vs_r"], r["pearson_p"] = pearsonr(t.sigma_e, t.r)
            rows_thr.append(r)
        t = protein_table(s, MAIN_NMIN)
        if len(t) < 8:
            continue
        tables[name] = t.assign(set=name)
        lo, hi = boot_rho(t.sigma_e.values, t.r.values)
        loo = [spearmanr(t.sigma_e.drop(i), t.r.drop(i))[0] for i in t.index]
        hard = (t.r <= t.r.quantile(1 / 3)).values
        pc = partial(t, "sigma_e", "r", ["true_sd"])
        pc3 = partial(t, "sigma_e", "r", ["true_sd", "mean_abs_mu", "n"])
        rows_main.append(dict(
            set=name, n_proteins=len(t), rho_sigma_r=spearmanr(t.sigma_e, t.r)[0], ci_lo=lo, ci_hi=hi,
            p=spearmanr(t.sigma_e, t.r)[1], pearson=pearsonr(t.sigma_e, t.r)[0], pearson_p=pearsonr(t.sigma_e, t.r)[1],
            loo_min=min(loo), loo_max=max(loo), partial_given_spread=pc[0], partial_given_spread_p=pc[1],
            partial_given_spread_absmu_n=pc3[0], partial_given_spread_absmu_n_p=pc3[1],
            rho_inter_r=spearmanr(t.sigma_inter, t.r)[0], rho_intra_r=spearmanr(t.sigma_intra, t.r)[0],
            rho_sigma_mae=spearmanr(t.sigma_e, t.mae)[0], rho_sigma_mae_p=spearmanr(t.sigma_e, t.mae)[1],
            rho_sigma_nrmse=spearmanr(t.sigma_e, t.nrmse)[0], rho_spread_r=spearmanr(t.true_sd, t.r)[0],
            auroc_flag_hardest_third=auroc(t.sigma_e.values, hard),
            auroc_flag_by_spread=auroc(-t.true_sd.values, hard)))
    thr = pd.DataFrame(rows_thr)
    main_t = pd.DataFrame(rows_main)
    allp = pd.concat(tables.values(), ignore_index=True)
    thr.to_csv(OUT / "threshold_sensitivity.csv", index=False, float_format="%.4g")
    main_t.to_csv(OUT / f"protein_flagging_N{MAIN_NMIN}.csv", index=False, float_format="%.4g")
    allp.to_csv(OUT / f"per_protein_N{MAIN_NMIN}.csv", index=False, float_format="%.4g")
    # pooled on within-set ranks (percentiles), permutation within set
    allp["sig_pct"] = allp.groupby("set").sigma_e.rank(pct=True)
    allp["r_pct"] = allp.groupby("set").r.rank(pct=True)
    obs = spearmanr(allp.sig_pct, allp.r_pct)[0]
    null = []
    for _ in range(5000):
        perm = allp.groupby("set").sig_pct.transform(lambda x: RNG.permutation(x.values))
        null.append(spearmanr(perm, allp.r_pct)[0])
    p_perm = (np.sum(np.array(null) <= obs) + 1) / (len(null) + 1)
    K["pooled_within_set"] = dict(n_proteins=len(allp), sets=list(tables), rho=round(obs, 3), perm_p_one_sided=round(p_perm, 4))
    K["main"] = {r.set: {k: (round(val, 3) if isinstance(val, float) else val) for k, val in r._asdict().items()
                         if k not in ("Index", "set")} for r in main_t.itertuples()}
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, default=str)
    figure(tables, thr)
    print(main_t.round(3).to_string(index=False))
    print(thr[["set", "nmin", "n_proteins", "rho_sigma_vs_r", "p_r", "rho_sigma_vs_mae", "p_mae"]].round(3).to_string(index=False))
    print(json.dumps(K["pooled_within_set"]))


def figure(tables, thr):
    names = list(tables)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(2.55 * (len(names) + 1) + 0.6, 2.9),
                             gridspec_kw=dict(wspace=0.45))
    for ax, name, col in zip(axes, names, cfg.SERIES):
        t = tables[name]
        ax.scatter(t.sigma_e, t.r, s=np.clip(np.sqrt(t.n) * 2.2, 10, 90), color=col, alpha=0.8, edgecolor="white", lw=0.5)
        b = np.polyfit(t.sigma_e, t.r, 1)
        xx = np.linspace(t.sigma_e.min(), t.sigma_e.max(), 10)
        ax.plot(xx, np.polyval(b, xx), color=C["ink2"], lw=1, ls="--")
        sr = spearmanr(t.sigma_e, t.r)
        ax.set_title(f"{name}\n{len(t)} proteins, ρ={sr[0]:+.2f} (p={sr[1]:.3f})", loc="left", fontsize=7.5)
        ax.set_xlabel("mean σ_e of protein")
    axes[0].set_ylabel("per-protein Pearson r")
    ax = axes[-1]
    for name, col in zip(names, cfg.SERIES):
        q = thr[thr.set == name]
        ax.plot(q.nmin, q.rho_sigma_vs_r, "-o", color=col, lw=1.6, ms=4, label=name)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    ax.set_xlabel("minimum variants per protein")
    ax.set_ylabel("ρ(mean σ_e, per-protein r)")
    ax.set_title("threshold sensitivity", loc="left", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig9_protein_flagging.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
