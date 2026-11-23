"""Protein flagging with sigma_e = EDL-only (intra) epistemic std, threshold series + tercile means. 2026-10-01."""
import config as cfg, numpy as np, pandas as pd, json
from scipy.stats import pearsonr, spearmanr
out = {}
for ds in cfg.DATASETS:
    v = pd.read_csv(cfg.CACHE_DIR / ds / "variants.csv")
    for k, lab in ((1, "singles"), (2, "doubles")):
        s = v[v.n_mutated == k]
        if not len(s): continue
        name = f"{cfg.DATASET_LABEL[ds]} {lab}"; res = dict(n_variants_all=len(s), n_proteins_all=int(s.pdb.nunique()),
                                                          mean_sigma=float(s.intra_epistemic_std.mean()), mae=float(s.abs_err.mean()))
        ser = []
        for nmin in [5, 10, 15, 20, 30]:
            rows = [dict(pdb=p, n=len(g), r=pearsonr(g.pred_mu, g.true_ddg)[0], sig=g.intra_epistemic_std.mean())
                    for p, g in s.groupby("pdb") if len(g) >= nmin and g.true_ddg.std() > 0]
            t = pd.DataFrame(rows)
            if len(t) < 4: ser.append(dict(nmin=nmin, n_prot=len(t))); continue
            r, p = pearsonr(t.sig, t.r); rho, prho = spearmanr(t.sig, t.r)
            e = dict(nmin=nmin, n_prot=len(t), n_var=int(t.n.sum()), pearson=round(r, 3), p=round(p, 4), spearman=round(rho, 3), p_rho=round(prho, 4))
            if nmin == 15 and len(t) >= 9:
                t = t.sort_values("sig"); th = np.array_split(t, 3)
                e.update(r_low_third=round(th[0].r.mean(), 3), r_mid_third=round(th[1].r.mean(), 3), r_high_third=round(th[2].r.mean(), 3),
                         sig_cut=[round(th[0].sig.max(), 3), round(th[2].sig.min(), 3)])
            ser.append(e)
        res["series"] = ser; out[name] = res
(cfg.OUT_DIR / "deepdive_protein_flagging").mkdir(parents=True, exist_ok=True)
json.dump(out, open(cfg.OUT_DIR / "deepdive_protein_flagging" / "intra_thresholds.json", "w"), indent=1)
for k, r in out.items():
    print(k, {x: r[x] for x in ("n_variants_all", "n_proteins_all", "mean_sigma", "mae")})
    for e in r["series"]: print("   ", e)
