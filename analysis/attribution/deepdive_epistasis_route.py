"""Through which features does the model's non-additivity run? (MegaScale-D test, existing IG only)

Mutation IG is complete for every variant: sum_f A_f = mu - mu_null. For a double AB matched to its two singles,
    dA_f = A_AB,f - A_A,f - A_B,f      and      sum_f dA_f = E_pred_null = (mu_AB - mu0_AB) - (mu_A - mu0_A) - (mu_B - mu0_B)
so the model's own non-additivity decomposes exactly over features. Nothing is re-run.

Also: (i) split by head (gamma_epi vs gamma_base + gamma_phys), (ii) input non-additivity of the features themselves
(dX_f = dx_AB - dx_A - dx_B; zero for amino-acid properties, non-zero for MadraX terms at contact), and
(iii) a consistency check across independent forward passes: for distant pairs the double's IG should be close to
the sum of the singles' IGs.

    python deepdive_epistasis_route.py      # ~30 s
Outputs -> <OUT_DIR>/deepdive_epistasis_route/ (key_numbers.json, tables/*.csv, fig27)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

import config as cfg
import figures  # noqa: F401  (shared rcParams)

DS = "megascale_test"
C = cfg.C
F = list(cfg.FEATURES)
OUT = cfg.OUT_DIR / "deepdive_epistasis_route"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(cfg.SEED)
N_BOOT = 1000
GROUPS = {
    "hydrophobic / solvation": ["Hydrophobicity", "Polar Solvation", "Hydrophobic Solvation"],
    "packing": ["Volume", "VdW Clashes", "VdW Energy"],
    "backbone": ["Cyclic", "Backbone Entropy", "Hydrogen Bonds"],
    "electrostatics": ["Electrostatics", "Charge"],
}
GROUPS["other"] = [f for f in F if f not in sum(GROUPS.values(), [])]
SHOW = ["Volume", "VdW Clashes", "VdW Energy", "Hydrophobicity", "Polar Solvation", "Hydrophobic Solvation",
        "Electrostatics", "Hydrogen Bonds", "Cyclic", "Backbone Entropy"]
BINS = ["<6 A", "6-9 A", ">9 A"]
K = {}


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def srho(a, b):
    return pearsonr(rankdata(a), rankdata(b))[0]


def psr(x, y, z):
    """partial Spearman of x and y given z (linear on ranks)"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return pearsonr(ex, ey)[0]


def boot_groups(keys, n=N_BOOT):
    keys = np.asarray(keys)
    u = np.unique(keys)
    by = {p: np.where(keys == p)[0] for p in u}
    for _ in range(n):
        yield np.concatenate([by[p] for p in RNG.choice(u, len(u))])


def ci(vals):
    vals = np.asarray(vals, float)
    return [r3(np.nanpercentile(vals, 2.5)), r3(np.nanpercentile(vals, 97.5))]


def gsum(M):
    return {g: M[:, [F.index(f) for f in fs]].sum(1) for g, fs in GROUPS.items()}


# ---------------------------------------------------------------------------- load and match
def load():
    cd = cfg.CACHE_DIR / DS
    v = pd.read_csv(cd / "variants.csv")
    A = {k: np.load(cd / f"attr_{k}.npz")["F_tot"] for k in ("mut_mu", "mut_base", "mut_epi")}
    A["mut_phys"] = A["mut_mu"] - A["mut_base"] - A["mut_epi"]
    DX = np.load(cd / "geom.npz")["dx_all"]
    one = cfg.THREE_TO_ONE
    sv = v[v.n_mutated == 1]
    key = {f"{p}::{one[w]}{i}{one[m]}": j for j, p, w, i, m in zip(sv.index, sv.pdb, sv.wt1, sv.pos1, sv.mt1)}
    d = v[v.n_mutated == 2].copy()
    ia = np.array([key.get(f"{p}::{one[w]}{i}{one[m]}", -1) for p, w, i, m in zip(d.pdb, d.wt1, d.pos1, d.mt1)])
    ib = np.array([key.get(f"{p}::{one[w]}{i}{one[m]}", -1) for p, w, i, m in zip(d.pdb, d.wt2, d.pos2, d.mt2)])
    ok = (ia >= 0) & (ib >= 0)
    d, ia, ib = d[ok], ia[ok], ib[ok]
    iab = d.index.values
    ep = pd.read_csv(cfg.OUT_DIR / DS / "tables" / "epistasis_per_variant.csv")
    ep = ep.set_index(ep.pdb + "|" + ep.mutation)
    q = ep.loc[(d.pdb + "|" + d.mutation).values]
    d["E_exp"], d["add_exp"], d["E_pred"], d["E_pred_null"] = q.E_exp.values, q.add_exp.values, q.E_pred.values, q.E_pred_null.values
    d = d.reset_index(drop=True)
    dA = {k: A[k][iab] - A[k][ia] - A[k][ib] for k in A}
    S = {k: A[k][ia] + A[k][ib] for k in A}
    DAB = {k: A[k][iab] for k in A}
    dX = DX[iab] - DX[ia] - DX[ib]
    d["bin"] = pd.cut(d.d12, [0, 6, 9, np.inf], right=False, labels=BINS).astype(str)
    d["pair"] = d.pdb + "_" + d.pos1.astype(str) + "_" + d.pos2.astype(str)
    d["E_attr"] = dA["mut_mu"].sum(1)
    d["E_epi"] = dA["mut_epi"].sum(1)
    d["E_rest"] = d.E_attr - d.E_epi
    K["n"] = dict(doubles_matched=int(len(d)), site_pairs=int(d.pair.nunique()), proteins=int(d.pdb.nunique()),
                  per_bin={b: int((d.bin == b).sum()) for b in BINS},
                  pairs_per_bin={b: int(d[d.bin == b].pair.nunique()) for b in BINS},
                  max_abs_Eattr_minus_Epred_null=r3(np.nanmax(np.abs(d.E_attr - d.E_pred_null))),
                  median_abs_Eattr_minus_Epred_null=r3(np.nanmedian(np.abs(d.E_attr - d.E_pred_null))),
                  max_abs_dX_amino_acid_props=r3(np.abs(dX[:, :11]).max()),
                  max_abs_dX_sasa=r3(np.abs(dX[:, cfg.SASA_IDX]).max()))
    return d, dA, S, DAB, dX


# ---------------------------------------------------------------------------- 1. variance shares of E_pred
def shares(d, dA):
    rows = []
    E = d.E_attr.values
    Eexp = d.E_exp.values
    okx = np.isfinite(Eexp)
    # experimental residual after removing the global/saturation term (linear in the experimental additive sum)
    res = np.full(len(d), np.nan)
    res[okx] = Eexp[okx] - np.polyval(np.polyfit(d.add_exp[okx], Eexp[okx], 1), d.add_exp[okx])
    d["E_exp_res"] = res
    G = gsum(dA["mut_mu"])
    feats = {**{f: dA["mut_mu"][:, F.index(f)] for f in F}, **{f"[{g}]": x for g, x in G.items()},
             "[head: gamma_epi]": d.E_epi.values, "[head: base + phys]": d.E_rest.values}
    for b in ["all"] + BINS:
        m = np.ones(len(d), bool) if b == "all" else (d.bin == b).values
        mm = m & okx
        for nm, x in feats.items():
            vs = np.cov(x[m], E[m])[0, 1] / np.var(E[m], ddof=1)
            es = np.cov(x[mm], Eexp[mm])[0, 1] / np.cov(E[mm], Eexp[mm])[0, 1]
            er = np.cov(x[mm], res[mm])[0, 1] / np.cov(E[mm], res[mm])[0, 1]
            rows.append(dict(bin=b, feature=nm, mean_dA=x[m].mean(), mean_abs_dA=np.abs(x[m]).mean(),
                             var_share=vs, exp_share=es, exp_share_after_saturation=er,
                             rho_vs_E_exp=srho(x[mm], Eexp[mm]),
                             prho_vs_E_exp_given_add=psr(x[mm], Eexp[mm], d.add_exp.values[mm])))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "epistasis_feature_shares.csv", index=False, float_format="%.4g")
    K["overall"] = {b: dict(mean_E_pred=r3(E[(d.bin == b).values if b != 'all' else slice(None)].mean()),
                            rho_E_pred_E_exp=r3(srho(E[((d.bin == b).values if b != 'all' else np.ones(len(d), bool)) & okx],
                                                     Eexp[((d.bin == b).values if b != 'all' else np.ones(len(d), bool)) & okx])))
                    for b in ["all"] + BINS}
    K["shares"] = {f"{r.bin}|{r.feature}": dict(var=r3(r.var_share), exp=r3(r.exp_share),
                                                exp_after_sat=r3(r.exp_share_after_saturation),
                                                prho=r3(r.prho_vs_E_exp_given_add), mean=r3(r.mean_dA))
                   for r in t.itertuples() if r.feature.startswith("[") or r.feature in SHOW}
    return t


def boot_contact(d, dA):
    """CIs for the <6 A shares: protein bootstrap and site-pair bootstrap."""
    m = ((d.bin == "<6 A") & np.isfinite(d.E_exp)).values
    s = d[m].reset_index(drop=True)
    G = gsum(dA["mut_mu"][m])
    X = {**G, "[head: gamma_epi]": s.E_epi.values, "[head: base + phys]": s.E_rest.values}
    out = {}
    for unit, keys in (("protein", s.pdb.values), ("site_pair", s.pair.values)):
        bs = {k: ([], [], []) for k in X}
        for i in boot_groups(keys, 500):
            E, Ee, res = s.E_attr.values[i], s.E_exp.values[i], s.E_exp_res.values[i]
            for k, x in X.items():
                bs[k][0].append(np.cov(x[i], E)[0, 1] / np.var(E, ddof=1))
                bs[k][1].append(np.cov(x[i], res)[0, 1] / np.cov(E, res)[0, 1])
                bs[k][2].append(psr(x[i], Ee, s.add_exp.values[i]))
        out[unit] = {k: dict(var_share_ci=ci(v[0]), exp_share_after_sat_ci=ci(v[1]), prho_ci=ci(v[2]))
                     for k, v in bs.items()}
    K["contact_ci"] = out


# ---------------------------------------------------------------------------- 2. inputs vs network
def inputs_vs_network(d, dA, dX):
    rows = []
    for b in ["all"] + BINS:
        m = np.ones(len(d), bool) if b == "all" else (d.bin == b).values
        for f in SHOW:
            j = F.index(f)
            x, a = dX[m, j], dA["mut_mu"][m, j]
            rows.append(dict(bin=b, feature=f, is_madrax=j >= 12, mean_abs_dX=np.abs(x).mean(),
                             frac_dX_nonzero=(np.abs(x) > 0.05).mean() if j >= 12 else (np.abs(x) > 1e-6).mean(),
                             mean_abs_dA=np.abs(a).mean(),
                             rho_dA_dX=srho(a, x) if np.ptp(x) > 0 else np.nan))
        # share of E_pred carried by features whose input is exactly additive (amino-acid properties + SASA)
        E = d.E_attr.values[m]
        aa = dA["mut_mu"][m][:, :12].sum(1)
        rows.append(dict(bin=b, feature="[amino-acid props + SASA: input additive]",
                         var_share=np.cov(aa, E)[0, 1] / np.var(E, ddof=1), mean_abs_dA=np.abs(aa).mean()))
        mx = dA["mut_mu"][m][:, 12:].sum(1)
        rows.append(dict(bin=b, feature="[MadraX terms]", var_share=np.cov(mx, E)[0, 1] / np.var(E, ddof=1),
                         mean_abs_dA=np.abs(mx).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "inputs_vs_network.csv", index=False, float_format="%.4g")
    K["inputs_vs_network"] = {f"{r.bin}|{r.feature}": {k: r3(getattr(r, k)) for k in
                                                       ("mean_abs_dX", "mean_abs_dA", "rho_dA_dX", "var_share")
                                                       if hasattr(r, k)}
                              for r in t.itertuples()}
    return t


# ---------------------------------------------------------------------------- 3. consistency across forward passes
def consistency(d, S, DAB):
    rows = []
    for b in BINS:
        m = (d.bin == b).values
        for k in ("mut_mu", "mut_rest"):
            if k == "mut_mu":
                ab, s = DAB["mut_mu"][m], S["mut_mu"][m]
            else:  # without the doubles-only head
                ab, s = DAB["mut_mu"][m] - DAB["mut_epi"][m], S["mut_mu"][m]
            num = (ab * s).sum(1)
            cos = num / (np.linalg.norm(ab, axis=1) * np.linalg.norm(s, axis=1) + 1e-12)
            r = dict(bin=b, target=k, n=int(m.sum()), median_cosine=np.median(cos), q25=np.quantile(cos, .25),
                     ratio_mass=np.median(np.abs(ab).sum(1) / (np.abs(s).sum(1) + 1e-12)))
            for f in ["Volume", "VdW Clashes", "Hydrophobicity", "Polar Solvation", "Electrostatics", "Cyclic"]:
                j = F.index(f)
                r[f"rho_{f}"] = srho(ab[:, j], s[:, j])
                r[f"slope_{f}"] = np.polyfit(s[:, j], ab[:, j], 1)[0]
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "consistency_double_vs_sum_of_singles.csv", index=False, float_format="%.4g")
    K["consistency"] = {f"{r['bin']}|{r['target']}": {k: r3(x) if isinstance(x, (float, np.floating)) else x
                                                      for k, x in r.items()} for r in rows}
    return t


# ---------------------------------------------------------------------------- figure
def figure(t, cons):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8, 1]))
    gnames = [f"[{g}]" for g in GROUPS]
    cols = [C["blue"], C["orange"], C["violet"], C["red"], C["neutral"]]
    a = ax[0]
    left = np.zeros(len(BINS))
    for g, c in zip(gnames, cols):
        vals = np.array([t[(t.bin == b) & (t.feature == g)].var_share.iloc[0] for b in BINS])
        pos, neg = np.clip(vals, 0, None), np.clip(vals, None, 0)
        a.barh(range(len(BINS)), pos, left=left, color=c, height=0.6, label=g.strip("[]"))
        a.barh(range(len(BINS)), neg, color=c, height=0.6, alpha=0.5)
        left += pos
    a.set_yticks(range(len(BINS)), [f"d12 {b}" for b in BINS])
    a.axvline(0, color=C["ink2"], lw=0.6)
    a.set(title="a  Share of the model's non-additivity", xlabel="variance share of E_pred")
    a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    a = ax[1]
    ci_ = K["contact_ci"]["site_pair"]
    names = [g for g in GROUPS]
    vals = [t[(t.bin == "<6 A") & (t.feature == f"[{g}]")].prho_vs_E_exp_given_add.iloc[0] for g in names]
    lo = [ci_[g]["prho_ci"][0] for g in names]
    hi = [ci_[g]["prho_ci"][1] for g in names]
    y = np.arange(len(names))
    a.errorbar(vals, y, xerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="o", color=C["violet"], ms=4, lw=0.8)
    a.set_yticks(y, names)
    a.axvline(0, color=C["ink2"], lw=0.6)
    a.set(title="b  Contact pairs (<6 Å): which feature group tracks E_exp",
          xlabel="partial ρ(dA_group, E_exp | additive), 95% site-pair CI")
    a = ax[2]
    h = ["[head: gamma_epi]", "[head: base + phys]"]
    for i, (nm, c) in enumerate(zip(h, [C["aqua"], C["ink2"]])):
        vals = [t[(t.bin == b) & (t.feature == nm)].prho_vs_E_exp_given_add.iloc[0] for b in BINS]
        a.bar(np.arange(3) + (i - 0.5) * 0.38, vals, width=0.38, color=c, label=nm.strip("[]").replace("head: ", ""))
    a.set_xticks(range(3), BINS)
    a.axhline(0, color=C["ink2"], lw=0.6)
    a.legend(fontsize=6.5)
    a.set(title="c  Which head tracks E_exp", ylabel="partial ρ with E_exp | additive")
    a = ax[3]
    q = cons[cons.target == "mut_rest"]
    fs = ["Volume", "VdW Clashes", "Hydrophobicity", "Polar Solvation", "Electrostatics", "Cyclic"]
    for i, b in enumerate(BINS):
        a.plot(range(len(fs)), [q[q.bin == b][f"rho_{f}"].iloc[0] for f in fs], "o-", ms=3.5, lw=0.9,
               color=[C["red"], C["yellow"], C["blue"]][i], label=f"d12 {b}")
    a.set_xticks(range(len(fs)), fs, rotation=35, ha="right")
    a.set(ylim=(0, 1.02), ylabel="ρ(IG double, sum of IG singles)", title="d  Consistency across forward passes")
    a.legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig27_epistasis_route.{ext}")
    plt.close(fig)


def main():
    d, dA, S, DAB, dX = load()
    t = shares(d, dA)
    boot_contact(d, dA)
    inputs_vs_network(d, dA, dX)
    cons = consistency(d, S, DAB)
    figure(t, cons)
    json.dump(K, open(OUT / "key_numbers.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: K[k] for k in ("n", "overall")}, indent=1))


if __name__ == "__main__":
    main()
