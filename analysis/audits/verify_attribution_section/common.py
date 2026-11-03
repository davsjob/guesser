"""Shared helpers for verifying the attribution section (2026-10-02).

Reads only the attribution cache and tables; writes to results/audits/verify_attribution_section.
sigma_e in the manuscript = sigma_intra = sqrt(mean_s beta_s/(v_s(alpha_s-1))) = column `intra_epistemic_std`.
Column `epistemic_std` = sqrt(sigma_intra^2 + sigma_inter^2) (the old total), `inter_epistemic_std` = seed SD of mu.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

PKG = Path(__file__).resolve().parents[2] / "attribution"
sys.path.insert(0, str(PKG))
import config as cfg  # noqa: E402
from paths import AUDITS_OUT, REF_DIR, RESULTS  # noqa: E402

HERE = AUDITS_OUT / "verify_attribution_section"
HERE.mkdir(parents=True, exist_ok=True)

F = list(cfg.FEATURES)
FI = {f: i for i, f in enumerate(F)}
DS = "megascale_test"
CD = cfg.CACHE_DIR / DS
OUTV3 = cfg.OUT_DIR
B = 2000
RNG = np.random.default_rng(20261002)
CHARGE = {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1}


def r3(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 3)


def load_v():
    v = pd.read_csv(CD / "variants.csv")
    from deepdive_attribution_mechanisms import substitution_class
    v["cls"] = [substitution_class(w, m) for w, m in zip(v.wt1, v.mt1)]
    v["cls2"] = [substitution_class(w, m) if isinstance(w, str) else "" for w, m in zip(v.wt2, v.mt2)]
    v["g"], v["g0"] = v.f_mut_epistemic, v.f0_mut_epistemic
    v["gm"] = v.g - v.g0
    v["tmu"] = v.pred_mu - v.mu_null
    v["serr"] = v.pred_mu - v.true_ddg
    v["site"] = v.pdb + "_" + v.pos1.astype(str) + np.where(v.n_mutated == 2, "_" + v.pos2.astype(str), "")
    v["sig_intra"] = v.intra_epistemic_std
    return v


def load_A(keys=("mut_mu", "ctx_mu", "mut_epistemic", "ctx_epistemic")):
    return {k: np.load(CD / f"attr_{k}.npz")["F_tot"].astype(float) for k in keys}


class PB:
    """Protein bootstrap with one fixed set of multinomial weights per protein set, so that every
    statistic computed from the same PB object is paired across groups."""

    def __init__(self, pdb, n=B, rng=RNG):
        self.pdb = np.asarray(pdb)
        self.u, self.code = np.unique(self.pdb, return_inverse=True)
        P = len(self.u)
        self.W = rng.multinomial(P, np.full(P, 1.0 / P), size=n).astype(float)

    def sums(self, x, mask=None):
        x = np.asarray(x, float)
        m = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
        s = np.bincount(self.code[m], weights=x[m], minlength=len(self.u))
        c = np.bincount(self.code[m], minlength=len(self.u)).astype(float)
        return s, c

    def mean(self, x, mask=None):
        """Returns (point estimate, bootstrap draws)."""
        s, c = self.sums(x, mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s.sum() / c.sum(), (self.W @ s) / (self.W @ c)


def ci(d, lo=2.5, hi=97.5):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return [r3(np.percentile(d, lo)), r3(np.percentile(d, hi))]


def boot_idx(pdb, n, rng=RNG):
    pdb = np.asarray(pdb)
    u = np.unique(pdb)
    by = {p: np.where(pdb == p)[0] for p in u}
    for _ in range(n):
        yield np.concatenate([by[p] for p in rng.choice(u, len(u))])


def srho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(rankdata(a), rankdata(b))[0]


def save(name, obj):
    p = HERE / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: r3(o) if np.isscalar(o) else str(o)))
    print("wrote", p)
