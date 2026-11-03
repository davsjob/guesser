"""Stage 1: extract compact per-variant arrays from an explain_s_std_a_h5_v3 file.

Usage:
    python extract.py --dataset megascale_test --stage scalars
    python extract.py --dataset megascale_test --stage geom
    python extract.py --dataset megascale_test --stage attr --key mut_mu
    python extract.py --dataset megascale_test --stage all      # everything (small files)

Outputs in cache/<dataset>/ :
    variants.csv         one row per variant: predictions, UQ, IG scalars, parsed mutation, RSA, d12
    geom.npz             per-node distance to mutated site, changed flag; per-variant dX sums
    attr_<key>.npz       per-variant feature sums (total / mutated nodes), node-mass by shell, etc.

Stages are separate so each fits in a short job; every stage is idempotent.
"""
import argparse
import re
import time

import h5py
import numpy as np
import pandas as pd

import config as cfg

MUT_RE = re.compile(r"'([A-Za-z0-9])',\s*'([A-Z]{3})',\s*(-?\d+),\s*'([A-Z]{3})'")


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def _open(ds):
    return h5py.File(cfg.EXPL_DIR / cfg.DATASETS[ds], "r")


def _outdir(ds):
    d = cfg.CACHE_DIR / ds
    d.mkdir(parents=True, exist_ok=True)
    return d


def _offsets(f):
    no = f["mutation/node_offset"][:]
    return no, np.diff(no), len(no) - 1


def _segsum(arr, no):
    """Sum rows of arr within each variant segment (node_offset boundaries)."""
    return np.add.reduceat(arr, no[:-1], axis=0)


# ----------------------------------------------------------------------------------------------
def stage_scalars(ds):
    f = _open(ds)
    M, P, I = f["mutation"], f["prediction"], f["ig"]
    df = pd.DataFrame({
        "pdb": _dec(M["pdb"][:]), "mutation": _dec(M["mutation"][:]),
        "n_mutated": M["n_mutated"][:], "n_nodes": M["n_nodes"][:], "true_ddg": M["true_ddg"][:],
        "wt_identity_match": M["wt_identity_match"][:],
    })
    df["mapping_ok"] = pd.Series(_dec(M["mapping_status"][:])).str.startswith("verified").values
    for k in ["pred_mu", "pred_gamma_phys", "pred_gamma_base", "pred_gamma_epi", "pred_v", "pred_alpha",
              "pred_beta", "aleatoric_std", "intra_epistemic_std", "inter_epistemic_std", "epistemic_std",
              "total_std"]:
        df[k] = P[k][:]
    for k in ["f_mut_mu", "f0_mut_mu", "completeness_err_mut_mu", "attr_mass_mut_mu",
              "f_ctx_mu", "f0_ctx_mu", "completeness_err_ctx_mu", "attr_mass_ctx_mu",
              "f_mut_epistemic", "f0_mut_epistemic", "f0_ctx_epistemic", "completeness_err_mut_epistemic",
              "completeness_err_ctx_epistemic", "attr_mass_mut_epistemic", "attr_mass_ctx_epistemic",
              "completeness_err_mut_epi", "completeness_err_mut_base"]:
        df[k] = I[k][:]
    df = df.rename(columns={"f0_mut_mu": "mu_null"})
    df["abs_err"] = (df.pred_mu - df.true_ddg).abs()
    # parse mutation strings: [('A', 'TRP', 1, 'ASP'), ...]
    parsed = df.mutation.apply(MUT_RE.findall)
    df["sites"] = parsed.apply(lambda p: ";".join(f"{c}:{w}{r}{m}" for c, w, r, m in p))
    df["wt1"] = parsed.apply(lambda p: p[0][1] if p else "")
    df["mt1"] = parsed.apply(lambda p: p[0][3] if p else "")
    df["pos1"] = parsed.apply(lambda p: int(p[0][2]) if p else -1)
    df["wt2"] = parsed.apply(lambda p: p[1][1] if len(p) > 1 else "")
    df["mt2"] = parsed.apply(lambda p: p[1][3] if len(p) > 1 else "")
    df["pos2"] = parsed.apply(lambda p: int(p[1][2]) if len(p) > 1 else -1)
    df["seq_sep"] = np.where(df.n_mutated == 2, (df.pos2 - df.pos1).abs(), np.nan)
    df.to_csv(_outdir(ds) / "variants_base.csv", index=False)
    print(f"[scalars] {ds}: {len(df)} variants")


# ----------------------------------------------------------------------------------------------
def _pdb_ca(path):
    ca = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                key = (line[21], int(line[22:26]), line[26].strip())
                ca.setdefault(key, (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return ca


def stage_geom(ds):
    t0 = time.time()
    f = _open(ds)
    no, nn, N = _offsets(f)
    V = int(no[-1])
    var_node = np.repeat(np.arange(N), nn)
    ism = f["node/is_mutated"][:].astype(bool)
    # ---- distances from graph edges (exact for Calpha <= 9 A) ----
    eo = f["mutation/edge_offset"][:]
    var_edge = np.repeat(np.arange(N), np.diff(eo))
    EI = f["edge/edge_index"][:].astype(np.int64)
    dist = f["edge/edge_attr"][:, 1].astype(np.float64)
    gs, gd = EI[:, 0] + no[var_edge], EI[:, 1] + no[var_edge]
    d = np.full(V, np.inf)
    d[ism] = 0.0
    m = ism[gs]
    np.minimum.at(d, gd[m], dist[m])
    m = ism[gd]
    np.minimum.at(d, gs[m], dist[m])
    d12 = np.full(N, np.nan)
    both = ism[gs] & ism[gd] & (gs != gd)
    tmp = np.full(N, np.inf)
    np.minimum.at(tmp, var_edge[both], dist[both])
    d12[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
    d_source = "edges (exact <= 9 A; beyond 9 A = inf)"
    # ---- exact Calpha distances from local PDBs where available ----
    pdir = cfg.pdb_dir(ds)
    if pdir is not None:
        pdb = _dec(f["mutation/pdb"][:])
        chain, rid, icode = _dec(f["node/res_chain"][:]), f["node/res_id"][:], _dec(f["node/res_icode"][:])
        nm = f["mutation/n_mutated"][:]
        cache, dmax_check = {}, 0.0
        for i in range(N):
            p = pdb[i]
            if p not in cache:
                cache[p] = _pdb_ca(pdir / p)
            ca = cache[p]
            r = slice(no[i], no[i + 1])
            X = np.array([ca[(c, int(x), k.strip())] for c, x, k in zip(chain[r], rid[r], icode[r])])
            mi = np.where(ism[r])[0]
            D = np.linalg.norm(X[:, None, :] - X[None, mi, :], axis=-1).min(1)
            fin = np.isfinite(d[r])
            dmax_check = max(dmax_check, float(np.abs(D[fin] - d[r][fin]).max()))
            d[r] = D
            if nm[i] == 2 and len(mi) == 2:
                d12[i] = float(np.linalg.norm(X[mi[0]] - X[mi[1]]))
        d_source = f"PDB Calpha ({pdir}); max |PDB - edge| on edge-covered nodes = {dmax_check:.2e} A"
    # ---- feature changes ----
    wt = f["node/raw_node_features_wt"][:].astype(np.float64)
    mt = f["node/raw_node_features_mt"][:].astype(np.float64)
    dx = mt - wt
    changed = (np.abs(dx) > 1e-6).any(1)
    ismf = ism[:, None].astype(np.float64)
    out = dict(
        node_dist=d.astype(np.float32), node_changed=changed, node_is_mut=ism,
        dx_self=_segsum(dx * ismf, no), dx_all=_segsum(dx, no),
        adx_self=_segsum(np.abs(dx) * ismf, no), adx_neigh=_segsum(np.abs(dx) * (1 - ismf), no),
        n_changed_neigh=np.bincount(var_node, weights=(changed & ~ism).astype(float), minlength=N),
        d12=d12,
    )
    # SASA / RSA at mutated sites
    sasa_mut = np.bincount(var_node, weights=wt[:, cfg.SASA_IDX] * ism, minlength=N) / np.maximum(
        np.bincount(var_node, weights=ism.astype(float), minlength=N), 1)
    resn = _dec(f["node/res_name_3"][:])
    maxasa = np.array([cfg.TIEN_MAX_ASA.get(r, 180.0) for r in resn])
    rsa_node = wt[:, cfg.SASA_IDX] / maxasa
    rsa_mean = np.bincount(var_node, weights=rsa_node * ism, minlength=N) / np.maximum(
        np.bincount(var_node, weights=ism.astype(float), minlength=N), 1)
    rsa_min = np.full(N, np.inf)
    np.minimum.at(rsa_min, var_node[ism], rsa_node[ism])
    out.update(sasa_mut=sasa_mut, rsa_mean=rsa_mean, rsa_min=rsa_min)
    np.savez_compressed(_outdir(ds) / "geom.npz", **out, d_source=np.array(d_source))
    print(f"[geom] {ds}: {N} variants, {V} nodes; {d_source}; {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------------------------
def stage_attr(ds, key):
    t0 = time.time()
    f = _open(ds)
    no, nn, N = _offsets(f)
    g = np.load(_outdir(ds) / "geom.npz", allow_pickle=True)
    d, changed, ism = g["node_dist"], g["node_changed"], g["node_is_mut"]
    var_node = np.repeat(np.arange(N), nn)
    A = f["node"][cfg.ATTR_KEYS[key]][:].astype(np.float64)
    ismf = ism[:, None].astype(np.float64)
    mass = np.abs(A).sum(1)
    # shell index: 0 self, 1 (0,6], 2 (6,9], 3 >9
    sh = np.where(ism, 0, np.where(d <= 6.0, 1, np.where(d <= 9.0, 2, 3)))
    mshell = np.bincount(var_node * 4 + sh, weights=mass, minlength=N * 4).reshape(N, 4)
    nshell = np.bincount(var_node * 4 + sh, minlength=N * 4).reshape(N, 4)
    seg_max = np.maximum.reduceat(mass, no[:-1])
    seg_max_mut = np.maximum.reduceat(mass * ism, no[:-1])
    # feature-resolved mass on neighbours (for dX-share vs attribution-share contrast)
    out = dict(
        F_tot=_segsum(A, no), F_self=_segsum(A * ismf, no),
        absF_tot=_segsum(np.abs(A), no), absF_neigh=_segsum(np.abs(A) * (1 - ismf), no),
        mass_shell=mshell, n_shell=nshell,
        mass_changed_neigh=np.bincount(var_node, weights=mass * (changed & ~ism), minlength=N),
        mass_unchanged=np.bincount(var_node, weights=mass * (~changed & ~ism), minlength=N),
        top_node_is_mut=seg_max_mut >= seg_max - 1e-12,
    )
    np.savez_compressed(_outdir(ds) / f"attr_{key}.npz", **out)
    print(f"[attr] {ds}/{key}: {time.time() - t0:.1f}s")


def stage_merge(ds):
    """Join variants_base.csv with per-variant geometry scalars -> variants.csv."""
    od = _outdir(ds)
    df = pd.read_csv(od / "variants_base.csv")
    g = np.load(od / "geom.npz", allow_pickle=True)
    for k in ["sasa_mut", "rsa_mean", "rsa_min", "d12", "n_changed_neigh"]:
        df[k] = g[k]
    df["rsa_class"] = pd.cut(df.rsa_mean, [-np.inf, cfg.RSA_CORE, cfg.RSA_SURF, np.inf],
                             labels=["core", "boundary", "surface"], right=False)
    df.to_csv(od / "variants.csv", index=False)
    print(f"[merge] {ds}: variants.csv written")


# ----------------------------------------------------------------------------------------------
def stage_seeds(ds, key):
    """Cross-seed agreement of attribution maps (streams one seed at a time).

    Per variant: mean pairwise cosine across the 8 seeds of (a) the full node x feature matrix,
    (b) the 23-feature sum vector, (c) the per-node |attribution| mass vector; plus per-seed
    feature sums F_seed [M, N, 23]. Mean pairwise cosine = (||sum_s u_s||^2 - M) / (M (M - 1)).
    """
    t0 = time.time()
    f = _open(ds)
    no, nn, N = _offsets(f)
    var_node = np.repeat(np.arange(N), nn)
    D = f["node_per_seed"][cfg.ATTR_KEYS[key]]
    M = D.shape[0]
    U_full = np.zeros(D.shape[1:], dtype=np.float32)
    U_node = np.zeros(D.shape[1], dtype=np.float32)
    F_seed = np.zeros((M, N, D.shape[2]), dtype=np.float32)
    for s in range(M):
        A = D[s].astype(np.float32)
        nrm = np.sqrt(_segsum((A.astype(np.float64) ** 2).sum(1), no)) + 1e-12
        U_full += A / nrm[var_node][:, None].astype(np.float32)
        mass = np.abs(A).sum(1)
        nm_ = np.sqrt(_segsum(mass.astype(np.float64) ** 2, no)) + 1e-12
        U_node += mass / nm_[var_node].astype(np.float32)
        F_seed[s] = _segsum(A.astype(np.float64), no)
        del A
    cos_full = (_segsum((U_full.astype(np.float64) ** 2).sum(1), no) - M) / (M * (M - 1))
    cos_node = (_segsum(U_node.astype(np.float64) ** 2, no) - M) / (M * (M - 1))
    Fn = F_seed / (np.linalg.norm(F_seed, axis=2, keepdims=True) + 1e-12)
    cos_feat = ((Fn.sum(0) ** 2).sum(1) - M) / (M * (M - 1))
    # rank agreement of the 23-feature vectors across seeds (mean pairwise Spearman)
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 2, F_seed)
    R = (R - R.mean(2, keepdims=True)) / (R.std(2, keepdims=True) + 1e-12)
    rs = [np.mean(R[a] * R[b], axis=1) for a in range(M) for b in range(a + 1, M)]
    rank_feat = np.mean(rs, axis=0)
    np.savez_compressed(_outdir(ds) / f"seeds_{key}.npz", cos_full=cos_full, cos_node=cos_node,
                        cos_feat=cos_feat, rank_feat=rank_feat, F_seed=F_seed)
    print(f"[seeds] {ds}/{key}: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(cfg.DATASETS))
    ap.add_argument("--stage", required=True, choices=["scalars", "geom", "attr", "merge", "seeds", "all"])
    ap.add_argument("--key", choices=list(cfg.ATTR_KEYS), help="attr stage: which attribution array")
    a = ap.parse_args()
    if a.stage in ("scalars", "all"):
        stage_scalars(a.dataset)
    if a.stage in ("geom", "all"):
        stage_geom(a.dataset)
    if a.stage == "attr":
        stage_attr(a.dataset, a.key)
    if a.stage == "all":
        for k in cfg.ATTR_KEYS:
            stage_attr(a.dataset, k)
    if a.stage in ("merge", "all"):
        stage_merge(a.dataset)
    if a.stage == "seeds":
        stage_seeds(a.dataset, a.key)
