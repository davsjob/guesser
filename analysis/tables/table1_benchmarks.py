"""Table 1 (GUESSER columns) and Table S3: accuracy of the eight-model ensemble on every benchmark.

Forward mutations; the prediction is the mean of the members' mu. For the six double-mutant deep
mutational scans only the Spearman correlation with the measured fitness is reported.
Writes results/tables/table1_benchmarks.csv and table_s3_dms.csv."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from paths import RESULTS, pred_files

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def ensemble(ds):
    seeds = [pd.read_csv(f) for f in pred_files(ds)]
    seeds = [d[d.direction == "Forward"].reset_index(drop=True) for d in seeds]
    e = seeds[0][["pdb", "mutation", "true_ddg", "num_muts"]].copy()
    for d in seeds[1:]:
        assert (d.mutation.values == e.mutation.values).all()
    e["mu"] = np.mean([d.pred_ddg.values for d in seeds], axis=0)
    return e.dropna(subset=["true_ddg"])


def row(name, d):
    return dict(dataset=name, n=len(d), r=pearsonr(d.true_ddg, d.mu)[0], rho=spearmanr(d.true_ddg, d.mu)[0],
                rmse=float(np.sqrt(((d.mu - d.true_ddg) ** 2).mean())))


mega = ensemble("mega_test")
rows = [row("MegaScale-S", mega[mega.num_muts == 1]),
        row("FireProt-HF", ensemble("fireprot_hf")),
        row("SSYM", ensemble("ssym")),
        row("S669", ensemble("s669")),
        row("MegaScale-D", mega[mega.num_muts == 2]),
        row("PTMUL-D", ensemble("ptmul_d"))]

DMS = [("avGFP", "dms_avgfp"), ("cgreGFP", "dms_cgregfp"), ("ppluGFP2", "dms_pplugfp2"),
       ("amacGFP", "dms_amacgfp"), ("HIS3", "dms_his3"), ("KRAS", "dms_kras")]
dms = pd.DataFrame([dict(assay=a, n=len(d), rho=spearmanr(d.true_ddg, d.mu)[0]) for a, d in ((a, ensemble(k)) for a, k in DMS)])
dms.loc[len(dms)] = dict(assay="Mean", n=int(dms.n.sum()), rho=dms.rho.mean())
rows.append(dict(dataset="DMS, six assays", n=int(dms.n.iloc[-1]), r=np.nan, rho=dms.rho.iloc[-1], rmse=np.nan))

t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "table1_benchmarks.csv", index=False, float_format="%.4f")
dms.to_csv(OUT / "table_s3_dms.csv", index=False, float_format="%.4f")
print(t1.round(2).to_string(index=False))
print(dms.round(2).to_string(index=False))
