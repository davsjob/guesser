"""Manuscript Fig. 2 (figure_1_nature_performance) and Fig. 4 (figure_2_uncertainty_calibration).

s* = sqrt(MSE / mean(sigma_e^2)) on the MegaScale validation set (forward, ensemble mean), with
sigma_e = sigma_intra = sqrt(mean over members of beta / (v (alpha - 1))). The paper used s* = 1.708.
Run with the working directory set to results/figures (run_all.sh does this)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np

import nature_figures as nf

val = nf.load_benchmark_data("mega_val", split="All", base_path=nf.PRED)
nf.S_STAR = float(np.sqrt(np.mean(val["error"] ** 2) / np.mean(val["sigma_edl_epi"] ** 2)))
print(f"s* (validation, sigma_intra) = {nf.S_STAR:.3f}")
df_test = nf.load_benchmark_data("mega_test", split="All", base_path=nf.PRED)
benchmarks = nf.load_all_canonical_benchmarks(nf.PRED)
nf.generate_nature_figure_1(df_test, benchmarks, out_dir=".")
nf.generate_figure_2(out_dir=".")
