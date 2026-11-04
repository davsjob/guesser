# Paper analysis

This folder holds every script behind the manuscript's figures and quoted numbers. Each script
reads the data folder (`GUESSER_DATA`) and writes to `results/` (`GUESSER_RESULTS`). Both
locations are set in [`paths.py`](paths.py). [`run_all.sh`](run_all.sh) runs everything in
dependency order.

```bash
GUESSER_DATA=/path/to/data bash analysis/run_all.sh
```

Throughout the paper, sigma_e = sigma_intra = sqrt(mean over the 8 members of beta / (v (alpha − 1))),
in the column `intra_epistemic_std` (or `sigma_intra`). The validation scale factor is s* = 1.708.
Only forward mutations are used unless a script says otherwise.

## attribution/

This is the analysis of the Integrated-Gradients explanations. It produces the attribution
section of the paper, Fig. 3 and the inputs of Figs. 6 and 9.

| script | role |
| --- | --- |
| `config.py` | feature names, tier/burial/shell definitions, palette (paths from `paths.py`) |
| `extract.py` | one pass over each explanation H5 → `results/cache/<ds>/` (per-variant table, geometry, attribution sums) |
| `analyses.py` | A0 sanity, A1 driving forces, A2 environment, A3 localisation, A4 epistasis, A5 uncertainty → `results/attribution/<ds>/` |
| `figures.py` | per-dataset attribution figures and the cross-dataset comparison |
| `deepdive_*.py` | the deep dives quoted in the text: feature direction, substitution-class mechanisms, composition bias, uncertainty attribution, structure checks (DSSP, pLDDT), epistasis route, reverse/antisymmetry, structure quality, protein flagging, epistasis vs distance |
| `epi_pair_distances.py` | side-chain-aware distances of MegaScale double-mutant site pairs |
| `fig_epistasis_3panel.py`, `fig_epistasis_nature.py` | Fig. 3 |

The definitions used in these scripts:

- Mutation IG (`signed_attr_<t>`) integrates x_diff from 0 to x_mt − x_wt, with the WT
  features fixed, and sums to μ − μ_null.
- Environment IG (`signed_attr_<t>_wt`) integrates x_wt from the per-graph node mean to x_wt,
  with x_diff fixed.
- Burial is relative SASA with the Tien et al. 2013 maximum ASA: core < 0.20, boundary
  0.20–0.50, surface ≥ 0.50.
- CIs are 95% protein-cluster bootstraps.
- Non-additivity is E = ΔΔG_AB − (ΔΔG_A + ΔΔG_B), with the singles taken from the MegaScale
  lookup for E_exp and from the model's own single-mutant predictions for E_calc.

## audits/

| script | role |
| --- | --- |
| `verify_attribution_section/part_*.py` | independent recomputation of every number in the attribution section |
| `binned_selective_prediction.py` | selective prediction globally and within predicted-ddG bins (Fig. 5) |
| `normalized_selective_prediction.py` | the same with bin-normalised uncertainty |
| `uq_measures_benchmarks.py` | AUCO and rank correlations of each uncertainty measure on every benchmark |
| `triage_intra.py`, `underprediction_check.py` | numbers for the triage and underprediction statements |

## figures/

These scripts run with `results/figures/` as the working directory, which `run_all.sh` sets.
`build_uq_cache.py` first writes the per-variant ensemble tables in `data/`. The other scripts
make one manuscript figure each; see the table in the top-level README.
`fig_performance_calibration.py` draws Figs. 2 and 4 with the paper's plotting-notebook code,
which is in `nature_figures.py`. The structure renders (`fig1_render_structure.py`, `fig_positional_render.py`) use
`pymol-open-source`.
