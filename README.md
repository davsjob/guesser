# GUESSER

**G**raph network for **U**ncertainty-aware, **E**xplainable **S**tability **S**hifts from
**E**vidential **R**egression.

GUESSER predicts the change in protein stability (ΔΔG) caused by single and double mutations,
together with an uncertainty for each prediction, from interpretable physicochemical and
energetic descriptors of the local protein structure. The uncertainty and the prediction are
both explained with integrated gradients and can be mapped back onto the structure.

This repository has the code, the eight trained ensemble members, a CLI, a quickstart
notebook, and the analysis code behind the figures and tables of the paper:

> D. Sjöberg, L. Koenekoop, J. Åqvist, H. Gutiérrez-de-Terán. *Evidential Graph Neural
> Networks for Explainable and Uncertainty-Aware Thermostability Prediction in Protein
> Engineering.*

Sign convention: ΔΔG > 0 is destabilizing (kcal mol⁻¹).

## How it works

* **Local graphs.** Residues with atoms within 9 Å of the Cα of a mutated residue form the
  graph; residues whose Cα atoms are within 9 Å are connected by edges carrying distance and
  side-chain orientation. Each node has 23 features: 11 amino-acid descriptors, SASA and 11
  MadraX energy terms (Table S2). Wild-type and mutant graphs share the topology and differ
  only in node features. Double mutants with overlapping neighbourhoods form one graph;
  distant ones form two disconnected components.
* **Graph encoder.** For each residue, the wild-type features and the mutation-induced
  changes (MT − WT) are concatenated and encoded in one pass by four graph attention (GATv2)
  layers with residual connections. The resulting representation feeds a global head and a
  multi-mutation (epistasis) head, active only for two or more mutations, and the summed
  MadraX energy differences feed a linear physics prior: μ = γ_base + γ_epi + γ_phys.
* **Evidential regression.** The evidential head predicts the Normal-Inverse-Gamma parameters
  (γ, ν, α, β). Over the eight ensemble members, the prediction is μ = mean(γ), the epistemic
  uncertainty is σ_e = √(mean β/(ν(α−1))), the aleatoric uncertainty is
  σ_a = √(mean β/(α−1)), and 1/Φ = mean 1/(2ν+α) is the inverse total evidence. σ_e is
  calibrated with one scale factor fitted on the MegaScale validation set, s* = 1.71.
* **Integrated gradients.** The *mutation attribution* integrates the mutation-induced
  feature changes from a null mutation, with the wild-type environment fixed; the
  *environment attribution* integrates the wild-type features from the graph mean. Both are
  computed for μ and log σ_e, per residue and per feature, and averaged over the ensemble.

## Install

```bash
conda env create -f environment.yml   # Python 3.10 and the versions used for the paper
conda activate guesser
```

Or `pip install -e .` in an existing environment (torch ≥ 2.4, torch-geometric ≥ 2.6).

Building graphs from PDB files needs [MadraX](https://github.com/grogdrinker/madrax). We
recommend our fork, which adds faster pair search and fixes batch-dependent energies
(Linux x86-64, Python 3.7–3.12):

```bash
pip install -e ".[structures]"     # or: pip install "git+https://github.com/davsjob/madrax.git"
```

## Quickstart

```bash
# Ensemble prediction on the bundled example graphs
guesser predict --graphs examples/example_graphs.pt --out preds.csv

# Your own structure and mutations; A:L45F+A:G50A is one double mutant
guesser predict --pdb my_protein.pdb --mutations A:L45F A:G50A A:L45F+A:G50A --out preds.csv

# A mutation list over several structures (columns pdb, mutation, optional chain and ddg;
# see examples/mutations.csv); with measured ddg it also reports r, ρ and RMSE
guesser predict --csv mutations.csv --pdb-dir structures/ --out preds.csv
```

[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) shows ensemble prediction,
uncertainty and an integrated-gradients explanation of a double mutant, and runs on a CPU in
about a minute. From Python:

```python
import torch
from guesser.inference import predict
graphs = torch.load("examples/example_graphs.pt", weights_only=False)
ens = predict(graphs)            # the eight members in weights/
```

### Output columns

| column | meaning |
| --- | --- |
| `pred_ddg` | μ, the ensemble mean prediction (kcal mol⁻¹) |
| `pred_phys`, `pred_base`, `pred_epi` | γ_phys, γ_base, γ_epi (γ_epi is exactly 0 for single mutants) |
| `intra_epistemic_std` | σ_e, the epistemic uncertainty |
| `calibrated_std` | s*·σ_e with s* = 1.71; the predictive distribution is N(μ, (s*σ_e)²) |
| `aleatoric_std` | σ_a, the aleatoric uncertainty |
| `inv_phi` | 1/Φ, the inverse total evidence (unitless; for ranking only) |
| `inter_epistemic_std`, `epistemic_std`, `total_std` | spread of μ between members, and its combinations with σ_e and σ_a |

`--member-dir` also writes one file per ensemble member with the forward and reverse
predictions and the raw (ν, α, β).

## Training

`guesser train` reproduces one ensemble member with the settings of Table S1: four GATv2 layers
of width 256 with four heads, dropout 0.2, AdamW (learning rate 10⁻⁴, weight decay 10⁻⁴,
cosine annealing over 100 epochs), batch size 128, evidential regularizer weight 0.05
annealed from 0 over the first 30 epochs, antisymmetry weight λ_anti = 1.0, and early stopping
on validation r with patience 25. Each training variant is used forward and reversed by
swapping the wild-type and mutant graphs. The data are the MegaScale single and double
mutations with the ThermoMPNN-D splits; the released members used seeds 42–49.

```bash
# graphs (needs MadraX); a sharded version is in scripts/slurm/build_graphs.sbatch
guesser build-graphs --table Megascale-S-train.xlsx Megascale-D-train.xlsx --pdb-dir pdb/ --out graphs/train.pt
guesser build-graphs --table Megascale-S-val.xlsx Megascale-D-val.xlsx --pdb-dir pdb/ --out graphs/val.pt

# one ensemble member (scripts/slurm/train_ensemble.sbatch trains all eight)
guesser train --train graphs/train.pt --val graphs/val.pt --seed 42 --out-dir runs/ --name guesser_S42

# predictions and integrated-gradients explanations with your members
guesser predict --graphs graphs/s669.pt --weights runs/guesser_S4*_best_weights.pt --out s669.csv
guesser explain --dataset graphs/s669.pt --weight_file weights/guesser_S42.pt --seed 42 --pdb_dirs s669_pdbs/ --out_prefix out/s669_S42
guesser merge-explanations --seed_files "out/s669_S*_raw_data.pt" --out_prefix out/s669 --save_h5
```

The ablations of Table S5 use the same command with one change each (seeds 42–45):

| Table S5 configuration | change |
| --- | --- |
| Singles only | `--mutation_type single` |
| MadraX features only | `--feature_set madrax_only` |
| No MadraX features | `--feature_set no_madrax` |
| No physics module | `--use_physics false` |
| No epistasis module | `--use_epistasis false` |
| EDL λ = 0.01, 0.1, 1.0 | `--edl_lambda_reg 0.01` (etc.) |
| Radius 7, 11, 13 Å | graphs built with `guesser build-graphs --radius 7` (etc.) |

## Reproducing the figures and tables

Download the data folder from Zenodo (DOI: TODO): the per-member predictions on every
benchmark, the integrated-gradients explanation files, two reference files and the MegaScale
structures (layout in [`analysis/paths.py`](analysis/paths.py)). Then:

```bash
GUESSER_DATA=/path/to/data bash analysis/run_all.sh
```

This writes every figure and table to `results/paper/` under the manuscript's numbering
(`Figure_1` … `Figure_9`, `Table_1`, `Table_2`, `Table_S3`, `Table_S4`), with the
intermediate tables and key numbers of each analysis in `results/`.

| in the paper | script |
| --- | --- |
| Fig. 1 Overview of GUESSER | `analysis/figures/fig1_overview_AB.py` |
| Fig. 2 Performance on the MegaScale test sets and across benchmarks | `analysis/figures/fig_performance_calibration.py` |
| Fig. 3 Non-additivity of double mutants | `analysis/attribution/fig_epistasis_3panel.py`, `fig_epistasis_nature.py` |
| Fig. 4 Calibration and error ranking of the uncertainty | `analysis/figures/fig_performance_calibration.py` |
| Fig. 5 Error rejection among predictions of similar size | `analysis/audits/binned_selective_prediction.py`, `analysis/figures/figS4_selective_prediction.py` |
| Fig. 6 Choosing predicted neutral and stabilizing variants | `analysis/figures/fig_selection_data.py`, `fig_selection_bars.py` |
| Fig. 7 Protein-level uncertainty | `analysis/attribution/deepdive_protein_flagging.py`, `analysis/figures/figS5_protein_flagging.py` |
| Fig. 8 Error-prone positions within a protein | `analysis/figures/positional_csvs.py`, `positional_parts/`, `fig_positional_render.py`, `assemble_per_position.py` |
| Fig. 9 Burial-dependent rules and their uncertainty | `analysis/figures/fig4_rules_trust.py` |
| Table 1, Table S3 | `analysis/tables/table1_benchmarks.py` |
| Table 2 | `analysis/audits/uq_measures_benchmarks.py` |
| Table S4 | `analysis/attribution/protein_flagging_intra_thresholds.py` |
| Attribution results (Section 3.7) | `analysis/attribution/` and `analysis/audits/verify_attribution_section/` |

See [`analysis/README.md`](analysis/README.md) for the individual analyses. A released
explanation file can be read on its own with `guesser.explain.h5.ExplanationFile` (numpy and
h5py only); `ef.readme` describes every field.

## Repository layout

```
guesser/          the model package
  model.py        GUESSER
  structure.py    neighbourhoods, Cα graph, SASA, MadraX
  graphs.py       graph construction, mutation parsing
  features.py     the 23 node features
  datasets.py     graph pairs, forward/reverse augmentation
  training.py     evidential loss with antisymmetry, train()
  inference.py    ensemble prediction and uncertainties
  explain/        integrated gradients: per-member run, ensemble merge, HDF5 format
  cli.py          the guesser command
analysis/         figures and tables of the paper
weights/          guesser_S42.pt … guesser_S49.pt, the eight ensemble members
examples/         41 bundled graph pairs with reference outputs
notebooks/        quickstart.ipynb
scripts/          data-folder export, shard merging, SLURM examples
tests/            pytest suite
```

## Citation

See `CITATION.cff`.

## License and third-party data

The code is under the MIT license (`LICENSE`). MadraX is a separate dependency under the
Apache 2.0 license. The training and benchmark data (MegaScale, FireProt-HF, SSYM, S669, PTMUL-D and the
ProteinGym deep mutational scans) belong to their original authors; please cite them when
you use the deposited data.
