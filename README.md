# GUESSER

GUESSER is an explainable, uncertainty-aware graph neural network for the change in protein
folding stability (ddG) caused by single and multiple point mutations.

* **Siamese graph model.** It compares the wild-type and mutant residue neighbourhoods. The
  prediction is split into three additive parts: a linear physics prior on MadraX energy
  differences, a structural GNN head, and an epistasis head for multi-mutants.
* **Evidential uncertainty.** A Normal-Inverse-Gamma head gives aleatoric and epistemic
  uncertainty from one forward pass. Eight independently seeded members form the released
  ensemble.
* **Integrated-Gradients explanations.** Each prediction is attributed to features and
  residues, with exact completeness bookkeeping.
* **Thermodynamic antisymmetry.** Training includes the reverse mutation and penalises
  ddG(fwd) + ddG(rev).

This repository has the code that trained the released model, the 8 checkpoints, a CLI, a
quickstart notebook, and the analysis code behind every figure and number in the paper.

> Sign convention: ddG > 0 means the mutation is destabilising (kcal/mol).

## Install

```bash
conda env create -f environment.yml   # Python 3.10 + the versions used for the paper
conda activate guesser
```

To use an existing environment instead, run `pip install -e .` (torch ≥ 2.4 and torch-geometric ≥ 2.6).

**MadraX** is needed only to build graphs from PDB files. It is not needed for prediction on
prebuilt graphs, training, or explanations. Install it from its upstream repository. The
paper used commit `bd247bd`:

```bash
pip install "git+https://bitbucket.org/grogdrinker/madrax.git@bd247bdabd26c68c578c6da18b5c939e6bf847bf"
```

MadraX is also developed at [github.com/grogdrinker/madrax](https://github.com/grogdrinker/madrax),
where it ships compiled energy modules for specific platforms and Python versions. Python 3.10
on x86-64 Linux is the safe choice.

## Quickstart

```bash
# Ensemble prediction on the 41 bundled example graphs (no MadraX needed)
guesser predict --graphs examples/example_graphs.pt --out preds.csv

# Your own structure and mutations (needs MadraX); A:L45F+A:G50A is one double mutant
guesser predict --pdb my_protein.pdb --mutations A:L45F A:G50A A:L45F+A:G50A --out preds.csv
```

[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) runs on a CPU in about a minute. It
shows ensemble prediction, uncertainty, the antisymmetry check, and an Integrated-Gradients
explanation of one double mutant.

From Python:

```python
import torch
from guesser.inference import predict
graphs = torch.load("examples/example_graphs.pt", weights_only=False)
ens = predict(graphs)            # the 8 released members in weights/
```

### Output columns (ensemble CSV)

| column | meaning |
| --- | --- |
| `pred_ddg` | ensemble mean of mu = `pred_phys` + `pred_base` + `pred_epi` |
| `pred_phys`, `pred_base`, `pred_epi` | physics prior, structural head, epistasis head (exactly 0 for single mutants) |
| `intra_epistemic_std` | sqrt(mean over members of beta / (v (alpha - 1))); this is sigma_e in the paper |
| `inter_epistemic_std` | standard deviation of mu across members (ddof = 1) |
| `epistemic_std` | sqrt(intra² + inter²) |
| `aleatoric_std` | sqrt(mean over members of beta / (alpha - 1)) |
| `total_std` | sqrt(aleatoric² + epistemic²) |

`--member-dir` also writes one CSV per member with forward and reverse rows and the raw NIG
parameters (`v`, `alpha`, `beta`). These files have the same format as the paper's
benchmark files.

## Model

`guesser.model.GUESSER` (hidden 256, 4 layers, about 2.16 M parameters):

* **Graphs.** Nodes are the residues with any atom within 9 Å of a mutated residue's CA.
  Edges connect pairs with CA–CA ≤ 9 Å, with features [cos(CA→CB_i, CA→CB_j), distance]. Each
  node has 23 features: 11 amino-acid properties, SASA, and 11 MadraX per-residue energy
  terms. The WT and MT graphs share topology and differ only in node features. MadraX runs in
  "isolated" mode: one wild-type call per structure and one call per mutant, so features
  don't depend on batching.
* **Backbone.** Four damped, pre-norm residual blocks of GATv2 plus an RBF distance-message
  branch, applied to [x_wt, x_mt − x_wt].
* **Heads.** gamma_phys is linear in the summed MadraX differences. gamma_base comes from the
  mean pool plus the mutated-site pool. gamma_epi comes from attentional pooling over
  mutated nodes, hard-gated to 0 for single mutants. The evidential head (v, alpha, beta)
  sees the normalised structural, epistatic and physics latents.
* **Training.** Megascale (Tsuboyama et al. 2023) S + D splits as in ThermoMPNN, with reverse
  augmentation. The loss is NIG NLL plus 0.05·|y − mu|·(2v + alpha), with the regulariser
  ramped in over 30 epochs, plus an antisymmetry penalty of weight 1.0. The optimiser is
  AdamW (lr 1e-4, weight decay 1e-4) with a cosine schedule, batch size 128 and 100 epochs.
  The checkpoint with the best validation Pearson r is kept. Seeds are 42–49.

All of these settings are the defaults of `guesser train` and `guesser.training.TrainConfig`.

## Reproducing the paper's figures and numbers

Download the data folder from Zenodo (DOI: TODO). It holds the per-member predictions, the
explanation HDF5 files, two reference files and the MegaScale structures (layout in
[`analysis/paths.py`](analysis/paths.py)). Then run:

```bash
GUESSER_DATA=/path/to/data bash analysis/run_all.sh     # writes results/
```

| manuscript figure | output in `results/figures/` | script |
| --- | --- | --- |
| Fig. 1 workflow overview | `fig1_overview_AB` | `analysis/figures/fig1_overview_AB.py` |
| Fig. 2 predictive performance | `figure_1_nature_performance` | `analysis/figures/fig_performance_calibration.py` |
| Fig. 3 epistasis | `figure_epistasis_nature` | `analysis/attribution/fig_epistasis_3panel.py`, `fig_epistasis_nature.py` |
| Fig. 4 uncertainty calibration | `figure_2_uncertainty_calibration` | `analysis/figures/fig_performance_calibration.py` |
| Fig. 5 selective prediction | `figS4_selective_prediction` | `analysis/audits/binned_selective_prediction.py`, `analysis/figures/figS4_selective_prediction.py` |
| Fig. 6 protein flagging | `figS5_protein_flagging` | `analysis/attribution/deepdive_protein_flagging.py`, `analysis/figures/figS5_protein_flagging.py` |
| Fig. 7 per-position uncertainty | `figure_per_position` | `analysis/figures/positional_csvs.py`, `positional_parts/`, `fig_positional_render.py` (PyMOL), `assemble_per_position.py` |
| Fig. 8 selection | `figure_selection` | `analysis/figures/fig_selection_data.py`, `fig_selection_bars.py` |
| Fig. 9 attribution rules and trust | `fig4_rules_trust` | `analysis/figures/fig4_rules_trust.py` |
| SI: selection curves, triage | `figure_selection_curves`, `figure_3_PPV_improvement` | `fig_selection.py`, `triage_hero.py` |

Numbers quoted in the text come from the key-number JSONs and tables in `results/attribution/`
(attribution analyses and deep dives) and `results/audits/` (attribution-section checks,
selective prediction, UQ measures, triage, underprediction). See [`analysis/README.md`](analysis/README.md).

## Rerunning the pipeline from structures

```bash
# 1. graphs (needs MadraX; GPU recommended); sharded example in scripts/slurm/build_graphs.sbatch
guesser build-graphs --table Megascale-S-train.xlsx Megascale-D-train.xlsx --pdb-dir pdb/ --out graphs/train_r9.pt
guesser build-graphs --table Megascale-S-val.xlsx Megascale-D-val.xlsx --pdb-dir pdb/ --out graphs/val_r9.pt

# 2. one member per seed (scripts/slurm/train_ensemble.sbatch runs all 8)
guesser train --train graphs/train_r9.pt --val graphs/val_r9.pt --seed 42 --out-dir runs/ --name guesser_S42

# 3. predictions with your members (or the released ones by default)
guesser predict --graphs graphs/s669_r9.pt --weights runs/guesser_S4*_best_weights.pt --out s669.csv --member-dir s669_members/

# 4. explanations: one run per member, then the ensemble merge (+ self-contained HDF5)
guesser explain --dataset graphs/s669_r9.pt --weight_file weights/guesser_S42.pt --seed 42 --pdb_dirs s669_pdbs/ --out_prefix out/s669_S42
guesser merge-explanations --seed_files "out/s669_S*_raw_data.pt" --out_prefix out/s669 --save_h5
```

Read a released explanation file with `guesser.explain.h5.ExplanationFile`. That module needs
only numpy and h5py, so it can also be copied on its own. `ef.readme` describes every field.

### What has been checked against the original training code

The released code is a cleaned-up copy of the research code, with prototype names replaced,
dead options removed and no hard-coded paths. The checks in this table were run when it
was extracted:

| component | check | result |
| --- | --- | --- |
| model | same init for a given seed, same outputs, all 7 architecture variants | bit-identical |
| checkpoints | all 8 load with `strict=True` into the new class | OK |
| training loop | 4 epochs on the same graphs and seed, single-threaded CPU | weights and epoch logs bit-identical |
| inference + ensemble | the paper's graphs (rebuilt from the explanation files) vs the saved per-seed CSVs for Ssym, Ssym-rev, S669, PTmul, FireProt-HF and Megascale-test | max abs. difference 2e-5 (float32, GPU vs CPU) |
| IG explainer | all 17 attribution arrays vs the released HDF5 (Ssym, seed 42) | bit-identical |
| paper analysis | `analysis/run_all.sh` on the exported data vs the outputs of the paper's final runs: 176 tables and key-number files | 169 identical; 7 differ in the 3rd decimal or less (grouped-CV AUCs, two bootstrap CIs, two AUROCs) |
| manuscript figures | regenerated vs the submitted PNGs | 7 of 10 pixel-identical; Fig. 1, Fig. 5 and the SI selection curves differ by a few pixels (Fig. 1's schematic glyph now uses the released explanations); Fig. 7b needs PyMOL |

`pytest tests/` re-runs a small version of these checks against reference values from the
released runs.

Notes on exact reproduction:

* **Multithreading.** On CPU and GPU, PyG scatter operations use atomics, so repeated
  training runs differ in the last bits unless run single-threaded or deterministically.
  The released weights come from GH200 GPUs.
* **Graph order.** The training data order affects shuffling. `build-graphs` processes
  structures sorted by PDB name and keeps table order within a structure.
  `scripts/merge_shards.py` restores that order after a sharded build. Bit-exact retraining
  needs the graph caches deposited on Zenodo.
* **MadraX numerics.** MadraX energies can differ slightly between devices.
* **Coordinate noise.** `coord_noise_sigma = 0.2` is configured as in the original runs.
  The local graphs carry no CA coordinates, so this augmentation had no effect on the
  released model. It is kept so that random number use matches.

## Repository layout

```
guesser/
  model.py        GUESSER (+ GUESSER_MODEL_KWARGS, load_model)
  structure.py    neighbourhoods, CA graph, SASA, MadraX wrappers
  graphs.py       build_graphs (isolated MadraX), mutation parsing
  features.py     node feature definitions, feature-set ablations
  datasets.py     graph-pair datasets, reverse augmentation, collate
  training.py     TrainConfig, evidential loss, train()
  inference.py    predict(), per-member and ensemble outputs, metrics
  explain/        Integrated Gradients: core, residue mapping, per-seed run, merge, HDF5 format
  cli.py          the guesser command
analysis/         paper analysis: paths.py, run_all.sh, attribution/, audits/, figures/
weights/          guesser_S42.pt ... guesser_S49.pt (released ensemble, state dicts)
examples/         41 bundled graph pairs + reference outputs from the released runs
notebooks/        quickstart.ipynb
scripts/          export_release_data.py (builds the data folder), merge_shards.py, SLURM examples
tests/            pytest suite
```

Names in the research code, for readers of the paper's supplementary material:

| here | research code |
| --- | --- |
| `GUESSER` | `SiameseEvidentialResGAT_Ultimate` (`models_30_07.py`) |
| `DampedResidualGATBlock`, `RBFExpansion`, `EpistasisPooling` | `ResRBFGATBlock_Damped`, `RBFLayer`, `SetTransformerEpistasis` |
| `guesser.training.train` | `training_29_07.train_model` (evidential path) via `main_30_07.py` |
| `guesser.graphs.build_graphs` | `mutations_multiradius.process_mutations_multiradius(madrax_mode="isolated")` |
| `guesser.inference` | `benchmark_mega_predictions_edl_30_07.py` + ensemble aggregation |
| `guesser.explain.*` | `explain_s_std_a.py`, `explain_s_std_a_core.py`, `merge_explain_seeds.py`, `s_std_a_explain_h5.py` |
| released GUESSER ensemble | run `clean_hero` ("hero"), seeds S42–S49; internal model name S_Std_A |

## Citation

See `CITATION.cff`. The article reference will be added on publication.

## License and third-party data

The code is under the MIT license (`LICENSE`). MadraX is a separate dependency under its own
license and is not redistributed here. The training and benchmark data (Megascale, S669,
Ssym, FireProt, PTmul, ProteinGym) belong to their original authors; please cite them when
you use the deposited graphs.
