"""
Where the analysis reads data and writes results. Every analysis script imports this.

    GUESSER_DATA     the downloaded data folder (default: <repo>/data), layout:
        predictions/<dataset>_predictions_S<seed>.csv   per-member predictions (8 seeds, Forward + Reverse rows)
        explanations/<dataset>_explanations.h5          ensemble Integrated-Gradients explanations
        reference/megascale_singles_lookup.json         measured single-mutant ddG (for non-additivity)
        reference/pdb_metadata_cache.json               experimental method and resolution per PDB
        pdb/megascale/*.pdb                              MegaScale structures
    GUESSER_RESULTS  output folder (default: <repo>/results)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GUESSER_DATA", REPO / "data")).resolve()
RESULTS = Path(os.environ.get("GUESSER_RESULTS", REPO / "results")).resolve()

PRED_DIR = DATA / "predictions"
EXPL_DIR = DATA / "explanations"
REF_DIR = DATA / "reference"
MEGASCALE_PDB_DIR = DATA / "pdb" / "megascale"

ATTRIBUTION_OUT = RESULTS / "attribution"       # attribution pipeline tables, figures, key numbers
AUDITS_OUT = RESULTS / "audits"                 # selective prediction, UQ measures, attribution checks
FIGURES_OUT = RESULTS / "figures"               # manuscript figures and their intermediate data
CACHE_DIR = RESULTS / "cache"                   # per-variant arrays extracted from the H5 files

for _d in (ATTRIBUTION_OUT, AUDITS_OUT, FIGURES_OUT, CACHE_DIR, FIGURES_OUT / "data"):
    _d.mkdir(parents=True, exist_ok=True)

PRED = str(PRED_DIR) + "/"   # for scripts that build glob patterns by string concatenation


def pred_files(dataset):
    """The 8 per-member prediction CSVs of a dataset, sorted by seed."""
    files = sorted(PRED_DIR.glob(f"{dataset}_predictions_S*.csv"))
    if not files:
        raise FileNotFoundError(f"no {dataset}_predictions_S*.csv in {PRED_DIR} (set GUESSER_DATA)")
    return files
