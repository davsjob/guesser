#!/usr/bin/env python
"""
Assemble the data folder that the analysis code (and the Zenodo deposit) uses, from the
research folders of the paper's runs. Run once by the authors; users download the result.

    python scripts/export_release_data.py --preds clean_hero_preds --expl explanations_v2_clean_hero \
        --singles megascale_singles_lookup.json --pdb_meta pdb_metadata_cache.json \
        --megascale_pdb Megascale/pdb --out data [--link]

Layout written (GUESSER_DATA):
    predictions/<dataset>_predictions_S<seed>.csv   per-member predictions, Forward + Reverse rows
    explanations/<dataset>_explanations.h5          ensemble IG explanations (+ _summary.csv)
    reference/megascale_singles_lookup.json         measured single-mutant ddG (non-additivity)
    reference/pdb_metadata_cache.json               experimental method / resolution per PDB
    reference/pdb_checks/{1pse,1tuc,2k5h}.pdb       wwPDB entries for the DSSP check and the 2K5H render
    pdb/megascale/*.pdb                              MegaScale structures (distances, DSSP, pLDDT)
--link makes symlinks instead of copies (for a local check).
"""
import argparse
import glob
import os
import re
import shutil


def put(src, dst, link):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    if link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="folder with seed_<S>/preds/<ds>_predictions_<tag>_S<S>.csv")
    ap.add_argument("--expl", required=True, help="folder with s_std_a_<ds>_raw_data.h5 and _summary.csv")
    ap.add_argument("--singles", required=True)
    ap.add_argument("--pdb_meta", required=True)
    ap.add_argument("--megascale_pdb", required=True)
    ap.add_argument("--pdb_checks", nargs="+", default=[], help="wwPDB files with HELIX/SHEET records (1pse, 1tuc, 2k5h)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()

    n = 0
    for f in sorted(glob.glob(os.path.join(a.preds, "seed_*", "preds", "*_predictions_*_S*.csv"))):
        m = re.match(r"(.+)_predictions_.+_S(\d+)\.csv$", os.path.basename(f))
        put(f, os.path.join(a.out, "predictions", f"{m.group(1)}_predictions_S{m.group(2)}.csv"), a.link); n += 1
    print(f"{n} prediction files")
    for f in sorted(glob.glob(os.path.join(a.expl, "s_std_a_*_raw_data.h5"))):
        ds = os.path.basename(f)[len("s_std_a_"):-len("_raw_data.h5")]
        put(f, os.path.join(a.out, "explanations", f"{ds}_explanations.h5"), a.link)
        s = f.replace("_raw_data.h5", "_summary.csv")
        if os.path.exists(s):
            put(s, os.path.join(a.out, "explanations", f"{ds}_explanations_summary.csv"), a.link)
        print("explanations:", ds)
    put(a.singles, os.path.join(a.out, "reference", "megascale_singles_lookup.json"), a.link)
    put(a.pdb_meta, os.path.join(a.out, "reference", "pdb_metadata_cache.json"), a.link)
    pdbs = sorted(glob.glob(os.path.join(a.megascale_pdb, "*.pdb")))
    for p in pdbs:
        put(p, os.path.join(a.out, "pdb", "megascale", os.path.basename(p)), a.link)
    print(f"{len(pdbs)} MegaScale PDBs")
    for p in a.pdb_checks:
        put(p, os.path.join(a.out, "reference", "pdb_checks", os.path.basename(p).lower()), a.link)


if __name__ == "__main__":
    main()
