#!/usr/bin/env bash
# Regenerate every analysis table, key-number JSON and manuscript figure from the data folder.
#   GUESSER_DATA=/path/to/data bash analysis/run_all.sh        (about 1 h on a laptop; the MegaScale H5 is 6.7 GB)
# Outputs go to $GUESSER_RESULTS (default: results/). Needs numpy, pandas, scipy, h5py, matplotlib,
# seaborn, scikit-learn; the structure render of Fig. 8b also uses PyMOL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python}
RES=$($PY -c "import sys; sys.path.insert(0, '$HERE'); import paths; print(paths.RESULTS)")
step() { echo; echo "=== $*"; }

step "1. attribution: extract per-variant arrays from the explanation files, then tables and figures"
cd "$HERE/attribution"
if [ "${SKIP_EXTRACT:-0}" != 1 ]; then   # SKIP_EXTRACT=1 reuses results/cache from an earlier run
for d in megascale_test s669 ssym ptmul fphf; do $PY extract.py --dataset "$d" --stage all; done
# per-seed attributions: seed stability (all sets) and the uncertainty deep dive (MegaScale)
for d in megascale_test s669 ssym ptmul fphf; do
    for k in mut_mu ctx_mu; do $PY extract.py --dataset "$d" --stage seeds --key "$k"; done
done
$PY extract.py --dataset megascale_test --stage seeds --key mut_epistemic
fi
$PY analyses.py --all
$PY figures.py --all

step "2. attribution deep dives"
for s in deepdive_feature_direction deepdive_attribution_mechanisms deepdive_composition_bias \
         deepdive_uncertainty_attribution deepdive_structure_checks deepdive_epistasis_route \
         deepdive_reverse_antisymmetry deepdive_uq_structure deepdive_protein_flagging \
         protein_flagging_intra_pooled protein_flagging_intra_thresholds \
         epi_pair_distances deepdive_epistasis_distance deepdive_doubles_error_distance \
         fig_epistasis_3panel fig_epistasis_nature; do
    echo "--- $s"; $PY "$s.py"
done

step "3. audits: attribution-section checks, selective prediction, uncertainty measures (Table 2)"
cd "$HERE/audits/verify_attribution_section"
for s in part_a_mu part_a2_dssp part_b_sigma part_c_doubles_limits part_d_predbins part_e_table23 part_f_doubles_predbins; do
    echo "--- $s"; $PY "$s.py"
done
cd "$HERE/audits"
cd "$RES/audits"
for m in singles doubles plot; do $PY "$HERE/audits/normalized_selective_prediction.py" "$m"; done
for s in binned_selective_prediction uq_measures_benchmarks; do
    echo "--- $s"; $PY "$HERE/audits/$s.py"
done

step "4. manuscript figures (working directory: $RES/figures)"
cd "$RES/figures"
export PYTHONPATH="$HERE/figures${PYTHONPATH:+:$PYTHONPATH}"
F="$HERE/figures"
$PY "$F/build_uq_cache.py"
$PY "$F/uq_metrics.py"
$PY "$F/uq_numbers_intra.py"
mkdir -p fig1_parts; $PY "$F/fig1_extract_ig.py"
if $PY -c "import pymol" 2>/dev/null; then $PY "$F/fig1_render_structure.py"; else cp "$F/assets/2k5h_L74.png" "$F/assets/2k5h_L74_nodes.csv" fig1_parts/; fi
$PY "$F/fig1_overview_AB.py"                   # Fig. 1
$PY "$F/fig_performance_calibration.py"        # Fig. 2 and Fig. 4
# Fig. 3 figure_epistasis_nature was written in step 2 (attribution/fig_epistasis_nature.py)
$PY "$F/figS4_selective_prediction.py"         # Fig. 5
$PY "$F/figS5_protein_flagging.py"             # Fig. 7
$PY "$F/positional_csvs.py" ""                 # Fig. 8 (per-position data)
mkdir -p positional_parts; (cd positional_parts && $PY "$F/positional_parts/parts_a.py" && $PY "$F/positional_parts/parts.py")
if $PY -c "import pymol" 2>/dev/null; then $PY "$F/fig_positional_render.py"; else echo "PyMOL not installed: skipping the 2K5H render (Fig. 8b)"; fi
[ -f fig_positional_render_sigma.png ] && $PY "$F/assemble_per_position.py" || echo "skipping Fig. 8 assembly (no render)"
$PY "$F/fig_selection_data.py"                 # Fig. 6
$PY "$F/fig_selection_bars.py" data/figure_selection_numbers.csv .
$PY "$F/fig4_rules_trust.py"                   # Fig. 9

step "5. tables and the manuscript figures under their paper names"
$PY "$HERE/tables/table1_benchmarks.py"        # Table 1 and Table S3
P="$RES/paper"; mkdir -p "$P"
cp "$RES/tables/table1_benchmarks.csv" "$P/Table_1.csv"
cp "$RES/audits/uq_measures_benchmarks/table.csv" "$P/Table_2.csv"
cp "$RES/tables/table_s3_dms.csv" "$P/Table_S3.csv"
cp "$RES/attribution/deepdive_protein_flagging/intra_thresholds.json" "$P/Table_S4.json"
for pair in 1:fig1_overview_AB 2:figure_1_nature_performance 3:figure_epistasis_nature 4:figure_2_uncertainty_calibration \
            5:figS4_selective_prediction 6:figure_selection 7:figS5_protein_flagging 8:figure_per_position 9:fig4_rules_trust; do
    n=${pair%%:*}; f=${pair#*:}
    for e in pdf png; do [ -f "$RES/figures/$f.$e" ] && cp "$RES/figures/$f.$e" "$P/Figure_$n.$e"; done
done
echo; echo "done: results in $RES (manuscript figures and tables in $RES/paper)"
