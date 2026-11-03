#!/bin/bash
#SBATCH --job-name=edlgnn_explain
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
# IG explanations for one graph set with all 8 members, then the ensemble merge (+ HDF5).
#   sbatch scripts/slurm/explain.sbatch graphs/s669_r9.pt /path/to/s669_pdbs out/s669
set -euo pipefail
GRAPHS=$1; PDBS=$2; PREFIX=$3; W=${WEIGHTS:-weights}
mkdir -p "$(dirname "$PREFIX")"
run() { CUDA_VISIBLE_DEVICES=$1 guesser explain --dataset "$GRAPHS" --weight_file "$W/guesser_S$2.pt" --seed "$2" \
            --pdb_dirs "$PDBS" --max_nodes_per_pass 50000 --out_prefix "${PREFIX}_S$2" > "${PREFIX}_S$2.log" 2>&1; }
for round in "42 43 44 45" "46 47 48 49"; do
    i=0; for s in $round; do run $i $s & i=$((i+1)); done; wait
done
guesser merge-explanations --seed_files "${PREFIX}_S*_raw_data.pt" --out_prefix "$PREFIX" --save_h5
