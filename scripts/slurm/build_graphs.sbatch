#!/bin/bash
#SBATCH --job-name=edlgnn_graphs
#SBATCH --gpus-per-node=1
#SBATCH -t 24:00:00
# Build r = 9 A graph pairs for one split, sharded with a job array:
#   sbatch --array=0-31 scripts/slurm/build_graphs.sbatch train 32
# then merge the shards (in shard order) with scripts/merge_shards.py.
# Set DATA to the folder with the Megascale spreadsheets and PDBs (see README, "Data").
set -euo pipefail
SPLIT=$1; NS=$2; K=${SLURM_ARRAY_TASK_ID:-0}
DATA=${DATA:?set DATA=/path/to/data}
mkdir -p graphs/shards
guesser build-graphs \
    --table "$DATA/spreadsheets/Megascale-S-$SPLIT.xlsx" "$DATA/spreadsheets/Megascale-D-$SPLIT.xlsx" \
    --pdb-dir "$DATA/pdb" --radius 9 --edge-cutoff 9 \
    --shard-id "$K" --num-shards "$NS" --out "graphs/shards/${SPLIT}_shard${K}of${NS}.pt"
