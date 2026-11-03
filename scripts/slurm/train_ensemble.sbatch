#!/bin/bash
#SBATCH --job-name=edlgnn_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH -t 48:00:00
# Train the 8-member ensemble (seeds 42-49), 4 GPUs x 2 rounds. Defaults of `guesser train`
# are the settings of the released model.
set -euo pipefail
TRAIN=${TRAIN:-graphs/train_r9.pt}; VAL=${VAL:-graphs/val_r9.pt}; OUT=${OUT:-runs/guesser}
mkdir -p "$OUT"
run() { CUDA_VISIBLE_DEVICES=$1 guesser train --train "$TRAIN" --val "$VAL" --seed "$2" \
            --out-dir "$OUT" --name "guesser_S$2" > "$OUT/train_S$2.log" 2>&1; }
for round in "42 43 44 45" "46 47 48 49"; do
    i=0; for s in $round; do run $i $s & i=$((i+1)); done; wait
done
