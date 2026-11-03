#!/usr/bin/env python
"""Concatenate graph shards built with --shard-id k --num-shards n.

    python scripts/merge_shards.py graphs/shards/train_shard{}of32.pt 32 graphs/train_r9.pt

The result is sorted by (PDB, table row), the order an unsharded build produces.
"""
import sys

import torch

pattern, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
items = []
for k in range(n):
    items += torch.load(pattern.format(k), weights_only=False)
items.sort(key=lambda it: (it["pdb"], it["row"]))
torch.save(items, out)
print(f"{len(items)} graph pairs -> {out}")
