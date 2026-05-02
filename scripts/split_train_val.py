"""Split train_clean.json into train/val at the MULTI-HOP sample level.

Splitting before single-hop expansion ensures all hops of the same original
question stay in the same split, preventing train/val leakage (otherwise
hop 1 of question X could end up in train and hop 2 in val).

Usage:
    python scripts/split_train_val.py \
        --input      data/wiki_coe/train_clean.json \
        --train-out  data/wiki_coe/train_split.json \
        --val-out    data/wiki_coe/val_split.json \
        --val-frac   0.05 \
        --seed       42
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n_val = int(round(len(data) * args.val_frac))
    val_idx = set(indices[:n_val])

    train = [s for i, s in enumerate(data) if i not in val_idx]
    val = [s for i, s in enumerate(data) if i in val_idx]

    os.makedirs(os.path.dirname(args.train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.val_out) or ".", exist_ok=True)
    with open(args.train_out, "w") as f:
        json.dump(train, f)
    with open(args.val_out, "w") as f:
        json.dump(val, f)

    print(f"Input:  {args.input}  ({len(data)} samples)")
    print(f"Train:  {args.train_out}  ({len(train)} samples, {100*len(train)/len(data):.2f}%)")
    print(f"Val:    {args.val_out}  ({len(val)} samples, {100*len(val)/len(data):.2f}%)")
    print(f"Seed:   {args.seed}")


if __name__ == "__main__":
    main()
