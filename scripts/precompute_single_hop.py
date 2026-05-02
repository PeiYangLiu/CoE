"""Precompute single-hop expansion offline.

train_clean.json was produced by the offline bbox picker (`scripts/filter_oob_samples.py`
+ CoE picker pipeline) and already contains per-hop resolved data under the
`_resolved_hops` key. Runtime CoEDataset._expand_to_single_hop re-runs the
picker for each sample, hitting blob for 76k _bboxes.json files which hangs
blobfuse under 64-rank load.

This script flattens `_resolved_hops` into an already-expanded single-hop
dataset. CoEDataset detects the pre-expanded format (all num_hops==1) and
skips the runtime expansion entirely.

Usage:
    python scripts/precompute_single_hop.py \
        --input  data/wiki_coe/train_clean.json \
        --output data/wiki_coe/train_single_hop.json
"""
import argparse
import json
import os
import re


_RELATION_MAP = {
    "date of birth": "date of birth",
    "date of death": "date of death",
    "country of citizenship": "country of citizenship",
    "place of birth": "place of birth",
    "place of death": "place of death",
    "director": "director",
    "publication date": "publication date",
    "publisher": "publisher",
    "performer": "performer",
    "composer": "composer",
    "screenwriter": "screenwriter",
    "father": "father",
    "mother": "mother",
    "spouse": "spouse",
    "occupation": "occupation",
}


def _relation_from_q(question: str) -> str:
    m = re.match(r"what is the (.+?) of ", question.lower())
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    out = []
    dropped = 0
    for s in data:
        hops = s.get("_resolved_hops", [])
        if not hops:
            dropped += 1
            continue
        for h in hops:
            if not h.get("bbox") or not h.get("screenshot"):
                continue
            out.append({
                "question": h["question"],
                "answer": h["answer"],
                "evidence_value": h["answer"],
                "type": s.get("type", ""),
                "num_hops": 1,
                "evidence_chain": [{
                    "entity": h.get("entity", ""),
                    "screenshot": h["screenshot"],
                    "bboxes": [h["bbox"]],
                    "sentence_ids": h.get("sentence_ids", []),
                }],
            })

    print(f"Flattened {len(data)} multi-hop -> {len(out)} single-hop samples "
          f"(dropped {dropped} without _resolved_hops)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
