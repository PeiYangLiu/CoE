"""Rebuild Phase 2 multi-hop training data with clean bboxes.

train_split.json / val_split.json have evidence_chain bboxes that were
produced by the weak char-n-gram matcher in build_wiki_coe.py. However
each sample also carries _resolved_hops, which were produced by the
strict stopword-based jaccard + value-in-text picker (scripts/
filter_oob_samples.py, mirrored from dataset.py::_pick_paragraph).

This script:
  1. Replaces evidence_chain[i].bboxes with the resolved-hop bboxes
     (matched by entity name, multi-bbox when a given entity appears in
     several resolved hops).
  2. Drops samples where any evidence_chain entity has NO matching
     resolved hop (upstream picker couldn't find a paragraph containing
     the value / within image bounds — unreliable supervision).
  3. Writes <name>.clean.json next to the input.

Usage:
    python scripts/rebuild_multihop_bboxes.py \\
        --inputs data/wiki_coe/train_split.json data/wiki_coe/val_split.json
"""
import argparse, json, os
from collections import defaultdict, Counter


def reconcile(sample):
    resolved = sample.get("_resolved_hops", [])
    if not resolved:
        return None, "no_resolved"
    # Group resolved hops by entity, preserving order so repeats map 1:1.
    ent2queue = defaultdict(list)
    for h in resolved:
        if h.get("bbox") and h.get("entity"):
            ent2queue[h["entity"]].append(h)

    new_chain = []
    # Consume resolved hops in entity-matched order; collect ALL bboxes for
    # a given ev_chain entry when multiple resolved hops share its entity.
    seen = defaultdict(int)
    for ev in sample["evidence_chain"]:
        q = ent2queue.get(ev["entity"], [])
        if not q:
            return None, "missing_hop"
        idx = seen[ev["entity"]]
        if idx >= len(q):
            # more ev_chain entries for this entity than resolved ones
            return None, "missing_hop"
        h = q[idx]
        seen[ev["entity"]] += 1

        new_ev = dict(ev)
        new_ev["bboxes"] = [h["bbox"]]
        sub_q = (h.get("question") or "").strip()
        if sub_q:
            new_ev["sub_query"] = sub_q
        new_chain.append(new_ev)

    out = dict(sample)
    out["evidence_chain"] = new_chain
    changed = any(new_chain[i]["bboxes"] != sample["evidence_chain"][i]["bboxes"]
                  for i in range(len(new_chain)))
    return out, ("changed" if changed else "same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args()

    for in_path in args.inputs:
        print(f"\n=== {in_path} ===", flush=True)
        with open(in_path) as f:
            data = json.load(f)
        kept = []
        stats = Counter()
        n_hops_before = 0; n_hops_after = 0
        for s in data:
            out, status = reconcile(s)
            stats[status] += 1
            if out is not None:
                kept.append(out)
                n_hops_before += len(s["evidence_chain"])
                n_hops_after += len(out["evidence_chain"])
        print(f"  input:  {len(data)}")
        print(f"  kept:   {len(kept)}")
        print(f"  stats:  {dict(stats)}")
        print(f"  hops:   {n_hops_before} -> {n_hops_after}")

        out_path = in_path.replace(".json", ".clean.json")
        with open(out_path, "w") as f:
            json.dump(kept, f)
        print(f"  wrote:  {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

        # hop-type breakdown
        typ = Counter(s.get("num_hops") for s in kept)
        print(f"  num_hops dist: {dict(typ)}")


if __name__ == "__main__":
    main()
