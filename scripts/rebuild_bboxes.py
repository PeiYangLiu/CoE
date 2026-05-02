#!/usr/bin/env python
"""
Regenerate all `_bboxes.json` files from saved `_layout.json` snapshots.

Uses the FIXED `_compute_bboxes_from_layout` in data/build_wiki_coe.py which:
  - collects float/infobox regions as (x_left, y_top, y_bottom) triples
  - only clips a text line's right edge when the line vertically overlaps
    a float region (instead of clipping every line to the global float left)

Runs multi-process since it's pure CPU / JSON I/O.
"""
import argparse
import json
import os
import sys
import time
from glob import glob
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_wiki_coe import _compute_bboxes_from_layout  # noqa: E402


def _one(layout_path: str) -> tuple:
    """Read layout, compute bboxes, write. Returns (path, n_paras, err_or_none)."""
    out_path = layout_path.replace("_layout.json", "_bboxes.json")
    try:
        with open(layout_path, "r") as f:
            layout = json.load(f)
        bboxes = _compute_bboxes_from_layout(layout)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(bboxes, f)
        os.replace(tmp, out_path)
        return (layout_path, len(bboxes), None)
    except Exception as e:
        return (layout_path, 0, repr(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--screenshots_dir", required=True,
                   help="dir containing *_layout.json and *_bboxes.json")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0=all")
    args = p.parse_args()

    layouts = sorted(glob(os.path.join(args.screenshots_dir, "*_layout.json")))
    if args.limit:
        layouts = layouts[: args.limit]
    print(f"Found {len(layouts)} layout files. Using {args.workers} workers.", flush=True)

    t0 = time.time()
    n_ok = 0
    n_err = 0
    errs = []
    with Pool(args.workers) as pool:
        for i, (path, n_paras, err) in enumerate(pool.imap_unordered(_one, layouts, chunksize=32), 1):
            if err is None:
                n_ok += 1
            else:
                n_err += 1
                if len(errs) < 20:
                    errs.append((os.path.basename(path), err))
            if i % 2000 == 0 or i == len(layouts):
                dt = time.time() - t0
                rate = i / max(dt, 1e-6)
                eta = (len(layouts) - i) / max(rate, 1e-6)
                print(f"  [{i}/{len(layouts)}] ok={n_ok} err={n_err}  "
                      f"{rate:.0f} files/s  ETA {eta:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.1f}s. ok={n_ok} err={n_err}")
    if errs:
        print("First errors:")
        for p, e in errs:
            print(f"  {p}: {e}")


if __name__ == "__main__":
    main()
