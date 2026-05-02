"""Parallel pre-trim of bottom whitespace for all screenshots.

Input : data/wiki_coe/screenshots/*.png       (originals, untouched)
Output: data/wiki_coe/screenshots_trimmed/*.png
Meta  : data/wiki_coe/trim_manifest.jsonl     (one record per file)

Algorithm (conservative):
  - Convert to grayscale
  - Find rows where std(row pixels) > THRESH (= has any text/graphics)
  - Keep rows [0 .. last_content_row + MARGIN]
  - Never grow the image (if computed new_h >= h, skip trimming)
  - Bbox coords are NOT remapped: all GT bboxes live in the top content
    region, so trimming the bottom doesn't change any (x1,y1,x2,y2).

Skips outputs that already exist with correct orig_size (idempotent /
resumable).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
import numpy as np

THRESH = 5.0
MARGIN = 20


def _trim_one(args):
    src_path, dst_path = args
    try:
        im = Image.open(src_path)
        orig_mode = im.mode
        w, h = im.size
        # grayscale scan (cheap)
        arr = np.asarray(im.convert("L"))
        row_std = arr.std(axis=1)
        content = np.where(row_std > THRESH)[0]
        if len(content) == 0:
            new_h = h  # all blank, keep as-is
        else:
            new_h = int(content[-1]) + 1 + MARGIN
            new_h = min(new_h, h)
        if new_h >= h:
            # no meaningful trim; just save as copy (preserve format)
            im.convert("RGB").save(dst_path, "PNG", optimize=False)
            return {"file": src_path.name, "orig_h": h, "new_h": h,
                    "rows_removed": 0, "skipped": True}
        trimmed = im.convert("RGB").crop((0, 0, w, new_h))
        trimmed.save(dst_path, "PNG", optimize=False)
        return {"file": src_path.name, "orig_h": h, "new_h": new_h,
                "rows_removed": h - new_h, "skipped": False,
                "orig_w": w}
    except Exception as e:
        return {"file": src_path.name, "error": str(e)[:200]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/wiki_coe/screenshots")
    p.add_argument("--dst", default="data/wiki_coe/screenshots_trimmed")
    p.add_argument("--manifest", default="data/wiki_coe/trim_manifest.jsonl")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="reprocess files that already exist in dst")
    args = p.parse_args()

    src_dir = Path(args.src).resolve()
    dst_dir = Path(args.dst).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()

    all_src = sorted(src_dir.glob("*.png"))
    print(f"[scan] {len(all_src)} source images in {src_dir}")

    # resumable: skip files already present in dst
    if args.force:
        work = [(p, dst_dir / p.name) for p in all_src]
    else:
        work = []
        skipped_existing = 0
        for p in all_src:
            tgt = dst_dir / p.name
            if tgt.exists() and tgt.stat().st_size > 0:
                skipped_existing += 1
            else:
                work.append((p, tgt))
        print(f"[resume] {skipped_existing} already done; processing {len(work)} new")

    if not work:
        print("nothing to do")
        return

    t0 = time.time()
    done = 0
    errors = 0
    rows_saved_total = 0
    pixels_saved_total = 0

    # Open manifest in append mode (resumable)
    mf = open(manifest_path, "a", buffering=1)

    with Pool(args.workers) as pool:
        for rec in pool.imap_unordered(_trim_one, work, chunksize=8):
            done += 1
            if "error" in rec:
                errors += 1
            else:
                rows_saved_total += rec.get("rows_removed", 0)
                if "orig_w" in rec:
                    pixels_saved_total += rec["rows_removed"] * rec["orig_w"]
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if done % 500 == 0 or done == len(work):
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(work) - done) / max(rate, 1e-6)
                print(f"[{done}/{len(work)}] "
                      f"{rate:.1f} img/s, "
                      f"pixels_saved={pixels_saved_total/1e9:.2f} G, "
                      f"errors={errors}, eta={eta/60:.1f}min", flush=True)

    mf.close()
    print(f"\n[done] {done} processed, {errors} errors, "
          f"{rows_saved_total/1e6:.1f}M rows saved, "
          f"{pixels_saved_total/1e9:.2f}G pixels saved")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
