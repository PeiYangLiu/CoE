"""Offline filter that drops training samples whose target evidence is out
of the screenshot (OOB) and cannot be localized to any in-bounds paragraph.

Root cause: `data/build_wiki_coe.py` caps screenshot height at 15000px
(`MAX_PAGE_HEIGHT`), but the layout JSON covers the full HTML document
(can be 50k+ px). Paragraphs beyond the screenshot are unreachable as
training targets.

Runtime behavior in `data/dataset.py._pick_paragraph` already filters
OOB paragraphs, but the sample is silently dropped at load time. This
offline script makes the drop explicit in `train.json` / `test.json`
by producing a cleaned copy plus a report.

Reads:  INPUT_JSON, screenshots/<name>.png, screenshots/<name>_bboxes.json
Writes: OUTPUT_JSON (cleaned), REPORT_JSON (drop reasons + counts)

Usage:
    python scripts/filter_oob_samples.py \
        --input  ~/conversationhub/liupeiyang/dataset/wiki_coe/train.json \
        --output ~/conversationhub/liupeiyang/dataset/wiki_coe/train_clean.json \
        --shots  ~/conversationhub/liupeiyang/dataset/wiki_coe/screenshots \
        --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# Picker mirrored from data/dataset.py._pick_paragraph (hybrid tier).
# Kept in-sync manually; any logic change there should be reflected here.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset("""
    a an the and or but if then else when of in on at to from by for with as
    is are was were be been being have has had do does did will would shall
    should may might must can could this that these those he she it they we
    you i me him her his hers its their theirs our ours your yours my mine
    who whom whose which what where why how there here not no yes so than
    because about into through during before after above below up down out
    off over under again further once all any both each few more most other
    some such only own same too very s t just now also which
""".split())


def _content_tokens(text: str):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) >= 2]


def _jaccard_tokens(a: str, b: str) -> float:
    """target-containment: |target ∩ para| / |target| (see dataset.py _coverage_score)."""
    ta = set(_content_tokens(a))
    tb = set(_content_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def _value_in_text(value: str, text: str) -> bool:
    """Continuous (punctuation-insensitive) substring match. See dataset.py."""
    def _norm(s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", " ", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    v = _norm(value); t = _norm(text)
    if not v:
        return False
    return f" {v} " in f" {t} "


def _img_height(path: str) -> Optional[int]:
    try:
        with Image.open(path) as im:
            return im.size[1]
    except Exception:
        return None


def _pick_paragraph(
    bboxes_path: str,
    img_path: str,
    value: str,
    target_sent: str,
) -> Tuple[Optional[Dict], str]:
    """Returns (chosen_paragraph_or_None, status_reason)."""
    if not os.path.exists(bboxes_path):
        return None, "no_bboxes_file"
    try:
        paras = json.load(open(bboxes_path))
    except Exception:
        return None, "bad_bboxes_file"
    if not target_sent:
        return None, "no_target_sent"

    img_h = _img_height(img_path) if img_path else None

    cands = []
    n_oob = 0
    for p in paras:
        x1, y1, x2, y2 = p["bbox"]
        if (x2 - x1) < 80 or (y2 - y1) < 5:
            continue
        if img_h is not None and y2 > img_h:
            n_oob += 1
            continue
        if len(p["text"].split()) < 8:
            continue
        sim = _jaccard_tokens(target_sent, p["text"])
        vi = _value_in_text(value, p["text"])
        score = sim + (0.2 if vi else 0.0)
        cands.append((score, sim, vi, y1, p))

    if not cands:
        return None, ("no_candidates_oob" if n_oob else "no_candidates")
    cands.sort(key=lambda c: (-c[0], -c[1], c[3]))
    best = cands[0]
    if best[0] < 0.4:
        return None, ("below_threshold_oob" if n_oob else "below_threshold")
    return best[4], ("val_match" if best[2] else "sim_only")


def _humanize_relation(r: str) -> str:
    return r.replace("_", " ")


# ---------------------------------------------------------------------------
# Per-sample worker.
# ---------------------------------------------------------------------------
_WORKER_SHOTS_DIR: str = ""
_WORKER_RAW_CTX: Dict[str, List[str]] = {}


def _worker_init(shots_dir: str, raw_ctx: Dict[str, List[str]]) -> None:
    global _WORKER_SHOTS_DIR, _WORKER_RAW_CTX
    _WORKER_SHOTS_DIR = shots_dir
    _WORKER_RAW_CTX = raw_ctx


def _process_sample(sample) -> Tuple[Optional[Dict], Dict[str, int]]:
    shots_dir = _WORKER_SHOTS_DIR
    raw_ctx = _WORKER_RAW_CTX
    reasons: Counter = Counter()
    evidences = sample.get("evidences") or []
    chain = sample.get("evidence_chain") or []
    if not evidences or not chain:
        reasons["no_evidence_or_chain"] += 1
        return None, dict(reasons)

    ent2ev = {ev.get("entity", ""): ev for ev in chain}
    kept_hops: List[Dict] = []

    for evi in evidences:
        if len(evi) != 3:
            reasons["malformed_triple"] += 1
            continue
        e, rel, v = evi
        v = str(v)

        ev_entry = ent2ev.get(e)
        if ev_entry is None:
            for ce, ev in ent2ev.items():
                if ce.startswith(e) or e.startswith(ce):
                    ev_entry = ev
                    break
        if ev_entry is None:
            reasons["no_chain_entry"] += 1
            continue

        sids = ev_entry.get("sentence_ids") or []
        ctx = raw_ctx.get(ev_entry.get("entity", ""), [])
        target_sent = ctx[sids[0]] if (sids and sids[0] < len(ctx)) else ""

        shot = ev_entry.get("screenshot", "")
        bboxes_path = os.path.join(shots_dir, shot.replace(".png", "_bboxes.json"))
        img_path = os.path.join(shots_dir, shot)
        chosen, status = _pick_paragraph(bboxes_path, img_path, v, target_sent)
        reasons[status] += 1
        if chosen is None:
            continue

        kept_hops.append(
            {
                "question": f"What is the {_humanize_relation(rel)} of {e}?",
                "answer": v,
                "entity": ev_entry.get("entity", e),
                "screenshot": shot,
                "bbox": chosen["bbox"],
                "sentence_ids": sids,
            }
        )

    if not kept_hops:
        reasons["sample_dropped_all_hops_oob"] += 1
        return None, dict(reasons)

    if len(kept_hops) < len(evidences):
        reasons["sample_partial_hops_dropped"] += 1

    out = dict(sample)
    out["_resolved_hops"] = kept_hops
    out["_n_hops_kept"] = len(kept_hops)
    out["_n_hops_total"] = len(evidences)
    return out, dict(reasons)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw train/test.json")
    ap.add_argument("--output", required=True, help="Cleaned JSON output path")
    ap.add_argument("--shots", required=True, help="screenshots/ directory")
    ap.add_argument(
        "--raw_ctx",
        default=None,
        help="Optional raw JSON with entity->[sentences] (defaults to --input)",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, drop samples where any hop is OOB (default keeps samples with at least one resolvable hop).",
    )
    ap.add_argument("--report", default=None, help="Path for JSON drop report")
    args = ap.parse_args()

    in_path = os.path.expanduser(args.input)
    out_path = os.path.expanduser(args.output)
    shots_dir = os.path.expanduser(args.shots)

    print(f"Loading {in_path} ...", flush=True)
    data = json.load(open(in_path))
    print(f"  {len(data)} samples", flush=True)

    # Build entity->sentences map. In this dataset, train.json itself has
    # the context under each sample's "context" field; we flatten across
    # the whole file once so all workers share it.
    raw_ctx: Dict[str, List[str]] = {}
    ctx_src = args.raw_ctx or in_path
    print(f"Building context map from {ctx_src} ...", flush=True)
    ctx_data = data if ctx_src == in_path else json.load(open(os.path.expanduser(ctx_src)))
    for s in ctx_data:
        for item in s.get("context", []) or []:
            if not isinstance(item, list) or len(item) < 2:
                continue
            ent, sents = item[0], item[1]
            if ent and ent not in raw_ctx:
                raw_ctx[ent] = list(sents)
    print(f"  {len(raw_ctx)} entity contexts", flush=True)

    tasks = data

    kept: List[Dict] = []
    agg: Counter = Counter()
    n_samples_dropped = 0

    if args.workers <= 1:
        _worker_init(shots_dir, raw_ctx)
        for t in tasks:
            out, r = _process_sample(t)
            for k, v in r.items():
                agg[k] += v
            if out is None:
                n_samples_dropped += 1
            elif args.strict and out.get("_n_hops_kept", 0) < out.get("_n_hops_total", 0):
                n_samples_dropped += 1
            else:
                kept.append(out)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(shots_dir, raw_ctx),
        ) as ex:
            for i, (out, r) in enumerate(ex.map(_process_sample, tasks, chunksize=64)):
                for k, v in r.items():
                    agg[k] += v
                if out is None:
                    n_samples_dropped += 1
                elif args.strict and out.get("_n_hops_kept", 0) < out.get("_n_hops_total", 0):
                    n_samples_dropped += 1
                else:
                    kept.append(out)
                if (i + 1) % 2000 == 0:
                    print(
                        f"  processed {i+1}/{len(tasks)}  "
                        f"kept={len(kept)}  dropped={n_samples_dropped}",
                        flush=True,
                    )

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(kept, f)
    print(
        f"\nWrote {out_path}: kept={len(kept)} / input={len(data)} "
        f"(dropped={n_samples_dropped}, {100*n_samples_dropped/max(1,len(data)):.1f}%)",
        flush=True,
    )

    report = {
        "input": in_path,
        "output": out_path,
        "n_input": len(data),
        "n_kept": len(kept),
        "n_dropped": n_samples_dropped,
        "hop_status_counts": dict(agg),
        "strict": args.strict,
    }
    report_path = args.report or (out_path + ".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}", flush=True)
    print("\nHop-level status breakdown:")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
