"""Distributed offline evaluation for CoE checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoProcessor

# Repo-root imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data.dataset import CoEDataset
from training.eval_metrics import compute_sample_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("offline_eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_file", default="data/wiki_coe/release_strict_chainsplit_20260428_1338/test.json")
    p.add_argument("--image_dir", default="data/wiki_coe/screenshots_trimmed")
    p.add_argument("--mode", default="multi_hop", choices=["single_hop", "multi_hop"])
    p.add_argument("--image_max_pixels", type=int, default=1048576)
    p.add_argument("--eval_resolution", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4,
                   help="per-rank generate batch size")
    p.add_argument("--prefetch", type=int, default=3,
                   help="number of batches to prefetch on CPU threads (0=off)")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--max_samples", type=int, default=-1,
                   help="sample limit (-1 = all)")
    p.add_argument("--candidate_top_k", type=int, default=5,
                   help="number of candidate images per sample")
    p.add_argument("--candidate_distractor_strategy", default="global",
                   choices=["global", "same_deck"])
    p.add_argument("--candidate_no_shuffle", action="store_true",
                   help="keep gold images first instead of shuffling candidate order")
    p.add_argument("--candidate_seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--save_preview_chars", type=int, default=2000,
                   help="truncate pred text in jsonl to this length")
    p.add_argument("--resume", action="store_true",
                   help="append to existing rank*.jsonl and skip already-written idx values")
    p.add_argument("--progress_every", type=int, default=20,
                   help="write rank progress JSON every N newly processed local samples")
    p.add_argument("--fsync_every", type=int, default=0,
                   help="fsync shard JSONL every N newly processed local samples (0=flush only)")
    return p.parse_args()


def main():
    args = parse_args()

    # --- distributed setup ---
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # --- load model & processor ---
    logger.info(f"[rank={rank}] loading model from {args.model_path}")
    t0 = time.time()
    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    logger.info(f"[rank={rank}] model ready in {time.time()-t0:.1f}s")

    # --- dataset ---
    candidate_cfg = {}
    if args.candidate_top_k > 0:
        candidate_cfg = {
            "top_k": args.candidate_top_k,
            "distractor_strategy": args.candidate_distractor_strategy,
            "shuffle": not args.candidate_no_shuffle,
            "seed": args.candidate_seed,
            "preserve_evidence_order": True,
        }
    val_dataset = CoEDataset(
        data_file=args.val_file,
        image_dir=args.image_dir,
        mode=args.mode,
        augmentation={},
        candidate_config=candidate_cfg,
        split="val",
        image_max_pixels=args.image_max_pixels,
        eval_resolution=args.eval_resolution,
    )
    n_total = len(val_dataset)
    if args.max_samples > 0:
        n_total = min(n_total, args.max_samples)
    logger.info(f"[rank={rank}] val size = {len(val_dataset)}, using {n_total}")

    # --- shard ---
    owned_indices = [i for i in range(n_total) if i % world_size == rank]
    logger.info(f"[rank={rank}] owns {len(owned_indices)} samples")

    shard_path = out_dir / f"rank{rank}.jsonl"
    progress_path = out_dir / f"rank{rank}.progress.json"
    BASE_METRIC_KEYS = ("answer_em", "answer_contains",
                        "bbox_localized", "bbox_recall", "bbox_precision",
                        "bbox_all_gt_hit", "parse_ok")
    IMAGE_METRIC_KEYS = ("image_hop_acc", "image_chain_order_acc",
                         "gold_image_recall", "gold_image_precision",
                         "gold_image_all_hit", "image_bbox_joint_recall",
                         "image_bbox_joint_all_gt_hit")
    agg = {k: 0.0 for k in BASE_METRIC_KEYS + IMAGE_METRIC_KEYS}
    agg.update({"count": 0.0, "failed": 0.0, "image_metric_count": 0.0})
    image_metrics_enabled = args.mode != "single_hop"

    def _accumulate_metrics(m: dict) -> None:
        for k in BASE_METRIC_KEYS:
            agg[k] += float(m.get(k, 0.0))
        valid = int(m.get("image_metric_valid", 0))
        for k in IMAGE_METRIC_KEYS:
            agg[k] += float(m.get(k, 0.0)) * valid
        agg["image_metric_count"] += valid
        agg["count"] += 1

    def _metrics_from_counts(counts: dict) -> dict:
        c = max(1.0, counts.get("count", 0.0))
        metrics = {k: counts.get(k, 0.0) / c for k in BASE_METRIC_KEYS}
        img_c = max(1.0, counts.get("image_metric_count", 0.0))
        if counts.get("image_metric_count", 0.0) > 0:
            metrics.update({k: counts.get(k, 0.0) / img_c for k in IMAGE_METRIC_KEYS})
        return metrics

    def _atomic_write_json(path: Path, obj: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        os.replace(tmp, path)

    def _load_completed_indices() -> set[int]:
        done: set[int] = set()
        if not (args.resume and shard_path.exists()):
            return done
        owned_set = set(owned_indices)
        with shard_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    idx = int(rec["idx"])
                    if idx not in owned_set or idx in done:
                        continue
                    done.add(idx)
                    m = rec.get("metrics") or {}
                    _accumulate_metrics(m)
                except Exception:
                    logger.warning(f"[rank={rank}] ignoring malformed resume line in {shard_path}")
        if done:
            logger.info(f"[rank={rank}] resume: loaded {len(done)} completed rows from {shard_path}")
        return done

    completed_indices = _load_completed_indices()
    indices = [i for i in owned_indices if i not in completed_indices]
    if completed_indices:
        logger.info(
            f"[rank={rank}] resume: remaining {len(indices)}/{len(owned_indices)} local samples"
        )

    # Pad on the left so generation continues from the right end for every
    # sequence in the batch (the tokenizer's default is often right-pad,
    # which would put pad tokens BETWEEN prompt and generation and break
    # decoding).
    tokenizer.padding_side = "left"
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"

    def _load_sample(idx: int):
        """Return (sample, prompt_text) or None on failure. Runs on CPU."""
        try:
            sample = val_dataset[idx]
        except Exception as e:
            logger.warning(f"[rank={rank}] val[{idx}] load failed: {e}")
            return None
        messages = [m for m in sample["messages"] if m["role"] != "assistant"]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return sample, prompt_text

    def _prepare_batch(batch_idx):
        """CPU-side: load all samples + run processor. Returns dict ready to
        .to(device), or None if the whole batch failed."""
        loaded = [_load_sample(i) for i in batch_idx]
        valid = [(i, *l) for i, l in zip(batch_idx, loaded) if l is not None]
        n_failed = len(batch_idx) - len(valid)
        if not valid:
            return {"batch_idx": batch_idx, "valid_idx": [], "samples": [],
                    "inputs": None, "n_failed": n_failed, "proc_failed": 0}
        vidxs   = [v[0] for v in valid]
        samples = [v[1] for v in valid]
        prompts = [v[2] for v in valid]
        flat_images = []
        for s in samples:
            imgs = s.get("images") or []
            flat_images.extend(imgs)
        proc_kwargs = dict(text=prompts, padding=True, return_tensors="pt")
        if flat_images:
            proc_kwargs["images"] = flat_images
        try:
            inputs = processor(**proc_kwargs)
        except Exception as e:
            logger.warning(f"[rank={rank}] processor failed ({len(valid)} samples): {e}")
            return {"batch_idx": batch_idx, "valid_idx": [], "samples": [],
                    "inputs": None, "n_failed": n_failed, "proc_failed": len(valid)}
        return {"batch_idx": batch_idx, "valid_idx": vidxs, "samples": samples,
                "inputs": inputs, "n_failed": n_failed, "proc_failed": 0}

    t_run = time.time()
    fout = open(shard_path, "a" if args.resume else "w", buffering=1)
    bs = args.batch_size
    n_done = len(completed_indices)
    last_progress_at = n_done
    last_fsync_at = n_done

    def _write_rank_progress(n_done_local: int, final: bool = False) -> None:
        counts = dict(agg)
        progress = {
            "rank": rank,
            "model_path": args.model_path,
            "val_file": args.val_file,
            "out_dir": str(out_dir),
            "owned_total": len(owned_indices),
            "n_done": n_done_local,
            "n_remaining": max(0, len(owned_indices) - n_done_local),
            "n_failed": int(counts.get("failed", 0)),
            "n_image_metric_valid": int(counts.get("image_metric_count", 0)),
            "metrics": _metrics_from_counts(counts),
            "counts": counts,
            "final": final,
            "updated_at": time.time(),
        }
        _atomic_write_json(progress_path, progress)

    def _write_partial_summary() -> None:
        if rank != 0:
            return
        counts = {k: 0.0 for k in BASE_METRIC_KEYS + IMAGE_METRIC_KEYS}
        counts.update({"count": 0.0, "failed": 0.0, "image_metric_count": 0.0})
        ranks = []
        for r in range(world_size):
            pp = out_dir / f"rank{r}.progress.json"
            if not pp.exists():
                continue
            try:
                prog = json.loads(pp.read_text())
            except Exception:
                continue
            ranks.append({
                "rank": r,
                "n_done": prog.get("n_done", 0),
                "owned_total": prog.get("owned_total", 0),
                "final": bool(prog.get("final", False)),
            })
            for k in counts:
                counts[k] += float((prog.get("counts") or {}).get(k, 0.0))
        summary = {
            "partial": True,
            "model_path": args.model_path,
            "val_file": args.val_file,
            "n_val_total": len(val_dataset),
            "n_evaluated": int(counts["count"]),
            "n_failed": int(counts["failed"]),
            "n_image_metric_valid": int(counts.get("image_metric_count", 0)),
            "metrics": _metrics_from_counts(counts),
            "counts": counts,
            "candidate": candidate_cfg,
            "ranks": ranks,
            "updated_at": time.time(),
        }
        _atomic_write_json(out_dir / "summary.partial.json", summary)

    def _flush_metrics(n_done: int):
        dt = time.time() - t_run
        rate = n_done / max(dt, 1e-6)
        eta = (len(owned_indices) - n_done) / max(rate, 1e-6)
        logger.info(
            f"[rank={rank}] {n_done}/{len(owned_indices)} "
            f"em={agg['answer_em']}/{agg['count']} "
            f"bbox={agg['bbox_localized']}/{agg['count']} "
            f"({rate:.2f} it/s, eta {eta/60:.1f}min)"
        )
        _write_rank_progress(n_done, final=False)
        _write_partial_summary()

    _write_rank_progress(n_done, final=False)
    _write_partial_summary()

    batches = [indices[s:s + bs] for s in range(0, len(indices), bs)]

    # Prefetch CPU preprocessing using a thread pool. The GPU processes
    # batch k while threads build batches k+1 .. k+prefetch.
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    if args.prefetch > 0:
        pool = ThreadPoolExecutor(max_workers=args.prefetch)
        futures = deque()
        # seed the pipeline
        for b in batches[: args.prefetch]:
            futures.append(pool.submit(_prepare_batch, b))
        next_to_submit = args.prefetch
    else:
        pool = None
        futures = deque()
        next_to_submit = 0

    for b_i, batch_idx in enumerate(batches):
        if pool is not None:
            prep = futures.popleft().result()
            if next_to_submit < len(batches):
                futures.append(pool.submit(_prepare_batch, batches[next_to_submit]))
                next_to_submit += 1
        else:
            prep = _prepare_batch(batch_idx)

        agg["failed"] += prep["n_failed"] + prep["proc_failed"]
        if prep["inputs"] is None:
            n_done += len(batch_idx)
            if n_done - last_progress_at >= args.progress_every or n_done >= len(owned_indices):
                _flush_metrics(n_done)
                last_progress_at = n_done
            continue

        inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                  for k, v in prep["inputs"].items()}
        vidxs   = prep["valid_idx"]
        samples = prep["samples"]

        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
        except Exception as e:
            logger.warning(f"[rank={rank}] batch generate failed ({len(vidxs)} samples): {str(e)[:200]}")
            agg["failed"] += len(vidxs)
            n_done += len(batch_idx)
            if n_done - last_progress_at >= args.progress_every or n_done >= len(owned_indices):
                _flush_metrics(n_done)
                last_progress_at = n_done
            continue

        input_len = inputs["input_ids"].shape[1]
        gen_ids_batch = out_ids[:, input_len:]
        for j, (idx_g, sample_g) in enumerate(zip(vidxs, samples)):
            pred_text = tokenizer.decode(gen_ids_batch[j], skip_special_tokens=True)
            gold = {
                "answer": sample_g.get("answer", ""),
                "evidence_chain": sample_g.get("evidence_chain", []),
                "image_metrics_enabled": image_metrics_enabled,
            }
            m = compute_sample_metrics(pred_text, gold, iou_thresh=args.iou_thresh)
            _accumulate_metrics(m)

            rec = {
                "idx": idx_g,
                "question": sample_g.get("question", ""),
                "gold_answer": gold["answer"],
                "gold_evidence_chain": gold["evidence_chain"],
                "candidate_screenshots": sample_g.get("candidate_screenshots", []),
                "gold_image_ids": sample_g.get("gold_image_ids", []),
                "pred_text": pred_text[: args.save_preview_chars],
                "metrics": m,
                "em": int(m["answer_em"]),
                "contains": int(m["answer_contains"]),
                "bbox": int(m["bbox_localized"]),
                "parse_ok": int(m["parse_ok"]),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += len(batch_idx)
        fout.flush()
        if args.fsync_every > 0 and n_done - last_fsync_at >= args.fsync_every:
            os.fsync(fout.fileno())
            last_fsync_at = n_done

        if n_done - last_progress_at >= args.progress_every or n_done >= len(owned_indices):
            _flush_metrics(n_done)
            last_progress_at = n_done
    if pool is not None:
        pool.shutdown(wait=True)
    _write_rank_progress(n_done, final=True)
    _write_partial_summary()
    fout.close()

    logger.info(f"[rank={rank}] done: {agg} in {time.time()-t_run:.1f}s")

    # --- all-reduce aggregate ---
    REDUCE_KEYS = list(BASE_METRIC_KEYS) + list(IMAGE_METRIC_KEYS) + [
        "count", "failed", "image_metric_count"
    ]
    if world_size > 1:
        buf = torch.tensor(
            [agg[k] for k in REDUCE_KEYS],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        vals = buf.cpu().tolist()
        total = {k: v for k, v in zip(REDUCE_KEYS, vals)}
    else:
        total = agg

    if rank == 0:
        metrics = _metrics_from_counts(total)
        summary = {
            "model_path": args.model_path,
            "val_file": args.val_file,
            "n_val_total": len(val_dataset),
            "n_evaluated": int(total["count"]),
            "n_failed": int(total["failed"]),
            "n_image_metric_valid": int(total.get("image_metric_count", 0)),
            "metrics": metrics,
            "counts": {k: total[k] for k in REDUCE_KEYS},
            "iou_thresh": args.iou_thresh,
            "max_new_tokens": args.max_new_tokens,
            "candidate": candidate_cfg,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        # merge shards for convenience
        merged = out_dir / "all.jsonl"
        with open(merged, "w") as fout:
            for r in range(world_size):
                sp = out_dir / f"rank{r}.jsonl"
                if sp.exists():
                    fout.write(sp.read_text())
        logger.info(f"[rank=0] summary: {json.dumps(summary['metrics'], indent=2)}")
        logger.info(f"[rank=0] wrote {out_dir}/summary.json and {merged}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
