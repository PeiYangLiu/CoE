"""vLLM offline batch inference for Wiki-CoE multi-hop evaluation.

Loads a Qwen3-VL CoE checkpoint and runs greedy generation over an entire
val/test JSON file using vLLM's Offline Batch Inference API. One process
holds the model on `tensor_parallel_size` GPUs and processes the dataset
in a single, dynamically-scheduled batch (continuous batching).

Outputs match scripts/offline_full_eval.py:
  <out_dir>/all.jsonl         per-sample {idx, question, gold_*, pred_text, metrics}
  <out_dir>/summary.json      aggregate metrics

Usage:
  python scripts/vllm_full_eval.py \
    --model_path checkpoints/coe_phase2/best \
    --val_file   data/wiki_coe/release_strict_chainsplit_20260428_1338/test.json \
    --image_dir  data/wiki_coe/screenshots_trimmed \
    --out_dir    eval_out/coe_phase2_best_test_full_vllm \
    --tensor_parallel_size 8 \
    --candidate_top_k 5
"""
from __future__ import annotations
import argparse, json, os, sys, time, logging
from pathlib import Path

# Repo-root imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data.dataset import CoEDataset
from training.eval_metrics import compute_sample_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm_eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_file", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", default="multi_hop", choices=["single_hop", "multi_hop"])
    p.add_argument("--image_max_pixels", type=int, default=1048576)
    p.add_argument("--eval_resolution", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=16384,
                   help="vLLM max model length (prompt + generation)")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--limit_mm_per_prompt_image", type=int, default=8,
                   help="vLLM upper bound on images per prompt")
    p.add_argument("--max_num_seqs", type=int, default=64,
                   help="vLLM max concurrent sequences")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--candidate_top_k", type=int, default=5)
    p.add_argument("--candidate_distractor_strategy", default="global",
                   choices=["global", "same_deck"])
    p.add_argument("--candidate_no_shuffle", action="store_true")
    p.add_argument("--candidate_seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--save_preview_chars", type=int, default=2000)
    p.add_argument("--prep_workers", type=int, default=40,
                   help="ThreadPool workers for sample prep producer")
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graphs for debugging.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- import vLLM lazily so other tools that import this file won't pay the cost
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    # ---- dataset & prompt builder ----
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
    logger.info(f"val size = {len(val_dataset)}, evaluating {n_total}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ---- vLLM init (do this BEFORE prepping prompts so we don't OOM RAM) ----
    logger.info(f"loading vLLM model from {args.model_path} on TP={args.tensor_parallel_size}")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt_image},
        enforce_eager=args.enforce_eager,
        mm_processor_kwargs={"min_pixels": 256 * 28 * 28,
                             "max_pixels": args.image_max_pixels},
    )
    logger.info(f"vLLM ready in {time.time()-t0:.1f}s")

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop=None,
    )

    # ---- streaming pipeline: producer threads prep samples into bounded
    # queue; main thread pulls batches, sends to vLLM, scores results.
    # This keeps RAM bounded to ~queue_size+batch_size samples instead of
    # materializing all 35K prompts up front.
    # ---------------------------------------------------------------------
    import threading, queue as _queue
    from concurrent.futures import ThreadPoolExecutor

    BASE_KEYS = ("answer_em", "answer_contains",
                 "bbox_localized", "bbox_recall", "bbox_precision",
                 "bbox_all_gt_hit", "parse_ok")
    IMG_KEYS = ("image_hop_acc", "image_chain_order_acc",
                "gold_image_recall", "gold_image_precision",
                "gold_image_all_hit", "image_bbox_joint_recall",
                "image_bbox_joint_all_gt_hit")
    image_metrics_enabled = (args.mode != "single_hop")

    agg = {k: 0.0 for k in BASE_KEYS + IMG_KEYS}
    agg.update({"count": 0.0, "image_metric_count": 0.0, "failed": 0.0})

    QUEUE_SIZE = max(args.max_num_seqs * 2, 128)
    q: _queue.Queue = _queue.Queue(maxsize=QUEUE_SIZE)
    SENTINEL = object()
    n_failed_prep = 0
    n_failed_lock = threading.Lock()

    def _build_one(i):
        try:
            s = val_dataset[i]
        except Exception as e:
            logger.warning(f"val[{i}] load failed: {e}")
            return None
        msgs = [m for m in s["messages"] if m["role"] != "assistant"]
        prompt_text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return s, prompt_text

    def _producer(executor: ThreadPoolExecutor):
        """Submit build_one futures, push results into bounded queue in order."""
        nonlocal n_failed_prep
        # use map (lazy iterator) -- futures complete in order, queue.put blocks
        # when full so we never run ahead of the consumer.
        for res in executor.map(_build_one, range(n_total), chunksize=4):
            if res is None:
                with n_failed_lock:
                    n_failed_prep += 1
                continue
            q.put(res)
        q.put(SENTINEL)

    out_file = out_dir / "all.jsonl"
    fout = out_file.open("w", buffering=1)
    n_done = 0
    t_start = time.time()
    BATCH = max(args.max_num_seqs, 16)

    pool = ThreadPoolExecutor(max_workers=args.prep_workers)
    prod_thread = threading.Thread(target=_producer, args=(pool,), daemon=True)
    prod_thread.start()
    logger.info(f"streaming pipeline started: prep_workers={args.prep_workers}, "
                f"queue_size={QUEUE_SIZE}, vLLM batch={BATCH}")

    def _drain_one_batch():
        nonlocal n_done
        # collect up to BATCH items (or until sentinel)
        batch_samples = []
        batch_prompts = []
        eof = False
        while len(batch_samples) < BATCH:
            item = q.get()
            if item is SENTINEL:
                eof = True
                break
            s, p = item
            batch_samples.append(s)
            batch_prompts.append(p)
        if not batch_samples:
            return False, eof  # nothing to do, eof seen

        requests = [
            {"prompt": p, "multi_modal_data": {"image": s["images"]}}
            for s, p in zip(batch_samples, batch_prompts)
        ]
        outs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        for s, out in zip(batch_samples, outs):
            pred_text = out.outputs[0].text if out.outputs else ""
            gold = {
                "answer": s.get("answer", ""),
                "evidence_chain": s.get("evidence_chain", []),
                "image_metrics_enabled": image_metrics_enabled,
            }
            m = compute_sample_metrics(pred_text, gold, iou_thresh=args.iou_thresh)
            for k in BASE_KEYS:
                agg[k] += float(m[k])
            valid = int(m.get("image_metric_valid", 0))
            for k in IMG_KEYS:
                agg[k] += float(m[k]) * valid
            agg["image_metric_count"] += valid
            agg["count"] += 1

            rec = {
                "idx": s.get("sample_id", -1),
                "question": s.get("question", ""),
                "gold_answer": gold["answer"],
                "gold_evidence_chain": gold["evidence_chain"],
                "candidate_screenshots": s.get("candidate_screenshots", []),
                "gold_image_ids": s.get("gold_image_ids", []),
                "pred_text": pred_text[: args.save_preview_chars],
                "metrics": m,
                "em": int(m["answer_em"]),
                "contains": int(m["answer_contains"]),
                "bbox": int(m["bbox_localized"]),
                "parse_ok": int(m["parse_ok"]),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += len(batch_samples)
        return True, eof

    eof_seen = False
    while not eof_seen or not q.empty():
        had_work, eof_seen = _drain_one_batch()
        if not had_work and eof_seen:
            break
        if had_work:
            dt = time.time() - t_start
            rate = n_done / max(dt, 1e-6)
            eta = max(0, (n_total - n_done)) / max(rate, 1e-6)
            logger.info(f"[stream] {n_done}/{n_total} "
                        f"({rate:.2f} sample/s, eta {eta/60:.1f}min)  "
                        f"qsize={q.qsize()} prep_failed={n_failed_prep}")

    prod_thread.join()
    pool.shutdown(wait=True)
    fout.close()

    agg["failed"] = float(n_failed_prep)
    logger.info(f"all done in {(time.time()-t_start)/60:.1f}min, "
                f"n_done={n_done}, prep_failed={n_failed_prep}")
    logger.info(f"wrote {out_file}")

    # ---- summary ----
    c = max(1.0, agg["count"])
    metrics = {k: agg[k] / c for k in BASE_KEYS}
    img_c = max(1.0, agg.get("image_metric_count", 0.0))
    if agg.get("image_metric_count", 0.0) > 0:
        metrics.update({k: agg[k] / img_c for k in IMG_KEYS})

    summary = {
        "model_path": args.model_path,
        "val_file": args.val_file,
        "n_val_total": len(val_dataset),
        "n_evaluated": int(agg["count"]),
        "n_failed": int(agg["failed"]),
        "n_image_metric_valid": int(agg.get("image_metric_count", 0)),
        "metrics": metrics,
        "counts": {k: agg[k] for k in BASE_KEYS + IMG_KEYS + ("count", "image_metric_count", "failed")},
        "iou_thresh": args.iou_thresh,
        "max_new_tokens": args.max_new_tokens,
        "candidate": candidate_cfg,
        "engine": "vllm",
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
