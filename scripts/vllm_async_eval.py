"""vLLM AsyncLLMEngine offline batch eval.

Uses asyncio + AsyncLLMEngine to keep continuous batching saturated:
- A producer task feeds new requests into the engine as fast as we can
  build them (bounded by `inflight_limit` so RAM stays bounded).
- A consumer awaits each request's async generator and writes results.

This avoids the trap of `LLM.generate(batch=N)` which is synchronous
per-batch and underutilizes GPU, and the OOM trap of submitting all
35K prompts at once with their PIL images.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, sys, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from data.dataset import CoEDataset
from training.eval_metrics import compute_sample_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vllm_async_eval")


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
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--limit_mm_per_prompt_image", type=int, default=8)
    p.add_argument("--max_num_seqs", type=int, default=128,
                   help="vLLM max concurrently scheduled sequences")
    p.add_argument("--inflight_limit", type=int, default=192,
                   help="upper bound of in-flight requests; controls peak RAM "
                        "for buffered PIL images. Should be >= max_num_seqs.")
    p.add_argument("--prep_workers", type=int, default=40,
                   help="ThreadPool workers for sample prep")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--candidate_top_k", type=int, default=5)
    p.add_argument("--candidate_distractor_strategy", default="global",
                   choices=["global", "same_deck"])
    p.add_argument("--candidate_no_shuffle", action="store_true")
    p.add_argument("--candidate_seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--save_preview_chars", type=int, default=2000)
    p.add_argument("--enforce_eager", action="store_true")
    return p.parse_args()


async def main_async(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoProcessor

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

    # ----- spin up AsyncLLM ------
    logger.info(f"loading AsyncLLM model from {args.model_path} on TP={args.tensor_parallel_size}")
    t0 = time.time()
    engine_args = AsyncEngineArgs(
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
        disable_log_stats=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    logger.info(f"AsyncLLM ready in {time.time()-t0:.1f}s")

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
    )

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

    out_file = out_dir / "all.jsonl"
    fout = out_file.open("w", buffering=1)
    fout_lock = asyncio.Lock()

    # ----- producer side: prep samples in a thread pool, feed bounded queue ----
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=args.prep_workers)
    loop = asyncio.get_running_loop()

    inflight_sem = asyncio.Semaphore(args.inflight_limit)
    n_done = 0
    n_failed = 0
    t_start = time.time()
    last_log_t = t_start

    def _build_one(i):
        """Runs in worker thread."""
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

    async def _process_one(idx_global: int):
        nonlocal n_done, n_failed, last_log_t
        # gate: bounded in-flight to keep RAM sane
        await inflight_sem.acquire()
        try:
            built = await loop.run_in_executor(pool, _build_one, idx_global)
            if built is None:
                n_failed += 1
                return
            s, prompt_text = built

            req_id = f"req-{idx_global}"
            req_input = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": s["images"]},
            }

            # AsyncLLM.generate returns an async iterator of RequestOutput.
            # We just want the final one (after the stream completes).
            final_out = None
            async for out in engine.generate(req_input, sampling, req_id):
                final_out = out

            pred_text = ""
            if final_out is not None and final_out.outputs:
                pred_text = final_out.outputs[0].text

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
            line = json.dumps(rec, ensure_ascii=False)
            async with fout_lock:
                fout.write(line + "\n")

            n_done += 1
            now = time.time()
            if n_done % 64 == 0 or now - last_log_t > 30:
                last_log_t = now
                rate = n_done / max(now - t_start, 1e-6)
                eta = (n_total - n_done) / max(rate, 1e-6)
                logger.info(f"[async] {n_done}/{n_total}  "
                            f"({rate:.2f} sample/s  eta {eta/60:.1f}min)  "
                            f"failed={n_failed}")
        except Exception as e:
            n_failed += 1
            logger.warning(f"req {idx_global} crashed: {e}")
        finally:
            inflight_sem.release()

    logger.info(f"streaming pipeline started: prep_workers={args.prep_workers}, "
                f"inflight_limit={args.inflight_limit}, max_num_seqs={args.max_num_seqs}")

    # Schedule all tasks; semaphore in _process_one keeps inflight bounded.
    tasks = [asyncio.create_task(_process_one(i)) for i in range(n_total)]
    await asyncio.gather(*tasks)

    fout.close()
    pool.shutdown(wait=True)

    agg["failed"] = float(n_failed)
    logger.info(f"all done in {(time.time()-t_start)/60:.1f}min  n_done={n_done}  failed={n_failed}")

    # ----- summary -----
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
        "engine": "vllm_async",
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"wrote {out_dir}/summary.json")

    # graceful shutdown of engine workers
    try:
        await engine.shutdown()  # vLLM 0.11 AsyncLLM
    except Exception:
        pass


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
