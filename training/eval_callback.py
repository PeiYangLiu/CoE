"""Generative evaluation callback for CoE training.

Runs model.generate() on the validation set at each HF Trainer eval step,
parses model output JSON, and computes:
  - answer_em        : SQuAD-style exact match on the `answer` field
  - answer_contains  : gold-as-substring (lenient)
  - bbox_localized   : relaxed paragraph-level localization (IoU>=0.3 OR
                       predicted-bbox-center inside GT bbox)
  - parse_ok         : whether output was parseable JSON

Works with DeepSpeed ZeRO-3: generate() gathers sharded parameters per
forward pass. Each rank processes a disjoint shard of the val set; results
are all-gathered and rank 0 computes + logs aggregate metrics via
`trainer.log(...)`, which propagates to swanlab / wandb / console.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .eval_metrics import compute_sample_metrics

logger = logging.getLogger(__name__)


class GenerativeEvalCallback(TrainerCallback):
    def __init__(
        self,
        processor,
        val_dataset,
        max_samples: int = 256,
        max_new_tokens: int = 256,
        iou_thresh: float = 0.3,
        batch_size: int = 1,
        generation_kwargs: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ):
        self.processor = processor
        self.val_dataset = val_dataset
        self.max_samples = max_samples
        self.max_new_tokens = max_new_tokens
        self.iou_thresh = iou_thresh
        self.batch_size = max(1, int(batch_size))
        self.generation_kwargs = generation_kwargs or {}
        self.save_dir = save_dir
        self._trainer = None  # set externally after Trainer is constructed

    def set_trainer(self, trainer):
        self._trainer = trainer

    # Trainer wires this into each evaluate(); we piggyback on it.
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or self.val_dataset is None:
            return
        trainer = self._trainer
        save_jsonl_path = None
        if self.save_dir is not None:
            step = getattr(state, "global_step", -1)
            save_jsonl_path = os.path.join(
                self.save_dir,
                f"step_{step}",
                f"rank_{int(os.environ.get('RANK', '0'))}.jsonl",
            )
        self._run_eval(
            model, state=state, n_samples=self.max_samples,
            save_jsonl_path=save_jsonl_path, trainer=trainer,
        )

    def run_full_eval(self, model, save_jsonl_path, state=None):
        """Public entrypoint for a post-training full-val-set inference.
        Iterates the ENTIRE val dataset, saves per-sample predictions to
        `save_jsonl_path` (rank 0 only), and logs aggregate metrics.
        """
        return self._run_eval(
            model, state=state, n_samples=len(self.val_dataset),
            save_jsonl_path=save_jsonl_path, trainer=self._trainer,
        )

    def _run_eval(self, model, state, n_samples, save_jsonl_path, trainer):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))

        n_total = min(n_samples, len(self.val_dataset))
        # Under ZeRO-3, model.generate() triggers async param prefetching
        # that does not play well with autoregressive multi-forward. We
        # materialize all parameters once for the duration of the eval
        # (cheap on H100-80G for an 8B model: ~16GB bf16), which avoids
        # "still have inflight params" errors. All ranks iterate the same
        # indices so NCCL collectives in the forward pass stay in sync.
        indices = list(range(n_total))
        # Detect ZeRO-3 by checking whether any parameter is DS-partitioned.
        # World-size alone is insufficient: single-GPU runs still use ZeRO-3
        # when configured, and will hit "inflight params" without gather.
        is_zero3 = any(hasattr(p, "ds_id") for p in model.parameters())
        needs_nccl_sync = dist.is_initialized() and dist.get_world_size() > 1

        t0 = time.time()
        local_metrics = {
            "answer_em": 0, "answer_contains": 0,
            "bbox_localized": 0, "parse_ok": 0, "count": 0,
        }
        local_records = []  # per-sample predictions (for save_jsonl_path)

        model.eval()
        tokenizer = self.processor.tokenizer
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or eos_id
        # Left-padding is required for batched decoder-only generation so all
        # sequences end at the same position before new tokens are produced.
        orig_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"

        # Use a no-op context if gather isn't available.
        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters as _GP
            def _make_gather():
                return _GP(list(model.parameters()), modifier_rank=None, enabled=is_zero3)
        except Exception:
            from contextlib import nullcontext
            def _make_gather():
                return nullcontext()

        # Shard: rank r owns samples where idx % world_size == r. Build the
        # owner's list up front, then consume it in chunks of `batch_size`.
        my_indices = [i for i in indices if (i % max(world_size, 1)) == rank]

        def _prepare_batch(batch_idxs):
            """Load samples, apply chat template, run processor. Returns
            (inputs_dict_on_device, list_of_samples, list_of_kept_idxs) or
            (None, [], []) if the whole batch failed."""
            prompts, imgs_per_sample, kept_samples, kept_idxs = [], [], [], []
            for idx in batch_idxs:
                try:
                    sample = self.val_dataset[idx]
                except Exception as e:
                    logger.warning(f"[rank={rank}] val[{idx}] load failed: {e}")
                    continue
                if sample is None:
                    continue
                messages = [m for m in sample["messages"] if m["role"] != "assistant"]
                prompt_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                prompts.append(prompt_text)
                imgs_per_sample.append(sample.get("images") or [])
                kept_samples.append(sample)
                kept_idxs.append(idx)
            if not prompts:
                return None, [], []
            proc_kwargs = dict(text=prompts, padding=True, return_tensors="pt")
            # Flatten images across the batch (processor handles per-sample
            # image grouping via image tokens in each prompt).
            flat_images = [im for ims in imgs_per_sample for im in ims]
            if flat_images:
                proc_kwargs["images"] = flat_images
            try:
                batch_inputs = self.processor(**proc_kwargs)
                batch_inputs = {k: v.to(model.device) if torch.is_tensor(v) else v
                                for k, v in batch_inputs.items()}
            except Exception as e:
                logger.warning(f"[rank={rank}] batch processor failed on {kept_idxs}: {e}")
                return None, [], []
            return batch_inputs, kept_samples, kept_idxs

        # Process owner's shard in chunks.
        # Gather all ZeRO-3 shards ONCE for the full eval. All ranks enter and
        # exit this collective together (their code paths don't diverge until
        # after the `with` block starts). For an 8B bf16 model this is ~16GB
        # on top of the already-gathered activations. Inside the block,
        # model.generate() is purely local — no extra NCCL collectives
        # needed, so per-rank work can freely diverge.
        with _make_gather():
            for chunk_start in range(0, len(my_indices), self.batch_size):
                chunk = my_indices[chunk_start:chunk_start + self.batch_size]

                batch_inputs, batch_samples, batch_idxs = _prepare_batch(chunk)
                if batch_inputs is None:
                    continue

                out_ids = None
                for attempt in range(2):
                    try:
                        with torch.no_grad():
                            out_ids = model.generate(
                                **batch_inputs,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=pad_id,
                                eos_token_id=eos_id,
                                synced_gpus=False,
                                **self.generation_kwargs,
                            )
                        break
                    except Exception as e:
                        msg = str(e)
                        # ZeRO-3 prefetch cache warmup may raise 'inflight params'
                        # on the first generate() even when params are gathered.
                        # One retry after a sync almost always succeeds.
                        if attempt == 0 and ("in flight" in msg or "inflight" in msg):
                            torch.cuda.synchronize()
                            out_ids = None
                            continue
                        logger.warning(
                            f"[rank={rank}] batched generate failed on {batch_idxs}: "
                            f"{msg[:200]}; falling back per-sample"
                        )
                        out_ids = None
                        break

                # Per-sample fallback on batched failure (OOM / shape issues).
                if out_ids is None:
                    for idx in batch_idxs:
                        s_inputs, s_samples, _ = _prepare_batch([idx])
                        if s_inputs is None:
                            continue
                        try:
                            with torch.no_grad():
                                oids = model.generate(
                                    **s_inputs,
                                    max_new_tokens=self.max_new_tokens,
                                    do_sample=False, num_beams=1,
                                    pad_token_id=pad_id, eos_token_id=eos_id,
                                    synced_gpus=False,
                                    **self.generation_kwargs,
                                )
                            padded = s_inputs["input_ids"].shape[1]
                            text = tokenizer.decode(oids[0, padded:], skip_special_tokens=True)
                            mm = compute_sample_metrics(text, {
                                "answer": s_samples[0].get("answer", ""),
                                "evidence_chain": s_samples[0].get("evidence_chain", []),
                            }, iou_thresh=self.iou_thresh)
                            for k in ("answer_em", "answer_contains",
                                      "bbox_localized", "parse_ok"):
                                local_metrics[k] += mm[k]
                            local_metrics["count"] += 1
                            if save_jsonl_path is not None:
                                local_records.append({
                                    "idx": int(idx),
                                    "question": s_samples[0].get("question", ""),
                                    "gold_answer": s_samples[0].get("answer", ""),
                                    "gold_evidence_chain": s_samples[0].get("evidence_chain", []),
                                    "pred_text": text,
                                    "metrics": mm,
                                })
                        except Exception as e:
                            logger.warning(f"[rank={rank}] val[{idx}] fallback failed: {e}")
                    continue

                # Decode each row. With left-padding, the generated tokens
                # start at column index = padded prompt length.
                padded_len = batch_inputs["input_ids"].shape[1]
                for bi, sample in enumerate(batch_samples):
                    gen_tokens = out_ids[bi, padded_len:]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    m = compute_sample_metrics(text, {
                        "answer": sample.get("answer", ""),
                        "evidence_chain": sample.get("evidence_chain", []),
                    }, iou_thresh=self.iou_thresh)
                    for k in ("answer_em", "answer_contains",
                              "bbox_localized", "parse_ok"):
                        local_metrics[k] += m[k]
                    local_metrics["count"] += 1
                    if save_jsonl_path is not None:
                        local_records.append({
                            "idx": int(batch_idxs[bi]),
                            "question": sample.get("question", ""),
                            "gold_answer": sample.get("answer", ""),
                            "gold_evidence_chain": sample.get("evidence_chain", []),
                            "pred_text": text,
                            "metrics": m,
                        })

        # Restore tokenizer padding side.
        tokenizer.padding_side = orig_padding_side

        dt = time.time() - t0
        logger.info(f"[rank={rank}] gen-eval local: {local_metrics} ({dt:.1f}s)")

        # All-reduce metrics across ranks (each rank processed a shard).
        if dist.is_initialized() and dist.get_world_size() > 1:
            try:
                buf = torch.tensor(
                    [local_metrics[k] for k in ("answer_em", "answer_contains",
                                                "bbox_localized", "parse_ok", "count")],
                    device=model.device, dtype=torch.float64,
                )
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                vals = buf.detach().cpu().tolist()
                total = {
                    "answer_em": vals[0], "answer_contains": vals[1],
                    "bbox_localized": vals[2], "parse_ok": vals[3], "count": vals[4],
                }
            except Exception as e:
                logger.warning(f"[rank={rank}] all_reduce failed: {e}; using rank-local")
                total = local_metrics
        else:
            total = local_metrics

        is_rank0 = int(os.environ.get("RANK", "0")) == 0
        if state is not None:
            is_rank0 = bool(state.is_world_process_zero)

        # Gather per-sample records across ranks before saving (rank 0 writes).
        if save_jsonl_path is not None and dist.is_initialized() and dist.get_world_size() > 1:
            try:
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local_records)
                all_records = [r for sub in gathered if sub for r in sub]
            except Exception as e:
                logger.warning(f"[rank={rank}] all_gather_object failed: {e}; using local")
                all_records = local_records
        else:
            all_records = local_records

        if is_rank0:
            c = max(1.0, total["count"])
            step_val = state.global_step if state is not None else -1
            report = {
                "eval_gen_answer_em": total["answer_em"] / c,
                "eval_gen_answer_contains": total["answer_contains"] / c,
                "eval_gen_bbox_localized": total["bbox_localized"] / c,
                "eval_gen_parse_ok": total["parse_ok"] / c,
                "eval_gen_samples": total["count"],
            }
            logger.info(f"[gen-eval] step={step_val} {json.dumps(report)}")
            if trainer is not None:
                try:
                    trainer.log(report)
                except Exception as e:
                    logger.warning(f"trainer.log failed: {e}")

            if save_jsonl_path is not None:
                try:
                    os.makedirs(os.path.dirname(save_jsonl_path) or ".", exist_ok=True)
                    all_records.sort(key=lambda r: r.get("idx", 0))
                    with open(save_jsonl_path, "w") as f:
                        for rec in all_records:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    summary_path = save_jsonl_path.rsplit(".", 1)[0] + "_summary.json"
                    with open(summary_path, "w") as f:
                        json.dump({**report, "num_saved": len(all_records)}, f, indent=2)
                    logger.info(
                        f"[gen-eval] wrote {len(all_records)} predictions to "
                        f"{save_jsonl_path} (+{summary_path})"
                    )
                except Exception as e:
                    logger.warning(f"failed to save predictions: {e}")

        model.train()
