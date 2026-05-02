"""Two-phase curriculum training for Chain of Evidence."""

import os
import sys
import logging
import argparse
import faulthandler
import signal
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import yaml
import torch
# Fix PyTorch 2.6+ weights_only=True incompatibility with DeepSpeed
import torch.serialization
try:
    import deepspeed.runtime.fp16.loss_scaler as _ds_ls
    import deepspeed.runtime.zero.config as _ds_zc
    _safe = []
    for mod in [_ds_ls, _ds_zc]:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                _safe.append(obj)
    torch.serialization.add_safe_globals(_safe)
except Exception:
    pass
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CoEDataset
from models.coe_model import load_coe_model
from training.collator import CoEDataCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BlobMirrorCallback(TrainerCallback):
    """Mirror freshly-saved checkpoints from a local dir to a blob mount.

    Runs only on the global-rank-0 process. Uses a best-effort `cp -r` into
    the blob mount in a background thread so training is not
    blocked by blob I/O. If the mount is unavailable, logs a warning and
    continues — local checkpoints are still intact for resume.
    """

    def __init__(self, local_dir: str, blob_dir: str):
        self.local_dir = local_dir
        self.blob_dir = blob_dir

    def on_save(self, args, state, control, **kwargs):
        try:
            if int(os.environ.get("RANK", "0")) != 0:
                return
        except Exception:
            pass
        step = state.global_step
        src = os.path.join(self.local_dir, f"checkpoint-{step}")
        if not os.path.isdir(src):
            return
        import shutil, threading
        os.makedirs(self.blob_dir, exist_ok=True)
        dst = os.path.join(self.blob_dir, f"checkpoint-{step}")

        def _copy():
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
                logger.info(f"[BlobMirror] mirrored checkpoint-{step} -> blob")
                # Keep only the newest resumable checkpoint on the blob side.
                # The final exported model lives separately under `best/`.
                ckpts = sorted(
                    [d for d in os.listdir(self.blob_dir)
                     if d.startswith("checkpoint-") and
                        os.path.isdir(os.path.join(self.blob_dir, d))],
                    key=lambda x: int(x.split("-")[1]),
                )
                keep = 1
                for old in ckpts[:-keep]:
                    shutil.rmtree(os.path.join(self.blob_dir, old), ignore_errors=True)
            except Exception as e:
                logger.warning(f"[BlobMirror] copy failed: {e}")

        threading.Thread(target=_copy, daemon=True).start()


class BestCheckpointCallback(TrainerCallback):
    """Keep a full copy of the best checkpoint under `<output_dir>/best`."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str,
        greater_is_better: bool = True,
        best_dir_name: str = "best",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_dir = os.path.join(checkpoint_dir, best_dir_name)
        self.best_value = None

    def _latest_metric(self, state):
        for item in reversed(state.log_history):
            if self.metric_name in item:
                return item[self.metric_name]
        return None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.greater_is_better:
            return value > self.best_value
        return value < self.best_value

    def on_save(self, args, state, control, **kwargs):
        try:
            if int(os.environ.get("RANK", "0")) != 0:
                return
        except Exception:
            pass

        import json
        import shutil

        step = state.global_step
        src = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        if not os.path.isdir(src):
            logger.warning("[BestCheckpoint] checkpoint dir missing: %s", src)
            return

        latest_link = os.path.join(self.checkpoint_dir, "latest")
        latest_tmp = latest_link + ".tmp"
        try:
            if os.path.lexists(latest_tmp):
                os.unlink(latest_tmp)
            os.symlink(os.path.basename(src), latest_tmp)
            os.replace(latest_tmp, latest_link)
        except Exception as e:
            logger.warning("[BestCheckpoint] latest symlink update failed: %s", e)

        value = self._latest_metric(state)
        if value is None:
            logger.warning(
                "[BestCheckpoint] metric %s not found at step %s; best not updated",
                self.metric_name,
                step,
            )
            return
        try:
            value = float(value)
        except Exception:
            logger.warning(
                "[BestCheckpoint] metric %s is non-numeric (%r); best not updated",
                self.metric_name,
                value,
            )
            return

        if not self._is_better(value):
            logger.info(
                "[BestCheckpoint] step=%s %s=%.6f did not beat best=%.6f",
                step,
                self.metric_name,
                value,
                self.best_value,
            )
            return

        tmp = f"{self.best_dir}.tmp-{step}"
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        logger.info(
            "[BestCheckpoint] new best at step=%s: %s=%.6f; copying %s -> %s",
            step,
            self.metric_name,
            value,
            src,
            self.best_dir,
        )
        try:
            shutil.copytree(src, tmp, symlinks=True)
            with open(os.path.join(tmp, "best_metric.json"), "w") as f:
                json.dump(
                    {
                        "step": step,
                        "metric": self.metric_name,
                        "value": value,
                        "source_checkpoint": os.path.basename(src),
                    },
                    f,
                    indent=2,
                )
            if os.path.exists(self.best_dir):
                shutil.rmtree(self.best_dir, ignore_errors=True)
            os.replace(tmp, self.best_dir)
            self.best_value = value
            logger.info("[BestCheckpoint] best updated: %s", self.best_dir)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            logger.warning("[BestCheckpoint] best copy failed: %s", e)


def _is_valid_checkpoint(ckpt_dir: str) -> bool:
    """Heuristic completeness check for a HF/DeepSpeed checkpoint dir."""
    if not os.path.isdir(ckpt_dir):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    # DeepSpeed writes a global_step* subdir with sharded optimizer state
    has_ds = any(
        d.startswith("global_step") and os.path.isdir(os.path.join(ckpt_dir, d))
        for d in os.listdir(ckpt_dir)
    )
    return has_ds


def _find_resume_checkpoint(*search_dirs: str) -> Optional[str]:
    """Return the newest VALID checkpoint across the given directories."""
    best_step = -1
    best_path = None
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.startswith("checkpoint-"):
                continue
            path = os.path.join(d, name)
            try:
                step = int(name.split("-")[1])
            except Exception:
                continue
            if step > best_step and _is_valid_checkpoint(path):
                best_step = step
                best_path = path
    return best_path


def _prune_old_checkpoints(output_dir: str, keep: int = 1) -> None:
    """Keep only the newest `checkpoint-*` dirs; never touches `best/`."""
    if not output_dir or not os.path.isdir(output_dir):
        return
    ckpts = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-")[1])
        except Exception:
            continue
        ckpts.append((step, path))
    ckpts.sort()
    import shutil
    for _, path in ckpts[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def create_deepspeed_config(cfg: dict) -> dict:
    """Generate DeepSpeed config from training config."""
    ds_cfg = cfg.get("deepspeed", {})
    zero_stage = ds_cfg.get("zero_stage", 3)

    zero_opt = {
        "stage": zero_stage,
        "offload_optimizer": {
            "device": "cpu" if ds_cfg.get("offload_optimizer") else "none",
        },
        "offload_param": {
            "device": "cpu" if ds_cfg.get("offload_param") else "none",
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": ds_cfg.get("reduce_bucket_size", 5e8),
        "allgather_bucket_size": ds_cfg.get("allgather_bucket_size", 5e8),
        "stage3_prefetch_bucket_size": ds_cfg.get("stage3_prefetch_bucket_size", 5e8),
        "stage3_param_persistence_threshold": ds_cfg.get("stage3_param_persistence_threshold", 1e6),
        "stage3_max_live_parameters": ds_cfg.get("stage3_max_live_parameters", 1e9),
        "stage3_max_reuse_distance": ds_cfg.get("stage3_max_reuse_distance", 1e9),
        "stage3_gather_16bit_weights_on_model_save": True,
    }

    config = {
        "zero_optimization": zero_opt,
        "bf16": {"enabled": True},
        "gradient_clipping": cfg["training"].get("max_grad_norm", 1.0),
        "train_micro_batch_size_per_gpu": cfg["training"]["per_device_batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "wall_clock_breakdown": False,
    }
    return config


def train(config_path: str, local_rank: int = -1,
          data_root: str = None, output_dir: str = None, model_cache: str = None):
    """Main training function."""
    cfg = load_config(config_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Override paths from CLI args.
    if data_root:
        # Prefer pre-expanded single-hop (fast init), then *_clean (filtered),
        # finally the raw multi-hop JSON.
        def _pick(*names):
            for n in names:
                p = os.path.join(data_root, n)
                if os.path.exists(p):
                    return p
            return os.path.join(data_root, names[-1])
        # Prefer the 95/5 train/val split (split at multi-hop level to avoid
        # hop-level leakage); fall back to pre-split single-hop / clean / raw.
        data_cfg["train_file"] = _pick(
            "train_split_single_hop.json", "train_single_hop.json",
            "train_clean.json", "train.json",
        )
        data_cfg["val_file"]   = _pick(
            "val_split_single_hop.json", "val_split.json",
            "test_single_hop.json", "test_clean.json", "test.json",
        )
        data_cfg["image_dir"] = os.path.join(data_root, "screenshots")
    if output_dir:
        train_cfg["output_dir"] = output_dir
    if model_cache:
        os.environ["HF_HOME"] = model_cache

    # Dump a Python backtrace from every process if we appear hung. Write
    # to stderr so it lands in std_log_process_*.txt which is mirrored to
    # blob. SIGUSR1 triggers on-demand dumps too (`kill -USR1 <pid>`).
    # Periodic auto-dump is OFF by default now that init hangs are fixed;
    # set FAULTHANDLER_INTERVAL=<sec> to re-enable for debugging.
    rank = int(os.environ.get("RANK", "0"))
    try:
        faulthandler.enable(file=sys.stderr)
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        _fh_interval = int(os.environ.get("FAULTHANDLER_INTERVAL", "0"))
        if _fh_interval > 0:
            faulthandler.dump_traceback_later(_fh_interval, repeat=True, file=sys.stderr, exit=False)
            logger.info(f"[rank={rank}] faulthandler armed (stderr, {_fh_interval}s interval)")
        else:
            logger.info(f"[rank={rank}] faulthandler armed (stderr, SIGUSR1 only)")
    except Exception as e:
        logger.warning(f"faulthandler setup failed: {e}")

    # For Phase 2, load Phase 1 full finetune checkpoint
    phase1_ckpt = os.environ.get("PHASE1_CKPT")
    if phase1_ckpt and os.path.exists(phase1_ckpt):
        # Full finetune: load from Phase 1 checkpoint directory instead of base model
        logger.info(f"Loading Phase 1 checkpoint from: {phase1_ckpt}")
        model_cfg["name"] = phase1_ckpt

    # Pre-initialize the torch process group with a LONG watchdog timeout
    # (default is 10 min, which is too short: straggler ranks on variable
    # image sizes + blobfuse I/O jitter can occasionally take >10min on a
    # single training step, causing NCCL reduce_scatter to time out and the
    # whole job to abort mid-training). 1 hour is defensive but safe.
    try:
        import torch.distributed as _dist
        if not _dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            _dist.init_process_group(backend=backend, timeout=timedelta(seconds=3600))
            logger.info(
                f"[rank={os.environ.get('RANK','?')}] initialized process group "
                f"with backend={backend}, timeout=3600s"
            )
    except Exception as e:
        logger.warning(f"Explicit init_process_group failed (will fall back to default): {e}")

    # Build DeepSpeed config first and install it as the HF "zero3 init"
    # context BEFORE from_pretrained. With ZeRO-3, each rank only allocates
    # its parameter shard during from_pretrained, avoiding 32 ranks
    # concurrently materializing the full 16GB model on CPU (which can
    # stall disk I/O or hit CPU-RAM pressure).
    ds_config = create_deepspeed_config(cfg)
    ds_config_dir = os.environ.get("LOCAL_CKPT_DIR", train_cfg["output_dir"])
    os.makedirs(ds_config_dir, exist_ok=True)
    ds_config_path = os.path.join(ds_config_dir, "ds_config.json")
    import json
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)

    _ds_hf_config = None
    if ds_config.get("zero_optimization", {}).get("stage", 0) == 3:
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
        except ImportError:
            from transformers.deepspeed import HfDeepSpeedConfig  # older path
        # Must keep reference alive until after from_pretrained returns.
        _ds_hf_config = HfDeepSpeedConfig(ds_config)
        logger.info(f"[rank={rank}] HfDeepSpeedConfig installed (ZeRO-3 init context active)")

    # Load model
    logger.info(f"[rank={rank}] calling load_coe_model(...) at {time.time():.1f}")
    t0 = time.time()
    model, processor = load_coe_model(
        model_name=model_cfg["name"],
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        use_flash_attention=model_cfg.get("use_flash_attention", True),
        gradient_checkpointing=False,  # handled by Trainer below with use_reentrant=False
    )
    logger.info(f"[rank={rank}] load_coe_model done in {time.time()-t0:.1f}s")

    # Prepare augmentation config
    aug_cfg = data_cfg.get("augmentation", {})
    candidate_cfg = data_cfg.get("candidate", {})

    # Load datasets
    train_dataset = CoEDataset(
        data_file=data_cfg["train_file"],
        image_dir=data_cfg["image_dir"],
        mode=data_cfg.get("mode", "multi_hop"),
        augmentation=aug_cfg,
        candidate_config=candidate_cfg,
        split="train",
        image_max_pixels=data_cfg.get("image_max_pixels", 1048576),
        train_resolutions=data_cfg.get("train_resolutions", [768, 1024, 1280]),
        eval_resolution=data_cfg.get("eval_resolution", 1024),
        single_hop_include_reasoning=data_cfg.get("single_hop_include_reasoning", True),
    )

    # Validation: 5% holdout from train (split at multi-hop level). Kept small
    # (~4.7k single-hop samples) so periodic eval doesn't dominate training time.
    val_file = data_cfg.get("val_file")
    if val_file and os.path.exists(val_file):
        val_dataset = CoEDataset(
            data_file=val_file,
            image_dir=data_cfg["image_dir"],
            mode=data_cfg.get("mode", "multi_hop"),
            augmentation={},  # no augmentation for eval
            candidate_config=candidate_cfg,
            split="val",
            image_max_pixels=data_cfg.get("image_max_pixels", 1048576),
            train_resolutions=data_cfg.get("train_resolutions", [768, 1024, 1280]),
            eval_resolution=data_cfg.get("eval_resolution", 1024),
            single_hop_include_reasoning=data_cfg.get("single_hop_include_reasoning", True),
        )
        logger.info(f"[rank={rank}] val_dataset loaded: {len(val_dataset)} samples from {val_file}")
        # HF Trainer's built-in eval computes eval_loss over the entire
        # eval_dataset each time evaluate() fires. With tall screenshots at
        # 8M-pixel resolution this costs ~20 min per eval pass and dominates
        # wall time. We only use HF's evaluate() as a trigger for our generative
        # callback, so shrink the dataset passed to Trainer to a tiny subset.
        # The callback iterates the full val_dataset independently.
        import torch.utils.data as tud
        hf_eval_subset = tud.Subset(val_dataset, list(range(min(16, len(val_dataset)))))
    else:
        val_dataset = None
        hf_eval_subset = None
        logger.info(f"[rank={rank}] no val_file found; skipping validation")

    # Data collator (resizing is done inside the dataset now)
    collator = CoEDataCollator(
        processor=processor,
        max_length=data_cfg.get("max_length", 4096),
    )

    # ------------------------------------------------------------------
    # Checkpoint strategy: by default write directly to the configured
    # output_dir. Cloud jobs can opt into fast local scratch by setting
    # LOCAL_CKPT_DIR, in which case BlobMirrorCallback mirrors to output_dir.
    # ------------------------------------------------------------------
    blob_output_dir = train_cfg["output_dir"]
    local_output_dir = os.environ.get("LOCAL_CKPT_DIR", blob_output_dir)
    os.makedirs(local_output_dir, exist_ok=True)
    # If blob and local resolve to the SAME directory, disable mirroring to
    # avoid the rmtree+copytree race that silently wipes checkpoints.
    try:
        if blob_output_dir and os.path.realpath(os.path.abspath(blob_output_dir)) == os.path.realpath(local_output_dir):
            logger.info("blob_output_dir == local_output_dir (realpath); disabling BlobMirror.")
            blob_output_dir = local_output_dir
    except Exception:
        pass
    logger.info(f"Local ckpt dir: {local_output_dir}")
    logger.info(f"Blob mirror dir: {blob_output_dir}")

    if train_cfg.get("save_total_limit", 1) != 1:
        logger.info(
            "Using save_total_limit=%s from config",
            train_cfg.get("save_total_limit"),
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=True,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 500) if val_dataset else None,
        eval_on_start=bool(val_dataset) and train_cfg.get("eval_on_start", True),
        load_best_model_at_end=False,
        deepspeed=ds_config_path,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 40),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=train_cfg.get("dataloader_prefetch_factor", 4),
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", ["swanlab"]),
        run_name=train_cfg.get("run_name", None),
        ddp_timeout=train_cfg.get("ddp_timeout", 3600),  # align with torch PG timeout above
        save_total_limit=train_cfg.get("save_total_limit", 1),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Callbacks
    callbacks = []
    if blob_output_dir and blob_output_dir != local_output_dir:
        callbacks.append(BlobMirrorCallback(local_output_dir, blob_output_dir))
    best_ckpt_cfg = train_cfg.get("best_checkpoint", {})
    if best_ckpt_cfg.get("enabled", False):
        callbacks.append(
            BestCheckpointCallback(
                checkpoint_dir=local_output_dir,
                metric_name=best_ckpt_cfg.get("metric", "eval_gen_answer_em"),
                greater_is_better=best_ckpt_cfg.get("greater_is_better", True),
                best_dir_name=best_ckpt_cfg.get("dir_name", "best"),
            )
        )

    # Generative eval callback: answer EM + relaxed bbox localization
    eval_cb = None
    eval_cfg = train_cfg.get("gen_eval", {})
    if val_dataset is not None and eval_cfg.get("enabled", True):
        from training.eval_callback import GenerativeEvalCallback
        gen_extra = {}
        if "repetition_penalty" in eval_cfg:
            gen_extra["repetition_penalty"] = float(eval_cfg["repetition_penalty"])
        eval_cb = GenerativeEvalCallback(
            processor=processor,
            val_dataset=val_dataset,
            max_samples=eval_cfg.get("max_samples", 256),
            max_new_tokens=eval_cfg.get("max_new_tokens", 256),
            iou_thresh=eval_cfg.get("iou_thresh", 0.3),
            batch_size=eval_cfg.get("batch_size", 4),
            generation_kwargs=gen_extra or None,
            save_dir=eval_cfg.get("save_dir", os.path.join(blob_output_dir or local_output_dir, "eval_preds")),
        )
        callbacks.append(eval_cb)
        logger.info(
            f"Generative eval callback enabled: max_samples={eval_cb.max_samples} "
            f"bs={eval_cb.batch_size} save_dir={eval_cb.save_dir} "
            f"iou_thresh={eval_cb.iou_thresh} max_new_tokens={eval_cb.max_new_tokens}"
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=hf_eval_subset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if eval_cb is not None:
        eval_cb.set_trainer(trainer)

    # Check for existing checkpoint to resume from (local first, blob fallback)
    resume_from = _find_resume_checkpoint(local_output_dir, blob_output_dir)
    if resume_from:
        # If the valid checkpoint is only on blob, stage it to local first so
        # DeepSpeed resumes from fast storage.
        if resume_from.startswith(blob_output_dir):
            import shutil
            staged = os.path.join(local_output_dir, os.path.basename(resume_from))
            if not os.path.isdir(staged):
                logger.info(f"Staging blob checkpoint {resume_from} -> {staged}")
                shutil.copytree(resume_from, staged)
            resume_from = staged
        logger.info(f"Resuming from checkpoint: {resume_from}")
    else:
        logger.info("No valid checkpoint found; starting fresh.")

    # Train
    logger.info(f"Starting training: {train_cfg['num_epochs']} epochs")
    if train_cfg.get("max_steps", -1) > 0:
        logger.info(f"  Max steps override: {train_cfg['max_steps']}")
    logger.info(f"  Mode: {data_cfg.get('mode', 'multi_hop')}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Batch size: {train_cfg['per_device_batch_size']} × {train_cfg['gradient_accumulation_steps']} (grad accum)")
    logger.info(f"  Learning rate: {train_cfg['learning_rate']}")

    trainer.train(resume_from_checkpoint=resume_from)

    # Full val-set inference (post-training): save all predictions + metrics.
    if eval_cb is not None and val_dataset is not None:
        try:
            preds_dir = os.path.join(local_output_dir, "final_eval")
            os.makedirs(preds_dir, exist_ok=True)
            save_path = os.path.join(preds_dir, "val_predictions.jsonl")
            logger.info(f"Running full val-set inference ({len(val_dataset)} samples)")
            eval_cb.run_full_eval(trainer.model, save_jsonl_path=save_path,
                                  state=trainer.state)
            if int(os.environ.get("RANK", "0")) == 0 and blob_output_dir != local_output_dir:
                import shutil
                blob_preds = os.path.join(blob_output_dir, "final_eval")
                if os.path.exists(blob_preds):
                    shutil.rmtree(blob_preds, ignore_errors=True)
                shutil.copytree(preds_dir, blob_preds)
                logger.info(f"Mirrored final_eval to {blob_preds}")
        except Exception as e:
            logger.warning(f"Final full-val eval failed: {e}")

    if int(os.environ.get("RANK", "0")) == 0:
        _prune_old_checkpoints(local_output_dir, keep=1)

    if train_cfg.get("save_final_model", True):
        # Save final model — write locally, then mirror to blob.
        final_dir_name = train_cfg.get("final_model_dir", "best")
        final_local = os.path.join(local_output_dir, final_dir_name)
        trainer.save_model(final_local)
        processor.save_pretrained(final_local)
        logger.info(f"Saved final model locally to {final_local}")
        if int(os.environ.get("RANK", "0")) == 0:
            _prune_old_checkpoints(local_output_dir, keep=1)
        try:
            if int(os.environ.get("RANK", "0")) == 0 and blob_output_dir != local_output_dir:
                import shutil
                final_blob = os.path.join(blob_output_dir, final_dir_name)
                if os.path.exists(final_blob):
                    shutil.rmtree(final_blob, ignore_errors=True)
                shutil.copytree(final_local, final_blob)
                logger.info(f"Mirrored final model to {final_blob}")
        except Exception as e:
            logger.warning(f"Failed to mirror final model to blob: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CoE model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_root", type=str, default=None, help="Override data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--model_cache", type=str, default=None, help="HF model cache dir")
    args = parser.parse_args()

    train(args.config, args.local_rank, args.data_root, args.output_dir, args.model_cache)
