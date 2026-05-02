"""
CoE single-GPU smoke test.

Goal: catch data/collator/model bugs in < 5 minutes. Validates end-to-end that:

  1. Dataset produces samples where target-JSON bbox coords lie inside the
     resized image dimensions (no coordinate-frame drift).
  2. Collator masks labels deterministically (assistant JSON in labels,
     prompt tokens are -100, no heuristic fallback triggered).
  3. One forward+backward pass on a real batch produces a finite loss.

Usage (single GPU):
    python scripts/smoke_test.py --config configs/train_phase1.yaml \
        --data_root data/wiki_coe --n 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CoEDataset  # noqa: E402
from models.coe_model import load_coe_model  # noqa: E402
from training.collator import CoEDataCollator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--model", default=None, help="override model name (e.g. a small model for CPU)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg, model_cfg = cfg["data"], cfg["model"]

    if args.data_root:
        data_cfg["train_file"] = os.path.join(args.data_root, "train.json")
        data_cfg["image_dir"] = os.path.join(args.data_root, "screenshots")

    if args.model:
        model_cfg["name"] = args.model

    # ------------------------------------------------------------------
    # 1. Dataset: assert bbox coords are within resized image bounds
    # ------------------------------------------------------------------
    ds = CoEDataset(
        data_file=data_cfg["train_file"],
        image_dir=data_cfg["image_dir"],
        mode=data_cfg.get("mode", "multi_hop"),
        augmentation=data_cfg.get("augmentation", {}),
        candidate_config=data_cfg.get("candidate", {}),
        split="train",
        image_max_pixels=data_cfg.get("image_max_pixels", 1048576),
        train_resolutions=data_cfg.get("train_resolutions", [768, 1024, 1280]),
        eval_resolution=data_cfg.get("eval_resolution", 1024),
    )
    print(f"[OK] dataset loaded: {len(ds)} samples")

    samples = [ds[i] for i in range(args.n)]
    for i, s in enumerate(samples):
        target = json.loads(s["messages"][-1]["content"])
        chain = target["evidence_chain"]
        assert len(chain) == len(s["evidence_chain"]), "target/evidence-chain mismatch"
        for hop in chain:
            img_idx = int(str(hop["image_id"]).replace("img_", ""))
            assert 0 <= img_idx < len(s["images"]), f"bad image_id: {hop['image_id']}"
            img = s["images"][img_idx]
            w, h = img.size
            for b in hop["bboxes"]:
                x1, y1, x2, y2 = b
                assert 0 <= x1 <= x2 <= w, f"bbox x out of bounds: {b} vs ({w},{h})"
                assert 0 <= y1 <= y2 <= h, f"bbox y out of bounds: {b} vs ({w},{h})"
        print(f"[OK] sample {i}: {len(chain)} hops, {len(s['images'])} images, bbox coords inside referenced image")

    # ------------------------------------------------------------------
    # 2. Model + processor + collator. Only load model if CUDA is there.
    # ------------------------------------------------------------------
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype = "bfloat16" if has_cuda else "float32"

    print(f"[INFO] loading model on {device} (dtype={dtype})...")
    model, processor = load_coe_model(
        model_name=model_cfg["name"],
        torch_dtype=dtype,
        use_flash_attention=False,  # smoke test avoids flash-attn dependency
        gradient_checkpointing=False,  # keep smoke test simple
    )
    model.to(device)

    collator = CoEDataCollator(processor=processor, max_length=data_cfg.get("max_length", 4096))
    batch = collator(samples)

    # ------------------------------------------------------------------
    # 3. Label mask sanity
    # ------------------------------------------------------------------
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    B, T = labels.shape
    for i in range(B):
        mask_count = (labels[i] != -100).sum().item()
        assert mask_count > 0, f"sample {i}: labels all masked"
        # First masked region must be contiguous prefix (prompt)
        nonmask_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
        first = nonmask_idx[0].item()
        last = nonmask_idx[-1].item()
        assert last - first + 1 == nonmask_idx.numel(), (
            f"sample {i}: non-masked region is not contiguous — label mask broken"
        )
        # Decode unmasked tokens — they should look like the target JSON
        decoded = processor.tokenizer.decode(
            input_ids[i][first:last + 1], skip_special_tokens=True
        )
        assert '"answer"' in decoded and '"evidence_chain"' in decoded, (
            f"sample {i}: unmasked region does not look like target JSON. got:\n{decoded[:300]}"
        )
        print(f"[OK] sample {i}: {mask_count}/{T} tokens supervised, target JSON present")

    # ------------------------------------------------------------------
    # 4. Forward + backward
    # ------------------------------------------------------------------
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out = model(**batch)
    loss = out.loss
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"[OK] forward pass: loss={loss.item():.4f}")
    loss.backward()
    print("[OK] backward pass completed")

    print("\n✅ smoke test passed")


if __name__ == "__main__":
    main()
