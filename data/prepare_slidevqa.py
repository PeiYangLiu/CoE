"""
SlideVQA Data Preparation

Converts the SlideVQA dataset into the CoE training format with
bounding box annotations and multi-hop evidence chains.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_slidevqa(data_dir: str, split: str = "train") -> List[dict]:
    """Load SlideVQA annotations from the official format."""
    ann_file = os.path.join(data_dir, f"{split}.json")
    with open(ann_file, "r") as f:
        data = json.load(f)
    return data


def convert_slidevqa_to_coe(
    raw_data: List[dict],
    image_dir: str,
) -> List[dict]:
    """
    Convert SlideVQA samples into the CoE format.
    
    SlideVQA provides:
    - question, answer
    - slide deck ID and slide indices as evidence
    - bounding box annotations on slides (if available)
    
    We reformat to match the CoE evidence chain structure.
    """
    coe_data = []

    for sample in tqdm(raw_data, desc="Converting SlideVQA"):
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        deck_id = sample.get("deck_name", sample.get("deck_id", ""))
        evidence_slides = sample.get("evidence", [])

        if not evidence_slides:
            continue

        evidence_chain = []
        all_valid = True

        for ev in evidence_slides:
            slide_idx = ev.get("slide_idx", ev.get("page", 0))
            bboxes = ev.get("bboxes", [])

            # Construct image filename
            img_name = f"{deck_id}_slide_{slide_idx}.png"
            img_path = os.path.join(image_dir, deck_id, img_name)

            # Try alternative naming conventions
            if not os.path.exists(img_path):
                alt_name = f"slide_{slide_idx:04d}.png"
                img_path = os.path.join(image_dir, deck_id, alt_name)
            if not os.path.exists(img_path):
                alt_name = f"{slide_idx}.png"
                img_path = os.path.join(image_dir, str(deck_id), alt_name)

            if not os.path.exists(img_path):
                logger.warning(f"Missing slide image: {img_path}")
                all_valid = False
                break

            img = Image.open(img_path)
            w, h = img.size

            # Validate and clip bounding boxes
            valid_bboxes = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    x1 = max(0, min(bbox[0], w))
                    y1 = max(0, min(bbox[1], h))
                    x2 = max(0, min(bbox[2], w))
                    y2 = max(0, min(bbox[3], h))
                    if x2 > x1 and y2 > y1:
                        valid_bboxes.append([x1, y1, x2, y2])

            evidence_chain.append({
                "entity": f"{deck_id}/slide_{slide_idx}",
                "screenshot": os.path.relpath(img_path, image_dir),
                "width": w,
                "height": h,
                "bboxes": valid_bboxes,
            })

        if not all_valid or not evidence_chain:
            continue

        coe_data.append({
            "question": question,
            "answer": str(answer),
            "type": sample.get("question_type", "multi_hop"),
            "num_hops": len(evidence_chain),
            "evidence_chain": evidence_chain,
        })

    return coe_data


def main():
    parser = argparse.ArgumentParser(description="Prepare SlideVQA for CoE")
    parser.add_argument("--data_dir", required=True, help="SlideVQA annotation directory")
    parser.add_argument("--image_dir", required=True, help="SlideVQA slide images directory")
    parser.add_argument("--output_dir", default="data/slidevqa", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_file = os.path.join(args.data_dir, f"{split}.json")
        if not os.path.exists(split_file):
            logger.info(f"Skipping {split} (not found)")
            continue

        raw = load_slidevqa(args.data_dir, split)
        coe = convert_slidevqa_to_coe(raw, args.image_dir)

        out_path = os.path.join(args.output_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(coe, f, indent=2)
        logger.info(f"Saved {len(coe)} {split} samples to {out_path}")


if __name__ == "__main__":
    main()
