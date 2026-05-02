"""
Visualization utilities for Chain of Evidence

Draws bounding boxes on document screenshots to visualize
the evidence chain for a given prediction.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Distinct colors for different hops
HOP_COLORS = [
    (255, 0, 0),      # red
    (0, 128, 255),     # blue
    (0, 200, 0),       # green
    (255, 165, 0),     # orange
    (148, 0, 211),     # purple
    (255, 20, 147),    # pink
    (0, 206, 209),     # cyan
    (255, 215, 0),     # gold
]


def draw_evidence_chain(
    images: List[Image.Image],
    evidence_chain: List[dict],
    question: str = "",
    answer: str = "",
    line_width: int = 3,
    font_size: int = 16,
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Visualize the evidence chain by drawing bounding boxes on document images.
    
    Args:
        images: list of document screenshot images
        evidence_chain: list of evidence steps with bboxes
        question: the query (for title)
        answer: the predicted answer (for title)
        line_width: bbox line width
        font_size: text font size
        output_path: if provided, save the visualization
        
    Returns:
        Combined visualization image.
    """
    annotated = []

    for step in evidence_chain:
        hop = step.get("hop", len(annotated) + 1)
        image_id = step.get("image_id", f"img_{hop - 1}")
        bboxes = step.get("bboxes", [])
        sub_query = step.get("sub_query", "")

        # Determine which image this step refers to
        img_idx = _parse_img_id(image_id)
        if img_idx is None or img_idx >= len(images):
            continue

        img = images[img_idx].copy()
        draw = ImageDraw.Draw(img)
        color = HOP_COLORS[hop % len(HOP_COLORS)]

        # Draw bounding boxes
        for bbox in bboxes:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Draw rectangle with thick border
                for offset in range(line_width):
                    draw.rectangle(
                        [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                        outline=color,
                    )

                # Draw semi-transparent fill
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                fill_color = (*color, 40)  # alpha=40
                overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(img)

        # Draw hop label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        label = f"Hop {hop}"
        if sub_query:
            label += f": {sub_query[:50]}"
        draw.text((10, 10), label, fill=color, font=font)

        annotated.append(img)

    if not annotated:
        return Image.new("RGB", (800, 100), (255, 255, 255))

    # Combine images horizontally with spacing
    _combine_with_header(annotated, question, answer)

    total_width = sum(img.width for img in annotated) + 20 * (len(annotated) - 1)
    max_height = max(img.height for img in annotated)

    # Add header space
    header_height = 60
    combined = Image.new("RGB", (total_width, max_height + header_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Draw header
    try:
        header_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except (IOError, OSError):
        header_font = ImageFont.load_default()

    if question:
        draw.text((10, 5), f"Q: {question[:120]}", fill=(0, 0, 0), font=header_font)
    if answer:
        draw.text((10, 30), f"A: {answer}", fill=(0, 128, 0), font=header_font)

    # Paste annotated images
    x_offset = 0
    for img in annotated:
        combined.paste(img, (x_offset, header_height))
        x_offset += img.width + 20

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.save(output_path)
        logger.info(f"Saved visualization to {output_path}")

    return combined


def _combine_with_header(images, question, answer):
    """Helper to scale images to uniform height."""
    if not images:
        return
    target_height = min(img.height for img in images)
    target_height = min(target_height, 800)
    for i, img in enumerate(images):
        if img.height != target_height:
            scale = target_height / img.height
            new_w = int(img.width * scale)
            images[i] = img.resize((new_w, target_height), Image.LANCZOS)


def _parse_img_id(image_id: str) -> Optional[int]:
    if image_id and image_id.startswith("img_"):
        try:
            return int(image_id.split("_")[1])
        except (IndexError, ValueError):
            pass
    return None


def visualize_batch(
    results_file: str,
    image_dir: str,
    output_dir: str,
    max_samples: int = 50,
):
    """
    Visualize a batch of predictions from an evaluation results file.
    
    Args:
        results_file: path to eval_results.json
        image_dir: directory containing screenshots
        output_dir: where to save visualizations
        max_samples: max number of samples to visualize
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    predictions = results.get("predictions", [])
    os.makedirs(output_dir, exist_ok=True)

    for idx, pred in enumerate(predictions[:max_samples]):
        evidence_chain = pred.get("evidence_chain", [])
        if not evidence_chain:
            continue

        # Load images
        images = []
        for step in evidence_chain:
            screenshot = step.get("screenshot", "")
            path = os.path.join(image_dir, screenshot) if screenshot else ""
            if path and os.path.exists(path):
                images.append(Image.open(path).convert("RGB"))
            else:
                images.append(Image.new("RGB", (800, 600), (240, 240, 240)))

        draw_evidence_chain(
            images=images,
            evidence_chain=evidence_chain,
            question=pred.get("question", ""),
            answer=pred.get("answer", ""),
            output_path=os.path.join(output_dir, f"sample_{idx:04d}.png"),
        )

    logger.info(f"Saved {min(len(predictions), max_samples)} visualizations to {output_dir}")
