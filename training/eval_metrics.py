"""Evaluation metrics for CoE: answer accuracy + relaxed bbox localization.

The model is trained to emit a JSON object:
    {"answer": "...", "evidence_chain": [{"hop":1,"image_id":"img_0","bboxes":[[x1,y1,x2,y2]], ...}, ...]}

Relaxed bbox localization: we do NOT require the predicted box to match the
ground-truth paragraph bbox exactly. A prediction counts as "correctly located"
if its IoU with the GT box is >= `iou_thresh` OR its center falls inside the GT
box. This matches the paper's evaluation intent ("locate the correct evidence
paragraph"), not pixel-perfect overlap.
"""
from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Answer normalization + EM (SQuAD-style)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def normalize_answer(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        s = ", ".join(str(x) for x in s)
    elif not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = s.translate(_PUNCT_TBL)
    s = _ARTICLES.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def answer_em(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def answer_contains(pred: str, gold: str) -> int:
    """Lenient: gold appears as substring in pred (useful for free-form)."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return 0
    return int(g in p)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_output(text: str) -> dict:
    """Try to extract the JSON object. Returns {} on failure."""
    if not text:
        return {}
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the first {...} blob (greedy)
    m = _JSON_RE.search(text)
    if not m:
        return {}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        # Try to fix trailing text after last closing brace
        depth = 0
        end = -1
        for i, c in enumerate(blob):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            try:
                return json.loads(blob[:end])
            except Exception:
                return {}
    return {}


def _clean_bbox_list(val: Any) -> List[List[float]]:
    """Normalize either [x1,y1,x2,y2] or [[...], ...] to list-of-boxes."""
    if isinstance(val, (list, tuple)) and len(val) == 4 and all(
        isinstance(x, (int, float)) for x in val
    ):
        return [[float(x) for x in val]]

    cleaned = []
    if isinstance(val, (list, tuple)):
        for b in val:
            if isinstance(b, (list, tuple)) and len(b) == 4:
                try:
                    cleaned.append([float(x) for x in b])
                except Exception:
                    continue
    return cleaned


def extract_pred_chain(parsed: dict) -> List[Dict[str, Any]]:
    """Return normalized predicted evidence hops.

    Supports two schemas:
      - Simplified single-hop:   {"bbox": [x1,y1,x2,y2], ...}
      - Multi-hop evidence_chain: {"evidence_chain": [{"bboxes": [[...]], ...}, ...]}
    """
    out = []
    if not isinstance(parsed, dict):
        return out

    # Simplified single-hop form: top-level "bbox" (or "bboxes") as list.
    for key in ("bbox", "bboxes"):
        if key in parsed:
            cleaned = _clean_bbox_list(parsed[key])
            if cleaned:
                return [{
                    "hop": 1,
                    "image_id": str(parsed.get("image_id", "") or ""),
                    "bboxes": cleaned,
                }]

    chain = parsed.get("evidence_chain", [])
    if not isinstance(chain, list):
        return out
    for i, ev in enumerate(chain):
        if not isinstance(ev, dict):
            continue
        cleaned = _clean_bbox_list(ev.get("bboxes", []))
        hop = ev.get("hop", i + 1)
        try:
            hop = int(hop)
        except Exception:
            hop = i + 1
        out.append({
            "hop": hop,
            "image_id": str(ev.get("image_id", "") or ""),
            "bboxes": cleaned,
        })
    return out


def extract_pred_bboxes(parsed: dict) -> List[Tuple[int, List[List[float]]]]:
    """Return list of (hop_idx, bboxes) from parsed output."""
    return [(ev["hop"], ev["bboxes"]) for ev in extract_pred_chain(parsed)]


# ---------------------------------------------------------------------------
# Bbox geometry
# ---------------------------------------------------------------------------

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_in(pred: List[float], gt: List[float]) -> bool:
    cx = (pred[0] + pred[2]) / 2
    cy = (pred[1] + pred[3]) / 2
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def _match(p, g, iou_thresh):
    return _iou(p, g) >= iou_thresh or _center_in(p, g)


def image_chain_stats(
    pred_chain: List[Dict[str, Any]],
    gt_chain: List[dict],
    iou_thresh: float = 0.3,
) -> dict:
    """Image-selection/order metrics for topK candidate-image evaluation."""
    gt_ids = [str(ev.get("image_id", "") or "") for ev in gt_chain]
    gt_ids = [img_id for img_id in gt_ids if img_id]
    if not gt_ids:
        return {
            "valid": 0,
            "image_hop_acc": 0.0,
            "image_chain_order_acc": 0,
            "gold_image_recall": 0.0,
            "gold_image_precision": 0.0,
            "gold_image_all_hit": 0,
            "image_bbox_joint_recall": 0.0,
            "image_bbox_joint_all_gt_hit": 0,
        }

    pred_ids = [str(ev.get("image_id", "") or "") for ev in pred_chain]
    n_hops = len(gt_ids)

    hop_hits = 0
    joint_hits = 0
    for i, gt_ev in enumerate(gt_chain[:n_hops]):
        gt_img = str(gt_ev.get("image_id", "") or "")
        pred_ev = pred_chain[i] if i < len(pred_chain) else {}
        pred_img = str(pred_ev.get("image_id", "") or "")
        if gt_img and pred_img == gt_img:
            hop_hits += 1

        gt_boxes = [
            [float(x) for x in b]
            for b in gt_ev.get("bboxes", [])
            if isinstance(b, (list, tuple)) and len(b) == 4
        ]
        pred_boxes = pred_ev.get("bboxes", []) if isinstance(pred_ev, dict) else []
        bbox_hit = False
        if not gt_boxes:
            bbox_hit = len(pred_boxes) == 0
        else:
            for p in pred_boxes:
                for g in gt_boxes:
                    if _match(p, g, iou_thresh):
                        bbox_hit = True
                        break
                if bbox_hit:
                    break
        if gt_img and pred_img == gt_img and bbox_hit:
            joint_hits += 1

    gold_unique = {img_id for img_id in gt_ids if img_id}
    pred_unique = {img_id for img_id in pred_ids if img_id}
    selected_gold = gold_unique & pred_unique

    return {
        "valid": 1,
        "image_hop_acc": hop_hits / max(1, n_hops),
        "image_chain_order_acc": int(len(pred_ids) == n_hops and pred_ids == gt_ids),
        "gold_image_recall": len(selected_gold) / max(1, len(gold_unique)),
        "gold_image_precision": len(selected_gold) / max(1, len(pred_unique)) if pred_unique else 0.0,
        "gold_image_all_hit": int(gold_unique.issubset(pred_unique)),
        "image_bbox_joint_recall": joint_hits / max(1, n_hops),
        "image_bbox_joint_all_gt_hit": int(joint_hits == n_hops),
    }


def bbox_localized(pred_bboxes: List[List[float]],
                   gt_bboxes: List[List[float]],
                   iou_thresh: float = 0.3) -> int:
    """Legacy relaxed metric: any pred vs any GT matches.
    Kept for backward compatibility; new code should use bbox_stats().
    """
    if not pred_bboxes or not gt_bboxes:
        return 0
    for p in pred_bboxes:
        for g in gt_bboxes:
            if _match(p, g, iou_thresh):
                return 1
    return 0


def bbox_stats(pred_bboxes: List[List[float]],
               gt_bboxes: List[List[float]],
               iou_thresh: float = 0.3) -> dict:
    """Per-sample bbox stats that treat EACH gt bbox independently.

    Returns:
      - recall      : fraction of gt bboxes matched by some pred bbox
      - precision   : fraction of pred bboxes that match some gt bbox
      - all_gt_hit  : 1 iff every gt bbox has a matching pred (strict localization)
      - any_hit     : 1 iff any gt matched (legacy behaviour)
      - n_gt / n_pred
    If there are no gt bboxes (unans / offtopic), returns all-ones when the
    model also emits no bboxes (correct "no-evidence" behaviour).
    """
    n_gt, n_pred = len(gt_bboxes), len(pred_bboxes)

    if n_gt == 0 and n_pred == 0:
        return {"recall": 1.0, "precision": 1.0, "all_gt_hit": 1,
                "any_hit": 1, "n_gt": 0, "n_pred": 0}
    if n_gt == 0 and n_pred > 0:
        # Model predicted bboxes when it should have abstained.
        return {"recall": 0.0, "precision": 0.0, "all_gt_hit": 0,
                "any_hit": 0, "n_gt": 0, "n_pred": n_pred}
    if n_pred == 0:
        return {"recall": 0.0, "precision": 0.0, "all_gt_hit": 0,
                "any_hit": 0, "n_gt": n_gt, "n_pred": 0}

    gt_hit = [0] * n_gt
    pred_hit = [0] * n_pred
    for i, p in enumerate(pred_bboxes):
        for j, g in enumerate(gt_bboxes):
            if _match(p, g, iou_thresh):
                gt_hit[j] = 1
                pred_hit[i] = 1
    recall = sum(gt_hit) / n_gt
    precision = sum(pred_hit) / n_pred
    return {
        "recall": recall,
        "precision": precision,
        "all_gt_hit": int(all(gt_hit)),
        "any_hit": int(any(gt_hit)),
        "n_gt": n_gt, "n_pred": n_pred,
    }


# ---------------------------------------------------------------------------
# Per-sample metric
# ---------------------------------------------------------------------------

def compute_sample_metrics(pred_text: str, sample: dict, iou_thresh: float = 0.3) -> dict:
    """
    sample must contain:
      - answer: str
      - evidence_chain: list of {bboxes: [[x1,y1,x2,y2], ...], ...}  (in resized coords)
    """
    parsed = parse_model_output(pred_text)
    pred_answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""

    em = answer_em(pred_answer, sample.get("answer", ""))
    contains = answer_contains(pred_answer, sample.get("answer", ""))

    # Flatten GT bboxes across hops
    gt_chain = sample.get("evidence_chain", [])
    gt_all: List[List[float]] = []
    for ev in gt_chain:
        for b in ev.get("bboxes", []):
            if len(b) == 4:
                gt_all.append([float(x) for x in b])

    # Flatten pred bboxes
    pred_chain = extract_pred_chain(parsed)
    pred_all: List[List[float]] = []
    for ev in pred_chain:
        pred_all.extend(ev.get("bboxes", []))

    loc = bbox_localized(pred_all, gt_all, iou_thresh=iou_thresh)
    stats = bbox_stats(pred_all, gt_all, iou_thresh=iou_thresh)
    parse_ok = int(bool(parsed))
    image_stats = image_chain_stats(pred_chain, gt_chain, iou_thresh=iou_thresh)
    if not sample.get("image_metrics_enabled", True):
        image_stats["valid"] = 0

    return {
        "answer_em": em,
        "answer_contains": contains,
        "bbox_localized": loc,               # legacy any-hit
        "bbox_recall": stats["recall"],      # fraction of GT bboxes hit
        "bbox_precision": stats["precision"],
        "bbox_all_gt_hit": stats["all_gt_hit"],
        "bbox_n_gt": stats["n_gt"],
        "bbox_n_pred": stats["n_pred"],
        "parse_ok": parse_ok,
        "image_metric_valid": image_stats["valid"],
        "image_hop_acc": image_stats["image_hop_acc"],
        "image_chain_order_acc": image_stats["image_chain_order_acc"],
        "gold_image_recall": image_stats["gold_image_recall"],
        "gold_image_precision": image_stats["gold_image_precision"],
        "gold_image_all_hit": image_stats["gold_image_all_hit"],
        "image_bbox_joint_recall": image_stats["image_bbox_joint_recall"],
        "image_bbox_joint_all_gt_hit": image_stats["image_bbox_joint_all_gt_hit"],
    }
