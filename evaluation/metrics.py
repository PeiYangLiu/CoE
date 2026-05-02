"""
CoE Evaluation Metrics

Implements the three evaluation dimensions described in the paper:
1. Exact Match (EM) - answer accuracy
2. Localization Accuracy (Loc-Acc) - IoU >= 0.7 for bounding boxes
3. Chain Accuracy (Chain-Acc) - complete reasoning chain correctness
4. Retrieval metrics (Recall@k, MRR) for the retrieval component
"""

import re
import string
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer accuracy (Exact Match)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (lowercased, no articles/punct)."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(c for c in s if c not in string.punctuation)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match between predicted and ground truth answers."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Bounding box IoU and Localization Accuracy
# ---------------------------------------------------------------------------

def compute_iou(bbox_pred: List[int], bbox_gt: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Each bbox is [x1, y1, x2, y2].
    """
    x1 = max(bbox_pred[0], bbox_gt[0])
    y1 = max(bbox_pred[1], bbox_gt[1])
    x2 = min(bbox_pred[2], bbox_gt[2])
    y2 = min(bbox_pred[3], bbox_gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = max(0, bbox_pred[2] - bbox_pred[0]) * max(0, bbox_pred[3] - bbox_pred[1])
    area_gt = max(0, bbox_gt[2] - bbox_gt[0]) * max(0, bbox_gt[3] - bbox_gt[1])
    union = area_pred + area_gt - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def localization_accuracy(
    pred_bboxes: List[List[int]],
    gt_bboxes: List[List[int]],
    iou_threshold: float = 0.7,
) -> Tuple[float, List[float]]:
    """
    Compute localization accuracy using greedy IoU matching.
    
    For each ground-truth bbox, find the best matching predicted bbox.
    A localization is correct if IoU >= threshold.
    
    Args:
        pred_bboxes: list of predicted [x1, y1, x2, y2]
        gt_bboxes: list of ground-truth [x1, y1, x2, y2]
        iou_threshold: IoU threshold (default 0.7)
        
    Returns:
        (accuracy, list of per-bbox IoU scores)
    """
    if not gt_bboxes:
        return 1.0 if not pred_bboxes else 0.0, []

    if not pred_bboxes:
        return 0.0, [0.0] * len(gt_bboxes)

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)))
    for i, gt in enumerate(gt_bboxes):
        for j, pred in enumerate(pred_bboxes):
            iou_matrix[i, j] = compute_iou(pred, gt)

    # Greedy matching: for each GT, find best unmatched pred
    matched_ious = []
    used_preds = set()

    for i in range(len(gt_bboxes)):
        best_iou = 0.0
        best_j = -1
        for j in range(len(pred_bboxes)):
            if j not in used_preds and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j

        matched_ious.append(best_iou)
        if best_j >= 0:
            used_preds.add(best_j)

    correct = sum(1 for iou in matched_ious if iou >= iou_threshold)
    accuracy = correct / len(gt_bboxes)
    return accuracy, matched_ious


# ---------------------------------------------------------------------------
# Chain Accuracy
# ---------------------------------------------------------------------------

def chain_accuracy(
    pred_chain: List[dict],
    gt_chain: List[dict],
    iou_threshold: float = 0.7,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute chain accuracy: all hops must be correctly localized.
    
    Args:
        pred_chain: list of predicted evidence steps
            [{"image_id": "img_0", "bboxes": [[x1,y1,x2,y2]]}]
        gt_chain: list of ground-truth evidence steps
            [{"entity": "...", "bboxes": [[x1,y1,x2,y2]]}]
            
    Returns:
        (chain_acc, details) where chain_acc is 1.0 if all hops correct
    """
    if len(gt_chain) == 0:
        return 1.0, {"per_hop": []}

    per_hop_results = []

    for hop_idx, gt_step in enumerate(gt_chain):
        gt_bboxes = gt_step.get("bboxes", [])

        # Match to predicted chain by hop index
        if hop_idx < len(pred_chain):
            pred_step = pred_chain[hop_idx]
            pred_bboxes = pred_step.get("bboxes", [])
        else:
            pred_bboxes = []

        hop_acc, hop_ious = localization_accuracy(pred_bboxes, gt_bboxes, iou_threshold)
        per_hop_results.append({
            "hop": hop_idx + 1,
            "accuracy": hop_acc,
            "ious": hop_ious,
            "correct": hop_acc == 1.0,
        })

    # Chain is correct only if ALL hops are correct
    all_correct = all(h["correct"] for h in per_hop_results)
    return float(all_correct), {"per_hop": per_hop_results}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_entities: List[str],
    gold_entities: List[str],
    k: int = 5,
) -> float:
    """Compute Recall@k: fraction of gold entities in top-k retrieved."""
    if not gold_entities:
        return 1.0
    retrieved_set = set(retrieved_entities[:k])
    gold_set = set(gold_entities)
    return len(retrieved_set & gold_set) / len(gold_set)


def mean_reciprocal_rank(
    retrieved_entities: List[str],
    gold_entities: List[str],
) -> float:
    """Compute MRR: reciprocal rank of first relevant result."""
    gold_set = set(gold_entities)
    for rank, entity in enumerate(retrieved_entities, 1):
        if entity in gold_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_coe(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Run full CoE evaluation on a set of predictions.
    
    Args:
        predictions: list of parsed model outputs
            [{"answer": "...", "evidence_chain": [...]}]
        ground_truths: list of ground-truth samples
            [{"answer": "...", "evidence_chain": [...]}]
            
    Returns:
        Dict with EM, F1, Loc-Acc, Chain-Acc metrics.
    """
    assert len(predictions) == len(ground_truths), (
        f"Prediction count ({len(predictions)}) != GT count ({len(ground_truths)})"
    )

    em_scores = []
    f1_scores = []
    loc_accs = []
    chain_accs = []
    per_type_metrics: Dict[str, Dict[str, list]] = {}

    for pred, gt in zip(predictions, ground_truths):
        # Answer metrics
        em = exact_match(pred.get("answer", ""), gt.get("answer", ""))
        f1 = f1_score(pred.get("answer", ""), gt.get("answer", ""))
        em_scores.append(em)
        f1_scores.append(f1)

        # Chain accuracy
        pred_chain = pred.get("evidence_chain", [])
        gt_chain = gt.get("evidence_chain", [])
        c_acc, c_details = chain_accuracy(pred_chain, gt_chain, iou_threshold)
        chain_accs.append(c_acc)

        # Per-hop localization accuracy
        for hop_info in c_details.get("per_hop", []):
            loc_accs.append(hop_info["accuracy"])

        # Per question type
        q_type = gt.get("type", "unknown")
        if q_type not in per_type_metrics:
            per_type_metrics[q_type] = {"em": [], "chain_acc": [], "loc_acc": []}
        per_type_metrics[q_type]["em"].append(em)
        per_type_metrics[q_type]["chain_acc"].append(c_acc)
        for hop_info in c_details.get("per_hop", []):
            per_type_metrics[q_type]["loc_acc"].append(hop_info["accuracy"])

    results = {
        "EM": np.mean(em_scores) * 100,
        "F1": np.mean(f1_scores) * 100,
        "Loc-Acc": np.mean(loc_accs) * 100 if loc_accs else 0.0,
        "Chain-Acc": np.mean(chain_accs) * 100,
        "num_samples": len(predictions),
    }

    # Per-type breakdown
    for q_type, metrics in per_type_metrics.items():
        results[f"EM_{q_type}"] = np.mean(metrics["em"]) * 100
        results[f"Chain-Acc_{q_type}"] = np.mean(metrics["chain_acc"]) * 100
        if metrics["loc_acc"]:
            results[f"Loc-Acc_{q_type}"] = np.mean(metrics["loc_acc"]) * 100

    return results
