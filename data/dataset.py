"""
CoE Dataset Classes

PyTorch datasets for Wiki-CoE and SlideVQA with:
- Single-hop and multi-hop data formatting
- Data augmentation (spatial, resolution variation, evidence permutation)
- Prompt construction for Qwen3-VL conversation format
"""

import os
import json
import random
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a visual evidence localization assistant. Given a question and a set of candidate document screenshots, you must:
1. Select only the candidate image(s) that contain evidence needed to answer the question. Some candidates may be distractors and should not appear in the evidence chain.
2. Output the evidence_chain in the logical reasoning order required by the question, not in the order the images are shown.
3. For each reasoning step (hop), identify the correct candidate image_id and provide precise bounding boxes [x1, y1, x2, y2] in pixel coordinates of that input image.
4. Along with each hop, write a short natural-language sub-query describing what information you are looking for at that step.
5. After locating all evidence, produce the final answer by reading and combining information from the bounded regions.

Output your response as a JSON object with the following structure (evidence_chain FIRST, then answer):
{
  "evidence_chain": [
    {"hop": 1, "image_id": "img_0", "bboxes": [[x1, y1, x2, y2]], "sub_query": "what you are looking for at hop 1"},
    {"hop": 2, "image_id": "img_1", "bboxes": [[x1, y1, x2, y2]], "sub_query": "what you are looking for at hop 2"}
  ],
  "answer": "your final answer"
}

Bounding box coordinates are in pixel space of the selected input image (top-left origin, +x right, +y down). Each candidate image is labeled as img_0, img_1, etc. in the order it appears. Use those exact image_id labels."""

SINGLE_HOP_SYSTEM_PROMPT = """You are a visual evidence localization assistant. Given a question and a document screenshot, you must:
1. Briefly reason (1-3 sentences) about where on the slide the evidence is and how you derive the answer from it. Omit the reasoning (use an empty string) only when the answer is trivial from the slide and needs no explanation.
2. Identify ALL evidence regions needed to answer the question and provide their bounding boxes [x1, y1, x2, y2] in pixel coordinates of the input image (top-left origin, +x right, +y down). If the question requires combining multiple pieces of evidence, list one bbox per piece. If the question cannot be answered from the image, return an empty list [].
3. Read the text inside those regions and produce the final answer. If the question cannot be answered from the image (e.g., information is not on the slide, or the question is unrelated to the slide), respond with a brief refusal.

Output your response as a JSON object with this exact key order (reasoning FIRST, bbox MIDDLE, answer LAST):
{"reasoning": "your brief reasoning", "bbox": [[x1, y1, x2, y2], ...], "answer": "your answer"}"""

SINGLE_HOP_SYSTEM_PROMPT_NO_REASONING = """You are a visual evidence localization assistant. Given a question and a document screenshot, you must:
1. Identify ALL evidence regions needed to answer the question and provide their bounding boxes [x1, y1, x2, y2] in pixel coordinates of the input image (top-left origin, +x right, +y down). If the question requires combining multiple pieces of evidence, list one bbox per piece. If the question cannot be answered from the image, return an empty list [].
2. Read the text inside those regions and produce the final answer. If the question cannot be answered from the image (e.g., information is not on the slide, or the question is unrelated to the slide), respond with a brief refusal.

Output your response as a JSON object with this exact key order (bbox FIRST, answer LAST):
{"bbox": [[x1, y1, x2, y2], ...], "answer": "your answer"}"""


def format_user_message_single_hop(question: str, image_path: str) -> List[dict]:
    """Format a single-hop question with one image for Qwen3-VL."""
    return [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"Question: {question}\n\nLocate the evidence in the document and answer the question."},
    ]


MULTI_HOP_USER_INSTRUCTION = (
    "Analyze the candidate documents step by step. Some images may be irrelevant "
    "distractors. For each reasoning hop, choose the relevant candidate image "
    "(img_0, img_1, ...), locate the evidence with bounding boxes, and keep the "
    "evidence_chain in the logical order needed to answer the question. Then "
    "provide the final answer."
)


def format_user_message_multi_hop(question: str, image_paths: List[str]) -> List[dict]:
    """Format a multi-hop question with multiple images for Qwen3-VL."""
    content = []
    for i, path in enumerate(image_paths):
        content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": f"[img_{i}]"})

    content.append({
        "type": "text",
        "text": (
            f"\nQuestion: {question}\n\n"
            f"{MULTI_HOP_USER_INSTRUCTION}"
        ),
    })
    return content


def format_target(
    answer: str,
    evidence_chain: List[dict],
    image_id_map: Dict[str, str],
    *,
    single_hop: bool = False,
    reasoning: str = "",
    include_reasoning: bool = True,
) -> str:
    """
    Format the target output as a JSON string.

    For single_hop=True we emit a MUCH shorter form to avoid the model spending
    most of its loss budget on scaffold tokens (hop/image_id/sub_query).
    Default shape (include_reasoning=True): {"reasoning": "...", "bbox": [...], "answer": "..."}
    With include_reasoning=False:           {"bbox": [...], "answer": "..."}

    For multi-hop we keep the richer evidence_chain structure.

    Args:
        answer: final answer string
        evidence_chain: list of evidence dicts from dataset
        image_id_map: mapping from entity/screenshot key to img_N id (multi-hop)
        single_hop: emit simplified single-hop schema
        reasoning: optional chain-of-thought rationale (single-hop only); empty string
            means "no reasoning needed", which is still explicitly serialized to
            keep the output schema consistent across samples.
        include_reasoning: when False (single-hop only), drop the reasoning field
            from the target entirely. Used by Wiki-CoE Phase1 where training
            samples have no reasoning labels and forcing "reasoning":""
            supervision was found to suppress the model's CoT and hurt EM.
    """
    if single_hop:
        first = evidence_chain[0]
        bboxes = first.get("bboxes", []) or []
        # Always emit as list-of-bboxes to keep a single consistent schema,
        # preserving ALL evidence bboxes (previously only bboxes[0] was kept,
        # which destroyed supervision on multi-evidence / arithmetic questions).
        bboxes_clean = [list(b) for b in bboxes if isinstance(b, (list, tuple)) and len(b) == 4]
        if include_reasoning:
            target = {"reasoning": reasoning or "", "bbox": bboxes_clean, "answer": answer}
        else:
            target = {"bbox": bboxes_clean, "answer": answer}
        return json.dumps(target, ensure_ascii=False)

    chain = []
    for hop_idx, ev in enumerate(evidence_chain):
        entity_key = ev.get("entity", ev.get("screenshot", ""))
        screenshot_key = ev.get("screenshot", "")
        img_id = (
            ev.get("image_id")
            or image_id_map.get(entity_key)
            or image_id_map.get(screenshot_key)
            or f"img_{hop_idx}"
        )
        chain.append({
            "hop": hop_idx + 1,
            "image_id": img_id,
            "bboxes": ev["bboxes"],
            "sub_query": ev.get("sub_query", ""),
        })

    target = {
        "evidence_chain": chain,
        "answer": answer,
    }
    return json.dumps(target, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------

def _humanize_relation(rel: str) -> str:
    """Convert 2Wiki relation slugs to natural phrases.
    e.g. 'date_of_birth' -> 'date of birth'."""
    if not rel:
        return rel
    return rel.replace("_", " ").strip()


PATCH_SIZE = 32  # Qwen3-VL visual block (patch_size 16 × merge_size 2)


def _round_to_patch(n: int) -> int:
    return max(PATCH_SIZE, (n // PATCH_SIZE) * PATCH_SIZE)


def resize_image_and_bboxes(
    image: Image.Image,
    bboxes: List[List[int]],
    target_long_side: int,
    max_pixels: Optional[int] = None,
) -> Tuple[Image.Image, List[List[int]], float]:
    """
    Canonical resize for CoE visual inputs.

    The configured resolution controls the document scale while preserving the
    original aspect ratio. If the result exceeds ``max_pixels``, the image is
    isotropically shrunk until it fits the visual-token budget.

    Rounds to PATCH_SIZE multiples (required by Qwen-VL).
    Bounding boxes are scaled by the SAME (possibly different per-axis after
    rounding) factors.
    Returns (new_image, new_bboxes, avg_scale).
    """
    w, h = image.size
    scale = target_long_side / w
    new_w, new_h = int(w * scale), int(h * scale)

    if max_pixels and new_w * new_h > max_pixels:
        shrink = (max_pixels / (new_w * new_h)) ** 0.5
        scale *= shrink
        new_w, new_h = int(w * scale), int(h * scale)

    new_w = _round_to_patch(new_w)
    new_h = _round_to_patch(new_h)
    # Recompute effective per-axis scale after patch rounding
    sx = new_w / w
    sy = new_h / h

    image = image.resize((new_w, new_h), Image.LANCZOS)
    new_bboxes = []
    for b in bboxes:
        x1, y1, x2, y2 = b
        new_bboxes.append([
            max(0, min(new_w, int(x1 * sx))),
            max(0, min(new_h, int(y1 * sy))),
            max(0, min(new_w, int(x2 * sx))),
            max(0, min(new_h, int(y2 * sy))),
        ])
    # Return an isotropic scale for callers that need a scalar (avg of x,y)
    return image, new_bboxes, (sx + sy) / 2


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CoEDataset(Dataset):
    """
    Unified dataset for CoE training/evaluation.
    
    Supports both Wiki-CoE and SlideVQA data in CoE format.
    Handles single-hop and multi-hop modes.
    """

    def __init__(
        self,
        data_file: str,
        image_dir: str,
        mode: str = "multi_hop",       # "single_hop" or "multi_hop"
        augmentation: Optional[dict] = None,
        candidate_config: Optional[dict] = None,
        max_images: int = 10,           # max images per sample
        split: str = "train",
        raw_data_file: Optional[str] = None,  # for single-hop target sentences
        image_max_pixels: int = 1048576,
        train_resolutions: Optional[List[int]] = None,
        eval_resolution: int = 1024,
        single_hop_include_reasoning: bool = True,
    ):
        self.image_dir = image_dir
        self.mode = mode
        self.max_images = max_images
        self.split = split
        self.image_max_pixels = image_max_pixels
        self.train_resolutions = train_resolutions or [768, 1024, 1280]
        self.eval_resolution = eval_resolution
        self.single_hop_include_reasoning = single_hop_include_reasoning

        # Load data
        with open(data_file, "r") as f:
            self.data = json.load(f)

        # Load context sentences for single-hop expansion
        self._context = {}
        if raw_data_file and os.path.exists(raw_data_file):
            with open(raw_data_file, "r") as f:
                raw_data = json.load(f)
            for r in raw_data:
                for ctx in r.get("context", []):
                    if isinstance(ctx, list) and len(ctx) == 2:
                        if ctx[0] not in self._context:
                            self._context[ctx[0]] = ctx[1]

        # Augmentations (only enabled flags; resizing is unified below)
        aug_cfg = augmentation or {}
        self.spatial_augmentation = aug_cfg.get("spatial", False)
        self.resolution_variation = aug_cfg.get("resolution_variation", False)
        self.evidence_permutation = aug_cfg.get("evidence_permutation", False)

        cand_cfg = candidate_config or {}
        self.candidate_top_k = int(cand_cfg.get("top_k", 0) or 0)
        self.candidate_strategy = cand_cfg.get("distractor_strategy", "global")
        self.candidate_shuffle = bool(cand_cfg.get("shuffle", True))
        self.candidate_seed = int(cand_cfg.get("seed", 42))
        self.candidate_preserve_evidence_order = bool(cand_cfg.get("preserve_evidence_order", True))
        self._all_screenshots: List[str] = []
        self._screenshots_by_deck: Dict[str, List[str]] = {}

        # For single-hop mode, expand multi-hop samples into individual hops.
        # If the input file was pre-expanded offline (every sample already
        # has num_hops==1 and one evidence_chain entry), skip the expensive
        # per-sample blob I/O and use self.data directly. This avoids 64
        # ranks concurrently reading ~50k _bboxes.json files from blob at
        # training start, which caused blobfuse to hang for >10min.
        if mode == "single_hop":
            pre_expanded = (
                len(self.data) > 0
                and all(s.get("num_hops") == 1 for s in self.data[:32])
                and all(len(s.get("evidence_chain", [])) == 1 for s in self.data[:32])
            )
            if pre_expanded:
                self.samples = self.data
                logger.info(f"Detected pre-expanded single-hop data; skipped runtime expansion.")
            else:
                self.samples = self._expand_to_single_hop()
        else:
            self.samples = self.data

        logger.info(
            f"Loaded {len(self.samples)} samples ({mode} mode) from {data_file}"
        )
        self._build_candidate_pools()
        if self.candidate_top_k > 0:
            logger.info(
                f"Candidate topK enabled: top_k={self.candidate_top_k}, "
                f"strategy={self.candidate_strategy}, shuffle={self.candidate_shuffle}"
            )

    @staticmethod
    def _deck_key_from_screenshot(screenshot: str) -> str:
        stem = os.path.basename(screenshot)
        for marker in ("__page_", "_page_"):
            if marker in stem:
                return stem.split(marker)[0]
        return ""

    @staticmethod
    def _page_num_from_screenshot(screenshot: str) -> Optional[int]:
        stem = os.path.basename(screenshot)
        for marker in ("__page_", "_page_"):
            if marker in stem:
                rest = stem.split(marker, 1)[1]
                num = rest.split(".", 1)[0]
                try:
                    return int(num)
                except Exception:
                    return None
        return None

    def _build_candidate_pools(self) -> None:
        seen = set()
        def _add(ss: str) -> None:
            if not ss or ss in seen:
                return
            seen.add(ss)
            self._all_screenshots.append(ss)
            deck = self._deck_key_from_screenshot(ss)
            if deck:
                self._screenshots_by_deck.setdefault(deck, []).append(ss)

        for sample in self.samples:
            for ev in sample.get("evidence_chain", []):
                _add(ev.get("screenshot"))

        if self.candidate_top_k > 0 and self.candidate_strategy == "same_deck" and os.path.isdir(self.image_dir):
            for root, _, files in os.walk(self.image_dir):
                for name in files:
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        _add(os.path.relpath(os.path.join(root, name), self.image_dir))

    @staticmethod
    def _scale_bboxes_to_size(
        bboxes: List[List[int]],
        orig_size: Tuple[int, int],
        new_size: Tuple[int, int],
    ) -> List[List[int]]:
        w, h = orig_size
        new_w, new_h = new_size
        sx, sy = new_w / w, new_h / h
        out = []
        for b in bboxes:
            x1, y1, x2, y2 = b
            out.append([
                max(0, min(new_w, int(x1 * sx))),
                max(0, min(new_h, int(y1 * sy))),
                max(0, min(new_w, int(x2 * sx))),
                max(0, min(new_h, int(y2 * sy))),
            ])
        return out

    def _candidate_rng(self, idx: int):
        if self.split == "train":
            return random
        return random.Random(self.candidate_seed + idx)

    def _sample_distractors(self, sample: dict, gold_screenshots: List[str], idx: int) -> List[str]:
        need = max(0, self.candidate_top_k - len(gold_screenshots))
        if need <= 0:
            return []
        gold_set = set(gold_screenshots)
        rng = self._candidate_rng(idx)

        explicit = sample.get("candidate_screenshots") or sample.get("distractor_screenshots") or []
        explicit_pool = []
        for item in explicit:
            ss = item.get("screenshot") if isinstance(item, dict) else item
            if ss and ss not in gold_set:
                explicit_pool.append(ss)
        if explicit_pool:
            pool = list(dict.fromkeys(explicit_pool))
        elif self.candidate_strategy == "same_deck" and gold_screenshots:
            deck = self._deck_key_from_screenshot(gold_screenshots[0])
            pool = [ss for ss in self._screenshots_by_deck.get(deck, []) if ss not in gold_set]
            gold_pages = [
                p for p in (self._page_num_from_screenshot(ss) for ss in gold_screenshots)
                if p is not None
            ]
            if gold_pages:
                pool.sort(key=lambda ss: (
                    min(abs((self._page_num_from_screenshot(ss) or 10**9) - gp) for gp in gold_pages),
                    self._page_num_from_screenshot(ss) or 10**9,
                    ss,
                ))
            else:
                pool.sort()
            return pool[:need]
        else:
            pool = [ss for ss in self._all_screenshots if ss not in gold_set]

        if len(pool) <= need:
            return pool[:need]
        return rng.sample(pool, need)

    def _expand_to_single_hop(self) -> List[dict]:
        """
        Expand multi-hop samples into single-hop training instances.

        For each evidence triple (entity, relation, value), we:
          1. Load the entity page's paragraph bboxes from `<screenshot>_bboxes.json`.
          2. Filter paragraphs whose text contains `value` (case-insensitive).
          3. Among the candidates, pick the paragraph most similar (Jaccard token
             overlap) to the 2Wiki supporting-fact sentence text; tie-break by
             topmost position (earlier in page = more likely lead fact).
          4. If no paragraph contains the value, DROP the sample (upstream 2Wiki
             noise; can't produce a correct supervision signal).
          5. Emit a single-hop sample with exactly ONE bbox — avoiding the
             "multiple evidences share the same bboxes" bug in earlier versions.
        """
        import re
        from collections import Counter

        _STOPWORDS = frozenset("""
            a an the and or but if then else when of in on at to from by for with as
            is are was were be been being have has had do does did will would shall
            should may might must can could this that these those he she it they we
            you i me him her his hers its their theirs our ours your yours my mine
            who whom whose which what where why how there here not no yes so than
            because about into through during before after above below up down out
            off over under again further once all any both each few more most other
            some such only own same too very s t just now also which
        """.split())

        def _content_tokens(text: str):
            toks = re.findall(r"[a-z0-9]+", text.lower())
            return [t for t in toks if t not in _STOPWORDS and len(t) >= 2]

        def _coverage_score(target: str, para: str) -> float:
            """Fraction of target's content tokens that appear in the paragraph.

            Rationale: union-based Jaccard penalizes long paragraphs (real
            intros) while min-denominator inflates short footnotes whose
            tokens are a subset of target. Target-containment favors
            paragraphs that fully cover the target regardless of extra
            content, which matches how 2Wiki supporting sentences relate to
            the Wikipedia paragraph they were extracted from.
            """
            ta = set(_content_tokens(target))
            tb = set(_content_tokens(para))
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta)

        def _value_in_text(value: str, text: str) -> bool:
            """Continuous (punctuation-insensitive) substring match.

            Avoids false positives where value tokens happen to appear
            scattered across an unrelated listing (e.g. an ancestry table
            that mentions "Frederick Francis" and "Frederik IX" separately
            would otherwise satisfy all-tokens-in for value
            "Frederick IX of Denmark").
            """
            def _norm(s: str) -> str:
                s = re.sub(r"[^a-z0-9]+", " ", s.lower())
                return re.sub(r"\s+", " ", s).strip()

            v = _norm(value)
            t = _norm(text)
            if not v:
                return False
            return f" {v} " in f" {t} "

        def _pick_paragraph(screenshot: str, value: str, target_sent: str):
            """Pick the paragraph most similar to the 2Wiki supporting sentence.

            Candidates: paragraphs with >= 8 tokens and reasonable bbox size,
            fully within screenshot bounds.

            Rank by jaccard(target_sent, para_text). Whether the paragraph
            contains the evidence `value` is only a tie-breaker — when two
            paragraphs have the same similarity, prefer the one that literally
            contains the value; second tie-break is earlier y1 (top of page).

            Accept only if best sim >= 0.4, else drop (upstream 2Wiki/crawl
            noise: target content not present on the page).
            """
            bbp = os.path.join(
                self.image_dir,
                screenshot.replace(".png", "_bboxes.json"),
            )
            if not os.path.exists(bbp):
                return None
            try:
                with open(bbp, "r") as f:
                    paras = json.load(f)
            except Exception:
                return None
            if not target_sent:
                return None

            # Filter out paragraphs whose bbox lies outside the screenshot
            # bounds. The layout JSON covers the full HTML document, but
            # screenshots may be truncated in height — paragraphs beyond
            # the image are unusable as training targets.
            img_h = None
            try:
                img_path = os.path.join(self.image_dir, screenshot)
                if os.path.exists(img_path):
                    with Image.open(img_path) as _im:
                        img_h = _im.size[1]
            except Exception:
                img_h = None

            candidates = []
            for p in paras:
                x1, y1, x2, y2 = p["bbox"]
                if (x2 - x1) < 80 or (y2 - y1) < 5:
                    continue
                if img_h is not None and y2 > img_h:
                    continue
                if len(p["text"].split()) < 8:
                    continue
                sim = _coverage_score(target_sent, p["text"])
                val_in = _value_in_text(value, p["text"])
                # val_in is a strong signal because evidence values are
                # almost always proper nouns (people, places, dates) and
                # 2Wiki sids occasionally point to the wrong sentence.
                # Add a bonus so val_in candidates beat similarly-scored
                # non-val_in ones, but coverage still dominates large gaps.
                score = sim + (0.2 if val_in else 0.0)
                candidates.append((score, sim, val_in, y1, p))

            if not candidates:
                return None
            candidates.sort(key=lambda c: (-c[0], -c[1], c[3]))
            # Threshold on the boosted score: either coverage alone is
            # strong enough, or coverage + val_in bonus clears the bar.
            if candidates[0][0] < 0.4:
                return None
            return candidates[0][4]

        single_hop_samples = []
        n_skipped_no_value = 0
        n_skipped_no_chain_entry = 0

        for sample in self.data:
            evidences = sample.get("evidences", [])
            evidence_chain = sample["evidence_chain"]

            if evidences:
                entity_to_ev = {ev["entity"]: ev for ev in evidence_chain}

                for evi in evidences:
                    if len(evi) != 3:
                        continue
                    evi_entity, relation, value = evi
                    value = str(value)

                    # Match to evidence_chain entry (gives us screenshot filename)
                    ev_entry = entity_to_ev.get(evi_entity)
                    if ev_entry is None:
                        for chain_entity, ev in entity_to_ev.items():
                            if chain_entity.startswith(evi_entity) or evi_entity.startswith(chain_entity):
                                ev_entry = ev
                                break
                    if ev_entry is None:
                        n_skipped_no_chain_entry += 1
                        continue

                    # Supporting-fact sentence text (for Jaccard tie-break)
                    target_sent = ""
                    sids = ev_entry.get("sentence_ids", [])
                    ctx_sents = self._context.get(ev_entry.get("entity", ""), [])
                    if sids and ctx_sents and sids[0] < len(ctx_sents):
                        target_sent = ctx_sents[sids[0]]

                    # Pick paragraph on this page using hybrid value+jaccard.
                    screenshot = ev_entry["screenshot"]
                    chosen = _pick_paragraph(screenshot, value, target_sent)
                    if chosen is None:
                        n_skipped_no_value += 1
                        continue

                    new_ev = {
                        "entity": ev_entry.get("entity", evi_entity),
                        "screenshot": screenshot,
                        "bboxes": [chosen["bbox"]],
                        "sentence_ids": sids,
                    }
                    single_hop_samples.append({
                        "question": f"What is the {_humanize_relation(relation)} of {evi_entity}?",
                        "answer": value,
                        "evidence_value": value,
                        "_target_sentence": target_sent,
                        "type": sample.get("type", ""),
                        "num_hops": 1,
                        "evidence_chain": [new_ev],
                    })
            else:
                for ev in evidence_chain:
                    single_hop_samples.append({
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "type": sample.get("type", ""),
                        "num_hops": 1,
                        "evidence_chain": [ev],
                    })

        if n_skipped_no_value or n_skipped_no_chain_entry:
            logger.info(
                f"_expand_to_single_hop: kept {len(single_hop_samples)} samples "
                f"(dropped {n_skipped_no_value} with no paragraph containing value, "
                f"{n_skipped_no_chain_entry} with no matching chain entry)"
            )
        return single_hop_samples

    def _bbox_contains_value(self, screenshot: str, bbox: List[int], value: str) -> bool:
        """Check if the paragraph at this bbox contains the evidence value."""
        bboxes_path = os.path.join(
            self.image_dir,
            screenshot.replace(".png", "_bboxes.json"),
        )
        if not os.path.exists(bboxes_path):
            return True  # can't verify, assume ok

        try:
            with open(bboxes_path, "r") as f:
                paras = json.load(f)
        except Exception:
            return True

        for para in paras:
            if para["bbox"] == bbox:
                return value.lower() in para["text"].lower()
        return True  # bbox not found in paras, assume ok

    def _find_bbox_by_value(self, screenshot: str, value: str,
                            target_sentence: str = "") -> Optional[List[int]]:
        """
        Find the best bbox for a single-hop sample.
        
        Strategy: among all paragraphs that contain the answer value,
        pick the one most similar to the target sentence. This handles
        cases where Wikipedia split one sentence into multiple paragraphs,
        or where common words like "American" appear in many places.
        
        If target_sentence is empty, falls back to topmost wide match.
        """
        bboxes_path = os.path.join(
            self.image_dir,
            screenshot.replace(".png", "_bboxes.json"),
        )
        if not os.path.exists(bboxes_path):
            return None

        try:
            with open(bboxes_path, "r") as f:
                paras = json.load(f)
        except Exception:
            return None

        import re
        from collections import Counter

        def _sim(a, b):
            ta = Counter(re.findall(r'[a-z0-9]+', a.lower()))
            tb = Counter(re.findall(r'[a-z0-9]+', b.lower()))
            common = sum((ta & tb).values())
            if common == 0:
                return 0.0
            p = common / max(sum(tb.values()), 1)
            r = common / max(sum(ta.values()), 1)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Filter: paragraphs that contain the answer (case insensitive)
        candidates = []
        val_lower = value.lower()
        for para in paras:
            x1, y1, x2, y2 = para["bbox"]
            w = x2 - x1
            if w < 100 or (y2 - y1) < 5:
                continue
            if val_lower in para["text"].lower():
                sim = _sim(target_sentence, para["text"]) if target_sentence else 0.0
                candidates.append((sim, w, y1, para["bbox"]))

        if not candidates:
            return None

        # Sort: highest similarity to target sentence, then wider, then earlier
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
        return candidates[0][3]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        evidence_chain = sample["evidence_chain"]
        evidences = sample.get("evidences", [])

        # Build entity -> sub_query mapping from evidences
        entity_sub_queries = {}
        for evi in evidences:
            if len(evi) == 3:
                entity, relation, value = evi
                entity_sub_queries[entity] = f"What is the {_humanize_relation(relation)} of {entity}?"

        # TopK candidate training/eval supervises evidence-chain order, so keep
        # the reasoning order unless explicitly opted out.
        can_permute_chain = not (self.candidate_top_k > 0 and self.candidate_preserve_evidence_order)

        # Optionally permute evidence order during training
        if self.evidence_permutation and can_permute_chain and self.split == "train" and len(evidence_chain) > 1:
            evidence_chain = list(evidence_chain)
            random.shuffle(evidence_chain)

        # Load images and apply a SINGLE canonical resize (images + bboxes together)
        images = []
        image_paths = []
        augmented_chain = []
        image_id_map = {}

        # Pick a resolution target for this sample (shared across all hops so that
        # coordinates in the target JSON are consistent with what the model sees).
        if self.split == "train" and self.resolution_variation:
            target_long_side = random.choice(self.train_resolutions)
        else:
            target_long_side = self.eval_resolution

        def _load_image(screenshot: str) -> Tuple[Image.Image, str]:
            img_path = os.path.join(self.image_dir, screenshot)
            if not os.path.exists(img_path):
                # Strict mode by default: a missing screenshot silently
                # replaced with a blank canvas will train/eval the model on
                # noise and is extremely hard to notice. Opt out explicitly
                # via env var COE_ALLOW_MISSING_IMAGES=1.
                if os.environ.get("COE_ALLOW_MISSING_IMAGES", "0") != "1":
                    raise FileNotFoundError(
                        f"Missing screenshot: {img_path}. "
                        f"Set COE_ALLOW_MISSING_IMAGES=1 to fall back to blank image."
                    )
                logger.warning(f"Missing image (allowed): {img_path}")
                img = Image.new("RGB", (1024, 768), (255, 255, 255))
            else:
                img = Image.open(img_path).convert("RGB")
            return img, img_path

        candidate_mode = self.mode != "single_hop" and self.candidate_top_k > 0

        if candidate_mode:
            gold_screenshots = list(dict.fromkeys(ev["screenshot"] for ev in evidence_chain))
            candidate_screenshots = gold_screenshots + self._sample_distractors(sample, gold_screenshots, idx)
            rng = self._candidate_rng(idx)
            if self.candidate_shuffle and len(candidate_screenshots) > 1:
                rng.shuffle(candidate_screenshots)

            candidate_sizes: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
            for cand_idx, screenshot in enumerate(candidate_screenshots):
                img, img_path = _load_image(screenshot)
                orig_size = img.size

                if self.split == "train" and self.spatial_augmentation:
                    jittered = int(target_long_side * random.uniform(0.9, 1.1))
                else:
                    jittered = target_long_side

                img, _, _ = resize_image_and_bboxes(
                    img, [],
                    target_long_side=jittered,
                    max_pixels=self.image_max_pixels,
                )

                images.append(img)
                image_paths.append(img_path)
                img_id = f"img_{cand_idx}"
                image_id_map[screenshot] = img_id
                candidate_sizes[screenshot] = (orig_size, img.size)

            for ev in evidence_chain:
                screenshot = ev["screenshot"]
                bboxes = [list(map(int, b)) for b in ev.get("bboxes", [])]
                if screenshot in candidate_sizes:
                    orig_size, new_size = candidate_sizes[screenshot]
                    bboxes = self._scale_bboxes_to_size(bboxes, orig_size, new_size)

                img_id = image_id_map.get(screenshot, "")
                if ev.get("entity"):
                    image_id_map[ev["entity"]] = img_id

                augmented_chain.append({
                    **ev,
                    "image_id": img_id,
                    "bboxes": bboxes,
                    "sub_query": (ev.get("sub_query") or "").strip() or
                                  entity_sub_queries.get(ev.get("entity", ""), ""),
                })
        else:
            for hop_idx, ev in enumerate(evidence_chain):
                screenshot = ev["screenshot"]
                img, img_path = _load_image(screenshot)

                bboxes = [list(map(int, b)) for b in ev.get("bboxes", [])]

                if self.split == "train" and self.spatial_augmentation:
                    jittered = int(target_long_side * random.uniform(0.9, 1.1))
                else:
                    jittered = target_long_side

                img, bboxes, _ = resize_image_and_bboxes(
                    img, bboxes,
                    target_long_side=jittered,
                    max_pixels=self.image_max_pixels,
                )

                images.append(img)
                image_paths.append(img_path)

                entity_key = ev.get("entity", screenshot)
                img_id = f"img_{hop_idx}"
                image_id_map[entity_key] = img_id
                image_id_map[screenshot] = img_id

                augmented_chain.append({
                    **ev,
                    "image_id": img_id,
                    "bboxes": bboxes,
                    # Prefer sub_query already present in the sample (populated by
                    # scripts/rebuild_multihop_bboxes.py from _resolved_hops). Fall
                    # back to entity-lookup for legacy data. Strip so stored ""
                    # still triggers fallback.
                    "sub_query": (ev.get("sub_query") or "").strip() or
                                  entity_sub_queries.get(ev.get("entity", ""), ""),
                })

        # Build prompt
        if self.mode == "single_hop":
            system_prompt = (
                SINGLE_HOP_SYSTEM_PROMPT if self.single_hop_include_reasoning
                else SINGLE_HOP_SYSTEM_PROMPT_NO_REASONING
            )
            user_content = format_user_message_single_hop(question, image_paths[0])
        else:
            system_prompt = SYSTEM_PROMPT
            user_content = format_user_message_multi_hop(question, image_paths)

        # Build target
        target_str = format_target(
            answer, augmented_chain, image_id_map,
            single_hop=(self.mode == "single_hop"),
            reasoning=sample.get("reasoning", "") if self.mode == "single_hop" else "",
            include_reasoning=self.single_hop_include_reasoning,
        )

        # Build conversation in Qwen3-VL format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_str},
        ]

        return {
            "messages": messages,
            "images": images,
            "image_paths": image_paths,
            "candidate_screenshots": [
                os.path.relpath(p, self.image_dir) if p.startswith(self.image_dir) else p
                for p in image_paths
            ],
            "gold_image_ids": [ev.get("image_id", "") for ev in augmented_chain],
            "mode": self.mode,
            "question": question,
            "answer": answer,
            "evidence_chain": augmented_chain,
            "sample_id": idx,
        }


class CoEEvalDataset(Dataset):
    """
    Evaluation dataset that provides raw samples without augmentation.
    Returns images and questions for inference pipeline.
    Applies the SAME canonical resize as training to keep coordinate frames aligned.
    """

    def __init__(
        self,
        data_file: str,
        image_dir: str,
        image_max_pixels: int = 1048576,
        eval_resolution: int = 1024,
    ):
        self.image_dir = image_dir
        self.image_max_pixels = image_max_pixels
        self.eval_resolution = eval_resolution
        with open(data_file, "r") as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} eval samples from {data_file}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        evidence_chain = sample["evidence_chain"]

        gt_images = []
        gt_image_paths = []
        gt_chain = []
        for ev in evidence_chain:
            path = os.path.join(self.image_dir, ev["screenshot"])
            gt_image_paths.append(path)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                img = Image.new("RGB", (1024, 768), (255, 255, 255))

            bboxes = [list(map(int, b)) for b in ev.get("bboxes", [])]
            img, bboxes, _ = resize_image_and_bboxes(
                img, bboxes,
                target_long_side=self.eval_resolution,
                max_pixels=self.image_max_pixels,
            )
            gt_images.append(img)
            gt_chain.append({**ev, "bboxes": bboxes})

        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample.get("type", ""),
            "num_hops": sample.get("num_hops", len(evidence_chain)),
            "evidence_chain": gt_chain,
            "gt_images": gt_images,
            "gt_image_paths": gt_image_paths,
            "sample_id": idx,
        }
