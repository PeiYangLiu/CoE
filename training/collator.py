"""
Data Collator for CoE VLM Training

Handles multi-image conversation formatting for Qwen3-VL fine-tuning.
Resizing is done in the dataset (so targets and images share a coordinate
frame); this collator ONLY tokenizes and builds loss masks.
"""

import logging
from typing import Dict, List, Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class CoEDataCollator:
    """
    Collator for CoE training with Qwen3-VL.

    Labels are masked deterministically using the prompt length obtained
    from `apply_chat_template(..., add_generation_prompt=True)` on everything
    up to (but not including) the assistant turn. No fragile subsequence
    searching, no 70% fallback.
    """

    def __init__(
        self,
        processor,
        max_length: int = 4096,
        **_ignored,
    ):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        batch_texts = []
        batch_prompt_texts = []
        batch_images = []

        for feat in features:
            messages = feat["messages"]
            images = feat.get("images", [])

            # Full sequence (with assistant target)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            # Prompt-only sequence (system + user + assistant-header, no answer)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )

            batch_texts.append(full_text)
            batch_prompt_texts.append(prompt_text)
            batch_images.append(images if images else None)

        all_images = []
        for imgs in batch_images:
            if imgs:
                all_images.extend(imgs)

        proc_kwargs = dict(text=batch_texts, padding=True, return_tensors="pt")
        if all_images:
            proc_kwargs["images"] = all_images
        inputs = self.processor(**proc_kwargs)

        # Tokenize prompts separately (no images) to get boundary lengths.
        # We intentionally use the SAME processor so special tokens line up.
        # Image placeholders in the prompt text expand to the same image-token
        # count as in the full text (since images are identical up to prompt),
        # keeping prompt_len correct.
        prompt_kwargs = dict(text=batch_prompt_texts, padding=True, return_tensors="pt")
        if all_images:
            prompt_kwargs["images"] = all_images
        prompt_inputs = self.processor(**prompt_kwargs)

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        input_ids = inputs["input_ids"]
        prompt_ids = prompt_inputs["input_ids"]

        for i in range(input_ids.size(0)):
            # Count non-pad tokens in the prompt = boundary.
            p_nonpad = (prompt_ids[i] != pad_id).sum().item()
            labels[i, :p_nonpad] = -100
            # Also mask padding in the full sequence.
            labels[i][input_ids[i] == pad_id] = -100

        inputs["labels"] = labels
        return inputs
