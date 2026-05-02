"""
CoE Model Wrapper

Wraps Qwen3-VL for Chain of Evidence generation:
- Unified generation of answer + bounding boxes
- Output parsing (extract answer, evidence chain from JSON)
"""

import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_coe_model(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype: str = "bfloat16",
    use_flash_attention: bool = True,
    gradient_checkpointing: bool = True,
    quantization_bits: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Load Qwen3-VL model and processor.
    
    Args:
        model_name: HuggingFace model ID
        torch_dtype: "bfloat16" or "float16"
        use_flash_attention: enable flash attention 2
        gradient_checkpointing: enable gradient checkpointing
        quantization_bits: 4 or 8 for quantized inference
        
    Returns:
        (model, processor) tuple
    """
    dtype = getattr(torch, torch_dtype, torch.bfloat16)

    # Quantization config for inference
    bnb_config = None
    if quantization_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if quantization_bits else None,
        "trust_remote_code": True,
    }
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, **model_kwargs
    )

    # Apply Liger Kernel fused ops (RMSNorm, SwiGLU, RoPE, fused CE).
    # Supported for qwen3_vl / qwen3_vl_text. Opt-out with DISABLE_LIGER=1.
    if os.environ.get("DISABLE_LIGER") != "1":
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=model)
            logger.info("Liger kernel patches applied")
        except Exception as e:
            logger.warning(f"Liger kernel not applied ({e}); continuing without it")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Gradient checkpointing: only enable here if caller explicitly asks
    # (HF Trainer handles it via TrainingArguments with use_reentrant=False
    # during training; enabling twice causes ZeRO-3 hangs).
    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    return model, processor


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_coe_output(text: str) -> Optional[dict]:
    """
    Parse the model's generated text into structured CoE output.
    
    Attempts to extract JSON from the generated text, handling
    common generation artifacts.
    
    Returns:
        Dict with "answer" and "evidence_chain", or None if parsing fails.
    """
    # Try direct JSON parse
    text = text.strip()
    try:
        result = json.loads(text)
        if "answer" in result:
            return _validate_and_clean(result)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block from markdown code fence
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if "answer" in result:
                return _validate_and_clean(result)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if "answer" in result:
                return _validate_and_clean(result)
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract answer from text
    logger.warning(f"Failed to parse CoE output, attempting fallback extraction")
    return _fallback_parse(text)


def _validate_and_clean(result: dict) -> dict:
    """Validate and clean parsed CoE output."""
    answer = str(result.get("answer", ""))
    chain = result.get("evidence_chain", [])

    cleaned_chain = []
    for step in chain:
        hop = step.get("hop", len(cleaned_chain) + 1)
        image_id = step.get("image_id", f"img_{hop - 1}")
        bboxes = step.get("bboxes", [])
        sub_query = step.get("sub_query", "")

        # Validate bboxes
        valid_bboxes = []
        for bbox in bboxes:
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    coords = [int(float(c)) for c in bbox]
                    x1, y1, x2, y2 = coords
                    if x2 > x1 and y2 > y1:
                        valid_bboxes.append(coords)
                except (ValueError, TypeError):
                    continue

        cleaned_chain.append({
            "hop": hop,
            "image_id": image_id,
            "bboxes": valid_bboxes,
            "sub_query": sub_query,
        })

    return {
        "answer": answer,
        "evidence_chain": cleaned_chain,
    }


def _fallback_parse(text: str) -> Optional[dict]:
    """Last-resort extraction of answer from free text."""
    # Look for "answer" patterns
    patterns = [
        r'"answer"\s*:\s*"([^"]*)"',
        r"answer[:\s]+(.+?)(?:\n|$)",
        r"The answer is[:\s]+(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return {"answer": m.group(1).strip(), "evidence_chain": []}
    return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class CoEGenerator:
    """High-level inference wrapper for CoE model."""

    def __init__(
        self,
        model,
        processor,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        question: str,
        images: list,
        system_prompt: str = None,
    ) -> dict:
        """
        Generate CoE output for a question with document images.
        
        Args:
            question: the user query
            images: list of PIL images (document screenshots)
            system_prompt: optional override for system prompt
            
        Returns:
            Parsed CoE output dict with answer and evidence chain.
        """
        from data.dataset import MULTI_HOP_USER_INSTRUCTION, SYSTEM_PROMPT

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # Build image paths (use PIL objects directly)
        image_content = []
        for i, img in enumerate(images):
            image_content.append({"type": "image", "image": img})
            image_content.append({"type": "text", "text": f"[img_{i}]"})

        image_content.append({
            "type": "text",
            "text": (
                f"\nQuestion: {question}\n\n"
                f"{MULTI_HOP_USER_INSTRUCTION}"
            ),
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_content},
        ]

        # Process with Qwen3-VL processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        # Decode only the generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Parse output
        result = parse_coe_output(text_output)
        if result is None:
            result = {"answer": text_output.strip(), "evidence_chain": []}

        result["raw_output"] = text_output
        return result
