# Chain of Evidence (CoE)

**Pixel-Level Visual Attribution for Iterative Retrieval-Augmented Generation**

CoE is a visual attribution framework that performs multi-hop reasoning directly over document screenshots. Given a question and a top-5 candidate set, it selects the ordered evidence images, localizes supporting regions with bounding boxes, and generates the final answer.

## Architecture

```
CoE/
├── configs/                    # Training configurations
│   ├── train_phase1.yaml       # Wiki-CoE Phase I
│   ├── train_phase2.yaml       # Wiki-CoE Phase II
│   ├── train_slidevqa_phase1.yaml
│   └── train_slidevqa_phase2.yaml
├── data/
│   ├── build_wiki_coe.py       # Wiki-CoE dataset construction pipeline
│   ├── prepare_slidevqa.py     # SlideVQA data preparation
│   └── dataset.py              # PyTorch Dataset classes with augmentation
├── models/
│   └── coe_model.py            # CoE model wrapper for Qwen3-VL
├── training/
│   ├── train.py                # Two-phase curriculum training script
│   └── collator.py             # VLM data collator
├── inference/
│   └── visualize.py            # Bounding box visualization
├── evaluation/
│   └── metrics.py              # EM, Loc-Acc, Chain-Acc metrics
└── scripts/
    ├── run_train.sh            # Training launcher
    ├── run_eval.sh             # Evaluation launcher
    └── offline_full_eval.py    # Distributed top-5 evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

### Wiki-CoE (from 2WikiMultiHopQA)

```bash
# Download 2WikiMultiHopQA first
python data/build_wiki_coe.py \
    --source path/to/2wikimultihopqa/train.json \
    --output_dir data/wiki_coe \
    --screenshot_dir data/wiki_coe/screenshots
```

### SlideVQA

```bash
# Download SlideVQA dataset first
python data/prepare_slidevqa.py \
    --data_dir path/to/slidevqa/annotations \
    --image_dir path/to/slidevqa/images \
    --output_dir data/slidevqa
```

## Training

CoE uses a two-phase curriculum learning strategy:

### Phase I: Single-hop Evidence Localization
```bash
bash scripts/run_train.sh wiki_phase1 8
bash scripts/run_train.sh slidevqa_phase1 8
```
- Trains visual grounding on individual document screenshots

### Phase II: Multi-hop Reasoning
```bash
bash scripts/run_train.sh wiki_phase2 8
bash scripts/run_train.sh slidevqa_phase2 8
```
- Requires Phase I checkpoint
- Trains top-5 candidate evidence-chain generation

## Evaluation

```bash
# Evaluate on Wiki-CoE
bash scripts/run_eval.sh wiki_coe checkpoints/coe_phase2/best 8

# Evaluate on SlideVQA
bash scripts/run_eval.sh slidevqa checkpoints/slidevqa_phase2/best 8
```

### Metrics
- **EM**: Exact Match answer accuracy
- **Loc-Acc**: Joint image-and-box localization accuracy. A box match is accepted when IoU >= 0.3 or the predicted center falls inside the ground-truth region.
- **Chain-Acc**: Ordered evidence-image chain accuracy.

## Model Output Format

The model generates structured JSON with answer and visual evidence chain:

```json
{
  "answer": "Christopher Nolan",
  "evidence_chain": [
    {
      "hop": 1,
      "image_id": "img_0",
      "bboxes": [[120, 340, 580, 390]],
      "sub_query": "director of Inception"
    },
    {
      "hop": 2,
      "image_id": "img_1",
      "bboxes": [[200, 156, 640, 210]],
      "sub_query": "nationality of Christopher Nolan"
    }
  ]
}
```

Bounding box coordinates `[x1, y1, x2, y2]` are in pixel space of the selected input image.

## Key Design Decisions

- **Candidate image IDs**: Model outputs `img_0`, `img_1`, etc. instead of entity names to avoid hallucination
- **Top-5 candidate setting**: Candidate screenshots include evidence documents and distractors; Wiki-CoE uses global distractors and SlideVQA uses same-deck distractors
- **Pixel-space coordinates**: Bounding boxes are emitted in the coordinate frame of the input screenshot
- **Structured generation**: The assistant target contains the evidence chain first, followed by the answer
- **Curriculum learning**: Phase I (grounding) → Phase II (reasoning) prevents optimization interference

## Citation

```bibtex
@inproceedings{coe2025,
  title={Chain of Evidence: Pixel-Level Visual Attribution for Iterative Retrieval-Augmented Generation},
  author={...},
  booktitle={...},
  year={2025}
}
```
