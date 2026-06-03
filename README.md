# Nutrition Table Detection

Fine-tuned **Qwen2-VL-7B-Instruct** to detect nutrition tables in product images. Given an image, the model outputs all bounding box coordinates as text.

**Model**: [`MayaKD/qwen2-vl-7b-nutrition`](https://huggingface.co/MayaKD/qwen2-vl-7b-nutrition)  
**Dataset**: [`openfoodfacts/nutrition-table-detection`](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection)  
**Task**: image → normalized bounding boxes (0–1000 scale), encoded as text tokens.

The project runs in two phases, in order: first **fine-tuning** the model, then **inference** with the fine-tuned model. This README is organized the same way.

---

## Repository Layout

```
nutrition_table/
├── nutrition_table_fine_tuning/    # Phase 1 — training code
│   ├── src/
│   │   ├── train.py                # Multi-stage training loop (LoRA, merge, eval)
│   │   ├── model_utils.py          # Model loading, adapter merge, LoRA target selection
│   │   ├── eval_utils.py           # IoU / Precision / Recall / F1 evaluation
│   │   ├── viz_utils.py            # Bounding box visualization
│   │   └── dataset/                # format_data (coord/box tags), collators
│   ├── configs/exp*.yaml           # Per-experiment configs
│   └── runs/                       # Training outputs (adapters, merged models)
│
├── nutrition_table_inference/      # Phase 2 — inference & benchmarking
│   ├── src/
│   │   ├── inference_eval.py       # Per-sample HF and vLLM latency + accuracy benchmark
│   │   ├── vllm_throughput.py      # vLLM serving throughput measurement
│   │   ├── eval_hf_dataset2.py     # Batched HF evaluation over full dataset
│   │   ├── eval_vllm_dataset.py    # vLLM evaluation over full dataset
│   │   └── quantize_qwen2vl_gptq.py# 4-bit GPTQ quantization (HF works, vLLM blocked)
│   └── model/Qwen2-VL-7B/
│       └── final_model_exp13/      # Active inference model (local serving copy)
│
├── README.md                       # This file — landing page
├── project_report.md               # Concise technical writeup
└── CLAUDE.md                       # Full operational reference (commands, gotchas)
```

---

## Phase 1 — Fine-Tuning

### Approach

Multi-stage **LoRA** fine-tuning of `Qwen/Qwen2-VL-7B-Instruct`. Up to 3 stages; after each stage the adapters are merged into the base so the next stage starts from clean full weights (not stacked adapters).

| Stage | LoRA Targets | Purpose |
|-------|-------------|---------|
| 1 — Vision warmup *(optional)* | Last 8 vision blocks | Orient vision encoder to the task |
| 2 — Full vision | All vision blocks + merger MLP | Full visual feature learning |
| 3 — Joint | Vision (full) + LLM self-attn + MLP | End-to-end vision–language alignment |

The shipped model (**exp13**) uses **2 stages** — warmup was dropped after it showed no IoU gain. Training uses `SFTTrainer` + `accelerate`, BF16 + Flash Attention 2 + fused AdamW, best checkpoint by `eval_loss` with early stopping. See [`project_report.md`](project_report.md) for the multi-GPU merge-corruption bug and its fix.

### Running Training

```bash
cd nutrition_table_fine_tuning
source .venv/bin/activate

# Single GPU
python src/train.py --config configs/exp13.yaml

# Multi-GPU (2 GPUs)
accelerate launch --multi_gpu --mixed_precision=bf16 src/train.py --config configs/exp13.yaml
```

### Fine-Tuning Results

Accuracy is mean IoU on the 123-sample validation set.

| Run | Stages | GPUs | Time | Mean IoU |
|-----|:------:|:----:|:----:|:--------:|
| repro_11 | 3 | 1× RTX Pro 6000 | 150 min | 0.886 |
| **repro_12** | **3** | **2× RTX Pro 6000** | **89 min** | **0.893** |
| **exp13 (shipped)** | **2** | **2× RTX Pro 6000** | — | **0.82** |

Final shipped model (**exp13**): Mean IoU **0.82** · Precision **0.91** · Recall **0.89** · F1 **0.90**.

---

## Phase 2 — Inference

Inference uses the fine-tuned **exp13** model. Two backends are supported with identical accuracy (same weights); they differ only in speed and memory.

### Quick Start (HuggingFace)

```bash
pip install "transformers>=4.49" qwen-vl-utils accelerate torch
```

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

REPO = "MayaKD/qwen2-vl-7b-nutrition"
SYSTEM = (
    "You are a Vision Language Model specialized in interpreting visual data from product images.\n"
    "Your task is to analyze the provided product images and detect the nutrition tables in a certain format.\n"
    "Focus on delivering accurate, succinct answers based on the visual information. "
    "Avoid additional explanation unless absolutely necessary."
)
PROMPT = "Detect the bounding boxes of all nutrition tables in the image."

# Merged fine-tuned weights live in a subfolder; the processor is the stock Qwen2-VL one
model = Qwen2VLForConditionalGeneration.from_pretrained(
    REPO, subfolder="exp13_mod121925_joint/merged",
    dtype=torch.bfloat16, device_map="auto",
)
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": [
        {"type": "image", "image": "https://static.openfoodfacts.org/images/products/27563564/2.jpg"},
        {"type": "text", "text": PROMPT},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)

out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
print(processor.batch_decode(trimmed, skip_special_tokens=False)[0])
```

The model emits each detection as
`<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x_min, y_min),(x_max, y_max)<|box_end|>`,
with coordinates on a **0–1000** scale.

### vLLM Serving (production)

vLLM must serve from a **local path** (it cannot load from an HF subfolder), so download/clone the `exp13_mod121925_joint/merged` weights locally first.

```bash
# Start the server (separate terminal; --host 0.0.0.0 for remote access)
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/final_model_exp13 \
  --dtype bfloat16 --host 0.0.0.0 --port 8000

# Then call the OpenAI-compatible endpoint
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/final_model_exp13",
       "messages": [{"role": "user", "content": [
         {"type": "text", "text": "Detect the bounding boxes of all nutrition tables in the image."},
         {"type": "image_url", "image_url": {"url": "https://static.openfoodfacts.org/images/products/27563564/2.jpg"}}
       ]}],
       "max_tokens": 300, "temperature": 0.0}'
```

### Inference Results (exp13, 123 val samples, single GPU)

| Backend | Config | Mean Latency (ms) | P95 (ms) | Throughput | VRAM |
|---------|--------|:-----------------:|:--------:|:----------:|:----:|
| HF | batch=4 | 481 | 483 | ~2 req/s | 20 GB |
| HF | batch=1 | 890 | 989 | 1.1 req/s | 17 GB |
| vLLM | concurrency=4 | 394 | 482 | **10 req/s** | 72 GB |
| vLLM | concurrency=1 | 1,378 | 1,506 | 0.7 req/s | 72 GB |

- **HF** — best for offline evaluation and dataset processing (batch_size=4).
- **vLLM** — best for production serving (~9× HF throughput at concurrency=4; 72 GB is intentional pre-allocation for continuous batching).

### Quantization

4-bit GPTQ via `GPTQModel` works for **HF inference**. vLLM serving of the quantized model is **blocked** — Qwen2-VL is a vision-language model and GPTQModel's tensor naming for VLMs doesn't match vLLM's GPTQ loader (`KeyError: layers.10.mlp.down_proj.g_idx`). Unresolved.

---

## Docs

- [`project_report.md`](project_report.md) — approach, results, and code overview
- [`CLAUDE.md`](CLAUDE.md) — all commands, configs, and operational notes

---

## Roadmap

- [ ] Add advanced loss functions (Dice, IoU)
- [ ] Implement balanced sampling for rare and small objects
- [ ] Resolve vLLM serving for the 4-bit GPTQ quantized model
- [ ] Provide dataset preprocessing scripts

---

## License

Distributed under the **MIT License**. Fine-tunes **Qwen2-VL** (developed by Alibaba Cloud), available under the [Apache-2.0 License](https://huggingface.co/Qwen).

## Author

Maintained by [MKDehdashti](https://github.com/MKDehdashti). Contributions and feedback are welcome.

Repository: [github.com/MKDehdashti/qwen2-vl-nutrition-table](https://github.com/MKDehdashti/qwen2-vl-nutrition-table)
