# Nutrition Table Detection — Project Report

Fine-tuned **Qwen2-VL-7B-Instruct** to detect nutrition tables in product images. Given an image, the model outputs all bounding boxes as text tokens. Trained on [`openfoodfacts/nutrition-table-detection`](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection), evaluated on 123 validation samples.

**Model on HuggingFace**: [`MayaKD/qwen2-vl-7b-nutrition-vllm`](https://huggingface.co/MayaKD/qwen2-vl-7b-nutrition-vllm) (merged weights, ready to serve) · [`MayaKD/qwen2-vl-7b-nutrition`](https://huggingface.co/MayaKD/qwen2-vl-7b-nutrition) (training artifacts)

The project has two phases, in order: **(1) fine-tuning** the model, then **(2) inference** with the fine-tuned model. The sections below follow that order.

---

# 1. Fine-Tuning

### Multi-Stage LoRA Fine-Tuning

The pipeline applies LoRA progressively across up to 3 stages. After each stage, adapters are merged into the base model — so each stage starts from clean full weights, not stacked adapters.

| Stage | LoRA Targets | Purpose |
|-------|-------------|---------|
| 1 — Vision warmup *(optional)* | Last 8 vision blocks | Orient vision encoder to the task |
| 2 — Full vision | All vision blocks + merger MLP | Full visual feature learning |
| 3 — Joint | Vision (full) + LLM self-attn + MLP | End-to-end vision–language alignment |

**exp13 (final model) uses 2 stages** — warmup was dropped after separate comparisons showed it cost training time without a matching IoU gain once stage 2 already targets the full vision encoder. Note that the run table below is not a controlled ablation: exp13 differs from the `repro_*` runs in batch size and schedule as well as stage count.

Base model `Qwen/Qwen2-VL-7B-Instruct`, trained with HuggingFace `SFTTrainer` + `accelerate`, BF16 + Flash Attention 2 + fused AdamW. Best checkpoint selected on `eval_loss` with early stopping (patience 3).

### Data Format

Dataset boxes are stored as `(y_min, x_min, y_max, x_max)` normalized `[0, 1]`. Training converts them to `(x_min, y_min, x_max, y_max)` scaled to `[0, 1000]` and encodes them using Qwen2-VL's native box tokens:

```
<|object_ref_start|>nutrition-table<|object_ref_end|><|box_start|>(x0, y0),(x1, y1)<|box_end|>
```

Multiple boxes are concatenated; samples with no table produce `"No table found."`.

### Key Engineering Problem: Multi-GPU Merge Corruption

During exp10, LoRA merging on multi-GPU silently corrupted weights — merged-model IoU dropped from **0.794 → 0.300** while the in-memory model scored correctly. All processes were calling `merge_and_unload()` + `save_pretrained()` simultaneously, writing conflicting shards.

**Fix**: main-process-only save with `accelerator.is_main_process` guard + `wait_for_everyone()` barrier. After the fix, pre- and post-merge evaluations match within ~1%.

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

Training accuracy is reported as mean IoU on the 123-sample validation set.

| Run | Stages | GPUs | Time | Final Mean IoU |
|-----|:------:|:----:|:----:|:--------------:|
| repro_9 | 3 | 2× RTX Pro 6000 | 97 min | 0.870 |
| repro_11 | 3 | 1× RTX Pro 6000 | 150 min | 0.886 |
| **repro_12** | **3** | **2× RTX Pro 6000** | **89 min** (15+31+43) | **0.893** |
| **exp13 (final)** | **2** | **2× RTX Pro 6000** | — | **0.82** |

The shipped model is **exp13**. Its full accuracy on the validation set:

| Metric | Value | |
|--------|:-----:|--|
| **Mean IoU** | **0.82** | primary metric |
| Precision@0.5 | ≤ 0.91 | upper bound |
| Recall@0.5 | ≤ 0.89 | upper bound |
| F1@0.5 | ≤ 0.90 | upper bound |

> **Metric correction.** The original precision/recall implementation counted every IoU-matrix
> cell above 0.5 instead of matching one-to-one, so N overlapping predictions against a single
> ground-truth box scored N true positives and recall could exceed 1.0. `metrics.py` now does
> greedy one-to-one matching, with a regression test on the exact failure case. The removed
> bias was strictly upward, so true values sit at or below those shown. Only aggregate
> statistics were persisted, so quantifying the gap would mean re-running inference on a GPU.
> Mean IoU never used the matching path, is unaffected, and is the metric to judge the model on.

The three-stage schedule reached **0.893**, so there is roughly 7 IoU points of headroom above the shipped model. exp13 is the two-stage production run: it is the artifact that is published, reproducible, and benchmarked throughout this report. The `repro_*` runs were exploratory and their weights were not preserved. Productionizing the three-stage schedule is the top open item.

---

# 2. Inference

Inference runs the **fine-tuned exp13 model**. Two backends are supported — HuggingFace `transformers` and vLLM. Accuracy is identical across backends (same weights, Mean IoU 0.82); they differ only in speed and memory.

### Backends

- **HuggingFace `transformers`** — best for offline evaluation and dataset processing. Use `batch_size=4`, or 16 if VRAM allows. 17–30 GB VRAM.
- **vLLM** — best for production serving and multi-user throughput. Use `concurrency=4` when P95 matters, `8` to maximize throughput; avoid 16. Pre-allocates ~72 GB for continuous batching (intentional, not a leak).

vLLM must be served from a **local path** (it cannot load from an HF subfolder).

### Inference Results (exp13, 123 val samples, single GPU)

**vLLM under concurrent load** (`vllm_throughput.py`, real `asyncio.Semaphore` concurrency):

| Concurrency | Mean Latency | P95 | Throughput |
|:-----------:|:------------:|:---:|:----------:|
| 1 | 1,378 ms | 1,506 ms | 0.73 req/s |
| 4 | 399 ms | 467 ms | 9.89 req/s |
| **8** | **435 ms** | 1,056 ms | **17.82 req/s** |
| 16 | 16,528 ms | 22,653 ms | 0.94 req/s |

Throughput scales to **17.82 req/s at c=8** — 24× the sequential rate — then collapses at c=16 (0.94 req/s, 22.7 s P95) as the scheduler saturates. c=4 is the best latency/throughput compromise; c=8 maximizes throughput.

**HuggingFace `transformers`** (real batching: one padded `generate()` per batch):

| Batch | Mean Latency/sample | P95 | VRAM |
|:-----:|:-------------------:|:---:|:----:|
| 1 | 890 ms | 989 ms | 17.2 GB |
| 4 | 475 ms | 483 ms | 20.4 GB |
| 16 | 321 ms | 533 ms | 29.9 GB |

vLLM pre-allocates ~72 GB for KV cache regardless of load, so HF remains the better option for offline work on constrained hardware.

> **Measurement caveat.** `inference_eval.py`'s vLLM path issues requests sequentially, so its `--batch_size` flag does not batch — it only rescales the reported per-sample latency. Accuracy from that path is valid; its vLLM *latency* figures are not, and are excluded here. All vLLM performance numbers above come from `vllm_throughput.py`. Raw JSONs are committed under `nutrition_table_inference/outputs/`.

### Quantization

4-bit GPTQ via `GPTQModel` produces a working 6.9 GB model (from 16.6 GB) that runs correctly under HF inference.

vLLM serving is blocked: Qwen2-VL is a vision-language model rather than a standard `AutoModelForCausalLM`, and GPTQModel's tensor naming for VLM layers doesn't match vLLM's GPTQ loader (`KeyError: layers.10.mlp.down_proj.g_idx`). `AutoGPTQ` was tried as an alternative and failed earlier, at CUDA extension compilation. Unresolved — it needs an upstream fix or a different quantization toolchain.

---

## Code Structure

```
nutrition_table_fine_tuning/src/        # Phase 1 — fine-tuning
  train.py              Multi-stage training loop (LoRA, merge, eval per stage)
  model_utils.py        Model loading, adapter merge, regex-based LoRA target selection
  eval_utils.py         IoU / Precision / Recall / F1; box parsing from generated text
  dataset/data_utils.py format_data (coord conversion, box tag assembly), parse_boxes_from_text
  dataset/collators.py  collate_fn: tokenization, label masking, numeric_only ablation mode
  configs/exp*.yaml     Per-experiment configs (13 experiments, exp1 → exp13)

nutrition_table_inference/src/          # Phase 2 — inference
  inference_eval.py     Per-sample HF and vLLM latency + accuracy benchmark
  vllm_throughput.py    vLLM serving throughput under concurrent load
  eval_hf_dataset2.py   Batched HF evaluation over full dataset
  eval_vllm_dataset.py  vLLM full-dataset evaluation
  quantize_qwen2vl_gptq.py  4-bit GPTQ quantization (HF working, vLLM blocked)
```
