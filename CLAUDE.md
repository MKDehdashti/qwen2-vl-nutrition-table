# Nutrition Table Detection — CLAUDE.md

Fine-tuning and inference project for detecting nutrition table bounding boxes in food product images.
Model: **Qwen2-VL-7B-Instruct** (always 7B, no other size variants).
Task: given an image → predict normalized bounding box coordinates as text.
Dataset: [`openfoodfacts/nutrition-table-detection`](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) (HuggingFace public dataset).
Eval set: **123 validation samples**.
Metrics: Mean IoU, Precision@0.5, Recall@0.5, F1@0.5.
W&B project: `nutrition-table-vl`.
HuggingFace repo: `MayaKD/qwen2-vl-7b-nutrition`.
GitHub repo: `MKDehdashti/qwen2-vl-nutrition-table` (public; renamed from `qwen_runpod`).

---

## Repository Layout

```
nutrition_table/
├── nutrition_table_fine_tuning/    # Training code
│   ├── src/
│   │   ├── train.py               # Main training loop (multi-stage, merge, eval)
│   │   ├── model_utils.py         # Model loading, adapter merge, processor setup
│   │   ├── eval_utils.py          # IoU / Precision / Recall / F1 evaluation
│   │   ├── wandb_utils.py         # W&B initialization and loss callback
│   │   ├── viz_utils.py           # Bounding box visualization
│   │   └── dataset/
│   │       ├── data_utils.py      # format_data (coord conversion, box tags), parse_boxes_from_text
│   │       └── collators.py       # collate_fn: tokenization, label masking, numeric_only mode
│   ├── configs/exp*.yaml          # Per-experiment configs (stages, LoRA, lr, etc.)
│   ├── (runs/)                    # GONE from disk — training outputs live only on HF now
│   ├── .venv/                     # Fine-tuning Python environment
│   └── cleanup.sh                 # Clears HF/pip/torch caches, recreates workspace dirs
│
├── nutrition_table_inference/      # Inference and evaluation code
│   ├── src/
│   │   ├── inference_eval.py      # Per-sample HF and vLLM latency + accuracy benchmark
│   │   ├── vllm_throughput.py     # vLLM serving throughput measurement
│   │   ├── eval_hf_dataset2.py    # Batched HF evaluation over full dataset
│   │   ├── eval_vllm_dataset.py   # vLLM evaluation over full dataset
│   │   ├── quantize_qwen2vl_gptq.py  # 4-bit GPTQ quantization (partially working)
│   │   ├── model_utils.py         # Inference-side model loading helpers
│   │   └── viz_utils_infer.py     # Visualization for inference outputs
│   ├── model/Qwen2-VL-7B/
│   │   └── final_model_exp13/     # Active inference model (keep — all inference commands point here)
│   ├── env_infer/                  # Inference Python environment (separate from fine-tuning)
│   ├── .secrets                    # HF_TOKEN, WANDB_API_KEY (per-subproject, gitignored)
│   └── cleanup.sh                 # Same cache-cleaning script
│
├── README.md                       # Public landing page
├── project_report.md               # Concise technical report (interview-facing)
└── CLAUDE.md                       # This file (full operational reference)
```

> Docs split: `README.md` is the landing page, `project_report.md` is the concise writeup, `CLAUDE.md` (this file) holds the full command/config/gotcha reference. The old `Notes_*.md`, `results_summary.md`, `inference_summary.md`, and per-subfolder READMEs were consolidated into these three and removed.

---

## Environments

**Two separate venvs** — do not cross-contaminate them.

```bash
# Fine-tuning
source /workspace/projects/nutrition_table/nutrition_table_fine_tuning/.venv/bin/activate

# Inference
source /workspace/projects/nutrition_table/nutrition_table_inference/env_infer/bin/activate
```

To load secrets:
```bash
export $(grep -v '^#' /workspace/projects/nutrition_table/nutrition_table_inference/.secrets | xargs)
```

To recreate the inference env from scratch:
```bash
cd /workspace/projects/nutrition_table/nutrition_table_inference
rm -rf env_infer
python3 -m venv env_infer
source env_infer/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_inference.txt
```

---

## Training

### Run training (from fine_tuning dir)

```bash
cd /workspace/projects/nutrition_table/nutrition_table_fine_tuning
source .venv/bin/activate

# Single GPU
python src/train.py --config configs/exp13.yaml

# Multi-GPU (2 GPUs)
accelerate launch --multi_gpu --mixed_precision=bf16 src/train.py --config configs/exp13.yaml
```

Debug configs merge on top of `configs/exp1.yaml` (any config with "debug" in the name triggers this).

### Architecture

- **Base model**: `Qwen/Qwen2-VL-7B-Instruct`
- **Method**: LoRA (PEFT) applied to frozen base; adapters merged into base between stages
- **Trainer**: HuggingFace `SFTTrainer` + `accelerate`
- **Precision**: BF16 + Flash Attention 2 + fused AdamW (`optim="adamw_torch_fused"`)
- **Quantization during training**: optional 4-bit NF4 (BitsAndBytes) — off in exp13
- **Best model selection**: `metric_for_best_model="eval_loss"`, `load_best_model_at_end=true`
- **Early stopping**: patience=3, threshold=0.001

### Multi-Stage Training Strategy

The pipeline supports up to 3 stages. Each stage trains fresh LoRA adapters, then merges them into the base before the next stage starts (clean full-weight model each time, not adapter-on-adapter).

| Stage (general) | Target Layers | Purpose |
|-----------------|--------------|---------|
| 1 — Vision warmup *(optional)* | Last 8 vision blocks (`attn.qkv`, `attn.proj`) | Light orientation of vision encoder |
| 2 — Full vision | All vision `blocks.*.attn` + `blocks.*.mlp` + `merger.mlp` | Full visual feature learning |
| 3 — Joint | Full vision targets + LLM `q/k/v/o_proj` + `gate/up/down_proj` | End-to-end vision-language alignment |

**exp13 is a 2-stage experiment** — stage 1 (warmup) was dropped because experiments showed it added training time without measurable gain when stage 2 already targets the full vision encoder:

| exp13 Stage | Name | Epochs | Batch | Grad Accum | LR | LoRA r/α |
|-------------|------|:------:|:-----:|:----------:|:--:|:--------:|
| 1 — Full vision | `exp13_mod121925_full_visio` | 6 | 2 | 16 | 1e-4 | 16/32 |
| 2 — Joint | `exp13_mod121925_joint` | 6 | 1 | 32 | 1e-5 | 16/32 |

Task prompt: `"Detect the bounding boxes of all nutrition tables in the image."`
Image resolution: `min_pixels=784`, `max_pixels=705600`. `max_seq_length` is intentionally not set (truncation breaks image tokens).

### Critical Bug: Multi-GPU Adapter Merging (Fixed)

During `exp10`, adapter merging after multi-GPU training silently corrupted weights: merged-model IoU dropped from ~0.79 to ~0.30 while the in-memory (pre-merge) model scored correctly. Multiple `_merge_test_repro_*` variants isolated the root cause:

- **Cause**: all processes simultaneously called `merge_and_unload()` + `save_pretrained()`, writing conflicting/partial weight shards
- **Fix**: only `is_main_process` performs the merge and save; `wait_for_everyone()` sync barrier before and after
- **Additional fix**: each stage loads from the previously merged directory as its base, not raw adapters

After the fix, pre-merge and post-merge evaluations agree within ~1%.

### Evaluation during training

```bash
python src/eval_utils.py \
  --from_dir runs/<exp_name>/checkpoint-N \
  --config configs/<exp>.yaml \
  --tag strict_eval \
  --n 123
```

This command previously did not work — `eval_utils.py` had no `argparse` and no `__main__`.
It now has both, plus `--model_id`, `--use_adapters`, `--quantized` and `--out`.

### Tests

```bash
pip install pytest && python -m pytest     # from the repo root
```

`src/metrics.py` and `src/prompts.py` are dependency-free (no torch) so the correctness suite
runs anywhere in seconds; tests needing torch skip themselves. Both files are duplicated into
each subproject and tests assert the copies stay byte-identical — **edit one, copy it over the
other**.

**CI is active**: `.github/workflows/ci.yml` runs pytest on Python 3.10 and 3.12 plus a ruff
smoke lint, on every push to `main` and every PR. pytest and ruff are version-pinned so a
future upstream release cannot turn the build red on its own.

Note that the local `.venv` push token lacks GitHub's `workflow` scope, so **changes to
`.github/workflows/*` cannot be pushed from this machine** — edit that file through the GitHub
web UI, or use a token with `workflow` scope.

### Training Results

> **Note**: "1B" and "2B" in the experiment names below refer to **batch sizes** (1 or 2 per device), NOT model parameter counts. The model is always Qwen2-VL-7B (7 billion parameters).

**Early experiments (exp1–exp11)** — pre-merge-bug-fix, rough numbers:

| Experiment | Config | Final Mean IoU |
|-----------|--------|:--------------:|
| exp1 (vision blocks 20–23) | 4 blocks | 0.37 |
| exp1 (8 blocks 16–23) | 8 blocks | 0.35 |
| exp1 (all vision) | full vision | 0.34 |
| exp9 | single-stage answer format | 0.35 |
| exp10 | two-stage (language + 4 vision → + IoU loss) | 0.35 |
| exp11 | two-stage (full vision → language + full vision) | Stage1: 0.29, Stage2: 0.33 |

**Post-fix scaling study (repro series on same 7B base, different batch/GPU configs):**

| Run | Batch/GPU | GPUs | Time | Stage 1 IoU | Stage 2 IoU | Stage 3 IoU |
|-----|-----------|:----:|:----:|:-----------:|:-----------:|:-----------:|
| repro_15 | bs=1 | 1× RTX Pro 6000 | 120 min | 0.63 | 0.839 | 0.845 |
| repro_9  | bs=1 | 2× RTX Pro 6000 | 97 min  | 0.76 | 0.85  | 0.87  |
| repro_11 | bs=2 | 1× RTX Pro 6000 | 150 min | 0.76 | 0.879 | 0.886 |
| **repro_12** | **bs=1** | **2× RTX Pro 6000** | **89 min** | **0.75** | **0.852** | **0.893** |

Best training result: **mean IoU 0.893** (repro_12, 3-stage, 2 GPUs, 89 min total).

**Final model (exp13) accuracy on 123-sample val set:**

| Metric | Value |
|--------|:-----:|
| Mean IoU | **0.82** |
| Precision@0.5 | 0.90–0.92 |
| Recall@0.5 | 0.88–0.90 |
| F1@0.5 | 0.89–0.91 |

The 0.82 vs 0.893 gap is because they are different experiments: repro_12 used a 3-stage schedule tuned over many iterations; exp13 is a cleaner 2-stage production run.

---

## Inference

### Load secrets first

```bash
source /workspace/projects/nutrition_table/nutrition_table_inference/env_infer/bin/activate
export $(grep -v '^#' /workspace/projects/nutrition_table/nutrition_table_inference/.secrets | xargs)
```

### HF inference (offline, batched)

```bash
cd /workspace/projects/nutrition_table/nutrition_table_inference

# Unquantized
python -m src.inference_eval \
  --backend hf \
  --model model/Qwen2-VL-7B/final_model_exp13 \
  --run_name exp13_hf_bs4 \
  --batch_size 4

# 4-bit quantized — NOTE: not on disk anymore. Pull MayaKD/qwen2-vl-7b-gptq-nutrition
# (private) to this path first, or the command will fail.
python -m src.inference_eval \
  --backend hf \
  --model model/Qwen2-VL-7B/final_model_exp13_quantized_4bit \
  --run_name exp13_hf_q_bs4 \
  --batch_size 4
```

### vLLM serving

```bash
cd /workspace/projects/nutrition_table/nutrition_table_inference
source env_infer/bin/activate

# If 'python' resolves to the wrong interpreter, fix PATH first:
export PATH="/workspace/projects/nutrition_table/nutrition_table_inference/env_infer/bin:$PATH"
hash -r

# Start server in a separate terminal (runs in foreground)
# --host 0.0.0.0 is required for remote/RunPod access
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000

# Test the server is up
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "/workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13",
        "messages": [{
          "role": "user",
          "content": [
            {"type": "text", "text": "Detect the bounding boxes of all nutrition tables in the image."},
            {"type": "image_url", "image_url": {"url": "https://static.openfoodfacts.org/images/products/27563564/2.jpg"}}
          ]
        }],
        "max_tokens": 300,
        "temperature": 0.0
      }'

# Per-sample latency + accuracy eval against running server
python -m src.inference_eval \
  --backend vllm \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --run_name exp13_vllm_bs4 \
  --batch_size 4 \
  --server_url http://127.0.0.1:8000/v1

# Serving throughput benchmark. c=8 peaks throughput (17.82 req/s); c=4 is the
# better latency/throughput compromise; c=16 collapses. This is the ONLY valid
# way to benchmark vLLM here — see the note in Inference Results.
python -m src.vllm_throughput \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --run_name vllm_c4 --concurrency 4

# Full dataset eval via vLLM
python -m src.eval_vllm_dataset \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --num_samples 123 \
  --name exp13_vllm
```

### Batched HF dataset eval

```bash
python -m src.eval_hf_dataset2 \
  --from_dir model/Qwen2-VL-7B/final_model_exp13 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --split val --num_samples 123 \
  --batch_size 4 --max_new_tokens 256 \
  --name exp13_eval \
  --out_dir outputs/hf_eval/exp13_eval
```

### Qwen2-VL sequence length notes

Qwen2-VL has **no fixed max sequence length** — do not set `max_seq_length` or truncate (it breaks image tokens). vLLM controls this via `max_model_len` (ceiling) and `max_num_batched_tokens` (scheduling); chunking is transparent.

---

## Inference Results

Accuracy is identical across backends (same exp13 weights, Mean IoU 0.82 — see Training Results above). These benchmarks measure speed and memory only.

### Inference benchmark (exp13 final model, 123 val samples, single GPU)

**HF offline batching** (real batching: one padded `generate()` per batch) — source `outputs/eval/exp13_hf_bs*_1.json`:

| Backend | Batch | Mean IoU | Mean Latency (ms/sample) | P95 (ms) | GPU Mem (GB) |
|---------|:-----:|:--------:|:------------------------:|:--------:|:------------:|
| HF | 1 | 0.82 | 890.32 | 989.31 | 17.16 |
| HF | 4 | 0.82 | **475.29** | 483.48 | 20.37 |
| HF | 16 | 0.82 | 320.91 | 532.85 | 29.92 |

**vLLM serving concurrency** (real async concurrency) — source `outputs/throughput/vllm_c*.json`:

| Concurrency | Mean Latency (ms) | P95 (ms) | Requests/s | GPU Mem (GB) |
|:-----------:|:-----------------:|:--------:|:----------:|:------------:|
| 1 | 1,377.62 | 1,505.59 | 0.73 | 72.51 |
| 4 | 398.39 | 466.67 | 9.89 | 72.51 |
| **8** | **434.59** | 1,056.33 | **17.82** | 72.51 |
| 16 | 16,527.85 | 22,652.71 | 0.94 | 72.51 |

Throughput peaks at **c=8 (17.82 req/s)**; c=16 falls off a cliff (0.94 req/s, 22.7 s P95) as the scheduler thrashes. c=4 is the best latency/throughput compromise if P95 matters; c=8 if raw throughput matters.

vLLM's 72 GB memory usage is intentional pre-allocation for continuous batching, not a bug.

> **Do not benchmark vLLM with `inference_eval.py --batch_size`.** `vllm_infer_batch()` issues requests in a sequential `for` loop, so `--batch_size` only changes the divisor at `inference_eval.py:296` — it does not batch or parallelize anything. Any vLLM "batch size" comparison from that script is measuring run-to-run noise. Use `vllm_throughput.py` (real `asyncio.Semaphore` concurrency) for all vLLM performance claims. The HF path in the same script *does* batch properly and its numbers are valid.

**When to use which backend:**
- **HF**: model evaluation, research, offline dataset processing (batch_size=4, or 16 if VRAM allows)
- **vLLM**: production API, multi-user serving (c=4 for latency-sensitive, c=8 for max throughput; never c=16)

---

## Quantization

4-bit GPTQ via `GPTQModel` (now a package-relative module, run with `-m` like the others):
```bash
cd /workspace/projects/nutrition_table/nutrition_table_inference
python -m src.quantize_qwen2vl_gptq
```

**Status**: the local copy at `model/Qwen2-VL-7B/final_model_exp13_quantized_4bit` is gone from disk, but the weights are **safe on HF** at `MayaKD/qwen2-vl-7b-gptq-nutrition` (private, 6.94 GB, 2 shards + `quantize_config.json` + `quant_log.csv`). Re-download from there rather than re-running the quantization.

It works for **HF inference**. vLLM serving is **blocked**:
- Error: `KeyError: 'layers.10.mlp.down_proj.g_idx'`
- Root cause: Qwen2-VL is a vision-language model, not a standard `AutoModelForCausalLM`; GPTQModel's output tensor naming for VLMs does not match vLLM's GPTQ loader expectations
- `AutoGPTQ` was also tried but failed at CUDA extension compilation (ninja build error)
- **Unresolved** — vLLM + quantized Qwen2-VL serving needs upstream fix or different quant tool

---

## Cache / Disk Cleanup

```bash
# Normal (clears HF, pip, torch, wandb caches)
bash /workspace/projects/nutrition_table/nutrition_table_inference/cleanup.sh

# Check disk
df -h / /workspace
```

---

## HuggingFace Upload

Repo `MayaKD/qwen2-vl-7b-nutrition` contains two folders mirroring the local run structure (both fully uploaded, 28 files each):
- `exp13_mod121925_full_visio/` — exp13 stage 1 (full vision) adapter + checkpoint-102 + merged model
- `exp13_mod121925_joint/` — exp13 stage 2 (joint) adapter + checkpoint-102 + merged model

**The entire local `nutrition_table_fine_tuning/runs/` tree is gone** (verified 2026-08-14) — not just the `merged/` and `checkpoint-102/` subfolders, but the final adapters too. HF is now the *only* copy of all training output. To retrain, resume, or recover any adapter, merged weights, or optimizer/scheduler/RNG state, pull from `MayaKD/qwen2-vl-7b-nutrition` — both stages are complete there (57 files, 34 GB, including `training_args.bin`, `rng_state_{0,1}.pth`, `scheduler.pt`, `optimizer.pt`).

The active inference model at `model/Qwen2-VL-7B/final_model_exp13/` is a separate flat copy kept on disk (see below).

**Upload gotcha — cgroup memory limit**: this container has a ~3.8 GB cgroup memory limit. `upload_folder` OOMs because it tries to hash all files at once. Must upload file-by-file:

```python
api.upload_file(path_or_fileobj=local_path, path_in_repo=hf_path, repo_id=REPO_ID, ...)
```

Use `nohup` so the process survives session timeouts:
```bash
nohup bash -c 'source .venv/bin/activate && python3 upload_script.py' > upload.log 2>&1 &
```

**vLLM cannot load from HF subfolder**: vLLM doesn't support the `subfolder=` parameter. The model must be served from a local path. Do not delete `model/Qwen2-VL-7B/final_model_exp13/` — all inference commands depend on it. It is a flat copy of the joint stage (merged weights + adapter + tokenizer) kept locally for serving.

---

## HuggingFace Space (Triton + vLLM proxy)

Space: `MayaKD/nutrition-table-detector-triton-vllm-proxy` (private by default).

Architecture: **Gradio (7860) → Triton (8000) → vLLM (8008)**

- `launch.sh` starts vLLM on port **8008**, then Triton, then Gradio.
- Triton Python backend (`model_repository/vllm_model/1/model.py`) calls vLLM at `localhost:8008`.
- Gradio (`app.py`) calls Triton at `localhost:8000`.

**Two HF model repos — use the right one:**
- `MayaKD/qwen2-vl-7b-nutrition` — training backup; weights are in subfolders (`exp13_.../merged/`). Cannot be loaded directly by vLLM.
- `MayaKD/qwen2-vl-7b-nutrition-vllm` — flat copy of final merged weights at repo root. **This is the one vLLM serves.** `launch.sh` and `model.py` must both reference this name.

**Prompt must match training exactly** — the model is sensitive to prompt deviations:
```
system: "You are a Vision Language Model specialized in interpreting visual data from product images.\nYour task is to analyze the provided product images and detect the nutrition tables in a certain format.\nFocus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."
user:   "Detect the bounding boxes of all nutrition tables in the image."
```
The system message must be defined at module level (no indentation on continuation lines) — indenting the string literal adds leading spaces to each line and changes the string the model sees.

**GPU recommendation**: L4 (24 GB, $0.80/hr) — right-sized for this model. A100 (80 GB) allocates a massive KV cache and takes ~7 min to start. A10G small has only 15 GB system RAM, which is tight for two heavy processes.

**vLLM cold-start on L4**: ~2–3 min. The wait loop in `launch.sh` must be long enough (≥300 s) or Triton/Gradio will start before vLLM is ready, causing `Connection refused` errors.

---

## Dataset Annotation Notes

The `openfoodfacts/nutrition-table-detection` val set has 123 samples:
- **117 samples have exactly 1 bounding box** (95%)
- 5 samples have 2 boxes, 1 sample has 3 boxes
- Some multi-box annotations are noise (e.g. a 2×1 pixel box alongside the real one)

The dataset annotates only the **official EU-format nutrition declaration** (the standardized "Per 100g" table). Colorful summary panels, "Per portion" columns, and other non-standard nutrition displays are **not annotated** even when visually present. A product image showing two side-by-side nutrition columns (one colorful, one standard) will have only one ground-truth box. The model correctly learns this: it detects the EU-format table and ignores the rest.

**Implication for demos**: crowded multi-product shelf images are out of distribution. Use single-product images (one label in frame) for reliable results.

---

## Known Issues / Gotchas

1. **Do not set `max_seq_length`** in configs — it truncates image tokens and breaks training/inference.
2. **Multi-GPU merge guard**: `save_pretrained` in the merge step must be wrapped in `if accelerator.is_main_process` + `wait_for_everyone()`. Without this, merged weights are silently corrupted (IoU drops from ~0.79 to ~0.30).
3. **1B/2B terminology in older notes**: these mean batch_size=1 or batch_size=2, not model size. Model is always 7B.
4. **Two separate venvs**: `.venv` (fine-tuning) and `env_infer` (inference) — do not mix.
5. **Canonical model path**: the inference model base path is `/workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/`. Some older/stray commands may reference outdated paths — always use this one.
6. **vLLM + 4-bit quantization**: unresolved as of last experiment (see Quantization section).
7. **cgroup memory limit (~3.8 GB)**: despite large system RAM, this container is cgroup-limited. Any Python process that tries to hold multiple large files in memory (e.g. `upload_folder`, loading multiple model shards) will be OOM-killed. Upload files one at a time; be cautious with scripts that load large tensors outside of GPU memory.
8. **cleanup.sh nukes the HF cache**: `~/.cache` and `/workspace/.cache` are deleted by cleanup.sh. If you ever load a model from HF Hub (not local disk), it will need to re-download after cleanup. Always use local paths for inference.
9. **`model/Qwen2-VL-7B/final_model_exp13/` is the active inference model** — keep it while this box is in use, but it *does* have a remote fallback: `MayaKD/qwen2-vl-7b-nutrition-vllm` is a flat repo (weights at root, no subfolder) that vLLM can serve directly, and its shards are byte-for-byte the same size as the local ones. Losing the local copy costs a re-download, not the model. (The local dir additionally carries `adapter_model.safetensors`, `optimizer.pt`, `trainer_state.json` and friends — those are backed up under `exp13_mod121925_joint/checkpoint-102/` in `MayaKD/qwen2-vl-7b-nutrition`.)
10. **HF Space model name must match vLLM**: `launch.sh` serves `MayaKD/qwen2-vl-7b-nutrition-vllm`; `model.py` must send requests with exactly that string as the `model` field — any mismatch returns 404.
11. **Published precision/recall/F1 are upper bounds**: they were computed by counting every IoU-matrix cell above 0.5 instead of matching one-to-one, so overlapping predictions overcounted true positives (recall could exceed 1.0). Fixed in `metrics.py` with greedy matching + a regression test. **The committed numbers have not been re-measured — that needs a GPU.** Mean IoU is unaffected (threshold-free, never used the matching path).
12. **All prompts come from `prompts.py`**: never hardcode a prompt string. The repo previously held three divergent variants — the training prompt, a singular-phrasing variant defaulted into several eval scripts, and a strict-format instruction hardcoded inside `call_vllm()` that silently overrode its own `prompt` argument. The last one produced the 0.573 result in `outputs/vllm_eval/`.
13. **Benchmarks used a placeholder system prompt**: every committed JSON records `"system_text": "System message"`, not the real training system message. That default is retained so the numbers stay reproducible; pass `--system_text` to test the trained configuration. The HF Space uses the real system message, so Space output and benchmark numbers are not strictly comparable.
