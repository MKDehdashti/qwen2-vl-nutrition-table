# Commands
source /workspace/projects/nutrition_table/nutrition_table_fine_tuning/.venv/bin/activate
source /workspace/projects/nutrition_table/nutrition_table_inference/env_infer/bin/activate

#################################################################
to recreate the env:
cd /workspace/projects/nutrition_table/nutrition_table_inference

# blow away the old broken venv
rm -rf env_infer

# create a fresh one
python3 -m venv env_infer

# activate it
source env_infer/bin/activate

# upgrade pip in the new env
python -m pip install --upgrade pip

Quick sanity check:
which python
which pip
python -c "import sys; print(sys.executable)"

python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_inference.txt
#################################################################
Quantization:
We tried:
Using GPTQModel to quantize → vLLM hits missing keys (rotary_pos_emb.inv_freq, then g_idx etc.).
Installing AutoGPTQ from PyPI → version metadata weirdness (0.7.1 vs 0.7.1+cu1281).
Installing AutoGPTQ from local clone with --no-build-isolation → now stuck at C++/CUDA extension compile failure (ninja).
Qwen2-VL isn’t an AutoModelForCausalLM – AutoGPTQ doesn’t know how to quantize that vision-language architecture out of the box.
The vLLM error:
KeyError: 'layers.10.mlp.down_proj.g_idx'
tells us:
vLLM has detected “this is a GPTQ-style checkpoint”, so it expects GPTQModel-style tensor names (including g_idx).
For some layer(s) (layers.10.mlp.down_proj), the checkpoint doesn’t have the expected GPTQ quant tensors – i.e., the quantization pass didn’t fully produce them in the way vLLM expects.
That’s an issue with how the checkpoint was produced by GPTQModel, not with AutoGPTQ.
So to make Option 2 work (quantized vLLM), we need to:
Use GPTQModel in the “vLLM-compatible” way, and
Ensure the saved folder looks like a standard HF/GPTQModel repo:
config.json (copied from your BF16 model)
tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
quant weights (model-00001-of-0000x.safetensors, etc.)
GPTQModel’s own quantize_config.json or similar.
You already added the config/tokenizer files – good. The missing piece is: making sure GPTQModel is saving weights in the exact format vLLM expects.






#################################################################
to quantize: 
python /workspace/projects/nutrition_table/nutrition_table_inference/src/quantize_qwen2vl_gptq.py

#################################################################
vllm:

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/projects/nutrition-table3/model/Qwen2-VL-7B/quantized_gptqmodel \
  --dtype float16 \
  --quantization gptq \
  --host 0.0.0.0 \
  --port 8000

# BF16, exp12
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --dtype bfloat16 \
  --port 8000

#################################################################
inference eval:
hf, unquantized:
python -m src.inference_eval --backend hf --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 --run_name exp13_hf_bs16_1 --batch_size 16

hf, quantized:
python -m src.inference_eval --backend hf --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13_quantized_4bit --run_name exp13_hf_bs4_1 --batch_size 4

vllm, unquantized:
python -m src.inference_eval --backend vllm --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 --run_name exp13_vllm_bs16_1 --batch_size 16 --server_url http://127.0.0.1:8000/v1

vllm, quantized:
python -m src.inference_eval --backend vllm --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13_quantized_4bit --run_name exp13_vllm_q_bs1_1 --batch_size 1 --server_url http://127.0.0.1:8000/v1

vllm_throughput:
python -m src.vllm_throughput \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --run_name vllm_c4 --concurrency 4

python -m src.eval_vllm_dataset \
  --model "/workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13_quantized_4bit" \
  --num_samples 123 \
  --name "exp13_quantized"

to fix python path issue:
export PATH="/workspace/projects/nutrition_table/nutrition_table_inference/env_infer/bin:$PATH"
hash -r
which python

test vllm:
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "model/Qwen2-VL-7B/final_model",
        "messages": [
          {
            "role": "user",
            "content": [
              { "type": "text", "text": "Detect the nutrition table bounding box." },
              {
                "type": "image_url",
                "image_url": { "url": "https://static.openfoodfacts.org/images/products/27563564/2.jpg" }
              }
            ]
          }
        ],
        "max_tokens": 300,
        "temperature": 0.0
      }'

######################################################
HF batch inference:

python -m src.eval_hf_dataset2 \
  --from_dir /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --split val --num_samples 123 \
  --batch_size 4 \
  --max_new_tokens 256 \
  --name exp13_2 \
  --out_dir outputs/hf_eval/exp13_2

cd /workspace/projects/nutrition_table/nutrition_table_inference
python -m src.eval_hf_dataset \
  --model /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --from_dir /workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13 \
  --num_samples 123 --split val --max_new_tokens 256 \
  --out_dir outputs/hf_eval --name exp13_hf


 
  accelerate launch --multi_gpu --mixed_precision=bf16 src/train.py --config configs/exp10_merge_test_repro.yaml


run viz:
python train.py --config configs/exp10.yaml --precheck-idx 20
python train.py --config configs/exp10.yaml --postcheck-idx 5,42 --postcheck-split val


  python src/viz_utils.py --from_dir runs/exp2_numeric_resume_20250911/lang_vision --idx 15

 python src/eval_utils.py \
  --from_dir runs/exp9_joint_training/checkpoint-220 \
  --config configs/exp9.yaml \
  --tag strict_eval \
  --n 123



python src/eval_utils.py \
  --from_dir runs/exp1-2_20250925_1837/exp1-2_lang_vision/checkpoint-400 \
  --config configs/exp1-2.yaml \
  --tag strict_eval_flash \
  --n 123 \
  --print_iou_per_sample

#################################################################
max sequence length for qwen2-vl:
Think in three layers:

Layer 1 — Model semantics (Qwen / Transformers)

No fixed max sequence length

No truncation by default

Multimodal tokens must stay intact

Layer 2 — Tokenization / preprocessing

You control image size

You avoid truncation

You let sequences be as long as needed

Layer 3 — vLLM runtime engine

max_model_len: absolute ceiling (capability)

max_num_batched_tokens: scheduling / memory control

Chunking is transparent and lossless
###############################################################3
# Results
    # My data format
        # My Eval:
            # with vision blocks 20-23:
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3654
            # loss': 2.04
            # 'eval_loss': 0.27

            # with whole vision
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3421
            # loss': 2.3
            # 'eval_loss': 0.30

            # with 8 vision 16-23:
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3547
            # loss': 2.02
            # 'eval_loss': 0.27

        # Optimistic
            # with vision blocks 20-23:
            # Mean IoU before: 0.3218
            # Mean IoU after: 0.3345
            # loss': 2.02
            # 'eval_loss': 0.27
            # train_runtime': 3179

        # Strict

    # Answer data format
        # Strict
            # with vision blocks 20-23:
            # Mean IoU before: 0.3218
            # Mean IoU after: 0.3362
            # loss': 1.13
            # 'eval_loss': 0.15
            # train_runtime': 3323

    # train 9
        # Mean IoU after:0.3532
        # loss': 1.20
        # 'eval_loss': 0.15

    # train 10 (two stages, 1: language + 4 vision blocks, 2: same with iou loss )
        # Mean IoU after: 0.3532

    # train 11 (two stages, 1: full vision, 2: language and full vision)
        # Mean IoU after stage 1: 0.29
        # Mean IoU after stage 2: 0.33




W&B chart:
Project: nutrition-table-vl
└── Group: exp1_20240902_123456
    ├── meta         → experiment setup only (YAML config, dataset sizes, seed, etc.)
    ├── baseline     → baseline evaluation (IoU, PR curve, sample images)
    ├── vision       → stage 1 training + eval metrics
    └── lang_vision  → stage 2 training + eval metrics

  