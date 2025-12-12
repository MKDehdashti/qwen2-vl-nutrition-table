# Commands
source /workspace/projects/nutrition-table3/.venv/bin/activate
source /workspace/projects/nutrition_table_inference/env_infer/bin/activate

vllm serve model/Qwen2-VL-7B/final_model --dtype bfloat16

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/projects/nutrition-table3/model/Qwen2-VL-7B/quantized_gptqmodel \
  --dtype float16 \
  --quantization gptq \
  --host 0.0.0.0 \
  --port 8000

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

archive env:
tar -czf venv_backup.tar.gz .venv
rm -rf .venv
re-install:
tar -xzf venv_backup.tar.gz



 pip install -r projects/nutrition-table/requirements.txt

 accelerate launch --mixed_precision=bf16 --multi_gpu projects/nutrition-table/src/train.py
 accelerate launch --mixed_precision=bf16 /workspace/projects/nutrition-table/src/train.py --config configs/exp1.yaml
  accelerate launch --mixed_precision=bf16 src/train.py --config configs/exp3.yaml
  accelerate launch --multi_gpu --mixed_precision=bf16 src/train.py --config configs/exp10_merge_test_repro.yaml

  accelerate launch --num_processes=1 --mixed_precision=bf16 src/train.py --config configs/exp7_2.yaml


  python src_comb/train.py --config configs/exp10-2.yaml
    python src_flash_format/train.py --config configs/exp9_debug.yaml
    python src_flash_format/train.py --config configs/exp10.yaml

python src/train.py --config configs/exp10_merge_test_repro.yaml

  python src/cleanup.py --deep

run viz:
python train.py --config configs/exp10.yaml --precheck-idx 20
python train.py --config configs/exp10.yaml --postcheck-idx 5,42 --postcheck-split val


  python src/viz_utils.py --from_dir runs/exp2_numeric_resume_20250911/lang_vision --idx 15



 python -u src/train.py --config configs/exp1_debug.yaml
 python -u src/train.py --config con(.venv) root@8791c11f9ca0:/workspace#  accelerate launch --mixed_precision=bf16 /workspace/projects/nutrition-table/src/train.py --config configs/exp1.yaml

 python src/eval_utils.py \
  --from_dir runs/exp9_joint_training/checkpoint-220 \
  --config configs/exp9.yaml \
  --tag strict_eval \
  --n 123

  python src/eval_utils.py \
  --from_dir runs/exp10_joint_training/checkpoint-200 \
  --config configs/exp10.yaml \
  --tag strict_eval_exp10 \
  --n 123 \
  --print_iou_per_sample

python src/eval_utils.py \
  --from_dir runs/exp1-2_20250925_1837/exp1-2_lang_vision/checkpoint-400 \
  --config configs/exp1-2.yaml \
  --tag strict_eval_flash \
  --n 123 \
  --print_iou_per_sample


python src/eval_loose.py \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --from_dir runs/exp9_joint_training/checkpoint-220 \
  --n 123


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

    import os, glob, json, re
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
os.environ["PEFT_BACKEND"] = "HF"
from peft import LoraConfig, PeftModel
from accelerate import PartialState

# LoRA config template
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[]
)


def save(trainer, processor, training_args, metrics=None, tag=None):
    """
    Save adapters + merged full model + processor + metrics.
    """
    if trainer is not None:
        # Save raw adapter model
        trainer.model.save_pretrained(training_args.output_dir)

        # Try merging adapters into the base model
        try:
            if hasattr(trainer.model, "merge_and_unload"):
                print("🔄 Merging adapters into base model...")
                merged = trainer.model.merge_and_unload()
                merged.save_pretrained(training_args.output_dir)
                print(f"✅ Merged model saved to {training_args.output_dir}")
            else:
                print("⚠️ No merge_and_unload found, skipping merge.")
        except Exception as e:
            print(f"⚠️ Merge failed: {e}")

    # Save processor
    processor.save_pretrained(training_args.output_dir)

    # Save metrics
    if metrics:
        suffix = f"_{tag}" if tag else ""
        metrics_path = os.path.join(training_args.output_dir, f"metrics{suffix}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to {metrics_path}")

    print(f"✅ Saved to {training_args.output_dir}")


def _find_processor_src(base_out, adapter_path, base_id="Qwen/Qwen2-VL-7B-Instruct"):
    candidates = [base_out, adapter_path, base_id]
    config_names = ["preprocessor_config.json", "processor_config.json", "image_processor_config.json"]
    for c in candidates:
        if c is None:
            continue
        for name in config_names:
            if os.path.isfile(os.path.join(c, name)):
                return c
    return base_id


def load_model(dtype=torch.float32, quantized=False, use_adapters=True, from_dir=None, training_args=None):
    proj_root = os.path.dirname(os.path.dirname(__file__))
    local_models = os.path.join(proj_root, "models", "transformers")
    base_id = "Qwen/Qwen2-VL-7B-Instruct"

    local_path = os.path.join(local_models, base_id.replace("/", "_"))
    model_src = local_path if os.path.isdir(local_path) else base_id

    # Adapter path
    base_out = None
    if from_dir:
        base_out = from_dir
    elif training_args is not None and hasattr(training_args, "output_dir"):
        base_out = training_args.output_dir

    adapter_path = None
    if base_out:
        ckpts = sorted(glob.glob(os.path.join(base_out, "checkpoint-*")), key=os.path.getmtime)
        adapter_path = ckpts[-1] if ckpts else base_out

    # Quantization config
    quant_cfg = None
    if quantized:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load base model
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_src,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map={"": PartialState().process_index},
    )

    # Processor source
    proc_src = None
    if adapter_path:
        proc_src = _find_processor_src(base_out, adapter_path, base_id)
    if not proc_src:
        proc_src = model_src

    processor = Qwen2VLProcessor.from_pretrained(
        proc_src,
        min_pixels=224 * 224,
        max_pixels=900 * 28 * 28,
    )

    # No adapters
    if not adapter_path or not use_adapters:
        return base, processor

    # Load adapters if present
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    has_adapter = any(os.path.exists(os.path.join(adapter_path, f)) for f in adapter_files)
    if not has_adapter:
        return base, processor

    model = PeftModel.from_pretrained(base, adapter_path)
    return model, processor


def get_matching_modules(model, regex_list):
    names = [n for n, _ in model.named_modules()]
    out = set()

    for pat in regex_list:
        rx = re.compile(pat)
        for n in names:
            if rx.search(n):
                out.add(n)

    if not out:
        for pat in regex_list:
            for n in names:
                if n.endswith(pat):
                    out.add(n)

    if not out:
        sample = "\n".join(names[:30])
        raise ValueError(f"No modules matched any of {regex_list}. Sample names:\n{sample}")

    return sorted(out)



import os, yaml, torch
from datetime import datetime
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from dataset.data_utils import train_ds, test_ds
from dataset.collators import collate_fn
from model_utils import save, load_model, get_matching_modules
from eval_utils import run_and_log_eval
from wandb_utils import init_wandb, WandBLossCallback
from cleanup import clean_cache
from datasets import load_dataset


if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("⚠️ No GPU found — forcing CPU mode")
else:
    print("⚡ GPU available — running on CUDA")

state = PartialState()
is_main = state.is_main_process


def get_proj_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path):
    proj_root = get_proj_root()
    full_path = os.path.join(proj_root, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    with open(full_path) as f:
        return yaml.safe_load(f)


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for k, v in override.items():
        if k in merged:
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.environ.get("TRAIN_CONFIG"))
    args, _ = parser.parse_known_args()
    if not args.config:
        raise SystemExit("Missing config. Pass --config or set TRAIN_CONFIG.")
    if "debug" in os.path.basename(args.config):
        base_cfg = load_config("configs/exp1.yaml")
        override_cfg = load_config(args.config)
        cfg = deep_merge(base_cfg, override_cfg)
    else:
        cfg = load_config(args.config)

    if "debug" in cfg:
        dbg = cfg["debug"]
        print("🐛 Debug mode from config")
        dataset = load_dataset("openfoodfacts/nutrition-table-detection")
        dataset_small = dataset["train"].select(range(dbg.get("dataset_subset", 20)))
        cfg["dataset_debug"] = dataset_small
        if "wandb" not in cfg:
            cfg["wandb"] = {}
        cfg["wandb"]["name"] = cfg["wandb"].get("name", cfg["experiment"]) + dbg.get("wandb_suffix", "-debug")
        cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + dbg.get("wandb_tags", ["debug"])

    proj_root = get_proj_root()
    runs_root = os.path.join(proj_root, "runs")
    os.makedirs(runs_root, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = cfg["experiment"]

    run_dir = os.path.join(runs_root, f"{exp_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    if is_main:
        import wandb
        wandb.init(project="nutrition-table-vl", group=f"{exp_name}_{run_id}", job_type="meta",
                   name=f"meta_{exp_name}_{run_id}", config=cfg)
        wandb.finish()

    # === Baseline Eval ===
    if is_main:
        init_wandb(run_id, {}, None, stage_name="baseline", exp_name=exp_name)

        resume_ckpt = cfg.get("resume_from")
        if resume_ckpt and os.path.isdir(resume_ckpt):
            print("\n=== Baseline Evaluation (resume checkpoint) ===")
            run_and_log_eval("baseline", tag="baseline", use_adapters=True, from_dir=resume_ckpt)
        else:
            print("\n=== Baseline Evaluation (base model) ===")
            run_and_log_eval("baseline", tag="baseline", use_adapters=False)

        import wandb; wandb.finish()

    # === Stage loop ===
    prev_dir = None
    for stage_idx, stage_cfg in enumerate(cfg["stages"]):
        name = stage_cfg["name"]
        regex_targets = stage_cfg["regex_targets"]

        out_dir = os.path.join(run_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        training_args = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=stage_cfg["epochs"],
            per_device_train_batch_size=stage_cfg["batch_size"],
            gradient_accumulation_steps=stage_cfg["grad_accum_steps"],
            learning_rate=float(stage_cfg["lr"]),
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            bf16=True,
            tf32=True,
            report_to="wandb",
            logging_steps=10,
            eval_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}
        )
        training_args.remove_unused_columns = False

        # Always get target modules from a clean base model
        clean_base, _ = load_model(dtype=torch.bfloat16, use_adapters=False, quantized=True)
        peft_cfg = LoraConfig(
            r=stage_cfg["lora_r"],
            lora_alpha=stage_cfg["lora_alpha"],
            lora_dropout=stage_cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_matching_modules(clean_base, regex_targets)
        )
        del clean_base; torch.cuda.empty_cache()

        # Load model for training
        if stage_idx == 0:
            print(f"\n=== Stage: {name} (starting from base model) ===")
            base_model, proc = load_model(dtype=torch.bfloat16, use_adapters=False, quantized=True)
        else:
            print(f"\n=== Stage: {name} (resuming from {prev_dir}) ===")
            base_model, proc = load_model(dtype=torch.bfloat16, use_adapters=True, quantized=True, from_dir=prev_dir)

        if is_main:
            init_wandb(run_id, training_args, peft_cfg, stage_name=name, exp_name=exp_name)

        base_model.enable_input_require_grads()
        base_model.config.use_cache = False

        collator_cfg = cfg.get("collator", {})
        numeric_only = bool(collator_cfg.get("numeric_only", False))

        trainer = SFTTrainer(
            model=base_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=lambda ex: collate_fn(ex, proc, numeric_only=numeric_only),
            peft_config=peft_cfg,
            callbacks=[WandBLossCallback()]
        )

        trainer.train(resume_from_checkpoint=prev_dir if prev_dir else None)
        save(trainer, proc, training_args)

        run_and_log_eval(name, tag=f"post_{name}", use_adapters=True,
                         from_dir=out_dir, training_args=training_args)

        from viz_utils import quick_viz
        quick_viz(idx=20, from_dir=out_dir, split="train",
                  out_dir=out_dir, save_name=f"viz_post_{name}_train20.png",
                  use_adapters=True)

        del base_model, proc; torch.cuda.empty_cache()
        clean_cache(deep=False)

        if is_main:
            import wandb; wandb.finish()

        prev_dir = out_dir

    if is_main:
        print("✅ All stages complete")
