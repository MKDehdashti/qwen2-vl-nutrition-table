# train.py
import os, yaml, torch
from datetime import datetime
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel
from dataset.data_utils import get_datasets, format_data
from dataset.collators import collate_fn
from model_utils import save, load_model, get_matching_modules, load_processor_fixed
from eval_utils import evaluate_model
from wandb_utils import init_wandb, WandBLossCallback
from cleanup import clean_cache
from datasets import load_dataset
from transformers import EarlyStoppingCallback
import wandb
from viz_utils import quick_viz

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_DATASETS_DISABLE_CACHING"] = "0"

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
    parser.add_argument("--pre_viz", type=str, default=None, help="Run quick_viz before training: pass idx, 'none' to skip")
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
        raw = load_dataset("openfoodfacts/nutrition-table-detection")
        train_raw = raw["train"].select(range(dbg.get("dataset_subset", 20)))
        val_raw = raw["val"].select(range(dbg.get("dataset_subset", 20)))
        if cfg.get("format_data", False):
            train_ds = [format_data(ex, cfg) for ex in train_raw]
            test_ds = [format_data(ex, cfg) for ex in val_raw]
        else:
            train_ds, test_ds = train_raw, val_raw
        if "wandb" not in cfg:
            cfg["wandb"] = {}
        cfg["wandb"]["name"] = cfg["wandb"].get("name", cfg["experiment"]) + dbg.get("wandb_suffix", "-debug")
        cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + dbg.get("wandb_tags", ["debug"])
    else:
        train_ds, test_ds = get_datasets(cfg, format_data_flag=cfg.get("format_data", False))

    proj_root = get_proj_root()
    runs_root = os.path.join(proj_root, "runs")
    os.makedirs(runs_root, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = cfg["experiment"]
    run_dir = os.path.join(runs_root, f"{exp_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    processor = load_processor_fixed(model_id=cfg["model_id"], cfg=cfg)
    prev_dir = None

    subset_size = cfg.get("eval_subset_size", 12)
    if hasattr(test_ds, "select"):
        eval_subset = [test_ds[i] for i in range(min(subset_size, len(test_ds)))]
    else:
        eval_subset = test_ds[:subset_size]

    if args.pre_viz and args.pre_viz.lower() != "none":
        try:
            idx = int(args.pre_viz)
            print(f"🔍 Running quick_viz on sample {idx} before training...")
            quick_viz(idx=idx, from_dir=None, split="train", cfg=cfg)
        except ValueError:
            print(f"⚠️ Invalid pre_viz arg {args.pre_viz}, skipping")

    for stage_idx, stage in enumerate(cfg["stages"]):
        name = stage["name"]
        regex_targets = stage["regex_targets"]
        out_dir = os.path.join(run_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        training_args = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=stage["epochs"],
            per_device_train_batch_size=stage["batch_size"],
            per_device_eval_batch_size=stage.get("eval_batch_size", stage["batch_size"]),
            gradient_accumulation_steps=stage["grad_accum_steps"],
            learning_rate=float(stage["lr"]),
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=True if torch.cuda.is_available() else False,
            report_to="wandb",
            logging_steps=cfg.get("logging_steps", 10),
            eval_strategy=cfg.get("eval_strategy", "steps"),
            eval_steps=stage.get("eval_steps", cfg.get("eval_steps", 50)),
            save_strategy=cfg.get("save_strategy", "steps"),
            save_steps=stage.get("save_steps", cfg.get("save_steps", 50)),
            save_total_limit=2,
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=0.3,
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            lr_scheduler_type=stage.get("lr_scheduler_type", cfg.get("lr_scheduler_type", "linear")),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_accumulation_steps=1,
            include_inputs_for_metrics=False,
            prediction_loss_only=False,
        )
        training_args.remove_unused_columns = False

        clean_base, _ = load_model(model_id=cfg["model_id"], dtype=torch.bfloat16, use_adapters=False, cfg=cfg)
        peft_cfg = LoraConfig(
            r=stage["lora_r"],
            lora_alpha=stage["lora_alpha"],
            lora_dropout=stage["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_matching_modules(clean_base, regex_targets),
        )
        del clean_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if stage_idx == 0:
            print(f"\n=== Stage: {name} (starting from base model) ===")
            base_model, _ = load_model(model_id=cfg["model_id"], dtype=torch.bfloat16, use_adapters=False, cfg=cfg)
        else:
            print(f"\n=== Stage: {name} (starting from merged model of {prev_dir}) ===")
            base_model, _ = load_model(model_id=cfg["model_id"], dtype=torch.bfloat16, use_adapters=False, from_dir=prev_dir, cfg=cfg)

        if is_main:
            init_wandb(run_id, training_args, peft_cfg, stage_name=name, exp_name=exp_name)

        base_model.enable_input_require_grads()
        base_model.config.use_cache = False
        base_model.config.name_or_path = cfg["model_id"]

        collator_cfg = cfg.get("collator", {})
        numeric_only = bool(collator_cfg.get("numeric_only", False))

        trainer = SFTTrainer(
            model=base_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=lambda ex: collate_fn(ex, processor, cfg=cfg, numeric_only=numeric_only),
            peft_config=peft_cfg,
            callbacks=[WandBLossCallback(), EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
            compute_metrics=None,
            processing_class=processor,
        )

        trainer.train()

        full_metrics = evaluate_model(
            trainer.model,
            processor=processor,
            dataset=test_ds,
            n=123,
            strict=True,
            tag=f"{name}_full",
            cfg=cfg,
            training_args=training_args,
            plot=False,
        )
        print(f"\n📊 Stage {name} full evaluation (pre-merge):")
        print(full_metrics)
        if wandb.run is not None:
            wandb.log(full_metrics)

        save(trainer, None, training_args)

        if not isinstance(trainer.model, PeftModel):
            raise ValueError("Trainer model is not a PeftModel; nothing to merge.")

        merged = trainer.model.merge_and_unload()
        merged_out = os.path.join(out_dir, "merged")
        os.makedirs(merged_out, exist_ok=True)
        merged.save_pretrained(merged_out, safe_serialization=True)
        print(f"✅ Merged model saved to {merged_out}")

        merged_model, _ = load_model(model_id=cfg["model_id"], dtype=torch.bfloat16, use_adapters=False, from_dir=out_dir, cfg=cfg)
        merged_metrics = evaluate_model(
            merged_model,
            processor=processor,
            dataset=test_ds,
            n=123,
            strict=True,
            tag=f"{name}_merged",
            cfg=cfg,
            training_args=training_args,
            plot=False,
        )
        print(f"\n📊 Stage {name} merged evaluation:")
        print(merged_metrics)
        if wandb.run is not None:
            wandb.log(merged_metrics)

        del merged_model
        del merged
        del trainer
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clean_cache(deep=False)

        if is_main:
            wandb.finish()

        prev_dir = out_dir

    if is_main:
        print("✅ All stages complete")
