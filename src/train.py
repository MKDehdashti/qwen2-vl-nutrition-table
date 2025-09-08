import os, yaml, torch
from datetime import datetime
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from data_utils import train_ds, test_ds, collate_fn_consistent
from model_utils import save, load_model, get_matching_modules
from eval_utils import run_and_log_eval, test_one
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
    # src/train.py -> go 1 level up = src/, 2 levels up = nutrition-table/
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
    args, unknown = parser.parse_known_args()
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


    # Use full timestamp (date + hour + minute + second)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["experiment"]

    # Run-specific folder
    run_dir = os.path.join(runs_root, f"{exp_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Redirect W&B logs here
    os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    if is_main:
        import wandb
        wandb.init(project="nutrition-table-vl", group=f"{exp_name}_{run_id}", job_type="meta",
                   name=f"meta_{exp_name}_{run_id}", config=cfg)
        wandb.finish()

    if is_main:
        init_wandb(run_id, {}, None, stage_name="baseline", exp_name=exp_name)
        print("\n=== Baseline Evaluation ===")
        baseline_metrics = run_and_log_eval("baseline", tag="baseline",
                                            quantized=True, use_adapters=False, from_dir=None)
        m, p = load_model(dtype=torch.bfloat16, quantized=True, use_adapters=False)
        test_one(m, p, idx=20, tag="baseline", split="val", out_dir=run_dir)
        del m, p; torch.cuda.empty_cache()
        import wandb; wandb.finish()

    # === Stage loop ===
    for stage_cfg in cfg["stages"]:
        name = stage_cfg["name"]
        regex_targets = stage_cfg["regex_targets"]

        # Stage-specific folder
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

        peft_cfg = LoraConfig(
            r=stage_cfg["lora_r"],
            lora_alpha=stage_cfg["lora_alpha"],
            lora_dropout=stage_cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_matching_modules(
                load_model(dtype=torch.bfloat16, quantized=True, use_adapters=False)[0],
                regex_targets
            )
        )

        if is_main:
            init_wandb(run_id, training_args, peft_cfg, stage_name=name, exp_name=exp_name)

        print(f"\n=== Stage: {name} ===")
        base_model, proc = load_model(dtype=torch.bfloat16, quantized=True, use_adapters=False, from_dir=out_dir)
        base_model.enable_input_require_grads()
        base_model.config.use_cache = False

        trainer = SFTTrainer(
            model=base_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=lambda ex: collate_fn_consistent(ex, proc),
            peft_config=peft_cfg,
            callbacks=[WandBLossCallback()],
        )
        trainer.train()
        save(trainer, proc, training_args)

        eval_metrics = run_and_log_eval(name, tag=f"post_{name}",
                                        quantized=True, use_adapters=True,
                                        from_dir=out_dir, training_args=training_args)
        m, p = load_model(dtype=torch.bfloat16, quantized=True, use_adapters=True, from_dir=out_dir)
        test_one(m, p, idx=20, tag=f"post_{name}", split="val",
                 out_dir=out_dir, training_args=training_args)
        del m, p; torch.cuda.empty_cache()

        clean_cache(deep=False)

        if is_main:
            import wandb; wandb.finish()

    if is_main:
        print("✅ All stages complete")
