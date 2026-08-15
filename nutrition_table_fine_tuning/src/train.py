# train.py
import os, yaml, torch
from datetime import datetime
from accelerate import PartialState
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import EarlyStoppingCallback
import wandb

from dataset.data_utils import get_datasets, format_data, set_seed
from dataset.collators import collate_fn
from eval_utils import evaluate_model
from wandb_utils import init_wandb, WandBLossCallback
from cleanup import clean_cache
from model_utils import (
    load_processor_fixed,
    load_base,
    merge_adapters_to_dir,
    save_adapters,
    get_matching_modules,
)

os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_DATASETS_DISABLE_CACHING"] = "0"

# Seeding used to happen as a side effect of importing data_utils; it is now
# explicit so that importing a helper does not mutate global RNG state.
set_seed(0)

state = PartialState()
is_main = state.is_main_process

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available. This training script is GPU-only.")


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
        merged[k] = deep_merge(merged[k], v) if k in merged else v
    return merged


def parse_optional_int(v):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"none", "null", ""}:
        return None
    return int(v)


def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


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

    if not isinstance(cfg, dict) or "model_id" not in cfg or "task_prompt" not in cfg:
        raise SystemExit("Config must include model_id and task_prompt.")

    has_cuda = torch.cuda.is_available()
    cfg["use_flash"] = bool(cfg.get("use_flash", False) and has_cuda)

    quantized = bool(cfg.get("quantized", False))

    debug_eval = bool(cfg.get("debug_eval", False))
    debug_eval_limit = int(cfg.get("debug_eval_limit", 3))
    eval_n = debug_eval_limit if debug_eval else 123

    if "debug" in cfg:
        dbg = cfg["debug"]
        raw = load_dataset("openfoodfacts/nutrition-table-detection")
        train_raw = raw["train"].select(range(dbg.get("dataset_subset", 20)))
        val_raw = raw["val"].select(range(dbg.get("dataset_subset", 20)))
        if cfg.get("format_data", False):
            train_ds = [format_data(ex, cfg) for ex in train_raw]
            test_ds = [format_data(ex, cfg) for ex in val_raw]
        else:
            train_ds, test_ds = train_raw, val_raw
        cfg.setdefault("wandb", {})
        cfg["wandb"]["name"] = cfg["wandb"].get("name", cfg["experiment"]) + dbg.get("wandb_suffix", "-debug")
        cfg["wandb"]["tags"] = cfg["wandb"].get("tags", []) + dbg.get("wandb_tags", ["debug"])
    else:
        train_ds, test_ds = get_datasets(cfg, format_data_flag=cfg.get("format_data", True))

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

    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = has_cuda and not use_bf16
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    prev_merged_dir = None

    collator_cfg = cfg.get("collator", {})
    numeric_only = bool(collator_cfg.get("numeric_only", False))

    for stage_idx, stage in enumerate(cfg["stages"]):
        name = stage["name"]
        regex_targets = stage["regex_targets"]

        out_dir = os.path.join(run_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        merged_dir = os.path.join(out_dir, "merged")

        stage_base_src = cfg["model_id"] if stage_idx == 0 or prev_merged_dir is None else prev_merged_dir

        if is_main:
            if stage_idx == 0 or prev_merged_dir is None:
                print(f"\n=== Stage: {name} (starting from base model: {cfg['model_id']}) ===")
            else:
                print(f"\n=== Stage: {name} (starting from previous merged stage: {prev_merged_dir}) ===")
            print(f"    base_src   : {stage_base_src}")
            print(f"    adapters   : {out_dir}")
            print(f"    merged_out : {merged_dir}")

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
            tf32=True if has_cuda else False,
            report_to="wandb" if is_main else "none",
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
            warmup_steps=cfg.get("warmup_steps", 100),
            lr_scheduler_type=stage.get("lr_scheduler_type", cfg.get("lr_scheduler_type", "linear")),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_accumulation_steps=1,
            include_inputs_for_metrics=False,
            prediction_loss_only=False,
        )
        training_args.remove_unused_columns = False

        clean_base = load_base(cfg["model_id"], torch_dtype=torch_dtype, quantized=quantized, cfg=cfg)
        target_modules = get_matching_modules(clean_base, regex_targets)
        del clean_base
        torch.cuda.empty_cache()

        peft_cfg = LoraConfig(
            r=stage["lora_r"],
            lora_alpha=stage["lora_alpha"],
            lora_dropout=stage["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        base_model = load_base(stage_base_src, torch_dtype=torch_dtype, quantized=quantized, cfg=cfg)
        base_model.enable_input_require_grads()
        base_model.config.use_cache = False

        if is_main:
            init_wandb(run_id, training_args, peft_cfg, stage_name=name, exp_name=exp_name)

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
        if is_main:
            callbacks.insert(0, WandBLossCallback())

        trainer = SFTTrainer(
            model=base_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=lambda ex: collate_fn(ex, processor, cfg=cfg, numeric_only=numeric_only),
            peft_config=peft_cfg,
            callbacks=callbacks,
            compute_metrics=None,
            processing_class=processor,
        )

        trainer.train()
        state.wait_for_everyone()

        if is_main:
            pre_merge_metrics = evaluate_model(
                unwrap_model(trainer.model),
                processor=processor,
                dataset=test_ds,
                n=eval_n,
                strict=True,
                tag=f"{name}_pre_merge",
                cfg=cfg,
            )
            if wandb.run is not None:
                wandb.log({f"pre_merge/{k}": v for k, v in pre_merge_metrics.items()}, step=trainer.state.global_step)

            save_adapters(trainer, training_args)

            merged_inmem = merge_adapters_to_dir(
                base_src=stage_base_src,
                adapter_dir=out_dir,
                merged_dir=merged_dir,
                torch_dtype=torch_dtype,
                quantized=quantized,
                cfg=cfg,
            )

            post_metrics_inmem = evaluate_model(
                merged_inmem,
                processor=processor,
                dataset=test_ds,
                n=eval_n,
                strict=True,
                tag=f"{name}_merged_inmem",
                cfg=cfg,
            )

            del merged_inmem
            torch.cuda.empty_cache()

            merged_reload = load_base(merged_dir, torch_dtype=torch_dtype, quantized=quantized, cfg=cfg)
            post_metrics_reload = evaluate_model(
                merged_reload,
                processor=processor,
                dataset=test_ds,
                n=eval_n,
                strict=True,
                tag=f"{name}_merged_reload",
                cfg=cfg,
            )

            if wandb.run is not None:
                wandb.log({f"post_merge/inmem/{k}": v for k, v in post_metrics_inmem.items()}, step=trainer.state.global_step + 1)
                wandb.log({f"post_merge/reload/{k}": v for k, v in post_metrics_reload.items()}, step=trainer.state.global_step + 2)

            del merged_reload
            torch.cuda.empty_cache()
            wandb.finish()

        state.wait_for_everyone()

        del trainer
        del base_model
        torch.cuda.empty_cache()
        clean_cache(deep=False)

        prev_merged_dir = merged_dir

    if is_main:
        print("All stages complete")
