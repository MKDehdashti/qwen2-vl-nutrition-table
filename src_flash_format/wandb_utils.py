import os
import wandb
from transformers import TrainerCallback
from huggingface_hub import HfFolder

def _load_secrets():
    proj_root = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
    path = os.path.join(proj_root, ".secrets")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

# Load secrets into env
_load_secrets()

# Hugging Face login
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    HfFolder.save_token(hf_token)

# W&B login
wandb_token = os.getenv("WANDB_API_KEY")
if wandb_token:
    wandb.login(key=wandb_token)

class WandBLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is not None:
            wandb.log(logs, step=state.global_step)

def init_wandb(run_id, training_args, lora_config=None, stage_name=None, exp_name="exp1"):
    def get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "nutrition-table-vl"),
        entity=os.getenv("WANDB_ENTITY"),
        group=f"{exp_name}_{run_id}",   # one experiment group
        job_type=stage_name,            # baseline, vision, lang_vision, etc.
        name=f"{stage_name}_{run_id}",  # run name
        config={
            "run_id": run_id,
            "stage": stage_name,
            "learning_rate": get(training_args, "learning_rate"),
            "batch_size": get(training_args, "per_device_train_batch_size"),
            "accum_steps": get(training_args, "gradient_accumulation_steps"),
            "epochs": get(training_args, "num_train_epochs"),
            "gradient_checkpointing": get(training_args, "gradient_checkpointing"),
            "lr_scheduler": get(training_args, "lr_scheduler_type"),
            "warmup_ratio": get(training_args, "warmup_ratio"),
            "lora_r": getattr(lora_config, "r", None),
            "lora_alpha": getattr(lora_config, "lora_alpha", None),
            "lora_dropout": getattr(lora_config, "lora_dropout", None),
        },
    )