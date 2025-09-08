import os, shutil, subprocess, sys
from datasets import load_dataset
from accelerate import PartialState
import torch
import random, numpy as np
from qwen_vl_utils import process_vision_info

# Seed everything
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
np.random.seed(0); random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_id = "openfoodfacts/nutrition-table-detection"
HF_CACHE = os.environ.get("HF_DATASETS_CACHE", "/workspace/hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)

def build_messages(example, with_answer=True, prompt="Detect the bounding box of the nutrition table."):
    """
    Convert a dataset example into Qwen-style messages.
    - with_answer=True: include assistant ground-truth bbox
    - with_answer=False: only include user prompt
    - prompt: custom user prompt string
    """
    bboxes = example.get("objects", {}).get("bbox", [])
    if bboxes:
        x0, y0, x1, y1 = bboxes[0]
        answer_text = f"({int(x0*1000)},{int(y0*1000)}),({int(x1*1000)},{int(y1*1000)})"
    else:
        answer_text = "No table found."

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})

    return msgs



state = PartialState()
is_main = state.is_main_process
if is_main:
    _ = load_dataset(dataset_id, split="train", cache_dir=HF_CACHE)
    _ = load_dataset(dataset_id, split="val",   cache_dir=HF_CACHE)
state.wait_for_everyone()
train_ds = load_dataset(dataset_id, split="train", cache_dir=HF_CACHE, download_mode="reuse_cache_if_exists")
test_ds  = load_dataset(dataset_id, split="val",   cache_dir=HF_CACHE, download_mode="reuse_cache_if_exists")

def purge_root_caches():
    for p in ["~/.cache/huggingface", "~/.cache/pip", "/root/.cache"]:
        shutil.rmtree(os.path.expanduser(p), ignore_errors=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
    except Exception:
        pass

def set_workspace_caches(proj_root="/workspace/projects/nutrition-table"):
    """
    Configure cache + run directories.
    - Hugging Face & Torch cache -> <proj_root>/models
    - W&B logs, tmp, etc.        -> <proj_root>/runs/cache
    """

    model_cache = os.path.join(proj_root, "models")      # base models cached here
    run_cache   = os.path.join(proj_root, "runs", "cache")  # misc cache for runs

    os.makedirs(os.path.join(model_cache, "hub"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "torch"), exist_ok=True)

    os.makedirs(os.path.join(run_cache, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(run_cache, "wandb"), exist_ok=True)

    os.environ.update({
        # Model & dataset caches
        "HF_HOME": model_cache,
        "HF_HUB_CACHE": os.path.join(model_cache, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(model_cache, "transformers"),
        "HF_DATASETS_CACHE": os.path.join(model_cache, "datasets"),
        "TORCH_HOME": os.path.join(model_cache, "torch"),

        # General cache/tmp
        "XDG_CACHE_HOME": run_cache,
        "TMPDIR": os.path.join(run_cache, "tmp"),
        "TMP": os.path.join(run_cache, "tmp"),
        "TEMP": os.path.join(run_cache, "tmp"),

        # W&B logs
        "WANDB_DIR": os.path.join(run_cache, "wandb"),
    })


def collate_fn_consistent(examples, processor):
    full_msgs = [build_messages(e, with_answer=True) for e in examples]
    pref_msgs = [build_messages(e, with_answer=False) for e in examples]


    full_txts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in full_msgs]
    pref_txts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in pref_msgs]

    image_inputs = [process_vision_info(m)[0] for m in full_msgs]

    batch = processor(text=full_txts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    tok = processor.tokenizer
    for i, pref in enumerate(pref_txts):
        pref_ids = tok(pref, add_special_tokens=False).input_ids
        cut = len(pref_ids)
        labels[i, :cut] = -100

    special = {tok.pad_token_id,
               tok.convert_tokens_to_ids("<|vision_start|>"),
               tok.convert_tokens_to_ids("<|vision_end|>"),
               tok.convert_tokens_to_ids("<|image_pad|>")}
    for tid in special:
        if tid is not None:
            labels[labels == tid] = -100

    batch["labels"] = labels
    return batch

