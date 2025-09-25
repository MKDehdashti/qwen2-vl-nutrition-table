import os, shutil, subprocess, sys
from datasets import load_dataset
import torch, random, numpy as np
from qwen_vl_utils import process_vision_info

torch.manual_seed(0); torch.cuda.manual_seed_all(0)
np.random.seed(0); random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_id = "openfoodfacts/nutrition-table-detection"
HF_CACHE = os.environ.get("HF_DATASETS_CACHE", "/workspace/hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)

def build_messages(example, with_answer=True, prompt="Detect the bounding box of the nutrition table."):
    bboxes = example.get("objects", {}).get("bbox", [])
    if bboxes:
        x0, y0, x1, y1 = bboxes[0]
        answer_text = f"({int(x0*1000)},{int(y0*1000)}),({int(x1*1000)},{int(y1*1000)})"
    else:
        answer_text = "No table found."
    msgs = [{"role": "user", "content": [{"type": "image", "image": example["image"]}, {"type": "text", "text": prompt}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})
    return msgs

def load_splits():
    train = load_dataset(dataset_id, split="train", cache_dir=HF_CACHE, download_mode="reuse_cache_if_exists")
    val = load_dataset(dataset_id, split="val", cache_dir=HF_CACHE, download_mode="reuse_cache_if_exists")
    return train, val

train_ds, test_ds = load_splits()

def purge_root_caches():
    for p in ["~/.cache/huggingface", "~/.cache/pip", "/root/.cache"]:
        shutil.rmtree(os.path.expanduser(p), ignore_errors=True)
    try:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
    except Exception:
        pass

def set_workspace_caches(proj_root="/workspace/projects/nutrition-table"):
    model_cache = os.path.join(proj_root, "models")
    run_cache = os.path.join(proj_root, "runs", "cache")
    os.makedirs(os.path.join(model_cache, "hub"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(model_cache, "torch"), exist_ok=True)
    os.makedirs(os.path.join(run_cache, "tmp"), exist_ok=True)
    os.makedirs(os.path.join(run_cache, "wandb"), exist_ok=True)
    os.environ.update({
        "HF_HOME": model_cache,
        "HF_HUB_CACHE": os.path.join(model_cache, "hub"),
        "TRANSFORMERS_CACHE": os.path.join(model_cache, "transformers"),
        "HF_DATASETS_CACHE": os.path.join(model_cache, "datasets"),
        "TORCH_HOME": os.path.join(model_cache, "torch"),
        "XDG_CACHE_HOME": run_cache,
        "TMPDIR": os.path.join(run_cache, "tmp"),
        "TMP": os.path.join(run_cache, "tmp"),
        "TEMP": os.path.join(run_cache, "tmp"),
        "WANDB_DIR": os.path.join(run_cache, "wandb"),
    })
