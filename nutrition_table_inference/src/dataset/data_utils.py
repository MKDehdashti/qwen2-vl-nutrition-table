# dataset/data_utils.py
"""Dataset loading and message formatting (inference side).

Seeding is exposed as `set_seed()` rather than executed at import time: importing
a data helper should not mutate global RNG or cuDNN state.

Prompt strings come from prompts.py so that inference cannot silently drift from
the strings the model was trained on -- which is exactly what happened before,
when this module hardcoded a singular-phrasing variant of the task prompt.
"""

import random
import re

import numpy as np
import torch
from datasets import load_dataset

from ..prompts import SYSTEM_MESSAGE, TASK_PROMPT

dataset_id = "openfoodfacts/nutrition-table-detection"

# Backwards-compatible alias; prompts.py is the single source of truth.
system_message = SYSTEM_MESSAGE


def set_seed(seed: int = 0, deterministic: bool = True):
    """Seed every RNG this project touches. Call once from the entrypoint."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def format_data(sample, cfg=None, label="nutrition-table"):
    """Convert one dataset row into chat messages with box tokens.

    Dataset boxes are `(y_min, x_min, y_max, x_max)` normalized to [0, 1]; the
    model is trained on `(x_min, y_min, x_max, y_max)` scaled to [0, 1000].
    """
    prompt = (cfg or {}).get("task_prompt", TASK_PROMPT)

    objects = sample.get("objects", {})
    bboxes = objects.get("bbox", [])
    annotations = []

    for bbox in bboxes:
        y_min, x_min, y_max, x_max = bbox
        x_min = int(x_min * 1000)
        y_min = int(y_min * 1000)
        x_max = int(x_max * 1000)
        y_max = int(y_max * 1000)
        annotations.append(
            f"<|object_ref_start|>{label}<|object_ref_end|>"
            f"<|box_start|>({x_min}, {y_min}),({x_max}, {y_max})<|box_end|>"
        )

    assistant_text = " ".join(annotations) if annotations else "No table found."
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]
    return {"messages": messages}


def parse_boxes_from_text(text: str):
    """Extract `(x0, y0),(x1, y1)` pairs from generated text."""
    matches = re.findall(r"\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)", text or "")
    return [[float(x0), float(y0), float(x1), float(y1)] for x0, y0, x1, y1 in matches]


def _lazy_transform(cfg):
    """Batch transform for `Dataset.with_transform` -- formats rows on access."""
    def _apply(batch):
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        rows = [{k: batch[k][i] for k in keys} for i in range(n)]
        return {"messages": [format_data(r, cfg)["messages"] for r in rows]}
    return _apply


def get_datasets(cfg=None, format_data_flag=True, lazy=False):
    """Return `(train, val)`.

    `lazy=True` formats rows on access via `with_transform`, which avoids holding
    every decoded image in memory at once. The default stays eager because that
    is the path the published runs used; the lazy path is covered by tests but
    has not been run through a full training job.
    """
    raw = load_dataset(dataset_id)

    if not format_data_flag:
        return raw["train"], raw["val"]

    if lazy:
        fn = _lazy_transform(cfg)
        return raw["train"].with_transform(fn), raw["val"].with_transform(fn)

    train = [format_data(ex, cfg) for ex in raw["train"]]
    val = [format_data(ex, cfg) for ex in raw["val"]]
    return train, val
