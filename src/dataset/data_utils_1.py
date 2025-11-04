import os, torch, random, numpy as np, re
from datasets import load_dataset

# reproducibility
torch.manual_seed(0); torch.cuda.manual_seed_all(0)
np.random.seed(0); random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset_id = "openfoodfacts/nutrition-table-detection"

# -------------------------
#   SYSTEM PROMPT
# -------------------------
system_message = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

# -------------------------
#   FORMAT DATA
# -------------------------
def format_data(sample, label="nutrition-table"):
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
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": "Detect the bounding box of the nutrition table."},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]

    return {"messages": messages}

# -------------------------
#   PARSE BOXES
# -------------------------
def parse_boxes_from_text(text: str):
    matches = re.findall(r"\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)", text)
    return [[float(x0), float(y0), float(x1), float(y1)] for x0, y0, x1, y1 in matches]

# -------------------------
#   LOAD + FORMAT DATASETS
# -------------------------
def get_datasets(format_data_flag=True):
    # fresh download each run
    raw = load_dataset(dataset_id)

    if format_data_flag:
        # Format data and create Dataset objects
        train_data = [format_data(ex) for ex in raw["train"]]
        val_data = [format_data(ex) for ex in raw["val"]]
        
        # Convert to Dataset objects
        from datasets import Dataset
        train = Dataset.from_list(train_data)
        val = Dataset.from_list(val_data)
    else:
        train, val = raw["train"], raw["val"]

    return train, val
