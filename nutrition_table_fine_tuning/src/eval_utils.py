# eval_utils.py
"""Evaluate a checkpoint on the validation split.

Importable as `evaluate_model(...)` from train.py, and runnable directly:

    python src/eval_utils.py \
      --from_dir runs/<exp_name>/checkpoint-N \
      --config configs/<exp>.yaml \
      --tag strict_eval \
      --n 123

Metric definitions live in metrics.py.
"""

import argparse
import json
import os

import torch
import yaml

from metrics import detection_metrics
from viz_utils import run_inference_strict
from dataset.data_utils import parse_boxes_from_text


def evaluate_model(
    model,
    processor,
    n=123,
    dataset=None,
    strict=True,
    tag=None,
    cfg=None,
):
    """Mean IoU / precision@0.5 / recall@0.5 / F1@0.5 over the first `n` samples."""
    if not isinstance(cfg, dict) or "task_prompt" not in cfg:
        raise ValueError("cfg must include task_prompt")

    if dataset is None:
        raise ValueError("Dataset must be provided to avoid redundant loading.")

    model.eval()

    ds = dataset.select(range(min(n, len(dataset)))) if hasattr(dataset, "select") else dataset[:n]

    ious, precisions, recalls, f1s, results = [], [], [], [], []

    for idx in range(len(ds)):
        sample = ds[idx]
        msgs = sample["messages"]

        gt_text = ""
        if msgs and msgs[-1]["role"] == "assistant":
            content = msgs[-1]["content"]
            if isinstance(content, str):
                gt_text = content

        gt_boxes = parse_boxes_from_text(gt_text)

        image = None
        for c in msgs[1]["content"]:
            if c.get("type") == "image":
                image = c.get("image")

        metrics = run_inference_strict(
            model,
            processor=processor,
            image_or_url=image,
            cfg=cfg,
            dtype=getattr(model, "dtype", torch.float32),
        )
        if metrics is None:
            continue

        pred_boxes = metrics.get("objects", {}).get("bbox", [])
        iou_mean, precision, recall, f1 = detection_metrics(gt_boxes, pred_boxes)

        metrics["iou"] = iou_mean
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        ious.append(iou_mean)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        results.append(metrics)

    out = {}
    if ious:
        out["mean_iou"] = sum(ious) / len(ious)
    if precisions:
        out["precision@0.5"] = sum(precisions) / len(precisions)
    if recalls:
        out["recall@0.5"] = sum(recalls) / len(recalls)
    if f1s:
        out["f1@0.5"] = sum(f1s) / len(f1s)
    out["samples"] = len(ious)
    if tag:
        out["tag"] = tag

    return out


def _load_config(path):
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = path if os.path.isabs(path) else os.path.join(proj_root, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")
    with open(full_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from_dir", required=True,
                        help="Checkpoint or merged-model directory to evaluate")
    parser.add_argument("--config", required=True,
                        help="Experiment YAML, e.g. configs/exp13.yaml")
    parser.add_argument("--tag", default="eval", help="Label recorded in the output")
    parser.add_argument("--n", type=int, default=123, help="Number of val samples")
    parser.add_argument("--model_id", default=None,
                        help="Base model id; defaults to model_id from the config")
    parser.add_argument("--use_adapters", action="store_true",
                        help="Treat --from_dir as LoRA adapters over the base model")
    parser.add_argument("--quantized", action="store_true", help="Load in 4-bit")
    parser.add_argument("--out", default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    # Imported here so that `--help` works on a machine without a GPU.
    from dataset.data_utils import get_datasets
    from model_utils import load_model

    cfg = _load_config(args.config)
    model_id = args.model_id or cfg["model_id"]

    _, val = get_datasets(cfg)

    model, processor = load_model(
        model_id=model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantized=args.quantized,
        use_adapters=args.use_adapters,
        from_dir=args.from_dir,
        cfg=cfg,
    )

    results = evaluate_model(
        model, processor, n=args.n, dataset=val, strict=True, tag=args.tag, cfg=cfg,
    )
    results["from_dir"] = args.from_dir
    results["config"] = args.config

    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
