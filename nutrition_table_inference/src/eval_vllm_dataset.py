# eval_vllm_dataset.py
import os
import time
import json
import argparse
import subprocess
import numpy as np

from .dataset.data_utils import get_datasets, parse_boxes_from_text
from .inference_vllm import call_vllm, image_to_data_url
from .metrics import detection_metrics
from .prompts import SYSTEM_MESSAGE, TASK_PROMPT
from .viz_utils_infer import draw_box_0to1000


def get_gpu_memory_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        vals = [int(x) for x in out.decode().strip().splitlines()]
        return max(vals) / 1024.0
    except Exception:
        return None


def eval_vllm_dataset(
    server_url,
    api_key,
    model,
    split="val",
    num_samples=50,
    max_new_tokens=256,
    out_dir="outputs/vllm_eval",
    num_visuals=5,
):
    train, val = get_datasets(format_data_flag=True)
    ds = val if split == "val" else train
    n = min(num_samples, len(ds))

    os.makedirs(out_dir, exist_ok=True)

    ious, precisions, recalls, f1s, latencies = [], [], [], [], []

    for idx, ex in enumerate(ds[:n]):
        msgs = ex["messages"]

        gt_text = ""
        if msgs and msgs[-1]["role"] == "assistant":
            content = msgs[-1]["content"]
            if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                gt_text = content[0]["text"]
            elif isinstance(content, str):
                gt_text = content

        gt_boxes = parse_boxes_from_text(gt_text)

        image = None
        for c in msgs[1]["content"]:
            if c.get("type") == "image":
                image = c["image"]
        if image is None:
            continue

        data_url = image_to_data_url(image)

        t0 = time.perf_counter()
        text = call_vllm(
            server_url=server_url,
            api_key=api_key,
            model=model,
            prompt=TASK_PROMPT,
            img_data_url=data_url,
            max_new_tokens=max_new_tokens,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        pred_boxes = parse_boxes_from_text(text)

        iou_mean, precision, recall, f1 = detection_metrics(gt_boxes, pred_boxes)

        ious.append(iou_mean)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if idx < num_visuals:
            sample_pred = {"image": image, "objects": {"bbox": pred_boxes}}
            save_path = os.path.join(out_dir, f"vllm_{split}_{idx}.png")
            draw_box_0to1000(sample_pred, save_path=save_path)
            print("saved viz:", save_path)

    if not ious:
        raise RuntimeError("No samples produced metrics. Check dataset or parsing.")

    mem_gb = get_gpu_memory_gb()

    stats = {
        "samples": len(ious),
        "mean_iou": float(np.mean(ious)),
        "precision@0.5": float(np.mean(precisions)),
        "recall@0.5": float(np.mean(recalls)),
        "f1@0.5": float(np.mean(f1s)),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "gpu_memory_gb": float(mem_gb) if mem_gb is not None else None,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="outputs/vllm_eval")
    parser.add_argument("--num_visuals", type=int, default=5)
    parser.add_argument("--name", type=str, default="config")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stats = eval_vllm_dataset(
        server_url=args.server_url,
        api_key=args.api_key,
        model=args.model,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
        num_visuals=args.num_visuals,
    )

    print("Evaluation name:", args.name)
    for k, v in stats.items():
        print(f"{k}: {v}")

    out_path = os.path.join(args.out_dir, f"metrics_{args.name}.json")
    data = {"name": args.name, "model": args.model, "split": args.split, "stats": stats}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved metrics:", out_path)


if __name__ == "__main__":
    main()
