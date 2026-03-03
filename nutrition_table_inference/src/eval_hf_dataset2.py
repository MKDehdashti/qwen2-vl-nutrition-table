# eval_hf_dataset.py
import os
import time
import json
import argparse
import subprocess
import numpy as np
import torch
from torchvision.ops import box_iou

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from .dataset.data_utils import get_datasets, parse_boxes_from_text
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


def eval_hf_dataset(
    model,
    split="val",
    num_samples=123,
    max_new_tokens=256,
    out_dir="outputs/hf_eval",
    num_visuals=5,
    prompt="Detect the bounding box of the nutrition table.",
    use_fast=True,
):
    train, val = get_datasets(format_data_flag=True)
    ds = val if split == "val" else train
    n = min(num_samples, len(ds))

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    hf_model.eval()

    processor = Qwen2VLProcessor.from_pretrained(
        model,
        use_fast=use_fast,
    )

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

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=device, dtype=hf_model.dtype)
            else:
                inputs[k] = v.to(device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        in_len = inputs["input_ids"].shape[1]
        out_text = processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=False)[0]

        pred_boxes = parse_boxes_from_text(out_text)

        iou_mean, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        if gt_boxes and pred_boxes:
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            pr_t = torch.tensor(pred_boxes, dtype=torch.float32)
            if gt_t.ndim == 1:
                gt_t = gt_t.unsqueeze(0)
            if pr_t.ndim == 1:
                pr_t = pr_t.unsqueeze(0)

            ious_mat = box_iou(gt_t, pr_t)
            best_gt = ious_mat.max(dim=1)[0]
            best_pr = ious_mat.max(dim=0)[0]
            iou_mean = (best_gt.mean() + best_pr.mean()).item() / 2

            matched = (ious_mat > 0.5).sum().item()
            precision = matched / len(pr_t)
            recall = matched / len(gt_t)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)

        ious.append(iou_mean)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if idx < num_visuals:
            sample_pred = {"image": image, "objects": {"bbox": pred_boxes}}
            save_path = os.path.join(out_dir, f"hf_{split}_{idx}.png")
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=123)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="outputs/hf_eval")
    parser.add_argument("--num_visuals", type=int, default=10)
    parser.add_argument("--name", type=str, default="config")
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stats = eval_hf_dataset(
        model=args.model,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
        num_visuals=args.num_visuals,
        use_fast=args.use_fast,
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
