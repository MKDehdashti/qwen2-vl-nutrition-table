# inference_eval.py
import os
import time
import json
import argparse
import subprocess
import base64
from io import BytesIO

import numpy as np
import torch
import requests
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

from .dataset.data_utils import get_datasets, parse_boxes_from_text
from .metrics import detection_metrics
from .prompts import PLACEHOLDER_SYSTEM_TEXT, TASK_PROMPT
from .viz_utils_infer import draw_box_0to1000


DEFAULT_PROMPT = TASK_PROMPT  # kept as an alias; canonical string lives in prompts.py


def get_gpu_memory_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        vals = [int(x) for x in out.decode().strip().splitlines()]
        return max(vals) / 1024.0
    except Exception:
        return None


def image_to_data_url(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def extract_openai_content(data):
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return content


def round2(x):
    if x is None:
        return None
    return float(np.round(float(x), 2))


def round_stats(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating, int, np.integer)):
            out[k] = round2(v)
        else:
            out[k] = v
    return out


def get_gt_and_image(ex):
    msgs = ex["messages"]
    gt_text = ""
    if msgs and msgs[-1]["role"] == "assistant":
        content = msgs[-1]["content"]
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
            gt_text = content[0]["text"]
        elif isinstance(content, str):
            gt_text = content

    image = None
    for c in msgs[1]["content"]:
        if c.get("type") == "image":
            image = c["image"]
            break

    gt_boxes = parse_boxes_from_text(gt_text)
    return gt_boxes, image


def metrics_one(gt_boxes, pred_boxes):
    """Per-image (mean_iou, precision, recall, f1). See src/metrics.py."""
    return detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5)


def hf_infer_batch(hf_model, processor, images, prompt, system_text, max_new_tokens):
    t0_total = time.perf_counter()

    texts = []
    for img in images:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    device = next(hf_model.parameters()).device if torch.cuda.is_available() else "cpu"
    for k, v in inputs.items():
        inputs[k] = v.to(device=device, dtype=hf_model.dtype) if torch.is_floating_point(v) else v.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_model = time.perf_counter()
    with torch.no_grad():
        out_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_ms = (time.perf_counter() - t0_model) * 1000.0

    attn = inputs.get("attention_mask", None)
    if attn is None:
        in_lens = [inputs["input_ids"].shape[1]] * out_ids.shape[0]
    else:
        in_lens = attn.sum(dim=1).tolist()

    trimmed = []
    for i in range(out_ids.shape[0]):
        trimmed.append(out_ids[i, int(in_lens[i]):])

    out_texts = processor.batch_decode(trimmed, skip_special_tokens=False)
    total_ms = (time.perf_counter() - t0_total) * 1000.0
    return out_texts, model_ms, total_ms


def vllm_infer_batch(server_url, api_key, model, images, prompt, system_text, max_new_tokens):
    """Send `images` to the vLLM OpenAI endpoint ONE AT A TIME, sequentially.

    Despite the name, this does not batch: there is no concurrency here, so
    `--batch_size` does not change how requests are issued. It only changes the
    divisor used for per-sample latency in eval_dataset(). Do not use this path
    to compare vLLM batch sizes -- the differences are run-to-run noise.
    Use vllm_throughput.py (asyncio.Semaphore) for real concurrency benchmarks.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = server_url.rstrip("/") + "/chat/completions"

    t0_total = time.perf_counter()
    request_only_ms_sum = 0.0
    out_texts = []

    for img in images:
        data_url = image_to_data_url(img)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        t0_req = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        request_only_ms_sum += (time.perf_counter() - t0_req) * 1000.0

        out_texts.append(extract_openai_content(resp.json()))

    total_ms = (time.perf_counter() - t0_total) * 1000.0
    return out_texts, request_only_ms_sum, total_ms


def eval_dataset(
    backend,
    model,
    split="val",
    num_samples=123,
    batch_size=4,
    max_new_tokens=256,
    run_root="outputs/eval",
    run_name="run",
    num_visuals=8,
    prompt=DEFAULT_PROMPT,
    system_text=PLACEHOLDER_SYSTEM_TEXT,
    use_fast=True,
    server_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
):
    train, val = get_datasets(format_data_flag=True)
    ds = val if split == "val" else train
    n = min(num_samples, len(ds))

    run_dir = os.path.join(run_root, run_name)
    viz_dir = os.path.join(run_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    hf_model = None
    processor = None
    if backend == "hf":
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        processor = Qwen2VLProcessor.from_pretrained(model, use_fast=use_fast)

    ious, precisions, recalls, f1s = [], [], [], []
    batch_total_ms = []
    batch_model_ms = []

    kept = 0
    visuals_saved = 0
    batch_gt, batch_imgs = [], []

    def flush_batch():
        nonlocal kept, visuals_saved
        if not batch_imgs:
            return

        B = len(batch_imgs)

        if backend == "hf":
            out_texts, model_ms, total_ms = hf_infer_batch(
                hf_model, processor, batch_imgs, prompt, system_text, max_new_tokens
            )
            batch_model_ms.append(model_ms)
            batch_total_ms.append(total_ms)
        else:
            out_texts, request_only_ms, total_ms = vllm_infer_batch(
                server_url, api_key, model, batch_imgs, prompt, system_text, max_new_tokens
            )
            batch_model_ms.append(request_only_ms)
            batch_total_ms.append(total_ms)

        for i in range(B):
            pred_boxes = parse_boxes_from_text(out_texts[i])
            gt_boxes = batch_gt[i]
            iou_mean, precision, recall, f1 = metrics_one(gt_boxes, pred_boxes)

            ious.append(iou_mean)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            if visuals_saved < num_visuals:
                sample_pred = {"image": batch_imgs[i], "objects": {"bbox": pred_boxes}}
                save_path = os.path.join(viz_dir, f"{backend}_{split}_{kept}.png")
                draw_box_0to1000(sample_pred, save_path=save_path)
                visuals_saved += 1

            kept += 1

        batch_gt.clear()
        batch_imgs.clear()

    for ex in ds[:n]:
        gt_boxes, img = get_gt_and_image(ex)
        if img is None:
            continue
        batch_gt.append(gt_boxes)
        batch_imgs.append(img)
        if len(batch_imgs) == batch_size:
            flush_batch()

    flush_batch()

    if not ious:
        raise RuntimeError("No samples produced metrics. Check dataset or parsing.")

    mem_gb = get_gpu_memory_gb()
    eff_bs = batch_size if batch_size > 0 else 1

    per_sample_total = [x / eff_bs for x in batch_total_ms] if batch_total_ms else []
    per_sample_model = [x / eff_bs for x in batch_model_ms] if batch_model_ms else []

    stats = {
        "samples": int(len(ious)),
        "batch_size": int(batch_size),

        "mean_iou": float(np.mean(ious)),
        "precision@0.5": float(np.mean(precisions)),
        "recall@0.5": float(np.mean(recalls)),
        "f1@0.5": float(np.mean(f1s)),

        "latency_ms_per_sample_mean_end_to_end": float(np.mean(per_sample_total)) if per_sample_total else None,
        "latency_ms_per_sample_p95_end_to_end": float(np.percentile(per_sample_total, 95)) if per_sample_total else None,

        "latency_ms_per_sample_mean_without_preprocess": float(np.mean(per_sample_model)) if per_sample_model else None,
        "latency_ms_per_sample_p95_without_preprocess": float(np.percentile(per_sample_model, 95)) if per_sample_model else None,

        "gpu_memory_gb": float(mem_gb) if mem_gb is not None else None,
    }

    if backend == "vllm":
        stats["note_without_preprocess"] = "For vLLM this is request-only time (client excludes base64 encode), not pure server model-only."
    else:
        stats["note_without_preprocess"] = "For HF this is model.generate() time only (excludes processor + decode)."

    stats = round_stats(stats)

    json_path = os.path.join(run_dir, f"{run_name}.json")
    data = {
        "run_name": run_name,
        "backend": backend,
        "model": model,
        "split": split,
        "prompt": prompt,
        "system_text": system_text,
        "stats": stats,
        "artifacts": {"run_dir": run_dir, "viz_dir": viz_dir, "json_path": json_path},
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--run_root", type=str, default="outputs/eval")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--num_visuals", type=int, default=8)

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--system_text", type=str, default=PLACEHOLDER_SYSTEM_TEXT)
    parser.add_argument("--use_fast", action="store_true")

    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    args = parser.parse_args()

    data = eval_dataset(
        backend=args.backend,
        model=args.model,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        run_root=args.run_root,
        run_name=args.run_name,
        num_visuals=args.num_visuals,
        prompt=args.prompt,
        system_text=args.system_text,
        use_fast=args.use_fast,
        server_url=args.server_url,
        api_key=args.api_key,
    )

    print("saved:", data["artifacts"]["json_path"])
    for k, v in data["stats"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
