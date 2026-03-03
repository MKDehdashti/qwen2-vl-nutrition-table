# vllm_throughput.py
import os
import json
import time
import base64
import argparse
import asyncio
from io import BytesIO

import numpy as np
import aiohttp

from .dataset.data_utils import get_datasets


PROMPT = "Detect the bounding boxes of all nutrition tables in the image."


def image_to_data_url(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def round2(x):
    if x is None:
        return None
    return float(np.round(float(x), 2))


def round_stats(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating, int, np.integer)):
            out[k] = round2(v)
        elif isinstance(v, dict):
            out[k] = round_stats(v)
        elif isinstance(v, list):
            out[k] = [round_stats(i) if isinstance(i, dict) else (round2(i) if isinstance(i, (float, np.floating, int, np.integer)) else i) for i in v]
        else:
            out[k] = v
    return out


async def one_call(session, url, headers, model, system_text, img_data_url, max_new_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_data_url}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        txt = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {txt[:500]}")
        data = json.loads(txt)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    usage = data.get("usage", {}) or {}
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    return dt_ms, completion_tokens, total_tokens


async def run_benchmark(args, images_data_urls):
    url = args.server_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}

    sem = asyncio.Semaphore(args.concurrency)

    lat_ms = []
    completion_tokens = 0
    total_tokens = 0

    async with aiohttp.ClientSession() as session:

        async def run_one(img_data_url):
            async with sem:
                return await one_call(
                    session,
                    url,
                    headers,
                    args.model,
                    args.system_text,
                    img_data_url,
                    args.max_new_tokens,
                )

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(run_one(u)) for u in images_data_urls]

        for coro in asyncio.as_completed(tasks):
            dt_ms, c, t = await coro
            lat_ms.append(dt_ms)
            completion_tokens += c
            total_tokens += t

        wall_s = time.perf_counter() - t0

    return lat_ms, completion_tokens, total_tokens, wall_s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--model", type=str, required=True)

    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=123)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=16)

    p.add_argument("--run_root", type=str, default="outputs/throughput")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--system_text", type=str, default="System message")
    args = p.parse_args()

    train, val = get_datasets(format_data_flag=True)
    ds = val if args.split == "val" else train
    n = min(args.num_samples, len(ds))

    images = []
    for ex in ds[:n]:
        msgs = ex["messages"]
        img = None
        for c in msgs[1]["content"]:
            if c.get("type") == "image":
                img = c["image"]
                break
        if img is not None:
            images.append(img)

    images = images[:n]
    data_urls = [image_to_data_url(img) for img in images]

    lat_ms, comp_tokens, total_tokens, wall_s = asyncio.run(run_benchmark(args, data_urls))

    lat_ms = np.array(lat_ms, dtype=np.float64)
    samples = int(len(lat_ms))
    req_per_s = samples / wall_s if wall_s > 0 else None
    comp_tok_per_s = comp_tokens / wall_s if wall_s > 0 else None
    total_tok_per_s = total_tokens / wall_s if wall_s > 0 else None

    stats = {
        "backend": "vllm_openai_api",
        "samples": samples,
        "concurrency": int(args.concurrency),
        "max_new_tokens": int(args.max_new_tokens),
        "latency_ms_p50_end_to_end": float(np.percentile(lat_ms, 50)) if samples else None,
        "latency_ms_p95_end_to_end": float(np.percentile(lat_ms, 95)) if samples else None,
        "latency_ms_mean_end_to_end": float(np.mean(lat_ms)) if samples else None,
        "throughput_requests_per_s": float(req_per_s) if req_per_s is not None else None,
        "throughput_completion_tokens_per_s": float(comp_tok_per_s) if comp_tok_per_s is not None else None,
        "throughput_total_tokens_per_s": float(total_tok_per_s) if total_tok_per_s is not None else None,
        "completion_tokens_total": int(comp_tokens),
        "total_tokens_total": int(total_tokens),
        "wall_time_s": float(wall_s),
    }

    run_dir = os.path.join(args.run_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"{args.run_name}.json")

    payload = {
        "run_name": args.run_name,
        "model": args.model,
        "server_url": args.server_url,
        "split": args.split,
        "prompt": PROMPT,
        "system_text": args.system_text,
        "stats": round_stats(stats),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("saved:", out_path)
    for k, v in payload["stats"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
