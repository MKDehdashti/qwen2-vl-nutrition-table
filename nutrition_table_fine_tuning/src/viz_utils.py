# viz_utils.py
import os, torch, argparse
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from qwen_vl_utils import process_vision_info
from dataset.data_utils import parse_boxes_from_text, system_message as TRAIN_SYSTEM_MESSAGE
from model_utils import load_model

def _load_image(image_or_url):
    if isinstance(image_or_url, Image.Image):
        return image_or_url.convert("RGB")
    if isinstance(image_or_url, str):
        if image_or_url.startswith("http://") or image_or_url.startswith("https://"):
            import requests
            return Image.open(BytesIO(requests.get(image_or_url, timeout=30).content)).convert("RGB")
        return Image.open(image_or_url).convert("RGB")
    return image_or_url

def run_inference_strict(model, processor, image_or_url, cfg=None, max_new_tokens=512, dtype=None):
    cfg = cfg or {}
    inf_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}

    if "task_prompt" not in cfg:
        raise ValueError("cfg must include task_prompt")

    if dtype is None:
        dtype = getattr(model, "dtype", torch.float32)

    device = next(model.parameters()).device
    image = _load_image(image_or_url)

    sys_msg = inf_cfg.get("system_message", None) or TRAIN_SYSTEM_MESSAGE
    prompt = cfg["task_prompt"]

    max_new_tokens = int(inf_cfg.get("max_new_tokens", max_new_tokens))
    max_len = int(inf_cfg.get("max_seq_length", cfg.get("max_seq_length", 1024)))
    pad_to_multiple_of = inf_cfg.get("pad_to_multiple_of", cfg.get("pad_to_multiple_of", None))

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixel_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=pixel_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        pad_to_multiple_of=pad_to_multiple_of,
        add_special_tokens=False,
    )

    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(device=device, dtype=dtype)
        else:
            inputs[k] = v.to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(inf_cfg.get("do_sample", False)),
    )
    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = float(inf_cfg.get("temperature", 0.7))
        gen_kwargs["top_p"] = float(inf_cfg.get("top_p", 0.9))
    else:
        gen_kwargs["temperature"] = 0.0

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    in_len = inputs["input_ids"].shape[1]
    out = processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=False)[0]
    bboxes = parse_boxes_from_text(out)
    return {"answer": out, "image": image, "objects": {"bbox": bboxes}}

def draw_box_0to1000(sample, save_path=None):
    img = sample["image"]
    bboxes = sample["objects"]["bbox"]
    fig, ax = plt.subplots()
    ax.imshow(img)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        rect = patches.Rectangle(
            (x0 / 1000 * img.width, y0 / 1000 * img.height),
            (x1 - x0) / 1000 * img.width,
            (y1 - y0) / 1000 * img.height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

def quick_viz(idx=0, dataset=None, from_dir=None, split="val", out_dir=None, save_name=None, cfg=None):
    if not cfg or "model_id" not in cfg or "task_prompt" not in cfg:
        raise ValueError("Config must include model_id and task_prompt")
    if dataset is None:
        raise ValueError("Dataset must be provided to avoid reloading.")

    inf_cfg = cfg.get("inference", {}) if isinstance(cfg, dict) else {}
    quantized = bool(inf_cfg.get("quantized", False))
    use_adapters = bool(inf_cfg.get("use_adapters", True))

    model, processor = load_model(
        model_id=cfg["model_id"],
        dtype=torch.bfloat16 if quantized else torch.float32,
        quantized=quantized,
        use_adapters=use_adapters,
        from_dir=from_dir,
        cfg=cfg,
    )

    ex = dataset[idx]
    msgs = ex["messages"]
    image = None
    for c in msgs[1]["content"]:
        if c.get("type") == "image":
            image = c.get("image")

    sample = run_inference_strict(
        model,
        processor=processor,
        image_or_url=image,
        cfg=cfg,
        dtype=getattr(model, "dtype", torch.float32),
    )

    base_dir = out_dir or from_dir or "./outputs"
    os.makedirs(base_dir, exist_ok=True)
    save_name = save_name or f"viz_{split}_{idx}.png"
    save_path = os.path.join(base_dir, save_name)
    draw_box_0to1000(sample, save_path=save_path)
    print(f"✅ Visualization saved: {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_dir", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    print("⚠️ quick_viz CLI still requires dataset/cfg to be passed in code.")
