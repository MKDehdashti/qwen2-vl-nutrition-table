import os, re, torch, requests, argparse
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from qwen_vl_utils import process_vision_info
from dataset.data_utils import build_messages, train_ds, test_ds
from model_utils import load_model, load_processor_fixed

def run_inference_strict(model, processor, image_or_url, prompt, max_new_tokens=512, dtype=None):
    if dtype is None:
        dtype = getattr(model, "dtype", torch.float32)
    device = next(model.parameters()).device if torch.cuda.is_available() else "cpu"

    if isinstance(image_or_url, str) and not isinstance(image_or_url, Image.Image):
        image = Image.open(BytesIO(requests.get(image_or_url).content)).convert("RGB")
    else:
        image = image_or_url

    fake_example = {"image": image, "objects": {"bbox": []}}
    custom_prompt = prompt.strip() + "\nReturn only: (x0,y0),(x1,y1) with integers 0..1000."
    messages = build_messages(fake_example, with_answer=False, prompt=custom_prompt)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixel_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=pixel_inputs, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=False)[0]

    bboxes = []
    nums = re.findall(r"-?\d+", out)
    if len(nums) >= 4:
        x0, y0, x1, y1 = map(int, nums[:4])
        bboxes.append([x0, y0, x1, y1])

    return {"answer": out, "image": image, "objects": {"bbox": bboxes}}

def draw_box_0to1000(sample, save_path=None):
    img = sample["image"]
    bboxes = sample["objects"]["bbox"]
    fig, ax = plt.subplots(); ax.imshow(img)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        y0 /= 1000; x0 /= 1000; y1 /= 1000; x1 /= 1000
        x0_px = x0 * img.width; y0_px = y0 * img.height
        width = (x1 - x0) * img.width; height = (y1 - y0) * img.height
        rect = patches.Rectangle((x0_px, y0_px), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150); plt.close(fig)
    else:
        plt.show()

def quick_viz(idx=0, from_dir=None, split="val",
              prompt="Detect the bounding box of the nutrition table.",
              out_dir=None, save_name=None,
              quantized=True, use_adapters=True, cfg=None):
    if not cfg or "model_id" not in cfg:
        raise ValueError("Config must include 'model_id'")

    model, _ = load_model(
        model_id=cfg["model_id"],
        dtype=torch.bfloat16 if quantized else torch.float32,
        quantized=quantized,
        use_adapters=use_adapters,
        from_dir=from_dir,
        cfg=cfg
    )
    processor = load_processor_fixed(
        model_id=cfg["model_id"],
        min_pixels=cfg.get("debug", {}).get("min_pixels", 224*224) if cfg else 224*224,
        max_pixels=cfg.get("debug", {}).get("max_pixels", 900*28*28) if cfg else 900*28*28,
    )

    ds = test_ds if split == "val" else train_ds
    ex = ds[idx]
    sample = run_inference_strict(model, processor, ex["image"], prompt, dtype=model.dtype)

    base_dir = out_dir or from_dir or "./outputs"
    os.makedirs(base_dir, exist_ok=True)
    save_name = save_name or f"viz_{split}_{idx}.png"
    save_path = os.path.join(base_dir, save_name)
    draw_box_0to1000(sample, save_path=save_path)
    print(f"✅ Visualization saved: {save_path}")
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_dir", type=str, required=True, help="Checkpoint dir")
    parser.add_argument("--idx", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    parser.add_argument("--prompt", type=str, default="Detect the bounding box of the nutrition table.", help="Custom prompt")
    args = parser.parse_args()
    # minimal cfg for CLI
    cfg = {"model_id": "Qwen/Qwen2-VL-7B-Instruct"}
    quick_viz(idx=args.idx, from_dir=args.from_dir, split=args.split, prompt=args.prompt, cfg=cfg)
