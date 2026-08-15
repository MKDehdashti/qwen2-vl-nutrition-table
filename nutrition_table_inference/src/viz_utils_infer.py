# viz_utils_infer.py
import os, torch, requests, argparse
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .prompts import TASK_PROMPT


def run_inference_strict(model, processor, image_or_url, prompt,
                         max_new_tokens=512, dtype=None):
    from qwen_vl_utils import process_vision_info
    from .dataset.data_utils import parse_boxes_from_text

    if dtype is None:
        dtype = getattr(model, "dtype", torch.float32)
    device = next(model.parameters()).device if torch.cuda.is_available() else "cpu"

    if isinstance(image_or_url, str) and not isinstance(image_or_url, Image.Image):
        image = Image.open(BytesIO(requests.get(image_or_url).content)).convert("RGB")
    else:
        image = image_or_url

    fake_example = {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "System message"}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]}

    text = processor.apply_chat_template(fake_example["messages"], tokenize=False, add_generation_prompt=True)
    pixel_inputs, _ = process_vision_info(fake_example["messages"])
    inputs = processor(text=[text], images=pixel_inputs, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=0.0)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=False)[0]

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
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def quick_viz(idx=0, dataset=None, from_dir=None, split="val",
              prompt=TASK_PROMPT,
              out_dir=None, save_name=None,
              quantized=True, use_adapters=True, cfg=None):
    from .model_utils import load_model

    if not cfg or "model_id" not in cfg:
        raise ValueError("Config must include 'model_id'")
    if dataset is None:
        raise ValueError("Dataset must be provided to avoid reloading.")

    model, processor = load_model(
        model_id=cfg["model_id"],
        dtype=torch.bfloat16 if quantized else torch.float32,
        quantized=quantized,
        use_adapters=use_adapters,
        from_dir=from_dir,
        cfg=cfg
    )

    ex = dataset[idx]
    msgs = ex["messages"]

    image = None
    for c in msgs[1]["content"]:
        if c["type"] == "image":
            image = c["image"]

    sample = run_inference_strict(model, processor, image, prompt, dtype=model.dtype)

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
    parser.add_argument("--prompt", type=str, default=TASK_PROMPT)
    args = parser.parse_args()
    cfg = {"model_id": "Qwen/Qwen2-VL-7B-Instruct"}
    print("quick_viz standalone mode requires dataset to be passed in code, not via this CLI.")
