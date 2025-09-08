import re
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint
import torch
from qwen_vl_utils import process_vision_info
from data_utils import build_messages

def run_inference_strict(model, processor, image_or_url, prompt, max_new_tokens=1024, dtype=None):
    if dtype is None:
        dtype = getattr(model, "dtype", torch.float32)
    device = next(model.parameters()).device

    image = Image.open(BytesIO(requests.get(image_or_url).content)).convert("RGB") if isinstance(image_or_url, str) else image_or_url
    fake_example = {"image": image, "objects": {"bbox": []}}
    custom_prompt = prompt.strip() + "\nReturn only: (x0,y0),(x1,y1) with integers 0..1000. No words, no code fences."
    messages = build_messages(fake_example, with_answer=False, prompt=custom_prompt)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pixel_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=pixel_inputs, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, top_p=None, top_k=None)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    blocks = re.findall(r"<\|box_start\|>(.*?)<\|box_end\|>", out, flags=re.DOTALL)
    bboxes = []
    for block in blocks:
        pairs = re.findall(r"\(<\|coords\|>(\d+)<\|/coords\|>,<\|coords\|>(\d+)<\|/coords\|>\)", block)
        if len(pairs) >= 2:
            x0, y0 = map(int, pairs[0]); x1, y1 = map(int, pairs[1]); bboxes.append([x0, y0, x1, y1])
    if not bboxes:
        nums = re.findall(r"-?\d+(?:\.\d+)?", out)
        if len(nums) >= 4:
            x0, y0, x1, y1 = map(int, nums[:4]); bboxes.append([x0, y0, x1, y1])

    return {"answer": out, "image": image, "objects": {"bbox": bboxes}}

def draw_box_0to1000(sample, save_path=None):
    pprint(sample)
    img = sample["image"]
    bboxes = sample["objects"]["bbox"]
    fig, ax = plt.subplots(); ax.imshow(img)
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        y0 /= 1000; x0 /= 1000; y1 /= 1000; x1 /= 1000
        x0_px = x0 * img.width; y0_px = y0 * img.height
        width = (x1 - x0) * img.width; height = (y1 - y0) * img.height
        rect = patches.Rectangle((x0_px, y0_px), width, height, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()
