import os
import argparse
import base64
from io import BytesIO

import requests
from PIL import Image

from .dataset.data_utils import parse_boxes_from_text, system_message
from .viz_utils_infer import draw_box_0to1000



# --------------------------
# Load image
# --------------------------
def load_image_from_url(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    return img


# --------------------------
# Convert PIL to data URL
# --------------------------
def image_to_data_url(img, fmt="JPEG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# --------------------------
# Call vLLM Server
# --------------------------
def call_vllm(server_url, api_key, model, prompt, img_data_url, max_new_tokens):
    # payload = {
    #     "model": model,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": img_data_url
    #                     },
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": prompt,
    #                 },
    #             ],
    #         }
    #     ],
    #     "max_tokens": max_new_tokens,
    #     "temperature": 0.0,
    # }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_data_url
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Detect the bounding box of the nutrition table. "
                            "Respond only with coordinates in this format: "
                            "(x_min, y_min),(x_max, y_max). "
                            "If there are multiple tables, output multiple boxes "
                            "separated by spaces."
                        ),
                    },
                ],
            },
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }



    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = server_url.rstrip("/") + "/chat/completions"

    print("DEBUG: POST", url)
    print("DEBUG: model field =", payload["model"])

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    print("DEBUG: status =", resp.status_code)
    print("DEBUG: raw response text (first 500 chars):")
    print(resp.text[:500])

    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        text = "\n".join(texts)
    else:
        text = content

    return text




# --------------------------
# Main pipeline: load img → call model → parse boxes → draw → save
# --------------------------
def run_inference_vllm(server_url, api_key, model, image_url, prompt, output_path, max_new_tokens):
    img = load_image_from_url(image_url)
    data_url = image_to_data_url(img)

    # Call vLLM
    text = call_vllm(server_url, api_key, model, prompt, data_url, max_new_tokens)

    # Parse numeric XYXY boxes (0–1000)
    bboxes = parse_boxes_from_text(text)

    # Prepare sample for viz_utils
    sample = {"image": img, "objects": {"bbox": bboxes}}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Draw bounding boxes
    draw_box_0to1000(sample, save_path=output_path)

    return img, bboxes, text, output_path


# --------------------------
# CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/projects/nutrition-table3/model/Qwen2-VL-7B/quantized_gptqmodel",
    )
    parser.add_argument("--image_url", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Detect the bounding box of the nutrition table.")
    parser.add_argument("--output_path", type=str, default="outputs/vllm_viz.png")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    img, bboxes, text, out_path = run_inference_vllm(
        server_url=args.server_url,
        api_key=args.api_key,
        model=args.model,
        image_url=args.image_url,
        prompt=args.prompt,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
    )

    print("Saved image with boxes to:", out_path)
    print("Parsed boxes:", bboxes)
    print("Raw model output:")
    print(text)


if __name__ == "__main__":
    main()
