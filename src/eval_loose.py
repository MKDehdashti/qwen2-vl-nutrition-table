# eval_loose.py

import torch, re, argparse
from torchvision import ops
from model_utils import load_model, load_processor_fixed
from dataset.data_utils import test_ds, build_messages
from qwen_vl_utils import process_vision_info


def generate_text_from_sample(model, processor, sample, max_new_tokens=256, device="cuda"):
    text = processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
    image_input, video_input = process_vision_info(sample)
    inputs = processor(
        text=[text],
        images=image_input,
        padding=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    return output_text


def parse_output(output_text):
    bounding_boxes = []
    for output in output_text:
        if "<|box_start|>" in output and "<|box_end|>" in output:
            box_str = output.split("<|box_start|>")[1].split("<|box_end|>")[0]
            nums = re.findall(r"-?\d+", box_str)
        else:
            nums = re.findall(r"-?\d+", output)
        if len(nums) >= 4:
            x1, y1, x2, y2 = map(int, nums[:4])
            bounding_boxes.append((x1, y1, x2, y2))
    return bounding_boxes


def _to_xyxy(b):
    # dataset format is (y0, x0, y1, x1) → convert to (x0, y0, x1, y1)
    return [b[1], b[0], b[3], b[2]]


def eval_iou(model, processor, examples):
    iou_score = []
    bad_examples = []
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for i in range(len(examples)):
        print("Current Sample Index:", i, "out of", len(examples))
        ex = examples[i]

        # ground truth boxes (normalized floats, need scaling + swap)
        gt_boxes = ex.get("objects", {}).get("bbox", [])
        if not gt_boxes:
            continue
        gt_boxes = [[x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000] for (x0, y0, x1, y1) in gt_boxes]
        truth = torch.tensor([_to_xyxy(b) for b in gt_boxes], dtype=torch.float32)

        # user prompt → chat messages
        messages = build_messages(ex, with_answer=False, prompt="Detect the bounding box of the nutrition table.")
        output_text = generate_text_from_sample(model=model, processor=processor, sample=messages)
        print(output_text)

        try:
            bboxes_pred = parse_output(output_text)
            print("pred:", bboxes_pred, "truth:", truth.tolist())
            pred = torch.tensor(bboxes_pred, dtype=torch.float32)
            iou_score.append(ops.box_iou(pred, truth))
        except Exception as e:
            bad_examples.append(i)
            print("❌ parse error on example", i, ":", e)
            continue

    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"Evaluation took {elapsed_ms/1000:.1f}s wall-time (GPU kernels)")

    return iou_score, bad_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--from_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    cfg = {"model_id": args.model_id}
    model, processor = load_model(
        model_id=args.model_id,
        use_adapters=True,
        from_dir=args.from_dir,
        cfg=cfg
    )

    subset = test_ds.select(range(min(args.n, len(test_ds))))
    ious, bad = eval_iou(model, processor, subset)

    mean_iou = torch.stack([x.mean() for x in ious]).mean().item() if ious else 0.0
    print("📊 Mean IoU:", mean_iou)
    print("❌ Bad examples:", bad)
