import os
import torch
import wandb
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from viz_utils import run_inference_strict, draw_box_0to1000
from model_utils import load_model
from data_utils import train_ds, test_ds

def test_one(model, processor, idx=20, tag="baseline", split="val", out_dir=None, training_args=None):
    ds = test_ds if split == "val" else train_ds
    image = ds[idx]["image"]
    infer_dtype = getattr(model, "dtype", torch.float32)
    sample = run_inference_strict(model, processor, image, "Detect the bounding box of the nutrition table.", 1024, dtype=infer_dtype)

    base_dir = out_dir
    if base_dir is None and training_args is not None and hasattr(training_args, "output_dir"):
        base_dir = training_args.output_dir
    if base_dir is None:
        base_dir = "./outputs"
    os.makedirs(base_dir, exist_ok=True)

    out_file = os.path.join(base_dir, f"bbox_{tag}_{split}_{idx}.png")
    draw_box_0to1000(sample, save_path=out_file)
    print(f"✅ Saved {out_file}")

def evaluate_model(model, processor, n=50, strict=True, iou_thresholds=None, plot=True, tag="baseline", out_dir=None, training_args=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    infer_dtype = getattr(model, "dtype", torch.float32)
    dev = next(model.parameters()).device

    def _to_xyxy(b): return [b[1], b[0], b[3], b[2]]
    def _scale(boxes):
        if not boxes: return boxes
        mx = max(max(b) for b in boxes)
        return [[c * 1000.0 for c in b] for b in boxes] if mx <= 1.0 else boxes

    model.eval()
    per_image_ious = []
    all_tp, all_fp, all_fn = {t: 0 for t in iou_thresholds}, {t: 0 for t in iou_thresholds}, {t: 0 for t in iou_thresholds}
    n = min(n, len(test_ds))

    for idx in range(n):
        ex = test_ds[idx]
        gt_boxes = _scale(ex.get("objects", {}).get("bbox", []))
        if not gt_boxes: continue
        pred = run_inference_strict(model, processor, ex["image"], "Detect the bounding box of the nutrition table.", 1024, dtype=infer_dtype)
        pred_boxes = _scale(pred.get("objects", {}).get("bbox", []))
        if not pred_boxes: continue
        gt_t = torch.tensor([_to_xyxy(b) for b in gt_boxes], dtype=torch.float32, device=dev)
        pr_t = torch.tensor([_to_xyxy(b) for b in pred_boxes], dtype=torch.float32, device=dev)
        iou_mat = box_iou(gt_t, pr_t).cpu()
        if strict:
            used_preds, matched = set(), []
            for gi in range(iou_mat.size(0)):
                row = iou_mat[gi]
                best_pi = torch.argmax(row).item()
                best_iou = row[best_pi].item()
                if best_pi not in used_preds:
                    used_preds.add(best_pi)
                    matched.append(best_iou)
            if matched:
                per_image_ious.append(sum(matched) / len(matched))
        else:
            matched = [row.max().item() for row in iou_mat]
            if matched:
                per_image_ious.append(sum(matched) / len(matched))
        for thr in iou_thresholds:
            tp, fp = 0, 0
            matched_gt = set()
            for pi in range(iou_mat.size(1)):
                best_iou, gi = iou_mat[:, pi].max(dim=0)
                if best_iou.item() >= thr:
                    tp += 1; matched_gt.add(gi.item())
                else:
                    fp += 1
            fn = iou_mat.size(0) - len(matched_gt)
            all_tp[thr] += tp; all_fp[thr] += fp; all_fn[thr] += fn

    mean_iou = sum(per_image_ious) / len(per_image_ious) if per_image_ious else 0.0
    metrics = {"mean_iou": mean_iou}
    for thr in iou_thresholds:
        tp, fp, fn = all_tp[thr], all_fp[thr], all_fn[thr]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"precision@{thr}"] = precision
        metrics[f"recall@{thr}"] = recall
        metrics[f"f1@{thr}"] = f1
    ap_scores = []
    for thr in iou_thresholds:
        tp, fp, fn = all_tp[thr], all_fp[thr], all_fn[thr]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ap_scores.append(precision * recall)
    metrics["mAP"] = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    print(f"Mean IoU ({'strict' if strict else 'optimistic'}): {metrics['mean_iou']:.4f}")
    for thr in iou_thresholds:
        print(f"Precision@{thr}: {metrics[f'precision@{thr}']:.4f}")
        print(f"Recall@{thr}: {metrics[f'recall@{thr}']:.4f}")
        print(f"F1@{thr}: {metrics[f'f1@{thr}']:.4f}")
    print(f"mAP over {iou_thresholds}: {metrics['mAP']:.4f}")
    if plot:
        ths = list(iou_thresholds)
        precisions = [metrics[f"precision@{t}"] for t in ths]
        recalls = [metrics[f"recall@{t}"] for t in ths]
        plt.figure(figsize=(6, 4))
        plt.plot(ths, precisions, marker="o", label="Precision")
        plt.plot(ths, recalls, marker="s", label="Recall")
        plt.xlabel("IoU Threshold"); plt.ylabel("Score"); plt.title("Precision / Recall vs IoU Threshold")
        plt.legend(); plt.grid(True)
        # Pick a safe save path
        base_dir = None
        if out_dir:
            base_dir = out_dir
        elif training_args is not None and hasattr(training_args, "output_dir"):
            base_dir = training_args.output_dir
        else:
            base_dir = "./outputs"
            os.makedirs(base_dir, exist_ok=True)

        save_path = os.path.join(
            base_dir,
            f"precision_recall_curve_{'strict' if strict else 'optimistic'}_{tag}.png"
        )

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Precision/Recall curve saved to {save_path}")
        try: plt.show()
        except Exception: pass
        plt.close()
        metrics["pr_curve_path"] = save_path
    return metrics

def run_and_log_eval(model_or_path, processor=None, n=50, strict=True, tag="eval",
                     quantized=True, use_adapters=True, step=None, from_dir=None, training_args=None):
    # --- Load model + processor ---
    if isinstance(model_or_path, tuple):
        model, processor = model_or_path
    elif isinstance(model_or_path, str):
        if model_or_path == "baseline":
            model, processor = load_model(
                dtype=torch.bfloat16,
                quantized=quantized,
                use_adapters=False,
                from_dir=from_dir,
                training_args=training_args,
            )
        else:
            model, processor = load_model(
                dtype=torch.bfloat16,
                quantized=quantized,
                use_adapters=use_adapters,
                from_dir=from_dir,
                training_args=training_args,
            )
    else:
        model, processor = model_or_path, processor

    # --- Decide output dir safely ---
    out_dir = None
    if from_dir:
        out_dir = from_dir
    elif training_args is not None and hasattr(training_args, "output_dir"):
        out_dir = training_args.output_dir

    # --- Run evaluation ---
    metrics = evaluate_model(
        model,
        processor,
        n=n,
        strict=strict,
        tag=tag,
        out_dir=out_dir,
        training_args=training_args,
    )

    # --- Log to W&B ---
    log_dict = {f"{tag}_{k}": v for k, v in metrics.items() if k != "pr_curve_path"}
    if "pr_curve_path" in metrics:
        log_dict[f"{tag}_pr_curve"] = wandb.Image(metrics["pr_curve_path"])
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)

    print(f"📊 {tag} Metrics:", metrics)

    # --- Cleanup ---
    del model, processor
    torch.cuda.empty_cache()
    return metrics
