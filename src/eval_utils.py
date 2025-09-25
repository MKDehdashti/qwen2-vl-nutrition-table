import os, torch, wandb, argparse
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from viz_utils import run_inference_strict, draw_box_0to1000
from model_utils import load_model, load_processor_fixed
from dataset.data_utils import train_ds, test_ds


def test_one(model, idx=20, tag="baseline", split="val", out_dir=None,
             training_args=None, cfg=None):
    ds = test_ds if split == "val" else train_ds
    image = ds[idx]["image"]
    infer_dtype = getattr(model, "dtype", torch.float32)
    processor = load_processor_fixed(
        model_id=cfg["model_id"],
        min_pixels=cfg.get("debug", {}).get("min_pixels", 224 * 224) if cfg else 224 * 224,
        max_pixels=cfg.get("debug", {}).get("max_pixels", 900 * 28 * 28) if cfg else 900 * 28 * 28,
    )
    sample = run_inference_strict(
        model, processor, image,
        "Detect the bounding box of the nutrition table.", 1024,
        dtype=infer_dtype
    )

    base_dir = out_dir or getattr(training_args, "output_dir", "./outputs")
    os.makedirs(base_dir, exist_ok=True)
    out_file = os.path.join(base_dir, f"bbox_{tag}_{split}_{idx}.png")
    draw_box_0to1000(sample, save_path=out_file)
    print(f"✅ Saved {out_file}")


def evaluate_model(model, n=50, strict=True, iou_thresholds=None,
                   plot=True, tag="baseline", out_dir=None,
                   training_args=None, cfg=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    infer_dtype = getattr(model, "dtype", torch.float32)
    dev = next(model.parameters()).device
    processor = load_processor_fixed(
        model_id=cfg["model_id"],
        min_pixels=cfg.get("debug", {}).get("min_pixels", 224 * 224) if cfg else 224 * 224,
        max_pixels=cfg.get("debug", {}).get("max_pixels", 900 * 28 * 28) if cfg else 900 * 28 * 28,
    )

    def _to_xyxy(b): return [b[1], b[0], b[3], b[2]]

    def _scale(boxes):
        if not boxes:
            return boxes
        mx = max(max(b) for b in boxes)
        return [[c * 1000.0 for c in b] for b in boxes] if mx <= 1.0 else boxes

    model.eval()
    per_image_ious = []
    all_tp, all_fp, all_fn = {t: 0 for t in iou_thresholds}, {t: 0 for t in iou_thresholds}, {t: 0 for t in iou_thresholds}
    n = min(n, len(test_ds))

    for idx in range(n):
        ex = test_ds[idx]
        gt_boxes = _scale(ex.get("objects", {}).get("bbox", []))
        if not gt_boxes:
            continue
        pred = run_inference_strict(
            model, processor, ex["image"],
            "Detect the bounding box of the nutrition table.", 1024,
            dtype=infer_dtype
        )
        pred_boxes = _scale(pred.get("objects", {}).get("bbox", []))
        if not pred_boxes:
            continue

        gt_t = torch.tensor([_to_xyxy(b) for b in gt_boxes], dtype=torch.float32, device=dev)
        pr_t = torch.tensor([_to_xyxy(b) for b in pred_boxes], dtype=torch.float32, device=dev)
        iou_mat = box_iou(gt_t, pr_t).cpu()

        # mean IoU
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

        # precision / recall
        for thr in iou_thresholds:
            tp, fp = 0, 0
            matched_gt = set()
            for pi in range(iou_mat.size(1)):
                best_iou, gi = iou_mat[:, pi].max(dim=0)
                if best_iou.item() >= thr:
                    tp += 1
                    matched_gt.add(gi.item())
                else:
                    fp += 1
            fn = iou_mat.size(0) - len(matched_gt)
            all_tp[thr] += tp
            all_fp[thr] += fp
            all_fn[thr] += fn

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
    ap_scores = [metrics[f"precision@{t}"] * metrics[f"recall@{t}"] for t in iou_thresholds]
    metrics["mAP"] = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

    if plot:
        ths = list(iou_thresholds)
        precisions = [metrics[f"precision@{t}"] for t in ths]
        recalls = [metrics[f"recall@{t}"] for t in ths]
        plt.figure(figsize=(6, 4))
        plt.plot(ths, precisions, marker="o", label="Precision")
        plt.plot(ths, recalls, marker="s", label="Recall")
        plt.xlabel("IoU Threshold")
        plt.ylabel("Score")
        plt.title("Precision / Recall vs IoU Threshold")
        plt.legend()
        plt.grid(True)
        save_dir = out_dir or getattr(training_args, "output_dir", "./outputs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"precision_recall_curve_{tag}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        metrics["pr_curve_path"] = save_path
        print(f"📊 Precision/Recall curve saved to {save_path}")

    print(f"📊 {tag} metrics:", metrics)
    return metrics


def run_and_log_eval(stage_name, tag="eval", n=50, strict=True,
                     quantized=True, from_dir=None,
                     training_args=None, step=None, cfg=None):
    if not cfg or "model_id" not in cfg:
        raise ValueError("Config must include 'model_id'")

    results = {}

    # --- Run with adapters active ---
    model, _ = load_model(
        model_id=cfg["model_id"],
        dtype=torch.bfloat16,
        use_adapters=True,
        from_dir=from_dir,
        cfg=cfg
    )
    metrics_adapters = evaluate_model(model, n=n, strict=strict,
                                      tag=f"{tag}_adapters", out_dir=from_dir,
                                      training_args=training_args, cfg=cfg)
    results["adapters"] = metrics_adapters
    log_dict_adapters = {f"{tag}_adapters/{k}": v
                         for k, v in metrics_adapters.items()
                         if k != "pr_curve_path"}
    if "pr_curve_path" in metrics_adapters:
        log_dict_adapters[f"{tag}_adapters/pr_curve"] = wandb.Image(metrics_adapters["pr_curve_path"])
    if wandb.run is not None:
        wandb.log(log_dict_adapters, step=step)
    else:
        print(f"[no wandb run] {stage_name} adapters metrics:", log_dict_adapters)
    del model
    torch.cuda.empty_cache()

    # --- Run with adapters merged ---
    model, _ = load_model(
        model_id=cfg["model_id"],
        dtype=torch.bfloat16,
        use_adapters=False,  # merged
        from_dir=from_dir,
        cfg=cfg
    )
    metrics_merged = evaluate_model(model, n=n, strict=strict,
                                    tag=f"{tag}_merged", out_dir=from_dir,
                                    training_args=training_args, cfg=cfg)
    results["merged"] = metrics_merged
    log_dict_merged = {f"{tag}_merged/{k}": v
                       for k, v in metrics_merged.items()
                       if k != "pr_curve_path"}
    if "pr_curve_path" in metrics_merged:
        log_dict_merged[f"{tag}_merged/pr_curve"] = wandb.Image(metrics_merged["pr_curve_path"])
    if wandb.run is not None:
        wandb.log(log_dict_merged, step=step)
    else:
        print(f"[no wandb run] {stage_name} merged metrics:", log_dict_merged)
    del model
    torch.cuda.empty_cache()

    return results


# === CLI for standalone eval ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_dir", type=str, required=True, help="Checkpoint dir")
    parser.add_argument("--config", type=str, required=True, help="Config yaml path")
    parser.add_argument("--tag", type=str, default="manual_eval", help="Eval tag")
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_and_log_eval(
        stage_name="manual",
        tag=args.tag,
        n=args.n,
        from_dir=args.from_dir,
        cfg=cfg
    )
