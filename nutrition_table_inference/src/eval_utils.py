import os, torch, matplotlib.pyplot as plt, json
from torchvision.ops import box_iou
from viz_utils import run_inference_strict
from dataset.data_utils import parse_boxes_from_text

def evaluate_model(
    model,
    processor,
    n=123,
    dataset=None,
    strict=True,
    tag=None,
    cfg=None,
    training_args=None,
    plot=True,
    dump_path=None,
):
    model.eval()
    device = next(model.parameters()).device

    if dataset is None:
        raise ValueError("Dataset must be provided to avoid redundant loading.")

    ds = dataset.select(range(min(n, len(dataset)))) if hasattr(dataset, "select") else dataset[:n]

    ious, precisions, recalls, f1s, results = [], [], [], [], []
    per_sample = []

    for idx in range(len(ds)):
        sample = ds[idx]
        msgs = sample["messages"]

        gt_text = ""
        if msgs and msgs[-1]["role"] == "assistant":
            content = msgs[-1]["content"]
            if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
                gt_text = content[0]["text"]
            elif isinstance(content, str):
                gt_text = content

        gt_boxes = parse_boxes_from_text(gt_text)

        image = None
        for c in msgs[1]["content"]:
            if c["type"] == "image":
                image = c["image"]

        metrics = run_inference_strict(
            model,
            processor=processor,
            image_or_url=image,
            prompt="Detect the bounding box of the nutrition table.",
            dtype=getattr(model, "dtype", torch.float32),
        )
        if metrics is None:
            continue

        pred_boxes = metrics.get("objects", {}).get("bbox", [])
        iou_mean, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

        if gt_boxes and pred_boxes:
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            pr_t = torch.tensor(pred_boxes, dtype=torch.float32)
            if gt_t.ndim == 1: gt_t = gt_t.unsqueeze(0)
            if pr_t.ndim == 1: pr_t = pr_t.unsqueeze(0)

            ious_mat = box_iou(gt_t, pr_t)
            best_gt = ious_mat.max(dim=1)[0]
            best_pr = ious_mat.max(dim=0)[0]
            iou_mean = (best_gt.mean() + best_pr.mean()).item() / 2

            matched = (ious_mat > 0.5).sum().item()
            precision = matched / len(pr_t)
            recall = matched / len(gt_t)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["iou"] = iou_mean
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        ious.append(iou_mean)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        results.append(metrics)

        per_sample.append(
            {
                "idx": idx,
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                "iou": iou_mean,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    out = {}
    if ious: out["mean_iou"] = sum(ious) / len(ious)
    if precisions: out["precision@0.5"] = sum(precisions) / len(precisions)
    if recalls: out["recall@0.5"] = sum(recalls) / len(recalls)
    if f1s: out["f1@0.5"] = sum(f1s) / len(f1s)

    if plot and results and training_args:
        precisions_curve = [m.get("precision", 0.0) for m in results]
        recalls_curve = [m.get("recall", 0.0) for m in results]
        plt.figure()
        plt.plot(recalls_curve, precisions_curve, marker="o")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve ({tag})")
        pr_path = os.path.join(training_args.output_dir, f"precision_recall_curve_{tag}.png")
        plt.savefig(pr_path, bbox_inches="tight")
        plt.close()
        out["pr_curve_path"] = pr_path

    if dump_path is not None:
        with open(dump_path, "w") as f:
            json.dump(per_sample, f, indent=2)

    return out
