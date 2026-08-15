"""Detection metrics for nutrition-table bounding boxes.

Single source of truth for IoU, precision, recall and F1. This module is
duplicated verbatim into both subprojects (fine-tuning and inference) because
they are deployed independently with separate virtualenvs; `tests/test_metrics.py`
asserts the two copies stay byte-identical.

Deliberately pure-Python: no torch, no torchvision, no numpy. The metrics are
the part of the pipeline most worth testing, and keeping them dependency-free
means the test suite runs anywhere in about a second.

Boxes are `(x_min, y_min, x_max, y_max)` in any consistent unit; this codebase
uses a 0-1000 normalized scale. Area follows the torchvision `box_iou`
convention -- `(x_max - x_min) * (y_max - y_min)`, with no +1 pixel term -- so
results match the previous implementation for the IoU values themselves.
"""

from typing import List, Sequence, Tuple

Box = Sequence[float]


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes. Returns 0.0 for degenerate input."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def iou_matrix(gt: Sequence[Box], pred: Sequence[Box]) -> List[List[float]]:
    """Full |gt| x |pred| IoU matrix."""
    return [[iou(g, p) for p in pred] for g in gt]


def greedy_match(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching, highest IoU first.

    Each ground-truth box matches at most one prediction and vice versa, which
    is what makes precision and recall bounded by 1.0. Repeatedly takes the
    highest remaining IoU above the threshold and retires both its row and its
    column.

    Returns a list of `(gt_index, pred_index, iou)` tuples.
    """
    matrix = iou_matrix(gt, pred)
    matched: List[Tuple[int, int, float]] = []
    used_gt, used_pred = set(), set()

    while True:
        best = (0.0, -1, -1)
        for i, row in enumerate(matrix):
            if i in used_gt:
                continue
            for j, value in enumerate(row):
                if j in used_pred:
                    continue
                if value > best[0]:
                    best = (value, i, j)

        score, i, j = best
        if i < 0 or score <= iou_threshold:
            break

        matched.append((i, j, score))
        used_gt.add(i)
        used_pred.add(j)

    return matched


def mean_pairwise_iou(gt: Sequence[Box], pred: Sequence[Box]) -> float:
    """Symmetric mean IoU: average of best-per-ground-truth and best-per-prediction.

    This is the definition used for every published mean-IoU figure in this
    project. It is threshold-free and unaffected by the matching strategy, so
    numbers reported before and after the matching fix remain comparable.
    """
    if not gt or not pred:
        return 0.0
    matrix = iou_matrix(gt, pred)
    best_per_gt = [max(row) for row in matrix]
    best_per_pred = [max(col) for col in zip(*matrix)]
    return (
        sum(best_per_gt) / len(best_per_gt) + sum(best_per_pred) / len(best_per_pred)
    ) / 2.0


def detection_metrics(
    gt: Sequence[Box],
    pred: Sequence[Box],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Return `(mean_iou, precision, recall, f1)` for one image.

    Precision and recall use greedy one-to-one matching, so both are guaranteed
    to lie in [0, 1].

    Empty-input convention:
      * no ground truth and no prediction -> a correct negative, all 1.0
      * one side empty and the other not  -> all 0.0
    """
    if not gt and not pred:
        return 1.0, 1.0, 1.0, 1.0
    if not gt or not pred:
        return 0.0, 0.0, 0.0, 0.0

    true_positives = len(greedy_match(gt, pred, iou_threshold=iou_threshold))
    precision = true_positives / len(pred)
    recall = true_positives / len(gt)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return mean_pairwise_iou(gt, pred), precision, recall, f1
