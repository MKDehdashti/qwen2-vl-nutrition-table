"""Tests for the detection metrics.

The regression test at the bottom of this file is the one that matters: the
original implementation counted every IoU-matrix cell above threshold instead
of performing one-to-one matching, which let recall exceed 1.0 whenever the
model emitted two overlapping boxes for a single ground-truth table.
"""

import filecmp
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FT_METRICS = REPO / "nutrition_table_fine_tuning" / "src" / "metrics.py"
INF_METRICS = REPO / "nutrition_table_inference" / "src" / "metrics.py"

sys.path.insert(0, str(FT_METRICS.parent))
import metrics  # noqa: E402


# --------------------------------------------------------------------------
# iou
# --------------------------------------------------------------------------

def test_identical_boxes_have_iou_one():
    assert metrics.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_disjoint_boxes_have_iou_zero():
    assert metrics.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_touching_boxes_have_iou_zero():
    assert metrics.iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: 50 / (100 + 100 - 50).
    assert metrics.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_contained_box():
    # 5x5 inside 10x10: 25 / 100.
    assert metrics.iou([0, 0, 10, 10], [0, 0, 5, 5]) == pytest.approx(0.25)


def test_degenerate_box_does_not_divide_by_zero():
    assert metrics.iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert metrics.iou([0, 0, 10, 10], [5, 5, 5, 5]) == 0.0


def test_iou_is_symmetric():
    a, b = [10, 20, 110, 220], [50, 60, 150, 260]
    assert metrics.iou(a, b) == pytest.approx(metrics.iou(b, a))


# --------------------------------------------------------------------------
# greedy_match
# --------------------------------------------------------------------------

def test_each_ground_truth_matches_at_most_one_prediction():
    gt = [[100, 100, 500, 500]]
    pred = [[100, 100, 500, 500], [105, 105, 495, 495]]
    assert len(metrics.greedy_match(gt, pred)) == 1


def test_matching_prefers_the_higher_iou_pair():
    gt = [[0, 0, 100, 100]]
    poor = [10, 10, 100, 100]
    exact = [0, 0, 100, 100]
    (gt_idx, pred_idx, score), = metrics.greedy_match(gt, [poor, exact])
    assert (gt_idx, pred_idx) == (0, 1)
    assert score == pytest.approx(1.0)


def test_below_threshold_pairs_are_not_matched():
    gt = [[0, 0, 100, 100]]
    pred = [[90, 90, 190, 190]]  # ~0.005 IoU
    assert metrics.greedy_match(gt, pred) == []


def test_two_distinct_tables_both_match():
    gt = [[0, 0, 100, 100], [500, 500, 600, 600]]
    pred = [[500, 500, 600, 600], [0, 0, 100, 100]]
    assert len(metrics.greedy_match(gt, pred)) == 2


# --------------------------------------------------------------------------
# detection_metrics
# --------------------------------------------------------------------------

def test_perfect_single_detection():
    box = [100, 100, 500, 500]
    iou_mean, precision, recall, f1 = metrics.detection_metrics([box], [box])
    assert (iou_mean, precision, recall, f1) == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_correct_negative_scores_perfect():
    assert metrics.detection_metrics([], []) == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_missed_detection_scores_zero():
    assert metrics.detection_metrics([[0, 0, 10, 10]], []) == pytest.approx((0.0,) * 4)


def test_false_positive_on_empty_ground_truth_scores_zero():
    assert metrics.detection_metrics([], [[0, 0, 10, 10]]) == pytest.approx((0.0,) * 4)


def test_duplicate_prediction_is_penalised_as_a_false_positive():
    gt = [[100, 100, 500, 500]]
    pred = [[100, 100, 500, 500], [105, 105, 495, 495]]
    _, precision, recall, f1 = metrics.detection_metrics(gt, pred)
    assert precision == pytest.approx(0.5)   # one of two predictions is spurious
    assert recall == pytest.approx(1.0)      # the single table was found
    assert f1 == pytest.approx(2 / 3)


def test_mean_iou_is_unaffected_by_the_matching_threshold():
    # Published mean-IoU figures must stay comparable across the matching fix.
    gt = [[0, 0, 100, 100]]
    pred = [[10, 10, 110, 110]]
    direct = metrics.mean_pairwise_iou(gt, pred)
    assert metrics.detection_metrics(gt, pred)[0] == pytest.approx(direct)
    assert metrics.detection_metrics(gt, pred, iou_threshold=0.9)[0] == pytest.approx(direct)


# --------------------------------------------------------------------------
# regression: the bug this module was written to fix
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gt,pred",
    [
        ([[100, 100, 500, 500]], [[100, 100, 500, 500], [105, 105, 495, 495]]),
        ([[0, 0, 100, 100]], [[0, 0, 100, 100]] * 5),
        ([[0, 0, 100, 100], [0, 0, 101, 101]], [[0, 0, 100, 100]] * 4),
    ],
)
def test_precision_and_recall_never_exceed_one(gt, pred):
    """The original metric returned recall=2.0 and f1=1.33 for the first case.

    It counted every cell of the IoU matrix above threshold, so N overlapping
    predictions against one ground-truth box scored N true positives.
    """
    _, precision, recall, f1 = metrics.detection_metrics(gt, pred)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0


# --------------------------------------------------------------------------
# the two copies must not drift
# --------------------------------------------------------------------------

def test_metrics_module_is_identical_in_both_subprojects():
    assert FT_METRICS.exists() and INF_METRICS.exists()
    assert filecmp.cmp(FT_METRICS, INF_METRICS, shallow=False), (
        "metrics.py has diverged between the fine-tuning and inference packages; "
        "copy one over the other."
    )
