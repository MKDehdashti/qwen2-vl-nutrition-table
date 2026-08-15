"""Tests for prompt constants, box parsing, and coordinate conversion.

`format_data` and `parse_boxes_from_text` are round-trip partners: training
encodes boxes into text, evaluation decodes them back. A mismatch between them
would silently corrupt every metric, so they are tested together.
"""

import filecmp
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FT_SRC = REPO / "nutrition_table_fine_tuning" / "src"
FT_PROMPTS = FT_SRC / "prompts.py"
INF_PROMPTS = REPO / "nutrition_table_inference" / "src" / "prompts.py"

sys.path.insert(0, str(FT_SRC))
import prompts  # noqa: E402

datasets_available = True
try:  # data_utils pulls in torch + datasets, which CI may not install
    sys.path.insert(0, str(FT_SRC / "dataset"))
    import data_utils  # noqa: E402
except Exception:  # pragma: no cover
    datasets_available = False

needs_datasets = pytest.mark.skipif(
    not datasets_available, reason="torch/datasets not installed"
)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def test_prompt_copies_are_identical_in_both_subprojects():
    assert filecmp.cmp(FT_PROMPTS, INF_PROMPTS, shallow=False), (
        "prompts.py has diverged between subprojects; copy one over the other."
    )


def test_system_message_has_no_leading_whitespace_on_continuation_lines():
    """Indenting the literal would change the string the model sees."""
    for line in prompts.SYSTEM_MESSAGE.splitlines():
        assert line == line.lstrip(), f"leading whitespace: {line!r}"


def test_system_message_is_three_lines():
    assert len(prompts.SYSTEM_MESSAGE.splitlines()) == 3


def test_task_prompt_is_the_plural_training_phrasing():
    assert prompts.TASK_PROMPT == (
        "Detect the bounding boxes of all nutrition tables in the image."
    )


def test_legacy_prompt_is_distinct_from_the_canonical_one():
    assert prompts.LEGACY_TASK_PROMPT != prompts.TASK_PROMPT


# --------------------------------------------------------------------------
# parse_boxes_from_text
# --------------------------------------------------------------------------

@needs_datasets
def test_parses_a_single_box():
    text = (
        "<|object_ref_start|>nutrition-table<|object_ref_end|>"
        "<|box_start|>(100, 200),(300, 400)<|box_end|>"
    )
    assert data_utils.parse_boxes_from_text(text) == [[100.0, 200.0, 300.0, 400.0]]


@needs_datasets
def test_parses_multiple_boxes():
    text = "(1, 2),(3, 4) (10, 20),(30, 40)"
    assert data_utils.parse_boxes_from_text(text) == [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ]


@needs_datasets
def test_no_table_found_parses_to_no_boxes():
    assert data_utils.parse_boxes_from_text("No table found.") == []


@needs_datasets
def test_empty_and_none_input_are_safe():
    assert data_utils.parse_boxes_from_text("") == []
    assert data_utils.parse_boxes_from_text(None) == []


@needs_datasets
def test_tolerates_whitespace_between_coordinate_pairs():
    """The generator emits '),(' but a model may emit '), ('."""
    assert data_utils.parse_boxes_from_text("(1, 2), (3, 4)") == [[1.0, 2.0, 3.0, 4.0]]


# --------------------------------------------------------------------------
# format_data: coordinate conversion and round trip
# --------------------------------------------------------------------------

def _sample(bboxes):
    return {"image": object(), "objects": {"bbox": bboxes}}


@needs_datasets
def test_converts_yxyx_normalised_to_xyxy_thousandths():
    # Dataset order is (y_min, x_min, y_max, x_max) in [0, 1].
    out = data_utils.format_data(_sample([[0.1, 0.2, 0.3, 0.4]]))
    assert "(200, 100),(400, 300)" in out["messages"][-1]["content"]


@needs_datasets
def test_round_trip_encode_then_parse():
    boxes = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.9, 0.9]]
    text = data_utils.format_data(_sample(boxes))["messages"][-1]["content"]
    assert data_utils.parse_boxes_from_text(text) == [
        [200.0, 100.0, 400.0, 300.0],
        [500.0, 500.0, 900.0, 900.0],
    ]


@needs_datasets
def test_empty_annotation_produces_the_no_table_sentinel():
    out = data_utils.format_data(_sample([]))
    assert out["messages"][-1]["content"] == "No table found."


@needs_datasets
def test_format_data_uses_the_canonical_prompt_by_default():
    out = data_utils.format_data(_sample([]))
    user_text = [c for c in out["messages"][1]["content"] if c["type"] == "text"][0]
    assert user_text["text"] == prompts.TASK_PROMPT


@needs_datasets
def test_config_can_override_the_task_prompt():
    out = data_utils.format_data(_sample([]), cfg={"task_prompt": "custom"})
    user_text = [c for c in out["messages"][1]["content"] if c["type"] == "text"][0]
    assert user_text["text"] == "custom"


@needs_datasets
def test_system_role_carries_the_canonical_system_message():
    out = data_utils.format_data(_sample([]))
    assert out["messages"][0]["content"] == prompts.SYSTEM_MESSAGE


# --------------------------------------------------------------------------
# lazy dataset path
# --------------------------------------------------------------------------

@needs_datasets
def test_lazy_transform_matches_eager_formatting():
    """Exercises the with_transform path without downloading the real dataset."""
    hf_datasets = pytest.importorskip("datasets")

    rows = {
        "image": ["img-a", "img-b"],
        "objects": [{"bbox": [[0.1, 0.2, 0.3, 0.4]]}, {"bbox": []}],
    }
    ds = hf_datasets.Dataset.from_dict(rows).with_transform(
        data_utils._lazy_transform(None)
    )

    assert len(ds) == 2
    assert ds[0]["messages"][-1]["content"] == (
        data_utils.format_data({"image": "img-a", "objects": rows["objects"][0]})
        ["messages"][-1]["content"]
    )
    assert ds[1]["messages"][-1]["content"] == "No table found."


@needs_datasets
def test_seeding_is_not_an_import_side_effect():
    """set_seed must be callable, and importing the module must not have run it."""
    assert callable(data_utils.set_seed)
    import torch

    torch.manual_seed(1234)
    before = torch.randint(0, 10_000, (1,)).item()
    data_utils.set_seed(0)
    torch.manual_seed(1234)
    assert torch.randint(0, 10_000, (1,)).item() == before
