"""Canonical prompts. Single source of truth for training and inference.

The model was fine-tuned with `SYSTEM_MESSAGE` and `TASK_PROMPT` below. Inference
should use the same strings; anything else is an untested configuration.

Duplicated verbatim into both subprojects, which deploy independently;
`tests/test_prompts.py` asserts the copies stay identical.

Do not re-indent SYSTEM_MESSAGE. It is a triple-quoted literal at module level,
so leading whitespace on the continuation lines becomes part of the string the
model sees.
"""

SYSTEM_MESSAGE = """You are a Vision Language Model specialized in interpreting visual data from product images.
Your task is to analyze the provided product images and detect the nutrition tables in a certain format.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

TASK_PROMPT = "Detect the bounding boxes of all nutrition tables in the image."

# Superseded strings, kept only so their provenance is traceable.
#
# LEGACY_TASK_PROMPT was the default in several inference scripts while the
# training config used TASK_PROMPT. Measured accuracy was near-identical
# (mean IoU 0.82 either way), but the mismatch was unintentional.
LEGACY_TASK_PROMPT = "Detect the bounding box of the nutrition table."

# PLACEHOLDER_SYSTEM_TEXT is the two-word string that inference_eval.py and
# vllm_throughput.py passed as the system prompt. Every benchmark JSON committed
# under nutrition_table_inference/outputs/ was produced with this, NOT with
# SYSTEM_MESSAGE. The effect of the difference has not been measured cleanly:
# the one run that used the real SYSTEM_MESSAGE also changed the output-format
# instruction at the same time, so the two effects are confounded.
PLACEHOLDER_SYSTEM_TEXT = "System message"
