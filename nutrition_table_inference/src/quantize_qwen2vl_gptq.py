import os
import random
import shutil
import glob
import torch

from gptqmodel import GPTQModel, QuantizeConfig
from .dataset.data_utils import get_datasets

model_id = "/workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13"
quant_path = "/workspace/projects/nutrition_table/nutrition_table_inference/model/Qwen2-VL-7B/final_model_exp13_quantized_4bit"

# -------------------------
# 1) Build small calib set
# -------------------------
train, val = get_datasets(format_data_flag=True)

all_messages = [ex["messages"] for ex in train]
random.seed(0)
random.shuffle(all_messages)
calibration_dataset = all_messages[:4]

# -------------------------
# 2) Quant config
# -------------------------
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
    static_groups=False,
    sym=True,
    true_sequential=True,
)

# -------------------------
# 3) Load + quantize
# -------------------------
model = GPTQModel.load(
    model_id_or_path=model_id,
    quantize_config=quant_config,
    dtype=torch.float16,
)

model.quantize(calibration_dataset, batch_size=1)

# -------------------------
# 4) Save quantized weights
#    + copy HF metadata
# -------------------------
os.makedirs(quant_path, exist_ok=True)
model.save(quant_path)

# copy HF config/tokenizer files so vLLM can treat this as a full model dir
meta_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

for fname in meta_files:
    src = os.path.join(model_id, fname)
    if os.path.exists(src):
        shutil.copy2(src, quant_path)

# sentencepiece or other tokenizer model
for spm in glob.glob(os.path.join(model_id, "*.model")):
    shutil.copy2(spm, quant_path)

print("✅ Quantized model saved to:", quant_path)
