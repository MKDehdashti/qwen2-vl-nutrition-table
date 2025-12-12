import random
import torch
from gptqmodel import GPTQModel, QuantizeConfig
from dataset.data_utils import get_datasets

model_id = "/workspace/projects/nutrition-table3/model/Qwen2-VL-7B/final_model"
quant_path = "/workspace/projects/nutrition-table3/model/Qwen2-VL-7B/quantized_gptqmodel"

train, val = get_datasets(format_data_flag=True)

all_messages = [ex["messages"] for ex in train]
random.seed(0)
random.shuffle(all_messages)
calibration_dataset = all_messages[:4]

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
    static_groups=False,
    sym=True,
    true_sequential=True,
)

model = GPTQModel.load(
    model_id_or_path=model_id,
    quantize_config=quant_config,
    dtype=torch.bfloat16,
)

model.quantize(calibration_dataset, batch_size=1)
model.save(quant_path)
