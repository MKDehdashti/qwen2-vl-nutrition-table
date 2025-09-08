import os, glob, json, re
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from accelerate import PartialState

lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=[])

def save(trainer, processor, training_args, metrics=None, tag=None):
    if trainer is not None:
        trainer.model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if metrics:
        suffix = f"_{tag}" if tag else ""
        metrics_path = os.path.join(training_args.output_dir, f"metrics{suffix}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to {metrics_path}")
    print(f"✅ Saved to {training_args.output_dir}")

def _find_processor_src(base_out, adapter_path, base_id="Qwen/Qwen2-VL-7B-Instruct"):
    candidates = [base_out, base_id]
    config_names = ["preprocessor_config.json","processor_config.json","image_processor_config.json"]
    for c in candidates:
        for name in config_names:
            if os.path.isfile(os.path.join(c, name)):
                return c
    return base_id

def load_model(dtype=torch.float32, quantized=False, use_adapters=True, from_dir=None, training_args=None):
    # Figure out where to load from
    proj_root = os.path.dirname(os.path.dirname(__file__))
    local_models = os.path.join(proj_root, "models", "transformers")
    base_id = "Qwen/Qwen2-VL-7B-Instruct"

    # Check if local cache exists
    local_path = os.path.join(local_models, base_id.replace("/", "_"))
    model_src = local_path if os.path.isdir(local_path) else base_id

    # Adapter path (fine-tune checkpoints)
    base_out = None
    if from_dir:
        base_out = from_dir
    elif training_args is not None and hasattr(training_args, "output_dir"):
        base_out = training_args.output_dir

    adapter_path = None
    if base_out:
        ckpts = sorted(glob.glob(os.path.join(base_out, "checkpoint-*")), key=os.path.getmtime)
        adapter_path = ckpts[-1] if ckpts else base_out

    # Quantization config
    quant_cfg = None
    if quantized:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load base model
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_src,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map={"": PartialState().process_index},
    )

    # Processor source (prefer adapter path, fallback to base)
    proc_src = None
    if adapter_path:
        proc_src = _find_processor_src(base_out, adapter_path)
    if not proc_src:
        proc_src = model_src

    processor = Qwen2VLProcessor.from_pretrained(
        proc_src,
        min_pixels=224 * 224,
        max_pixels=900 * 28 * 28,
    )

    # Return base only if no adapters or adapters disabled
    if not adapter_path or not use_adapters:
        return base, processor

    # Load adapters if present
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    has_adapter = any(os.path.exists(os.path.join(adapter_path, f)) for f in adapter_files)
    if not has_adapter:
        return base, processor

    model = PeftModel.from_pretrained(base, adapter_path)
    return model, processor



def get_matching_modules(model, regex_list):
    names = [n for n, _ in model.named_modules()]
    out = set()

    import re
    # Regex search
    for pat in regex_list:
        rx = re.compile(pat)
        for n in names:
            if rx.search(n):
                out.add(n)

    # Fallback: suffix search
    if not out:
        for pat in regex_list:
            for n in names:
                if n.endswith(pat):
                    out.add(n)

    if not out:
        sample = "\n".join(names[:30])
        raise ValueError(f"No modules matched any of {regex_list}. Sample names:\n{sample}")

    return sorted(out)
