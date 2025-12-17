# model_utils
import os, json, re, torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
os.environ["PEFT_BACKEND"] = "HF"
from peft import LoraConfig, PeftModel
from accelerate import PartialState

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[]
)

def load_processor_fixed(model_id: str, cfg=None, min_pixels: int = 224*224, max_pixels: int = 900*28*28):
    if cfg is not None:
        min_pixels = cfg.get("min_pixels", min_pixels)
        max_pixels = cfg.get("max_pixels", max_pixels)
    return Qwen2VLProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)


def save(trainer, processor, training_args, metrics=None, tag=None):
    if trainer is not None:
        trainer.model.save_pretrained(training_args.output_dir)
    if metrics:
        suffix = f"_{tag}" if tag else ""
        metrics_path = os.path.join(training_args.output_dir, f"metrics{suffix}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved to {metrics_path}")
    print(f"✅ Adapters saved to {training_args.output_dir}")

def _detect_merged_dir(path):
    if path and os.path.isdir(os.path.join(path, "merged")):
        return os.path.join(path, "merged")
    return None

def load_model(model_id: str, dtype=torch.float32, quantized=False,
               use_adapters=False, from_dir=None, cfg=None):
    if not model_id:
        raise ValueError("model_id must be provided from YAML config")

    proj_root = os.path.dirname(os.path.dirname(__file__))
    local_models = os.path.join(proj_root, "models", "transformers")
    local_path = os.path.join(local_models, model_id.replace("/", "_"))
    model_src = local_path if os.path.isdir(local_path) else model_id

    merged = _detect_merged_dir(from_dir) if from_dir else None
    if merged:
        model_src = merged

    quant_cfg = None
    if quantized:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # ⚡ FlashAttention toggle
    use_flash = cfg.get("use_flash", False) if cfg else False

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_src,
        dtype=dtype,
        quantization_config=quant_cfg,
        device_map={"": PartialState().process_index},
        attn_implementation="flash_attention_2" if use_flash else "eager",
    )

    processor = load_processor_fixed(model_id=model_id, cfg=cfg)


    if merged or not use_adapters or not from_dir:
        return base, processor

    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    has_adapter = any(os.path.exists(os.path.join(from_dir, f)) for f in adapter_files)
    if not has_adapter:
        return base, processor

    model = PeftModel.from_pretrained(base, from_dir)
    return model, processor

# def merge_adapters_to_base(model_id: str, adapter_dir: str, dtype=torch.bfloat16):
#     base_with_adapters, _ = load_model(model_id=model_id, dtype=dtype,
#                                        use_adapters=True, from_dir=adapter_dir)
#     if not isinstance(base_with_adapters, PeftModel):
#         raise ValueError(f"No adapters to merge at {adapter_dir}")
#     merged = base_with_adapters.merge_and_unload()
#     out = os.path.join(adapter_dir, "merged")
#     os.makedirs(out, exist_ok=True)
#     merged.save_pretrained(out, safe_serialization=True)
#     print(f"✅ Merged model saved to {out}")
#     return out

def get_matching_modules(model, regex_list):
    names = [n for n, _ in model.named_modules()]
    out = set()
    for pat in regex_list:
        rx = re.compile(pat)
        for n in names:
            if rx.search(n):
                out.add(n)
    if not out:
        for pat in regex_list:
            for n in names:
                if n.endswith(pat):
                    out.add(n)
    if not out:
        sample = "\n".join(names[:30])
        raise ValueError(f"No modules matched {regex_list}. Sample:\n{sample}")
    return sorted(out)
