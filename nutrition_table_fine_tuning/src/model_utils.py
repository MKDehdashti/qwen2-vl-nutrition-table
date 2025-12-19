# model_utils.py
import os, json, re, torch
from accelerate import PartialState
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
os.environ["PEFT_BACKEND"] = "HF"
from peft import PeftModel


def load_processor_fixed(model_id: str, cfg=None, min_pixels: int = 224 * 224, max_pixels: int = 900 * 28 * 28):
    if cfg is not None:
        min_pixels = cfg.get("min_pixels", min_pixels)
        max_pixels = cfg.get("max_pixels", max_pixels)
    return Qwen2VLProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)


def save_adapters(trainer, training_args, metrics=None, tag=None):
    if trainer is not None:
        trainer.model.save_pretrained(training_args.output_dir)
    if metrics:
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(training_args.output_dir, f"metrics{suffix}.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)


def _attn_impl(cfg):
    return "flash_attention_2" if (cfg and cfg.get("use_flash", False)) else "eager"


def _quant_cfg(quantized: bool):
    if not quantized:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _device_map():
    return {"": PartialState().process_index}


def load_base(model_src: str, torch_dtype=torch.bfloat16, quantized=False, cfg=None):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. This project is GPU-only.")
    return Qwen2VLForConditionalGeneration.from_pretrained(
        model_src,
        torch_dtype=torch_dtype,
        quantization_config=_quant_cfg(quantized),
        device_map=_device_map(),
        attn_implementation=_attn_impl(cfg),
    )


def adapters_exist(adapter_dir: str):
    if not adapter_dir:
        return False
    if not os.path.isdir(adapter_dir):
        return False
    return (
        os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
        and (
            os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(adapter_dir, "adapter_model.bin"))
        )
    )


def resolve_merged_dir(path: str):
    if not path:
        return None
    cand = os.path.join(path, "merged")
    return cand if os.path.isdir(cand) else None


def load_base_plus_adapters(base_src: str, adapter_dir: str, torch_dtype=torch.bfloat16, quantized=False, cfg=None):
    base = load_base(base_src, torch_dtype=torch_dtype, quantized=quantized, cfg=cfg)
    if not adapters_exist(adapter_dir):
        raise ValueError(f"No adapters found in: {adapter_dir}")
    return PeftModel.from_pretrained(base, adapter_dir)


def merge_adapters_to_dir(base_src: str, adapter_dir: str, merged_dir: str, torch_dtype=torch.bfloat16, quantized=False, cfg=None):
    m = load_base_plus_adapters(base_src, adapter_dir, torch_dtype=torch_dtype, quantized=quantized, cfg=cfg)
    merged = m.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    return merged


def load_model(model_id: str, dtype=torch.bfloat16, quantized=False, use_adapters=False, from_dir=None, base_from_dir=None, cfg=None):
    processor = load_processor_fixed(model_id=model_id, cfg=cfg)

    base_src = model_id
    if base_from_dir:
        base_src = base_from_dir
    if from_dir and not use_adapters:
        md = resolve_merged_dir(from_dir)
        base_src = md or from_dir

    base = load_base(base_src, torch_dtype=dtype, quantized=quantized, cfg=cfg)

    if use_adapters and from_dir and adapters_exist(from_dir):
        return PeftModel.from_pretrained(base, from_dir), processor

    return base, processor


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
