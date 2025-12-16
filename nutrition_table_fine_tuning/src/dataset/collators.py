# collators.py
import torch, re
from qwen_vl_utils import process_vision_info

def collate_fn(examples, processor, cfg=None, numeric_only=False):
    cfg = cfg or {}
    max_len = int(cfg.get("max_seq_length", 1024))
    pad_to_multiple_of = cfg.get("pad_to_multiple_of", None)
    mask_prompt_labels = bool(cfg.get("mask_prompt_labels", False))

    texts, images = [], []
    prompt_lens = [] if mask_prompt_labels else None
    tok = processor.tokenizer

    for ex in examples:
        messages = ex["messages"]
        full_text = processor.apply_chat_template(messages, tokenize=False)
        image_input, _ = process_vision_info(messages)

        texts.append(full_text)
        images.append(image_input)

        if mask_prompt_labels:
            prompt_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            prompt_ids = tok(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_len,
            )["input_ids"]
            prompt_lens.append(len(prompt_ids))

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        pad_to_multiple_of=pad_to_multiple_of,
        add_special_tokens=False,
    )

    labels = batch["input_ids"].clone()

    if mask_prompt_labels:
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, labels.size(1))
            labels[i, :plen] = -100

    special_tokens = {
        tok.pad_token_id,
        tok.convert_tokens_to_ids("<|vision_start|>"),
        tok.convert_tokens_to_ids("<|vision_end|>"),
        tok.convert_tokens_to_ids("<|image_pad|>"),
    }
    for tid in special_tokens:
        if tid is not None:
            labels[labels == tid] = -100

    if numeric_only:
        numeric_ids = [tid for tok_str, tid in tok.get_vocab().items() if re.fullmatch(r"[0-9]+", tok_str)]
        numeric_ids = torch.tensor(numeric_ids, device=labels.device)
        for i in range(labels.size(0)):
            row = labels[i]
            non_numeric_mask = ~torch.isin(row, numeric_ids)
            row[non_numeric_mask & (row != -100)] = -100
            labels[i] = row

    batch["labels"] = labels
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    return batch
