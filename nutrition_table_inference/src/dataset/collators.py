import torch, re
from qwen_vl_utils import process_vision_info

def collate_fn(examples, processor, numeric_only=False):
    texts, images = [], []

    for ex in examples:
        messages = ex["messages"]
        text = processor.apply_chat_template(messages, tokenize=False)
        image_input, _ = process_vision_info(messages)
        texts.append(text)
        images.append(image_input)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    tok = processor.tokenizer

    # mask out special tokens
    special_tokens = {
        tok.pad_token_id,
        tok.convert_tokens_to_ids("<|vision_start|>"),
        tok.convert_tokens_to_ids("<|vision_end|>"),
        tok.convert_tokens_to_ids("<|image_pad|>")
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
