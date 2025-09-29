import torch, re
from qwen_vl_utils import process_vision_info
from dataset_flash_format.data_utils import build_messages

def collate_fn(examples, processor, numeric_only: bool = False):
    full_msgs = [build_messages(e, with_answer=True) for e in examples]
    pref_msgs = [build_messages(e, with_answer=False) for e in examples]

    full_txts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in full_msgs
    ]
    pref_txts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in pref_msgs
    ]

    image_inputs = [process_vision_info(m)[0] for m in full_msgs]

    batch = processor(text=full_txts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    tok = processor.tokenizer
    for i, pref in enumerate(pref_txts):
        # tokenize prefix (system+user, no assistant)
        pref_ids = tok(pref, add_special_tokens=False).input_ids
        cut = len(pref_ids)
        # mask prefix
        labels[i, :cut] = -100

    # mask vision-related special tokens
    special = {
        tok.pad_token_id,
        tok.convert_tokens_to_ids("<|vision_start|>"),
        tok.convert_tokens_to_ids("<|vision_end|>"),
        tok.convert_tokens_to_ids("<|image_pad|>")
    }
    for tid in special:
        if tid is not None:
            labels[labels == tid] = -100

    # optional: numeric-only training
    if numeric_only:
        numeric_token_ids = [
            tid for tok_str, tid in tok.get_vocab().items()
            if re.fullmatch(r"[0-9]+", tok_str)
        ]
        numeric_token_ids = torch.tensor(numeric_token_ids, device=labels.device)
        for i in range(labels.size(0)):
            row = labels[i]
            non_numeric_mask = ~torch.isin(row, numeric_token_ids)
            row[non_numeric_mask & (row != -100)] = -100
            labels[i] = row

    batch["labels"] = labels
    return batch
