# Commands 
testing 

source /workspace/projects/nutrition-table/.venv/bin/activate

 pip install -r projects/nutrition-table/requirements.txt

 accelerate launch --mixed_precision=bf16 --multi_gpu projects/nutrition-table/src/train.py
 accelerate launch --mixed_precision=bf16 /workspace/projects/nutrition-table/src/train.py --config configs/exp1.yaml
  accelerate launch --mixed_precision=bf16 src/train.py --config configs/exp1.yaml


 python -u src/train.py --config configs/exp1_debug.yaml
 python -u src/train.py --config con(.venv) root@8791c11f9ca0:/workspace#  accelerate launch --mixed_precision=bf16 /workspace/projects/nutrition-table/src/train.py --config configs/exp1.yaml
The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_processes` was set to a value of `1`
        `--num_machines` was set to a value of `1`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1083/1083 [00:02<00:00, 369.67 examples/s]
Generating val split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 123/123 [00:00<00:00, 505.00 examples/s]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1083/1083 [00:06<00:00, 174.63 examples/s]
Generating val split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 123/123 [00:00<00:00, 501.32 examples/s]
wandb: Using wandb-core as the SDK backend. Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: maryamkd8 (maryamdehdashti). Use `wandb login --relogin` to force relogin
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
⚡ GPU available — running on CUDA
wandb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
wandb: Tracking run with wandb version 0.18.5
wandb: Run data is saved locally in /workspace/wandb/run-20250906_234947-pop60wyc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run meta_exp1_20250906_234947
wandb: ⭐️ View project at https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: 🚀 View run at https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/pop60wyc
wandb:                                                                                
wandb: 🚀 View run meta_exp1_20250906_234947 at: https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/pop60wyc
wandb: ⭐️ View project at: https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250906_234947-pop60wyc/logs
wandb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
wandb: Tracking run with wandb version 0.18.5
wandb: Run data is saved locally in /workspace/wandb/run-20250906_234951-13u8um54
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run baseline_20250906_234947
wandb: ⭐️ View project at https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: 🚀 View run at https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/13u8um54

=== Baseline Evaluation ===
`torch_dtype` is deprecated! Use `dtype` instead!
config.json: 1.20kB [00:00, 5.47MB/s]
model.safetensors.index.json: 56.5kB [00:00, 148MB/s]
model-00005-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.09G/1.09G [00:09<00:00, 120MB/s]
model-00004-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:42<00:00, 90.1MB/s]
model-00001-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.90G/3.90G [00:43<00:00, 89.2MB/s]
model-00002-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:44<00:00, 87.3MB/s]
model-00003-of-00005.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:46<00:00, 83.6MB/s]
Fetching 5 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:46<00:00,  9.39s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:13<00:00,  2.74s/it]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 244/244 [00:00<00:00, 1.35MB/s]
preprocessor_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 347/347 [00:00<00:00, 1.27MB/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.56it/s]
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. Note that this behavior will be extended to all models in a future release.
tokenizer_config.json: 4.19kB [00:00, 9.38MB/s]
vocab.json: 2.78MB [00:00, 98.1MB/s]
merges.txt: 1.67MB [00:00, 160MB/s]
tokenizer.json: 7.03MB [00:00, 250MB/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 9341.43it/s]
chat_template.json: 1.05kB [00:00, 5.19MB/s]
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Mean IoU (strict): 0.3309
Precision@0.5: 0.2400
Recall@0.5: 0.2182
F1@0.5: 0.2286
Precision@0.75: 0.0600
Recall@0.75: 0.0545
F1@0.75: 0.0571
mAP over [0.5, 0.75]: 0.0278
📊 Precision/Recall curve saved to ./outputs/precision_recall_curve_strict_baseline.png
📊 baseline Metrics: {'mean_iou': 0.3309273260831833, 'precision@0.5': 0.24, 'recall@0.5': 0.21818181818181817, 'f1@0.5': 0.2285714285714286, 'precision@0.75': 0.06, 'recall@0.75': 0.05454545454545454, 'f1@0.75': 0.05714285714285715, 'mAP': 0.02781818181818182, 'pr_curve_path': './outputs/precision_recall_curve_strict_baseline.png'}
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.22s/it]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 8289.14it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 6543.38it/s]
{'answer': '(0,196),(982,805)<|im_end|>',
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=819x650 at 0x7C6F3EBEAAD0>,
 'objects': {'bbox': [[0, 196, 982, 805]]}}
✅ Saved /workspace/projects/nutrition-table/runs/exp1_20250906_234947/bbox_baseline_val_20.png
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:         baseline_f1@0.5 ▁
wandb:        baseline_f1@0.75 ▁
wandb:            baseline_mAP ▁
wandb:       baseline_mean_iou ▁
wandb:  baseline_precision@0.5 ▁
wandb: baseline_precision@0.75 ▁
wandb:     baseline_recall@0.5 ▁
wandb:    baseline_recall@0.75 ▁
wandb: 
wandb: Run summary:
wandb:         baseline_f1@0.5 0.22857
wandb:        baseline_f1@0.75 0.05714
wandb:            baseline_mAP 0.02782
wandb:       baseline_mean_iou 0.33093
wandb:  baseline_precision@0.5 0.24
wandb: baseline_precision@0.75 0.06
wandb:     baseline_recall@0.5 0.21818
wandb:    baseline_recall@0.75 0.05455
wandb: 
wandb: 🚀 View run baseline_20250906_234947 at: https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/13u8um54
wandb: ⭐️ View project at: https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20250906_234951-13u8um54/logs
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.45s/it]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5489.93it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 13934.56it/s]
wandb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
wandb: Tracking run with wandb version 0.18.5
wandb: Run data is saved locally in /workspace/wandb/run-20250906_235244-n7es2r7s
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vision_20250906_234947
wandb: ⭐️ View project at https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: 🚀 View run at https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/n7es2r7s

=== Stage: vision ===
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.46s/it]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5745.62it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 11155.06it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5384.22it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 4563.99it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 8594.89it/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
  0%|                                                                                                                                                                            | 0/170 [00:00<?, ?it/s]/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
{'loss': 3.3426, 'grad_norm': 2.879091501235962, 'learning_rate': 4.908536585365854e-05, 'entropy': 2.972109481692314, 'num_tokens': 284483.0, 'mean_token_accuracy': 0.45124878697097304, 'epoch': 0.3} 
{'eval_loss': 3.2739064693450928, 'eval_runtime': 42.7866, 'eval_samples_per_second': 2.875, 'eval_steps_per_second': 0.374, 'eval_entropy': 2.885738492012024, 'eval_num_tokens': 284483.0, 'eval_mean_token_accuracy': 0.4692540653049946, 'epoch': 0.3}                                                                                                                                                        
  6%|█████████▍                                                                                                                                                       | 10/170 [04:52<1:06:16, 24.86s/itwandb: WARNING Tried to log to step 10 that is less than the current step 11. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 3.304, 'grad_norm': 4.643190860748291, 'learning_rate': 4.603658536585366e-05, 'entropy': 2.9744035452604294, 'num_tokens': 568143.0, 'mean_token_accuracy': 0.4606848552823067, 'epoch': 0.59} 
{'eval_loss': 3.2321479320526123, 'eval_runtime': 40.4637, 'eval_samples_per_second': 3.04, 'eval_steps_per_second': 0.395, 'eval_entropy': 2.8937732875347137, 'eval_num_tokens': 568143.0, 'eval_mean_token_accuracy': 0.4712121468037367, 'epoch': 0.59}                                                                                                                                                       
 12%|██████████████████▉                                                                                                                                              | 20/170 [09:40<1:03:06, 25.24s/itwandb: WARNING Tried to log to step 20 that is less than the current step 21. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 3.2586, 'grad_norm': 5.499569416046143, 'learning_rate': 4.298780487804878e-05, 'entropy': 2.9275563955307007, 'num_tokens': 852983.0, 'mean_token_accuracy': 0.45657205730676653, 'epoch': 0.89}
{'eval_loss': 3.1777873039245605, 'eval_runtime': 40.5107, 'eval_samples_per_second': 3.036, 'eval_steps_per_second': 0.395, 'eval_entropy': 2.8985914140939713, 'eval_num_tokens': 852983.0, 'eval_mean_token_accuracy': 0.47439802810549736, 'epoch': 0.89}                                                                                                                                                     
 18%|████████████████████████████▊                                                                                                                                      | 30/170 [14:23<57:47, 24.77s/itwandb: WARNING Tried to log to step 30 that is less than the current step 31. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 3.2014, 'grad_norm': 1.666517734527588, 'learning_rate': 3.9939024390243905e-05, 'entropy': 2.985602421096609, 'num_tokens': 1133532.0, 'mean_token_accuracy': 0.46234093021742906, 'epoch': 1.18}
{'eval_loss': 3.1116139888763428, 'eval_runtime': 40.7838, 'eval_samples_per_second': 3.016, 'eval_steps_per_second': 0.392, 'eval_entropy': 2.899537906050682, 'eval_num_tokens': 1133532.0, 'eval_mean_token_accuracy': 0.47729002870619297, 'epoch': 1.18}                                                                                                                                                     
 24%|██████████████████████████████████████▎                                                                                                                            | 40/170 [19:03<53:40, 24.77s/itwandb: WARNING Tried to log to step 40 that is less than the current step 41. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 3.1102, 'grad_norm': 2.376655101776123, 'learning_rate': 3.6890243902439025e-05, 'entropy': 2.9712775349617004, 'num_tokens': 1416235.0, 'mean_token_accuracy': 0.45832641422748566, 'epoch': 1.47}
{'eval_loss': 3.0309271812438965, 'eval_runtime': 40.4485, 'eval_samples_per_second': 3.041, 'eval_steps_per_second': 0.396, 'eval_entropy': 2.8968042582273483, 'eval_num_tokens': 1416235.0, 'eval_mean_token_accuracy': 0.4782429300248623, 'epoch': 1.47}                                                                                                                                                     
 29%|███████████████████████████████████████████████▉                                                                                                                   | 50/170 [23:46<49:19, 24.66s/itwandb: WARNING Tried to log to step 50 that is less than the current step 51. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 3.0389, 'grad_norm': 1.5159759521484375, 'learning_rate': 3.384146341463415e-05, 'entropy': 2.992652577161789, 'num_tokens': 1698938.0, 'mean_token_accuracy': 0.4672131922096014, 'epoch': 1.77}
{'eval_loss': 2.936985492706299, 'eval_runtime': 40.8376, 'eval_samples_per_second': 3.012, 'eval_steps_per_second': 0.392, 'eval_entropy': 2.8939412981271744, 'eval_num_tokens': 1698938.0, 'eval_mean_token_accuracy': 0.4797574859112501, 'epoch': 1.77}                                                                                                                                                      
 35%|█████████████████████████████████████████████████████████▌                                                                                                         | 60/170 [28:29<45:07, 24.62s/itwandb: WARNING Tried to log to step 60 that is less than the current step 61. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.9302, 'grad_norm': 1.6873221397399902, 'learning_rate': 3.079268292682927e-05, 'entropy': 2.9601052893867976, 'num_tokens': 1981627.0, 'mean_token_accuracy': 0.468706800213343, 'epoch': 2.06}
{'eval_loss': 2.8256654739379883, 'eval_runtime': 40.6293, 'eval_samples_per_second': 3.027, 'eval_steps_per_second': 0.394, 'eval_entropy': 2.8921104818582535, 'eval_num_tokens': 1981627.0, 'eval_mean_token_accuracy': 0.4846924487501383, 'epoch': 2.06}                                                                                                                                                     
 41%|███████████████████████████████████████████████████████████████████                                                                                                | 70/170 [33:11<40:18, 24.18s/itwandb: WARNING Tried to log to step 70 that is less than the current step 71. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.7945, 'grad_norm': 2.0134410858154297, 'learning_rate': 2.7743902439024393e-05, 'entropy': 2.9557969003915785, 'num_tokens': 2265212.0, 'mean_token_accuracy': 0.4754586800932884, 'epoch': 2.35}
{'eval_loss': 2.7063043117523193, 'eval_runtime': 40.022, 'eval_samples_per_second': 3.073, 'eval_steps_per_second': 0.4, 'eval_entropy': 2.875041499733925, 'eval_num_tokens': 2265212.0, 'eval_mean_token_accuracy': 0.49072819016873837, 'epoch': 2.35}                                                                                                                                                        
 47%|████████████████████████████████████████████████████████████████████████████▋                                                                                      | 80/170 [37:52<37:19, 24.88s/itwandb: WARNING Tried to log to step 80 that is less than the current step 81. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.6721, 'grad_norm': 2.2260327339172363, 'learning_rate': 2.4695121951219512e-05, 'entropy': 2.949573004245758, 'num_tokens': 2548772.0, 'mean_token_accuracy': 0.4867727655917406, 'epoch': 2.65}
{'eval_loss': 2.578946828842163, 'eval_runtime': 40.1334, 'eval_samples_per_second': 3.065, 'eval_steps_per_second': 0.399, 'eval_entropy': 2.8513737618923187, 'eval_num_tokens': 2548772.0, 'eval_mean_token_accuracy': 0.5060273595154285, 'epoch': 2.65}                                                                                                                                                      
 53%|██████████████████████████████████████████████████████████████████████████████████████▎                                                                            | 90/170 [42:34<32:51, 24.65s/itwandb: WARNING Tried to log to step 90 that is less than the current step 91. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.5382, 'grad_norm': 1.8184367418289185, 'learning_rate': 2.1646341463414634e-05, 'entropy': 2.931483393907547, 'num_tokens': 2833692.0, 'mean_token_accuracy': 0.49595569297671316, 'epoch': 2.94}
{'eval_loss': 2.440950632095337, 'eval_runtime': 40.1915, 'eval_samples_per_second': 3.06, 'eval_steps_per_second': 0.398, 'eval_entropy': 2.843589887022972, 'eval_num_tokens': 2833692.0, 'eval_mean_token_accuracy': 0.5147630367428064, 'epoch': 2.94}                                                                                                                                                        
 59%|███████████████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 100/170 [47:17<28:57, 24.83s/itwandb: WARNING Tried to log to step 100 that is less than the current step 101. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
{'loss': 2.4007, 'grad_norm': 2.305936098098755, 'learning_rate': 1.8597560975609757e-05, 'entropy': 2.920383311525176, 'num_tokens': 3114922.0, 'mean_token_accuracy': 0.5125821047945868, 'epoch': 3.24}
{'eval_loss': 2.2932491302490234, 'eval_runtime': 40.0886, 'eval_samples_per_second': 3.068, 'eval_steps_per_second': 0.399, 'eval_entropy': 2.870581865310669, 'eval_num_tokens': 3114922.0, 'eval_mean_token_accuracy': 0.5272507220506668, 'epoch': 3.24}                                                                                                                                                      
 65%|████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                         | 110/170 [51:59<24:58, 24.97s/itwandb: WARNING Tried to log to step 110 that is less than the current step 111. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.2297, 'grad_norm': 2.4708642959594727, 'learning_rate': 1.554878048780488e-05, 'entropy': 2.9499155819416045, 'num_tokens': 3401230.0, 'mean_token_accuracy': 0.5184951700270176, 'epoch': 3.53}
{'eval_loss': 2.1592981815338135, 'eval_runtime': 40.1056, 'eval_samples_per_second': 3.067, 'eval_steps_per_second': 0.399, 'eval_entropy': 2.8952095359563828, 'eval_num_tokens': 3401230.0, 'eval_mean_token_accuracy': 0.5369326025247574, 'epoch': 3.53}                                                                                                                                                     
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                               | 120/170 [56:43<20:59, 25.19s/itwandb: WARNING Tried to log to step 120 that is less than the current step 121. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.1113, 'grad_norm': 2.1806273460388184, 'learning_rate': 1.25e-05, 'entropy': 2.969839358329773, 'num_tokens': 3682255.0, 'mean_token_accuracy': 0.5319751758128405, 'epoch': 3.83}            
{'eval_loss': 2.042262315750122, 'eval_runtime': 41.0528, 'eval_samples_per_second': 2.996, 'eval_steps_per_second': 0.39, 'eval_entropy': 2.920460432767868, 'eval_num_tokens': 3682255.0, 'eval_mean_token_accuracy': 0.5545112080872059, 'epoch': 3.83}                                                                                                                                                        
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 130/170 [1:01:24<16:12, 24.32s/itwandb: WARNING Tried to log to step 130 that is less than the current step 131. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.9785, 'grad_norm': 1.9203184843063354, 'learning_rate': 9.451219512195123e-06, 'entropy': 2.9864272346979455, 'num_tokens': 3961598.0, 'mean_token_accuracy': 0.5569248765329772, 'epoch': 4.12}
{'eval_loss': 1.9630459547042847, 'eval_runtime': 40.267, 'eval_samples_per_second': 3.055, 'eval_steps_per_second': 0.397, 'eval_entropy': 2.934274345636368, 'eval_num_tokens': 3961598.0, 'eval_mean_token_accuracy': 0.5779813788831234, 'epoch': 4.12}                                                                                                                                                       
 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 140/170 [1:06:02<12:04, 24.15s/itwandb: WARNING Tried to log to step 140 that is less than the current step 141. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.9397, 'grad_norm': 1.544498324394226, 'learning_rate': 6.402439024390244e-06, 'entropy': 2.9976709723472594, 'num_tokens': 4244589.0, 'mean_token_accuracy': 0.5735817566514015, 'epoch': 4.41}
{'eval_loss': 1.9179624319076538, 'eval_runtime': 40.3737, 'eval_samples_per_second': 3.047, 'eval_steps_per_second': 0.396, 'eval_entropy': 2.9554570019245148, 'eval_num_tokens': 4244589.0, 'eval_mean_token_accuracy': 0.5912522450089455, 'epoch': 4.41}                                                                                                                                                     
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                  | 150/170 [1:10:45<08:09, 24.49s/itwandb: WARNING Tried to log to step 150 that is less than the current step 151. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.8804, 'grad_norm': 1.5727089643478394, 'learning_rate': 3.3536585365853664e-06, 'entropy': 3.021562287211418, 'num_tokens': 4529872.0, 'mean_token_accuracy': 0.5819739528000355, 'epoch': 4.71}
{'eval_loss': 1.894426941871643, 'eval_runtime': 41.2224, 'eval_samples_per_second': 2.984, 'eval_steps_per_second': 0.388, 'eval_entropy': 2.9661597907543182, 'eval_num_tokens': 4529872.0, 'eval_mean_token_accuracy': 0.5962310247123241, 'epoch': 4.71}                                                                                                                                                      
 94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌         | 160/170 [1:15:31<04:10, 25.05s/itwandb: WARNING Tried to log to step 160 that is less than the current step 161. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.842, 'grad_norm': 1.6463017463684082, 'learning_rate': 3.0487804878048784e-07, 'entropy': 3.032598091077201, 'num_tokens': 4810050.0, 'mean_token_accuracy': 0.5856489554236207, 'epoch': 5.0}
{'eval_loss': 1.8870514631271362, 'eval_runtime': 40.2305, 'eval_samples_per_second': 3.057, 'eval_steps_per_second': 0.398, 'eval_entropy': 2.964444160461426, 'eval_num_tokens': 4810050.0, 'eval_mean_token_accuracy': 0.5977507531642914, 'epoch': 5.0}                                                                                                                                                       
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 170/170 [1:20:11<00:00, 23.55s/itwandb: WARNING Tried to log to step 170 that is less than the current step 171. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'train_runtime': 4813.178, 'train_samples_per_second': 1.125, 'train_steps_per_second': 0.035, 'train_loss': 2.621931423860438, 'epoch': 5.0}                                                           
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 170/170 [1:20:13<00:00, 28.31s/it]
✅ Saved to /workspace/projects/nutrition-table/runs/exp1_20250906_234947/vision
Loading checkpoint shards:  40%|███████████████████████████████████████████████████████▌                                                                                   | 2/5 [00:05<00:08,  2.72s/it]wandb: WARNING Tried to log to step 170 that is less than the current step 172. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.41s/it]
Mean IoU (strict): 0.3386
Precision@0.5: 0.2200
Recall@0.5: 0.2000
F1@0.5: 0.2095
Precision@0.75: 0.0800
Recall@0.75: 0.0727
F1@0.75: 0.0762
mAP over [0.5, 0.75]: 0.0249
📊 Precision/Recall curve saved to /workspace/projects/nutrition-table/runs/exp1_20250906_234947/vision/precision_recall_curve_strict_post_vision.png
📊 post_vision Metrics: {'mean_iou': 0.3386168011277914, 'precision@0.5': 0.22, 'recall@0.5': 0.2, 'f1@0.5': 0.20952380952380953, 'precision@0.75': 0.08, 'recall@0.75': 0.07272727272727272, 'f1@0.75': 0.0761904761904762, 'mAP': 0.024909090909090912, 'pr_curve_path': '/workspace/projects/nutrition-table/runs/exp1_20250906_234947/vision/precision_recall_curve_strict_post_vision.png'}
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.49s/it]
{'answer': '(10,10),(983,983)<|im_end|>',
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=819x650 at 0x7C6F2FF3C340>,
 'objects': {'bbox': [[10, 10, 983, 983]]}}
✅ Saved /workspace/projects/nutrition-table/runs/exp1_20250906_234947/vision/bbox_post_vision_val_20.png
🧹 Cleaning caches (mode: normal)...
WARNING: No matching packages
Files removed: 0 (0 bytes)
✅ Cleanup done. Current disk usage:
Filesystem      Size  Used Avail Use% Mounted on
overlay         200G  243M  200G   1% /
Filesystem                Size  Used Avail Use% Mounted on
mfs#euro.runpod.net:9421  2.0P  1.7P  298T  85% /workspace
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                    entropy ▄▄▁▅▄▆▃▃▃▂▁▃▄▅▆▇█
wandb:                      epoch ▁▁▂▂▃▃▄▄▅▅▅▆▆▇▇██
wandb:               eval/entropy ▃▄▄▄▄▄▄▃▁▁▃▄▅▆▇██
wandb:                  eval/loss ███▇▇▆▆▅▄▄▃▂▂▁▁▁▁
wandb:   eval/mean_token_accuracy ▁▁▁▁▁▂▂▂▃▃▄▅▆▇███
wandb:            eval/num_tokens ▁▁▂▂▃▃▄▄▅▅▅▆▆▇▇██
wandb:               eval/runtime █▂▂▃▂▃▃▁▁▁▁▁▄▂▂▄▂
wandb:    eval/samples_per_second ▁▇▇▆▇▆▆█████▅▇▇▅▇
wandb:      eval/steps_per_second ▁▇▇▆▇▆▆██▇██▅▇▇▅▇
wandb:                  grad_norm ▃▆█▁▃▁▁▂▂▂▂▃▂▂▁▁▁
wandb:              learning_rate ██▇▇▆▆▅▅▄▄▄▃▃▂▂▁▁
wandb:                       loss ███▇▇▇▆▅▅▄▄▃▂▂▁▁▁
wandb:        mean_token_accuracy ▁▁▁▂▁▂▂▂▃▃▄▅▅▇▇██
wandb:                 num_tokens ▁▁▂▂▃▃▄▄▅▅▅▆▆▇▇██
wandb:         post_vision_f1@0.5 ▁
wandb:        post_vision_f1@0.75 ▁
wandb:            post_vision_mAP ▁
wandb:       post_vision_mean_iou ▁
wandb:  post_vision_precision@0.5 ▁
wandb: post_vision_precision@0.75 ▁
wandb:     post_vision_recall@0.5 ▁
wandb:    post_vision_recall@0.75 ▁
wandb:              train/entropy ▄▄▁▅▄▆▃▃▃▂▁▃▄▅▆▇█
wandb:                train/epoch ▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇█████
wandb:          train/global_step ▁▁▁▁▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇██████
wandb:            train/grad_norm ▃▆█▁▃▁▁▂▂▂▂▃▂▂▁▁▁
wandb:        train/learning_rate ██▇▇▆▆▅▅▄▄▄▃▃▂▂▁▁
wandb:                 train/loss ███▇▇▇▆▅▅▄▄▃▂▂▁▁▁
wandb:  train/mean_token_accuracy ▁▁▁▂▁▂▂▂▃▃▄▅▅▇▇██
wandb:           train/num_tokens ▁▁▂▂▃▃▄▄▅▅▅▆▆▇▇██
wandb: 
wandb: Run summary:
wandb:                    entropy 3.0326
wandb:                      epoch 5
wandb:               eval/entropy 2.96444
wandb:                  eval/loss 1.88705
wandb:   eval/mean_token_accuracy 0.59775
wandb:            eval/num_tokens 4810050.0
wandb:               eval/runtime 40.2305
wandb:    eval/samples_per_second 3.057
wandb:      eval/steps_per_second 0.398
wandb:                  grad_norm 1.6463
wandb:              learning_rate 0.0
wandb:                       loss 1.842
wandb:        mean_token_accuracy 0.58565
wandb:                 num_tokens 4810050.0
wandb:         post_vision_f1@0.5 0.20952
wandb:        post_vision_f1@0.75 0.07619
wandb:            post_vision_mAP 0.02491
wandb:       post_vision_mean_iou 0.33862
wandb:  post_vision_precision@0.5 0.22
wandb: post_vision_precision@0.75 0.08
wandb:     post_vision_recall@0.5 0.2
wandb:    post_vision_recall@0.75 0.07273
wandb:                 total_flos 2.3231104536250368e+17
wandb:              train/entropy 3.0326
wandb:                train/epoch 5
wandb:          train/global_step 170
wandb:            train/grad_norm 1.6463
wandb:        train/learning_rate 0.0
wandb:                 train/loss 1.842
wandb:  train/mean_token_accuracy 0.58565
wandb:           train/num_tokens 4810050.0
wandb:                 train_loss 2.62193
wandb:              train_runtime 4813.178
wandb:   train_samples_per_second 1.125
wandb:     train_steps_per_second 0.035
wandb: 
wandb: 🚀 View run vision_20250906_234947 at: https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/n7es2r7s
wandb: ⭐️ View project at: https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: Synced 2 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20250906_235244-n7es2r7s/logs
config.json: 1.20kB [00:00, 3.86MB/s]
model.safetensors.index.json: 56.5kB [00:00, 97.6MB/s]
model-00005-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.09G/1.09G [00:04<00:00, 264MB/s]
model-00003-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:09<00:00, 427MB/s]
model-00001-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.90G/3.90G [00:15<00:00, 250MB/s]
model-00004-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:18<00:00, 207MB/s]
model-00002-of-00005.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.86G/3.86G [00:19<00:00, 198MB/s]
Fetching 5 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:19<00:00,  3.99s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.46s/it]
generation_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 244/244 [00:00<00:00, 1.11MB/s]
preprocessor_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 347/347 [00:00<00:00, 3.12MB/s]
Fetching 1 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.54it/s]
tokenizer_config.json: 4.19kB [00:00, 14.6MB/s]
vocab.json: 2.78MB [00:00, 10.9MB/s]
merges.txt: 1.67MB [00:00, 71.9MB/s]
tokenizer.json: 7.03MB [00:00, 110MB/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5315.97it/s]
chat_template.json: 1.05kB [00:00, 4.15MB/s]
wandb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
wandb: Tracking run with wandb version 0.18.5
wandb: Run data is saved locally in /workspace/wandb/run-20250907_011526-w5w63hcd
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run lang_vision_20250906_234947
wandb: ⭐️ View project at https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: 🚀 View run at https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/w5w63hcd

=== Stage: lang_vision ===
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.47s/it]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5029.14it/s]
Fetching 1 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10866.07it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5121.25it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5322.72it/s]
Fetching 1 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 9177.91it/s]
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
  0%|                                                                                                                                                                            | 0/340 [00:00<?, ?it/s]/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
{'loss': 3.3131, 'grad_norm': 2.75140643119812, 'learning_rate': 8.181818181818183e-05, 'entropy': 2.989158508181572, 'num_tokens': 142497.0, 'mean_token_accuracy': 0.44711317457258704, 'epoch': 0.15} 
{'eval_loss': 3.0667598247528076, 'eval_runtime': 41.1877, 'eval_samples_per_second': 2.986, 'eval_steps_per_second': 0.388, 'eval_entropy': 2.898722290992737, 'eval_num_tokens': 142497.0, 'eval_mean_token_accuracy': 0.4868935514241457, 'epoch': 0.15}                                                                                                                                                       
  3%|████▋                                                                                                                                                            | 10/340 [02:58<1:15:38, 13.75s/itwandb: WARNING Tried to log to step 10 that is less than the current step 11. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 2.7181, 'grad_norm': 2.978142738342285, 'learning_rate': 9.756838905775076e-05, 'entropy': 3.015145680308342, 'num_tokens': 284483.0, 'mean_token_accuracy': 0.48199847228825093, 'epoch': 0.3} 
{'eval_loss': 2.315303087234497, 'eval_runtime': 40.7995, 'eval_samples_per_second': 3.015, 'eval_steps_per_second': 0.392, 'eval_entropy': 2.9903403967618942, 'eval_num_tokens': 284483.0, 'eval_mean_token_accuracy': 0.5508011765778065, 'epoch': 0.3}                                                                                                                                                        
  6%|█████████▍                                                                                                                                                       | 20/340 [05:53<1:14:59, 14.06s/itwandb: WARNING Tried to log to step 20 that is less than the current step 21. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.9427, 'grad_norm': 1.3584388494491577, 'learning_rate': 9.452887537993922e-05, 'entropy': 3.135740429162979, 'num_tokens': 425967.0, 'mean_token_accuracy': 0.5597153153270483, 'epoch': 0.44}
{'eval_loss': 1.6549015045166016, 'eval_runtime': 40.6979, 'eval_samples_per_second': 3.022, 'eval_steps_per_second': 0.393, 'eval_entropy': 3.10811410844326, 'eval_num_tokens': 425967.0, 'eval_mean_token_accuracy': 0.6427571885287762, 'epoch': 0.44}                                                                                                                                                        
  9%|██████████████▏                                                                                                                                                  | 30/340 [08:51<1:14:15, 14.37s/itwandb: WARNING Tried to log to step 30 that is less than the current step 31. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.4263, 'grad_norm': 1.5015547275543213, 'learning_rate': 9.148936170212766e-05, 'entropy': 3.254803869128227, 'num_tokens': 568143.0, 'mean_token_accuracy': 0.6552670493721962, 'epoch': 0.59}
{'eval_loss': 1.26664400100708, 'eval_runtime': 40.632, 'eval_samples_per_second': 3.027, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.218745544552803, 'eval_num_tokens': 568143.0, 'eval_mean_token_accuracy': 0.695042796432972, 'epoch': 0.59}                                                                                                                                                           
 12%|██████████████████▉                                                                                                                                              | 40/340 [11:47<1:11:00, 14.20s/itwandb: WARNING Tried to log to step 40 that is less than the current step 41. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 1.1271, 'grad_norm': 1.8881850242614746, 'learning_rate': 8.844984802431611e-05, 'entropy': 3.2828898251056673, 'num_tokens': 712114.0, 'mean_token_accuracy': 0.6794045515358448, 'epoch': 0.74}
{'eval_loss': 0.9267840385437012, 'eval_runtime': 41.4161, 'eval_samples_per_second': 2.97, 'eval_steps_per_second': 0.386, 'eval_entropy': 3.2941516041755676, 'eval_num_tokens': 712114.0, 'eval_mean_token_accuracy': 0.7248168662190437, 'epoch': 0.74}                                                                                                                                                       
 15%|███████████████████████▋                                                                                                                                         | 50/340 [14:46<1:08:53, 14.26s/itwandb: WARNING Tried to log to step 50 that is less than the current step 51. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.8567, 'grad_norm': 0.502830445766449, 'learning_rate': 8.541033434650457e-05, 'entropy': 3.347445419430733, 'num_tokens': 852983.0, 'mean_token_accuracy': 0.7106821320950985, 'epoch': 0.89} 
{'eval_loss': 0.7532498240470886, 'eval_runtime': 41.1526, 'eval_samples_per_second': 2.989, 'eval_steps_per_second': 0.389, 'eval_entropy': 3.2703305035829544, 'eval_num_tokens': 852983.0, 'eval_mean_token_accuracy': 0.7557225301861763, 'epoch': 0.89}                                                                                                                                                      
 18%|████████████████████████████▍                                                                                                                                    | 60/340 [17:41<1:05:31, 14.04s/itwandb: WARNING Tried to log to step 60 that is less than the current step 61. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.78, 'grad_norm': 0.9936000108718872, 'learning_rate': 8.237082066869302e-05, 'entropy': 3.316706758279067, 'num_tokens': 990447.0, 'mean_token_accuracy': 0.7231331994900336, 'epoch': 1.03}  
{'eval_loss': 0.7350167632102966, 'eval_runtime': 41.3984, 'eval_samples_per_second': 2.971, 'eval_steps_per_second': 0.386, 'eval_entropy': 3.213566243648529, 'eval_num_tokens': 990447.0, 'eval_mean_token_accuracy': 0.7605811096727848, 'epoch': 1.03}                                                                                                                                                       
 21%|█████████████████████████████████▌                                                                                                                                 | 70/340 [20:34<59:43, 13.27s/itwandb: WARNING Tried to log to step 70 that is less than the current step 71. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7707, 'grad_norm': 0.5357077121734619, 'learning_rate': 7.933130699088146e-05, 'entropy': 3.281961715221405, 'num_tokens': 1133532.0, 'mean_token_accuracy': 0.7239036463201046, 'epoch': 1.18}
{'eval_loss': 0.7258878350257874, 'eval_runtime': 40.4746, 'eval_samples_per_second': 3.039, 'eval_steps_per_second': 0.395, 'eval_entropy': 3.191753178834915, 'eval_num_tokens': 1133532.0, 'eval_mean_token_accuracy': 0.7656724862754345, 'epoch': 1.18}                                                                                                                                                      
 24%|█████████████████████████████████████▉                                                                                                                           | 80/340 [23:31<1:01:32, 14.20s/itwandb: WARNING Tried to log to step 80 that is less than the current step 81. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7503, 'grad_norm': 0.695688784122467, 'learning_rate': 7.629179331306992e-05, 'entropy': 3.289782699942589, 'num_tokens': 1275194.0, 'mean_token_accuracy': 0.7292420141398906, 'epoch': 1.32}
{'eval_loss': 0.7124602794647217, 'eval_runtime': 40.9525, 'eval_samples_per_second': 3.003, 'eval_steps_per_second': 0.391, 'eval_entropy': 3.1649158746004105, 'eval_num_tokens': 1275194.0, 'eval_mean_token_accuracy': 0.7704784460365772, 'epoch': 1.32}                                                                                                                                                     
 26%|███████████████████████████████████████████▏                                                                                                                       | 90/340 [26:27<58:25, 14.02s/itwandb: WARNING Tried to log to step 90 that is less than the current step 91. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7431, 'grad_norm': 0.6020016074180603, 'learning_rate': 7.325227963525836e-05, 'entropy': 3.193069672584534, 'num_tokens': 1416235.0, 'mean_token_accuracy': 0.7270415917038917, 'epoch': 1.47}
{'eval_loss': 0.7006155252456665, 'eval_runtime': 40.8236, 'eval_samples_per_second': 3.013, 'eval_steps_per_second': 0.392, 'eval_entropy': 3.1551317274570465, 'eval_num_tokens': 1416235.0, 'eval_mean_token_accuracy': 0.771600779145956, 'epoch': 1.47}                                                                                                                                                      
 29%|███████████████████████████████████████████████▋                                                                                                                  | 100/340 [29:23<55:43, 13.93s/it/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
wandb: WARNING Tried to log to step 100 that is less than the current step 101. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7407, 'grad_norm': 0.707417905330658, 'learning_rate': 7.021276595744681e-05, 'entropy': 3.2689485162496568, 'num_tokens': 1558203.0, 'mean_token_accuracy': 0.7233220480382443, 'epoch': 1.62}
{'eval_loss': 0.6857258677482605, 'eval_runtime': 40.7492, 'eval_samples_per_second': 3.018, 'eval_steps_per_second': 0.393, 'eval_entropy': 3.148435801267624, 'eval_num_tokens': 1558203.0, 'eval_mean_token_accuracy': 0.7715542912483215, 'epoch': 1.62}                                                                                                                                                      
 32%|████████████████████████████████████████████████████▍                                                                                                             | 110/340 [32:21<53:43, 14.01s/itwandb: WARNING Tried to log to step 110 that is less than the current step 111. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7254, 'grad_norm': 0.8443538546562195, 'learning_rate': 6.717325227963525e-05, 'entropy': 3.255138251185417, 'num_tokens': 1698938.0, 'mean_token_accuracy': 0.7368183694779873, 'epoch': 1.77}
{'eval_loss': 0.6480585932731628, 'eval_runtime': 40.7965, 'eval_samples_per_second': 3.015, 'eval_steps_per_second': 0.392, 'eval_entropy': 3.1363893300294876, 'eval_num_tokens': 1698938.0, 'eval_mean_token_accuracy': 0.7709990218281746, 'epoch': 1.77}                                                                                                                                                     
 35%|█████████████████████████████████████████████████████████▏                                                                                                        | 120/340 [35:16<50:53, 13.88s/itwandb: WARNING Tried to log to step 120 that is less than the current step 121. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.7066, 'grad_norm': 0.7419446706771851, 'learning_rate': 6.413373860182371e-05, 'entropy': 3.232494702935219, 'num_tokens': 1842039.0, 'mean_token_accuracy': 0.7315182991325855, 'epoch': 1.92}
{'eval_loss': 0.5907226204872131, 'eval_runtime': 40.5985, 'eval_samples_per_second': 3.03, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.139438673853874, 'eval_num_tokens': 1842039.0, 'eval_mean_token_accuracy': 0.7814257480204105, 'epoch': 1.92}                                                                                                                                                       
 38%|█████████████████████████████████████████████████████████████▉                                                                                                    | 130/340 [38:14<49:46, 14.22s/itwandb: WARNING Tried to log to step 130 that is less than the current step 131. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6773, 'grad_norm': 0.8460584878921509, 'learning_rate': 6.109422492401215e-05, 'entropy': 3.2065197626749673, 'num_tokens': 1981627.0, 'mean_token_accuracy': 0.7424829991964194, 'epoch': 2.06}
{'eval_loss': 0.5725424289703369, 'eval_runtime': 41.926, 'eval_samples_per_second': 2.934, 'eval_steps_per_second': 0.382, 'eval_entropy': 3.1350245028734207, 'eval_num_tokens': 1981627.0, 'eval_mean_token_accuracy': 0.7847996167838573, 'epoch': 2.06}                                                                                                                                                      
 41%|██████████████████████████████████████████████████████████████████▋                                                                                               | 140/340 [41:08<45:41, 13.71s/itwandb: WARNING Tried to log to step 140 that is less than the current step 141. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6788, 'grad_norm': 0.8024686574935913, 'learning_rate': 5.805471124620061e-05, 'entropy': 3.2382289737463, 'num_tokens': 2121236.0, 'mean_token_accuracy': 0.7377568237483502, 'epoch': 2.21} 
{'eval_loss': 0.5638158321380615, 'eval_runtime': 40.6431, 'eval_samples_per_second': 3.026, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.111741364002228, 'eval_num_tokens': 2121236.0, 'eval_mean_token_accuracy': 0.7826682813465595, 'epoch': 2.21}                                                                                                                                                      
 44%|███████████████████████████████████████████████████████████████████████▍                                                                                          | 150/340 [44:02<43:56, 13.88s/itwandb: WARNING Tried to log to step 150 that is less than the current step 151. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6762, 'grad_norm': 1.1241364479064941, 'learning_rate': 5.5015197568389065e-05, 'entropy': 3.167374235391617, 'num_tokens': 2265212.0, 'mean_token_accuracy': 0.7416379414498806, 'epoch': 2.35}
{'eval_loss': 0.5633772611618042, 'eval_runtime': 40.1992, 'eval_samples_per_second': 3.06, 'eval_steps_per_second': 0.398, 'eval_entropy': 3.120953992009163, 'eval_num_tokens': 2265212.0, 'eval_mean_token_accuracy': 0.7831445820629597, 'epoch': 2.35}                                                                                                                                                       
 47%|████████████████████████████████████████████████████████████████████████████▏                                                                                     | 160/340 [46:59<42:09, 14.05s/itwandb: WARNING Tried to log to step 160 that is less than the current step 161. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6447, 'grad_norm': 0.9492358565330505, 'learning_rate': 5.1975683890577506e-05, 'entropy': 3.186590752005577, 'num_tokens': 2407496.0, 'mean_token_accuracy': 0.7506570398807526, 'epoch': 2.5}
{'eval_loss': 0.5624677538871765, 'eval_runtime': 41.5651, 'eval_samples_per_second': 2.959, 'eval_steps_per_second': 0.385, 'eval_entropy': 3.10147362947464, 'eval_num_tokens': 2407496.0, 'eval_mean_token_accuracy': 0.7830365225672722, 'epoch': 2.5}                                                                                                                                                        
 50%|█████████████████████████████████████████████████████████████████████████████████                                                                                 | 170/340 [49:56<39:49, 14.06s/itwandb: WARNING Tried to log to step 170 that is less than the current step 171. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6637, 'grad_norm': 0.8383882641792297, 'learning_rate': 4.893617021276596e-05, 'entropy': 3.229828396439552, 'num_tokens': 2548772.0, 'mean_token_accuracy': 0.7442171990871429, 'epoch': 2.65}
{'eval_loss': 0.5604153871536255, 'eval_runtime': 40.402, 'eval_samples_per_second': 3.044, 'eval_steps_per_second': 0.396, 'eval_entropy': 3.107339069247246, 'eval_num_tokens': 2548772.0, 'eval_mean_token_accuracy': 0.788380891084671, 'epoch': 2.65}                                                                                                                                                        
 53%|█████████████████████████████████████████████████████████████████████████████████████▊                                                                            | 180/340 [52:52<37:20, 14.00s/itwandb: WARNING Tried to log to step 180 that is less than the current step 181. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6562, 'grad_norm': 0.9034678339958191, 'learning_rate': 4.589665653495441e-05, 'entropy': 3.2413204312324524, 'num_tokens': 2691724.0, 'mean_token_accuracy': 0.7501236632466316, 'epoch': 2.8}
{'eval_loss': 0.5593393445014954, 'eval_runtime': 40.0965, 'eval_samples_per_second': 3.068, 'eval_steps_per_second': 0.399, 'eval_entropy': 3.1020372062921524, 'eval_num_tokens': 2691724.0, 'eval_mean_token_accuracy': 0.788214985281229, 'epoch': 2.8}                                                                                                                                                       
 56%|██████████████████████████████████████████████████████████████████████████████████████████▌                                                                       | 190/340 [55:47<35:08, 14.05s/itwandb: WARNING Tried to log to step 190 that is less than the current step 191. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.642, 'grad_norm': 1.4442687034606934, 'learning_rate': 4.2857142857142856e-05, 'entropy': 3.130261868238449, 'num_tokens': 2833692.0, 'mean_token_accuracy': 0.7479047104716301, 'epoch': 2.94}
{'eval_loss': 0.5550841689109802, 'eval_runtime': 40.0708, 'eval_samples_per_second': 3.07, 'eval_steps_per_second': 0.399, 'eval_entropy': 3.0803584903478622, 'eval_num_tokens': 2833692.0, 'eval_mean_token_accuracy': 0.7881014011800289, 'epoch': 2.94}                                                                                                                                                      
 59%|███████████████████████████████████████████████████████████████████████████████████████████████▎                                                                  | 200/340 [58:42<32:53, 14.10s/it/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
wandb: WARNING Tried to log to step 200 that is less than the current step 201. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6338, 'grad_norm': 1.6002836227416992, 'learning_rate': 3.981762917933131e-05, 'entropy': 3.1692280035752516, 'num_tokens': 2971062.0, 'mean_token_accuracy': 0.755868823100359, 'epoch': 3.09}
{'eval_loss': 0.555669903755188, 'eval_runtime': 40.544, 'eval_samples_per_second': 3.034, 'eval_steps_per_second': 0.395, 'eval_entropy': 3.0659454464912415, 'eval_num_tokens': 2971062.0, 'eval_mean_token_accuracy': 0.7870619148015976, 'epoch': 3.09}                                                                                                                                                       
 62%|██████████████████████████████████████████████████████████████████████████████████████████████████▊                                                             | 210/340 [1:01:35<30:10, 13.93s/itwandb: WARNING Tried to log to step 210 that is less than the current step 211. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6243, 'grad_norm': 1.2885738611221313, 'learning_rate': 3.677811550151976e-05, 'entropy': 3.1345924377441405, 'num_tokens': 3114922.0, 'mean_token_accuracy': 0.7580911792814732, 'epoch': 3.24}
{'eval_loss': 0.5554279685020447, 'eval_runtime': 39.9942, 'eval_samples_per_second': 3.075, 'eval_steps_per_second': 0.4, 'eval_entropy': 3.0620559602975845, 'eval_num_tokens': 3114922.0, 'eval_mean_token_accuracy': 0.7889479845762253, 'epoch': 3.24}                                                                                                                                                       
 65%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                        | 220/340 [1:04:32<28:22, 14.19s/itwandb: WARNING Tried to log to step 220 that is less than the current step 221. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6627, 'grad_norm': 1.164707064628601, 'learning_rate': 3.373860182370821e-05, 'entropy': 3.1592579245567323, 'num_tokens': 3259365.0, 'mean_token_accuracy': 0.744897547364235, 'epoch': 3.38}
{'eval_loss': 0.5552865862846375, 'eval_runtime': 40.6635, 'eval_samples_per_second': 3.025, 'eval_steps_per_second': 0.393, 'eval_entropy': 3.055802136659622, 'eval_num_tokens': 3259365.0, 'eval_mean_token_accuracy': 0.7895961590111256, 'epoch': 3.38}                                                                                                                                                      
 68%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                   | 230/340 [1:07:30<26:16, 14.34s/itwandb: WARNING Tried to log to step 230 that is less than the current step 231. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6379, 'grad_norm': 1.3237168788909912, 'learning_rate': 3.069908814589666e-05, 'entropy': 3.1374263137578966, 'num_tokens': 3401230.0, 'mean_token_accuracy': 0.7553096510469913, 'epoch': 3.53}
{'eval_loss': 0.5524808764457703, 'eval_runtime': 40.3113, 'eval_samples_per_second': 3.051, 'eval_steps_per_second': 0.397, 'eval_entropy': 3.0540761500597, 'eval_num_tokens': 3401230.0, 'eval_mean_token_accuracy': 0.7874614968895912, 'epoch': 3.53}                                                                                                                                                        
 71%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                               | 240/340 [1:10:26<23:53, 14.34s/itwandb: WARNING Tried to log to step 240 that is less than the current step 241. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.648, 'grad_norm': 1.9980448484420776, 'learning_rate': 2.765957446808511e-05, 'entropy': 3.185216689109802, 'num_tokens': 3541096.0, 'mean_token_accuracy': 0.7521466858685016, 'epoch': 3.68}
{'eval_loss': 0.5529633164405823, 'eval_runtime': 40.6018, 'eval_samples_per_second': 3.029, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.053284391760826, 'eval_num_tokens': 3541096.0, 'eval_mean_token_accuracy': 0.7889109291136265, 'epoch': 3.68}                                                                                                                                                      
 74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                          | 250/340 [1:13:22<21:16, 14.18s/itwandb: WARNING Tried to log to step 250 that is less than the current step 251. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6261, 'grad_norm': 1.2135779857635498, 'learning_rate': 2.462006079027356e-05, 'entropy': 3.0971182912588118, 'num_tokens': 3682255.0, 'mean_token_accuracy': 0.7561966717243195, 'epoch': 3.83}
{'eval_loss': 0.5548965930938721, 'eval_runtime': 40.317, 'eval_samples_per_second': 3.051, 'eval_steps_per_second': 0.397, 'eval_entropy': 3.0519398599863052, 'eval_num_tokens': 3682255.0, 'eval_mean_token_accuracy': 0.786674115806818, 'epoch': 3.83}                                                                                                                                                       
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 260/340 [1:16:16<18:21, 13.77s/itwandb: WARNING Tried to log to step 260 that is less than the current step 261. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6483, 'grad_norm': 1.2179299592971802, 'learning_rate': 2.1580547112462007e-05, 'entropy': 3.1427818775177, 'num_tokens': 3824215.0, 'mean_token_accuracy': 0.7509463503956795, 'epoch': 3.97}
{'eval_loss': 0.5534147620201111, 'eval_runtime': 40.6328, 'eval_samples_per_second': 3.027, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.052387550473213, 'eval_num_tokens': 3824215.0, 'eval_mean_token_accuracy': 0.7899944894015789, 'epoch': 3.97}                                                                                                                                                      
 79%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                 | 270/340 [1:19:12<16:36, 14.23s/itwandb: WARNING Tried to log to step 270 that is less than the current step 271. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6367, 'grad_norm': 1.3303731679916382, 'learning_rate': 1.8541033434650455e-05, 'entropy': 3.129096575272389, 'num_tokens': 3961598.0, 'mean_token_accuracy': 0.7548992358721219, 'epoch': 4.12}
{'eval_loss': 0.5498879551887512, 'eval_runtime': 41.0256, 'eval_samples_per_second': 2.998, 'eval_steps_per_second': 0.39, 'eval_entropy': 3.0497053116559982, 'eval_num_tokens': 3961598.0, 'eval_mean_token_accuracy': 0.7922178022563457, 'epoch': 4.12}                                                                                                                                                      
 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 280/340 [1:22:04<13:55, 13.93s/itwandb: WARNING Tried to log to step 280 that is less than the current step 281. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6186, 'grad_norm': 1.9992138147354126, 'learning_rate': 1.5501519756838906e-05, 'entropy': 3.11612055003643, 'num_tokens': 4104122.0, 'mean_token_accuracy': 0.7610760033130646, 'epoch': 4.27}
{'eval_loss': 0.5512025952339172, 'eval_runtime': 40.3506, 'eval_samples_per_second': 3.048, 'eval_steps_per_second': 0.397, 'eval_entropy': 3.041562244296074, 'eval_num_tokens': 4104122.0, 'eval_mean_token_accuracy': 0.7897184416651726, 'epoch': 4.27}                                                                                                                                                      
 85%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                       | 290/340 [1:25:02<11:46, 14.12s/itwandb: WARNING Tried to log to step 290 that is less than the current step 291. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6211, 'grad_norm': 1.54454505443573, 'learning_rate': 1.2462006079027356e-05, 'entropy': 3.093658486008644, 'num_tokens': 4244589.0, 'mean_token_accuracy': 0.7606501258909703, 'epoch': 4.41}
{'eval_loss': 0.5528536438941956, 'eval_runtime': 40.5388, 'eval_samples_per_second': 3.034, 'eval_steps_per_second': 0.395, 'eval_entropy': 3.0312169790267944, 'eval_num_tokens': 4244589.0, 'eval_mean_token_accuracy': 0.7923429422080517, 'epoch': 4.41}                                                                                                                                                     
 88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                  | 300/340 [1:27:57<09:20, 14.00s/itwandb: WARNING Tried to log to step 300 that is less than the current step 301. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
/workspace/projects/nutrition-table/.venv/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
{'loss': 0.6425, 'grad_norm': 1.5784112215042114, 'learning_rate': 9.422492401215805e-06, 'entropy': 3.1492397665977476, 'num_tokens': 4388079.0, 'mean_token_accuracy': 0.7513018161058426, 'epoch': 4.56}
{'eval_loss': 0.5527857542037964, 'eval_runtime': 40.5093, 'eval_samples_per_second': 3.036, 'eval_steps_per_second': 0.395, 'eval_entropy': 3.022752195596695, 'eval_num_tokens': 4388079.0, 'eval_mean_token_accuracy': 0.7899530455470085, 'epoch': 4.56}                                                                                                                                                      
 91%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉              | 310/340 [1:30:56<07:10, 14.36s/itwandb: WARNING Tried to log to step 310 that is less than the current step 311. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.613, 'grad_norm': 1.2532031536102295, 'learning_rate': 6.3829787234042555e-06, 'entropy': 3.1256235510110857, 'num_tokens': 4529872.0, 'mean_token_accuracy': 0.7622268632054329, 'epoch': 4.71}
{'eval_loss': 0.5527204275131226, 'eval_runtime': 40.5341, 'eval_samples_per_second': 3.034, 'eval_steps_per_second': 0.395, 'eval_entropy': 3.025266945362091, 'eval_num_tokens': 4529872.0, 'eval_mean_token_accuracy': 0.788676094263792, 'epoch': 4.71}                                                                                                                                                       
 94%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌         | 320/340 [1:33:51<04:41, 14.08s/itwandb: WARNING Tried to log to step 320 that is less than the current step 321. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6396, 'grad_norm': 2.0462918281555176, 'learning_rate': 3.343465045592705e-06, 'entropy': 3.1301055431365965, 'num_tokens': 4672715.0, 'mean_token_accuracy': 0.7488784119486809, 'epoch': 4.86}
{'eval_loss': 0.5521355271339417, 'eval_runtime': 40.6163, 'eval_samples_per_second': 3.028, 'eval_steps_per_second': 0.394, 'eval_entropy': 3.029614210128784, 'eval_num_tokens': 4672715.0, 'eval_mean_token_accuracy': 0.7901008687913418, 'epoch': 4.86}                                                                                                                                                      
 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎    | 330/340 [1:36:49<02:23, 14.36s/itwandb: WARNING Tried to log to step 330 that is less than the current step 331. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'loss': 0.6345, 'grad_norm': 2.530855178833008, 'learning_rate': 3.03951367781155e-07, 'entropy': 3.108948438595503, 'num_tokens': 4810050.0, 'mean_token_accuracy': 0.7575616179368435, 'epoch': 5.0}  
{'eval_loss': 0.5515103340148926, 'eval_runtime': 40.7144, 'eval_samples_per_second': 3.021, 'eval_steps_per_second': 0.393, 'eval_entropy': 3.0265931338071823, 'eval_num_tokens': 4810050.0, 'eval_mean_token_accuracy': 0.7909483015537262, 'epoch': 5.0}                                                                                                                                                      
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 340/340 [1:39:41<00:00, 12.72s/itwandb: WARNING Tried to log to step 340 that is less than the current step 341. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
{'train_runtime': 5982.9398, 'train_samples_per_second': 0.905, 'train_steps_per_second': 0.057, 'train_loss': 0.8860841148039874, 'epoch': 5.0}                                                         
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 340/340 [1:39:43<00:00, 17.60s/it]
✅ Saved to /workspace/projects/nutrition-table/runs/exp1_20250906_234947/lang_vision
Loading checkpoint shards:  40%|███████████████████████████████████████████████████████▌                                                                                   | 2/5 [00:04<00:07,  2.63s/it]wandb: WARNING Tried to log to step 340 that is less than the current step 342. Steps must be monotonically increasing, so this data will be ignored. See https://wandb.me/define-metric to log data out of order.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.41s/it]
Mean IoU (strict): 0.4727
Precision@0.5: 0.4600
Recall@0.5: 0.4182
F1@0.5: 0.4381
Precision@0.75: 0.1200
Recall@0.75: 0.1091
F1@0.75: 0.1143
mAP over [0.5, 0.75]: 0.1027
📊 Precision/Recall curve saved to /workspace/projects/nutrition-table/runs/exp1_20250906_234947/lang_vision/precision_recall_curve_strict_post_lang_vision.png
📊 post_lang_vision Metrics: {'mean_iou': 0.472729530967772, 'precision@0.5': 0.46, 'recall@0.5': 0.41818181818181815, 'f1@0.5': 0.43809523809523804, 'precision@0.75': 0.12, 'recall@0.75': 0.10909090909090909, 'f1@0.75': 0.1142857142857143, 'mAP': 0.10272727272727272, 'pr_curve_path': '/workspace/projects/nutrition-table/runs/exp1_20250906_234947/lang_vision/precision_recall_curve_strict_post_lang_vision.png'}
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.39s/it]
{'answer': '(0,0),(1000,1000)<|im_end|>',
 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=819x650 at 0x7C6F2FF58D00>,
 'objects': {'bbox': [[0, 0, 1000, 1000]]}}
✅ Saved /workspace/projects/nutrition-table/runs/exp1_20250906_234947/lang_vision/bbox_post_lang_vision_val_20.png
🧹 Cleaning caches (mode: normal)...
WARNING: No matching packages
Files removed: 0 (0 bytes)
✅ Cleanup done. Current disk usage:
Filesystem      Size  Used Avail Use% Mounted on
overlay         200G  243M  200G   1% /
Filesystem                Size  Used Avail Use% Mounted on
mfs#euro.runpod.net:9421  2.0P  1.7P  296T  85% /workspace
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                         entropy ▁▂▄▆▇█▇▇▇▅▆▆▆▅▆▄▅▆▆▄▅▄▄▄▅▃▄▄▃▃▄▄▄▃
wandb:                           epoch ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb:                    eval/entropy ▁▃▅▇██▇▆▆▆▅▅▅▅▅▅▅▅▅▄▄▄▄▄▄▄▄▄▄▃▃▃▃▃
wandb:                       eval/loss █▆▄▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:        eval/mean_token_accuracy ▁▂▅▆▆▇▇▇▇█████████████████████████
wandb:                 eval/num_tokens ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb:                    eval/runtime ▅▄▄▃▆▅▆▃▄▄▄▄▃█▃▂▇▂▁▁▃▁▃▂▃▂▃▅▂▃▃▃▃▄
wandb:         eval/samples_per_second ▄▅▅▆▃▄▃▆▄▅▅▅▆▁▆▇▂▆██▆█▆▇▆▇▆▄▇▆▆▆▆▅
wandb:           eval/steps_per_second ▃▅▅▆▃▄▃▆▅▅▅▅▆▁▆▇▂▆██▆█▅▇▆▇▆▄▇▆▆▆▆▅
wandb:                       grad_norm ▇█▃▄▅▁▂▁▂▁▂▂▂▂▂▃▂▂▂▄▄▃▃▃▅▃▃▃▅▄▄▃▅▇
wandb:                   learning_rate ▇███▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▃▃▃▃▃▂▂▂▂▁▁▁
wandb:                            loss █▆▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             mean_token_accuracy ▁▂▄▆▆▇▇▇▇▇▇▇▇█▇███████████████████
wandb:                      num_tokens ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb:         post_lang_vision_f1@0.5 ▁
wandb:        post_lang_vision_f1@0.75 ▁
wandb:            post_lang_vision_mAP ▁
wandb:       post_lang_vision_mean_iou ▁
wandb:  post_lang_vision_precision@0.5 ▁
wandb: post_lang_vision_precision@0.75 ▁
wandb:     post_lang_vision_recall@0.5 ▁
wandb:    post_lang_vision_recall@0.75 ▁
wandb:                   train/entropy ▁▂▄▆▇█▇▇▇▅▆▆▆▅▆▄▅▆▆▄▅▄▄▄▅▃▄▄▃▃▄▄▄▃
wandb:                     train/epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇▇██
wandb:               train/global_step ▁▁▁▁▁▂▂▂▂▂▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████
wandb:                 train/grad_norm ▇█▃▄▅▁▂▁▂▁▂▂▂▂▂▃▂▂▂▄▄▃▃▃▅▃▃▃▅▄▄▃▅▇
wandb:             train/learning_rate ▇███▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▃▃▃▃▃▂▂▂▂▁▁▁
wandb:                      train/loss █▆▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/mean_token_accuracy ▁▂▄▆▆▇▇▇▇▇▇▇▇█▇███████████████████
wandb:                train/num_tokens ▁▁▁▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▇▇▇▇▇███
wandb: 
wandb: Run summary:
wandb:                         entropy 3.10895
wandb:                           epoch 5
wandb:                    eval/entropy 3.02659
wandb:                       eval/loss 0.55151
wandb:        eval/mean_token_accuracy 0.79095
wandb:                 eval/num_tokens 4810050.0
wandb:                    eval/runtime 40.7144
wandb:         eval/samples_per_second 3.021
wandb:           eval/steps_per_second 0.393
wandb:                       grad_norm 2.53086
wandb:                   learning_rate 0.0
wandb:                            loss 0.6345
wandb:             mean_token_accuracy 0.75756
wandb:                      num_tokens 4810050.0
wandb:         post_lang_vision_f1@0.5 0.4381
wandb:        post_lang_vision_f1@0.75 0.11429
wandb:            post_lang_vision_mAP 0.10273
wandb:       post_lang_vision_mean_iou 0.47273
wandb:  post_lang_vision_precision@0.5 0.46
wandb: post_lang_vision_precision@0.75 0.12
wandb:     post_lang_vision_recall@0.5 0.41818
wandb:    post_lang_vision_recall@0.75 0.10909
wandb:                      total_flos 2.301152403716905e+17
wandb:                   train/entropy 3.10895
wandb:                     train/epoch 5
wandb:               train/global_step 340
wandb:                 train/grad_norm 2.53086
wandb:             train/learning_rate 0.0
wandb:                      train/loss 0.6345
wandb:       train/mean_token_accuracy 0.75756
wandb:                train/num_tokens 4810050.0
wandb:                      train_loss 0.88608
wandb:                   train_runtime 5982.9398
wandb:        train_samples_per_second 0.905
wandb:          train_steps_per_second 0.057
wandb: 
wandb: 🚀 View run lang_vision_20250906_234947 at: https://wandb.ai/maryamdehdashti/nutrition-table-vl/runs/w5w63hcd
wandb: ⭐️ View project at: https://wandb.ai/maryamdehdashti/nutrition-table-vl
wandb: Synced 2 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20250907_011526-w5w63hcd/logs
✅ All stages completefigs/exp1.yaml





# Results
    # My data format
        # My Eval:
            # with vision blocks 20-23:
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3654
            # loss': 2.04
            # 'eval_loss': 0.27

            # with whole vision
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3421
            # loss': 2.3
            # 'eval_loss': 0.30

            # with 8 vision 16-23:
            # Mean IoU before: 0.3309
            # Mean IoU after: 0.3547
            # loss': 2.02
            # 'eval_loss': 0.27

        # Optimistic
            # with vision blocks 20-23:
            # Mean IoU before: 0.3218
            # Mean IoU after: 0.3345
            # loss': 2.02
            # 'eval_loss': 0.27
            # train_runtime': 3179

        # Strict

    # Answer data format
        # Strict
            # with vision blocks 20-23:
            # Mean IoU before: 0.3218
            # Mean IoU after: 0.3362
            # loss': 1.13
            # 'eval_loss': 0.15
            # train_runtime': 3323

    # train 9
        # Mean IoU after:0.3532
        # loss': 1.20
        # 'eval_loss': 0.15

    # train 10 (two stages, 1: language + 4 vision blocks, 2: same with iou loss )
        # Mean IoU after: 0.3532

    # train 11 (two stages, 1: full vision, 2: language and full vision)
        # Mean IoU after stage 1: 0.29
        # Mean IoU after stage 2: 0.33




W&B chart:
Project: nutrition-table-vl
└── Group: exp1_20240902_123456
    ├── meta         → experiment setup only (YAML config, dataset sizes, seed, etc.)
    ├── baseline     → baseline evaluation (IoU, PR curve, sample images)
    ├── vision       → stage 1 training + eval metrics
    └── lang_vision  → stage 2 training + eval metrics