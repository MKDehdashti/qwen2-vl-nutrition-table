# Nutrition-Table Training Pipeline

This repository provides a training and evaluation pipeline for fine-tuning [Qwen-VL](https://huggingface.co/Qwen) (developed by Alibaba Cloud, Apache-2.0 License) on **nutrition table detection and segmentation** tasks.  
It implements a multi-stage LoRA training process with integrated experiment tracking using **Weights & Biases (W&B)**.

---

## Project Structure
- **configs/** → Experiment YAMLs (e.g., exp1.yaml, exp1_debug.yaml)  
- **src/** → Source code (train.py, utilities, evaluation scripts)  
- **models/** → Pretrained and fine-tuned weights (ignored by git)  
- **outputs/** → Evaluation results and plots (ignored by git)  
- **runs/** → Training logs and checkpoints (ignored by git)  
- **wandb/** → W&B logging cache (ignored by git)  

---

## Features
- Multi-stage training workflow (baseline → vision → language+vision)  
- LoRA fine-tuning with regex-based layer targeting  
- Configuration-driven experiments for reproducibility  
- Integrated training and evaluation loop  
- Weights & Biases integration for monitoring and comparison  
- Clean `.gitignore` to exclude large files, caches, and secrets  

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/MKDehdashti/qwen_runpod.git
cd qwen_runpod
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset
This project uses a custom nutrition table dataset containing images annotated with bounding boxes around table regions.  

- **Format**: COCO-style JSON annotations with bounding boxes  
- **Splits**: Training and validation sets handled automatically by the pipeline  
- **Preprocessing**: Image resizing, tokenizer alignment, and batching managed in `train.py`  

---

## Training & Evaluation
Training and evaluation are both handled by the main script.

Run a full experiment:
```bash
python src/train.py --config configs/exp1.yaml
```

For quick debugging:
```bash
python src/train.py --config configs/exp1_debug.yaml
```

---

## Metrics
The pipeline automatically evaluates during training and logs results to W&B, including:  
- Mean IoU  
- Precision, Recall, F1-score  
- mAP@0.5, mAP@0.75  
- Precision–recall curves  

Evaluation outputs are also saved locally under `outputs/`.

---

## Notes
- Large files and directories (`wandb/`, `runs/`, `outputs/`, `models/`, `.venv/`) are excluded via `.gitignore`.  
- Use Git branches (e.g., `exp/lora_fc`, `exp/data_balance`) to organize experiment variations.  
- W&B dashboards are recommended for comparing experiments across runs.  

---

## Requirements
- Python ≥ 3.9  
- PyTorch ≥ 2.1  
- Hugging Face: transformers, datasets, accelerate, peft, trl  
- Weights & Biases (wandb)  

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Roadmap
- [ ] Add advanced loss functions (Dice, IoU)  
- [ ] Extend LoRA coverage to vision MLP layers  
- [ ] Implement balanced sampling for rare and small objects  
- [ ] Provide dataset preprocessing scripts  

---

## License
This repository is distributed under the **MIT License**.  
It fine-tunes **Qwen-VL**, developed by **Alibaba Cloud**, which is available under the [Apache-2.0 License](https://huggingface.co/Qwen).  

---

## Author
Maintained by [MKDehdashti](https://github.com/MKDehdashti).  
Contributions and feedback are welcome.
