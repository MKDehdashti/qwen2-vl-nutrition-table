# Nutrition Table Detection (Qwen-VL Fine-Tuning)

This project fine-tunes **Qwen2-VL** for bounding-box detection of nutrition tables in images.
It uses Hugging Face `transformers`, `trl`, LoRA adapters, and Weights & Biases (W\&B) for experiment tracking.

---

## 🔧 Setup

### 1. Clone & enter project

```bash
cd projects/nutrition-table
```

### 2. Create environment (recommended)

```bash
python -m venv ../../envs/qwen
source ../../envs/qwen/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Training

### Stage 1 (Vision only)

```bash
accelerate launch --mixed_precision=bf16 src/train.py --config configs/stage1.yaml
```

### Stage 2 (Vision + Language)

```bash
accelerate launch --mixed_precision=bf16 src/train.py --config configs/stage2.yaml
```

---

## 📊 Evaluation

Run evaluation after training:

```bash
python src/eval_utils.py --config configs/eval.yaml
```

---

## 📂 Project Structure

```
nutrition-table/
├─ configs/              # configs for training, env, eval
│   ├─ default.env
│   ├─ stage1.yaml
│   ├─ stage2.yaml
├─ src/                  # source code
│   ├─ train.py
│   ├─ data_utils.py
│   ├─ model_utils.py
│   ├─ eval_utils.py
│   └─ viz_utils.py
├─ runs/                 # training outputs & checkpoints
├─ requirements.txt      # dependencies
├─ Notes.md              # personal notes & results
└─ README.md             # this file
```

---

## 📊 Results

### Stage 1

| Variant             | Mean IoU Before | Mean IoU After | Loss | Eval Loss | Runtime (s) | Notes       |
| ------------------- | --------------- | -------------- | ---- | --------- | ----------- | ----------- |
| Vision blocks 20-23 | 0.3309          | 0.3654         | 2.04 | 0.27      | 3200        | Best so far |
| Whole vision        | 0.3309          | 0.3421         | 2.30 | 0.30      | 3100        | Worse       |

### Stage 2

| Variant         | Mean IoU Before | Mean IoU After | Loss | Eval Loss | Runtime (s) | Notes |
| --------------- | --------------- | -------------- | ---- | --------- | ----------- | ----- |
| Stage2 baseline | 0.3309          | 0.3547         | 2.02 | 0.27      | 3179        |       |

---

## 📝 Observations

* Training loss plateaued after epoch 2 → try lower LR or fewer epochs.
* Stage 2 improves slightly but runtime increases.
* Possible next step: test with connectors instead of whole vision fine-tuning.

---

## 📌 To-Do / Next Experiments

* [ ] Try different LoRA target modules.
* [ ] Run with 5 epochs instead of 3.
* [ ] Add alternative collator masking.
* [ ] Compare strict vs optimistic IoU evaluation.
