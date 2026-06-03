# Nutrition Table Detection

Fine-tuned **Qwen2-VL-7B-Instruct** to detect nutrition tables in product images. Given an image, the model outputs all bounding box coordinates as text.

**Model**: [`MayaKD/qwen2-vl-7b-nutrition`](https://huggingface.co/MayaKD/qwen2-vl-7b-nutrition)  
**Dataset**: [`openfoodfacts/nutrition-table-detection`](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection)  
**Final accuracy**: Mean IoU 0.82 · Precision 0.91 · Recall 0.89 · F1 0.90  
**Best serving throughput**: 10 req/s (vLLM, concurrency=4)

---

## Structure

```
nutrition_table_fine_tuning/   LoRA fine-tuning pipeline (multi-stage, multi-GPU)
nutrition_table_inference/     Inference and benchmarking (HF + vLLM)
```

---

## Docs

- [`project_report.md`](project_report.md) — approach, results, and code overview
- [`CLAUDE.md`](CLAUDE.md) — all commands, configs, and operational notes

---

## Roadmap

- [ ] Add advanced loss functions (Dice, IoU)
- [ ] Implement balanced sampling for rare and small objects
- [ ] Resolve vLLM serving for the 4-bit GPTQ quantized model
- [ ] Provide dataset preprocessing scripts

---

## License

Distributed under the **MIT License**. Fine-tunes **Qwen2-VL** (developed by Alibaba Cloud), available under the [Apache-2.0 License](https://huggingface.co/Qwen).

## Author

Maintained by [MKDehdashti](https://github.com/MKDehdashti). Contributions and feedback are welcome.

Repository: [github.com/MKDehdashti/qwen2-vl-nutrition-table](https://github.com/MKDehdashti/qwen2-vl-nutrition-table)
