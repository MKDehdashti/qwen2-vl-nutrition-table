# Benchmark result files

Raw measurement output backing the results tables in [`../../README.md`](../../README.md)
and [`../../project_report.md`](../../project_report.md). Committed so the published
numbers can be checked against their source.

All accuracy figures are on the `openfoodfacts/nutrition-table-detection` **val** split.
The `.png` bounding-box visualizations produced alongside these runs are not committed
(regenerable); three representative ones live in [`../../docs/examples/`](../../docs/examples/).

## `eval/` — accuracy + per-sample latency (`inference_eval.py`)

| File | Samples | What it measures |
|------|:-------:|------------------|
| `exp13_hf_bs1_1.json` | 123 | HF backend, batch=1 |
| `exp13_hf_bs4_1.json` | 123 | HF backend, batch=4 — the reported operating point |
| `exp13_hf_bs16_1.json` | 123 | HF backend, batch=16 |
| `exp13_vllm_bs1_1.json` | 123 | vLLM, sequential — latency **not** comparable, see note |
| `exp13_vllm_bs4_1.json` | 123 | vLLM, sequential — latency **not** comparable, see note |

> **Note on the two vLLM files.** `inference_eval.py`'s vLLM path sends requests in a
> sequential loop, so `--batch_size` never batched anything; it only changed the latency
> divisor. The 378 ms vs 1,410 ms spread between these two files is run-to-run variance
> (cold server), not a batching effect. **Their accuracy numbers are valid; their latency
> numbers are not.** All published vLLM performance claims come from `throughput/` instead.
> The function now carries a docstring saying so.

## `throughput/` — vLLM serving concurrency (`vllm_throughput.py`)

Real async concurrency via `asyncio.Semaphore`. This is the valid vLLM benchmark.

| File | Concurrency | Mean latency | P95 | Throughput |
|------|:-----------:|:------------:|:---:|:----------:|
| `vllm_c1.json` | 1 | 1,377.62 ms | 1,505.59 ms | 0.73 req/s |
| `vllm_c4.json` | 4 | 398.39 ms | 466.67 ms | 9.89 req/s |
| `vllm_c8.json` | 8 | 434.59 ms | 1,056.33 ms | **17.82 req/s** |
| `vllm_c16.json` | 16 | 16,527.85 ms | 22,652.71 ms | 0.94 req/s |

Throughput peaks at c=8 and collapses at c=16 as the scheduler saturates.

## `hf_eval/` and `vllm_eval/` — full-dataset evaluation runs

| File | Samples | Mean IoU | Notes |
|------|:-------:|:--------:|-------|
| `hf_eval/metrics_exp13_hf.json` | 123 | 0.82 | Batched HF eval, batch=1 |
| `hf_eval/metrics_hf_bs4.json` | 123 | 0.816 | Batched HF eval, batch=4 |
| `vllm_eval/metrics_exp13_unquantized.json` | 123 | 0.81 | vLLM full-dataset eval of the shipped model |
| `vllm_eval/metrics_unquantized_older_training_run1219_2206.json` | 50 | 0.788 | **Superseded.** An earlier training run, kept for comparison |
| `vllm_eval/metrics_not_quantized_with_system_and_strict_format.json` | 50 | **0.573** | **Negative result — prompt ablation, not a model regression** |

### About the 0.573 result

That file records a **deliberate prompt-format ablation**, not a failure of the shipped
model. It applied a stricter output-format instruction on top of the system message, and
accuracy fell from ~0.82 to 0.573 on a 50-sample subset.

The finding is that this model is unusually sensitive to deviations in the prompt it was
fine-tuned with — changing the instruction wording degrades detection sharply even though
the weights are unchanged. That is why the exact training prompt is reproduced verbatim in
the README, the model cards, and the HF Space, and why it is treated as part of the model's
interface rather than a tunable knob. It is kept here as evidence for that claim.
