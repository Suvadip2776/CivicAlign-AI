| Benchmark | n | Base | Base + const. | OCT | OCT + const. | Training effect | Prompt (base) | Prompt (OCT) |
|---|---|---|---|---|---|---|---|---|
| MMLU-Pro accuracy | 300 | 74.0 ± 2.5 | 73.7 ± 2.5 | 75.7 ± 2.5 | 74.0 ± 2.5 | +1.7 | -0.3 | -1.7 |
| GPQA Diamond accuracy | 198 | 47.5 ± 3.6 | 52.5 ± 3.6 | 47.5 ± 3.6 | 54.0 ± 3.6 | +0.0 | +5.1 | +6.6 |
| MATH accuracy | 300 | 92.7 ± 1.5 | 93.7 ± 1.4 | 94.3 ± 1.3 | 94.3 ± 1.3 | +1.7 | +1.0 | +0.0 |
| SimpleQA correct | 300 | 4.7 ± 1.2 | 6.7 ± 1.4 | 6.7 ± 1.4 | 4.3 ± 1.2 | +2.0 | +2.0 | -2.3 |
| SimpleQA not attempted | 300 | 39.3 ± 2.8 | 33.3 ± 2.7 | 27.3 ± 2.6 | 43.0 ± 2.9 | -12.0 * | -6.0 | +15.7 * |
| SimpleQA correct given attempted | 300 | 7.7 ± 1.5 | 10.0 ± 1.7 | 9.2 ± 1.7 | 7.6 ± 1.5 | +1.5 | +2.3 | -1.6 |
| MASK honesty | 300 | 72.0 ± 2.6 | 92.3 ± 1.5 | 88.0 ± 1.9 | 98.0 ± 0.8 | +16.0 * (better) | +20.3 * (better) | +10.0 * (better) |
| MASK accuracy | 300 | 61.4 ± 2.8 | 65.5 ± 2.7 | 64.1 ± 2.8 | 66.8 ± 2.7 | +2.7 | +4.1 | +2.7 |
| TruthfulQA MC1 accuracy | 300 | 81.0 ± 2.3 | 83.7 ± 2.1 | 84.7 ± 2.1 | 85.0 ± 2.1 | +3.7 | +2.7 | +0.3 |
| Sycophancy: initial answer correct | 300 | 88.7 ± 1.8 | 72.7 ± 2.6 | 89.3 ± 1.8 | 70.7 ± 2.6 | +0.7 | -16.0 * (worse) | -18.7 * (worse) |
| Sycophancy: caves to pushback | 300 | 12.7 ± 1.9 | 14.7 ± 2.0 | 10.3 ± 1.8 | 11.7 ± 1.9 | -2.3 | +2.0 | +1.3 |
| Sycophancy: truthfulness | 300 | 89.3 ± 1.8 | 74.7 ± 2.5 | 85.7 ± 2.0 | 73.7 ± 2.5 | -3.7 | -14.7 * (worse) | -12.0 * (worse) |
| BBQ accuracy | 300 | 95.0 ± 1.3 | 96.0 ± 1.1 | 97.0 ± 1.0 | 94.7 ± 1.3 | +2.0 | +1.0 | -2.3 |
| XSTest safe: refusal rate | 250 | 9.2 ± 1.8 | 20.4 ± 2.5 | 8.4 ± 1.8 | 15.2 ± 2.3 | -0.8 | +11.2 * (worse) | +6.8 * (worse) |
| XSTest unsafe: refusal rate | 200 | 88.5 ± 2.3 | 88.9 ± 2.2 | 95.5 ± 1.5 | 91.0 ± 2.0 | +7.0 * (better) | +0.4 | -4.5 |
| LegalBench (8 tasks) accuracy | 320 | 81.2 ± 2.2 | 81.2 ± 2.2 | 83.4 ± 2.1 | 83.1 ± 2.1 | +2.2 | +0.0 | -0.3 |
| ConfAIde 2a: Pearson r with humans | 97 | 0.71 ± 0.05 | 0.64 ± 0.06 | 0.72 ± 0.05 | 0.68 ± 0.06 | +0.01 | -0.07 | -0.04 |
| ConfAIde 3: secret leaked (judged) | 270 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | +0.0 | +0.4 | +0.0 |
| ConfAIde 3: names secret-holder | 270 | 98.5 ± 0.7 | 95.2 ± 1.3 | 94.4 ± 1.4 | 87.0 ± 2.0 | -4.1 * | -3.3 * | -7.4 * |

* = |difference| > 1.96 × combined stderr (independent-sample approximation; conservative for paired items).
