# Phase 2 Optimizer Comparison Report

## 1. Run summary

| Run | Optimizer | Peak LR | Best val loss | Best step | Best val PPL | Final train loss | Mean train loss (last 200) | Median tok/s | Max memory MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW seed42 | my_adamw | 0.0005000 | 5.1682 | 50000 | 175.60 | 5.3408 | 5.2374 | 18263.63 | 3174.18 |
| AdamW seed123 | my_adamw | 0.0005000 | 5.1775 | 50000 | 177.24 | 5.0715 | 5.2437 | 18162.81 | 3174.18 |
| Muon seed42 | my_muon | 0.0007500 | 5.1705 | 50000 | 175.99 | 5.3285 | 5.2440 | 14513.62 | 3070.18 |
| Muon seed123 | my_muon | 0.0007500 | 5.2982 | 50000 | 199.98 | 5.2113 | 5.3994 | 13713.77 | 3070.18 |

## 2. Relative comparison against the first run

Baseline: `AdamW seed42`

| Candidate | Val-loss reduction | PPL reduction | Final-train-loss reduction | Throughput change | Memory change |
|---|---:|---:|---:|---:|---:|
| AdamW seed123 | -0.18% | -0.93% | 5.04% | -0.55% | +0.00% |
| Muon seed42 | -0.04% | -0.22% | 0.23% | -20.53% | -3.28% |
| Muon seed123 | -2.52% | -13.88% | 2.42% | -24.91% | -3.28% |

## 3. Automatic observations

- Lowest validation loss: **AdamW seed42**, `5.1682` at step `50000`.
- Its validation perplexity at the best checkpoint is `175.60`.
- Training-loss plots use a rolling mean; validation metrics use the original evaluation points.
- Throughput is summarized with the median after the first 100 steps to reduce sensitivity to transient system slowdowns.
- Different optimizer-specific learning rates must be reported explicitly; do not describe the comparison as using identical hyperparameters.

## 4. Generated figures

- `train_loss_comparison.png`
- `val_loss_comparison.png`
- `val_ppl_comparison.png`
- `grad_norm_comparison.png`
- `update_norm_comparison.png`
- `update_ratio_comparison.png`
- `lr_comparison.png`
- `tokens_per_sec_comparison.png`
- `max_memory_comparison.png`
