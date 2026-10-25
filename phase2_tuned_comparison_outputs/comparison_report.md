# Phase 2 Optimizer Comparison Report

## 1. Run summary

| Run | Optimizer | Peak LR | Best val loss | Best step | Best val PPL | Final train loss | Mean train loss (last 200) | Median tok/s | Max memory MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW 1.5e-4 | my_adamw | 0.0001500 | 5.3346 | 50000 | 207.39 | 5.5276 | 5.4215 | 17382.00 | 3174.18 |
| AdamW 5e-4 | my_adamw | 0.0005000 | 5.1682 | 50000 | 175.60 | 5.3408 | 5.2374 | 18263.63 | 3174.18 |
| Muon 1.5e-4 | my_muon | 0.0001500 | 5.6711 | 50000 | 290.34 | 5.8641 | 5.7893 | 13559.13 | 3070.18 |
| Muon 7.5e-4 | my_muon | 0.0007500 | 5.1705 | 50000 | 175.99 | 5.3285 | 5.2440 | 14513.62 | 3070.18 |

## 2. Relative comparison against the first run

Baseline: `AdamW 1.5e-4`

| Candidate | Val-loss reduction | PPL reduction | Final-train-loss reduction | Throughput change | Memory change |
|---|---:|---:|---:|---:|---:|
| AdamW 5e-4 | 3.12% | 15.33% | 3.38% | +5.07% | +0.00% |
| Muon 1.5e-4 | -6.31% | -40.00% | -6.09% | -21.99% | -3.28% |
| Muon 7.5e-4 | 3.08% | 15.14% | 3.60% | -16.50% | -3.28% |

## 3. Automatic observations

- Lowest validation loss: **AdamW 5e-4**, `5.1682` at step `50000`.
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
