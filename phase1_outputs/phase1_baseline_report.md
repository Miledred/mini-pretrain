# Phase 1 Baseline and Early Muon Audit Report

## 1. Audit scope

- Project root: `.`
- Logs directory: `logs/`
- Checkpoint directory: `checkpoints/`

## 2. Run inventory

| Run | Final step | Best val loss | Best step | Final train loss | Final val ppl | Avg tok/s last100 | Max memory MB | Checkpoints | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `my_adamw_L4_H4_D256_bs8_blk256_ts256_` | 50000 | 5.5539 | 50000 | 5.8432 | 258.2395 | 43747.50 | 2021.65 | best=True, latest=True, periodic=10 | complete_or_long |
| `my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel` | 50000 | 5.3346 | 50000 | 5.5276 | 207.3880 | 17663.53 | 3174.18 | best=True, latest=True, periodic=10 | complete_or_long |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr1p5e4_50k` | 50000 | 5.6711 | 50000 | 5.8641 | 290.3434 | 12245.12 | 3070.18 | best=True, latest=True, periodic=10 | complete_or_long |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_lr7p5e4_50k` | 50000 | 5.1705 | 50000 | 5.3285 | 175.9948 | 14530.47 | 3070.18 | best=True, latest=True, periodic=10 | complete_or_long |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_tune10k_lr3e4` | 10000 | 6.1684 | 10000 | 6.1410 | 477.4101 | 9930.44 | 3070.18 | best=True, latest=True, periodic=2 | possibly_incomplete |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_tune10k_lr5e4` | 10000 | 5.9771 | 10000 | 5.9513 | 394.3034 | 13327.98 | 3070.18 | best=True, latest=True, periodic=2 | possibly_incomplete |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_tune10k_lr7p5e4` | 10000 | 5.8523 | 10000 | 5.8197 | 348.0463 | 14330.16 | 3070.18 | best=True, latest=True, periodic=2 | possibly_incomplete |
| `my_muon_L8_H8_D512_bs8_blk256_ts256_muon_l8_tune10k_scale2` | 10000 | 6.3188 | 10000 | 6.3064 | 554.9267 | 14346.01 | 3070.18 | best=True, latest=True, periodic=2 | possibly_incomplete |

## 3. Interpretation checklist

For each baseline or tuning run, inspect the generated PNG files and answer these four questions:

1. Does `train_loss` decrease smoothly, or does it show spikes/instability?
2. Does `val_loss` keep decreasing, plateau, or rebound after a certain step?
3. Is `update_ratio` stable, vanishing, or suddenly increasing?
4. Is throughput stable enough for fair comparison with Muon later?

## 4. Files generated

- `run_inventory.csv`: machine-readable run summary.
- `<run_name>/*.png`: single-metric curves for direct insertion into slides.

## 5. Archived-status wording for supervisor update

> The AdamW baseline and the first Muon tuning runs have been archived with train/validation loss, perplexity, learning rate, gradient norm, update norm, update ratio, throughput, memory usage, and checkpoints. The complete Phase 2 comparisons and seed check are summarized separately in the Phase 2 output folders.
