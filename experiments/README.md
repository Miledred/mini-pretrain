# Experiment Archive / 实验归档

This directory is the portable, Git-tracked record of the completed `mini_pretrain` experiments. It preserves the evidence needed to inspect the results and regenerate figures without checking large virtual environments, tokenized datasets, TensorBoard events, or repeated training snapshots into normal Git history.

## What was run

| Phase | Scope | Primary conclusion | Evidence |
|---|---|---|---|
| Phase 1 | Custom AdamW L4 and L8 WikiText baselines | The L8 model improved validation loss from 5.5539 to 5.3346 at 50k steps, at a throughput and memory cost. | `../phase1_outputs/`, `../phase1_report_package/` |
| Phase 2 | 50k-step AdamW/Muon learning-rate comparison | Tuned AdamW `5e-4` reached 5.1682 best validation loss; tuned Muon `7.5e-4` reached 5.1705. The optimizers used different tuned learning rates. | `../phase2_comparison_outputs/`, `../phase2_tuned_comparison_outputs/` |
| Seed check | Seed 42 and 123 for the tuned L8 runs | AdamW was lower on both archived seeds (5.1682, 5.1775) than Muon (5.1705, 5.2982). Two seeds are not enough for a definitive statistical conclusion. | `../seed123_comparison_outputs/` |

The raw outputs support a cautious conclusion: on this model/data/budget, tuned AdamW is at least competitive with Muon and has materially higher throughput. Do not describe the comparison as identical-hyperparameter training: the best archived AdamW and Muon runs use `5e-4` and `7.5e-4` respectively.

## Preserved primary evidence

- `configs/<run>.json` — 14 resolved run configurations. They retain experimental hyperparameters, seeds, and Muon settings, while local absolute data paths were normalized to repository-relative `data/` paths.
- `raw_metrics/<run>/metrics.csv` — every unique per-step metrics file (about 530k rows total). These are the source records for the tables and figures. The duplicate recovered `bigmodel` CSV is deliberately omitted.
- `data/data_meta.json` — token counts, dtype, vocabulary size, and tokenizer metadata for the exact input build.
- `../phase1_outputs/`, `../phase1_report_package/`, `../phase2_comparison_outputs/`, `../phase2_tuned_comparison_outputs/`, and `../seed123_comparison_outputs/` — reports, plots, presentation decks, summary tables, and fixed-prompt samples.
- [`CHECKPOINTS.md`](CHECKPOINTS.md) — the final model-state retention manifest and GitHub Release asset checksums.

## Rebuild the input data

The original raw and tokenized data are intentionally not stored in Git. Recreate them with:

```bash
python build_wikitext.py
```

The build uses the Hugging Face dataset `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, and the GPT-2 tokenizer. The archived metadata records 114,140,917 training tokens and 239,536 validation tokens using `uint16` tokens. The original generated binaries had these SHA-256 values:

| File | SHA-256 |
|---|---|
| `train.bin` | `2DC7CA81C45510D442507706EB98C350EF0D8F190455AA1C243D0C13C704C722` |
| `val.bin` | `71191D75ADADD2D85EF52B7601C8296D9B13304C7A3652E9BD25A0E953828609` |
| `data_meta.json` | `826238B93653ADA025386B3DFEB4951B705FB40D36D1786AA360C7E6F71A7FCF` |

## Environment recorded for the archive

The runs were inspected in Python 3.13.4 with PyTorch 2.11.0+cu130 (CUDA 13.0), Transformers 5.5.0, Datasets 4.8.4, TensorBoard 2.20.0, Matplotlib 3.10.9, and NumPy 2.4.3. Install the project dependencies with `pip install -r requirements.txt`; use a CUDA-compatible PyTorch build appropriate to the target machine.

## Intentional exclusions

| Excluded material | Why it is not in normal Git |
|---|---|
| `.venv/` | Rebuildable from dependencies; machine-specific. |
| `data/train.*`, `data/val.*` | Regenerated from documented public source; metadata and hashes are retained. |
| TensorBoard event files | Redundant with the complete CSV metrics and can expose machine identifiers in filenames. |
| `latest.pt` and `step_*.pt` | Redundant temporal snapshots. Each run's final/best model is retained as a Release asset instead. |

The source revision in this archive is the reproducible starting point for future work. Historical runs predate this preservation commit, so their configuration, metrics, data provenance, and final checkpoint assets are retained explicitly rather than claiming a previously recorded source commit.
