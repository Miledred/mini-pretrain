# Final Checkpoint Release

The final/best checkpoint from every archived run is preserved as an asset of the [`experiment-archive-2026-10-25` GitHub Release](https://github.com/Miledred/mini-pretrain/releases/tag/experiment-archive-2026-10-25). This keeps multi-hundred-megabyte weights out of normal Git history while retaining the model state needed for sampling or resuming from the recorded final step.

`checkpoint_manifest.csv` is the machine-readable source of truth. It maps each release-asset name to its original run, optimizer settings, final/best step, validation loss, byte count, and SHA-256 checksum. Verify a downloaded asset against that checksum before use.

## Retention rule

- Included: all 14 `best.pt` files, one for each unique archived run.
- Excluded: `latest.pt` and `step_*.pt` files. For these runs, `best.pt` contains the full model, optimizer, scheduler, configuration, step, and best-validation-loss state. The periodic files are redundant intermediate snapshots.
- The local raw datasets and virtual environment are not release assets; their recreation procedure and provenance are recorded in [`README.md`](README.md).

For a downloaded asset named `best__<run>.pt`, sample with:

```bash
python src/sample.py --checkpoint path/to/best__<run>.pt
```

If the target machine uses a different GPU or CPU, use the `--device` option supported by `src/sample.py`.
