# Mini Pretrain

A lightweight PyTorch-based language model pretraining project for studying optimizer behavior and training dynamics.

This project builds a small GPT-style causal language modeling pipeline from data preprocessing to training, logging, checkpointing, and text generation. The current baseline focuses on a custom hand-written AdamW optimizer, with the long-term goal of extending the framework to compare AdamW with more advanced optimizers such as Shampoo, SOAP, and Muon.

## Overview

Modern large language model training relies heavily on optimization algorithms. Instead of only calling existing optimizer APIs, this project implements a custom AdamW optimizer and integrates it into a complete pretraining pipeline. The goal is to understand optimizer behavior through actual training experiments, including loss curves, gradient norms, update norms, throughput, GPU memory usage, and generated text samples.

The project currently supports:

- WikiText data preprocessing
- GPT-style causal language model training
- Custom AdamW optimizer implementation
- Memory-mapped token dataset loading
- TensorBoard visualization
- CSV metric logging
- Checkpoint saving and resume support
- Text generation from trained checkpoints

## Features

### Custom Optimizer

The project includes a hand-written AdamW optimizer implementation. It supports:

- First-moment and second-moment exponential moving averages
- Bias correction
- Decoupled weight decay
- Parameter grouping for weight decay and no-weight-decay parameters
- Compatibility with PyTorch optimizer interfaces

This makes the optimizer easy to inspect, modify, and extend in future experiments.

### GPT-style Language Model

The model is built using Hugging Face Transformers with a GPT-style causal language modeling architecture. The default configuration uses a medium-scale model setting:

- 8 Transformer layers
- 8 attention heads
- 512 embedding dimensions
- 256 context length

This creates a practical small-scale pretraining environment for studying optimizer behavior without requiring large-scale distributed training.

### Large-scale Token Loading

The project preprocesses raw text into binary token files and loads them through a memory-mapped dataset. This avoids loading the entire dataset into CPU memory during training and makes the pipeline more suitable for larger-scale experiments.

### Training Monitoring

The training script records multiple metrics:

- Training loss
- Validation loss
- Validation perplexity
- Learning rate
- Gradient norm
- Parameter norm
- Update norm
- Update ratio
- Step time
- Tokens per second
- GPU memory usage

These metrics are written to both TensorBoard and CSV files.

### Checkpointing

The training pipeline saves checkpoints during training, including:

- Model state
- Optimizer state
- Scheduler state
- Current training step
- Best validation loss
- Training configuration

This allows training to be resumed after interruption and makes it easier to compare different training runs.

## Project Structure

```text
.
├── build_wikitext.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model_factory.py
│   ├── optim_factory.py
│   ├── sample.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── trainer.py
│   └── utils.py
```

### File Description

| File | Description |
|---|---|
| `build_wikitext.py` | Downloads and preprocesses WikiText data, then writes tokenized binary files. |
| `src/config.py` | Stores training configuration, model size, optimizer settings, logging paths, and checkpoint settings. |
| `src/dataset.py` | Implements a memory-mapped token block dataset for efficient training data loading. |
| `src/model_factory.py` | Builds the GPT-style causal language model. |
| `src/optim_factory.py` | Implements the custom AdamW optimizer and optimizer builder. |
| `src/train.py` | Main training entry point. |
| `src/trainer.py` | Contains training step, evaluation logic, gradient statistics, and update statistics. |
| `src/tokenizer.py` | Builds the tokenizer. |
| `src/sample.py` | Loads a trained checkpoint and generates text samples. |
| `src/utils.py` | Utility functions such as random seed setup. |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mini_pretrain.git
cd mini_pretrain
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The main dependencies are:

```text
torch
transformers
datasets
numpy
tensorboard
```

A CUDA-enabled GPU is recommended for training.

## Dataset Preparation

Run the following command to build the WikiText dataset:

```bash
python build_wikitext.py
```

This script will:

1. Download WikiText from Hugging Face Datasets.
2. Clean empty lines and article title markers.
3. Save raw text files for inspection.
4. Tokenize the data with the GPT-2 tokenizer.
5. Save tokenized binary files for efficient training.
6. Save metadata such as tokenizer name, vocabulary size, token count, and token dtype.

Generated files will be saved under:

```text
data/
├── train.txt
├── val.txt
├── train.bin
├── val.bin
└── data_meta.json
```

These generated files are excluded from Git tracking.

## Training

Start training with:

```bash
python src/train.py
```

The default configuration can be modified in:

```text
src/config.py
```

Important training parameters include:

| Parameter | Description |
|---|---|
| `block_size` | Context length of each training sequence. |
| `batch_size` | Number of sequences per training batch. |
| `n_layer` | Number of Transformer layers. |
| `n_head` | Number of attention heads. |
| `n_embd` | Embedding dimension. |
| `learning_rate` | Peak learning rate. |
| `min_lr` | Minimum learning rate after cosine decay. |
| `max_steps` | Total number of training steps. |
| `eval_interval` | Number of steps between validation evaluations. |
| `warmup_steps` | Number of learning rate warmup steps. |
| `grad_clip_norm` | Gradient clipping threshold. |
| `save_every` | Number of steps between checkpoint saves. |

## TensorBoard Visualization

Training logs are saved under:

```text
logs/
```

Launch TensorBoard with:

```bash
tensorboard --logdir logs
```

Then open the displayed local URL in your browser.

The training dashboard includes:

- `train/loss`
- `train/grad_norm`
- `train/param_norm`
- `train/update_norm`
- `train/update_ratio`
- `train/lr`
- `val/loss`
- `val/ppl`
- `val/best_loss`
- `system/step_time_sec`
- `system/tokens_per_sec`
- `system/max_memory_mb`

## Checkpoints

Checkpoints are saved under:

```text
checkpoints/
```

Each training run creates a directory named according to the optimizer and model configuration.

Typical checkpoint files include:

```text
best.pt
latest.pt
step_5000.pt
step_10000.pt
...
```

- `best.pt` stores the checkpoint with the best validation loss.
- `latest.pt` stores the latest training state.
- `step_x.pt` stores periodic checkpoints.

Generated checkpoint files are excluded from Git tracking.

## Resume Training

To resume training, set `resume_from` in `src/config.py`:

```python
resume_from: str | None = "checkpoints/YOUR_RUN_NAME/latest.pt"
```

Then run:

```bash
python src/train.py
```

The script will restore:

- Model parameters
- Optimizer state
- Learning rate scheduler state
- Training step
- Best validation loss

## Text Generation

After training, update the checkpoint path in:

```text
src/sample.py
```

Example:

```python
checkpoint_path = "checkpoints/YOUR_RUN_NAME/best.pt"
```

Then run:

```bash
python src/sample.py
```

The script generates text from several fixed prompts using both greedy decoding and sampling.

Example prompts include:

```text
The history of science
In the early twentieth century
Artificial intelligence is
The theory of evolution states that
In modern society, technology has
```

## Current Status

The current version has completed a custom AdamW baseline training pipeline. It supports larger WikiText-based training, TensorBoard monitoring, checkpointing, and qualitative sampling.

The default model configuration is designed as a medium-scale experimental model rather than a toy single-batch example. It is suitable for studying optimizer behavior before moving to more advanced optimizer reproduction.

## Future Work

Planned extensions include:

- Implement Shampoo optimizer
- Implement SOAP optimizer
- Implement Muon optimizer
- Add optimizer comparison scripts
- Add experiment configuration files
- Add automatic plotting from CSV logs
- Add more stable sampling and evaluation utilities
- Compare training speed, convergence behavior, memory usage, and update statistics across optimizers

## Notes

The following generated files are not tracked by Git:

```text
data/
logs/
checkpoints/
runs/
*.pt
*.pth
*.ckpt
*.safetensors
events.out.tfevents.*
```

This keeps the repository lightweight and focused on source code.

## Motivation

This project is part of a longer-term study on optimization methods for neural network pretraining. By building the training loop, optimizer, data pipeline, logging system, and checkpoint mechanism manually, the project aims to provide a deeper understanding of how optimizer design affects training dynamics.

Instead of treating optimizers as black-box PyTorch modules, this repository is intended to serve as a controlled experimental framework for implementing and analyzing optimizer algorithms from the ground up.