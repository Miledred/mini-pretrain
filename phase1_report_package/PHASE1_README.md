# Phase 1 README：AdamW Baseline 整理报告

## 1. 阶段一目标

本阶段目标不是重新训练模型，而是把已经完成的 AdamW 预训练实验整理成可汇报、可复现、可继续对比的 baseline。具体包括：

1. 汇总 L4 与 L8 两组 AdamW 训练结果；
2. 检查 train loss、validation loss、perplexity、learning rate、gradient norm、update norm、update ratio、throughput、memory；
3. 固定 prompt 做 greedy / sampling 采样，作为后续 Muon 对比的 qualitative baseline；
4. 明确下一阶段 Muon optimizer 的实验计划。

---

## 2. 已完成的实验能力

当前 mini_pretrain 项目已经形成完整的小规模 GPT-style pretraining pipeline：

| 模块 | 已完成内容 | 阶段一产物 |
|---|---|---|
| 数据构建 | WikiText → GPT-2 tokenizer → `train.bin` / `val.bin` / `data_meta.json` | 可复用 tokenized 数据 |
| 模型 | GPT-style causal LM，支持 L4/L8 两档配置 | L4 debug；L8 formal baseline |
| 优化器 | 手写 `MyAdamW`，支持 bias correction、decoupled weight decay、参数分组 | 后续可扩展 Muon / SOAP |
| 训练系统 | warmup + cosine decay、gradient clipping、step-based training | 50k steps 完整训练 |
| 日志系统 | CSV + TensorBoard | loss / ppl / grad / update / speed / memory 曲线 |
| checkpoint | `best.pt`、`latest.pt`、每 5000 step periodic checkpoint | 可恢复、可采样、可比较 |
| 定性评估 | 固定 5 个 prompt，greedy + sampling | `sample_outputs_L4.md`、`sample_outputs_L8.md` |

---

## 3. Run Inventory

| Run | Final step | Best val loss | Best step | Final train loss | Final val ppl | Avg tok/s last100 | Max memory | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L4-H4-D256 | 50000 | 5.5539 | 50000 | 5.8432 | 258.24 | 43,747.50 | 2,021.65 MB | best/latest + 10 periodic |
| L8-H8-D512 | 50000 | 5.3346 | 50000 | 5.5276 | 207.39 | 17,663.53 | 3,174.18 MB | best/latest + 10 periodic |

### 关键对比

- L8 相比 L4 的 best validation loss 更低：`5.3346 vs 5.5539`，相对改善约 **3.95%**。
- L8 的 validation perplexity 更低：`207.39 vs 258.24`，相对降低约 **19.69%**。
- L8 代价更高：吞吐从约 `43.7k tok/s` 降到 `17.7k tok/s`，显存从约 `2.0GB` 增加到 `3.2GB`。
- 两组 best step 都在 `50000`，说明 validation loss 到训练结束仍在下降，没有明显 overfitting rebound。

---

## 4. 曲线结论

### 4.1 Loss 与 PPL

- L8 train loss 从约 11 降到约 5.53。
- L8 validation loss 从约 7.9 降到 5.3346。
- L8 validation perplexity 从约 2800 降到 207.39。
- validation loss 没有后期回升，因此当前 50k steps 训练是稳定的。

### 4.2 Optimizer Dynamics

- learning rate 按 warmup + cosine decay 正常变化。
- update norm 和 update ratio 在 warmup 后达到峰值，之后随 learning rate 平滑下降。
- gradient norm 有早期 transient peak，但后续整体 bounded，没有训练崩溃迹象。

### 4.3 System Metrics

- L4 速度快，适合作为 Muon smoke test / debug 模型。
- L8 指标更好，适合作为正式 AdamW baseline。
- 两组 checkpoint 都完整，后续可以直接 resume 或 sample。

---

## 5. 固定 Prompt 采样结果

固定使用 5 个 prompt，每个 checkpoint 记录 greedy 与 sampling。

| Prompt | prompt tokens | max_new_tokens | decoding |
|---|---:|---:|---|
| The history of science | 4 | 80 | Greedy + Sampling |
| In the early twentieth century | 5 | 80 | Greedy + Sampling |
| Artificial intelligence is | 4 | 80 | Greedy + Sampling |
| The theory of evolution states that | 6 | 80 | Greedy + Sampling |
| In modern society, technology has | 6 | 80 | Greedy + Sampling |

### 5.1 L8 Sampling 片段

| Prompt | L8 sampling excerpt |
|---|---|
| The history of science | `The history of science and literature both and in science it been by historians...` |
| In the early twentieth century | `In the early twentieth century it an in form classical, style music a of and music...` |
| Artificial intelligence is | `Artificial intelligence is to a, or. , ' is the of who that are to...` |
| The theory of evolution states that | `The theory of evolution states that are and, with, the of, in...` |
| In modern society, technology has | `In modern society, technology has more a of @@ jobs the rate time to...` |

### 5.2 定性分析

- Greedy decoding 明显容易重复，例如 `the of`、`as as`、连续逗号、`@@`。
- Sampling 比 greedy 更自然，能出现 science、literature、evolution、technology 等主题相关词。
- L8 的 quantitative metrics 明显优于 L4，但 raw generation 的可读性提升有限。
- 当前 checkpoint 只是 raw pretrained LM，没有做 SFT，因此不能用“是否会回答问题”来评价预训练是否失败。
- `@@` 很可能与 WikiText 原始文本格式和 tokenizer/detokenization 有关，后续可在数据清洗或解码后处理中处理。

---

## 6. 阶段一结论

阶段一已经完成 AdamW baseline 的整理和验证：

1. 两组 AdamW baseline 均完整训练到 50k steps；
2. L8 模型在 best val loss 和 val ppl 上明显优于 L4；
3. validation loss 在 50k 仍然下降，没有明显过拟合回升；
4. optimizer-level 指标已经记录，包括 grad norm、update norm、update ratio；
5. speed 和 memory 已经记录，可用于后续比较 Muon 的代价；
6. 固定 prompt 采样已经完成，为后续 qualitative comparison 提供基准。

一句话总结：

> 阶段一已经证明当前项目具备完整 pretraining baseline 能力；下一步可以从工程搭建转入 optimizer behavior comparison。

---

## 7. 下一阶段计划：Muon 接入与公平对比

| 阶段 | 实验 | 模型 | Steps | 成功标准 | 产物 |
|---|---|---|---:|---|---|
| 2.1 | Muon smoke test | L4-H4-D256 | 2k–5k | loss 不爆；update ratio 正常 | smoke metrics + 曲线 |
| 2.2 | 快速对比 | L4-H4-D256 | 10k | 与 AdamW 前 10k 对齐比较 | quick comparison table |
| 2.3 | 正式对比 | L8-H8-D512 | 50k | 同预算比较 val loss / ppl / speed / memory | AdamW vs Muon formal report |
| 2.4 | 定性复验 | L8 best checkpoint | fixed prompts | 使用同一 prompts 采样 | `sample_outputs_muon.md` |
| 2.5 | 可选 SFT sanity check | AdamW / Muon best checkpoint | 1–3 epochs | 区分 pretraining 能力与 instruction following 能力 | SFT appendix |

核心研究问题：

> 在相同 data / model / batch / scheduler / max_steps 下，Muon 是否比 AdamW 更快降低 validation loss？如果有效，它的代价是更慢吞吐、更高显存，还是更大的 update ratio？

---

## 8. 主要文件说明

| 文件/目录 | 用途 |
|---|---|
| `logs/<run_name>/metrics.csv` | 完整训练指标，曲线来源 |
| `logs/<run_name>/config.json` | 模型与训练配置 |
| `checkpoints/<run_name>/best.pt` | best validation loss 对应模型 |
| `checkpoints/<run_name>/latest.pt` | 最新训练状态，可 resume |
| `checkpoints/<run_name>/step_*.pt` | 周期性 checkpoint |
| `run_inventory.csv` | 阶段一机器可读 summary |
| `phase1_baseline_report.md` | 阶段一自动生成报告 |
| `sample_outputs_L4.md` | L4 固定 prompt 采样 |
| `sample_outputs_L8.md` | L8 固定 prompt 采样 |
| `phase1_adamw_baseline_report.pptx` | 给老师汇报用 PPT |
