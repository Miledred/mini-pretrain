# 阶段一：AdamW baseline 整理执行清单

## 目标

把已有 AdamW baseline 从“跑过一次”整理成“可以汇报、可以作为 Muon 对比基线”的标准实验结果。

## 需要得到的产物

1. `phase1_outputs/run_inventory.csv`
2. `phase1_outputs/phase1_baseline_report.md`
3. 每个 run 的曲线图：
   - `train_loss.png`
   - `val_loss.png`
   - `val_ppl.png`
   - `lr.png`
   - `grad_norm.png`
   - `update_norm.png`
   - `update_ratio.png`
   - `tokens_per_sec.png`
   - `max_memory_mb.png`
4. `phase1_outputs/sample_outputs.md`

## 运行步骤

把两个脚本放到项目的 `tools/` 目录下：

```bash
mkdir -p tools
cp phase1_baseline_audit.py tools/
cp phase1_sample_checkpoints.py tools/
```

在项目根目录运行：

```bash
python tools/phase1_baseline_audit.py --project-root .
```

如果只检查指定 run：

```bash
python tools/phase1_baseline_audit.py \
  --project-root . \
  --run-name my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel
```

采样固定 prompt：

```bash
python tools/phase1_sample_checkpoints.py \
  --project-root . \
  --checkpoint checkpoints/my_adamw_L8_H8_D512_bs8_blk256_ts256_bigmodel/best.pt \
  --out phase1_outputs/sample_outputs.md
```

## 阶段一完成判定

满足以下条件即可进入阶段二 Muon：

- 能找到 `logs/<run_name>/metrics.csv`
- 能找到 `checkpoints/<run_name>/best.pt` 和 `latest.pt`
- `phase1_baseline_report.md` 已生成
- `val_loss` 曲线已经确认：下降 / plateau / 回升三者之一
- 固定 prompt 的 greedy 与 sampling 输出已经保存
- 已经写出一句 AdamW baseline 结论

## 汇报用一句话

AdamW baseline 已经完成标准化整理：训练日志、验证损失、困惑度、学习率、梯度范数、更新范数、update ratio、吞吐、显存和 checkpoint 均已归档；下一步将在同一训练预算和同一日志协议下接入 Muon，比较收敛速度、验证损失和更新动态差异。
