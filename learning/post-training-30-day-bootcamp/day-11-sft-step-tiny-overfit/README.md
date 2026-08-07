# Day 11 — Qwen3-0.6B Tiny Overfit 与 SFT Step 语义

日期：`2026-08-06`

状态：`ready_for_gpu`（CPU 前置、冻结输入、脚本和上传包已于 2026-08-05 准备）

强度：工作日 4–5 小时

## 主要目标

用 `Qwen/Qwen3-0.6B-Base` 在一组极小、人工验证过的数据上刻意过拟合，把一个 SFT step 的 token、loss、gradient、optimizer、scheduler 和 checkpoint 语义逐项对齐。今天验证的是训练管线正确性，不是模型能力。

## 理论（60 分钟）

精读清单：[Day 11 — SFT step semantics 与 tiny overfit](../SCALING-BOOK-READING-GUIDE.md#day-11)。

- masked causal LM loss 的 numerator/denominator；batch 中 supervised-token 数不同时，平均方式如何影响梯度权重。
- micro batch、gradient accumulation、global batch 与 optimizer step 的关系。
- BF16 forward/backward、gradient clipping、AdamW update、LR scheduler、`zero_grad` 的正确顺序。
- parameter、gradient、optimizer state、scheduler、RNG、dataloader position 在 step 前后的变化。
- tiny overfit 只能证明 pipeline 能学这批 token，不能证明泛化、数据质量或 recipe 有效。

## Coding（90 分钟）

- 从 Day 08 golden cases 和 Day 09 审核通过的数据中冻结 16–32 条短样本及 hash。
- 固化 0.6B full-SFT config；显式记录 seed、dtype、max length、batch、accumulation、LR、warmup、clip 和 loss mask。
- 为前 3 个 optimizer steps 记录：
  `sample IDs -> supervised tokens -> loss -> grad norm -> clip -> LR -> parameter checksum -> global supervised tokens`。
- 保存一个中间 checkpoint，并验证 resume 后的 optimizer step、scheduler、RNG 与 sampler position；不把“只加载权重”记作 resume 成功。

## 训练 / 实验（120–150 分钟）

- 先对 tiny set 生成 Base 输出并计算 teacher-forced loss/token accuracy。
- 重复 tiny set 训练 50–150 optimizer steps；若预注册过拟合阈值提前达到则停止。
- 保存 early/final checkpoint，并对同一 tiny set 重新计算 loss、teacher-forced token accuracy 与 deterministic generation。
- Stop：零有效 label、sample ID 顺序异常、NaN、grad norm 持续为 0、checkpoint 无法恢复时立即停止排查。

## 资源与租卡

- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [ms-swift SFT documentation](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html)
- 推荐 1×RTX 4090 24GB；本实验拒绝低于 20 GiB、无原生 BF16 或非 CUDA 的设备。预计租卡窗口 1–3 小时。
- 完成后同步 config、tiny manifest、step trace 和 checkpoints，再关机。

## 已完成的 CPU 前置

- 冻结 18 条 tiny 样本：Day 08 的 10 条正向 golden cases，以及逐条内容检查的 Day 09 general/code/finance/math 各 2 条。
- 使用 exact Qwen3 tokenizer/template 审计得到 2,194 个 encoded tokens、904 个 shifted supervised tokens，所有样本无截断。
- 冻结 Base revision、模型文件 hash、数据 hash、chat-template hash、loss mask、FP32 参数/梯度/Adam state + BF16 autocast compute + FP32 loss、batch/accumulation、warmup、clip、停止阈值和磁盘 gate。
- 实现 supervised-token-weighted accumulation；不会把不同 label 数的 microbatch 做错误的等权平均。
- 实现前 3 optimizer steps 的逐步 trace、Base/early/final 同协议评估和 deterministic generation。
- 实现 6-step uninterrupted vs `3 + fresh-process resume + 3` 的 exact comparison；恢复 model、optimizer、scheduler、RNG 与 sampler cursor。
- 本地 7 个单元测试和 frozen-data deterministic rebuild 已通过。

## 明日唯一流程

详细命令见 [`AUTODL-RUNBOOK.md`](./AUTODL-RUNBOOK.md)。上传本地已准备的 `tmp/day11-ready-upload.tar` 和同名 `.sha256` 文件到 AutoDL 的 `/root/autodl-tmp/`，然后执行：

```bash
cd /root/autodl-tmp
sha256sum -c day11-ready-upload.tar.sha256
tar -xf day11-ready-upload.tar
cd /root/autodl-tmp/day11-ready
bash run-day11.sh
```

入口脚本会自动完成依赖检查、preflight、resume probe、main tiny overfit 和最终验收。只有 `DAY11-PASS.json` 存在且状态为 `day11_pass` 才算完成。

## 产物

- `../artifacts/configs/day11-qwen3-0.6b-tiny-overfit.yaml`
- `../artifacts/data/day11-tiny-overfit.jsonl`
- `../artifacts/data/day11-tiny-overfit-manifest.json`
- `../artifacts/logs/day11-first-three-steps.jsonl`
- `../artifacts/reports/day11-sft-step-audit.md`
- `../artifacts/reports/day11-resume-audit.md`

## 验收

- [ ] 前 3 个 optimizer steps 的 sample、有效 label、accumulation、gradient、LR 和参数变化能逐项对应。
- [ ] tiny set 达到预注册的过拟合阈值；默认目标为 non-padding assistant tokens 的 teacher-forced accuracy ≥95%，否则解释阻塞。
- [ ] checkpoint resume 恢复训练状态与样本进度，而不是仅能加载模型做 inference。
- [ ] Base/early/final 在完全相同 template 和 decoding 下可比较。
- [ ] 不把 command 成功、loss 有数字或 tiny-set 记忆当作真实 SFT 效果。

## Daily Log

### 一个 optimizer step 的完整证据

### 未达到/达到过拟合阈值的原因

### Day 12 第一动作
