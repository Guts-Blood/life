# Day 17 — Megatron SFT 主链路通读与运行准备

日期：`2026-08-12`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

完成 Megatron SFT 主链路的第一遍端到端通读，固定环境和 commit，并把第二天的 upstream smoke、Qwen SFT、插桩和停止条件写成可执行 runbook。

## 理论（45 分钟）

精读清单：[Day 17 — Megatron parallel ownership 与 checkpoint](../SCALING-BOOK-READING-GUIDE.md#day-17)。重画 TP/PP/DP process groups，并区分 model/optimizer/RNG/dataloader state。

## Repo 通读（180 分钟）

固定一个 Megatron commit，按调用而不是目录顺序读：

1. **入口与 loop（40 分钟）**：`pretrain_gpt.py`、`megatron/training/training.py`；找到 argument parse、initialize、model provider、dataset provider、train loop。
2. **data 与 model（45 分钟）**：`megatron/training/datasets/sft_dataset.py`、`megatron/core/models/gpt/gpt_model.py`、Transformer block/layer；标出 input、labels、loss mask 的 shape。
3. **parallel runtime（55 分钟）**：`parallel_state.py`、`pipeline_parallel/schedules.py`、`tensor_parallel/layers.py`/`mappings.py`；标出每个 collective 的 caller 与 group。
4. **optimizer/checkpoint（40 分钟）**：定位 distributed optimizer、`core/dist_checkpointing`、save/load 入口；列出 resume 所需状态。

禁止只读 README。调用链必须写到 `file:function`，看不懂的边标 `UNKNOWN`，第二天用 runtime 验证。

## Coding / 运行准备（45–60 分钟）

- Clone/fetch 后固定 Megatron、Megatron-SWIFT/Mcore-Bridge commit 与容器 digest。
- 准备 Run A：官方 Megatron 2-rank synthetic GPT，10–20 steps。
- 准备 Run B：同一份 Day 09 数据的 Qwen3-1.7B/4B Megatron SFT，10–30 steps。
- 准备 per-rank rank/group/memory logger；在 `forward_step` 或 hook 中插入 tensor-shape 观测点，保存为独立 diff。
- 完成 HF↔Megatron checkpoint 转换命令的 dry review，不在 GPU 上临时猜 model args。

## 训练 / 实验

- 今天不开始正式训练；只允许 CPU/config validation、容器 import test 和转换命令 dry-run。
- 成功标准不是出现 loss，而是 Day 18 的两个 run 能在不查临时网页、不改 model args 的情况下直接启动。

## 资源与租卡

- CPU/no-card；checkpoint conversion 如确实需要可短用 1×H100，不开始训练。
- 推荐直接使用与 Day 18 相同的官方/固定镜像，今天构建或拉取，明天不编译环境。
- 确认 Day 18 的同机 2×H100、NVLink/topology、磁盘和共享内存。

## 资料

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron-SWIFT Quick Start](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html)

## 验收

- [ ] 主链路图至少覆盖入口、dataset、model、schedule、TP mapping、optimizer、checkpoint。
- [ ] 每条边有 `file:function` 或显式 `UNKNOWN`，不是模块名列表。
- [ ] Run A/Run B 的 config、预期 process groups、日志和停止条件已写好。
- [ ] 插桩 diff、checkpoint conversion 和 Day 18 第一条命令已 dry review。

## Daily Log

### 读完的文件 / commit

### UNKNOWN edges

### 多卡 runbook

### Day 18 第一动作
