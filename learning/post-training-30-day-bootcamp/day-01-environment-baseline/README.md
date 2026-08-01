# Day 01 — 环境与可复现基线

日期：`2026-07-27`  
状态：`done`（用户确认 Day 01 已完成；保留原清单，不补写未记录的 artifact）

强度：工作日 4–5 小时

## 今日结果

建立可复现项目环境。今天不追求训练效果，追求以后任何实验都能回答“用什么代码、模型、数据和硬件跑的”。

## 时间安排

- 60 分钟：完成 [Day 01 精读清单](../SCALING-BOOK-READING-GUIDE.md#day-01)，建立 Roofline 术语表。
- 90 分钟：clone、安装并固定 `ms-swift` commit。
- 60 分钟：记录 GPU、CUDA、PyTorch、拓扑和存储情况。
- 60 分钟：创建实验命名与日志约定，测试 Qwen 模型下载/加载。
- 30 分钟：完成环境报告和进度表。

## 理论

精读清单：[Day 01 — 术语、上下界与阅读地图](../SCALING-BOOK-READING-GUIDE.md#day-01)。Core 是 Part 0 的全书地图与 Part 1 `Where Does the Time Go?`；今天停在 matmul 推导之前。

今日与 Day 02 合并进行互动阅读时，使用 [`DAY-01-02-READING-QUIZ.md`](./DAY-01-02-READING-QUIZ.md) 跟踪原始回答、点评和修正版结论。

- [x] Day 01 Reading Quiz：`6/6 completed`（完成于 `2026-07-24`）。
- [ ] GPU/环境基线与 CUDA smoke test。

电子版结构与显存图：

- [`../artifacts/reports/day-roadmaps/day-01-roofline-foundations-roadmap.svg`](../artifacts/reports/day-roadmaps/day-01-roofline-foundations-roadmap.svg)
- [`../artifacts/reports/training-memory-visual-guide/dense-transformer-training-memory.svg`](../artifacts/reports/training-memory-visual-guide/dense-transformer-training-memory.svg)
- [`../artifacts/reports/training-memory-visual-guide/qwen3-30b-a3b-moe-training-memory.svg`](../artifacts/reports/training-memory-visual-guide/qwen3-30b-a3b-moe-training-memory.svg)
- [`../artifacts/reports/training-memory-visual-guide/README.md`](../artifacts/reports/training-memory-visual-guide/README.md)

- 可复现训练的最小状态：code、config、model revision、dataset revision、environment、seed、hardware。
- 理解 CUDA driver、CUDA runtime、PyTorch build 和 Flash Attention 版本的关系。

## Coding

- 创建环境快照命令或脚本，能够一次导出关键版本、Git commit、GPU 拓扑和磁盘信息。
- 建立 run ID 与 artifacts 路径约定。

## 训练 / 实验

- 今天不做正式训练；只加载 Qwen config/tokenizer，并做一个 1–5 sample inference/CUDA smoke。
- 成功标准：环境能调用 GPU，模型/tokenizer 路径和缓存已确认。

## 资源与租卡

- 主要工作：本地 CPU 或 AutoDL 无卡模式。
- GPU：任意 1×CUDA GPU 30–60 分钟完成兼容性验证；若已确定后续都用 H100，可直接 1×H100。
- 数据盘：至少预留模型、cache、checkpoint 所需空间；今天记录，不盲目扩容。
- 完成后：保存环境信息并关机。参见 [`../AUTODL.md`](../AUTODL.md)。

## Core

- [ ] 在工作区 clone [ms-swift](https://github.com/modelscope/ms-swift)，记录 commit hash。
- [ ] `swift sft --help`、`swift infer --help` 能运行。
- [ ] 记录 `nvidia-smi`、`nvidia-smi topo -m`、Python、CUDA、Torch、Transformers、ms-swift 版本。
- [ ] 确认模型、数据、checkpoint、日志分别存在哪里，预估磁盘余量。
- [ ] 下载或至少成功读取 `Qwen/Qwen3-0.6B-Base` config/tokenizer。
- [ ] 更新 [`../PROGRESS.md`](../PROGRESS.md)。

## 建议命令

```bash
git clone https://github.com/modelscope/ms-swift.git vendor/ms-swift
cd vendor/ms-swift
git rev-parse HEAD
pip install -e .
python -m pip freeze
nvidia-smi
nvidia-smi topo -m
```

不要为了“最新”每天更新 `main`。从今天起固定 commit，除非遇到已确认的框架 bug。

## 必须回答

1. 当前 H100 的显存、driver 和 CUDA 版本是什么？
2. 模型与数据从 Hugging Face 还是 ModelScope 下载？缓存位置在哪里？
3. 训练日志、checkpoint 和 Git repo 是否在同一块盘？
4. 一次训练如何唯一标识其 code、config、dataset 和 seed？

## 产物

- `../artifacts/logs/environment.md`
- `../artifacts/logs/pip-freeze.txt`
- `../artifacts/logs/hardware-topology.txt`
- 本文件底部的 Daily Log

## Definition of Done

- 从新 shell 进入环境后，5 分钟内能打印所有关键版本并加载 Qwen tokenizer。
- 环境报告包含 ms-swift commit，不只写“最新版”。
- 磁盘和缓存路径明确，不会在长训练中途才发现空间不足。

## Daily Log

### 实际用时

### 今日 commit / 环境

### 遇到的问题

### 最重要结论

### Day 02 第一动作
