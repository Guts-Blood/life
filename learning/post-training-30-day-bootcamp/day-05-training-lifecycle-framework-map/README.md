# Day 05 — End-to-End Training Lifecycle 与框架边界

日期：`2026-07-31`

状态：`done`（guided quiz、`swift sft` codepath walkthrough 与 tiny causal-LM 状态机完成）

强度：工作日 4–5 小时

## 主要目标

从一条 raw sample 追到 checkpoint 与 eval，建立完整 training lifecycle；同时明确用户代码、训练 repo、框架、通信库和 CUDA 各自负责什么。结果是能追踪和诊断训练，不是搭一个 auto-train 平台。

## 理论（75 分钟）

精读清单：[Day 05 — Training lifecycle 与 framework boundaries](../SCALING-BOOK-READING-GUIDE.md#day-05)。

- 45 分钟：按状态变化梳理
  `raw data -> template/tokenize -> collate -> forward -> masked loss -> backward -> gradient sync/accumulation -> clip -> optimizer/scheduler -> zero_grad -> checkpoint/resume -> eval`。
- 30 分钟：保留《How To Scale Your Model》Part 5 的 DP/FSDP/TP/PP 小节，只回答某种并行改变了 lifecycle 的哪一步、哪类状态和哪次通信。
- 区分 model state、optimizer state、scheduler、RNG、sampler/dataloader progress；解释为什么只加载 model weights 不等于连续 resume。

## Coding / 源码追踪（135 分钟）

- 选择一条固定的 `swift sft` 命令，追踪 `CLI/config -> dataset/template -> collator -> model forward -> loss -> trainer step -> optimizer -> checkpoint`。
- 在每个边界记录 `input/output object`、状态所有者、关键默认值和失败后应查看的日志。
- 完成框架职责表：
  - 训练 repo/ms-swift：recipe、模型/数据适配、入口与配置；
  - Transformers/TRL：model/trainer 与目标函数；
  - Accelerate/DeepSpeed/FSDP/Megatron：进程、状态切分与训练 runtime；
  - PyTorch autograd/optimizer：本地 forward/backward/update；
  - NCCL：被调用后执行 collective；
  - CUDA/cuBLAS/Triton：单卡 kernel 与本地矩阵计算；
  - slime/SGLang：后续 online RL 的 orchestration 与 rollout，不属于今天的 SFT trainer 主链。

## 训练 / 实验（45–60 分钟）

- CPU dry path 或离线源码 trace；不做正式训练。
- 用一条合成 sample 手工记录每一阶段的 shape、dtype、device、有效 label tokens 和需要保存的恢复状态。
- 做一次 resume 纸面演练：分别遗漏 optimizer、RNG、dataloader progress，预测曲线或样本顺序会如何变化。

## 资源与租卡

- [ms-swift Pre-training and Fine-tuning](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [DeepSpeed Training](https://www.deepspeed.ai/training/)
- [Training at Scale](https://jax-ml.github.io/scaling-book/training/)
- CPU only；不为源码阅读启动 H100。

## 产物

- `../artifacts/reports/day05-training-lifecycle.mmd`
- `../artifacts/reports/day05-framework-responsibility-map.md`
- `../artifacts/reports/day05-resume-state-checklist.md`

## 验收

- [ ] 能不看文档完整讲述一个 SFT step 的顺序和状态变化。
- [ ] 能指出 sharding/collective 由哪层决定、哪层执行，而不说“CUDA 自动切分”。
- [ ] 能从一条 `swift sft` 命令定位 dataset、loss、optimizer 和 checkpoint 的主要边界。
- [ ] 能列出可连续 resume 所需状态，并解释遗漏任一关键状态的后果。
- [ ] 源码阅读以 boundary map 为止，不扩成 Megatron/slime 全仓通读。

## Daily Log

### 最意外的框架边界

### 一项隐藏默认值

### Day 06 第一动作
