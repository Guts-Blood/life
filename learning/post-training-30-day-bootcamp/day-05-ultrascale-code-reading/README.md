# Day 05 — Ultra-Scale 与三套框架分层

日期：`2026-07-31`  
状态：`not_started`  
强度：工作日 4–5 小时

## 今日结果

把 Day 04 的抽象映射到教学实现，并明确 ms-swift、Megatron、slime 各自负责哪一层。今天只建立阅读路线，不把快速浏览误写成“通读完成”。

## 资料

- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Picotron](https://github.com/huggingface/picotron)
- [Nanotron](https://github.com/huggingface/nanotron)
- [ms-swift](https://github.com/modelscope/ms-swift)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [slime](https://github.com/THUDM/slime)

## 时间安排

- 75 分钟：完成 Scaling Book Part 5 的 DP/FSDP/TP/PP 精读。
- 45 分钟：读 Ultra-Scale/Picotron 中对应短实现。
- 60 分钟：追踪 ms-swift 一条 SFT 命令从 CLI 到 trainer/config。
- 60 分钟：只读 Megatron/slime 的 project structure、入口和架构说明。
- 30 分钟：画三套框架的分层图和后续必读主链路。

## 理论

精读清单：[Day 05 — DP/FSDP/TP/PP 的 roofline](../SCALING-BOOK-READING-GUIDE.md#day-05)。每节只抓“切什么、复制什么、通信什么、临界条件、最常见误用”，再映射到实现。

- 把并行策略从公式映射到 collective、process group、参数 ownership 和训练生命周期。

## Coding

- 阅读 Picotron/ms-swift 真实代码，输出 concept-to-code map。
- 为 Megatron 标出 `entry -> data/model -> parallel runtime -> optimizer -> checkpoint`。
- 为 slime 标出 `entry -> Ray roles -> rollout/buffer/reward -> Megatron train -> weight sync`。

## 训练 / 实验

- 无正式训练。可以运行 CPU dry path 或最小单进程测试验证 CLI/config 解析。

## 资源与租卡

- CPU only 或 AutoDL 无卡模式。
- 如果 vendor repo/模型缓存已经在 AutoDL，使用无卡模式读代码；不要占用 H100。

## Core

- [ ] 在 Picotron 中定位 DP/FSDP/TP 至少三个实现入口。
- [ ] 找到 all-reduce、all-gather、reduce-scatter 对应代码。
- [ ] 在 ms-swift 找到 `swift sft` 的 CLI 入口、dataset/template 和 trainer 入口。
- [ ] 记录 framework 帮你隐藏了哪些复杂性。
- [ ] 记录哪部分属于 Transformers/TRL、哪部分属于 ms-swift、哪部分属于 DeepSpeed/Megatron。
- [ ] 能解释 slime 为什么是 Megatron+SGLang 之上的 RL orchestration，而不是另一个 trainer。

## Code Reading 模板

```text
Concept:
Educational implementation:
Production implementation:
ms-swift configuration:
Collective/communication:
State owned by each rank:
Open question:
```

## Stretch

- [ ] 用 debugger 或打印调用栈跑一次 `swift sft --help`/dry path。
- [ ] 找到 checkpoint save 和 resume 的最终实现位置。

## 产物

- `../artifacts/reports/concept-to-code-map.md`
- `../artifacts/reports/ms-swift-sft-call-chain.mmd`
- `../artifacts/reports/post-training-framework-stack.mmd`

## Definition of Done

- 不看文档，能够说出一条 `swift sft` 命令经过的主要层级。
- 至少找到三个 collective 在教学/生产代码中的真实位置。
- 能区分“理解算法”与“记住框架参数”。
- Megatron/slime 只完成了阅读计划和入口定位；明确记录为 `first_pass`，不能标记为通读。

## Daily Log

### 阅读过的 commit

### 最重要的调用链

### Day 06 第一动作
