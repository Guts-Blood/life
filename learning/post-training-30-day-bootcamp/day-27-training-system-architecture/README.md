# Day 27 — Training System 总图：框架分层与节点关系

日期：`2026-08-22`  
状态：`done`（guided Core Quiz `3/3`；slime architecture、node ledger 与 failure triage 合并 SVG 完成）<br>
强度：1 小时，仅阅读、画图与口述

## 课程转向

Day 27–30 不再以“换一个框架再跑通一次训练”为目标。Day 23/25 已经证明我们能借助工具完成 DPO/GRPO 实跑；接下来更重要的是建立可迁移的系统心智模型：看到任意 recipe 时，能判断谁在编排、谁在计算、谁持有状态、对象沿哪条边移动，以及故障首先属于哪一层。

Day 26 的 slime runtime blocker 保留为历史事实，但不阻塞本阶段的 CPU 源码学习。除非学习过程中出现只有 runtime 才能回答的具体问题，否则 Day 27–30 不租 GPU。

## 主要目标

先画出两条不同但相连的系统链：

```text
Offline SFT / DPO
raw data -> processor/template -> learner batch -> training engine
         -> PyTorch kernels/collectives -> optimizer state -> checkpoint/eval

Online RL
prompt source -> rollout engine -> environment/reward -> trajectory buffer
              -> learner batch -> training engine -> new weights
              -> weight sync -> next-version rollout
```

对图中每个节点固定回答四个问题：

1. 谁创建和调度它？
2. 它消费、产出什么对象？
3. 它持有哪些长期或临时状态？
4. 下游如何证明自己消费了正确版本？

## 60 分钟安排

- 10 分钟：回看 [Day 05 framework map](../day-05-training-lifecycle-framework-map/README.md)，区分 recipe、orchestrator、training engine、communication runtime 和 kernel。
- 20 分钟：分别画 offline training 与 online RL 两张图，不先写具体框架名，只写职责和对象。
- 20 分钟：把 `ms-swift / slime / Ray / SGLang / Megatron Core / PyTorch / NCCL / CUDA / reward sandbox` 放回对应节点；一个框架可以跨多个节点，但每项职责必须落到具体组件。
- 10 分钟：不看图口述一遍“一个 sample、一个 optimizer step、一个新 policy version”如何流动。

## 第一版职责表

| 层 | 要回答的问题 | 本阶段重点实例 |
|---|---|---|
| Experiment / recipe | 任务、数据、目标、超参和入口由谁定义？ | ms-swift recipe、slime launch config |
| Orchestration / control plane | 哪些角色何时启动、放在哪些资源上、谁等待谁？ | slime、Ray actors / placement |
| Data plane | prompt、trajectory、reward、learner batch 如何传递？ | slime `Sample` / `DataSource` / buffer |
| Rollout / serving | 谁持有推理权重与 KV cache，如何生成？ | SGLang rollout engines |
| Learner / training engine | 谁构造模型并执行 forward/backward/update？ | Megatron Core 或 HF/TRL 路径 |
| Distributed runtime | 谁定义 group，谁执行 collective？ | Megatron/PyTorch distributed、NCCL |
| Device compute | matmul、attention、optimizer kernel 在哪执行？ | CUDA、cuBLAS、Transformer Engine/Triton |
| Persistence / evidence | 哪些状态可恢复，如何绑定版本？ | distributed checkpoint、HF export、trajectory dump |

`ms-swift`、`slime` 都不是单一方框。它们会组装或委托多个组件；课程要追到真实边界，不能用“框架帮我做了”结束解释。

## 当日产物

- [`day27-slime-training-system-architecture.svg`](../artifacts/reports/day27-slime-training-system-architecture.svg)：把原计划的 layer map 与 node ledger 合并为一张可复查总图，覆盖 slime/Ray/SGLang/Megatron 职责、control/data/weight/evidence flow、对象/version 字段、state ownership 与 failure triage。

Node ledger 至少包含：`node / concrete component / process or Ray actor / input / output / owned state / upstream / downstream / observable evidence / likely failures`。

## 验收

- [x] 能解释 ms-swift、slime、Megatron、SGLang、Ray、PyTorch、NCCL、CUDA 各自处在哪一层，且不把它们说成互斥替代品。
- [x] 能区分 control flow、data flow、weight flow 和 checkpoint/evidence flow。
- [x] 能指出 online RL 相比 SFT 新增了哪些节点和反馈边。
- [x] 每条箭头都写出了传输对象和版本字段，而不是只连接两个框架名字。

## Daily Log

### 最容易混淆的两个边界

1. slime 定义 RL workflow 与对象语义；Ray 负责 actor/process/GPU placement 和异步执行，不理解 trajectory、reward 或 policy loss。
2. Ray actor `alive` 只证明进程存活；SGLang 的 model-ready、loaded policy version、weight hash 与 KV/cache barrier 必须由 serving 层证据证明。

### 仍然无法解释的三条边

概念链已能解释；以下三条在 Day 26 固定 runtime 中仍是 `RUNTIME UNKNOWN`，不能被 Day 27 CPU 图改写成已实跑：

1. Day 21 merged S1 → Megatron conversion/model load。
2. SGLang rollout → live reward → learner optimizer step。
3. Megatron full-weight sync → SGLang loaded-version ACK → next-version rollout。
