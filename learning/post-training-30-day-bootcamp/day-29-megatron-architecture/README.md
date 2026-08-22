# Day 29 — Megatron Architecture：进程组、状态所有权与训练 Step

日期：`2026-08-24`  
状态：`not_started`  
强度：4–5 小时；CPU 源码学习与已有 evidence 复盘

## 计划修订说明

旧计划要求 Day 29 在 slime 中跑最小 RL 闭环，因此被 Day 26 的 runtime no-go 阻塞。修订后的 Day 29 不再执行该 run，而是系统学习 Megatron。旧 no-go JSON 仍是原实验的有效历史记录，但不阻塞今天的学习任务。

## 主要目标

把 Megatron 从“另一个能跑训练的 backend”还原成一个具体训练引擎：它怎样构造并行模型，怎样组织 process groups，怎样调度 forward/backward，哪些 rank 持有哪些 state，以及 checkpoint 为什么必须携带并行布局与优化器信息。

完成后应能从一份 Megatron config 预测：

- world size 被哪些并行维度分解；
- 一条 sample、一个 activation 和一份 parameter 分别在哪些 rank 上出现；
- global batch 如何计算，哪些并行维度不增加样本数；
- 哪些 collective 出现在 forward、backward 和 optimizer 前后；
- OOM、hang、loss 分叉或 resume 失败首先应查哪层。

## 1. Megatron 在系统中的位置（30 分钟）

先写清边界：

```text
上游：recipe/config、model/data adapter、tokenized learner batch
Megatron：model construction + parallel groups + schedules + distributed optimizer/checkpoint
下游：PyTorch autograd/distributed -> NCCL collectives -> CUDA kernels
旁路：HF conversion/export、eval/serving、slime weight sync
```

Megatron 不负责定义业务 reward，不是 rollout orchestrator，也不自动替你决定数据质量、checkpoint promotion 或 eval protocol。

## 2. 并行维度与 process-group 图（60–75 分钟）

分别解释 DP、TP、PP、CP、EP 和 sequence parallel：

| 维度 | 切分对象 | 是否增加独立样本数 | 主要通信 | 最容易误判的问题 |
|---|---|---:|---|---|
| DP | batch / optimizer state（取决于 optimizer） | 是 | gradient reduce / parameter gather | 把 world size 都乘进 global batch |
| TP | layer 内 parameter/activation | 否 | layer 内 all-reduce / all-gather / reduce-scatter | head/vocab/layout 不可整除 |
| PP | layer stages / microbatches | 否 | stage 间 send/recv | bubble、stage imbalance、loss rank |
| CP | sequence/context | 否 | attention 所需的 context 通信 | 把 context shard 当 batch shard |
| EP | experts | 通常否 | token dispatch / combine | load imbalance、capacity/drop |

选一个小 world-size 示例，画出每个 rank 同时属于哪些 group。必须把“logical role”“OS process”“CUDA device”“distributed rank”分开写。

## 3. 一个训练 step 的源码链（90 分钟）

沿 Day 18 已验证的 Qwen3.5 路径追踪，不全仓通读：

```text
CLI / normalized args
-> model provider and HF↔MCore mapping
-> data iterator / batch / labels
-> forward_step and loss reduction
-> pipeline schedule
-> backward / gradient synchronization
-> optimizer and scheduler step
-> distributed checkpoint / export
```

每个节点记录：

```text
file:function
input object + shape/dtype
executing ranks / process groups
owned or mutated state
collective / point-to-point communication
output and downstream consumer
log or artifact that proves the edge ran
```

优先复用 [Day 18 report](../artifacts/reports/day18-qwen35-megatron-compatibility.md) 中的真实节点：loader、template/batch、provider/conversion、GDN、forward/loss、train step、distributed checkpoint 与 export。

## 4. State ownership ledger（45–60 分钟）

至少追踪以下状态：

- parameters / master weights；
- gradients；
- optimizer moments；
- activations；
- RNG states；
- scheduler / consumed samples or tokens；
- dataloader/sampler cursor；
- process-group and parallel-layout metadata；
- inference export。

对每项标记 `replicated / sharded / transient / persisted / required for continuation / required for exact resume`。只保存 HF weights 时，明确写出丢失了什么。

## 5. 用 Day 18 evidence 做反事实复盘（45 分钟）

对照 `TP=1, DP=2` 与 `TP=2, DP=1` 两个已完成 run，回答：

1. world size 同为 2，为什么 global batch 和 parameter ownership 不能用同一种解释？
2. 为什么 TP2 每卡显存下降不等于吞吐一定更高？
3. C2/C3 的首步 loss 接近能证明什么，不能证明什么？
4. distributed checkpoint、HF export 与 exact resume 各覆盖了不同的哪部分状态？

这是证据解释练习，不追加 GPU，也不把 two-row tiny overfit 外推成 scaling 结论。

## 6. 故障定位演练（30 分钟）

为以下现象写“首查节点 -> 需要的证据 -> 下一分叉”：

- 所有 rank import 成功但初始化 hang；
- TP1 正常、TP2 logits 分叉；
- step 0 OOM；
- loss 正常但 optimizer 后权重不变；
- checkpoint 可加载模型但无法连续训练；
- 导出 HF 后生成结果变化。

## Evidence-first 产物

- `../artifacts/reports/day29-megatron-process-group-map.mmd`
- `../artifacts/reports/day29-megatron-train-step-codepath.md`
- `../artifacts/reports/day29-megatron-state-ownership.md`
- `../artifacts/reports/day29-megatron-failure-tree.md`

## 验收

- [ ] 能从 `TP/PP/CP/EP/DP` 配置推导 group 关系和 global-batch 口径，不把 TP/PP/CP 乘进样本数。
- [ ] 能从一个 learner batch 讲到 optimizer step，并指出每次主要通信发生的原因。
- [ ] 能列出可连续 resume 所需状态，解释 model-only export 的边界。
- [ ] 能说明 Megatron 的上游适配层和下游 PyTorch/NCCL/CUDA 各自负责什么。
- [ ] 所有源码结论都绑定 `file:function` 或已有 runtime evidence；不以“命令跑通”代替架构解释。

## Daily Log

### 一句话解释 Megatron

### 最反直觉的 state ownership

### 仍需 runtime 才能回答的问题
