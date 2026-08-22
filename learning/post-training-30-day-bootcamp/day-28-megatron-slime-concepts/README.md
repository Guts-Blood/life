# Day 28 — Megatron × slime 概念准备：对象、状态与接口

日期：`2026-08-23`  
状态：`not_started`  
强度：1 小时，仅阅读与源码定位

## 主要目标

为 Day 29 的 Megatron 深挖和 Day 30 的 slime 深挖建立共同词汇。今天不追求把仓库“看完”，只把相同词在不同层里的含义拆开，并为两天的源码追踪找到稳定入口。

## 先分清六类对象

| 对象 | 关键问题 | 不能混同为 |
|---|---|---|
| config / recipe | 描述目标与拓扑，最终由谁解析成 runtime 对象？ | 正在运行的进程拓扑 |
| sample / trajectory | 包含哪些 token、mask、reward、status 和 policy version？ | 已可直接训练的 tensor batch |
| learner batch | 哪些维度被 DP/TP/PP/CP 切分，loss 如何归约？ | rollout engine 的 request batch |
| model state | 哪些 rank 持有参数、gradient、optimizer shard？ | rollout server 中可推理的权重副本 |
| policy version | rollout、old、current、reference 分别是哪一版？ | checkpoint step 的字符串别名 |
| checkpoint / export | 能否恢复 optimizer/RNG/data cursor，还是只能推理？ | weight sync 的瞬时传输 |

## 60 分钟安排

- 15 分钟：从 [Day 18 runtime report](../artifacts/reports/day18-qwen35-megatron-compatibility.md) 抽出 Megatron 的 8 个 codepath nodes，按 `input -> output -> state owner` 重写一遍。
- 15 分钟：从 [Day 26 runtime config](../artifacts/configs/day26-slime-qwen35-runtime.json) 抽出 slime 的 `DataSource -> rollout -> sample-to-train -> train -> weight-sync` 链。
- 15 分钟：为 DP、TP、PP、CP、EP、distributed optimizer、Ray actor、placement group、rollout engine、weight sync、policy staleness 各写一句“它改变什么 / 不改变什么”。
- 15 分钟：列出 Day 29/30 必须在源码中回答的问题，不先写猜测答案。

## 源码锚点

Megatron 路径以 Day 18 已验证节点为锚，不按目录漫游：

```text
model loader/provider
-> processor/template and batch
-> forward_step/loss
-> schedule/backward/train_step
-> distributed checkpoint
-> HF/inference export
```

slime 路径以 Day 26 pinned `v0.3.1` static audit 为学习快照：

```text
slime/rollout/data_source.py
-> slime/rollout/sglang_rollout.py
-> slime/ray/rollout.py
-> slime/ray/actor_group.py
-> Megatron learner / weight update
```

这些路径是源码阅读证据，不代表 Day 26 的 runtime 已通过；`static_pass` 与 `runtime_verified` 必须始终分开。

## 当日产物

- `../artifacts/reports/day28-megatron-slime-glossary.md`
- `../artifacts/reports/day28-source-reading-questions.md`

问题清单至少覆盖：对象 schema、进程/rank、state ownership、collective/transport、lifecycle、version、checkpoint 和 failure boundary。

## 验收

- [ ] 能解释为什么 slime 可以调用 Megatron，而两者不是同层的“二选一训练框架”。
- [ ] 能解释 SGLang rollout weights、Megatron training weights 与 resumable checkpoint 的区别。
- [ ] 能解释 Ray actor、GPU process、distributed rank 和 logical role 为什么不是同一个概念。
- [ ] Day 29/30 的源码问题都能落到一个具体对象或一条边，而不是“理解整体架构”这种不可验收目标。

## Daily Log

### 新术语 / 自己的话

### Day 29 首个源码问题
