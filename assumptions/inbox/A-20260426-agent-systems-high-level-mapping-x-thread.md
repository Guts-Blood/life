---
id: A-20260426-agent-systems-high-level-mapping-x-thread
title: Agent 系统状态转移映射 X Thread 草稿
status: inbox
domain: AI systems
created_at: 2026-04-26
updated_at: 2026-04-26
confidence: 0.5
source: derived from A-20260426-agent-systems-high-level-mapping
related_profile:
related_experiment:
review_at:
verified_at:
falsified_at:
supersedes:
---

# Agent 系统状态转移映射 X Thread 草稿

## 版本 1：偏系统理论

### 1/12

我最近在尝试用状态机、概率转移矩阵和收敛时间的视角，重新理解 Agent 系统。

一个高层直觉是：

Agent 不是单次生成答案，而是一个受控的随机状态转移系统。

### 2/12

如果把 Agent 当前所处的情况抽象成状态 `S_t`，那么每一步都可以看成：

```text
S_t -> S_{t+1}
```

这里的状态不只是 prompt，也不只是上下文窗口，而是整个系统当下可用于决策的抽象状态。

### 3/12

`S_t` 里可能包含：

- 用户目标
- 当前任务进度
- 已知约束
- 工具返回结果
- 当前不确定性
- 已经失败过的尝试
- 当前控制阶段
- 预算、权限和时间限制

状态设计得好不好，会直接决定 Agent 是否真的能稳定行动。

### 4/12

如果暂时不考虑 action，只看状态之间的变化，我们可以写成：

```text
P(s' | s)
```

也就是：当前在状态 `s` 时，下一步转移到 `s'` 的概率。

这就是 Agent 系统诱导出来的状态转移行为。

### 5/12

但这个 `P` 不是模型单独决定的。

它由很多东西共同诱导出来：

- model
- system prompt
- context
- action space
- tools
- planner
- harness
- verifier
- memory
- user feedback
- external environment

所以 Agent engineering 本质上是在塑造这个 `P`。

### 6/12

一个任务的目标通常不是某个唯一状态，而是一个目标状态集合：

```text
G = goal states
```

比如：问题被正确回答、代码通过测试、文件被正确生成、系统在信息不足时提出正确澄清问题。

### 7/12

同时也有失败状态集合：

```text
F = failure states
```

比如：错误答案、错误工具调用、死循环、过早完成、上下文污染、越权操作、在错误方向上持续推进。

所以我们真正关心的是：

```text
Pr(reach G before F)
```

### 8/12

这也是为什么我觉得 Agent 不能只用“单次输出质量”来评估。

更重要的是整个状态转移过程：

- 成功状态命中率
- 失败状态命中率
- false done 概率
- loop 概率
- 平均完成步数
- 工具错误率
- repair 次数

### 9/12

系统理论里会谈 mixing time。

但在 Agent 任务里，更直接的指标可能是 hitting time：

```text
tau_G = first time reaching G
```

也就是从初始状态出发，第一次命中目标状态集合需要多久。

### 10/12

工程上，`tau_G` 可以对应：

- 模型轮次
- 工具调用次数
- token cost
- wall-clock latency
- planner expansion 次数
- 用户介入次数
- repair 次数

也就是说，“更快收敛”最终会落到具体成本和体验上。

### 11/12

在这个框架里：

```text
Plan    = 对未来成功路径的假设
Harness = 对合法转移和恢复路径的控制
```

Plan 让系统更可能走上通向 `G` 的路径。

Harness 则阻止坏转移、检测 false done、处理 retry、控制终止，并提供 escape path。

### 12/12

所以我目前的高层理解是：

一个好的 Agent 系统，是在给定任务分布下，能够高概率、低成本、可恢复地从 `S_0` 到达目标状态集合 `G` 的受控随机状态转移系统。

Agent engineering 的核心，不只是写 prompt，而是把概率质量从坏路径搬到好路径上。

## 版本 2：更短的单帖

我最近在用状态机和概率转移矩阵的角度理解 Agent 系统。

一个高层直觉：

Agent 不是单次生成答案，而是一个受控随机状态转移系统。

```text
S_t -> S_{t+1}
```

系统每一步都从当前状态转移到下一个状态。这个转移概率 `P(s' | s)` 不是模型单独决定的，而是由 model、prompt、context、tools、planner、harness、verifier、memory、用户反馈和外部环境共同诱导出来的。

目标也不是某个唯一状态，而是目标状态集合 `G`。失败也不是一个点，而是失败状态集合 `F`。

所以我们真正想优化的是：

```text
maximize   Pr(reach G before F)
minimize   E[tau_G]
minimize   cost
```

其中 `tau_G` 是第一次到达目标状态集合的时间。

在这个框架里：

```text
Plan    = 对未来成功路径的假设
Harness = 对合法转移、验证、恢复和终止的控制
```

一个好的 Agent 系统，本质上是在把概率质量从 bad transitions、false done、loop、stuck state 搬到通往目标状态的路径上。

这也解释了为什么 Agent engineering 不只是 prompt engineering，而是 state abstraction、policy shaping、transition control、verification 和 recovery 的整体设计。

