---
id: A-20260426-agent-systems-high-level-mapping
title: Agent 系统的高层状态转移映射
status: inbox
domain: AI systems
created_at: 2026-04-26
updated_at: 2026-04-26
confidence: 0.5
source: user reflection
related_profile:
related_experiment:
review_at:
verified_at:
falsified_at:
supersedes:
---

# Agent 系统的高层状态转移映射

## 这篇文档想做什么

这篇文档先不深入讨论具体 harness、planner、tool schema、memory、verifier 的工程实现，而是先建立一个高层映射：

```text
状态空间 S
状态转移 P
目标状态 G
失败状态 F
收敛时间 / mixing time / hitting time
```

这些系统理论里的概念，如何和我们真正想要的 Agent 系统对齐。

核心目标是：先把抽象坐标系搭起来。之后再逐层拆解：

```text
high-level stochastic process
-> Agent 系统组件映射
-> plan 和 harness 在改变什么
-> 具体工程设计 checklist
-> eval 和 trace 如何估计转移质量
```

## 第一层抽象：Agent 是一个随机状态转移系统

一个 Agent 系统可以先被高层理解为：

```text
S_t -> S_{t+1}
```

其中 `S_t` 表示系统在第 `t` 步时的状态。这个状态不只是模型上下文，也不只是自然语言 prompt，而是整个 Agent 系统当下可用于决策的抽象状态。

它可能包含：

- 用户 query 的目标。
- 当前任务完成度。
- 当前上下文。
- 已经执行过的步骤。
- 工具返回结果。
- 已知约束。
- 当前不确定性。
- 当前控制阶段。
- 预算、权限、时间限制。

如果把所有可能状态组成一个集合，就得到状态空间：

```text
S = {s_1, s_2, ..., s_n}
```

真实 Agent 的状态空间通常非常大，甚至无法显式枚举。但 high level 上仍然可以把它看成一个状态空间，因为系统每一步都在从一个状态移动到另一个状态。

## 第二层抽象：Agent 的行为诱导出状态转移矩阵

如果暂时不考虑 action，只看状态到状态的变化，可以写成：

```text
P(s_j | s_i)
```

这表示系统当前处于状态 `s_i` 时，下一步转移到状态 `s_j` 的概率。

如果状态空间有限，就可以得到一个状态转移矩阵：

```text
P =
[
  P(s_1 | s_1)  P(s_2 | s_1)  ...  P(s_n | s_1)
  P(s_1 | s_2)  P(s_2 | s_2)  ...  P(s_n | s_2)
  ...
  P(s_1 | s_n)  P(s_2 | s_n)  ...  P(s_n | s_n)
]
```

在 Agent 系统里，这个 `P` 不是天然存在的单一对象，而是由多个组件共同诱导出来的：

```text
P = induced transition behavior of the whole Agent system
```

也就是说，`P` 由以下因素共同决定：

- 模型能力。
- system prompt。
- 当前上下文。
- action space。
- tool set。
- tool schema。
- planner。
- harness。
- verifier。
- memory。
- 用户反馈。
- 外部环境。

所以我们真正关心的不是「能不能显式写出完整矩阵 P」，而是：

```text
Agent 系统的设计，如何改变从坏状态到好状态的转移概率？
```

## 第三层抽象：目标不是单一状态，而是目标状态集合

一个任务通常不是要到达某个唯一状态，而是要到达一类满足目标的状态集合：

```text
G = goal states
```

例如：

- 用户的问题被正确回答。
- 代码修改完成，并且测试通过。
- 文件被正确生成。
- 决策建议满足约束和证据标准。
- 系统承认信息不足并向用户提出正确澄清问题。

同时也存在失败状态集合：

```text
F = failure states
```

例如：

- 错误答案。
- 错误工具调用。
- 死循环。
- 过早完成。
- 上下文污染。
- 违反权限或安全约束。
- 在错误方向上持续推进。

因此，一个好的 Agent 系统不是追求随机过程最终进入任意稳定分布，而是希望：

```text
Pr(reach G before F) 尽可能高
```

也就是：系统从初始状态 `S_0` 出发，更高概率先到达目标状态集合 `G`，而不是失败状态集合 `F`。

## 第四层抽象：收敛时间对应 Agent 的完成效率

从系统理论直觉看，我们希望状态转移过程尽快收敛到希望到达的区域。

可以先把它类比成 mixing time：

```text
系统从初始状态出发，经过多少步后进入稳定的高质量区域？
```

但在 Agent 任务里，更直接的指标通常是 hitting time：

```text
tau_G = first time reaching G
```

也就是第一次到达目标状态集合 `G` 的时间。

在工程里，`tau_G` 可以对应：

- 模型轮次。
- 工具调用次数。
- token 消耗。
- wall-clock latency。
- planner expansion 次数。
- 用户介入次数。
- repair 次数。

所以，高层目标可以写成：

```text
maximize   Pr(tau_G < tau_F)
minimize   E[tau_G]
minimize   cost(tau_G)
```

其中：

```text
tau_G = 第一次到达目标状态集合 G 的时间
tau_F = 第一次到达失败状态集合 F 的时间
```

这就是「更小 mixing time」在 Agent 系统里的工程化版本：不是为了混合到任意平稳分布，而是为了更快、更稳定、更低成本地进入目标区域。

## 第五层抽象：Agent 系统设计是在塑造转移概率

高层看，Agent 系统的每个组件都在改变状态转移概率。

| 系统理论概念 | Agent 系统对应物 | 作用 |
| --- | --- | --- |
| 状态空间 `S` | 所有可能任务状态、上下文状态、控制状态、环境状态 | 定义系统可能在哪里 |
| 当前状态 `S_t` | 当前任务表示、上下文、工具结果、计划进度 | 决定下一步从哪里出发 |
| 转移概率 `P(s' | s)` | 整个 Agent 系统从当前状态进入下一状态的倾向 | 描述系统自然会往哪里走 |
| 动作 `a_t` | 模型回复、工具调用、提问、规划、验证、修复 | 改变状态转移路径 |
| 策略 `pi(a | s)` | 模型和 prompt 在当前状态下选择动作的分布 | 决定下一步做什么 |
| 目标集合 `G` | 满足用户目标和质量标准的状态 | 定义什么算成功 |
| 失败集合 `F` | 错误、循环、伪完成、越权、污染状态 | 定义什么算失败 |
| 收敛 / hitting time | 完成任务所需步骤、时间、成本 | 衡量多快到达目标 |
| bottleneck | 正确路径和错误路径之间的窄桥 | 解释为什么纠错困难 |
| recurrent class | 系统反复游走的状态簇 | 对应 stuck、loop、反复 retry |
| absorbing state | done、false done、fatal error | 一旦进入就很难退出 |

因此，一个 Agent 系统的设计问题可以转换成：

```text
如何让 P 把更多概率质量分配给通往 G 的路径，
同时减少通往 F、loop、false done 的路径？
```

## 第六层抽象：Plan 和 Harness 的位置

在这个 high-level mapping 里，plan 和 harness 分别扮演不同角色。

### Plan 是对未来转移路径的假设

Plan 不是状态转移本身，而是 Agent 对未来状态转移路径的预测和选择：

```text
S_0 -> S_1 -> S_2 -> ... -> G
```

一个好的 plan 会提高系统走上成功路径的概率，因为它提前约束了中间状态和动作顺序。

它回答的是：

```text
从当前状态出发，哪些中间状态更可能通向目标？
```

### Harness 是对合法转移的控制

Harness 不是单纯包一层 wrapper，而是在控制状态转移过程：

```text
allowed actions
allowed state transitions
validation
retry
rollback
termination
user handoff
```

它回答的是：

```text
哪些转移可以发生？
哪些转移应该被阻止？
哪些转移需要验证？
哪些坏状态需要 escape path？
什么时候应该终止？
```

所以从高层看：

```text
Plan    = proposes likely path to G
Harness = constrains and corrects actual transitions
```

## 第七层抽象：好的 Agent 系统是什么

在这个框架下，一个好的 Agent 系统可以定义为：

```text
在给定任务分布下，能够高概率、低成本、可恢复地从初始状态 S_0 到达目标状态集合 G 的受控随机状态转移系统。
```

它应该满足：

- 成功状态的到达概率高。
- 失败状态的到达概率低。
- 到达目标状态的期望时间短。
- 从错误状态逃逸的概率高。
- 对初始状态扰动不敏感。
- 对工具失败和信息缺失有恢复机制。
- 不会轻易把 false done 当成 done。
- 能通过 trace 和 eval 估计自己的转移质量。

这也是为什么 Agent 系统不能只看单次模型输出质量。我们真正要评估的是整个状态转移过程的质量。

## 这一层先保留的问题

这篇文档只建立 high-level mapping，暂时不深入下列问题：

- 状态 `S_t` 应该如何表示？
- action space 应该如何分层？
- plan 如何改变路径分布？
- harness 如何定义有限控制状态？
- verifier 如何定义 `G` 和 `F`？
- memory 如何改变跨 episode 的转移概率？
- 如何用 trace 数据经验估计 `P(s' | s, a)`？
- 如何识别 bottleneck、stuck state 和 false done？

这些可以作为下一层文档继续拆。

## 当前核心假设

> Agent 系统可以先被高层理解成一个受控随机状态转移系统。模型、prompt、plan、harness、tool、verifier、memory 都是在共同塑造状态转移概率。好的 Agent 设计，就是让系统从 `S_0` 更高概率、更低成本、更少失败地进入目标状态集合 `G`。

