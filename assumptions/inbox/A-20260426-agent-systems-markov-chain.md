---
id: A-20260426-agent-systems-markov-chain
title: Agent 系统与 Markov Chain 的关系
status: inbox
domain: AI systems
created_at: 2026-04-26
updated_at: 2026-04-26
confidence: 0.55
source: user reflection
related_profile:
related_experiment:
review_at:
verified_at:
falsified_at:
supersedes:
---

# Agent 系统与 Markov Chain 的关系

## 核心直觉

一个 Agent 系统可以被理解成一个受控的随机状态转移系统。

用户 query 和当前上下文经过 Mapper 后，被压缩到某个状态空间中的初始状态 `S0`。Agent 每一步根据当前状态 `St` 选择动作 `at`，然后环境、工具、模型输出和 harness 共同把系统推进到下一个状态 `St+1`。

如果固定策略 `pi(a | s)`，并且状态表示足够完整，使得未来只依赖当前状态和动作，那么这个过程就接近一个 Markov Decision Process。在策略固定之后，它可以退化成一个由策略诱导出来的 Markov Chain。

更形式化地说：

```text
S0 = Mapper(user_query, current_context)
at ~ pi(a | St)
St+1 ~ T(s' | St, at)
```

固定策略后，诱导状态转移为：

```text
P_pi(s' | s) = sum_a pi(a | s) * T(s' | s, a)
```

所以，Agent 系统不是单纯在「生成回答」，而是在构造、约束、观测和修正一个随机状态转移过程。

## 目标状态

可以先做一个简化假设：`S0` 中包含 query 希望环境最终达到的状态。理想情况下，这个目标状态应该和上下文无关；更简化地看，也可以假设一开始所有需要达到的目标状态都是 `0`，Agent 的工作是逐步把目标状态从 `0` 推到完成态。

但在真实系统里，目标通常不是单个状态，而是一个目标状态集合 `G`：

```text
G = {所有满足用户真实意图、约束和质量标准的状态}
```

失败状态也应该被显式建模：

```text
F = {错误答案、错误工具调用、陷入循环、不可恢复的上下文污染、过度消耗资源等}
```

因此，一个好的 Agent 系统的目标更像是：

```text
maximize   Pr(reach G before F)
minimize   E[time_to_reach_G | success]
minimize   cost, latency, risk
```

这里比 `mixing time` 更直接的工程指标通常是 `hitting time`，也就是从初始状态到达目标集合 `G` 的期望时间。`mixing time` 描述的是分布收敛到平稳分布的速度，而 Agent 任务通常更关心「多快到达正确终点」和「到达错误吸收态的概率有多低」。

不过，用 mixing time 的直觉仍然有价值：我们希望系统快速进入目标状态附近的高概率区域，而不是在巨大的状态空间里长期游走。

## Agent 系统到底在做什么

### 1. 模型和 System Prompt 提供策略先验

模型和 system prompt 的作用，是提供一个更好的动作选择分布：

```text
pi(a | s) = P(next_action = a | current_state = s)
```

一个好的模型和 prompt 应该让概率质量集中到更有用的动作上：

- 在需要搜索时，更倾向于搜索而不是直接编造。
- 在需要澄清时，更倾向于提问而不是盲目执行。
- 在需要工具时，更倾向于调用正确工具。
- 在已有足够证据时，更倾向于收敛而不是继续发散。
- 在状态不确定时，更倾向于保留不确定性而不是假装确定。

换句话说，模型和 prompt 在塑造状态转移矩阵：它们改变从 `St` 到不同 `St+1` 的概率分布。

### 2. Harness 限制并塑造 action space

Harness 的核心价值不是简单地「限制模型」，而是把无限、模糊、难观测的动作空间变成更有限、更结构化、更可验证的动作空间。

例如：

- 工具列表限制了可选动作集合。
- JSON schema 限制了工具调用格式。
- 权限系统限制了危险动作。
- 状态机限制了某些动作只能在某些阶段发生。
- validator 把错误状态及时暴露出来。
- retry、backtrack、timeout 和 checkpoint 让系统能从坏转移中恢复。

从 Markov Chain 的角度看，限制 `at` 的本质是改变动作分布：

```text
pi_H(a | s) = P(a | s, harness constraints)
```

这会减少无效动作的概率，提高正确工具和正确步骤的概率，从而降低到达目标状态的期望时间。

同时，harness 也在限制状态空间爆炸。它把一些本来会发散的状态压回可控轨道，让系统不会在无穷多的自然语言中间状态里随机游走。

### 3. State Mapper 决定系统是否真的近似 Markov

如果 Mapper 把 query 和上下文映射成一个好的状态表示，那么当前状态 `St` 就足够描述下一步决策所需的信息。

如果状态太粗，会丢失关键变量：

- 用户真实约束没有进入 state。
- 当前工具执行结果没有进入 state。
- 已经失败过的尝试没有进入 state。
- 目标是否已经满足没有进入 state。

这会破坏 Markov 假设，因为未来并不只依赖当前状态，而是依赖被丢掉的历史。

如果状态太细，又会导致状态空间爆炸：

- 每个自然语言 token 都变成状态差异。
- 无关上下文导致过拟合。
- 近似相同的任务无法共享经验。
- 评估和调试变得困难。

所以 state design 的关键，是找到合适的抽象粒度：足够完整，能支持下一步决策；又足够压缩，能让系统稳定地复用策略。

### 4. Verification 改变转移后的吸收概率

没有验证的 Agent 系统，容易把错误状态当成完成态。

例如：

```text
St -> "看起来完成了" -> terminal
```

但这个 terminal 可能并不是目标集合 `G`，而是失败集合 `F`。

验证机制的作用，是把「伪完成态」重新打开：

```text
St -> candidate_done -> verify -> done
St -> candidate_done -> verify_failed -> repair
```

这相当于改变了吸收态结构。好的 verification 会减少错误吸收态的概率，提高真正到达 `G` 的概率。

### 5. Planning 和 Search 是对转移路径的主动选择

如果只做一步贪心选择，Agent 每次都从 `P(a | s)` 中选一个看起来最好的动作。

但复杂任务通常需要考虑多步路径：

```text
S0 -> S1 -> S2 -> ... -> G
```

planning、tree search、beam search、自我反思、多样本投票，本质上都是在估计不同转移路径的质量。

它们并不消除随机性，而是在用更多计算换取更好的路径选择：

- 提高成功路径被采样到的概率。
- 提前发现低质量分支。
- 避免短期看似合理、长期会卡住的动作。
- 在多个候选轨迹中选择更稳定的一条。

## 学术视角：抽象模型

从高层看，Agent 系统可以被建模为：

```text
State space:      S
Action space:     A
Policy:           pi(a | s)
Transition:       T(s' | s, a)
Target states:    G
Failure states:   F
Cost function:    C(s, a)
Verifier:         V(s) -> {pass, fail, uncertain}
Harness:          H(S, A, pi, T)
```

更完整的目标函数可以写成：

```text
maximize   Pr(tau_G < tau_F)
minimize   E[tau_G]
minimize   E[sum_t C(St, at)]
subject to quality, safety, latency constraints
```

其中：

```text
tau_G = first time reaching target states G
tau_F = first time reaching failure states F
```

这说明 Agent 设计的本质不是让模型「更聪明」这么简单，而是在做三个层面的设计：

- 状态空间设计：什么信息进入 `St`。
- 策略设计：什么动作在什么状态下更可能被选中。
- 转移控制：如何让坏转移更少、好转移更可达、错误状态更容易恢复。

## 工程视角：具体 trade off

### Action space 越小，越稳，但表达力越弱

限制工具、限制 schema、限制状态机，可以显著降低错误动作概率。

但如果限制过强，Agent 会失去解决非常规问题的能力。它可能很稳定，但只能解决被预先设计过的问题。

工程上的关键问题是：

```text
哪些动作必须开放？
哪些动作应该被结构化？
哪些动作应该被禁止？
哪些动作应该需要额外确认？
```

### State space 越小，越容易收敛，但越容易丢信息

压缩状态可以减少混乱，提高复用性，降低上下文成本。

但状态压缩过度会让模型看不到关键事实，导致它在错误假设下反复行动。

好的 state abstraction 不是越短越好，而是要能保留影响下一步决策的变量。

### Verification 越强，错误率越低，但成本越高

每一步都验证可以提高可靠性，但会增加：

- latency
- token cost
- 工具调用成本
- 实现复杂度
- 用户等待时间

所以 verification 应该和风险匹配。低风险任务可以轻验证，高风险任务需要强验证。

### Memory 可以降低重复探索，也可能制造粘性错误

记忆机制可以让 Agent 避免重复犯错，也可以让它更快识别用户偏好。

但错误记忆会制造 sticky state：一旦进入错误解释，后续每一步都被错误记忆强化，系统很难跳出来。

所以 memory 需要：

- provenance
- freshness
- confidence
- retraction
- conflict detection

### Harness 不只是 wrapper，而是系统的控制层

Harness 做的不是把模型包起来这么简单。它实际上在控制整个随机过程：

- 约束可选动作。
- 记录状态转移。
- 插入验证节点。
- 检测循环和卡死。
- 控制重试和回滚。
- 决定什么时候交还给用户。
- 决定什么时候终止。

因此，harness 是 Agent 系统里非常核心的工程层。

## Barbell Bridge 和 Stuck State

你提到的「杠铃桥」可以理解为状态空间中的 bottleneck。

状态空间里可能有两个大的区域：

```text
correct interpretation cluster  <->  narrow bridge  <->  wrong interpretation cluster
```

一旦 Agent 掉进错误解释区域，它仍然可以在这个区域内部做很多看似合理的状态转移，但跨回正确区域的概率很低。

这类问题常见于：

- 一开始误解用户意图。
- 过早选定错误计划。
- 错误工具返回结果被当成事实。
- 记忆中存在错误偏见。
- prompt 把模型锁进某种固定角色或固定流程。
- 上下文太长，关键纠错信号被稀释。

Stuck state 则是另一类问题：系统进入了一个不包含目标状态的 recurrent class 或近似吸收态。

例如：

```text
反复搜索 -> 反复总结 -> 发现信息不足 -> 再次搜索
```

或者：

```text
工具失败 -> retry -> 同样参数再失败 -> retry
```

避免这些情况，需要给系统设计 escape transition：

- loop detection
- max retry
- alternative hypothesis generation
- forced state reclassification
- backtracking
- reset context
- ask user
- switch tool
- lower-level diagnostic
- verifier-triggered repair

这些机制的本质，是在原本很低概率的逃逸路径上人为增加转移概率。

## 迭代视角：怎么训练和改进系统

可以把 Agent 系统的迭代看成对经验转移矩阵的估计和修正。

第一步，记录轨迹：

```text
(S0, a0, S1, outcome)
(S1, a1, S2, outcome)
(S2, a2, S3, outcome)
...
```

第二步，给状态和结果打标签：

- reached target
- reached false completion
- stuck
- loop
- tool error
- user clarification needed
- context missing
- wrong decomposition
- wrong action

第三步，估计哪些状态转移有问题：

- 哪些状态最容易进入失败态？
- 哪些动作在某类状态下成功率低？
- 哪些状态经常导致循环？
- 哪些任务的 hitting time 特别长？
- 哪些状态需要用户澄清但模型没有问？
- 哪些工具调用经常因为 schema 或参数错误失败？

第四步，选择修正手段：

- 如果问题是动作太发散，收紧 action space。
- 如果问题是状态不完整，改 Mapper 或 state schema。
- 如果问题是错误完成，加入 verifier。
- 如果问题是 stuck，加入 escape transition。
- 如果问题是工具误用，改 tool description 或参数 schema。
- 如果问题是模型策略差，改 system prompt、few-shot 或训练数据。
- 如果问题是长程规划差，引入 planner 或 search。

第五步，重新跑 eval，比较：

- success rate
- false completion rate
- average hitting time
- retry count
- loop rate
- tool error rate
- user intervention rate
- average cost

这就是把 Agent 迭代从「感觉 prompt 更好了」变成「状态转移结构更好了」。

## 设计哲学

一个好的 Agent 系统，本质上是在做概率质量的搬运。

它要让系统从初始状态 `S0` 更高概率、更低成本、更少风险地进入目标状态集合 `G`。模型提供基础策略，prompt 调整策略先验，harness 约束动作空间和状态空间，verifier 改善吸收态结构，memory 和 planning 改善长程路径选择。

所以可以把 Agent 系统的核心工作概括成一句话：

```text
Agent system = state abstraction + policy shaping + transition control + verification + recovery
```

它不是追求完全确定性，而是在一个高维、部分可观测、非平稳的状态空间里，让正确转移更容易发生，让错误转移更容易被发现，让坏状态更容易逃离，让目标状态更快被命中。

## 可检验假设

这套理解可以转化成一个可测试的工程假设：

> 如果我们能更好地定义状态空间、约束动作空间、插入验证节点、记录状态转移并识别 stuck/bottleneck 状态，那么 Agent 系统到达目标状态的成功率会提高，平均 hitting time 会下降，错误完成和循环会减少。

## 最小验证方式

选一类固定任务，构建两套 Agent harness：

- Baseline：较少约束，主要依赖模型自然规划。
- Structured：显式 state schema、有限 action space、工具 schema、verification、loop detection、escape transition。

对同一批任务记录轨迹并比较：

- 成功率是否提高。
- 平均步骤数是否降低。
- 工具错误率是否降低。
- stuck/loop 是否减少。
- 错误完成率是否降低。
- 用户澄清是否更早发生。

如果 structured harness 在这些指标上稳定优于 baseline，就支持这个假设：Agent 系统的关键不是单点 prompt，而是对状态转移过程的整体设计。
