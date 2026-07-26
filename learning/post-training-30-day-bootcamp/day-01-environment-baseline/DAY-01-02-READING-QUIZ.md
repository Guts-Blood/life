# Day 01–02 Reading Quiz — Roofline 到 Transformer Accounting

日期：`2026-07-23`–`2026-07-26`
状态：`in_progress`  
方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## 当前进度口径

- Day 01 已独立证明掌握：`6/6`，`100%`。
- Day 01 Quiz 状态：`completed`。
- 说明：这里只代表 Day 01 Reading Quiz 完成；GPU/环境基线任务仍需执行。

| Day 01 能力点 | 当前证据 | 判定 |
|---|---|---|
| 用公式和单位解释 `T_math=F/C`、`T_comms=M/W` | 概念理解，首次回答主要是表述不精确 | `passed` |
| 解释 `max` 下界与 `sum` 二项模型上界 | 首次回答正确解释完全 overlap 与完全串行 | `passed` |
| 正确判断 compute-bound / communication-bound | 第二次数值复测正确判断 compute-bound | `passed` |
| 解释理想下界为何未必可达 | 能区分二项模型内部上界与真实程序，并指出 CPU、kernel、同步等漏项 | `passed` |
| 计算并比较 operation/hardware arithmetic intensity | 正确计算 intensity，并在复测中用 `I_op/I_hw=2` 判断 compute-bound | `passed` |
| 读懂 Roofline 图，并解释“放得下不等于跑得快” | Roofline 区域判断和 capacity/bandwidth 数值复测均通过 | `passed` |

## Quiz 来源说明

不是照抄原书习题。

- 来源约束：`T_math`、`T_comms`、上下界、arithmetic intensity 与 Roofline 坐标来自 *How to Scale Your Model* Part 1。
- 自编部分：Q1 的组织方式、Q2 的数值、FSDP overlap 时间线、kernel/layer/step 分层、针对错误的追问，以及与 30B+ SFT 工程场景的连接。
- 自适应原则：后续题目根据学习者的回答动态生成；已经稳定掌握的点不重复机械考察，出现的误区会换一个场景复测。
- 原书 `A Few Problems to Work` 会在对应计划节点明确标注为“原书题”，不会伪装成自编题。

## 这次 Reading 在整体知识地图中的位置

```text
模型结构与 workload
        ↓
参数量 / FLOPs / 通信 bytes / activation bytes
        ↓
硬件 compute / bandwidth / memory capacity
        ↓
时间、显存、并行策略与训练成本预测
        ↓
用 profiler 和真实实验校准
```

Day 01 建立右半边的硬件约束语言：compute、bandwidth、capacity 与 Roofline。

Day 02 建立左半边的 workload 账本：从 tensor shape 推出 Transformer 的参数量与 FLOPs。

两天合起来的目标不是背公式，而是以后看到一个模型 config 和一组硬件规格时，能够先做一阶预测，再决定是否租卡、租几张卡，以及应该测什么。

## 今日阅读顺序

### Block A — Day 01（45–60 分钟）

1. Introduction：`High-Level Outline`、`Links to Sections`。
2. Part 1：`Where Does the Time Go?`，读到 arithmetic intensity 与 compute-/communication-bound 判据。
3. Part 1：`Visualizing rooflines`，理解横轴、纵轴、斜线区和平台区。
4. 停在 `Matrix multiplication` 之前；matmul roofline 留到 Day 03。

### Block B — Day 02（80–90 分钟）

1. Part 4：`Counting Dots`。
2. `Forward and reverse FLOPs`，自己推导训练为何约是 forward 的 3 倍。
3. `Transformer Accounting`，逐项跟踪 embedding、Q/K/V/O、gate/up/down。
4. `Global FLOPs and Params Calculation`，理解 `6 × parameters × tokens` 的近似和边界。
5. `A Few Problems to Work`：先做前两题，再展开答案。

## Quiz 模块

| 模块 | 希望真正掌握的整体能力 | 状态 |
|---|---|---|
| A. Roofline | 把一个 operation 的时间拆成 compute 与 data movement，并判断瓶颈 | `completed` |
| B. Tensor FLOPs | 只看 tensor shape 就能数 dot/matmul FLOPs | `completed` |
| C. Transformer accounting | 从 config 推导参数量、forward/backward FLOPs | `D2-Q4 completed; D2-Q5 in_progress` |
| D. 工程容量判断 | 区分 weights、optimizer、gradients、activations 与吞吐瓶颈 | `locked` |

题目总数预计 10–12 题；根据回答动态增加或跳过追问。

## 当前题目

### Q2 — 数值 Roofline 与 arithmetic intensity

状态：`in_progress`。

某个 operation：

- 需要 `6 × 10^12` FLOPs；
- 需要从 HBM 搬运 `120 GB = 1.2 × 10^11 bytes`；
- 硬件有效计算吞吐 `C = 600 TFLOPs/s = 6 × 10^14 FLOPs/s`；
- 有效 HBM 带宽 `W = 3 TB/s = 3 × 10^12 bytes/s`。

1. 算出 `T_math` 和 `T_comms`，换算成毫秒。
2. 给出理想下界和保守上界。
3. 判断它是 compute-bound 还是 memory-bound。
4. 计算 operation 的 arithmetic intensity `F/M`，以及硬件临界 intensity `C/W`；用这两个数再次解释第 3 问。

Q2.1 回答：

```text
T_math  = (6 × 10^12) / (6 × 10^14)
        = 0.01s
        = 10ms

T_comms = (1.2 × 10^11) / (3 × 10^12)
        = 0.04s
        = 40ms
```

判定：`passed`。

Q2.2 回答：

```text
ideal lower bound = max(10ms, 40ms) = 40ms
two-component upper bound = 10ms + 40ms = 50ms
bound = memory-bound
```

若 `C` 翻倍：

```text
T_math  = 5ms
T_comms = 40ms
new ideal lower bound = max(5ms, 40ms) = 40ms
```

结论：提高非瓶颈资源的速度没有改变理想下界，operation 仍然 memory-bound。判定：`passed`。

Q2.3 回答：

```text
I_op = F/M = 50 FLOPs/byte
I_hw = C/W = 200 FLOPs/byte
```

数值与单位正确。根据 intensity 判断 bound 时把方向说反，需要复测。

推导：

```text
T_math / T_comms
= (F/C) / (M/W)
= (F/M) / (C/W)
= I_op / I_hw
```

因此：

```text
I_op > I_hw  ⇔ T_math > T_comms  ⇔ compute-bound
I_op < I_hw  ⇔ T_math < T_comms  ⇔ memory-/communication-bound
```

本题 `50 < 200`，所以是 memory-bound。判定：`calculation passed; comparison needs_retry`。

Q2.3 复测：

```text
I_op = 300 FLOPs/byte
I_hw = 150 FLOPs/byte
I_op / I_hw = 2
```

回答：compute-bound，因为算法每 byte 需要的计算更多。判定：`passed`。

### Q3.1：Roofline 吞吐与图上区域

已知：

```text
C = 600 TFLOPs/s
W = 3 TB/s
I_op = 50 FLOPs/byte
```

回答：

```text
I_op × W
= 50 FLOPs/byte × 3 × 10^12 bytes/s
= 150 × 10^12 FLOPs/s
= 150 TFLOPs/s

P_achievable = min(600, 150) = 150 TFLOPs/s
```

判定：计算正确；该点位于单层理想 Roofline 的左侧斜线区，是 memory-bound。

区域与 bound 的关系：

```text
left sloped region  ⇔ I_op < I_hw ⇔ bandwidth/memory-bound
right flat plateau ⇔ I_op > I_hw ⇔ compute-bound
knee               ⇔ I_op = I_hw
```

限定：上述等价关系针对单个理想 Roofline ceiling。真实 profiler 中“没有达到峰值 `C`”本身不能单独证明 memory-bound；实际点还可能因不理想 shape、低效 kernel、launch/synchronization 或其他 overhead 落在理论 roofline 下方。

### Q3.2：放得下不等于训得动或跑得快

题目：

- 32B parameters
- BF16 weights 约 `64GB`
- 单卡显存 capacity `80GB`

原始回答：

1. 几乎不能 full SFT；除 weights 外还有 activation、optimizer states、KV cache，并且和 context length 强相关。
2. 能放进显存也不一定快，还取决于训练并行度和 batch-size 设置。
3. 尚不能清楚区分 memory capacity 和 memory bandwidth。

点评：

- full SFT 结论正确。activation 不是参数；标准 teacher-forced SFT 通常关闭持久 inference KV cache，但 Q/K/V 及 attention intermediates 会作为训练 activation，占用量随 context、micro-batch、layers、hidden size 和 checkpointing 改变。
- 还需计入 gradients、Adam first/second moments，以及某些实现中的 FP32 master weights。常见 full-training model-state 量级约为 `12–16 bytes/parameter`，32B 对应约 `384–512GB`，且尚未包含 activation 与临时 buffer。
- “放得下不等于快”回答方向正确，但性能还取决于 compute utilization、HBM/network bandwidth、arithmetic intensity、operator shape/mix、communication/overlap 和 runtime overhead。
- capacity 回答“同时能存多少 bytes、是否可行”；bandwidth 回答“每秒能搬多少 bytes、搬运需要多久”。

Q3.2 最终复测：

```text
capacity = 80GB
resident working set = 60GB
bytes moved per step = 300GB
effective bandwidth = 3TB/s
```

回答：

1. `60GB < 80GB`，working set 可以放入显存。
2. `300GB / 3TB/s = 0.1s = 100ms`。
3. capacity 从 `80GB` 增加到 `160GB`，bandwidth 不变，因此搬运时间仍为 `100ms`。
4. bandwidth 增加到 `6TB/s`，搬运时间减半为 `50ms`。

判定：`passed`。Day 01 Reading Quiz：`6/6 completed`。

## 对话记录

### Q1 原始回答

1. `T_math`、`T_comms` 分别是某一个算法在一个单位计算中，计算浮点数的时间和通信所消耗的时间。
2. 理想情况下计算和通信可以完全并行，所以 lower bound 是二者取 max；保守上界可以看作完全串行执行。
3. 如果 `T_math > T_comms`，认为是 communication-/memory-bound；反之是 compute-bound。
4. 不一定能达到下界；猜测是因为现实情况更加复杂，但原因暂不明确。

### Q1 点评与修正版

1. 对一个被分析的 operation，`T_math = F/C`，`T_comms = M/W`。两者的单位都是 seconds。
2. 若计算与搬运能完全重叠，较短的一项被隐藏，总时间至少是两者较大值；若完全不能重叠，则近似为二者之和。这一点回答正确。
3. 原回答把名称反了：`T_math > T_comms` 是 compute-bound；`T_comms > T_math` 是 communication-/memory-bound。
4. `max(T_math, T_comms)` 是理想化下界，不是实际性能保证。它可能假设了不可实现的完美 overlap；峰值或标称 `C/W` 也未必能被小矩阵、差的 shape、低效 kernel 达到；同时还忽略 kernel launch、同步、依赖、通信 latency/拥塞、数据加载和框架开销。
5. memory capacity 与 memory bandwidth 必须分开：capacity 决定 workload 是否放得下；bandwidth 决定 bytes 搬运需要多长时间。

### Q1 第一次追问澄清

第 1 问属于表述问题，不作为概念错误。准确表达为：

> 对选定的整个 operation，`T_math` 是以有效计算吞吐 `C` 完成 `F` FLOPs 所需的时间，即 `F/C`；`T_comms` 是以有效带宽 `W` 搬运 `M` bytes 所需的时间，即 `M/W`。

阻止完美 overlap 的典型关键路径：

- FSDP：当前层参数的 AllGather 完成后，该层 matmul 才能使用完整权重；反向计算出的 gradient bucket 就绪后才能发起 ReduceScatter。
- Data Parallel：一个 gradient bucket 必须等其中的梯度计算完成才能 AllReduce；最后未被后续 backward compute 隐藏的通信会暴露在 step 尾部。
- Tensor Parallel：local matmul 后的 AllReduce/ReduceScatter/AllGather 结果经常是下一项计算的输入，因此 collective 的尾部无法被依赖它的计算覆盖。
- Expert Parallel：MoE expert compute 前需要 AllToAll dispatch，之后需要 AllToAll combine；expert matmul 被夹在两个数据依赖边界之间。
- Pipeline Parallel：下一 stage 需要等待 activation Send/Recv，上一 stage 的 backward 需要等待 gradient Send/Recv；调度不充分时形成 pipeline bubble。
- CPU/GPU offload：需要的 tensor 尚未通过 H2D/PCIe 搬回 GPU 时，依赖它的 kernel 不能开始。

实际 bytes 与简单估算不同的典型原因：

- unfused operations 会把中间 tensor materialize 到 HBM，再由下一 kernel 读回；kernel fusion 可能让它留在寄存器或片上 SRAM。
- naive attention 可能写出并读回 `[B, H, T, T]` attention matrix；FlashAttention 通过分块保留局部状态，避免完整矩阵的 HBM 往返。
- matmul 需要 tiling；若 tile 不能有效复用或 cache 行为不理想，同一部分 weight/activation 可能被重复加载。
- cache 命中会让数据来自 L2/SRAM 而不是 HBM；cache miss、容量不足或 thrashing 则会增加 HBM traffic。
- activation checkpointing 减少保存/读取 activation 的 bytes，但反向时重新执行 forward，增加 FLOPs，并可能重新读取参数。
- layout conversion、transpose 后的 contiguous copy、padding、dtype cast/quant-dequant 会产生最简公式中没有列出的额外读写。

### Q1 第二次追问澄清

学习者正确指出：整体瓶颈取决于 high-level 算法与执行架构，matmul 只是 Transformer 中最主要、最适合建立一阶模型的一类 operation。必须先声明两个分析边界：

1. workload boundary：分析单个 kernel、一个 Transformer block、一个 training step，还是端到端训练；
2. data-movement boundary：分析 HBM↔计算核心、GPU↔GPU，还是 CPU↔GPU。

不同层次要分别建模：

- kernel level：每个 matmul、softmax、norm、elementwise kernel 都有自己的 `F_i`、`M_i`、shape 和 bound；
- layer/model level：operator mix、tensor shape、sequence length、batch、attention/MLP 比例、fusion 与 architecture 共同决定 FLOPs 和 bytes；
- distributed execution level：DP/FSDP/TP/PP/EP、micro-batch、shard shape、collective 与 overlap 改变 communication bytes 和 critical path；
- hardware/runtime level：GPU、memory hierarchy、network、kernel implementation 和实际利用率决定有效 `C_i/W_i`。

端到端时间通常不能只用一次 `max(total_F/C, total_M/W)` 得到。更可靠的一阶表达是沿执行 DAG 的关键路径，对各 operation 的计算、memory traffic、collective 和 overlap 分段估算，再用 profiler 校准。

### Q1 第一次数值复测

题目：

- Layer 5 compute：`12ms`
- Layer 6 parameter AllGather：`8ms`
- 两者同时开始并可完全并发
- 变体：AllGather 改为 `18ms`

原始回答：

1. `8ms` AllGather 不能被隐藏。
2. 必须等 compute 完成才能计算 Layer 6。
3. `18ms` AllGather 会暴露 `30ms`。
4. 属于 compute-bound。

点评：

- Layer 6 必须同时等待 Layer 5 activation 和 Layer 6 parameters，但这不妨碍二者的生产过程并发。
- `8ms` AllGather 在 `12ms` compute 内完成，可以被完全隐藏；Layer 6 在 `max(12, 8)=12ms` 开始。
- `18ms` AllGather 与 `12ms` compute 并发，总等待时间是 `max(12,18)=18ms`，不是相加；暴露的通信尾部是 `18-12=6ms`。
- 此时通信更慢，关键路径由 AllGather 决定，因此是 communication-bound。

本次暴露的误区：

> “Layer 6 依赖 Layer 5 的输出”只意味着 Layer 6 自己不能提前开始，不意味着 Layer 6 参数的预取不能与 Layer 5 compute 并发。

### Q1：Parameter AllGather 的数据依赖澄清

FSDP 下，每张 GPU 平时持有某个参数 tensor 的一个 persistent shard。例如四张卡分别持有 `W6_0`、`W6_1`、`W6_2`、`W6_3`；parameter AllGather 通过跨卡通信，让每张 GPU 临时得到完整的 `W6=[W6_0,W6_1,W6_2,W6_3]`。

这里需要区分两条独立的数据链：

```text
Layer 5 activation ── Layer 5 compute ──> Layer 6 input activation
Layer 6 weight shards ── AllGather ─────> Layer 6 full weights
                                             │
两者都 ready ────────────────────────────────┴─> Layer 6 compute
```

Layer 6 weight shards 在 Layer 5 compute 开始前就已经存在；AllGather 不需要等待 Layer 5 产生它们，所以可以预取。Layer 6 compute 则同时依赖 Layer 5 output activation 与完整 Layer 6 weights，必须等待两条链都完成。

“没有数据依赖”不等于“没有资源竞争”。AllGather 需要从各卡 HBM 读取 shard，经 NVLink/PCIe/网络发送和接收，再写入 full-parameter buffer；它还包含 collective 调度、同步与 latency，并可能与 concurrent compute 竞争 HBM bandwidth、互连和部分 GPU execution resources。因此完全 overlap 只是 Roofline 中的理想情况。

不是所有 AllGather 都能提前预取。能否 overlap 取决于被 gather 的 tensor 是否已经 ready：

- 下一层 parameter AllGather：参数 shard 已存在，通常可提前 prefetch；
- 某次 compute 输出的 activation AllGather：必须等该 activation 产生后才能开始；
- backward gradient ReduceScatter：必须等对应 gradient 或 gradient bucket ready 后才能开始。

### Q1 第二次数值复测

题目：

- Layer 5 compute：`20ms`
- Layer 6 parameter AllGather：`14ms`
- 两者从 `t=0` 并发

回答与判定：

1. AllGather 可以被完全隐藏：正确。
2. AllGather 在 `t=14ms` ready，但 Layer 6 必须等 Layer 5 activation，所以最早在 `t=20ms` 开始：正确。
3. 回答 exposed communication 为 `6ms`：术语方向错误。这里 `6ms` 是 communication 完成后剩余的 compute tail；exposed communication 为 `max(14-20, 0)=0ms`。
4. compute-bound：正确。

公式：

```text
hidden communication  = min(T_compute_window, T_comms)
exposed communication = max(T_comms - T_compute_window, 0)
next operation starts = max(T_compute_window, T_comms)
```

### Q1：理论下界与 profiler 复测

题目：

```text
T_math  = 12ms
T_comms = 8ms
ideal lower bound = 12ms
two-component upper bound = 20ms
profiled runtime = 27ms
```

原始回答：

1. 可能受矩阵大小、分片比例、峰值与平均/实际计算效率差异，以及不能完全并行影响。
2. 因为只能在一定程度上并行。
3. 认为第一问已经列出一些原因。

点评：

- 第一问有效：shape/sharding 会影响有效吞吐和数据量，peak `C/W` 可能高估实际值，imperfect overlap 会让运行时间偏离 `max` 下界。
- 第二问需要纠正：如果 `12ms` 和 `8ms` 是准确的完整成本，且系统只有这两项，那么 partial overlap 只能使时间落在 `[12ms, 20ms]`。它不能单独解释 `27ms > 20ms`。
- `20ms` 只是简化二项模型内部的上界，不是真实程序的绝对上界。出现 `27ms` 说明至少有一个假设失败：`T_math` 被低估、`T_comms` 被低估，或模型漏掉了额外 overhead/critical-path operations。

最终确认回答：

> `20ms` 只是模型内部的 upper bound，并不是真实程序的绝对 upper bound；实际还可能存在 CPU、kernel 和同步问题的额外消耗。

判定：`passed`。

### 当前最重要的误区

`T_math > T_comms` 时，瓶颈是计算，因此是 compute-bound，不是 communication-bound。判断 bound 时看“谁耗时更长”，名字就跟谁。

## Day 02 当前题目

### D2-Q1 — Counting Dots：从 shape 数参数与 FLOPs

给定：

```text
X [B,T,D]
W [D,F]
Y = X @ W → [B,T,F]
```

请回答：

1. `W` 有多少参数？
2. `Y` 一共有多少个 scalar outputs，也就是多少次长度为 `D` 的 dot product？
3. 每次 dot product 为什么近似需要 `2D` FLOPs？
4. 整个 forward matmul 近似需要多少 FLOPs？

只写符号推导，暂时不代数字。

#### D2-Q1 原始回答与点评

回答：

1. `W` 有 `D×F` 个参数：正确。
2. `Y` 有 `B×T×F` 个 scalar outputs / dot products：正确。
3. 猜测 `2D` 来自 BF16：错误。
4. forward matmul 为 `2×B×T×D×F` FLOPs：正确。

修正：

```text
one length-D dot product
= D multiplications + (D-1) additions
= 2D-1 FLOPs
≈ 2D FLOPs
```

FLOP count 与 BF16/FP32 dtype 无关；dtype 影响 bytes、数值精度和硬件可达到的 FLOPs/s。判定：`3/4 correct; dot-product reason needs_recheck`。

#### D2-Q1 复测与“一阶近似”澄清

当 `D=4096`：

```text
multiplications = 4096
additions = 4095
exact FLOPs = 4096 + 4095 = 8191
leading-order approximation = 2D = 8192
relative error = 1/8191 ≈ 0.012%
```

学习者正确指出 BF16/FP32 不改变数学 FLOP count；dtype 会改变 bytes、arithmetic intensity 与硬件对应精度的 FLOPs/s。

这里的“一阶近似”指 leading-order approximation：保留随规模增长最快、决定数量级的主导项，忽略相对越来越小的常数项或低阶项。

迁移题：

```text
P_layer = 4D² + 3DF + 2D, F = 4D
        = 16D² + 2D
        ≈ 16D²
        ∈ Θ(D²)
```

学习者正确识别出 `2D` 是低阶项，可以用极限/比值理解其相对贡献趋近于零。需区分：

- `16D²`：保留 leading term 及其系数，用于估算实际参数量/FLOPs。
- `Θ(D²)`：只描述随 `D` 的渐近增长阶，可以忽略常数系数。
- `D²`：若没有 `Θ` 或“正比于”的语境，不能替代 `16D²` 作为数值估算式。

判定：`passed`。

### D2-Q2 — Forward 与 Reverse FLOPs

目标：从 chain rule 和 tensor shape 推出一个 linear/matmul 在训练中的 forward、`dX` 与 `dW` 三次主要矩阵乘法，为理解 `6 × parameters × tokens` 做准备。

合并 token 维度，令 `N=B×T`：

```text
X:  [N,D]
W:  [D,F]
Y:  [N,F]
G = dL/dY: [N,F]
```

前向与反向：

```text
Y  = X  @ W    : [N,D] @ [D,F] → [N,F]
dX = G  @ Wᵀ   : [N,F] @ [F,D] → [N,D]
dW = Xᵀ @ G    : [D,N] @ [N,F] → [D,F]
```

学习者最初能从标量 chain rule 判断线性映射对输入的导数与 `W` 有关，但忘记了矩阵反向传播的具体乘法。教学后，已经正确识别：

- `L` 是 scalar loss；
- `G` 是 upstream gradient `dL/dY`；
- `dX` 来自 chain rule，但 reverse-mode 中使用线性映射的 transpose/adjoint；
- `dW` 需要累加所有 `N` 个 token/sample 对共享权重的贡献。

Shape 复测回答：

```text
dX = dY @ Wᵀ
   = [8,32] @ [32,16]
   → [8,16]

dW = Xᵀ @ dY
   = [16,8] @ [8,32]
   → [16,32]
```

FLOPs 推导：

```text
forward Y : 2NDF
backward dX: 2NDF
backward dW: 2NDF
total      : 6NDF = 3 × forward FLOPs
```

令 linear weight 参数量 `P=D×F`，处理的 token 数 `N=B×T`，则：

```text
training FLOPs ≈ 6 × N × P
```

这是一阶 matmul accounting；bias、norm、activation、softmax、loss、attention 的非线性/二次项以及 activation recomputation 等需要按分析边界额外处理。判定：`passed`。

### D2-Q3 — GQA 的 Q/K/V 维度

一般形式：

```text
Q width = Hq  × d_k
K width = Hkv × d_k
V width = Hkv × d_v
attention heads 拼接后的 width = Hq × d_v
```

标准 dot-product attention 要求 `d_q=d_k` 才能直接计算 `QKᵀ`；`d_v` 在数学上可以不同。模型 config 决定这些维度，训练学习的是 projection weights 的数值。

GQA 中多个 query heads 共享一个 KV head。Qwen3-30B-A3B 的 `Hq=32`、`Hkv=4`，因此每 8 个 query heads 共享一个 KV head。其 config 使用统一的 `head_dim=128`，所以 Q/K/V widths 分别为 `4096/512/512`。

复测使用不同的 value head dimension：

```text
Hq=32, Hkv=4, d_k=128, d_v=96

Q width = 32×128 = 4096
K width =  4×128 = 512
V width =  4×96  = 384
attention heads 拼接 width = 32×96 = 3072
```

学习者四项均正确。判定：`dimension constraints passed; projection parameter accounting pending`。

Attention score shape 澄清：

```text
Q_h: [T,d_k]
K_h: [T,d_k]
Q_h @ K_hᵀ: [T,d_k] @ [d_k,T] → [T,T]
```

学习者最初将 `[B,Hq,T,T]` 误解为 `[B,Hq,d_k,d_k]`；现已理解 `d_k` 是 dot product 中被收缩的 feature dimension，两个 `T` 分别表示 query position 与 key position。因此 `attention_weights[b,h,i,j]` 表示 batch `b`、head `h` 中，第 `i` 个 query token 对第 `j` 个 key token 的权重。判定：`passed`。

Qwen3-30B-A3B 单层 attention projection 参数账本（数学 `X@W` 记法，忽略 bias）：

```text
WQ [2048,4096]: 8,388,608
WK [2048, 512]: 1,048,576
WV [2048, 512]: 1,048,576
WO [4096,2048]: 8,388,608
total           : 18,874,368 ≈ 18.87M
```

学习者正确指出参数量就是矩阵维度乘积，总量为各 projection 相加；不再单独考察机械乘法。下一步考察 GQA 相对 MHA 改变了哪些账目。

GQA → MHA 复测：

```text
Hq 保持 32，Hkv 从 4 增加到 32

WQ: unchanged
WO: unchanged
WK: 8×
WV: 8×
KV cache total: 8×
```

KV cache 可写为 `2 × B × T × Hkv × d_head × bytes_per_element`；开头的 `2` 已同时计入 K 和 V，因此 K/V 各扩大 8 倍时，总 cache 也是扩大 8 倍而不是 16 倍。学习者三项均正确。判定：`passed`。

GQA attention-compute 复测：

学习者正确指出，减少 KV heads 只是让多个 query heads 共享 K/V；`Hq=32` 没有变化，32 个 query heads 仍然分别计算自己的 attention scores。因此 GQA 不会把核心 `QKᵀ` 和 `AV` 的逻辑 FLOPs 降低 8 倍：

```text
QKᵀ FLOPs ≈ 2 × B × Hq × T² × d_k
AV FLOPs   ≈ 2 × B × Hq × T² × d_v
```

GQA 主要减少 K/V projection 参数与 FLOPs、K/V activation、推理 KV cache 及相关 memory traffic；具体 kernel 仍可能因复用和带宽行为获得额外性能收益。判定：`D2-Q3 passed`。

### D2-Q4 — Gated MLP 与 MoE Expert Accounting

一个 SwiGLU expert：

```text
gate   = X @ W_gate
up     = X @ W_up
hidden = SiLU(gate) ⊙ up
output = hidden @ W_down

W_gate: [D,F]
W_up:   [D,F]
W_down: [F,D]
```

因此：

```text
P_one_expert = DF + DF + FD = 3DF
```

学习者正确指出，`W_gate/W_up/W_down` 是持久、可训练的 weight parameters；`gate/up/hidden` 是 forward 过程中产生的 intermediate activations，不应加入参数量，但训练时可能需要保存或重算并占用 activation memory。`SiLU` 和逐元素乘法本身没有可训练参数。

对于 Qwen3-30B-A3B 的 `D=2048, F=768`：

```text
P_one_expert = 3 × 2048 × 768
             = 4,718,592
             ≈ 4.72M
```

MoE total-vs-active 复测：

```text
E = 128 experts
K = 8 selected experts/token
Pe = 4,718,592 parameters/expert

total expert parameters/layer
= E × Pe
= 128 × 4,718,592
= 603,979,776
≈ 603.98M

active expert parameters/token/layer
= K × Pe
= 8 × 4,718,592
= 37,748,736
≈ 37.75M

active / total = K/E = 8/128 = 1/16 = 6.25%
```

学习者的 `≈604.16M` 来自使用四舍五入后的 `4.72M`，作为一阶估算正确。判定：`D2-Q4 passed`。

### D2-Q5 — MoE Capacity Cost vs Compute Cost

给定 Qwen3-30B-A3B：

```text
total parameters  ≈ 30.5B
active parameters ≈ 3.3B/token
```

学习者回答：

1. full-SFT weights 显存按 total parameters：正确。
2. gradients 与 Adam states 也按 total parameters，但理由表述为“事先不知道激活哪些 experts”：结论正确，理由需细化。
3. per-token 主要 matmul FLOPs 按 active parameters；实际不会为该 token 执行全部 128 experts：正确。

修正边界：所有 trainable expert weights 都是长期模型状态，并可能在不同 token、micro-batch 和 step 中被选中、产生梯度和更新，因此 full-SFT 的模型状态容量按 total trainable parameters 规划。某一 micro-batch 中未收到 token 的 expert 可能没有有效梯度，但常规容量估算仍需覆盖所有本地参数的 gradient/optimizer buffers；ZeRO/Expert Parallel 可以把这些状态跨 GPU 分片，但不会把全局模型从 total parameters 变成 active parameters。

当前判定：`capacity/compute answers correct; optimizer-state reason recheck pending`。

#### Training step、逐层 MoE 与 optimizer state

- 这里的 `step 1/2` 指连续的 optimizer/training steps，通常各自消费一个新的 global batch；不是同一 batch 中的两条 sequence。若使用 gradient accumulation，则多个 micro-batches/micro-steps 先累计梯度，最后一次 `optimizer.step()` 才构成这里所说的 optimizer step。
- 同一 batch 内的不同 sequence、不同 token 会独立路由；同一个 token 进入不同 decoder layers 时，也会由各层自己的 router 重新选择 experts。
- Qwen3-30B-A3B 有 48 个 decoder layers，官方 config 为 `decoder_sparse_step=1`、`mlp_only_layers=[]`。Transformers 实现会在 `(layer_idx+1) % decoder_sparse_step == 0` 时使用 `Qwen3MoeSparseMoeBlock`，因此该模型每层的 FFN/MLP 子模块都是 MoE；self-attention 仍为 dense GQA。其他 MoE 模型可以交替使用 dense MLP 与 MoE。
- Adam 并非逐 step 独立。它为每个 trainable parameter 保留一阶矩 `m` 与二阶矩 `v`：

```text
m_t = β1 m_(t-1) + (1-β1) g_t
v_t = β2 v_(t-1) + (1-β2) g_t²
```

当前更新显式依赖过去 steps 的状态；若每一步重置 `m/v`，算法就不再是正常的 Adam。某个 expert 当前 step 未激活时，其状态可能保持不变，或按具体 optimizer/zero-gradient/weight-decay 语义处理，但不能丢弃，因为它在未来 step 再次激活时仍需延续历史。

#### 为什么优化必须沿连续轨迹进行

训练目标可写成：

```text
J(θ) = E_z[ℓ(θ; z)]
```

每个 mini-batch 只提供总体目标梯度的随机估计 `g_t`。梯度下降依靠迭代逐步接近更优参数：

```text
θ_(t+1) = θ_t - η g_t(θ_t)
```

`θ_(t+1)` 同时是前面所有更新积累出的模型，也是下一批数据计算梯度的位置。如果每一步都重新回到相同初始参数 `θ_0`，则只会得到许多个互不累积的一步更新，无法沿 loss surface 持续前进。

例：`J(θ)=θ²/2`、`∇J=θ`、`η=0.1`、`θ_0=10`：

```text
连续训练: 10 → 9 → 8.1 → 7.29 → ... → 0
每步重置: 10 → 9；10 → 9；10 → 9；...
```

如果每个 batch 都在同一个 `θ` 上计算梯度后求平均，本质上只是组成一个更大的 batch、执行一次更新，并不等于多次迭代。

Checkpoint 只是把训练状态序列化到磁盘：

- 保存并恢复 `θ + optimizer m/v + optimizer step + scheduler + RNG + data position`，原则上等价于不中断地继续训练。
- 只加载更新后的 model weights `θ`，仍保留了模型参数的连续性，但重置了 Adam/scheduler 等历史，属于 warm start，后续轨迹会改变。
- 对无 momentum 的 vanilla SGD，若学习率、下一批数据和随机状态完全一致，只恢复 weights 可以与连续更新等价；Adam 不满足这个条件。

核对来源：

- https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py

## 完成标准

- [x] 能解释 Roofline 的时间上下界、单位和 arithmetic intensity。
- [x] 能从 contraction shape 推 FLOPs，而不是只记公式。
- [x] 能解释训练 matmul 为何常用约 `6 × 参数量 × token 数`。
- [ ] 能从 Qwen config 分解 Q/K/V/O 与 gated MLP 参数。
- [ ] 能说明上述近似忽略了什么，以及何时误差会变大。
- [ ] 完成 10–12 道自适应 Quiz，并记录至少 3 个被纠正的误区。
