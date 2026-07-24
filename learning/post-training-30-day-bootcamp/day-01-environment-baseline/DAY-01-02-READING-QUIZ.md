# Day 01–02 Reading Quiz — Roofline 到 Transformer Accounting

日期：`2026-07-23`–`2026-07-24`  
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
| B. Tensor FLOPs | 只看 tensor shape 就能数 dot/matmul FLOPs | `Day 02 Q1 in_progress` |
| C. Transformer accounting | 从 config 推导参数量、forward/backward FLOPs | `locked` |
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

## 完成标准

- [x] 能解释 Roofline 的时间上下界、单位和 arithmetic intensity。
- [ ] 能从 contraction shape 推 FLOPs，而不是只记公式。
- [ ] 能解释训练 matmul 为何常用约 `6 × 参数量 × token 数`。
- [ ] 能从 Qwen config 分解 Q/K/V/O 与 gated MLP 参数。
- [ ] 能说明上述近似忽略了什么，以及何时误差会变大。
- [ ] 完成 10–12 道自适应 Quiz，并记录至少 3 个被纠正的误区。
