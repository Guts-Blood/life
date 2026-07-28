# Day 04 Reading Quiz — Sharding、Collective 与分布式诊断

开始日期：`2026-07-27`

状态：`in_progress`

方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## Quiz 目标

| 模块 | 希望真正掌握的能力 | 状态 |
|---|---|---|
| A. Sharded matmul | 从 global/local shape 判断输出是完整块还是 partial sum | `completed` |
| B. Collective 与布局 | 从目标输出布局推出 AllGather、AllReduce、ReduceScatter 或 AllToAll | `completed` |
| C. 训练状态 | 说明 DP、FSDP/ZeRO、TP、PP 下参数、梯度、优化器状态和 activation 的布局 | `in_progress` |
| D. 配置与并行组 | 计算 world size、DP/TP/PP group 和 global batch | `not_started` |
| E. 故障诊断 | 对 OOM、hang、checkpoint mismatch 给出有证据顺序的首轮排查 | `not_started` |

题目会根据回答动态增加或跳过追问；不重复 Day 01–03 已经稳定掌握的纯 Roofline 推导。

## 当前进度

- D4-Q1：`passed`
- D4-Q2：`passed`
- D4-Q3：`passed`
- D4-Q4：`passed`
- D4-Q5：`passed`
- D4-Q6：`needs_retry`
- 当前重点：PP 的 stage-local optimizer state、边界 activation gradient，以及 pipeline bubble。

## 当前题目

### D4-Q1 — 从 local shard 推出 collective

有矩阵乘法：

```text
A[I,J] @ B[J,K] -> C[I,K]

A.shape = [8,12]
B.shape = [12,16]
world_size = 2
```

现在沿收缩维 `J` 把 `A` 和 `B` 对齐切到两个 rank：

```text
rank 0:
  A0 = A[:, 0:6]    # [8,6]
  B0 = B[0:6, :]    # [6,16]
  P0 = A0 @ B0      # [8,16]

rank 1:
  A1 = A[:, 6:12]   # [8,6]
  B1 = B[6:12, :]   # [6,16]
  P1 = A1 @ B1      # [8,16]
```

请先独立回答四点：

1. `P0` 或 `P1` 是否已经是完整的 `C`？为什么？请从 dot product 覆盖的 `J` 范围解释。
2. 用一个等式写出完整 `C` 与 `P0`、`P1` 的关系。
3. 如果希望两个 rank 最终都持有完整的 `C[8,16]`，应使用什么 collective？它做了哪两个逻辑动作？
4. 如果下游只希望每个 rank 持有 `C` 的一部分，而不是都复制完整 `C`，可以改用什么 collective？它与第 3 问的最终布局有何不同？

暂时不要讨论 ring 实现或通信成本；这里只考察 `global shape -> local computation -> output state -> collective` 的因果链。

## 回答与点评

### D4-Q1 第一轮回答

学习者回答：

```text
1. 肯定不是，这是被切分了的。
2. C = P0 + P1。
3. AllReduce。
4. ReduceScatter。
```

点评：

1. 结论正确，但“被切分了”还不够精确。这里切的是收缩维 `J`：`P0[i,k]` 只累加 `j=0..5`，`P1[i,k]` 只累加 `j=6..11`。两者的 shape 都已经是 `[8,16]`，缺少的是另一半 `J` 对应的数值贡献，而不是缺少一块输出坐标。因此它们是同形状的 partial sums。
2. `C=P0+P1` 正确；这里是同一 `[i,k]` 位置上的 elementwise sum。
3. `AllReduce` 正确。以 `SUM` 为 reduce op 时，它在逻辑上先规约各 rank 的 partial sums，再把完整结果复制到所有 rank；最终两个 rank 都持有相同的 `C[8,16]`。
4. `ReduceScatter` 正确。它先完成同样的 elementwise reduction，再把规约后的 `C` 沿约定维度切成互不重叠的 shards；每个 rank 只持有其中一块，而不是完整复制。

判定：`core collective mapping passed; explanation/layout needs_followup`。

### D4-Q1 定向追问

假设 `ReduceScatter` 最终约定沿 `K` 维切分：

```text
C.shape = [8,16]
world_size = 2
```

只回答两点：

1. rank 0 和 rank 1 最终分别持有 `C` 的哪个 slice？各自的 local shape 是什么？
2. 为什么不能把 `P0`、`P1` 沿 `K` 维直接 concat 成完整 `C`？

学习者回答：

```text
肯定不行；这是两个不同的事情，但暂时不知道怎样严格表达。
```

严格表达：

```text
P0[i,k] = Σ_{j∈J0} A[i,j]B[j,k]
P1[i,k] = Σ_{j∈J1} A[i,j]B[j,k]
```

`P0`、`P1` 的输出坐标域相同，都是所有 `(i,k)`；区别只在每个坐标累加了哪个 `J` 子集。因此它们是同一输出坐标的不同数值贡献，正确组合是 elementwise reduction：

```text
C[i,k] = P0[i,k] + P1[i,k]
```

沿 `K` concat 会产生 `[8,32]`：前 16 列仍只是 `J0` 的 partial sums，后 16 列仍只是 `J1` 的 partial sums。它既改变了目标 shape，也没有完成任何一个 `C[i,k]` 所需的完整求和。

只有当两个 rank 持有互不重叠的输出坐标时才能沿 `K` concat。例如：

```text
C0 = A @ B[:, 0:8]    # C[:, 0:8], shape [8,8]
C1 = A @ B[:, 8:16]   # C[:, 8:16], shape [8,8]
C  = concat_K(C0, C1)
```

判定：`passed`。学习者已区分两种情况；补齐形式化表达。

### D4-Q2 — AllGather 还是 AllReduce

现在不是切收缩维 `J`，而是沿输出维 `K` 切参数：

```text
W.shape = [12,16]

rank 0: W0 = W[:, 0:8]    # [12,8]
rank 1: W1 = W[:, 8:16]   # [12,8]
```

某个后续计算要求每个 rank 都临时持有完整的 `W[12,16]`。请回答：

1. 应使用 `AllGather` 还是 `AllReduce`？
2. 为什么？请用“输出坐标”或“数值贡献”解释。
3. collective 完成后，每个 rank 持有的 local shape 是什么？与操作前相比，参数 working set 变成多少倍？

学习者回答：

```text
1. AllGather。
2. 因为这是矩阵不同坐标的切分到聚集，而不是求和。
3. 都拥有完整的 slice，2 倍。
```

点评：

1. `AllGather` 正确。
2. 解释正确。`W0`、`W1` 对应互不重叠的 `K` 坐标，不是同一坐标的 partial contributions，因此应拼合坐标块，不做数值 reduction。
3. 倍数正确。更精确地说，完成后每个 rank 持有完整 tensor `W[12,16]`，不再只是一个 slice；每个 rank 的参数 working set 从 `12×8=96` 个元素变为 `12×16=192` 个元素，即 `2×`。

判定：`passed`。

### D4-Q3 — AllToAll 改变谁持有什么

考虑一个简化的 MoE token routing。两个 rank 分别承载 expert 0 和 expert 1。路由前：

```text
rank 0 持有:
  token a -> expert 0
  token b -> expert 1

rank 1 持有:
  token c -> expert 0
  token d -> expert 1
```

路由后希望：

```text
rank 0 持有: token a, token c    # 都交给 expert 0
rank 1 持有: token b, token d    # 都交给 expert 1
```

请回答：

1. 应使用哪个 collective？
2. 这个过程是否需要把 token 表示做 elementwise sum？
3. 它与 AllGather 的关键区别是什么：操作后每个 rank 是拥有全部 token，还是只拥有发给自己的 token？

学习者回答：

```text
1. AllToAll。
2. 不需要。
3. 只是重新分布。
```

点评：

- 三项均正确。AllToAll 根据目的地重新分布 token ownership，不做 elementwise reduction。
- 与 AllGather 的区别是：AllGather 后每个 rank 都拥有全部输入 shards；AllToAll 后每个 rank 只拥有路由到自己的 token。

判定：`passed`。Collective 与布局模块完成。

### D4-Q4 — DDP 与 FSDP：persistent state 和临时 working set

假设用两个 rank 训练同一个模型，优化器为 Adam，不考虑 CPU offload、activation checkpointing 和混合精度细节。

配置 A 是普通 DDP；配置 B 是 FSDP `FULL_SHARD`。请填写下表中的“复制”或“分片”，并回答 collective：

| 状态/时刻 | DDP | FSDP `FULL_SHARD` |
|---|---|---|
| step 之间持久保存的 parameters | ? | ? |
| optimizer states | ? | ? |
| gradient 同步完成后的 gradients | ? | ? |
| 某一层 matmul 执行前所需 parameters | ? | ? |

然后回答：

1. DDP 在 backward 中主要用哪个 collective 同步 gradients？
2. FSDP 在一层计算前为什么需要 AllGather parameters？
3. FSDP 在 backward 后为什么更适合 ReduceScatter gradients，而不是 AllReduce 后让每个 rank 都保留完整 gradients？

注意区分：

```text
persistent state：大部分时间由 rank 长期持有的状态
temporary working set：某层实际计算时临时 materialize 的状态
```

学习者第一轮回答：

```text
DDP 是复制，FSDP 是分片；DDP 只有数据是切分的。
1. AllToAll。
2. 因为没有完整数据分片。
3. 不需要，每层只需要那一层的 gradient。
```

点评：

- “DDP 复制模型状态、不同 rank 处理不同数据”总方向正确，但表格尚未逐项填写。
- 普通 DDP 的 gradient synchronization 是 `AllReduce`，不是 `AllToAll`。
- FSDP 在计算前 AllGather 的对象是该层的 parameter shards；目的不是获得完整训练数据，而是临时 materialize 普通 layer kernel 所需的完整 parameters。
- “每层只需要该层 gradient”解释了 layer-wise 生命周期的一部分，但没有回答为什么使用 ReduceScatter。各 data rank 基于不同 local micro-batch 得到不同的 gradient contributions，仍需跨 rank 求和/平均；ReduceScatter 在完成 reduction 的同时，只让每个 rank 保留自己负责的 parameter/optimizer shard 对应的 gradient shard。

正确状态表：

| 状态/时刻 | DDP | FSDP `FULL_SHARD` |
|---|---|---|
| step 之间持久保存的 parameters | 复制 | 分片 |
| optimizer states | 复制 | 分片 |
| gradient 同步完成后的 gradients | 复制 | 分片 |
| 某一层 matmul 执行前所需 parameters | 复制/完整 | 临时 AllGather 为复制/完整 |

判定：`DDP/FSDP direction recognized; collective objects need_retry`。

### D4-Q4 定向复测

两个 data-parallel rank 分别用不同 local micro-batch 算出同一层完整参数 `W` 的本地 gradient contribution：

```text
rank 0: G0 = ∂L_local0/∂W
rank 1: G1 = ∂L_local1/∂W
```

只回答三点：

1. 普通 DDP 做完 gradient AllReduce 后，rank 0、rank 1 各持有什么？用 `G0`、`G1` 写出结果。
2. FSDP 做 ReduceScatter 后，跨 rank 是否仍然完成了 `G0+G1` 的 reduction？每个 rank 最终保留完整 gradient 还是其中一个 shard？
3. FSDP 在某层 matmul 前 AllGather 的究竟是 training samples、activations、parameters，还是 gradients？

学习者第二轮回答：

```text
1. 完整的 gradient。
2. 是的，但是分别保留 G0、G1 的切片。
3. weights、gradient，以及矩阵乘法需要的参数和状态切片。
```

点评：

1. 正确。更完整地说，两个 rank 都持有相同的 aggregated full gradient，通常是 `(G0+G1)/2`；若只描述 SUM reduction，则写作 `G0+G1`。
2. “完成 reduction”正确，但最终对象需要纠正：每个 rank 保留的是聚合结果 `G0+G1` 的不同 shard，不是分别保留 `G0`、`G1` 的 shard。
3. FSDP forward matmul 前 AllGather 的是该层 parameters，例如 weight/bias。gradient 尚未产生；Adam optimizer states 不参与 forward matmul，仍保持分片。

最小符号例子：

```text
G0 = [a0, a1]
G1 = [b0, b1]

ReduceScatter-SUM:
rank 0 -> [a0 + b0]
rank 1 -> [a1 + b1]
```

最终确认回答：

```text
1. AllGather 计算所需的 parameter shards。
2. 每个 rank 保留 G0+G1 结果矩阵的一个 shard。
```

判定：`passed`。

### D4-Q5 — TP 中 activation 布局为什么会变化

仍然计算：

```text
X[8,12] @ W[12,16] -> Y[8,16]
```

两个 TP rank 有两种切法。

Case A：沿输出维 `K` 切 `W`：

```text
rank 0: W0 = W[:, 0:8]    # [12,8]
rank 1: W1 = W[:, 8:16]   # [12,8]
```

Case B：沿收缩维 `J` 切 `W`，并对齐切 `X`：

```text
rank 0: X0 = X[:, 0:6],  W0 = W[0:6, :]    # [8,6] @ [6,16]
rank 1: X1 = X[:, 6:12], W1 = W[6:12, :]   # [8,6] @ [6,16]
```

请回答：

1. Case A 中，每个 rank 计算出的 local output shape 是什么？两个输出是不同坐标的 shards，还是同一坐标的 partial sums？
2. Case A 若需要完整 `Y`，应该 concat/AllGather，还是 AllReduce？
3. Case B 中，两个 local output 的 shape 是什么？完整 `Y` 与它们是什么关系，需要哪种 reduction collective？
4. 根据这两个 case，“TP 的 activation 永远是分片的”这句话是否准确？为什么？

学习者回答：

```text
1. [8,8]，不同坐标。
2. AllGather/concat。
3. [8,16]，需要 AllReduce 相加。
4. 不准确；TP 可以分片但不一定永远分片，可能取决于矩阵大小。
   大多数情况下应该是分片的，本质是把矩阵计算分成 partial 矩阵计算。
```

点评：

1. 正确。Case A 的两个 `[8,8]` local outputs 对应互不重叠的 `K` 坐标。
2. 正确。需要完整 `Y` 时拼合不同坐标，使用 concat/AllGather，而不是数值 reduction。
3. 正确。Case B 的两个 `[8,16]` local outputs 是同一输出坐标的 partial sums，完整 `Y=P0+P1`，需要 AllReduce；若目标仍是分片输出，也可使用 ReduceScatter。
4. “不永远分片”正确，但布局不是由矩阵大小直接决定。矩阵大小会影响是否值得采用 TP、采用多大 TP degree；具体 activation 是复制、分片还是未规约 partial，取决于 parameter sharding 维度、当前计算产生的布局、下游期望布局和所选 collective。

需要修正的概括：

- column-parallel 产生的是不同输出坐标上的完整值，不是 partial sums；
- row-parallel 沿收缩维计算，才产生同一输出坐标的 partial sums；
- 经典 column-parallel/row-parallel 组合会让 activation 在复制与分片布局之间切换，以避免每层都 AllGather 完整 activation。

判定：`shape and collective passed; activation-layout cause needs_final_check`。

### D4-Q5 最终确认

按上述两个 case 填四个空：

```text
Case A（column-parallel）：
input X  = __________
output Y = __________（不做 AllGather 时）

Case B（row-parallel + AllReduce）：
input X  = __________
output Y = __________（AllReduce 后）
```

学习者回答：

```text
Case A:
input X  = [8,12]，完整复制
output Y = [8,8]，分片的 Y

Case B:
input X  = 分片的 X；W 也沿收缩维分片
output Y = AllReduce 后的完整 Y[8,16]
```

点评：四项均正确。

判定：`passed`。

### D4-Q6 — Pipeline Parallelism 切了什么

把一个 24-layer Transformer 切成两个 pipeline stages：

```text
rank 0 / stage 0: layers 0..11
rank 1 / stage 1: layers 12..23 + loss
```

暂时不叠加 DP、FSDP 或 TP，只看纯 PP。对一个 micro-batch：

```text
tokens -> stage 0 -> boundary hidden state h -> stage 1 -> loss
```

请回答：

1. 两个 rank 是否都持有完整模型 parameters 和完整 Adam optimizer states？各自实际持有什么？
2. forward 时，stage 0 需要把什么对象发送给 stage 1？这是 parameter AllGather、gradient AllReduce，还是 stage 间的 activation send/recv？
3. backward 时，stage 1 需要把什么对象传回 stage 0？
4. 为什么只使用一个 micro-batch 时会有明显 pipeline bubble？增加多个 micro-batches 主要改善的是模型显存、单次 matmul shape，还是 stage 间的时间重叠？

学习者第一轮回答：

```text
1. 取决于 FSDP 或 DDP；FSDP 不需要 Adam states，只需要本层 parameters，
   forward 还需保留 activation；DDP 需要。
2. 发送 activation X；具体可能取决于架构，FSDP 可能使用 ReduceScatter；
   主要目的是 activation send/recv。
3. backward 需要 gradient 和 Adam states，传完后 activation 可以清除。
4. 不理解 pipeline bubble；猜测多个 micro-batches 主要改善显存和 arithmetic intensity，
   但提高 OOM 风险。
```

点评：

1. 题目限定纯 PP。每个 stage 只持有自己负责的 layers 的 parameters、parameter gradients 和对应 Adam states。Adam states 是优化器更新所必需的；FSDP 也需要，只是按 data-parallel ranks 分片或选择 offload。
2. 核心正确：stage 0 向 stage 1 发送 boundary activation `h`。这是 stage 间 point-to-point send/recv，不是 FSDP ReduceScatter；FSDP/DP/TP 可以与 PP 正交叠加，但不改变这条 PP 边界语义。
3. stage 1 传回的是 `∂L/∂h`，即 boundary activation 的 gradient。parameter gradients 和 Adam states 都留在拥有相应 layers 的本地 stage，不跨 PP 边界传递。某个 activation 在其 backward 完成后才可释放；是否保存或重算还受 activation checkpointing 影响。
4. pipeline bubble 是某个 stage 因依赖尚未到达或流水线正在填充/排空而空闲的时间槽。增加 micro-batches 的主要收益是让不同 stages 同时处理不同 micro-batches，提高 stage 间时间重叠和设备利用率。若每个 micro-batch shape 不变，单次 matmul shape/arithmetic intensity 并不会因此自动改变；多个 in-flight micro-batches 通常会增加 activation memory，而不是改善显存。

只看 forward 的最小时间线：

```text
一个 micro-batch：
time:    t0      t1
stage 0: F0      idle
stage 1: idle    F0

四个 micro-batches：
time:    t0      t1      t2      t3      t4
stage 0: F0      F1      F2      F3      idle
stage 1: idle    F0      F1      F2      F3
```

首尾仍有 fill/drain bubble，但中间两个 stages 可以同时工作。训练还需加入 backward 和具体调度（如 1F1B），核心含义不变。

判定：`forward boundary object passed; PP state/backward/bubble need_retry`。

### D4-Q6 定向复测

只回答三点：

1. 纯 PP 中，stage 0 是否保存 stage 1 的 Adam states？它保存哪些 Adam states？
2. backward 跨 `stage 1 -> stage 0` 边界发送的是 parameter gradient，还是 `∂L/∂h`？
3. 保持单个 pipeline micro-batch 的 shape 不变，增加一次 schedule 中的 micro-batch 数量，主要减少什么空闲现象？通常会让 in-flight activation memory 增加还是减少？

学习者第二轮回答：

```text
1. 保存，跨 step 的 Adam states 给 backward 使用。
2. 传 gradient、optimizer states。
3. 减少 stage 0/1 之间的时间重叠；in-flight activation memory 通常增加。
```

点评：

1. 不正确。stage 0 只保存自己 layers 对应 parameters 的 Adam states，不保存 stage 1 的 Adam states。Adam states 的确跨 step 持久存在，但由 `optimizer.step()` 使用，不由 backward 使用。
2. 不正确。stage 1 向 stage 0 只发送 boundary activation gradient `∂L/∂h`。stage 1 的 parameter gradients 和 optimizer states 都留在 stage 1。
3. activation memory 增加正确；时间方向说反。增加 pipeline micro-batches 会增加 stages 之间的时间重叠，从而减少 fill/drain bubble 和 idle fraction。

严格区分三个阶段：

```text
forward:
stage 0 -- h --> stage 1

backward:
stage 0 <-- ∂L/∂h -- stage 1
各 stage 使用本地 parameters、保存/重算的 activations 和上游 gradient，
计算自己的 parameter gradients。

optimizer step:
stage 0: Adam(theta0, dtheta0, m0, v0)
stage 1: Adam(theta1, dtheta1, m1, v1)
```

`m0/v0` 与 `m1/v1` 是各 stage 本地持久状态，不跨 PP boundary 传输。

判定：`activation-memory direction passed; state ownership/backward object/overlap direction need_retry`。

### D4-Q6 最终确认

只用下面六个符号填空：

```text
theta0：stage 0 parameters
theta1：stage 1 parameters
dtheta0/dtheta1：对应 parameter gradients
m0,v0 / m1,v1：对应 Adam states
h：forward boundary activation
dh：∂L/∂h
```

```text
forward，stage 0 -> stage 1：________
backward，stage 1 -> stage 0：________

stage 0 本地 optimizer.step 使用：________
stage 1 本地 optimizer.step 使用：________

增加 pipeline micro-batches：
stage 时间重叠 ________，bubble ________，in-flight activation memory 通常 ________
```
