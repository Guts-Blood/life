# Day 03 Reading Quiz — Matmul Roofline 到 H100 SFT

日期：`2026-07-26`  
状态：`completed`  
方式：Socratic quiz；一次只讨论一道题，先回答、再纠错、再进入下一题。

## Quiz 目标

| 模块 | 希望真正掌握的能力 | 状态 |
|---|---|---|
| A. Matmul roofline | 从矩阵 shape 推导 FLOPs、HBM bytes 与 arithmetic intensity | `completed` |
| B. H100 critical point | 用实际/有效硬件规格判断 compute-bound 与 bandwidth-bound | `completed_core; actual_SKU_pending` |
| C. Network roofline | 将 collective 通信量、带宽、overlap 映射到多卡 step time | `completed` |
| D. SFT step-time 与 MFU | 从参数量、tokens、GPU 数和 MFU 估算训练时间，并说明误差来源 | `completed` |

## 当前进度

- D3-Q1：matmul FLOPs、HBM bytes、arithmetic intensity，`passed`。
- D3-Q2：用 `I_op/I_hw` 与 `T_math/T_HBM` 判断 bound，`passed`。
- D3-Q3：critical local tokens、micro-batch 与 accumulation，`concept passed; arithmetic corrected`。
- 能力模块：A、B Core、C、D 已完成；Day 03 Reading Quiz Core `100%`。
- 当前错误画像：没有方向性概念错误；出现一次临界值除法的机械错误。

## 出题原则

- 不重复 Day 01 已稳定掌握的基础定义，重点考察 matmul、local tokens 和真实 H100 的连接。
- 每题先做符号推导，再代数值；公式必须带单位。
- 区分硬件 theoretical peak、有效 benchmark 吞吐与真实训练 MFU。
- 题目根据回答动态增加或跳过追问。

## 当前题目

### D3-Q1 — 从矩阵 shape 推导 arithmetic intensity

在一张 GPU 上计算：

```text
X [N,D]
W [D,F]
Y = X @ W → [N,F]
```

假设：

- `X`、`W` 各从 HBM 读取一次，`Y` 向 HBM 写入一次；
- 三者都是 BF16，即 `2 bytes/element`；
- 暂时忽略 bias、kernel launch、workspace，以及 tile/cache 导致的额外流量。

请回答：

1. 写出 matmul 的 FLOPs `F_math`。
2. 写出最小 HBM 流量 `M_HBM`，单位为 bytes。
3. 写出 arithmetic intensity `I_op = F_math / M_HBM`。
4. 当 `D=F=4096`、`N=256` 时，算出 `F_math`、`M_HBM` 和 `I_op`。先保留完整表达式，再给近似值。

先不要判断它在 H100 上属于哪一侧；下一问会用硬件临界 intensity 做比较。

#### D3-Q1 回答与点评

学习者回答：

```text
F_math = 2NDF
M_HBM  = 2(ND + DF + NF) bytes
I_op   ≈ N
```

前两项正确。完整的 arithmetic intensity 为：

```text
I_op
= 2NDF FLOPs / [2(ND + DF + NF) bytes]
= NDF / (ND + DF + NF) FLOPs/byte
```

当 `D=F=d` 时：

```text
I_op = Nd / (d + 2N)
```

因此 `I_op ≈ N FLOPs/byte` 需要 `N ≪ d`，即 weight matrix 的 HBM 流量主导。它不是对所有 `N` 都成立；当 `N → ∞` 时，`I_op → d/2`，因为读 `X`、写 `Y` 的流量最终不可忽略。

代入 `d=4096`、`N=256`：

```text
F_math
= 2 × 256 × 4096 × 4096
= 8,589,934,592 FLOPs
≈ 8.59 GFLOPs

M_HBM
= 2 × (256×4096 + 4096×4096 + 256×4096)
= 37,748,736 bytes
= 36 MiB
≈ 37.75 MB

I_op
= 8,589,934,592 / 37,748,736
≈ 227.56 FLOPs/byte
≈ 0.889N FLOPs/byte
```

判定：`passed`。数值部分由答案补齐；需保留的关键边界是 `I_op≈N` 仅在 `N≪d` 时成立。

### D3-Q2 — H100 临界 intensity

沿用 D3-Q1 的 matmul，并先使用一组明确的有效值而非宣传峰值：

```text
effective compute C = 600 TFLOPs/s
effective HBM bandwidth W = 3 TB/s
operation intensity I_op ≈ 227.56 FLOPs/byte
```

请判断：

1. 硬件临界 intensity `I_hw=C/W` 是多少？
2. 该 matmul 在简化 Roofline 中属于 compute-bound 还是 HBM-bound？
3. 分别算出 `T_math`、`T_HBM` 与理想 Roofline 下界。

#### D3-Q2 回答与点评

学习者回答：

```text
I_hw = 200 FLOPs/byte
I_op = 227.56 FLOPs/byte > I_hw
因此是 compute-bound
```

两项均正确。`I_hw` 描述硬件 compute 与 HBM bandwidth 的比例；`I_op` 描述 operation 每搬运一个 byte 要做多少 FLOPs。二者比较等价于比较 `T_math` 与 `T_HBM`。

第 3 问需要沿用 D3-Q1 的总 FLOPs 与 bytes，但题面没有重新列出，导致题目不够自包含。补全输入：

```text
F_math = 8,589,934,592 FLOPs
M_HBM  = 37,748,736 bytes
C      = 600×10^12 FLOPs/s
W      = 3×10^12 bytes/s
```

于是：

```text
T_math
= F_math / C
= 8,589,934,592 / (600×10^12)
≈ 14.32 μs

T_HBM
= M_HBM / W
= 37,748,736 / (3×10^12)
≈ 12.58 μs

ideal Roofline lower bound
= max(T_math, T_HBM)
≈ 14.32 μs
```

由于 `T_math > T_HBM`，时间比较也得到 compute-bound。判定：`passed`。

### D3-Q3 — Critical local tokens

对于 `D=F=d=4096` 的 BF16 方阵 matmul：

```text
I_op(N) = Nd / (d + 2N)
I_hw    = 200 FLOPs/byte
```

请回答：

1. 解 `I_op(N_critical)=I_hw`，求临界 `N_critical`。
2. `N=128` 和 `N=256` 分别位于 Roofline 的哪一侧？
3. 在 Transformer 训练的 linear layer 中，`N` 通常由哪些维度合并而来？为什么提高 micro-batch size 可能提升 MFU，而只提高 gradient accumulation steps 通常不会扩大单次 GEMM 的 `N`？

#### D3-Q3 回答与点评

学习者正确把握了临界点的含义：`N>N_critical` 时进入 compute-bound 一侧，`N<N_critical` 时位于 HBM-bound 一侧。方程写对，但最后除法有机械误差：

```text
4096N = 200(4096 + 2N)
4096N = 819200 + 400N
3696N = 819200
N_critical = 819200 / 3696
           ≈ 221.65
```

实际离散 token-row 可把边界记为约 `222`。`N=182.21` 代回后只有约 `167.35 FLOPs/byte`，尚未达到硬件临界 intensity。若使用 `I_op≈N` 的 weight-dominated 近似，会粗略预测 `N_critical≈200`；完整公式计入 `X/Y` 流量后，临界点上移到约 `222`。

因此：

```text
N=128 → I_op≈120.47 FLOPs/byte < 200 → HBM-bound
N=256 → I_op≈227.56 FLOPs/byte > 200 → compute-bound
```

在普通 dense/token-wise Transformer linear layer 中，通常先把本 rank 当前 micro-batch 的 token 维度展平：

```text
X [B_micro, S, D]
→ X_flat [N, D]

N = B_micro × S
```

这里的 `N` 是单次 GEMM 在该 GPU/rank 上实际处理的 token rows：

- 有 padding 时，kernel 通常仍计算 padding rows，所以 GEMM 的 `N` 常是 padded `B_micro×S`，不等于有效 label-token 数。
- Data Parallel 的每个 rank 使用自己的 local micro-batch，因此看的是 per-rank `B_micro`，不是 global batch。
- 常见 Tensor Parallel linear 不一定切 token 维；Sequence/Context Parallel 则可能让某些 kernel 的 local `N` 进一步变小。

增大 `B_micro` 会直接扩大同一次 GEMM 的 `N`，让一份 weight 在更多 token rows 上复用，提高 arithmetic intensity，并通常改善 Tensor Core tile occupancy、摊薄 kernel-launch overhead。但它也会增加 activation memory；进入 compute-bound 后，继续增大 `N` 的收益会变小。

Gradient accumulation 不会把多次 forward/backward 合并成一个大 GEMM：

```text
micro-step 1: GEMM with N
micro-step 2: GEMM with N
...
micro-step G: GEMM with N
optimizer.step()
```

它把一个 optimizer step 的总 token 数扩大为约 `G×B_micro×S`，但每次 GEMM 的 `N` 仍是 `B_micro×S`，所以通常不会改善单次 kernel 的 arithmetic intensity 或 shape efficiency。

判定：`concept passed; arithmetic corrected; local-token interpretation taught`。

#### `B_micro` 与日常所说 batch size

学习者对第 3 问的原始理解正确：`N` 通常合并 sequence length 和 batch 维，增加 gradient accumulation steps 与单次 GEMM 的 `N` 无关。

之所以写 `B_micro`，是因为 “batch size” 在论文、训练框架和日常交流中可能指三个不同量：

```text
B_micro = 每个 GPU/rank、每个 micro-step 同时处理的 sequence 数
G       = gradient accumulation steps
D_dp    = data-parallel world size

B_global = B_micro × G × D_dp
```

`B_global` 是一次 `optimizer.step()` 在所有 data-parallel ranks 上合计消费的 sequence 数。若忽略 padding：

```text
tokens per optimizer step
= B_global × S
= B_micro × G × D_dp × S
```

但单个 data-parallel rank 上一次 token-wise linear GEMM 的行数仍通常是：

```text
N_local = B_micro × S
```

常见命名映射：

- Hugging Face `per_device_train_batch_size` 通常对应 `B_micro`。
- Megatron `micro_batch_size` 直接对应 `B_micro`。
- “effective batch size” 通常对应 `B_global`，但使用前仍应确认说的是 sequences、tokens 还是有效 label tokens。

配置方式取决于框架：

- 常见 Trainer 风格会设置 `per-device batch size`、`gradient accumulation steps` 和 GPU 数，由此得到 `B_global`。
- Megatron 风格可以显式设置 `micro_batch_size` 与目标 `global_batch_size`，再由框架推导需要多少个 micro-batches/accumulation steps。
- 原生 PyTorch `DataLoader(batch_size=...)` 在 DDP 中通常表示每个 process/rank 每次拿到的 local batch，不能直接当成跨所有 GPU 的 global batch。

所以，`B_micro` 通常最接近日常训练命令中的“单卡 batch size”，但 `B_global` 既可能是推导量，也可能是显式配置的目标量。无论配置入口如何，都必须满足：

```text
B_global = B_micro × G × D_dp
```

若不能整除，框架需要调整 micro-batch 数、丢弃/补齐样本或拒绝该配置。

### D3-Q3.1 — Micro-batch 与 accumulation 迁移检查

单卡、相同 sequence length，两个配置每个 optimizer step 都处理 `16,384` 个 token：

```text
A: B_micro=1, S=2048, gradient_accumulation=8
B: B_micro=4, S=2048, gradient_accumulation=2
```

只回答三个判断，不需要计算具体 FLOPs：

1. 两者单次 token-wise linear GEMM 的 `N` 分别是多少？
2. 哪个配置通常更可能得到较高的 matmul arithmetic intensity/MFU？为什么？
3. 哪个配置通常需要更多 activation memory？

讨论中已确认：

```text
A: N=1×2048=2048
B: N=4×2048=8192
```

B 通常有更高的 matmul arithmetic intensity/shape efficiency，也需要更多 activation memory。两者的 optimizer-step global tokens 相同，不代表单次 GEMM shape 相同。该迁移点已在 `B_micro/B_global` 追问中掌握，不再单独复测。

### D3-Q4 — Theoretical peak、有效吞吐、MFU 与 GPU-Util

先使用一组取整后的教学数值，不绑定具体 H100 SKU：

```text
BF16 theoretical peak C_peak = 1000 TFLOPs/s
实测大尺寸 GEMM吞吐 C_gemm = 700 TFLOPs/s
训练 step 平均有效吞吐 C_train = 350 TFLOPs/s
nvidia-smi GPU-Util = 99%
```

请回答：

1. 相对 theoretical peak 的 MFU 是多少？
2. `C_train/C_gemm` 是多少？它与第 1 问分别回答什么问题？
3. 为什么 `GPU-Util=99%` 仍然可以与较低 MFU 同时出现？至少说出三个原因。
4. 在运行前估算 step time 时，`C_peak`、`C_gemm`、`C_train` 应如何使用或标注，才能避免给出虚假精确的预测？

#### D3-Q4 回答与点评

学习者回答：

```text
MFU = C_train/C_peak = 0.35
C_train/C_gemm = 0.5
```

数值正确，但两个比例回答的问题不同：

```text
MFU_peak = C_train/C_peak = 35%
```

它回答训练 workload 实际完成的模型 FLOPs 占对应 dtype、dense/sparse 口径下硬件理论峰值的比例。

```text
η_vs_gemm = C_train/C_gemm = 50%
```

它回答训练吞吐相对“这类 shape/dtype 在当前软件与硬件上实测可达到的 GEMM ceiling”的比例。它通常更能区分不可达的硬件宣传峰值与 workload/runtime 带来的损失，但不是标准 MFU 定义。

学习者对 `GPU-Util=99%` 与低 MFU 可以共存的解释完整，正确指出 GPU-Util 更接近采样窗口中设备是否繁忙，而非 Tensor Core 完成了多少有用模型 FLOPs。列出的原因包括 HBM-bound、collective/搬运、小或不友好 shape、非 GEMM 算子、launch/同步/bubble，以及未充分使用 Tensor Core。判定：`passed`。

运行前预测应分层标注：

```text
T_peak_lower = F / (n_gpu × C_peak)
```

- `C_peak` 只给 theoretical compute lower bound；必须标注 H100 SKU、dtype、dense/sparse 口径，不能当成训练可持续吞吐。
- `C_gemm` 可给相似 shape/dtype 下更现实的 matmul-only lower bound，但仍不包含完整训练 step 的非 GEMM、通信与框架开销。
- `C_train` 是特定 model/config/software/topology 的经验校准值。已有相近 run 时可用于中心预测；没有时应使用多个 MFU scenario，例如 `30%/40%/50%`：

```text
T_math_scenario
= F / (n_gpu × C_peak × MFU_scenario)
```

最终 step-time 预测还需要单列 `T_HBM`、`T_network` 与其他 overhead，并给区间、假设和待 benchmark 项，而不是只报一个小数很多的单点值。

模块 B 的核心概念已通过；具体 AutoDL H100 SKU、拓扑与实测规格仍需在 worksheet 中补证据。

### D3-Q5 — Network Roofline（含 Ring AllReduce 提前预览）

范围澄清：

- Day 03 Part 1 `Network communication rooflines` 的 Core 是：给定通信量 `M` 与有效带宽 `W`，使用 `T_network=M/W`，再分析它与 compute 的 overlap 和关键路径。
- 从 sharding 推导为什么需要某种 collective，以及 AllGather/ReduceScatter/AllToAll 的 bytes，原计划在 Day 04 Part 3。
- DP/FSDP/TP 为什么触发相应 collective，原计划在 Day 05 Part 5。
- Ring/collective 在 GPU 上如何实现，以及 link bandwidth、NCCL `algbw/busbw` 和真实 topology，原计划在 Day 18 Part 12。

因此，下面的 Ring AllReduce 分解属于提前预览，用于说明 `M_rank` 从哪里来，不作为 Day 03 必须独立推导的考点。Day 03 只要求在 `M_rank` 已知后正确计算时间与 overlap。

一个 `p=8` 的 Data Parallel group 对大小为 `M=1 GB` 的 gradient bucket 做 Ring AllReduce。为避免单位歧义，本题全部使用十进制单位：

```text
1 GB = 10^9 bytes
measured effective per-rank transport bandwidth W_transport = 200 GB/s
通信发起后可用于 overlap 的 backward compute window = 6 ms
```

Ring AllReduce 每个 rank 的近似网络流量为：

```text
M_rank = 2(p-1)/p × M
```

请回答：

1. 每个 rank 需要传输多少 GB？
2. `T_network = M_rank/W_transport` 是多少 ms？
3. 可隐藏多少通信、暴露多少通信？依赖该 bucket 的下一阶段最早在多少 ms 后开始？
4. 为什么这里应该使用 measured effective bandwidth，而不能直接把 H100/NVLink 的宣传 link bandwidth 代入？

#### D3-Q5 术语修正与讲解

学习者正确指出通信时间应为 `communication bytes / bandwidth`。原题把 `W_alg` 称为 algorithm bandwidth，同时又用实际 ring traffic `M_rank` 作分子，容易与 NCCL benchmark 常见的 `algbw = logical message size / time` 定义冲突。题目现统一改用：

```text
W_transport = 实际 per-rank transport bytes / time
```

若直接读取 NCCL test 的 `algbw`，则应按该工具的 logical message-size 口径计算；其 `busbw` 才会应用 collective-specific traffic factor。不同工具字段不能只看同一个 `GB/s` 单位就混用。

`M=1 GB` 表示每个 rank 最初拥有一份 1GB local gradient bucket。AllReduce 的目标是让每个 rank 最终都拿到所有 ranks 的逐元素 sum/average。Ring 算法把 bucket 切成 `p=8` 个 chunk：

```text
chunk size = M/p = 1GB/8 = 0.125GB
```

第一阶段 ReduceScatter：

```text
p-1 = 7 rounds
每 round 每 rank 发送一个 0.125GB chunk
per-rank sent bytes = 7×0.125GB = 0.875GB
```

数据在环上传递时逐步与其他 rank 的对应 chunk reduce。阶段结束后，每个 rank 只持有一个已经完成全局 reduce 的 chunk。

第二阶段 AllGather：

```text
p-1 = 7 rounds
每 round 每 rank 再发送一个 0.125GB reduced chunk
per-rank sent bytes = 0.875GB
```

阶段结束后，每个 rank 重新收齐完整的 1GB reduced bucket。因此：

```text
M_rank
= 0.875GB + 0.875GB
= 2(p-1)/p × M
= 1.75GB
```

这里按每个 rank 的发送流量计；每个 rank 也会接收同量数据。若链路/benchmark 带宽使用 full-duplex、单向或 send+receive 合计口径，必须相应匹配，不能再盲目乘 2。

在本题明确的 transport-bandwidth 口径下：

```text
T_network = 1.75GB / 200GB/s
          = 0.00875s
          = 8.75ms
```

与 `6ms` compute window 同时开始时：

```text
hidden communication  = min(8.75, 6) = 6ms
exposed communication = max(8.75-6, 0) = 2.75ms
next dependent stage  = max(8.75, 6) = 8.75ms
```

学习者回答的 `6ms` 是可隐藏的通信量，不是全部通信时间或 exposed communication。

宣传 link bandwidth 不能直接用于 collective 时间预测，因为真实有效值还取决于 topology、单向/双向与 aggregate 口径、collective algorithm、协议与 chunk size、并发/拥塞、跨 PCIe switch 或跨节点链路，以及软件调度。应使用与目标 collective、message size、GPU 数和 topology 匹配的实测值。

当前判定：`Day 03 bytes/bandwidth concept correct; ring traffic treated as optional preview; terminology issue fixed`。

### D3-Q5.1 — Overlap 快速复测

仍使用：

```text
T_network = 8.75ms
compute overlap window = 10ms
```

只回答：

1. hidden communication 是多少？
2. exposed communication 是多少？
3. 下一依赖阶段最早在多少 ms 后开始？

#### D3-Q5.1 回答与点评

学习者回答：

```text
hidden communication  = 8.75ms
exposed communication = 1.25ms
next dependent stage  = 10ms
```

第 1、3 项正确。第 2 项需要纠正：

```text
hidden communication
= min(T_network, T_compute_window)
= min(8.75, 10)
= 8.75ms

exposed communication
= max(T_network - T_compute_window, 0)
= max(8.75 - 10, 0)
= 0ms

compute tail after communication finishes
= max(T_compute_window - T_network, 0)
= 10 - 8.75
= 1.25ms
```

`1.25ms` 是通信已经完成后仍在继续执行的 compute tail，不是 exposed communication。下一阶段依赖 compute 与 communication 都完成，因此最早开始时间仍为：

```text
max(8.75, 10) = 10ms
```

判定：`hidden and critical-path timing passed; exposed-communication label needs quick recheck`。

### D3-Q5.2 — 单点术语复测

仍使用同一时间线，只回答一句：

> `1.25ms` 具体是什么时间？为什么它不能称为 exposed communication？

学习者已理解 `1.25ms` 是 communication 完成后的 compute tail。更精确的定义是：

```text
exposed communication
= 没有被 eligible compute overlap 隐藏、
  因而留在目标 critical path 上的通信时间
```

它不必表示整台设备在该时间段绝对只执行 communication；关键是这段通信没有被所分析的 compute window 隐藏，并延迟了后续依赖操作。判定：`passed`。模块 C 完成；Ring traffic 细节仍只算后续章节预览。

### D3-Q6 — Qwen3-1.7B SFT step-time 与 MFU scenarios

使用取整后的 worksheet 假设：

```text
dense parameters P = 1.7×10^9
training FLOPs/token ≈ 6P

GPU count = 1
B_micro = 2 sequences
sequence length S = 2048 tokens
gradient accumulation G = 4

H100 BF16 dense peak C_peak = 1.0×10^15 FLOPs/s
```

请回答：

1. 每个 optimizer step 处理多少 token？
2. `F_step ≈ 6P×tokens_per_step` 是多少 FLOPs？
3. 先算 `MFU=100%` 的 theoretical compute lower bound，再算 `MFU=30%/40%/50%` 三种 scenario 的 step time。
4. 如果这里的 MFU 是用端到端训练 wall time 定义的，得到 `F_step/(C_peak×MFU)` 后，还能否再把同一 run 的 HBM、network、launch 等时间直接相加？解释 accounting boundary。

#### D3-Q6 回答与点评

学习者回答：

```text
tokens_per_step
= B_micro × S × G
= 2 × 2048 × 4
= 16,384 tokens

F_step
= 6 × 1.7×10^9 × 16,384
= 167,116,800,000,000 FLOPs
= 1.671168×10^14 FLOPs
≈ 167.12 TFLOPs
```

两项正确。训练 FLOP count 是对数学运算量的 accounting，本身不因 BF16/FP32 改变；dtype 决定 bytes、数值行为，以及分母应该使用哪一种硬件 peak/effective throughput。本题给出的 `C_peak` 已明确是 BF16 dense peak。

时间结果：

```text
T_step(MFU)
= F_step / (C_peak × MFU)

MFU=100%: 1.671168×10^14 / 1.0×10^15
        ≈ 0.1671s

MFU=30%: ≈ 0.5571s
MFU=40%: ≈ 0.4178s
MFU=50%: ≈ 0.3342s
```

第 4 问回答正确。MFU 的通用定义必须先固定同一个测量区间：

```text
MFU_end_to_end
= F_interval / (n_gpu × C_peak × T_interval)
```

本题的区间是一个 optimizer step，因此 `F_interval=F_step`、`T_interval=T_step_wall`：

```text
MFU_end_to_end
= F_step / (n_gpu × C_peak × T_step_wall)

T_step_wall
= F_step / (n_gpu × C_peak × MFU_end_to_end)
```

此前把分子写成 `F_model` 过于含糊，是记号错误；实际数值计算代入的是正确的 `F_step`。如果计时改为一个 micro-step，就必须同步改用 `F_micro_step`；如果测量多个 steps，分子也必须是这段区间内的 FLOPs 总和。

这个 `T_step_wall` 已经通过较低 MFU 吸收了未达到峰值的 matmul、HBM stall、network、launch/sync、bubble 等落在计时边界内的效率损失。之后再把同一批 overhead 直接相加会 double count。

两种合法但不能混用的建模方式：

```text
Top-down:
F_interval + assumed/measured end-to-end MFU
→ 直接预测匹配区间的 T_interval
```

```text
Bottom-up:
逐 operation 的 compute/HBM/network/overlap/overhead
→ 沿 critical path 预测 T_wall
→ 再计算 implied MFU
```

若某个“compute efficiency”只描述 GEMM kernel 而排除了 communication/launch 等时间，就不应把它称为 end-to-end MFU；此时可以在 bottom-up 模型中再加入其他分项。计时是否包含 data loading、checkpoint、eval 等也必须显式定义。

判定：`passed; numeric scenarios supplied; dtype/FLOP-count boundary reinforced`。

### D3-Q7 — 用真实 step time 反推 MFU

沿用 D3-Q6 的 workload：

```text
F_step = 1.671168×10^14 FLOPs
tokens_per_step = 16,384
C_peak = 1.0×10^15 FLOPs/s
```

现在 profiler 测得稳定区间：

```text
T_wall = 0.50s/optimizer-step
其中有效 label tokens 占全部 processed tokens 的 60%
```

请回答：

1. 实测 end-to-end MFU 是多少？
2. processed tokens/s 与 effective label tokens/s 分别是多少？
3. `30%/40%/50%` 三个预注册 scenario 中，哪个最接近实测？
4. 如果 prediction 与 profiler 仍不一致，列出三个必须检查的 accounting/profiler 边界。不要只写“真实情况复杂”。

#### D3-Q7 回答与点评

学习者写出的 MFU 公式正确，但科学计数法计算错误：

```text
MFU
= 1.671168×10^14 / (1×10^15×0.50)
= 1.671168×10^14 / 5×10^14
= 1.671168 / 5
= 0.3342336
≈ 33.42%
```

也可使用快速 sanity check：

```text
MFU = T_peak_lower / T_actual
    = 0.1671168 / 0.50
    ≈ 0.3342
```

实际 step time 约为理论下界的 3 倍，因此 MFU 应约为三分之一，不可能是 `10^-6`。

Throughput 必须除以 wall time：

```text
processed tokens/s
= 16,384 tokens / 0.50s
= 32,768 tokens/s

effective label tokens/s
= 32,768 × 0.60
= 19,660.8 label tokens/s
```

实测 MFU `33.42%` 位于 30% 与 40% scenario 之间，更接近 30%。从 step time 看也一致：

```text
actual             = 0.5000s
30% scenario       = 0.5571s  # absolute gap ≈ 0.0571s
40% scenario       = 0.4178s  # absolute gap ≈ 0.0822s
```

Prediction 与 profiler 不一致时，至少检查以下 accounting boundaries：

1. **FLOPs numerator**：`6P×tokens` 是否只是一阶近似；是否漏了 quadratic attention、embedding/lm-head、non-matmul ops、activation recomputation；MoE 是否错用 total/active parameters。
2. **Token denominator/grain**：使用的是 padded processed tokens、non-padding tokens 还是 label tokens；packing、variable length、micro-batch、accumulation、DP world size 是否与公式一致。
3. **Hardware denominator**：H100 SKU、BF16/FP8、dense/sparse peak、GPU 数是否匹配；不能把 sparse/FP8 宣传峰值用于 BF16 dense workload。
4. **Wall-time boundary**：测的是 micro-step 还是 optimizer step；是否包含 optimizer、grad clipping、data loading、logging、checkpoint/eval、CUDA synchronization；是否排除了 warmup/compile。
5. **Distributed critical path**：collective bytes/effective bandwidth/topology 是否正确；通信是 hidden 还是 exposed；是否有 straggler、load imbalance 或 pipeline bubble。
6. **Runtime stability**：是否处于 steady state；是否有 thermal/power throttling、动态时钟、allocator/cache/compilation 抖动。

判定：`formula concept passed; scientific notation and throughput arithmetic corrected; profiler boundaries taught`。

### D3-Q7.1 — 数值快速复测

同一 workload 改为：

```text
T_wall = 0.40s
F_step = 1.671168×10^14 FLOPs
tokens_per_step = 16,384
C_peak = 1.0×10^15 FLOPs/s
label fraction = 60%
```

请给出：

1. MFU；
2. processed tokens/s；
3. effective label tokens/s；
4. 30%/40%/50% 中最接近哪个 scenario。

#### D3-Q7.1 回答与点评

学习者回答：

```text
MFU ≈ 0.4175
processed tokens/s = 4096
label tokens/s = 4096×0.6
closest scenario = 40%
```

MFU 与 scenario 判断正确；更精确的 MFU 为：

```text
MFU
= 1.671168×10^14 / (1×10^15×0.40)
= 0.417792
≈ 41.78%
```

Throughput 少写了一个 0：

```text
processed tokens/s
= 16,384 / 0.40
= 40,960 tokens/s

effective label tokens/s
= 40,960 × 0.60
= 24,576 label tokens/s
```

Sanity check：

```text
0.40s/step = 2.5 steps/s
```

因此每秒 processed tokens 应为每 step token 数的 `2.5×`，必然大于 `16,384`，不可能是 `4,096`。

判定：`MFU formula and scenario passed; throughput factor-of-10 arithmetic corrected`。

### D3-Q8 — 最终综合：把点估计写成可验证预测

你准备向同事报告：

> “Qwen3-1.7B 在单张 H100 上预计约 `0.42s/optimizer-step`。”

请把这句话改写为一个不会制造虚假精确、且能被 profiler 验证的短预测。至少包含：

1. `known`：明确 workload 与硬件输入；
2. `estimated`：说明 `6P` 与 MFU scenario；
3. `predicted range`：给一个合理区间，不只报 `0.42s` 单点；
4. `must benchmark`：列出至少三个真实运行中要记录的指标。

#### D3-Q8 回答与最终修正

学习者给出的预测已经包含 model、单卡 H100、`B_micro=2`、`G=4` 和 `MFU=30%–50%` scenario，方向正确。需要修正三个边界：

1. `0.42s/step` 是时间；`MFU=0.42` 是无量纲比例，两者不能混写。恰好数值接近只是巧合。
2. 本题 `MFU=40%` 对应约 `0.4178s/step`；若中心 MFU 真取 `42%`，对应时间约为 `0.398s/step`。
3. `G=4` 表示四个 micro-batches 各自进行 forward/backward、计算并累积 gradient，之后执行一次 `optimizer.step()`；不是四个 batch 才计算一次 gradient。

可直接用于 worksheet 的版本：

```text
Known:
- dense model parameters P≈1.7B
- 1×H100；具体 SKU 与 BF16 dense peak 待实例核验
- worksheet 暂用 C_peak=1.0×10^15 FLOPs/s
- BF16 training
- B_micro=2 sequences/GPU/micro-step
- S=2048 tokens
- gradient accumulation G=4
- DP=1
- processed tokens/optimizer-step=2×2048×4=16,384

Estimated:
- model training FLOPs 使用 F_step≈6P×tokens
- F_step≈1.671168×10^14 FLOPs
- end-to-end MFU scenario=30%/40%/50%

Predicted:
- MFU=30% → ≈0.557s/optimizer-step
- MFU=40% → ≈0.418s/optimizer-step
- MFU=50% → ≈0.334s/optimizer-step
- 因此预注册范围约 0.33–0.56s/optimizer-step，
  中心情景约 0.42s/step（对应 MFU=40%）

Must benchmark:
- steady-state micro-step 与 optimizer-step wall time
- processed tokens/s 与 effective label tokens/s
- peak allocated/reserved HBM memory
- end-to-end MFU，并用 profiler 分解 GEMM、HBM stall、
  optimizer、launch/sync 与 data-loading 时间
```

若实际 instance 的 H100 SKU、clock/power limit 或 BF16 dense peak 不同，必须替换 `C_peak` 后重新计算。

判定：`final prediction structure passed with unit, accumulation, range, and benchmark corrections`。

## Quiz 完成总结

- Matmul FLOPs/HBM bytes/arithmetic intensity：`passed`。
- `I_op/I_hw`、critical local tokens 与 compute/HBM bound：`passed`。
- Micro-batch、global batch、gradient accumulation：`passed`。
- Theoretical peak、effective throughput、MFU、GPU-Util：`passed`。
- Network `M/W`、overlap 与 exposed communication：`passed`；Ring AllReduce 仅作 Day 04/18 提前预览。
- Qwen3-1.7B step-time/MFU scenario 与 profiler accounting boundary：`passed`。
- 仍需工程验证：实际 H100 SKU/topology/spec、worksheet、真实 profiler 与预注册预测对照。

其中第 3 问沿用 D3-Q1 已算出的：

```text
F_math = 8,589,934,592 FLOPs
M_HBM  = 37,748,736 bytes
```

#### D3-Q2 回答与点评

学习者回答：

```text
I_hw = 200 FLOPs/byte
I_op = 227.56 FLOPs/byte > I_hw
```

硬件临界 intensity 正确，但把 bound 的方向说反。直接比较两个时间：

```text
T_math / T_HBM
= (F_math/C) / (M_HBM/W)
= (F_math/M_HBM) / (C/W)
= I_op / I_hw
= 227.56 / 200
≈ 1.138 > 1
```

所以 `T_math > T_HBM`，瓶颈是 compute：

```text
I_op > I_hw  ⇔ compute-bound
I_op < I_hw  ⇔ HBM-bound
```

代入数值：

```text
T_math
= 8,589,934,592 FLOPs / (600×10^12 FLOPs/s)
≈ 14.32 μs

T_HBM
= 37,748,736 bytes / (3×10^12 bytes/s)
≈ 12.58 μs

ideal Roofline lower bound
= max(T_math, T_HBM)
≈ 14.32 μs
```

判定：`I_hw passed; bound direction needs_recheck`。第 3 问原题没有重列 D3-Q1 的 `F_math/M_HBM`，现已补齐题面并给出数值答案。

#### D3-Q2 方向复测

另一个 operation 的：

```text
I_op = 150 FLOPs/byte
I_hw = 200 FLOPs/byte
```

只需回答：它是 compute-bound 还是 HBM-bound？此时仅把 compute `C` 翻倍，理想运行时间会不会明显改善？

#### D3-Q1 回答与点评

学习者回答：

```text
F_math = 2NDF
M_HBM  = 2(ND + DF + NF) bytes
I_op   ≈ F
```

前两式正确。第三式需要修正：

```text
I_op
= F_math / M_HBM
= 2NDF / [2(ND + DF + NF)]
= NDF / (ND + DF + NF) FLOPs/byte
```

其中分母中的 `2` 是 BF16 的 `2 bytes/element`。若用 `b` 表示每个 element
的 bytes，更一般的公式是：

```text
I_op = 2NDF / [b(ND + DF + NF)]
```

代入 `D=F=4096`、`N=256`：

```text
F_math
= 2 × 256 × 4096 × 4096
= 8,589,934,592 FLOPs
≈ 8.59 GFLOPs

M_HBM
= 2 × (256×4096 + 4096×4096 + 256×4096) bytes
= 37,748,736 bytes
= 36 MiB
≈ 37.75 MB

I_op
= 8,589,934,592 / 37,748,736
≈ 227.56 FLOPs/byte
```

对于 `D=F` 的 BF16 方阵：

```text
I_op = ND / (D + 2N)
```

因此：

```text
N ≪ D  → I_op ≈ N
N ≫ D  → I_op ≈ D/2
```

关键直觉：增加本卡参与同一次 matmul 的 local tokens，可以让同一份 weight
被更多 token 复用，所以 arithmetic intensity 上升；但 `X/Y` 流量也随 `N`
增加，因此 intensity 最终会饱和，而不会无限增长。`I_op ≈ F` 一般不成立。

判定：`FLOPs and bytes passed; intensity formula corrected`。

### D3-Q2 — 放到一个 H100 有效 Roofline 上

继续使用 D3-Q1 的 matmul，并先采用 worksheet 中的一组有效值，而不是厂商
theoretical peak：

```text
C_effective = 600 TFLOPs/s
W_effective = 3 TB/s
I_op        ≈ 227.56 FLOPs/byte
```

请回答：

1. 硬件临界 intensity `I_hw = C_effective/W_effective` 是多少？
2. 这个 matmul 是 compute-bound 还是 HBM-bound？
3. 若 `N` 从 `256` 减半为 `128`，使用
   `I_op = ND/(D+2N)` 判断瓶颈是否改变。
