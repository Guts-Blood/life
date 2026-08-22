# Day 30 — slime Architecture：Online RL 编排与 Training System 集成

日期：`2026-08-25`  
状态：`not_started`  
强度：4–5 小时；CPU 源码学习、架构整合与口述答辩

## 主要目标

理解 slime 为什么存在，以及它如何把多个本来独立的系统组成 online RL loop：prompt/data source、SGLang rollout、environment/reward、trajectory buffer、Megatron learner、weight sync 和下一版 rollout。

本日不以“启动成功”或“loss 下降”验收，而以能否追清四条流验收：

1. **Control flow**：谁启动谁、谁等待谁、同步/异步边界在哪里；
2. **Data flow**：prompt、response、reward、mask、log-prob、advantage 如何变成 learner batch；
3. **Weight/version flow**：training weights 如何变成 rollout weights，下一批如何证明使用新版本；
4. **Evidence/recovery flow**：trajectory、checkpoint、server state 和 replay artifact 哪些可持久化、哪些不能恢复。

## 1. slime 在整个 training stack 中的位置（30–45 分钟）

```text
experiment config / custom data & reward
        |
        v
slime orchestration + Ray resource actors
   | rollout branch                 | learner branch
   v                                v
SGLang engines                  Megatron trainer
   |                                |
responses -> env/reward -> samples -> train batch/update
   ^                                |
   +--------- weight sync/version --+

下层共同依赖：PyTorch distributed / NCCL / CUDA
外部边界：sandbox、tools、storage、eval/observability
```

slime 的核心价值是 online RL 多角色编排、数据循环和权重更新协议，不是重新实现 Transformer forward/backward。Megatron 是 learner 的训练引擎之一，SGLang 是 rollout backend，Ray 负责分布式角色与资源生命周期；这些关系比 CLI 名称更重要。

## 2. 沿 pinned 源码追一轮数据（75–90 分钟）

以 Day 26 `v0.3.1` static audit 的真实路径为锚：

```text
RolloutDataSourceWithBuffer
-> generate_rollout_async
-> Sample / group / reward result
-> RolloutManager._convert_samples_to_train_data
-> RayTrainGroup.async_train
-> RayTrainGroup.update_weights
-> next rollout
```

再补齐 Qwen3.5 model spec、HF↔Megatron conversion 和 loss mask 的边界。每个节点记录：

- 所在 process / Ray actor / GPU role；
- 输入输出 schema 与 batch grain；
- policy / weight version；
- 持久状态与临时状态；
- transport（object store、RPC、NCCL、disk 等，以源码证据为准）；
- backpressure / retry / drop / replay 语义；
- 可观测证据和第一个失败点。

`static_pass` 只能支持结构解释。任何实际 transport、并发或数值行为若 Day 26 未运行，必须标 `RUNTIME UNKNOWN`。

## 3. 四条 flow 分开画（60 分钟）

不要把所有箭头挤进一张“框架架构图”。分别产出：

### Control flow

launcher/config -> Ray roles -> rollout/train phase ordering -> barriers/futures -> shutdown/restart。

### Data flow

prompt -> group responses -> environment result -> reward components -> masks/log-probs/advantages -> learner batch。标出 bad sample、timeout 和 zero-variance group 的分叉。

### Weight/version flow

SFT parent / initial weights -> Megatron shards -> optimizer update -> exported or transferred weights -> rollout engine load -> next batch policy version。区分 full sync、adapter sync、checkpoint 和 server reload。

### Evidence/recovery flow

raw trajectory、reward replay input、train-only batch、training checkpoint、rollout server/KV state、Ray actor state分别能否落盘和重放。说明“server 重启”“job continuation”“exact replay”不是一回事。

## 4. On-policy、并发与 staleness（35–45 分钟）

把 Day 27 原本的 GRPO/on-policy 内容放进系统图，而不是孤立背公式：

- rollout policy、old policy、current learner policy、reference policy 各在哪个节点；
- generation、buffer、training、weight sync 的延迟怎样形成 stale trajectory；
- policy version、sample timestamp/step 和 log-prob identity 至少需要哪些字段；
- importance correction 能处理的分布偏移边界，何时应 drop / re-sample / stop。

画一条同步 timeline 和一条异步 timeline，明确 staleness 首次可见的位置。

## 5. 不做“ms-swift → slime 迁移”，而做职责 crosswalk（30–40 分钟）

比较 Day 25 ms-swift GRPO 与 Day 26 slime 路径时，只比较职责：

| 问题 | ms-swift 路径中是谁负责？ | slime 路径中是谁负责？ | 接口对象是否相同？ |
|---|---|---|---|
| recipe/config 归一化 | | | |
| prompt/group construction | | | |
| rollout backend | | | |
| reward/environment | | | |
| advantage/loss batch | | | |
| learner engine | | | |
| placement/concurrency | | | |
| weight sync/version | | | |
| checkpoint/replay | | | |

同一个框架在不同 mode 下可能委托不同 backend；不要写成静态产品对比表。目标是以后遇到 verl、OpenRLHF 或自研系统时，也能沿相同问题定位组件。

## 6. 用已有 evidence 做系统推理（30–40 分钟）

联合解释三个事实，但不能把它们拼成不存在的 end-to-end success：

- Day 18 证明冻结 envelope 内 Megatron learner 的 Qwen3.5 conversion、TP/DP、checkpoint 与 export 路径可用；
- Day 25 证明 ms-swift 路径完成真实 GRPO update，但没有产生更优 candidate；
- Day 26 只完成 slime static codepath audit，live runtime 在 S0 因 image/topology/dependency identity 失败。

回答：这些证据分别覆盖 integrated graph 的哪些节点？哪些边仍完全未知？为什么“Megatron 通过 + slime static pass”不能推出“slime online RL 闭环通过”？

## 综合口述与故障演练（30 分钟）

不看笔记完成 20 分钟讲解：

1. 一条 prompt 如何成为 Megatron learner batch；
2. 一个 optimizer update 如何回到 SGLang rollout engine；
3. ms-swift、slime、Megatron、Ray、SGLang、PyTorch/NCCL/CUDA 的层级关系；
4. Day 26 S0 failure 应定位在哪一层，为什么不是算法失败；
5. 如果 reward 正常但 next rollout 没变化，按哪条证据链排查。

最后随机选择三个故障：rollout hang、reward 全零、learner OOM、weight version 不变、checkpoint 无法 resume、吞吐升高但 on-policy quality 下降，画出最小排查路径。

## Evidence-first 产物

- `../artifacts/reports/day30-slime-control-data-weight-flows.mmd`
- `../artifacts/reports/day30-slime-node-ledger.md`
- `../artifacts/reports/day30-ms-swift-slime-responsibility-crosswalk.md`
- `../artifacts/reports/day27-30-training-system-architecture.md`
- 20 分钟口述提纲与 `KNOWN / INFERRED / RUNTIME UNKNOWN` 清单

## 最终验收

- [ ] 能从 prompt 追到 rollout、reward、learner batch、Megatron update、weight sync 与 next-version rollout，每条边都有对象和 owner。
- [ ] 能区分 orchestration、rollout、training engine、distributed runtime 与 device compute，不再把它们统称为“训练框架”。
- [ ] 能解释至少三类 state：trajectory state、training state、serving state，以及各自的恢复边界。
- [ ] 能用相同 node ledger 阅读另一个 RL framework，而不依赖记住 slime CLI。
- [ ] Day 26 未验证的 runtime 边全部保留为 `UNKNOWN`；没有为完成课程而租卡或换模型补跑。
- [ ] 输出五个只有未来针对性实验才能回答的问题，每个问题都说明最小实验和所需证据。

## Final Log

### 一句话解释 slime

### Integrated graph 中最关键的反馈边

### 五个后续实验问题

1.
2.
3.
4.
5.
