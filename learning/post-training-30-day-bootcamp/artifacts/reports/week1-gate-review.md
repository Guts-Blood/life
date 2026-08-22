# Week 1 Gate Review

周期：`Day 01–07`
日期：`2026-08-02`
Day 完成率：`7/7`
Week 1 Gate：`4/5 passed; 1 partial`
总投入：`unknown`（仓库没有可靠用时记录，不补估算）

## 结论先行

可以进入 Day 08 的 CPU-only 数据契约工作，但不能启动 SFT。Week 1 已有证据足以支撑训练 lifecycle、框架职责、parallelism 配置语义和首轮故障定位；还没有真实 GPU、数据、frozen eval 或训练证据。Day 06 的 stage/scaling 对读和 crosswalk 已补齐；“能区分 SFT、DPO、online RL/RLVR 的信号来源”仍暂记为 `partial`，因为还需要学习者脱离文档独立口述。

决策图：[`week1-training-stage-decision-map.mmd`](./week1-training-stage-decision-map.mmd)

## 本周实际完成

- Day 01–03：完成 Roofline、Transformer accounting、H100 step-time/MFU 的互动 Quiz；真实环境、硬件 SKU、accounting worksheet 和 profiler 对照仍缺。
- Day 04：Quiz Core 完成。已能从 shard 维度选择 collective，使用 `world_size = DP × TP × PP` 和 `B_global = B_micro × accumulation × DP`，并对 OOM、hang、checkpoint mismatch 给出有顺序的首查证据。
- Day 05：完成 training lifecycle Quiz、`swift sft` 边界追踪和最小 causal-LM 状态机。已区分 data/template/tokenizer/collator、masked loss、gradient、optimizer、checkpoint/resume 与 eval 的状态边界。
- Day 06：完成 Tülu 3 × Applied Training 对读，写明三阶段的数据、上游 checkpoint、状态角色、scaling pressure，以及 memory/compute/time-cost 三类 feasibility。
- Day 07：完成 training-stage decision map 与本 gate review；没有启动 GPU。

## 最强证据

- 最可信的实验：`none`。本周只有纸面推导、源码追踪和 CPU 级状态机，尚无真实训练 run。
- 最可信的概念证据：Day 04 Quiz 的 parallel group/global batch 与故障诊断；Day 05 Quiz 的 sample-to-checkpoint lifecycle、continuous resume 和 framework boundary。
- 最可信的 Eval：`none`。Day 10 前没有 frozen manifest、逐样本 Base predictions 或 scorer 证据。
- 最有价值的失败：曾把 tokenizer 输出写成 `[T,D]`、把 gradient 与 optimizer state 混淆、把 communication 后的 compute tail 称为 exposed communication；定向复测后已纠正。这些错误说明后续必须继续用“对象、shape、状态所有者、发生时刻”作答，不能只背术语。

## 我的模型发生了什么变化

Week 1 后，训练不再被看成一个单独的 `train()` 调用，而是一条有状态、有证据边界的链：

```text
sample/version
-> template/tokenizer/loss mask
-> batch and forward
-> backward/accumulation/sync
-> clip/optimizer/scheduler
-> complete checkpoint state
-> frozen per-sample eval
```

DP/FSDP/TP/PP 改变的是数据副本、tensor/layer ownership、状态布局和通信，不会创造新的训练目标。是否进入 SFT、DPO 或 online RL，首先由目标行为和可用训练信号决定；显存、并行、吞吐与成本只决定该阶段能否以及如何执行。

## Week 1 Gate

| Gate | 结果 | 证据 / 缺口 |
|---|---|---|
| 区分 SFT、DPO 与 online RL/RLVR 的信号来源 | `partial` | Day 06 crosswalk 与决策图已写清 demonstration tokens、preference pairs、on-policy verifiable reward 三种信号；仍需独立口述复核。 |
| 解释 training lifecycle 与框架责任边界 | `passed_guided` | Day 05 Quiz 六模块完成，并有 `swift sft` codepath walkthrough。 |
| 用 DP/FSDP/TP/PP 做配置和首轮故障定位 | `passed_quiz` | Day 04 Quiz 五模块完成；真实 rank logs、memory snapshot 和 reshard 尚待后续实践。 |
| 解释为何先 data contract，再 frozen eval，之后才能解释训练效果 | `passed_review` | 若先训练，不知道哪些 token 被监督；若先看结果再定 eval，会引入 protocol/selection drift，无法把变化归因于训练。 |
| 写明 Week 2 模型、数据候选、eval slices、停止条件和未知项 | `passed_plan` | 见下一节；所有未固定的 source/revision/cache/GPU 信息均显式保留为未知。 |

## Week 2 证据顺序与进入门槛

```text
Day 08 data contract
-> Day 09 data quality, mixture, lineage, decontamination
-> Day 10 frozen eval plus Base per-sample baseline
-> Day 11 tiny overfit and resume proof
-> Day 12 controlled equal-token SFT and checkpoint selection
```

任何一步未通过，都不把下一步的 loss 或 aggregate score解释成模型能力变化。

| 项目 | 当前决定 | 未知 / 必须补证据 |
|---|---|---|
| Core model | `Qwen/Qwen3-0.6B-Base` | model/tokenizer revision、cache path、hash、实际磁盘占用未记录。 |
| Optional model | `Qwen/Qwen3-1.7B-Base` 只在 0.6B 链路通过后考虑 | 不属于 Day 08–12 的默认路径；不提前下载或开卡。 |
| Train data | 同一 source pool，候选 slices 为 general/math/code；A 为等 supervised-token 比例，B 提高一个预注册 target slice | 具体 dataset、source URL、license、revision、质量阈值、语言比例均未知；Day 08–09 固定。 |
| Eval | 90–150 条候选，general/math/code，分 dev 与 frozen test；保存逐样本输出 | 样本来源、reference、extractor/scorer version、允许退化阈值均未知；Day 09 去污染、Day 10 冻结。 |
| Tiny overfit | 16–32 条人工审核短样本，默认目标为 assistant token accuracy 至少 95% | 精确 sample IDs、hash、LR、max length、150 steps 内能否达标未知。 |
| SFT stop | 零有效 label、NaN、grad norm 持续为 0、sample 顺序异常、resume 不连续立即停；A/B 配置或 data hash 漂移则比较作废 | dev slice 的成功阈值和允许退化范围必须在 Day 10 预注册。 |
| Resource window | Day 08–09 CPU only；Day 10 候选 1×H100 约 1–2h；Day 11 候选约 2–3h；Day 12 候选约 4–6h | 缓存、磁盘、实际 H100 SKU/拓扑、租卡窗口和预算都未确认；确认前不启动实例。 |

## 三项需要真实证据证伪的假设

1. `H1 — pipeline learnability`：通过 Day 08/09 审计的 16–32 条 tiny set，能在 150 个 optimizer updates 内达到 non-padding assistant-token accuracy `>=95%`，且没有 NaN、零梯度或 resume 断裂。
2. `H2 — useful SFT checkpoint`：至少一个 SFT checkpoint 能改善 Day 10 预注册的 dev slice，同时 general/aggregate 的退化不超过训练前冻结的阈值。
3. `H3 — targeted mixture effect`：在相同 Base、seed、preprocessing、supervised-token budget、optimizer、hardware 和 eval protocol 下，`mix-B-targeted` 相比 `mix-A-balanced` 改善 target slice，且 general slice 退化不超过预注册阈值。

任何一项失败都应保留 per-sample bad cases 和最近的 state boundary，不能用“模型太小”或“数据不够”作无证据解释。

## 下周调整

- 删除：在数据契约和 frozen eval 前启动训练；同时探索多个模型/框架；用 aggregate score 代替逐样本证据。
- 保留：每次只改变一个变量；保存 config/data/model/template hashes；先写成功与停止条件。
- 加强：token-level label audit、supervised-token 口径、train/eval overlap、continuous resume、selected-checkpoint bad cases。

## 风险

- [x] 有为了赶进度跳过 Base baseline 的风险；用 Day 10 硬 gate 阻止。
- [x] 尚无逐样本 Eval；Day 10 前不允许声称模型变好。
- [x] 数据 source/revision 尚未固定；Day 08–09 是当前最大执行阻塞。
- [ ] 本周没有同时启动多个框架或 GPU run。
- [ ] 睡眠与身体状态没有记录，无法判断；下周 daily log 显式记录影响判断的情况。

## 最大阻塞

Week 2 的具体 dataset source/license/revision 与 model/tokenizer/cache/disk 状态尚未固定。Day 08 的第一项证据必须先补数据契约和缓存只读检查，不能跳到训练命令。

## Week 2 第一项证据

一条可人工核验的 SFT sample trace：

```text
raw messages
-> rendered text with pinned template hash
-> input IDs/tokens
-> labels and assistant-only supervised mask
-> shifted targets
```

必须同时记录 model/tokenizer revision、sample ID、max length、EOS/padding/truncation policy，并证明有效 label token 数大于 0。
