# Day 07 — 周末 Review：Training Stage Decision Map

日期：`2026-08-02`

状态：`done`（Week 1 Gate `4/5 passed; 1 partial`；stage-signal 独立口述复核待完成）

强度：1 小时，仅阅读/复盘

## 主要目标

把 Week 1 的 accounting、roofline、parallelism 和 lifecycle 压缩成一张“该不该进入下一训练阶段”的决策图，为 Week 2 的数据与 SFT 实验设门槛。

## 理论 / 复盘（60 分钟）

精读清单：[Day 07 — Training stage decision map](../SCALING-BOOK-READING-GUIDE.md#day-07)。

- 15 分钟：复查 Day 01–05 的核心证据，只保留能改变训练决策的结论。
- 30 分钟：画阶段图
  `Base -> SFT -> preference/DPO -> online RL/RLVR -> final eval`，每个节点写：
  - 目标行为与适用条件；
  - 所需 data contract 和上游 checkpoint；
  - objective 与输出 artifact；
  - 进入 gate、停止条件、主要 failure signal。
- 10 分钟：把 sharding、显存和成本作为每个阶段的执行约束挂到图上，不把它们误写成新的训练阶段。
- 5 分钟：写 Week 2 的证据顺序：`data contract -> data quality/lineage -> frozen eval -> tiny overfit -> controlled SFT`。

## Coding

无。

## 训练 / 实验

无；不要为“提前准备”启动 GPU。

## 资源与租卡

CPU only。只确认 Day 10–12 所需模型缓存、磁盘和 GPU 窗口，不启动实例。

## 产物

- [`../artifacts/reports/week1-training-stage-decision-map.mmd`](../artifacts/reports/week1-training-stage-decision-map.mmd)
- [`../artifacts/reports/week1-gate-review.md`](../artifacts/reports/week1-gate-review.md)

## Week 1 Gate

- [ ] 能区分 SFT、DPO 与 online RL/RLVR 改变模型的信号来源。（Day 06 crosswalk 与决策图已写，仍需独立口述复核。）
- [x] 能解释训练 lifecycle 与框架责任边界。（Day 05 guided quiz evidence。）
- [x] 能把 DP/FSDP/TP/PP 用于配置和故障定位，不要求自己实现 collective。（Day 04 quiz evidence；runtime evidence 后补。）
- [x] 能说明为什么 Day 08 必须先固定数据契约、Day 10 必须先冻结 eval，之后才能解释训练效果。
- [x] Week 2 的模型、数据候选、eval slices 和停止条件已写明；未知项被显式记录。

### 最大阻塞

Week 2 的具体 dataset source/license/revision，以及 model/tokenizer cache、磁盘和 GPU 窗口尚未固定。

### Week 2 第一项证据

完成一条 `raw messages -> rendered text -> input IDs/tokens -> labels -> supervised mask -> shifted targets` 的人工可核验 trace，并固定 tokenizer revision 与 template hash。
