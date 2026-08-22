# Day 12 — 受控 SFT Mixture 对照与 Checkpoint 轨迹

日期：`2026-08-07`

状态：`dev_eval_complete_no_eligible_checkpoint / frozen_test_unconsumed`

强度：工作日 4–5 小时人工工作；短训练可在当天稍后继续

## 主要目标

继续使用 `Qwen/Qwen3-0.6B-Base`，只改变 Day 09 冻结的 mixture 比例，运行 A/B 两条等 token-budget SFT；比较 early/mid/final checkpoint，学习如何从训练轨迹和 slice eval 得出有限结论。

## 理论（45 分钟）

精读清单：[Day 12 — Controlled SFT、mixture 与 checkpoints](../SCALING-BOOK-READING-GUIDE.md#day-12)。

- 单变量对照：same base、seed、source pool、preprocessing、supervised-token budget、optimizer 和 eval protocol。
- 训练 loss 不是模型选择指标；dev slice 用于选择 checkpoint，frozen test 只在选择后使用一次。
- early/mid/final 轨迹用于识别学习、平台期、过拟合和能力回退。
- mixture 效果是当前小模型、数据规模与 seed 下的证据，不外推为普遍规律。

## Coding（75 分钟）

- 用一份 base config 加两个只覆盖 dataset manifest 的 A/B 配置，自动 diff 并拒绝未授权差异。
- 每个 run manifest 保存 code/model/tokenizer/data/template/config/environment/seed hash。
- 按累计 supervised tokens 定义 25%/60%/100% 三个 checkpoint，而不是用 epoch 名称含糊比较。
- 统一导出曲线和逐 checkpoint dev predictions；先选 checkpoint，再对每个 run 的 selected checkpoint 跑一次 frozen test。

## 训练 / 实验（150–180 分钟启动与分析）

- A：`mix-A-balanced`；B：`mix-B-targeted`。两者从同一 Base checkpoint 独立启动。
- 固定总 supervised-token budget、effective batch、max length、LR schedule、warmup、dtype、seed 和硬件。
- 训练前写明假设：B 应改善预注册 target slice，同时 aggregate/general slice 的退化不得超过 Day 10 定义阈值。
- 每个 run 保存 early/mid/final；比较 train/dev loss、target/general/math/code slices、输出长度与 bad cases。
- 如果 A/B 之外出现配置差异、数据 hash 漂移或 resume 状态不完整，本次比较作废并记录，不通过事后解释挽救。

## 资源与租卡

- [Tülu 3 paper](https://arxiv.org/abs/2411.15124) 的 SFT mixture 与 evaluation 部分
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 的 post-training 总览
- 推荐 1×H100 80GB 顺序运行，预计总计 4–6 GPU 小时；也可使用同型号的单张 40GB 级 GPU，但 A/B 必须使用同一硬件。
- 所有 predictions、curves、manifests 和 selected checkpoint metadata 同步后关机。

## 产物

- `../artifacts/configs/day12-controlled-sft-A.yaml`
- `../artifacts/configs/day12-controlled-sft-B.yaml`
- `../artifacts/reports/day12-config-diff.md`
- `../artifacts/reports/day12-checkpoint-trajectory.md`
- `../artifacts/reports/day12-mixture-ablation.md`
- `../artifacts/reports/day12-cloud-dev-selection-outcome.json`

## 验收

- [x] 自动 diff 证明 A/B 唯一有意变量是 mixture manifest/比例。
- [x] 两个 run 的累计 supervised tokens、优化器设置和硬件一致。
- [x] A/B 均保存并评估 early/mid/final；同一 dev protocol 判定两条 run 均无 eligible checkpoint。
- [x] 因没有 eligible selected checkpoint，按预注册规则不读取 Frozen test。
- [x] 结论包含支持证据、反例/bad cases 和适用边界；没有把两次小实验写成通用规律。

## Optional Capstone Handoff（不增加 Day 12 Core）

Day 12 额外固化一个通用 `selected-checkpoint-promotion` manifest 模板，至少包含 parent checkpoint、data/config/template hashes、累计 label tokens、dev selection evidence、可推理 export 路径/hash、可继续训练 state 路径/hash与转换 parity。0.6B checkpoint 不能成为 4B/8B 权重起点；可复用的是 SFT recipe/selection/checkpoint contract。Capstone 的 S1 必须用同一个 exact manifest 分叉 direct RL 与 OPD。

## Qwen3.5-4B v2 Handoff

本日正式关闭 `qwen3-0.6b-day01-12-v1`：十轮 recovery C–L 全部完成，0 个候选通过联合 gate，frozen test 未消费。后续不会把 0.6B Base、Run E 或其他 descriptive peak 当作 4B policy parent。

Day 13+ 建立 `qwen35-4b-day13-plus-v2`。原始数据/provenance、scorer 思路和 promotion policy 可以重新审核后复用；旧 tokenizer/template 产出的 rendered text、token IDs、label spans、supervised-token schedule、comparison key 和 aggregate score 均不可沿用。完整迁移 gate 见 [`../QWEN35-4B-MIGRATION-PLAN.md`](../QWEN35-4B-MIGRATION-PLAN.md)。

## Daily Log

### 预注册假设与阈值

### Selected checkpoints

A：无；B：无。两条 run 的全部 checkpoint 均因 math slice 相对 Cloud
Base 退化超过 2 个 correct cases 而不合格。A-25/B-25 只记录为 descriptive
code peak，不晋级 frozen confirmation。

### 最可信和最不可信的结论

最可信：两条 SFT recipe 都显著改善 code，并把 Base 的普遍 max-token
ceiling 行为降下来；但同时严重损害 math，且 code 都在 25% 后回退。

最不可信：B mixture 普遍优于 A。三个 matched-budget code 净胜为
`+2/-1/+2`，没有一次达到预注册的 +3 threshold，且两边都未通过 guardrail。

### Day 13 Reading 问题
