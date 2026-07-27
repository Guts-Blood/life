# Day 12 — 受控 SFT Mixture 对照与 Checkpoint 轨迹

日期：`2026-08-07`

状态：`not_started`

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

## 验收

- [ ] 自动 diff 证明 A/B 唯一有意变量是 mixture manifest/比例。
- [ ] 两个 run 的累计 supervised tokens、优化器设置和硬件一致。
- [ ] A/B 均保存 early/mid/final，并用同一 dev protocol 选出 checkpoint。
- [ ] Frozen test 只评 selected checkpoints，保存逐样本结果和 slice 变化。
- [ ] 结论包含支持证据、反例/bad cases 和适用边界；没有把两次小实验写成通用规律。

## Daily Log

### 预注册假设与阈值

### Selected checkpoints

### 最可信和最不可信的结论

### Day 13 Reading 问题
