# Day 29 — 最小可行 slime 闭环、Reward 修改与 Train-only Replay

日期：`2026-08-24`  
状态：`not_started`  
强度：4–5 小时人工工作；GPU wall time 由 Day 26 的最小 recipe 决定

## 主要目标

在 pinned slime `v0.3.0` 上跑通最小真实闭环，并用 debug dump 做 train-only replay；随后只修改一个可单测的 reward component 重跑。Core 是正确性闭环，不是 8×H100 规模复现。

## 理论（30–45 分钟）

精读清单：[Day 29 — slime runtime gates](../SCALING-BOOK-READING-GUIDE.md#day-29)。

- 从 Day 28 gate 表复述 Sample/DataSource/rollout/reward/train/weight version。
- 预写 zero variance、reward hacking、truncation、KL/entropy anomaly、stale weight 和 rollout bottleneck 的 signature。
- 冻结 baseline/modified reward、frozen correctness prompts 和停止条件。

## Coding / Runtime gates（60 分钟）

- 重新验证 reward adapter unit tests 和 pinned resolved config。
- 使用框架已有 trace/debug 能力；若证据不足，只增加轻量、可移除的日志。
- 必须记录 rollout/group IDs、status、reward components、loss mask、policy/weight version、train step 和 sync duration。

## 训练 / 实验（180 分钟启动与分析）

严格按 gate 顺序：

1. **G0 config/import**：两条 prompt 能加载 model、tokenizer、reward 和资源 placement。
2. **G1 rollout-only**：生成最小 batch，保存 raw trajectory/debug dump，离线重算 reward。
3. **G2 train-only replay**：从同一 dump 启动 learner，完成 optimizer step；再次 replay 对比首 step loss/metrics，并说明 exactness。
4. **G3 full loop**：完成 `rollout → reward → train update → weight sync → next rollout`，证明下一批使用新 weight version。
5. **G4 controlled change**：只修改一个 reward component/权重，unit tests 通过后再完成至少一个 update，并比较逐样本行为和 frozen correctness。

每个 gate 失败就停在该层，保留 dump/log；不跳过 replay 直接扩大规模。Core 只要求最小支持的 model/topology 和 1–3 updates，不要求收敛。

## 资源与租卡

- Core：使用 Day 26 在 pinned tag 上验证的**最小受支持 topology**和最小 model/recipe；开卡前已有明确 GPU-hour 上限。
- 如果最小 recipe 是多卡，优先同机拓扑；不得当天临时猜测缩容参数。
- Stretch：8×H100 官方规模 recipe、5–10 updates 和吞吐分析。只有 Core 全部通过且预算明确时才做。
- Reward 无有效 variance、持续 OOM/NaN、trajectory 无法落盘或 weight version 无法验证时停止。

## Evidence-first 产物

- `../artifacts/logs/day29-slime-trajectories/`
- `../artifacts/reports/day29-slime-minimum-loop.md`
- baseline/modified reward diff 与 tests
- train-only replay comparison、weight-version timeline

## 验收

- [ ] G1 raw trajectory/reward 可离线重算。
- [ ] G2 从保存 dump 真正完成 train-only optimizer step，并报告 replay exactness。
- [ ] G3 完成闭环且下一轮 rollout 有新 weight-version 证据。
- [ ] G4 只改变一个 reward component，并同时报告 frozen correctness、length、KL/entropy 和 group variance。
- [ ] 8×H100 若未执行仍可完成 Core；不能把 Stretch 写成毕业条件。

## Optional Capstone Export Contract（不扩大 Day 29 Core）

保留 runtime/container lock、resolved placement、raw token IDs/text、response/tool/environment masks、rollout/reward/train replay pack、policy/weight-version timeline 与 next-version evidence。Day 39 OPD 复用同一 gate 形状，但会增加只读 teacher scoring role；不得把 slime 的 reference/reward model 自动等同于 OPD teacher。

## Daily Log

### Pinned config / GPU topology / hours

### Gate results

### Train-only replay

### Reward 修改前后

### Weight-version timeline

### Day 30 第一动作
