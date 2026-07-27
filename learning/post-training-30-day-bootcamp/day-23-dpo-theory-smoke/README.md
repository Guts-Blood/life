# Day 23 — DPO 推导与小模型真实 Smoke

日期：`2026-08-18`
状态：`not_started`
强度：4–5 小时

## 主要目标

从 Bradley–Terry/RLHF 假设推到 DPO objective，并用 Day 22 冻结的 preference data 在小模型上完成一次真实 forward、backward、checkpoint 和 held-out 对比。

## 理论（90 分钟）

精读清单：[Day 23 — DPO objective](../SCALING-BOOK-READING-GUIDE.md#day-23)。

- Policy/reference、chosen/rejected、implicit reward 与 β。
- 手推 `log πθ - log πref` 在 chosen/rejected 两边的作用。
- Sequence log-prob 的 token mask、sum/average 口径和 length bias。
- DPO 不需要在线 rollout，但仍会过拟合偏好数据或偏好 artifact。

必须在看高层 Trainer 前，用四个标量 log-prob 手算一遍 loss 和梯度方向。

## Coding（75 分钟）

- 写最小 DPO loss 单元测试：交换 chosen/rejected、改变 reference、改变 β 时方向符合预期。
- 在真实 tokenizer 输出上核对 prompt token 不参与 response log-prob，chosen/rejected template 完全一致。
- 固定 ms-swift commit/model revision/完整 config，并保存 reference policy 的来源和冻结方式。
- 为 train/dev/held-out 输出逐 pair margin、length bucket、source slice。

## 训练 / 实验（120–150 分钟）

- Model：使用 Day 12 已通过 frozen eval/生成检查的最小 SFT checkpoint，优先 Qwen3-0.6B 阶梯；Base 只能用于 loss/encode 对照，不能冒充 DPO 的合格策略起点。
- Data：只使用 Day 22 frozen train，dev 选 checkpoint，held-out 只做最后确认。
- 先跑 5-step overfit gate，确认 chosen margin 朝正确方向变化；再跑 20–50 optimizer steps。
- 保存 early/final checkpoint；按 Day 21 policy 比较 dev pair accuracy/margin、held-out、length-matched 和 source-held-out slices。
- 记录 train loss、chosen/rejected rewards/margins、KL proxy、response lengths 和逐 pair 结果。Smoke 的目标是验证机制，不宣称能力提升。

## 资源与租卡

- 1×H100 80GB，预计 2–4 小时；小模型也可用 A100/L40S。
- 5-step gate 未通过时不扩大 steps。
- checkpoint、config、逐 pair predictions 和 manifest 同步后关机。

## Evidence-first 产物

- `../artifacts/scripts/test_dpo_loss.py`
- `../artifacts/configs/day23-dpo/`
- `../artifacts/reports/day23-dpo-smoke.md`

## 验收

- [ ] 能不看代码推导并解释 DPO objective。
- [ ] 手写 loss tests 覆盖 chosen/rejected、reference 和 β。
- [ ] 真实训练完成 optimizer step、保存并重载 checkpoint。
- [ ] held-out 与 length/source slices 在训练前冻结，结果保存到 pair 粒度。
- [ ] 结论明确区分“训练机制正确”和“模型能力提升”。

## Daily Log

### 手推公式

### Training gate

### Held-out / bias slices

### Day 24 第一动作
