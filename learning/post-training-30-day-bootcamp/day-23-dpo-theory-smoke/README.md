# Day 23 — Qwen3.5 Coding DPO：Objective、Reference 与真实 Smoke

日期：`2026-08-18`
状态：`not_started`
强度：4–5 小时

## 主要目标

从 Bradley–Terry/RLHF 假设推到 DPO objective，并只从 Day 21 **promoted Qwen3.5 SFT anchor** 启动 coding DPO smoke。`Qwen/Qwen3.5-4B-Base` 和任何 Day 01–12 checkpoint 都不能冒充合格的 DPO policy/reference 起点。

## Hard Prerequisites

- Day 21 promotion manifest 唯一、hash 完整，resumable checkpoint 与 inference export parity 已通过。
- Day 22 `day22-qwen35-experimental-ai-assisted-manifest.json` 必须由独立 validator 验证为 `completed_experimental_ai_assisted`；它绑定的 train/dev/held-out、processor/template 和 sandbox evidence 已冻结。该实验路径不宣称 formal human-reviewed readiness。
- pinned ms-swift runtime 已验证如何加载 SFT adapter/merged path、构造 frozen reference，并执行 LoRA DPO；CLI 以该 checkout 的文档和 `--help` 为准。
- 资源 preflight 已把 policy、reference、optimizer、activations 和 temporary logits 纳入显存测量。

任一 prerequisite 不满足，状态写 `blocked_missing_qwen35_sft_anchor_or_contract`，不回退 0.6B，也不从 Base 冷启动。

## 理论（75 分钟）

精读清单：[Day 23 — DPO objective](../SCALING-BOOK-READING-GUIDE.md#day-23)。

- Policy/reference、chosen/rejected、implicit reward 与 β。
- 手推 `log πθ - log πref` 在 chosen/rejected 两边的作用。
- Sequence log-prob 的 response mask、sum/average 口径与 length bias。
- DPO 不需要在线 rollout，但仍会过拟合 preference/test artifacts。

在看 Trainer 前，用四个标量 log-prob 手算 loss 与梯度方向。

## Coding（75 分钟）

- 写最小 DPO loss tests：交换 chosen/rejected、改变 reference/β/mask 时方向符合预期。
- 在 Qwen3.5 processor 输出上证明 prompt tokens 不参与 response log-prob，chosen/rejected 的 prompt/template 完全一致。
- 从 Day 21 manifest 解析 policy parent 与 frozen reference 的逻辑身份；无论 runtime 用独立模型、adapter disable 或 merged weights 实现，都保存 resolved mapping 和 hashes。
- 记录 LoRA target/trainable params、ViT/aligner freeze、policy/reference immutability 与完整 config diff。
- 为 train/dev/held-out 输出逐 pair margin、length/source/test-family/status slices。

## 训练 / 实验（120–150 分钟）

- Parent/reference：Day 21 promoted SFT anchor；reference 始终冻结。
- Policy：从同一 anchor 初始化新的 DPO trainable state；不得在 reference 上原地更新。
- Data：只使用 Day 22 experimental close manifest 绑定的 frozen train；dev 选择 checkpoint；preference held-out 只在选定后确认一次。
- 先跑 5-step overfit/mechanism gate，确认 chosen margin 方向；再运行 20–50 optimizer steps，保存 early/final。
- 比较 dev pair accuracy/margin、length-matched/source/test-family slices、sandbox correctness、response length 和 general/math/format guardrails。
- 记录 train loss、chosen/rejected log-prob/reward/margin、KL proxy、grad norm、memory 和逐 pair 结果。Smoke 只验证机制和局部行为，不宣称通用 coding 提升。

## 资源与租卡

- 以 preflight 为准；1×H100 80GB 只作为 LoRA DPO planning 上限，不是保证或最低声明。若 frozen reference + policy 超过余量，先停止并重算 sharding/topology。
- 不在正式 run 中临时加入 quantization、offload、ZeRO 或更短序列来掩盖 preflight 失败。
- checkpoint、reference/policy mapping、configs、逐 pair predictions 和 manifests 同步后关机。

## Evidence-first 产物

- `../artifacts/scripts/test_day23_qwen35_dpo_loss.py`
- `../artifacts/configs/day23-qwen35-coding-dpo/`
- `../artifacts/reports/day23-qwen35-coding-dpo-smoke.md`

## 验收

- [ ] 能不看代码推导并解释 DPO objective、mask 与 reference。
- [ ] policy/reference 都由 Day 21 promoted SFT anchor 派生，Base/v1 权重未参与。
- [ ] Reference immutability、LoRA/vision freeze 与 trainable inventory 有 runtime evidence。
- [ ] 真实 optimizer step、checkpoint save/reload 和逐 pair dev/held-out evidence 完整。
- [ ] 结论区分“机制正确”“preference metric 变化”和“真实 coding correctness”。

## Daily Log

### Anchor / reference / policy mapping

### 手推公式与 loss tests

### Training gate

### Dev / held-out / sandbox slices

### Day 24 第一动作
