# Day 23 — Qwen3.5 Coding DPO：Objective、Reference 与真实 Smoke

日期：`2026-08-18`
状态：`closed_no_candidate`（terminal detail：`terminal_dev_not_qualified`）；CPU 与 GPU gates 于 `2026-08-14` 提前执行，最终 qualification dev NO-GO
强度：4–5 小时

## 主要目标

从 Bradley–Terry/RLHF 假设推到 DPO objective，并只从 Day 21 **promoted Qwen3.5 SFT anchor** 启动 coding DPO smoke。`Qwen/Qwen3.5-4B-Base` 和任何 Day 01–12 checkpoint 都不能冒充合格的 DPO policy/reference 起点。

## GPU Optimizer Hard Prerequisites

- Day 21 promotion manifest 唯一、hash 完整，resumable checkpoint 与 inference export parity 已通过。
- Day 22 `day22-qwen35-experimental-ai-assisted-manifest.json` 必须由独立 validator 验证为 `completed_experimental_ai_assisted`；它绑定的 train/dev/held-out、processor/template 和 sandbox evidence 已冻结。该实验路径不宣称 formal human-reviewed readiness。
- pinned ms-swift runtime 已验证如何加载 SFT adapter/merged path、构造 frozen reference，并执行 LoRA DPO；CLI 以该 checkout 的文档和 `--help` 为准。
- 资源 preflight 已把 policy、reference、optimizer、activations 和 temporary logits 纳入显存测量。

这些 prerequisite 是启动真实 optimizer 的硬门槛，不阻塞理论、数据编译、processor/template、loss 与 CLI 的 CPU preflight。任一 GPU prerequisite 不满足就停止，不回退 0.6B，也不从 Base 冷启动。

## CPU Preflight（2026-08-14）

CPU 主链已经完成，独立 validator 状态为 `valid_cpu_ready_gpu_pending`：

- Day 21/22 trust roots、self-hash 与 cross-binding 通过；Day 22 experimental 200 pairs，split `154/17/29`，formal-human 仍未宣称 ready。
- 新增独立 RLHF-only alias `day23_qwen3_5_dpo_target_v1`，没有修改 Day 20 training-only 历史 alias。
- pinned processor 对 200 pairs / 400 branches 重编码，400/400 与 Day 22 frozen token-label evidence exact match；max input 381，`max_length=512` 下零截断。
- DPO scalar/gradient/mask oracle、torch 与 pinned ms-swift `DPOTrainer` 交叉验证通过。
- 5-step 与 30-step 配置均通过真实 JSON CLI parse。5-step 固定 4 对、不加载 dev、dataset/dataloader 均不 shuffle；两个 run 都必须 fresh start。
- 27 个 Day 23 tests、11 个 loss tests、272 个 Day 20–22 regression tests，共 `310 PASS`。

完整 CPU 结果与开卡 pipeline 见 [`day23-qwen35-coding-dpo-cpu-preflight.md`](../artifacts/reports/day23-qwen35-coding-dpo-cpu-preflight.md)。这些 remaining gates 随后均已在目标 GPU runtime 中执行；最终结果见下节。

## GPU Closeout（2026-08-14）

本节以下内容记录最初 Day 23 mechanism 协议的历史 closeout。之后另建的 append-only、隔离 RSI lineage 从 promoted S1 fresh start 继续搜索，并最终训练出 qualification checkpoint；最终事实以 [qualification closeout](../artifacts/reports/day23-qwen35-dpo-qualification.md) 为准。

G0 remote S1 payload、双卡 DDP、fresh LoRA / frozen reference、一步 optimizer、显存、checkpoint save 与独立新进程 reload 全部通过。训练使用 2×RTX PRO 6000；G2 的最低 observed free fraction 为 `87.06%`，远高于 `15%` 硬门。

5-step mechanism gate 最终为 terminal NO-GO：

- Attempt 1（双卡 B2 / GA1）：mean margin improvement `+0.00168481`，但仅 `1/4` pair 改善。
- 唯一授权的 topology correction attempt 2（双卡 B1 / GA4）：mean `+0.00541587`，仍仅 `1/4` pair 改善。
- 两次都是 task602/604/605 退化、只有 task610 改善；训练、冻结、reference、checkpoint 与显存证据均正常，因此是科学机制门失败，不是 infrastructure failure。

Attempt 2 的 one-shot claim 已消费，协议明确禁止第三次尝试。30-step smoke、dev selection、preference heldout 与 post-selection guardrails 均未运行；两个 failure checkpoint 都被隔离，不能作为 candidate/resume。完整逐 pair 数值、远端路径、self/file hashes 与 claim boundary 见 [`day23-qwen35-coding-dpo-smoke.md`](../artifacts/reports/day23-qwen35-coding-dpo-smoke.md)。

## Isolated RSI continuation 与最终 qualification（2026-08-14）

早期 mechanism 协议关闭后，没有覆写其失败或复用失败 checkpoint。后续采用独立 lineage：Day 23 `rsi-v0003`–`rsi-v0005` → `goal-0003/csearch-v0001` → `goal-0004/qual-v0001`。`csearch-v0001` 选出 `4.3e-6@step20`，随后从 promoted S1 fresh start，用 full train154、双卡 B8/GA2 完成 30-step refit，并保存有效 checkpoint-20。

一次性 dev17 的最终结果是 `8/17` positive pairs、mean margin `-0.00071079`、length-matched mean `-0.00225322`，没有达到 `>=13/17` 且两个 mean `>0` 的冻结门。因此 checkpoint 存在但不合格；full112、E2B 与 heldout 均未运行，实例已关机。完整 checkpoint 路径、hash 和 gate 证据见 [`day23-qwen35-dpo-qualification.md`](../artifacts/reports/day23-qwen35-dpo-qualification.md)。

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

Git 只归档代码骨架与学习报告。运行时 JSON/JSONL 数据、eval 明细、配置、charter state 与 checkpoint 本体留在本地或外部 artifact store；报告只记录其路径、hash、冻结门和结论。

- `../artifacts/scripts/test_day23_qwen35_dpo_loss.py`
- `../artifacts/scripts/validate_day23_qwen35_dpo.py`
- `../artifacts/data/day23-qwen35-coding-dpo-data-manifest.json`
- `../artifacts/data/day23-qwen35-dpo-next-blind-manifest.json`
- `../artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json`
- `../artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json`
- `../artifacts/configs/day23-qwen35-coding-dpo/`
- `../artifacts/reports/day23-qwen35-coding-dpo-cpu-preflight.md`
- `../artifacts/reports/day23-qwen35-coding-dpo-smoke.md`
- `../artifacts/reports/day23-qwen35-dpo-qualification.md`

## 验收

- [ ] 能不看代码推导并解释 DPO objective、mask 与 reference。
- [x] policy/reference 都由 Day 21 promoted SFT anchor 派生，Base/v1 权重未参与。
- [x] Reference immutability、LoRA/vision freeze 与 trainable inventory 有 runtime evidence。
- [ ] 合格 candidate 的 optimizer、checkpoint、dev、guardrail 与 held-out evidence 完整；当前 optimizer/checkpoint/dev 已有证据，但 dev gate 失败。
- [x] 结论区分“机制正确”“preference metric 变化”和“真实 coding correctness”。

## Daily Log

### Anchor / reference / policy mapping

Policy/reference 均从 promoted merged S1 派生；policy 使用 fresh LoRA，reference 通过禁用 fresh adapter 回到同一 merged S1。GPU runtime 已验证 496 个 LoRA tensors 可训练、723 个 parent tensors 版本不变，Base/v1 权重未参与。

### 手推公式与 loss tests

sigmoid DPO、implicit rewards、β、chosen/rejected swap、reference margin、response-only sum mask 与解析/数值梯度测试通过；已与 torch 和 pinned `DPOTrainer.dpo_loss` 交叉验证。

### Training gate

`CLOSED_NO_CANDIDATE`。原 mechanism 的 G0–G2 通过；G3 两个 5-step 尝试均只有 `1/4` pair 改善。Attempt 2 为 one-shot，terminal failure 后原协议的 G4/G5 不启动。后续隔离 lineage 从 promoted S1 fresh start 完成 30-step qualification refit，但没有通过一次性 dev gate。

### Dev / held-out / sandbox slices

原 mechanism 没有产生 candidate，因此没有读取 dev。后续隔离 qualification 消费了一次性 dev17，结果为 `8/17` positive、overall/length-matched mean 均为负；full112、E2B 与 heldout 未运行或消费。

### Day 24 第一动作

Day 23 已以合法零候选结果关闭。Day 24/25 如继续，必须从 promoted S1 独立启动；不得使用或 resume 任一 Day 23 failure checkpoint。
