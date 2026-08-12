# Day 17 — Qwen3.5 LoRA Exact Checkpoint Resume 与可复现性

日期：`2026-08-12`
状态：`blocked_no_selected_resumable_day16_candidate`
强度：4–5 小时

审计结论（`2026-08-10`）：**不能关闭为 pass，也不应现在租卡。** Day 11 是旧 0.6B/full-SFT 的 exact-resume 方法证据；Day 18 是 Qwen3.5/Megatron 的 fresh continuation，但没有 uninterrupted comparator；Day 20 是目标 HF LoRA 路径，却只有三份 `resumable=false` 的 probe checkpoint。详见 [`Day 17 Gap Audit`](../artifacts/reports/day17-gap-audit.md)、[`Resume Equivalence Status`](../artifacts/reports/day17-qwen35-resume-equivalence.md) 与机器可读 [`Readiness Gate`](../artifacts/configs/day17-qwen35-resume-readiness.json)。

## 当前阻塞

Day 16 已以 [`no eligible candidate`](../artifacts/reports/day16-gap-audit.md) 收束。Day 20 保存的三份 probe checkpoint 都明确是 `resumable=false`，实际目录没有 optimizer、scheduler 或 RNG state；也没有 selected config、main checkpoint 或 provisional anchor，因此当前不能按本日的 anchor continuity 协议启动 Run A/B。

原计划保留。只有新 SFT charter 产生 selected、完整 resumable 的 LoRA checkpoint，或用户另行批准一个明确标注为 method-only、不得晋级的 resume run，本日才可解锁。

## 主要目标

证明 Day 16 的 Qwen3.5 LoRA coding SFT checkpoint 不只是“能加载 adapter”，而是能恢复 Base/adapter 关系、optimizer、scheduler、RNG 和 dataloader cursor，使中断续训与不中断训练经过相同样本和更新。Day 01–12 的 resume 结果继续作为历史方法证据，但旧 state/checkpoint 不进入本日对照。

## 理论（60 分钟）

精读清单：[Day 17 — Checkpoint/resume state](../SCALING-BOOK-READING-GUIDE.md#day-17)。

- Frozen Base revision、LoRA adapter、optimizer moments、scheduler step 和 merged inference export 的职责边界。
- Python、NumPy、Torch CPU/CUDA RNG、dropout、sampler epoch、dataset cursor、worker seed 与 accumulation boundary。
- Qwen3.5 processor/template、ViT/aligner freeze policy、resolved loader/GatedDeltaNet backend 为什么也是可复现状态。
- “加载成功”“生成可用”“逐 step 数值一致”“bitwise exact”是不同等级。

先列 state inventory，并写明缺失每一项会从哪个阶段分叉。

## Coding（90 分钟）

- 扩展 checkpoint audit：Base repo/revision/hash、adapter tensors/config、trainable parameter names、optimizer/scheduler/scaler、RNG、sampler/dataloader、processor/template、runtime/kernel versions 与 freeze policy。
- 为每个 step 保存 sample IDs、render hash、有效 label tokens、LR、loss、grad norm、adapter checksum 和累计 supervised tokens。
- 比较两条 run 的逐 step timeline，以及若干 adapter/optimizer tensor 的 checksum/`max_abs_diff`。
- 实现真正退出进程再恢复的脚本；禁止在同一 Python 进程中伪装 resume，也禁止用 merged inference export 冒充 resumable state。

## 训练 / 实验（120–150 分钟）

使用 Day 16 冻结的 Base、LoRA target、数据顺序、packing setting、optimizer 与 runtime：

- Run A：从 step 0 连续训练到 step 40。
- Run B1：从相同 Base 和 adapter 初始化训练到 step 20，保存完整 resumable checkpoint，正常退出。
- Run B2：新进程从 B1 恢复到 step 40。
- 比较 step 21–40 的 sample/render IDs、LR、loss、grad norm、adapter 权重、optimizer state 与 final merged-export generation。
- Failure probe：复制 checkpoint 后有意遗漏 sampler cursor 或 scheduler state，只跑 3–5 steps，证明 audit 能定位分叉；不覆盖正确 checkpoint。

如果当前 kernel/framework 无法 bitwise deterministic，必须固定硬件和 deterministic 设置，报告首个分叉 step、误差量级与具体 nondeterministic source；不能把“曲线相似”称为 exact resume。

## 资源与租卡

- 使用 Day 16 的同一 GPU topology、镜像、worker 数和 kernel backend；预计 3–5 小时窗口。
- 正确 checkpoint、故障注入副本和 merged inference export 分目录保存。
- 当前 GPU 预算为 `0`。新 selected/resumable candidate 出现后，复用 candidate 的同一单卡 topology；若 candidate 使用单卡 H800 80GB，Day 17 也使用单卡 H800，无需额外引入 TP。
- 若用户另批 method-only 短 run，只能验证工具链，不能关闭 candidate continuity gate，也不得把结果晋级为下游策略起点。

## Evidence-first 产物

- `../artifacts/scripts/checkpoint_audit_qwen35.py`
- `../artifacts/reports/day17-qwen35-resume-equivalence.md`
- Run A/B configs、sample/render timeline、state manifests、adapter/optimizer tensor diff 与 export parity

当前仅生成 fail-closed 状态报告和 readiness contract；runtime auditor 与 Run A/B evidence 仍须等真实 candidate 解锁，不能用静态文件冒充实验完成。

## 验收

- [x] 所需 Base、adapter、optimizer、scheduler、RNG、dataloader、processor/template 与 freeze policy 字段已进入 readiness inventory。
- [ ] 真实 selected candidate 的上述状态全部落盘并通过 integrity audit。
- [ ] 确实退出并启动新进程。
- [ ] 给出 exactness 等级和逐 step 证据，不只给最终 loss。
- [ ] 故障注入在预期位置造成并定位分叉。
- [ ] resumable state 与 merged inference export 没有混淆。

## Daily Log

### Qwen3.5 state inventory

字段合同已冻结在 [`day17-qwen35-resume-readiness.json`](../artifacts/configs/day17-qwen35-resume-readiness.json)；尚无 eligible checkpoint 可实例化。

### Run A vs Run B

`not_run_blocked`。Day 20 main 未启动，三份 probe checkpoint 均不可恢复。

### 首个分叉与原因

未运行，不能报告数值分叉。当前流程分叉发生在 entry gate：缺少 selected/resumable candidate。

### Resume/export boundary

Day 18 已证明 full-state checkpoint 与 model-only export 的角色边界；目标 HF LoRA candidate 尚无双路径 runtime evidence。

### Day 18 第一动作

此标题是原课程顺序的历史占位。Day 18 standalone 已提前完成；当前第一动作改为批准新 SFT charter，而不是自动进入下一次 GPU run。
