# Day 17 Gap Audit — Qwen3.5 LoRA Exact Resume

审计日期：`2026-08-10`

结论：`blocked_no_selected_resumable_candidate`

## 一句话判断

Day 17 原本不是再做一轮普通 SFT，而是对 Day 16 选出的 Qwen3.5 LoRA candidate 做一次**训练状态连续性证明**：比较 `0→40` 不间断轨迹与 `0→20→退出进程→20→40` 恢复轨迹，并证明 sample、LR、loss、adapter、optimizer、RNG 与数据游标在 step 21–40 对齐。

这个目标尚未完成，也不能由已有的“能保存 adapter”“能重新加载生成”或“另一训练栈能 continuation”替代。当前正确处置是保持 blocked，追加 GPU 预算为 `0`，等新的 SFT charter 产生 selected、完整 resumable 的 LoRA candidate 后再执行。

## 原计划到底要证明什么

```text
同一冻结训练合同
├── Run A：step 0 → 40，不中断
└── Run B
    ├── B1：step 0 → 20，保存 full resumable checkpoint，退出进程
    └── B2：新进程加载 checkpoint，step 20 → 40

比较 step 21–40
├── sample IDs / render hash / label tokens
├── LR / loss / grad norm / cumulative supervised tokens
├── LoRA adapter / optimizer / scheduler checksums
├── Python / NumPy / Torch CPU / CUDA RNG
├── sampler epoch / dataloader cursor / accumulation boundary
└── final adapter 与 merged inference export 的生成 parity

故障注入
└── 删除 sampler cursor 或 scheduler state，再跑 3–5 steps，确认 auditor 能定位首个分叉
```

验收层级必须分开：

1. checkpoint 可以加载；
2. adapter/merged export 可以生成；
3. fresh process 可以继续更新；
4. 中断与不中断的逐 step 轨迹数值一致；
5. bitwise exact。

Day 17 的核心是第 4 层；第 5 层若受 kernel nondeterminism 限制，必须报告首个分叉 step、`max_abs_diff` 与已定位来源，不能用“曲线相似”代替。

## 现有证据交叉检查

| 证据 | 实际覆盖 | 能否关闭 Day 17 | 边界 |
|---|---|---:|---|
| Day 11 `Qwen3-0.6B-Base` full SFT | 6-step uninterrupted 对 3+fresh-process-resume+3；model/Adam/scheduler/RNG/cursor exact compare 通过 | 否 | 旧 0.6B lineage、full SFT、不同 processor/runtime；只能证明方法做过 |
| Day 18 Qwen3.5 Megatron C4 | 新进程加载 model/optimizer/RNG，从 step 3 继续到 step 5，随后 export/reload | 否 | full-parameter Megatron TP2；没有 uninterrupted 5-step comparator，报告明确不声称 exact resume |
| Day 20 Qwen3.5 HF LoRA probes | 三档各完成 103 steps / 16k supervised tokens，并保存 adapter checkpoint | 否 | 三份 checkpoint 的完整性记录均为 `resumable=false`；实际没有 optimizer、scheduler、RNG state |
| Day 20 resume 代码与 13 项单测 | 会拒绝 partial/tampered checkpoint，并只为完整 main checkpoint staging resume evidence | 否 | 这是静态实现 readiness；main 未启动，resume 路径没有真实运行证据 |

2026-08-10 本地复核：`python3 -m unittest -v test_day20_runner.py` 为 `13/13 pass`；三份 probe integrity JSON 均满足 `status=complete`、`global_step=103`、`resumable=false`，且 checkpoint 目录均不存在 `optimizer.pt`、`scheduler.pt` 或 `rng_state*.pth`。

## Gate 判定

| Gate | 状态 | 当前证据/缺口 |
|---|---|---|
| G0 selected Qwen3.5 LoRA candidate | fail | Day 16 合法结果为 no eligible candidate；S1 不存在 |
| G1 complete resumable checkpoint | fail | Day 20 只有 non-resumable probe adapter checkpoint |
| G2 state inventory schema | partial | 原计划字段和 Day 20 integrity 规则已列明；尚无真实 candidate manifest |
| G3 fresh-process continuation | partial | Day 11 与 Day 18 各自证明；目标 HF LoRA lineage 未运行 |
| G4 uninterrupted vs resumed step parity | missing | 没有目标 lineage 的 Run A/B timeline、tensor diff 或首个分叉记录 |
| G5 failure injection | missing | Day 20 单测覆盖文件缺失/篡改，但没有目标训练轨迹上的 scheduler/cursor divergence probe |
| G6 resumable/export boundary | partial | 文档和 Day 18 C5 已区分 model-only export；目标 LoRA candidate 无双路径实证 |

## 查漏补缺处置

本次补齐：

- 本 gap audit，明确原计划、证据等级和不能外推的边界；
- [`day17-qwen35-resume-equivalence.md`](day17-qwen35-resume-equivalence.md)，以 `not_run_blocked` 记录正式结果槽，不伪造数值；
- [`day17-qwen35-resume-readiness.json`](../configs/day17-qwen35-resume-readiness.json)，机器可读 entry gate、state inventory 和未来 Run A/B 合同；
- 课程 README、PROGRESS、迁移合同和资源计划的状态交叉链接。

本次不补造：

- `artifacts/scripts/checkpoint_audit_qwen35.py`；
- Run A/B config、timeline、state manifest、adapter/optimizer tensor diff；
- failure-injection runtime evidence；
- exact-resume pass、candidate、S1 或 promotion 结论。

原因是 auditor 必须绑定新的训练框架、checkpoint schema、dataset cursor 和实际 selected candidate。现在写一个无法用真实 checkpoint 验证的脚本，只会增加“代码存在但证据不存在”的歧义。

## 解锁条件与下一次执行

只有同时满足以下条件才为 Day 17 租卡：

1. 新 SFT charter 已批准并冻结 data/eval comparison key、LR、LoRA targets、packing、batch、seed 和 runtime；
2. 有合法 selected Qwen3.5 LoRA candidate；
3. checkpoint 实际包含 adapter、optimizer、scheduler、RNG、trainer/global step 与可验证的数据游标；
4. Run A 与 Run B 能从同一初始化、同一数据顺序重新开始，而不是拿 probe 尾部临时续跑；
5. 保存 checkpoint、故障注入副本、merged export 和逐 step evidence 的磁盘预算已冻结。

解锁后复用 candidate 的**同一单卡 topology**即可；4B LoRA continuity 不需要为了模型切分额外引入 TP。若新 candidate 冻结在单卡 H800 80GB，就用同一张 H800 做 Run A/B，避免把多卡调度与通信 nondeterminism 引入 exactness 对照。计划窗口仍为 3–5 小时，但必须先以 5-step smoke 验证 state inventory 和 deterministic setting。

## 最终决定

- Day 17：保持 `blocked_no_selected_resumable_candidate`，不关闭为 pass。
- 当前 GPU：`0`，不租卡。
- 已完成的是方法复核、证据分级和 entry contract；未完成的是 Qwen3.5 HF LoRA 的真实 exact-resume 实验。
- 下一主线动作不是 Day 17 补跑，而是先批准新的 SFT charter 并产生 selected/resumable candidate。
