# Day 17 — Qwen3.5 LoRA Resume Equivalence Status

审计日期：`2026-08-10`

状态：`not_run_blocked`

Exact resume proven：`false`

## Verdict

没有可报告的 Qwen3.5 HF LoRA Run A/B 等价性结果。Day 16 没有 selected candidate，Day 20 三份 LoRA probe checkpoint 都是 `resumable=false`，因此原定 `0→40` 对 `0→20→fresh-process-resume→40` 实验不能合法启动。

本文件是正式结果槽的 fail-closed 状态记录，不是 pass 报告。完整理由与证据交叉检查见 [`day17-gap-audit.md`](day17-gap-audit.md)。

## 当前 exactness 等级

| 能力 | 当前结论 |
|---|---|
| Qwen3.5 LoRA adapter 可保存/加载 | supported by Day 20 probes |
| 目标 checkpoint 包含完整训练状态 | false |
| 目标 HF LoRA fresh-process continuation | not run |
| uninterrupted 与 resumed step parity | not run |
| bitwise exact | not established |
| failure-injection divergence localization | not run on target lineage |
| merged export parity | not run for a selected candidate |

## 缺失产物

- `artifacts/scripts/checkpoint_audit_qwen35.py`
- Run A/B configs 与 immutable initialization manifest
- step 1–40 sample/render/label/LR/loss/grad/token timeline
- step 20 complete resumable checkpoint state manifest
- adapter/optimizer/scheduler tensor checksums 与 `max_abs_diff`
- correct-resume 和 scheduler/cursor failure-injection 对照
- selected candidate 的 adapter/reload/merged-export generation parity

## 允许重新打开的条件

`selected_candidate=true`、`checkpoint.resumable=true`、所需 state files 和数据游标全部通过 integrity audit，且新 SFT charter 冻结了相同初始化与 Run A/B 对照合同。满足前述条件前，本状态只能是 `not_run_blocked`。
