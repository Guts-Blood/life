# Day 16 Gap Audit — Controlled LoRA SFT 与 Candidate Closeout

审计日期：`2026-08-09`
状态：`closed_no_eligible_candidate_by_day20_evidence`
追加 GPU 运行：`0`
活动权重：`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`

## 结论

Day 16 以 `no_eligible_qwen35_sft_anchor` 收束，不再为了凑出 early/mid/final checkpoint 而启动主训练。

Day 20 的 canonical run 已经执行了 Day 16 所需的受控 LoRA 前置选择：同一 Base、同一数据顺序、同一 LoRA language-module 白名单和三个预注册学习率，各完成 16,000 supervised tokens、103 optimizer steps、5-step safety gate、adapter checkpoint 与固定 32 条 diagnostic eval。三档学习率都触发至少一个不依赖 E2B 的硬失败，因此缺失 E2B correctness 不会改变“不得进入 main”的结论。

这不是一个成功的 SFT anchor，也不是 Day 20 probe 的追认晋级。合法退出状态本来就包括 `no eligible candidate`；本次只把已经存在的更强运行证据映射回 Day 16，并保持 Base active。

## Evidence crosswalk

| Day 16 要求 | 实际证据 | 判定 |
|---|---|---|
| exact Qwen3.5 Base v2 parent | 本地且不纳入 Git 的 `remote-runs/day20-qwen35-lora-20260809T042005Z/DAY20-MANIFEST.json` 固定 Base snapshot SHA-256 `a176a3c9…195b` 与 exact revision；运行包身份由仓库内 [`run_provenance.json`](../../day-20-qwen35-balanced-lora-sft/reports/day20-probe-analysis-20260809/evidence/run_provenance.json) 记录 | `covered` |
| 新数据/token budget，不继承 0.6B | Day 20 六源合同；probe 821 records、精确 16,000 supervised tokens；main 13,197 records / 256,000 tokens 只准备未训练 | `covered_by_actual_day20_contract` |
| LoRA language-only scope | 三个 probe 共用 248 个 target modules、496 个 trainable parameter tensors；target regex 必须经过 `language_model.layers`，ViT/aligner/embedding/lm_head 排除 | `covered` |
| 5-step safety 与真实训练更新 | 三档 probe safety 均 `pass`，随后各完成 103 optimizer steps 并保存 16,000-token adapter checkpoint | `covered` |
| 预注册 candidate decision | [`probe_metrics.json`](../../day-20-qwen35-balanced-lora-sft/reports/day20-probe-analysis-20260809/evidence/probe_metrics.json) 逐候选记录必要门禁失败 | `no_eligible_candidate` |
| early/mid/final main candidates | LR gate 无通过项，`main` 按合同未启动 | `not_applicable_after_fail_closed_probe_gate` |
| packing correctness/throughput parity | Day 15 未提供 packing contract；Day 20 固定 `packing=false`、`padding_free=false`，未做 P1 | `not_run_packing_remains_false` |
| confirmation split | 没有 candidate，因此未创建或消费新的 confirmation split | `unconsumed_but_future_contract_missing` |

## Candidate 判定复算

固定 diagnostic 每个候选 32 条：General、Math、Finance、Code 各 8 条。Base 的 Math 为 `7/8`，Code 静态 execution eligible 为 `8/8`。预注册必要条件要求 Math/Finance/Code 各自最多比 Base 退化 1，且 Code 静态 eligible 至少 `7/8`。

| Candidate | General | Math | Finance | Code 静态 eligible | 与 E2B 无关的失败 |
|---|---:|---:|---:|---:|---|
| Base | 0/8 | 7/8 | 3/8 | 8/8 | baseline |
| LoRA `1e-5` | 0/8 | 4/8 | 4/8 | 8/8 | Math 比 Base 低 3，超过允许的 1 |
| LoRA `3e-5` | 5/8 | 5/8 | 4/8 | 6/8 | Math 低 2；Code eligible 低于 7 |
| LoRA `1e-4` | 6/8 | 6/8 | 4/8 | 0/8 | Code eligible 低于 7 |

本轮没有 E2B 结果，也没有正式 `PROBE-SELECTION.json`。这里不补造这两个 artifact。因为每个 LoRA 已先触发其它必要条件，E2B 最好结果也无法让它们通过完整 gate；因此可以确定“不进入 main”，但不能声称 HumanEval correctness 或候选总分排名。

## 原计划产物处置

| 原 Day 16 产物 | 处置 |
|---|---|
| `artifacts/configs/day16-qwen35-lora-sft/` | 不复制；实际运行配置保留在 canonical Day 20 run 的三个 `configs/probe-lr-*.json`。 |
| `artifacts/data/day16-qwen35-coding-sft-manifest.json` | 不补造；实际数据身份是 Day 20 `DAY20-MANIFEST.json`，两者不能改名冒充。 |
| `artifacts/logs/day16-qwen35-sft-step-metrics.jsonl` | 不复制；三条实际 step metrics 保留在 Day 20 `evidence/probe-*/attempt-*/step-metrics.jsonl`。 |
| `artifacts/reports/day16-packing-parity.md` | 已生成非执行结论：packing 未通过 parity，因此固定为 `false`。 |
| `artifacts/reports/day16-qwen35-sft-trajectory.md` | 已生成 Day 20 → Day 16 trajectory crosswalk。 |
| `artifacts/reports/day16-qwen35-provisional-anchor.json` | 已生成 machine-readable `no eligible anchor` 结论；checkpoint 为 `null`。 |

## Packing 边界

原 Day 16 只允许在 Day 15 packing contract tests 通过后运行 P0/P1。Day 15 Close 明确没有 packing 结论；Day 20 则有意固定 unpacked。最安全且不增加 GPU 成本的合法结论是 `packing=false`，不是把未跑的 parity 写成 `pass`。

未来只有新 SFT charter 确实需要 packing 时，才单独比较 boundary、label mask、position IDs、cross-sample attention isolation、loss、有效 label tokens/s、step time 与峰值显存。当前没有吞吐收益声明。

## 后续边界

- `S1` 仍不存在；Base 保持 active。
- Day 17 没有 selected/resumable Day 16 candidate，不能按 anchor continuity 路径启动。若只做 method-only resume，必须另行批准且不得晋级。
- Day 21 当前没有 candidate 可 audit；在新 candidate charter 产生候选前保持 blocked。
- DPO/GRPO 继续 blocked until `S1`。
- 下一份 SFT charter 应先统一 Code continuation target 的 byte-0 缩进合同、版本化 eval adapter/comparison key、恢复独立 sandbox 证据，再预注册新的 LR probe；不得原地改写 Day 20 evidence-bound runner。

最终结论：`Day 16 closed with no eligible candidate; packing remains false; no S1; Base remains active.`
