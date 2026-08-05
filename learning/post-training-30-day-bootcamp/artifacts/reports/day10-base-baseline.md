# Day 10 Qwen3-0.6B Base Baseline

日期：`2026-08-05`

状态：`dev_automated_eval_complete / code_sandbox_scored / human_review_pending`

## 结论

112 条 `dev` baseline 已完整生成并通过结构、顺序、hash、token limit 和 split 验证；没有生成或查看 48 条 `frozen_test` 输出。28 条 code completion 已在逐样本新建的 E2B 强隔离 sandbox 中完成评分，且没有 infrastructure/null 结果。最终四-slice macro 为 **13.39%**（15/112）；其中 code pass@1 为 **0.00%**（0/28）。

这个 Base baseline 的主要价值是给后续 SFT 建立纵向锚点，而不是公平衡量 Qwen3-0.6B-Base 的横向能力。模型是 non-instruction-tuned Base，却通过 chat template 做 free-form generation；它经常把指令当成待续写文本，112/112 都生成到各 slice 的 token ceiling，没有自然 EOS。

## Dev metrics

| Slice | Correct / scored | Accuracy | Wilson 95% | Parse error | Wrong / failed | Sandbox pending | Ceiling hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| general | 2 / 28 | 7.14% | 1.98%–22.65% | 17 | 9 | 0 | 28 |
| math | 12 / 28 | 42.86% | 26.51%–60.93% | 0 | 16 | 0 | 28 |
| finance | 1 / 28 | 3.57% | 0.63%–17.71% | 16 | 11 | 0 | 28 |
| code | 0 / 28 | 0.00% | 0.00%–12.06% | 0 | 28 | 0 | 28 |

| Aggregate | Result |
|---|---|
| Final four-slice macro | 13.39%（15 / 112） |
| General/math/finance diagnostic macro | 17.86%（15 / 84） |
| Macro uncertainty | 预注册的 stratified bootstrap 尚未计算；不伪造 pooled Wilson |
| Input/output tokens | 27,654 / 43,904 |
| Summed generation latency | 1,272.058 s（约 21m12s） |

Wilson 区间只用于固定小型 pilot 集上的敏感度提示。当前是配额构造的 28-per-slice manifest，不应把这些区间解释为对完整用户任务总体的覆盖保证。

## Repeatability

同一模型、manifest、decoding、CPU runtime 下各跑 10 条 A/B repeat：

- 两次均为 10 条，sample order 一致；
- `output_token_ids`：0 mismatch；
- `raw_output_hash`：0 mismatch；
- `scorer_result`：0 mismatch；
- `comparison_key` 与 `run_hash`：0 mismatch。

两份 JSONL 的 exact file hash 不同，是因为 wall-clock timestamp 与 latency 是运行观测字段；语义输出完全一致。

| Artifact | Exact SHA-256 |
|---|---|
| repeat A | `0411c568ac6746ac06a50f29bd91b98d56e75a09554e1f74f2f85e265933d365` |
| repeat B | `513b27eb3eca1ee5c57667707a2e95fc33cc5433b66f32a9aac850197a7eced2` |
| repeat run hash | `sha256:5a9952525bda657057d0f63f69ff4dc00d437e40e46dbed10343ed6391a3290f` |

## Evidence identity

| Item | Value |
|---|---|
| Predictions exact SHA-256 | `9760a4d297751c97e00edf7df6087e0eaccbc6b91137a2e3fcee15e290be6260` |
| Run hash | `sha256:49408d403fdc72f9f125c9aeda158e0d6f22d80cbce05ac74f69b9d9b535de2d` |
| Generation comparison key | `sha256:b0ecfbf25d6879f05b60c945319685189d1b0737e75d9968de3f50ff6c3a7788` |
| Complete comparison key | `sha256:850a69854fe0a945d7b86f3c71a59a7977e09dd42a2df96ca7548b6068bd0f14` |
| Code execution protocol hash | `sha256:9571c64301029e230ab2d01e1b8daf8a6dcf7c4a5c8b0696e21d9b073a730c87` |
| Code run hash | `sha256:147616a3cd6f02975a2be509b6420703f4334631db0a1f5d2ca421ef6b51a626` |
| Code sidecar exact SHA-256 | `d50be508bf284beac8b8e731fd521a68547dccf928d7206938a64d65b264509c` |
| Code results semantic hash | `sha256:b91139dc8c535a18a6ea51abdfed75f89bbb008e6bbf5913001e96642d1b2f14` |
| Summary semantic hash | `sha256:6cbbb641767e258779cb18bd33517bbb6deb08af07580c45d614b6853cc3ab17` |
| Summary exact SHA-256 | `25def33ed69a2c6752b878cf3772d74139709949629ee2d6d580b4c161c37bed` |

The analyzer independently reconstructs manifest, protocol, execution, comparison and run hashes; rejects duplicate IDs, order drift, frozen records, field-hash mismatches and incomplete slice counts before aggregating.

## Initial failure taxonomy

这只是自动证据支持的初始 taxonomy，不能替代尚未完成的人审：

1. **Format/parse failure**：general 17、finance 16；Base 没有稳定遵循最终答案标记。
2. **Capability/wrong answer**：在成功解析的结果中，general 9、math 16、finance 11 为错误答案。
3. **Generation-control failure**：112/112 hit ceiling，说明 EOS/指令遵循在该 Base setup 下失效。
4. **Code execution failure**：code 0/28；25 条 `syntax_error`、3 条 `runtime_error`，没有 assertion-level wrong answer，也没有 sandbox infrastructure failure。
5. **Scorer defects found before freeze**：dry run 暴露过 MMLU 和 TAT-QA 两个 false extraction，均在正式 v3 baseline 前修复并回归测试。

30 条 dev-only audit packet 已确定性生成，四个 slice 分布为 general 7、math 14、code 4、finance 5；全部 `human_review.status=pending`，直到用户逐条确认 generation、parse、scorer 与 capability 分类。不能把预生成 packet 误写成“已完成 30 条人审”。packet 已合并 code sidecar，四条 code review 行记录为 3 条 `syntax_error` 与 1 条 `runtime_error`；合并器 provenance 更新后已 byte-exact 重建，packet exact SHA-256 为 `9b7da2940950a8ef8a34da4ef6bb404046e6d04f557df0241c9072dc6e5837df`。

## Remaining gates

- 完成 30 条 dev 人工复核和 taxonomy 修订；
- 如果 Day 12 使用 H100/CUDA 做 eval，在相同 GPU execution protocol 下重跑 Base；
- 在 A/B checkpoints 都选定前，继续保持 `frozen_test` 未消费。

## Source artifacts

- `../eval/day10-qwen3-0.6b-base-predictions.jsonl`
- `../eval/day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl`
- `day10-base-baseline-summary.json`
- `../eval/day10-qwen3-0.6b-base-repeatability-a-10.jsonl`
- `../eval/day10-qwen3-0.6b-base-repeatability-b-10.jsonl`
- `../eval/day10-base-human-review-packet-30.jsonl`
- `../../day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json`
- `../logs/post-training-sandbox-environment.md`
- `day10-eval-protocol.md`
