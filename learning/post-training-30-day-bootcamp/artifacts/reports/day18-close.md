# Day 18 Close — Qwen3.5 × Megatron Compatibility

Close 日期：`2026-08-10`

最终状态：`closed_pass_c0_c5`

追加 GPU：`0`

## 结论

Day 18 正式关闭，不补跑 GPU。原协议要求的 C0–C5 已在 `day18-qwen35-20260808T073811Z` 全部通过；本次 Close 重新验证了本地 evidence bundle、不可变 PASS manifest、65 项 evidence inventory、问题账本和 16 项 preparation/finalizer 测试，没有发现会推翻原结论的硬缺口。

Day 18 证明的是冻结 envelope 内的 Qwen3.5-4B Megatron compatibility、真实 optimizer update、distributed checkpoint/fresh-process continuation、HF export/reload parity，以及相同两行 fixture 的 learnability。它不产生 S1，也不证明 vision、长序列、多节点、吞吐 scaling、泛化或 exact-resume equivalence。

## 原协议验收

| Gate | Close 判定 | 关键证据 |
|---|---|---|
| C0 HF→MCore conversion | pass | 738 mappings；15 个 MTP tensors exact round-trip |
| C1 single-rank parity | pass | 两行 loss-token argmax mismatch 为 0；loss/logit tolerance 通过 |
| C2 `TP=1, DP=2` | pass | 3 个真实 update；model/Adam/RNG checkpoint 和 fresh loadcheck 通过 |
| C3 `TP=2, DP=1` | pass | 3 个真实 update；TP2 checkpoint 通过 |
| C4 fresh continuation/export | pass | 新进程从 iteration 3 恢复到 5；HF export/reload parity 通过 |
| C5 150-step tiny overfit | pass | 150/150 updates；72/72 teacher-forced tokens；vision/aligner bitwise unchanged |

Finalizer 总结：10/10 required gates、8 个 runtime codepath nodes、65 项 hashed evidence、17 类 resolved/observation problems、`open_problem_count=0`。

## Closeout 重新验证

`2026-08-10` 在本地执行：

- evidence tar SHA-256：`2afc7ad1b39bdde3aad40832ceb78ab1e01116ee4f7bbc19ccf85beb4570c9c0`，与报告一致；
- tar 共 113 个 members，并包含 `DAY18-PASS.json` 与 `evidence/day18-final-summary.json`；
- PASS manifest 的 65 项 inventory 全部存在，SHA-256 和 byte size 均匹配；
- `required_gates=10`、`status=day18_pass`、`failures=[]`、`open_problem_count=0`；
- `python3 -m unittest -v test_day18_preparation.py`：`16/16 pass`；
- adjacent SHA sidecar 已从 AutoDL 绝对路径改为本地 basename，`shasum -a 256 -c` 可直接执行。

机器可读结果见 [`day18-closeout.json`](day18-closeout.json)。

## 状态收尾

- [`run-contract.json`](../configs/day18-qwen35-megatron/run-contract.json) 保留 `prepared_locally/...runtime_unverified`，因为它是运行前预注册合同；不回写历史状态。
- [`day18-qwen35-megatron-compatibility.md`](day18-qwen35-megatron-compatibility.md) 保留运行时上下文，同时追加当前 Close 状态和 Day 15 后续关闭说明。
- [`AUTODL-RUNBOOK.md`](../../day-18-megatron-minimum-codepath/AUTODL-RUNBOOK.md) 标为历史 handoff；实际 run 已完成，不应照 runbook 自动重跑。
- 课程 README、PROGRESS、迁移合同、Artifacts 索引与 AutoDL 预算统一更新为 `closed_pass` / `0 additional GPU`。

## 权重与 checkpoint 留存边界

本地保留并验证的是 6.8 MiB evidence bundle；它故意不包含 C2–C4 full-state checkpoint、C4/C5 HF export 或 C5 model-only checkpoint。报告记录的远端 canonical run 是：

```text
/root/autodl-tmp/runs/day18-qwen35-20260808T073811Z/
```

Close 时没有重新连接 AutoDL 验证该目录是否仍存在，因此 retention 状态写为 `unknown_remote_retention_evidence_bundle_retained`。这不阻塞 Day 18 的历史证据关闭，但有以下边界：

- 不能声称这些大 payload 当前仍可直接 reload；
- 若后续必须复用原 export/checkpoint，先尝试从原 AutoDL 数据盘恢复并重新 hash；
- 若远端已删除，只能从 frozen Base、合同和脚本重新构建，不能用本地 evidence tar 冒充权重备份。

## 不补跑事项

以下是明确排除的扩展范围，不是 Day 18 Close 缺口：

- vision/image/video path；
- 长序列、更多 TP/PP/CP 组合或多节点；
- throughput scaling 或生产性能基准；
- 一般化能力、正式 coding SFT candidate 或 S1；
- uninterrupted 对 resumed 的逐 step exact parity。该诊断已降级为顺序课程 Day 20 Optional R。

## 最终决定

- Day 18：`closed_pass_c0_c5`。
- 原实验协议：通过，不需要重跑。
- GPU 预算：追加 `0`。
- 可复用结论：冻结 envelope 内的 Megatron runtime/compatibility/learnability evidence。
- 下游：不自动创建 S1；Day 21 仍等待新的合法 SFT candidate。
