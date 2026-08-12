# Artifacts

这里存放跨天复用的真实产物。Day 文件夹用于当天 tracking 和链接，较大的训练输出/checkpoint 可以放在工作区之外，只在这里记录路径和元数据。

| 目录 | 用途 |
|---|---|
| `configs/` | 可复现 YAML/JSON/launch commands |
| `data/` | 数据 schema、manifest、hash、split 说明；避免提交大数据 |
| `eval/` | frozen eval、scorers、逐样本 predictions |
| `logs/` | 精简日志、曲线导出、环境快照 |
| `reports/` | SFT/DPO/GRPO/Eval/Capacity 正式报告与每日 SVG 学习路线图 |
| `runbooks/` | OOM、checkpoint、NCCL、reward collapse 排查手册 |
| `scripts/` | memory estimator、数据验证、generation、scoring 等脚本 |

## 命名

```text
run-YYYYMMDD-HHMM-<model>-<purpose>
report-dayXX-<topic>.md
config-dayXX-<topic>.yaml
capstone-<T0|T1|T2|S0|S1|S2|S3>-<purpose>.<ext>
```

## Day 13+ 活动 Lineage

- `qwen3-0.6b-day01-12-v1`：不可改写的历史 lineage；保留原 tokenizer、template、data/eval、checkpoint 与失败证据。
- `qwen35-4b-day13-plus-v2`：新的活动 lineage；模型固定为 `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`。Day 15 已关闭；Day 16 已以 no-candidate 收束，见 [`Day 15 Close`](reports/day15-close.md)、[`Day 16 Gap Audit`](reports/day16-gap-audit.md) 与 machine-readable [`no-anchor`](reports/day16-qwen35-provisional-anchor.json)。Day 17 的 [`Gap Audit`](reports/day17-gap-audit.md)、[`blocked equivalence status`](reports/day17-qwen35-resume-equivalence.md) 和 [`readiness gate`](configs/day17-qwen35-resume-readiness.json) 已落盘，状态不变；runtime 设计移入顺序课程 [`Day 20 Optional R`](../day-20-weekend-training-failures/README.md#optional-r--exact-resume-failure-lab)，不再是 S1 硬 gate。当前仍无 S1。
- 机器可读的迁移状态与 checkpoint graph：[`configs/qwen35-4b-migration-contract.json`](configs/qwen35-4b-migration-contract.json)。
- Day 18 Megatron compatibility 已 `closed_pass_c0_c5`：[`Close 报告`](reports/day18-close.md)、[`machine-readable closeout`](reports/day18-closeout.json) 与原 [`compatibility report`](reports/day18-qwen35-megatron-compatibility.md)。本地 evidence bundle 已验证，但完整 C2–C5 payload 未归档到 Git。
- v1/v2 的 rendered text、token IDs、label spans、token budget、eval comparison key 和 aggregate score 不可混用。

## 大文件规则

- 不把模型权重和完整 checkpoint 提交到 Git。
- 为外部 checkpoint 保存绝对路径、模型 hash、config 和产生它的 command。
- Eval 保存逐样本结果和 prompt/config hash。
- 删除任何 checkpoint 前，先确认它是否是后续实验的唯一可复现依赖。

## Capstone 外部 Checkpoint Registry

Qwen3.5-4B 权重不提交到 Git。活动图是 `S0 Base -> S1 selected coding SFT -> S2 direct coding RL`；teacher/OPD 角色 `T0/T1/T2/S3` 当前为 deferred/unselected，只有单独批准的 charter v2 才可创建。每个外部 checkpoint manifest 至少记录：

- role/ID（`T0/T1/T2/S0/S1/S2/S3`）与 parent checkpoint hash；
- model/config/processor/tokenizer/template revisions 与 hashes；
- architecture、loader、task modality、training scope、freeze policy、trainable-module coverage；
- training framework/config/code/container/hardware；

仓库根目录的 `.gitignore` 同时排除 `*.safetensors`、checkpoint 目录、`remote-runs/` 与重复的 `snapshots/`。需要共享权重时使用外部 artifact store，并只提交 manifest、hash、选择结论和可重放的小型证据。
- TP/DP/PP、distributed optimizer 与 checkpoint shard metadata；
- resumable state 路径和 inference export 路径；
- conversion command、source/destination hashes 与 parity evidence；
- dataset/eval suite、teacher 或 policy version（适用时）；
- train/rollout/teacher-scoring GPU-hours。

`artifacts/checkpoints/` 只存 registry/manifest，不存大权重本体。
