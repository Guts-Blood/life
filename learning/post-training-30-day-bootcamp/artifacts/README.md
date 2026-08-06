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

## 大文件规则

- 不把模型权重和完整 checkpoint 提交到 Git。
- 为外部 checkpoint 保存绝对路径、模型 hash、config 和产生它的 command。
- Eval 保存逐样本结果和 prompt/config hash。
- 删除任何 checkpoint 前，先确认它是否是后续实验的唯一可复现依赖。

## Capstone 外部 Checkpoint Registry

8B/4B 权重不提交到 Git。每个外部 checkpoint manifest 至少记录：

- role/ID（`T0/T1/T2/S0/S1/S2/S3`）与 parent checkpoint hash；
- model/config/tokenizer/template revisions；
- training framework/config/code/container/hardware；
- TP/DP/PP、distributed optimizer 与 checkpoint shard metadata；
- resumable state 路径和 inference export 路径；
- conversion command、source/destination hashes 与 parity evidence；
- dataset/eval suite、teacher 或 policy version（适用时）；
- train/rollout/teacher-scoring GPU-hours。

`artifacts/checkpoints/` 只存 registry/manifest，不存大权重本体。
