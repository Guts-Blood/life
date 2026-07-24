# Artifacts

这里存放跨天复用的真实产物。Day 文件夹用于当天 tracking 和链接，较大的训练输出/checkpoint 可以放在工作区之外，只在这里记录路径和元数据。

| 目录 | 用途 |
|---|---|
| `configs/` | 可复现 YAML/JSON/launch commands |
| `data/` | 数据 schema、manifest、hash、split 说明；避免提交大数据 |
| `eval/` | frozen eval、scorers、逐样本 predictions |
| `logs/` | 精简日志、曲线导出、环境快照 |
| `reports/` | SFT/DPO/GRPO/Eval/Capacity 正式报告 |
| `runbooks/` | OOM、checkpoint、NCCL、reward collapse 排查手册 |
| `scripts/` | memory estimator、数据验证、generation、scoring 等脚本 |

## 命名

```text
run-YYYYMMDD-HHMM-<model>-<purpose>
report-dayXX-<topic>.md
config-dayXX-<topic>.yaml
```

## 大文件规则

- 不把模型权重和完整 checkpoint 提交到 Git。
- 为外部 checkpoint 保存绝对路径、模型 hash、config 和产生它的 command。
- Eval 保存逐样本结果和 prompt/config hash。
- 删除任何 checkpoint 前，先确认它是否是后续实验的唯一可复现依赖。

