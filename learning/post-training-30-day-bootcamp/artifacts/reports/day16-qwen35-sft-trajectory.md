# Day 16 Qwen3.5 LoRA SFT Trajectory Crosswalk

日期：`2026-08-09`
来源 run：`day20-qwen35-lora-20260809T042005Z`
状态：`probe_complete_main_not_started_no_eligible_candidate`

## 冻结训练身份

- Parent：`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`
- Hardware：实际训练使用单卡 `NVIDIA H800 PCIe`；主机可见 2 卡，但 Day 20 合同只允许 GPU 0。
- Runtime：HF Transformers；ms-swift commit `565a1ad586a21d24b23931c52d2c62b49c39bee8`；Torch `2.10.0+cu126`；Transformers `5.12.1`；PEFT `0.19.1`。
- Scope：BF16、`max_length=2304`、batch `2`、gradient accumulation `4`、`packing=false`、`padding_free=false`。
- LoRA：rank 8、alpha 16、dropout 0.05；248 个 language-module targets、496 个 trainable parameter tensors；ViT、aligner、embedding 与 lm_head 不在 target regex 内。
- Probe data：821 records，General/Math/Finance/Code 各 4,000 supervised tokens，总计 16,000。

## 训练轨迹

| LR | Safety | Optimizer steps | Supervised tokens | Adapter checkpoint SHA-256 | Diagnostic 结果 |
|---:|---|---:|---:|---|---|
| `1e-5` | 5-step pass | 103 | 16,000 | `9c558612…3ff0` | Math `4/8`，相对 Base `-3`；失败 |
| `3e-5` | 5-step pass | 103 | 16,000 | `2735f44a…1b9` | Math `5/8`、Code eligible `6/8`；失败 |
| `1e-4` | 5-step pass | 103 | 16,000 | `f9ffcb58…138` | Math `6/8`、Code eligible `0/8`；失败 |

三条训练 summary 的 self-hash 分别为：

- `1e-5`：`84ac6e45dd0ddefe6efb80f2a543127e9a51cf36e10e27d023c2c6e4ad7a2a75`
- `3e-5`：`27274659f99631d60dd86cd9ec03fc2fa808130ed6698f46a20ef012e93f2c90`
- `1e-4`：`51b9e2edd072cc78621272dadc5551778dcbea6b373f678c41ee798108263d55`

本地 evidence 文件（位于被 `.gitignore` 排除的 `remote-runs/`，仓库只保留 provenance、self-hash 和汇总）：

- `remote-runs/day20-qwen35-lora-20260809T042005Z/evidence/probe-1e-5/training-summary.json`
- `remote-runs/day20-qwen35-lora-20260809T042005Z/evidence/probe-3e-5/training-summary.json`
- `remote-runs/day20-qwen35-lora-20260809T042005Z/evidence/probe-1e-4/training-summary.json`
- [`run provenance`](../../day-20-qwen35-balanced-lora-sft/reports/day20-probe-analysis-20260809/evidence/run_provenance.json)
- [`candidate metrics`](../../day-20-qwen35-balanced-lora-sft/reports/day20-probe-analysis-20260809/evidence/probe_metrics.json)

## 停止点

所有 probe 都证明 LoRA 能更新、保存且通过短安全检查；它们没有证明模型质量门禁通过。每个 probe 至少违反一个不依赖 E2B 的必要条件，所以主训练没有启动，64k/153.6k/256k early/mid/final checkpoints 不存在。

本报告不创建 formal `PROBE-SELECTION.json`，不挑选“失败中最好”的 LR，不合并 adapter，也不登记 provisional anchor。
