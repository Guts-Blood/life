# Day 23 Qwen3.5 Coding DPO — CPU Preflight

执行日期：`2026-08-14`（提前完成原计划 Day 23 的 CPU 阶段）

状态：`CPU_READY_GPU_PENDING`（当时状态；GPU 阶段随后以 `CLOSED_NO_CANDIDATE` 结束）

轨道：`experimental_ai_assisted`；不宣称 formal human-reviewed readiness

> 后续状态（`2026-08-14`）：G0–G2 已通过；两个 5-step mechanism topology 均只有 `1/4` pair 改善，terminal NO-GO。30-step/dev/heldout 未运行。最终证据见 [`day23-qwen35-coding-dpo-smoke.md`](day23-qwen35-coding-dpo-smoke.md)。

## 结论

Day 23 的 CPU 阶段已完成，可以在开卡后直接进入远端 payload 与 GPU runtime gates。当前没有 CPU blocker；仍然禁止直接启动 optimizer，直到远端 S1 实体、policy/reference 拓扑、显存、冻结范围和 save/reload 均在目标 GPU runtime 中实证通过。

本轮没有加载模型权重、没有访问 AutoDL、没有执行 optimizer step，也没有消费 preference held-out。

## 已关闭的 CPU gates

| Gate | 结果 | 证据 |
|---|---:|---|
| Day 21 promoted S1 compact ancestry | PASS | 固定 promotion/key/merged-export trust roots；self-hash、cross-binding、S1 role 与 parity 状态重验 |
| Day 22 experimental close | PASS | 独立 validator：200 pairs，split `154/17/29`，formal-ready 仍为 false |
| Day 23 deterministic compilation | PASS | train/dev/heldout JSONL 可 byte-for-byte 重建；heldout 不进入 trainer args |
| pinned ms-swift dataset loader | PASS | strict/no auto mapping；`154/17/29` 全保留，0 silent deletion |
| Qwen3.5 RLHF template | PASS | 新 alias `day23_qwen3_5_dpo_target_v1`；Day 20 历史 alias 未修改 |
| chosen/rejected encoding | PASS | 200 pairs / 400 branches 与 Day 22 frozen token-label evidence exact match |
| response mask / coding boundary | PASS | 400/400 response-only，400/400 首个四空格 token 受监督，400/400 零截断 |
| length envelope | PASS | input `73–381` tokens；response `7–249`；冻结 `max_length=512` |
| DPO scalar oracle | PASS | sigmoid DPO、swap/reference/beta/mask/gradient；torch 与 pinned `DPOTrainer.dpo_loss` 交叉验证 |
| real JSON CLI parse | PASS | 两个 stage 均走 `parse_yaml_args → parse_args(RLHFArguments)`，不是直接构造 dataclass |
| mechanism loader | PASS | 固定 train 前 4 对，顺序/text exact；dev 0 条；dataset/dataloader shuffle 均关闭 |
| tamper/fail-closed | PASS | upstream trust roots、compiler/source inventory、processor assets、claims、CPU gate inventory 与 heldout leakage 均受独立 validator 约束 |

## 冻结产物

- Data manifest：`8827c6cf6db8994cf59e6bcb218b9ab1d510b2218fb380cc13129dfa16aac65c`
- Processor audit summary：`60e0ee95156e6433bc5325d8cf980dbe53f211aee8b509c67c6da59bfda34272`
- CPU run contract：`9d85ff6e149000ab3a4fea160eda5804c96a9c41655848dfb4613e0043c037cc`
- Real JSON CLI argument audit：`2de31f509ff1eb00fb46f52ebf2bd6e7bdc2ca469240e0a0645b1c10907d46d2`
- Independent bundle status：`valid_cpu_ready_gpu_pending`

核心路径：

- `artifacts/data/day23-qwen35-coding-dpo-{train,dev,heldout}.jsonl`
- `artifacts/data/day23-qwen35-coding-dpo-data-manifest.json`
- `artifacts/eval/day23-qwen35-coding-dpo-processor-audit-summary.json`
- `artifacts/eval/day23-qwen35-coding-dpo-argument-audit.json`
- `artifacts/configs/day23-qwen35-coding-dpo/cpu-run-contract.json`
- `artifacts/scripts/validate_day23_qwen35_dpo.py`

## Runtime evidence 与边界

CPU processor runtime：Python `3.11.15`、Transformers `5.12.1`、torch `2.9.1`、PEFT `0.18.0`、TRL `0.29.1`、Accelerate `1.12.0`、Datasets `4.8.4`、Hugging Face Hub `1.27.0`、ms-swift `4.5.0.dev0`。

实际 import 已证明来自 clean vendor checkout `565a1ad586a21d24b23931c52d2c62b49c39bee8`；processor/template、JSON CLI、RLHF args、reference context 与 DPO trainer 的关键源码均已 hash 绑定。为兼容 pinned loader，仅在隔离的 `tmp/day22-token-runtime` 中把 Datasets 固定到 `4.8.4`，没有修改主环境。

`pip check` 仅报告缺少 `cpm-kernels`。它不影响本轮 text-only、无权重 CPU gate；GPU learner runtime 必须重新建立完整 CUDA lock 并处理该依赖。`decord` 的 warning 同样未进入 text-only 数据，但 GPU runtime preflight 仍须记录最终依赖集。

## 冻结训练意图

共同配置：DPO sigmoid、`beta=0.1`、fresh LoRA rank 8 / alpha 16、LR `5e-6`、batch 1 / GA 8、bf16、gradient checkpointing、`max_length=512`、packing/padding-free 均关闭、vision/aligner 冻结、15% 最低空闲显存门槛。

Reference 逻辑身份与 policy 初始状态均为 Day 21 promoted merged S1。可执行 JSON 必须省略 `adapters`、`ref_adapters`、`ref_model`、`resume_from_checkpoint`；这些字段的 ms-swift 默认值分别是空、空、None、None。原因是 pinned JSON CLI 会把显式空列表/null 展开成非法 argv。GPU binding 还必须把 dataset、val_dataset、external plugin 转成绝对路径，关闭 `add_version`，并只输出合法 ms-swift keys。

两个 run 都从同一 S1 fresh start，互不 resume：

1. `mechanism_5step`：固定 4 对；无 dev；dataset/dataloader 不 shuffle；在 batch 1 / GA 8 下循环这些样本直到 5 个 optimizer steps；step 5 保存；产物永不作为 candidate。
2. `bounded_smoke_30step`：完整 train；step 15/30 save+eval；dev 只用于外部逐 pair selector。trainer native aggregate 不是最终 selector。

Preference heldout 只允许在 checkpoint 选定后消费一次。coding sandbox、general/math/format guardrails 同样在 selection 之后执行，并分别报告机制、preference 指标和真实 coding correctness。

## 开卡后的 pipeline

| 顺序 | Gate | 必须满足的退出条件 |
|---:|---|---|
| G0 | Remote S1 payload | promotion/key/export/checkpoint/shards 均存在；bytes/SHA-256/checkout commit 与本地 trust roots 一致 |
| G1 | GPU topology + one step | merged S1 + fresh policy LoRA；reference 为禁用 fresh adapter 的同一 S1；仅 target regex 可训练；vision/aligner/embedding/lm_head 冻结 |
| G2 | Memory + save/reload | policy/reference/optimizer/activation/logits 全计入；峰值后 ≥15% 余量；一步 finite；新进程 reload parity 通过 |
| G3 | 5-step mechanism | 4 对中至少 3 对 reward margin 改善；所有 loss/reward/grad finite；该 checkpoint 不晋级 |
| G4 | Fresh 30-step smoke | 独立从 S1 启动；15/30 均 save+reload；外部逐 pair dev selector 冻结唯一 winner |
| G5 | One-shot confirmation | 只对 winner 消费 preference heldout，并跑 sandbox/general/math/format guardrails；生成最终 smoke report |

任一 gate 失败就停止，不临时用 quantization、offload、ZeRO、缩短序列或 Base checkpoint 绕过。

## 验证记录

- Day 23 unit/tamper tests：`27/27 PASS`
- Required DPO loss tests（pinned torch/ms-swift）：`11/11 PASS`
- Day 20 regression：`158/158 PASS`
- Day 21 regression：`5/5 PASS`
- Day 22 regression：`109/109 PASS`
- 合计：`310 PASS`
- Data、processor、run-contract、argument-audit 四个 `--mode check`：全部 exit `0`
- Day 22 independent close validator：`valid_completed_experimental_ai_assisted`
- Day 23 independent validator：`valid_cpu_ready_gpu_pending`

## Remaining gates（preflight 时点）

- `pending_remote_s1_payload_verification`
- `pending_gpu_reference_lora_memory_save_reload`

这两项在本报告生成时是 GPU optimizer 的硬 gate，不否定 CPU 阶段已经完成；它们随后均通过。最终 pipeline 在更后的 5-step mechanism gate 以零候选关闭，见 closeout 报告。
