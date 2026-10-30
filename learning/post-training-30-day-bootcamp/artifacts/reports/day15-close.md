# Day 15 Close — Qwen3.5-4B Onboarding 关闭报告

关闭日期：`2026-08-09`
关闭状态：`closed_superseded_by_day18_20`
追加 GPU 运行：`0`
活动权重：`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`（Base 继续 active）

## 决定

关闭 Day 15，不再为了复刻原定执行顺序而重跑一套 onboarding、tiny-overfit 和显存 smoke。

这不是把未执行的原计划补写成 `pass`。关闭理由是：Day 18–20 后续 standalone 实验已经用更长、更真实、部分更严格的路径回答了 Day 15 的核心问题——固定的 Qwen3.5-4B Base 能否在已审计环境中正确加载、保持 text-only/视觉冻结边界、完成真实训练更新、保存/恢复/导出，并在单卡 H800 上运行 LoRA。继续按 Day 15 原顺序重复这些问题，预期新增信息低于时间和 GPU 成本。

状态因此使用 `closed_superseded_by_day18_20`，而不是 `done`、`pass` 或 `acceptance_complete`。

## 关闭所依据的证据

| 证据 | 已实际证明的内容 | 不能外推的内容 |
|---|---|---|
| [`qwen35-4b-base-s0.json`](../checkpoints/qwen35-4b-base-s0.json) | exact revision、模型/processor 类型、9,319,737,856 bytes 权重、config/tokenizer/processor/两份 safetensors 的 SHA-256 | 本地下载清单本身不证明 GPU loader 或训练 |
| [`day18-qwen35-megatron-compatibility.md`](day18-qwen35-megatron-compatibility.md) | Python 3.12、冻结的 ms-swift/Transformers/PyTorch/CUDA/kernel 栈；HF↔MCore parity；text-only、ViT/aligner 冻结；2×H800 TP/DP；optimizer updates；fresh-process full-state continuation；HF export/reload；150-step two-row overfit | 不是 Day 15 的单卡 LoRA 路径；C4 没有 uninterrupted 5-step 数值对照；不证明泛化或 S1 |
| [`day-19-qwen35-sft-comparison/README.md`](../../day-19-qwen35-sft-comparison/README.md) | Qwen3.5 response-boundary adapter、HumanEval solution/completion adapter、Base C0 与 Full-SFT 诊断；确认评测错位与真实训练退化可以同时存在 | A/B/E 使用 standalone 诊断协议；没有 promoted S1 |
| [`day-20-qwen35-balanced-lora-sft/README.md`](../../day-20-qwen35-balanced-lora-sft/README.md) 与已同步 run evidence | 单卡 H800、HF Transformers、BF16、`max_length=2304`、真实 Qwen3.5 template 现场编码、监督 token 复核、语言模块白名单 LoRA、三个 LR 各 16,000 supervised tokens / 103 optimizer steps、adapter checkpoint 与 trainable inventory | 三个 probe 均未通过选择 gate；没有 main run、winner merge、exact LoRA resume 或 S1 |

## 原 Day 15 Gate 的证据映射

| 原 Gate | 关闭判定 | 理由与去向 |
|---|---|---|
| Gate 0：revision、环境、loader、scope | `covered_by_superseding_evidence` | S0 registry 冻结 revision/file hashes；Day 18 冻结完整 runtime、conditional-generation loader、processor、GDN/MTP 和 text-only freeze scope；Day 20 在同一 revision 上运行单卡 HF LoRA。 |
| Gate 1：processor/template/loss contract | `operationally_covered_not_protocol_equivalent` | Day 18/20 使用真实 Qwen3.5 processor/template，Day 20 对训练记录现场编码并按 `labels[1:] != -100` 复核监督 token，Day 19 修正 response/HumanEval 边界。原计划的独立 `10 positive + 10 edge` golden audit 没有生成，不补造。 |
| Gate 2：Day 09 全量 retokenize 与 manifest v2 | `not_executed_superseded` | 原定 7,860 条 Day 09 parent pool 的 Qwen3.5 manifest 没有生成。实际 LoRA 使用 Day 20 的六源 balanced contract、文件/行 hash、现场 tokenizer/template 审计和独立 supervised-token budget；两者不能冒充同一数据 lineage。未来新 SFT run 必须继承其实际数据合同，不引用不存在的 Day 15 manifest。 |
| Gate 3：新 Base selection dev baseline | `diagnostic_baseline_exists_not_original_split` | Day 19 untouched C0 与 Day 20 Base diagnostic 已运行，Qwen3.5 v2 response/scoring path 已产生实际预测证据；原定的新 selection/confirmation split 与 comparison key 没有冻结。任何后续 S1 promotion 仍须显式冻结其实际 selection/confirmation 合同。 |
| Gate 4：LoRA tiny-overfit 与 fresh-process resume | `learnability_covered_resume_gap_carried` | Day 20 的三条 LoRA probe 均完成 103 steps，比 one-step/tiny smoke 更强地证明 HF LoRA 能更新和保存；Day 18 证明 full-state fresh-process continuation 与 export/reload。没有 LoRA uninterrupted-vs-resume 数值等价证据，该缺口移交 Day 17，不写成已通过。 |
| Gate 5：显存与长度 preflight | `operational_capacity_covered` | Day 18 在 2×H800 完成 full-parameter TP/DP、checkpoint/export 并记录峰值；Day 20 在单卡 H800、BF16 LoRA、`max_length=2304` 下完成真实 103-step runs。它已覆盖原计划 2048 上限的实际可运行性，但不产生 packing throughput 结论。 |

## 原计划产物处置

以下原计划产物没有按 Day 15 协议产生，并且不会事后拼接或伪造：

- `artifacts/configs/day15-qwen35-transition.json`
- `artifacts/data/day15-qwen35-processor-template-contract.json`
- `artifacts/data/day15-qwen35-sft-manifest.json`
- `artifacts/eval/day15-qwen35-base-dev-predictions.jsonl`
- `artifacts/logs/day15-qwen35-tiny-overfit/`
- `artifacts/reports/day15-qwen35-onboarding.md`

本报告是关闭决策与 evidence crosswalk，不是上述 artifact 的替代内容，也不改变它们从未生成这一事实。

## 保留的缺口与后续归属

| 缺口 | 后续归属 | 约束 |
|---|---|---|
| LoRA uninterrupted vs fresh-process exact resume | Day 17 | 必须比较 optimizer、scheduler、RNG、sampler/data cursor、step/LR 与数值轨迹；Day 18 continuation 不能替代 exact LoRA parity。 |
| packing correctness/throughput parity | Day 16 | 只有 boundary/mask/position parity 通过后才允许采用 packing。Day 15 close 不产生 packing 结论。 |
| canonical SFT candidate 与 selection/confirmation contract | Day 16 / Day 21 或下一份明确批准的 SFT charter | Day 20 无 passing probe，不能手工挑选最优失败项。 |
| promoted `S1` | Day 21 promotion gate | 当前仍不存在；Base 继续 active。 |
| DPO / GRPO ancestry | Day 23 / Day 25 | 在 `S1` 产生前继续 blocked，禁止从 Base 或未过门槛 adapter 起步。 |

Day 13/14 的阅读与复盘状态不因本关闭决定自动改为完成；如继续维护顺序课程，可作为文档 backlog 单独处理，但不再作为重复运行 Day 15 GPU 实验的理由。

## 最终边界

允许陈述：

- Qwen3.5-4B Base 的 exact revision、文件身份和两个实际训练 runtime 已有可审计证据。
- 冻结 text-only scope 后，Megatron full training 与 HF LoRA 均完成了真实 optimizer updates；单卡 H800 LoRA 容量不是当前 blocker。
- Day 15 onboarding 的核心执行风险已被后续更强证据消解，因此关闭且不重跑。

禁止陈述：

- Day 15 原定 M2–M5 artifact 全部按协议通过。
- Day 09 的 7,860 条数据已经生成 Qwen3.5 manifest v2。
- LoRA exact resume、packing parity 或新的 confirmation split 已完成。
- Day 19 Full-SFT 或 Day 20 任一 LoRA probe 是合格 `S1`。

最终结论：`Day 15 closed by superseding evidence; no rerun; no promoted S1; Base remains active.`
