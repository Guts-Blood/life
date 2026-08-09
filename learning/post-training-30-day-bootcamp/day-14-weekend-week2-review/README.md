# Day 14 — 周末 Review：Week 2 Evidence 与 Qwen3.5 Readiness Contract

日期：`2026-08-09`

状态：`not_started`

强度：1 小时，仅阅读/复盘

## 主要目标

审计 Day 08–13 的证据链，并冻结 `Qwen/Qwen3.5-4B-Base` 的迁移 readiness contract。Week 2 的 Qwen3-0.6B 证据继续作为历史方法验证；它不能自动证明新模型的数据、训练或显存链路已经可用。今天不补跑训练。

## 理论 / 复盘（60 分钟）

精读清单：[Day 14 — Week 2 evidence review](../SCALING-BOOK-READING-GUIDE.md#day-14)。

- 15 分钟：v1 数据证据——contract/template hash、有效 label、lineage、质量抽样、mixture 与 decontamination；标记哪些只对旧 tokenizer 成立。
- 15 分钟：v1 训练/评测证据——tiny-overfit、resume、Day 12 `no eligible checkpoint` 与 `frozen_test_unconsumed`；不事后晋级旧 checkpoint。
- 20 分钟：填写下方 Qwen3.5 v2 readiness contract。
- 10 分钟：每项结论写成 `Claim -> Evidence -> Alternative explanation -> Limit`，只把方法和 raw IDs/scorers 交给 v2，不迁移模型结论。

## Coding

无。发现缺失 artifact 时记录缺口，不在周末临时补脚本、下载权重或重写 v1 manifest。

## 训练 / 实验

无；CPU only，不启动 GPU。

## 资源与租卡

- Day 08–13 的 manifests、logs、predictions、curves 与 reports（只读）。
- [Weekly review template](../templates/weekly-review.md)。
- CPU only；今天不新增数据、不改 scorer、不启动训练。

## 产物

- `../artifacts/reports/week2-evidence-review.md`
- `../artifacts/reports/week2-claim-evidence-table.md`
- `../artifacts/reports/qwen35-v2-readiness-contract.md`

## Qwen3.5 v2 Readiness Contract

下列字段今天只允许写 `pending` 或明确值；不得凭记忆填写 revision/version：

| Gate | 必须冻结/验证的对象 | Day 15 前状态 |
|---|---|---|
| Model | `Qwen/Qwen3.5-4B-Base` exact revision、weight/config hash、license record | pending |
| Loader | resolved model class、`AutoProcessor`/tokenizer、remote-code policy | pending |
| Scope | text-only coding；ViT/aligner train/freeze policy | pending |
| Runtime | ms-swift/Transformers/PyTorch/CUDA 与必要 kernel versions | pending |
| Data | raw IDs 可复用；render/tokens/labels/token budgets 全部 v2 重建 | pending |
| Eval | scorer/task schema 可复用；旧 dev 降为诊断；新 v2 selection dev/confirmation 与 Base predictions | pending |
| Training | LoRA target、tiny-overfit、fresh-process resume、SFT promotion policy | pending |
| Capacity | load/infer/train peak VRAM、CPU RAM、disk、sequence cap | pending |

任一项未通过时，Day 15 停在 onboarding，不开始 packing/optimizer ablation。

## Week 2 Historical Gate

- [ ] SFT data contract、template 和 loss mask 有逐 token 证据。
- [ ] 训练数据可追溯，mixture 用 supervised tokens 报告，并完成 train/eval 去污染。
- [ ] Frozen Base baseline 早于训练且协议未静默变化。
- [ ] tiny overfit 证明 step pipeline 可学习并能恢复，但未被误写为能力提升。
- [ ] Controlled A/B 唯一变量成立；否则实验明确标记 invalid。
- [ ] selected checkpoint 的改善与退化都有逐样本证据和适用边界。
- [ ] 明确记录 Day 12 没有 eligible checkpoint，frozen test 未消费。
- [ ] Week 3 第一优先级固定为 v2 onboarding，不用堆 run 掩盖未知。

### 本周最强证据

### 被否决/降级的结论

### Week 3 唯一优先实验：Qwen3.5 v2 onboarding
