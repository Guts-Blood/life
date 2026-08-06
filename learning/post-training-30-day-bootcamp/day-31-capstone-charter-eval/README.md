# Day 31 — Capstone Charter、Domain Eval 与版本冻结

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；CPU only

## 主要目标

把“8B SFT+RL teacher → <=4B OPD student”从愿望变成不可随结果漂移的实验 charter。默认领域为可执行 tool calling；若替换领域，必须在今天完成并冻结理由。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 31 — Capstone Charter](../SCALING-BOOK-READING-GUIDE.md#day-31)。

## 输入

- Day 08 tokenizer/template/loss contract；
- Day 09 lineage/decontamination 方法；
- Day 10/21 eval schema、selection policy 与 consumed-set 记录；
- Day 24 verifier/trajectory schema；
- Day 30 post-training design 与未决问题。
- [`../templates/capstone-charter.yaml`](../templates/capstone-charter.yaml)、checkpoint promotion 与 OPD trajectory schema 模板。

## 工作内容

1. 固定 teacher/student 的 exact model、tokenizer、config revisions；Core 优先同一家族、共享 vocabulary。
2. 定义 `T0/T1/T2/S0/S1/S2/S3` Core checkpoint graph，以及可选 `S1d/S3d/S4`、每条边的 objective 和唯一自变量。
3. 分开构造 `sft_train`、shared `policy_train`、`teacher_advantage_probe`、`trace_cold_start`、`capstone_dev`、`capstone_frozen`；按 tool/scenario/template family 做 group split 与 overlap audit。
4. 冻结 E2E scorer、environment、timeout/error semantics、tool schemas 和 raw execution evidence。
5. 预注册 primary metric、guardrails、minimum meaningful difference、CI、selection 和 `inconclusive`。
6. 写 three-budget accounting：student updates、student rollout tokens、total GPU-hours；teacher preparation 单列。

如果 Day 10/12 的 frozen set 已揭盲或驱动过决策，今天必须新建 capstone frozen set，不能改名复用。

## Evidence-first 产物

- `../artifacts/configs/capstone/capstone-charter.yaml`
- `../artifacts/data/capstone/{sft-train,policy-train,teacher-advantage-probe,trace-cold-start,dev,frozen}-manifest.json`
- `../artifacts/reports/capstone/eval-protocol.md`
- `../artifacts/reports/capstone/selection-and-budget-policy.md`

## 验收

- [ ] checkpoint graph 的每条边只有一个声明的训练信号变化。
- [ ] dev/frozen group split、overlap report 与 hashes 可重建。
- [ ] Tool call 至少覆盖 call/no-call、wrong tool、invalid args、execution 和 multi-turn recovery。
- [ ] Teacher/student tokenizer JSON、vocab/token-ID map、special tokens、template hashes 相等，20 条 golden prompts 的 token IDs 完全一致。
- [ ] Scorer 可从 raw trajectory 离线重算，不依赖训练日志中的 aggregate。
- [ ] Frozen access policy 与 consumption record 路径已写明。
- [ ] 未启动 GPU；Day 32 所需模型/config 已缓存或有确定下载计划。

## Handoff

Day 32 只能使用今天冻结的 model revisions、task manifests 和预算；发现根本 schema 错误时创建 charter v2，不覆盖 v1。
