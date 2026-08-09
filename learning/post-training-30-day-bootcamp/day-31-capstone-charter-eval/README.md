# Day 31 — Qwen3.5-4B Charter、Coding Eval 与版本冻结

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；CPU only

## 主要目标

把唯一活动 policy `S0 = Qwen/Qwen3.5-4B-Base@<immutable revision>`、coding task、数据和评测变成不可随结果漂移的实验 charter。model ID 已定；immutable revision 必须继承 Core Day 15 已验收的 revision/hash，今天只复核并写入 capstone artifact，不重新跟随官方仓 `main`。若 Day 15 未冻结成功，本日 blocked。Teacher/OPD 保持 `deferred_unselected`，不选择 8B、9B 或任何其他 teacher。

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

1. 复核并继承 Core Day 15 的 S0 exact immutable revision、config/weights、processor/tokenizer、loader class/version、modality 和 rendered-template hashes。
2. 定义唯一活动 checkpoint graph `S0 -> S1 coding SFT -> S2 direct coding RL`；`T0/T1/T2/S3/S1d/S3d` 全部写成 `deferred_unselected`，teacher model/revision 为 `null`。
3. 冻结 text-only coding training scope；记录 language model、vision tower、multimodal projector、MTP 与 adapter target 的 trainable/frozen/module-coverage contract。Day 32 前不替用户选择 full/LoRA/QLoRA。
4. 分开构造 `sft_train`、`policy_train`、`capstone_dev`、`capstone_frozen`；按 language/runtime/problem family/template 做 group split 与 overlap audit。Teacher probe/trace manifests 只保留 deferred schema，不生成数据。
5. 冻结 code extraction、sandbox/runtime、deterministic tests、timeout/error semantics、scorer 和 raw execution evidence。
6. 预注册 primary metric、guardrails、minimum meaningful difference、CI、selection 和 `inconclusive`。
7. 写活动 policy 预算：SFT/RL trained tokens、optimizer updates、rollout tokens 和 total GPU-hours；teacher cost 字段保持 deferred/null。
8. 用 20 条 golden coding prompts 冻结 rendered text、input IDs、attention/label mask、processor output schema 和 decode roundtrip hashes。

Day 10/12 的旧 dev 已驱动多轮选择；旧 frozen test 虽未消费，也绑定 v1 协议。今天必须新建 capstone dev/frozen manifests，不能把旧 manifest 改名复用；若要候选复用未消费 raw IDs，必须不查看内容并提供脚本化 crosswalk 与新 processor/render hashes。

## Evidence-first 产物

- `../artifacts/configs/capstone/capstone-charter.yaml`
- `../artifacts/data/capstone/{sft-train,policy-train,dev,frozen}-manifest.json`
- deferred teacher manifest slots（值为 `null`，不生成 teacher data）
- `../artifacts/reports/capstone/eval-protocol.md`
- `../artifacts/reports/capstone/selection-and-budget-policy.md`

## 验收

- [ ] S0 model ID 固定为 `Qwen/Qwen3.5-4B-Base`，immutable revision 与 Core Day 15 完全相同并写入；`main` 未被当成 revision。
- [ ] checkpoint graph 只有 S0/S1/S2 活动，且每条边只有一个声明的训练信号变化。
- [ ] dev/frozen group split、overlap report 与 hashes 可重建。
- [ ] Coding eval 覆盖 parse/compile、tests、runtime error、timeout、unsafe action 与 general guardrails。
- [ ] Processor/loader/modality/training-scope/module-coverage contract 完整；20 条 golden prompts 的 render/token/mask hashes 可重建。
- [ ] Scorer 可从 raw trajectory 离线重算，不依赖训练日志中的 aggregate。
- [ ] Frozen access policy 与 consumption record 路径已写明。
- [ ] `teacher_model_id/revision` 为 `null`，teacher/OPD 状态为 `deferred_unselected`，未启动 teacher 选型或训练。
- [ ] 未启动 GPU；Day 32 所需 Qwen3.5-4B model/processor/config 已缓存或有确定下载计划。

## Handoff

Day 32 只能使用今天冻结的 S0 revision、processor/loader、task manifests 和预算。发现根本 schema 错误时创建 policy charter 修订版，不覆盖旧 artifact；未来若选择 teacher，必须另建 teacher-extension charter v2，不能回填当前 v1。
