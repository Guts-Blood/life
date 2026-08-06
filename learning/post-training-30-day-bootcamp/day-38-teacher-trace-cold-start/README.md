# Day 38 — Teacher-trace Cold Start 与 Distillation Data Audit

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；teacher generation + 小型 SFT

## 主要目标

从冻结 `T2` 生成可追溯 teacher traces，审计并完成一个受限 off-policy cold-start ablation `S1d`。Core OPD 仍从 S1 开始；只有 Day 39 证明分布不兼容时，才允许另开 `S1d -> S3d` 恢复路线。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 38 — Teacher-trace Cold Start](../SCALING-BOOK-READING-GUIDE.md#day-38)。

## 数据契约

每条 trace 至少包含：

- prompt/tool/environment manifest ID；
- teacher checkpoint/template/generation hashes；
- raw token IDs/text、tool calls、observations 与 final state；
- verifier components、status、length 与错误证据；
- acceptance/rejection reason；
- teacher trace artifact hash 和 parent prompt ID。

过滤不能只保留“格式漂亮”的答案。至少按 task success、tool family、difficulty、length 和 failure/recovery slice 审计，并保留被拒 trace 的统计。

## 实验

1. 从 train prompts 生成 teacher traces，不访问 dev/frozen references。
2. 人工审查和 deterministic verifier 共同形成 frozen accepted manifest。
3. 从 exact S1 进行短 cold-start SFT，预算在看 OPD 前固定。
4. 在 dev 上确认 `S1d` 没有破坏 guardrails，并测量它与 T2 的 token/support overlap。
5. 不把 S1d 当最终 distilled model，也不默认把它设为 S3 起点；它是 offline-KD baseline 和潜在 recovery checkpoint。

## Evidence-first 产物

- `../artifacts/data/capstone/teacher-traces-manifest.json`
- `../artifacts/reports/capstone/day38-teacher-trace-audit.md`
- `../artifacts/checkpoints/capstone/S1d-cold-start-manifest.json`
- rejected/accepted slice summary、S1→S1d dev comparison

## 验收

- [ ] Teacher traces 与 T2/prompt/environment 全链路可追踪。
- [ ] Eval references、frozen prompts 或 hidden verifier states 未进入 cold-start data。
- [ ] S1d budget、selection 和 guardrails 已预注册并执行；Core S3 仍声明从 S1 开始。
- [ ] 保留 S1；S1d 没有覆盖共同分叉点。
- [ ] 若 S1d 已达到 T2 或没有 OPD 学习空间，重新评估问题而非机械进入 OPD；若后续使用 S1d，结果命名为 S3d。
