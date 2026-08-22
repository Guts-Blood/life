# Day 38 — Deferred Teacher-trace Cold Start 与 Data Audit

状态：`deferred_unselected`
日期：`unscheduled_after_day30`
强度：当前 0 GPU；仅独立 teacher-extension charter v2 激活后执行

## 当前执行状态

当前没有 teacher model/revision 或 T2，本页不得执行、不得生成 traces、不得创建 S1d。活动 capstone policy charter v1 在 Day 37 后直接进入 Day 41；目录与本文保留为未来 extension 模板。

## 主要目标

若未来 charter v2 激活，从已 promotion 的 frozen `T2` 生成可追溯 teacher traces，并完成受限 off-policy cold-start ablation `S1d`。Extension Core OPD 仍从 exact S1 开始；只有后续 gate 证明分布不兼容时，才允许另开 `S1d -> S3d`。

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

- [ ] 独立 teacher-extension charter v2、用户 teacher decision 与 promoted T2 均存在；否则本页保持 deferred 且不产生 artifacts。
- [ ] 激活后 teacher traces 与 T2/prompt/environment 全链路可追踪。
- [ ] Eval references、frozen prompts 或 hidden verifier states 未进入 cold-start data。
- [ ] S1d budget、selection 和 guardrails 已预注册并执行；Core S3 仍声明从 S1 开始。
- [ ] 保留 S1；S1d 没有覆盖共同分叉点。
- [ ] 若 S1d 已达到 T2 或没有 OPD 学习空间，重新评估问题而非机械进入 OPD；若后续使用 S1d，结果命名为 S3d。
