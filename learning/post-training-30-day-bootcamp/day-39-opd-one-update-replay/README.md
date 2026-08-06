# Day 39 — OPD One-update、Teacher Scoring 与 Replay Gate

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；多卡短 smoke

## 主要目标

在 Core 起点 `S1` 上完成一条可手工追踪的 OPD update：student 自己 rollout，冻结 T2 在 student-visited prefixes 上评分，distillation loss 更新 student，并能从保存 batch 重放。只有明确记录的分布不兼容 failure 才允许另开 `S1d` recovery run。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 39 — OPD One-update](../SCALING-BOOK-READING-GUIDE.md#day-39)。

## OPD 最小证据链

```text
prompt + student policy version
  -> student rollout token IDs / response mask
  -> exact teacher input prefix/token alignment
  -> teacher log-prob or top-k distribution
  -> student old/current log-prob
  -> divergence / token advantage
  -> distillation loss mask
  -> optimizer update
  -> new student policy version
```

环境/tool observation tokens 默认不参与 student loss；若实现不同，必须在 charter v2 中声明并用 token trace 证明。

## Gate 顺序

1. `G0 placement`：student train、student rollout、teacher scoring 和 environment workers 的 GPU bundles 正确。
2. `G1 same-model sanity`：条件允许时以 teacher=student 做近零信号 smoke，记录 train/inference engine 数值差异。
3. `G2 teacher scoring`：单条 student rollout 的 teacher token alignment 和 log-prob 可人工核查。
4. `G3 one update`：KL direction/temperature/top-k、stop-gradient、mask 和 reduction 与 resolved implementation 一致。
5. `G4 replay`：从持久化 batch 重算 teacher/student quantities 和首步 loss；说明 exactness。
6. `G5 version`：下一次 rollout 明确使用更新后的 student version，teacher version保持不变。

若 Core S1 gate 失败而 S1d gate 成功，后续 checkpoint 命名为 `S3d`，并且若要声称 OPD objective 优于 direct RL，必须从 S1d 另跑 matched direct-RL control。

## 必存字段

- prompt/trajectory/group IDs；
- student rollout policy version 与 checkpoint hash；
- teacher key/checkpoint/tokenizer/template hash；
- student/teacher token log-probs 或 top-k payload；
- token overlap/mass diagnostics（实现支持时）；
- KL estimator/direction、temperature、coefficient、task-reward coefficient；
- response/distillation/tool/environment masks；
- loss、grad norm、policy version 和 runtime status。
- rollout policy lag；Core 要求 teacher scoring/update 前的 student batch 为当前版本，跨阶段最大 `policy_lag <= 1`。

## Evidence-first 产物

- `../artifacts/logs/capstone/day39-opd-one-update/`
- `../artifacts/reports/capstone/day39-opd-gates.md`
- `../artifacts/data/capstone/opd-replay-batch/`
- resolved configs、placement map、teacher/student version timeline

## 验收

- [ ] Student rollout 而非 teacher trace 驱动了本次 on-policy batch。
- [ ] Teacher/student token alignment 与 loss mask 可逐 token 审计。
- [ ] Teacher 全程冻结，student 确实完成 optimizer update。
- [ ] 保存 batch 可 replay；线上/离线差异有容差和原因。
- [ ] 未把 task reward、teacher advantage 与 normalized RL advantage 混成同名字段。

## Stop Conditions

出现 tokenizer mismatch、teacher signal 全零/爆炸、mask 错位、teacher 被更新或 replay 不可解释时停止，不启动 Day 40。
