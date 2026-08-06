# Day 35 — Domain RL Contract 与 Teacher Readiness

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时；CPU/config dry run 为主

## 主要目标

将 Day 24–29 的小模型 online RL contract 迁移到 `T1`，在开大规模 rollout 前冻结 domain environment、reward、资源 placement、weight sync 和 teacher promotion rule。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 35 — Domain RL Contract](../SCALING-BOOK-READING-GUIDE.md#day-35)。

## 工作内容

1. 将 capstone prompts 映射为 prompt/group/trajectory/turn/tool/environment schemas。
2. 冻结 deterministic reward components、错误码、timeout、幂等性与 raw execution evidence。
3. 构造 reward-hacking tests：只满足格式、冗长绕过、重复调用、伪造 observation、调用不存在工具。
4. 固定 rollout/train policy versions、old/ref/current log-prob、loss mask 与 stale threshold。
5. 解析 8B actor training、rollout engine、reward workers 的 TP/DP/resource placement。
6. 预注册 `T2` selection，以及 Day 37 在 S1 落盘后才能执行的 teacher advantage gate。
7. 完成 rollout-only → reward replay → one-update → weight-sync → next-version rollout runbook。

## Teacher promotion rule

Day 36 先要求 T2 相对 T1 在 domain dev 有预注册提升；S1 尚未产生，因此只能把 T2 锁定为 teacher candidate。Day 37 创建 S1 后再验证 T2 相对 S1 具有可迁移的新能力并完成最终 promotion。训练 reward 上升、格式更像或输出更长都不够。

## Evidence-first 产物

- `../artifacts/configs/capstone/day35-8b-domain-rl/`
- `../artifacts/reports/capstone/day35-domain-rl-contract.md`
- reward/verifier tests、placement plan、teacher-promotion policy
- resolved CLI/config 与 1-rollout/1-update runbook

## 验收

- [ ] Reward 可从 raw trajectory 离线重算并通过 hacking tests。
- [ ] Environment/tool tokens 与 trainable assistant tokens 的 mask 明确。
- [ ] Actor/rollout/ref/reward 的版本与资源 ownership 明确。
- [ ] 8B TP checkpoint 能被 rollout backend 加载，转换路径有 hash。
- [ ] GPU-hour cap 与每级停止条件已冻结；今天不启动未通过 dry gate 的长 run。
