# Day 35 — Deferred Teacher Domain RL Readiness

状态：`deferred_unselected`
日期：`unscheduled_after_day30`
强度：当前 0 GPU；仅独立 teacher-extension charter v2 激活后执行

## 当前执行状态

本页保留未来 teacher domain-RL readiness 模板，当前不得执行，也不为 teacher 预留资源。活动 S2 的 coding environment/reward contract 直接继承 Day 24–29 与 Day 31，不依赖本页或任何 T1/T2。

## 主要目标

若未来 charter v2 激活，将已验证的 online RL contract 迁移到用户明确选择并完成 SFT 的 `T1`，在 teacher rollout 前冻结 environment、reward、placement、weight sync 和 promotion rule。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 35 — Domain RL Contract](../SCALING-BOOK-READING-GUIDE.md#day-35)。

## 工作内容

1. 将 capstone prompts 映射为 prompt/group/trajectory/turn/tool/environment schemas。
2. 冻结 deterministic reward components、错误码、timeout、幂等性与 raw execution evidence。
3. 构造 reward-hacking tests：只满足格式、冗长绕过、重复调用、伪造 observation、调用不存在工具。
4. 固定 rollout/train policy versions、old/ref/current log-prob、loss mask 与 stale threshold。
5. 解析已选择 teacher 的 actor training、rollout engine、reward workers 的 TP/DP/resource placement。
6. 预注册 `T2` selection，以及 exact S1 落盘后才能执行的 extension teacher-advantage gate。
7. 完成 rollout-only → reward replay → one-update → weight-sync → next-version rollout runbook。

## Teacher promotion rule

激活后先要求 T2 相对 T1 在 extension dev 有预注册提升；在 exact S1 可用后，再验证 T2 相对 S1 具有可迁移的新能力并完成最终 promotion。该 gate 属于 charter v2，不回写当前 Day 37。训练 reward 上升、格式更像或输出更长都不够。

## Evidence-first 产物

- `../artifacts/configs/capstone/day35-teacher-domain-rl/`
- `../artifacts/reports/capstone/day35-domain-rl-contract.md`
- reward/verifier tests、placement plan、teacher-promotion policy
- resolved CLI/config 与 1-rollout/1-update runbook

## 验收

- [ ] 独立 teacher-extension charter v2 与用户 model decision 已存在；否则本页保持 deferred。
- [ ] 激活后 reward 可从 raw trajectory 离线重算并通过 hacking tests。
- [ ] Environment/tool tokens 与 trainable assistant tokens 的 mask 明确。
- [ ] Actor/rollout/ref/reward 的版本与资源 ownership 明确。
- [ ] 已选择 teacher checkpoint 能被 rollout backend 加载，转换路径有 hash。
- [ ] GPU-hour cap 与每级停止条件已冻结；今天不启动未通过 dry gate 的长 run。
