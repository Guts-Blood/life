# Day 24 — Online RL Dataflow 与 Reward/Verifier Contract

日期：`2026-08-19`
状态：`not_started`
强度：4–5 小时

## 主要目标

在使用框架前，能逐对象解释 online RL：prompt 如何变成 grouped trajectories、reward/advantage 和 log-prob 如何产生、训练后新权重如何进入下一轮 rollout。

## 理论（75 分钟）

精读清单：[Day 24 — Online RL dataflow](../SCALING-BOOK-READING-GUIDE.md#day-24)。

```text
prompt/data source
  -> policy rollout + environment/tools
  -> verifier/reward components
  -> filtering/grouping/advantage
  -> old/ref/current log-prob + loss mask
  -> optimizer update
  -> checkpoint/weight sync
  -> next policy version + eval
```

必须区分：

- prompt、completion、trajectory、turn/segment、group 和 training sample。
- raw reward、reward component、normalized reward、advantage。
- rollout/old policy、current train policy、reference policy 和 weight version。
- truncated、timeout、failed、aborted sample 的处理。

## Coding（120 分钟）

- 定义最小 trajectory schema：IDs、tokens/text、response span、loss mask、reward components、status、policy version、old/ref log-prob、metadata。
- 写 deterministic verifier contract：输入/输出、单位、错误码、timeout、版本、幂等性和 raw evidence。
- 用两个 prompts × 每组四条 completions 构造纯 CPU mini-pipeline，计算 group reward/advantage，并导出训练 batch。
- 写 contract tests：零方差 group、全失败 group、截断、格式对但答案错、重复调用一致、reward component 缺失。

## 实验（75–90 分钟）

- 对同一批 synthetic trajectories 比较 accuracy-only 与 accuracy+format 两个 reward。
- 展示 reward normalization 前后，以及 length/status slice。
- 有意构造一个“奖励升高但任务正确率不升”的 hack case。
- 画出每个 object 的 producer、consumer、持久化位置和 replay key。

## 资源与租卡

- CPU only；今天不启动 RL 训练。
- 可以复用 Day 23 模型输出，但不得为了生成更多样本租 GPU。

## Evidence-first 产物

- `../artifacts/data/trajectory-schema.json`
- `../artifacts/scripts/test_reward_contract.py`
- `../artifacts/reports/day24-online-rl-dataflow.md`

## 验收

- [ ] 能手工跟踪一个 trajectory 到训练 loss，字段不丢失。
- [ ] raw reward、normalized reward、advantage 没有混用。
- [ ] verifier 有版本、错误/timeout 语义和单元测试。
- [ ] 至少一个 reward-hacking case 被 frozen correctness metric 揭示。
- [ ] producer→schema→consumer 图可用于 Day 25/26 对照真实框架。

## Optional Capstone Trajectory Schema v2

保留 Day 24 RL schema v1，不覆盖。为 Day 31–42 fork [`../templates/opd-trajectory-schema-v2.json`](../templates/opd-trajectory-schema-v2.json)，在既有字段上增加可选：

- `student_rollout_checkpoint_hash/policy_version`；
- `teacher_key/checkpoint_hash/tokenizer_template_hash`；
- exact rendered input/token IDs 与 alignment hash；
- teacher sampled-token log-prob 或 top-k payload/hash；
- student old/current log-prob；
- KL/divergence estimator、direction、temperature/top-k/coefficient；
- `distillation_mask` 与 tool/environment spans；
- task reward 与 distillation signal 的不同字段；
- deterministic replay key、policy lag 和 teacher-serving status。

Day 24 Core 只需保证 schema 可向后扩展；不启动 teacher，不假定 slime v0.3.0 原生支持 OPD。

## Daily Log

### Object contracts

### Reward normalization

### Reward-hacking example

### Day 25 第一动作
