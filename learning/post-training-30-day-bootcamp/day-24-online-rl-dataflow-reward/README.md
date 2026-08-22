# Day 24 — Coding Online RL Trajectory 与 Reward/Sandbox Contract

日期：`2026-08-19`
状态：`closed_pass_cpu_contract`（`2026-08-16` 提前执行）
强度：4–5 小时

## 主要目标

在调用 RL 框架前，逐对象解释 coding online RL：prompt 如何变成 grouped code trajectories，sandbox/tests 如何产生可重放 reward，log-prob/advantage 如何构造，训练后新 Qwen3.5 policy 如何进入下一轮 rollout。Day 01–12 的 prompts/bad cases 可作历史选题线索，但旧 generations、token IDs、policy hashes 和 trajectories 不进入 v2 contract。

## 理论（60 分钟）

精读清单：[Day 24 — Online RL dataflow](../SCALING-BOOK-READING-GUIDE.md#day-24)。

```text
frozen coding prompt + tests
  -> Qwen3.5 SFT policy rollout
  -> parse/build/run in versioned sandbox
  -> reward components + raw execution evidence
  -> filtering/grouping/advantage
  -> old/ref/current log-prob + response/tool/environment masks
  -> LoRA optimizer update
  -> checkpoint/weight sync
  -> next policy version + frozen coding eval
```

必须区分 prompt、completion、code artifact、tool call、execution attempt、trajectory、group 和 training sample；也必须区分 raw test result、reward component、normalized reward、advantage，以及 rollout/old/current/reference policy。

## Trajectory Schema（90 分钟）

最小 schema 至少包含：

- run/trajectory/group/prompt/completion IDs 与 deterministic replay key；
- Qwen3.5 Base/SFT-parent/current policy hashes，processor/tokenizer/template hash，rollout backend/version；
- exact rendered input、input/output token IDs、response span 与 loss mask；
- code language、files/patch、tool/environment spans；
- sandbox image/environment/test manifest hashes、timeout、exit code、stdout/stderr 与 raw test cases；
- parse/compile/test/style/safety reward components、aggregate reward、normalization、advantage；
- old/ref/current log-prob、policy/weight version、generation config；
- `ok/truncated/timeout/compile_error/runtime_error/infra_error/aborted` 等 status。

Core 固定为 text-only coding；`modality=text` 和无 image/video tokens 必须显式记录。不要因为模型有视觉模块就省略 processor/model-class provenance。

## Reward/Sandbox Contract（75 分钟）

- 定义 deterministic verifier：输入、输出单位、错误码、timeout、版本、幂等性、raw evidence 与隔离边界。
- `infra_error/timeout` 不得自动等同于代码错误；invalid parse、compile fail、partial tests 与 full pass 使用不同字段。
- 用两个 prompts × 每组四条 completions 构造 CPU mini-pipeline，重放测试并计算 group reward/advantage。
- Contract tests 覆盖：零方差 group、全 infra failure、截断、格式正确但测试失败、只过公开样例、重复调用一致、reward component 缺失。

## Reward Hacking 实验（45–60 分钟）

- 对同一 trajectories 比较 tests-only 与 tests+format/style 两个 reward 组合。
- 展示 normalization 前后、length/status/test-family slices。
- 构造至少一个“reward 升高但 hidden correctness 不升”的 case，例如硬编码公开样例、吞掉异常或输出格式投机。
- 画出每个 object 的 producer、schema、transport、consumer、持久化位置与 replay key。

## 资源与租卡

- CPU only；今天不启动 RL 训练，也不为了补 trajectories 租 GPU。
- 可以复用 Day 23 已保存的 v2 outputs，但必须复制到新 trajectory namespace 并记录 parent hash。
- Sandbox 只运行不可信代码于隔离环境，不允许直接在训练主进程执行。

## Evidence-first 产物

- `../artifacts/data/day24-coding-trajectory-schema-v2.json`
- `../artifacts/scripts/test_day24_coding_reward_contract.py`
- `../artifacts/reports/day24-coding-online-rl-dataflow.md`

## 验收

- [x] 能从一个 coding trajectory 手工追到 sandbox evidence、reward、advantage 和训练 loss 的输入边界；真实 log-prob/loss 明确留给 Day 25，未伪造。
- [x] Raw result、reward component、aggregate/normalized reward 与 advantage 没有混用。
- [x] Processor/model/policy/sandbox/test versions 与 replay key 完整。
- [x] Infra failure 与 wrong answer 区分，verifier 有 candidate timeout、infra retry、全 infra 与幂等单测。
- [x] `mbpp:task:602 sample:03` 的满 reward 被 frozen correctness `1/2` 揭示为 weak-test false positive。
- [x] producer→schema→consumer 图、Day 25 薄 adapter 接口与持久化位置已落入报告。

## Daily Log

### Object / policy contracts

- 冻结 train-only `mbpp:task:602`、`mbpp:task:604`，每组四条 promoted-S1 completion；Day 23 failure checkpoint 未使用。
- Trajectory v2 保存 exact token IDs、processor/tokenizer/template、S1 parent、test/sandbox/verifier/reward hashes；text-only modality 显式记录。
- Day 24 CPU contract 不计算 old/current/reference log-prob 或 loss mask；字段为 `null` 并带 `not_computed_day24_cpu_contract`。

### Sandbox / status semantics

- 新增 `day24_coding_verifier.py`，不修改冻结的 Day 22 scorer。每条 completion 使用 fresh、禁网 E2B VM；每个 testcase 使用 fresh Python process/cwd。
- `parse/compile/wrong_answer/runtime/candidate_timeout` 为模型结果；`infra_error` reward 为 `null`，受控重试一次仍失败则整个 G=4 group invalid。
- Day 22 原测试 `13/13` 通过；Day 24 contract tests `22/22` 通过（常规 run 跳过 live class），显式 live E2B integration `1/1` 通过。

### Reward normalization / hacking

- Live mini-pipeline：8 trajectories × 2 replays = 16 evidence；`ok=10`、`wrong_answer=6`、`infra_error=0`。
- `tests_only` 与 `tests_format_style` 共用 raw evidence；G=4 与 pinned ms-swift 对齐，使用 sample std (`ddof=1`) 且 normalization denominator 加 `1e-4`。Zero-variance fixture 产生全零 advantage；全 infra fixture 不产生 advantage/update eligibility。
- `task:602 sample:03` reward tests `3/3`、frozen eval `1/2`；`task:604 sample:04` 的 style/format bonus 将 aggregate 从 `0` 提到 `.1`，frozen correctness 仍为 `0`。

### Replay key / version map

- 8/8 trajectories 的两次 semantic result hash 一致；event hash 保留 UUID/time/duration 差异。
- 产物：trajectory schema、8-row frozen input、16-row verifier evidence、4-row group rewards，以及 `artifacts/reports/day24-coding-online-rl-dataflow.md`。

### Day 25 第一动作

实现 pinned ms-swift 的薄 reward adapter：只做 completion/task 转换、batch verifier 调用、evidence 持久化和 scalar reward 返回；不得在 adapter 重写 tests、reward weights 或 group normalization。随后先完成 rollout-only G1，再进入 one-update G2。
