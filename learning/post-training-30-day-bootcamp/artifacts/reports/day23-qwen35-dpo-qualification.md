# Day 23 Qwen3.5 Coding DPO — Qualification Closeout

执行日期：`2026-08-14`

最终状态：`TERMINAL_DEV_NOT_QUALIFIED`

结论：成功训练并保存了一个 DPO checkpoint，但它没有通过冻结的一次性 dev gate，因此不是合格 candidate。

## Lineage

本轮没有复用旧 SFT RSI 的版本号：

- 旧 SFT：`goal-0001`，`rsi-v0001` / `rsi-v0002`
- Day 23 DPO 探索：`goal-0002`，`rsi-v0003` → `rsi-v0005`
- 隔离 candidate search：`goal-0003-day23-dpo-candidate-search`，`csearch-v0001`
- 唯一 finalist qualification：`goal-0004-day23-dpo-qualification`，`qual-v0001`

`csearch-v0001` 冻结的 winner 是 `learning_rate=4.3e-6`、checkpoint step `20`。Qualification 从 promoted merged S1 fresh start，未 resume search checkpoint。

## Full-154 refit

- authoritative root：`/root/autodl-tmp/runs/day23-qwen35-dpo-qualification-v0001-20260814T130800Z`
- run：`full154_refit_lr_4p3e_6_step_20`
- data：frozen train `154` pairs
- runtime：2×RTX PRO 6000，DDP world size `2`
- batch：per-device `8`，gradient accumulation `2`，nominal global batch `32`
- schedule：cosine horizon `30` steps；step `20` 是唯一 candidate，step `30` 仅保留为 noncandidate
- seed / data seed：`20260820`
- training：`30/30` steps 完成，约 `59.48s`，train loss 约 `0.6917`
- peak memory：约 `44.99 GiB`；未触发 `20%` free-memory gate

Candidate checkpoint：

`/root/autodl-tmp/runs/day23-qwen35-dpo-qualification-v0001-20260814T130800Z/outputs/full154_refit_lr_4p3e_6_step_20/checkpoint-20`

- success receipt self SHA-256：`401864224ace8e135f4b8e602e2a3b89c472099a5558ef6f53738089dbaec0c9`
- checkpoint files aggregate SHA-256：`44fea0d4508124ca15f9dd4161078a45dd143ac716aa191b23c9e64069aebd98`
- adapter SHA-256：`9d9df99a16b7eaeb2ee8697317df9c764ed756b479f24730fc7b6b86e84ce676`

训练、LoRA 更新、parent freeze、reference disable-adapter、DDP 与 checkpoint evidence 均通过。该 checkpoint 是有效训练产物，但资格取决于后续 dev gate。

## One-shot dev result

Dev 共 `17` pairs，冻结门同时要求：

- positive reward-margin pairs `>=13/17`
- mean reward margin `>0`
- length-matched mean reward margin `>0`
- `17/17` finite

实际结果：

| Metric | Observed | Gate | Result |
|---|---:|---:|---|
| Positive pairs | `8/17` | `>=13/17` | FAIL |
| Mean reward margin | `-0.0007107868` | `>0` | FAIL |
| Length-matched mean reward margin | `-0.0022532248` | `>0` | FAIL |
| Finite rows | `17/17` | `17/17` | PASS |

- dev evaluation SHA-256：`6d5b3f447b875c6a1174b4a93f3e0419144c9d1308dba9dbb21f3eb4bbbb6020`
- global dev claim SHA-256：`6ea2231d659f34f0c548a4e437c73b0d299ebbaf20616d9cc8571d243085b6e1`
- `heldout_consumed=false`

这是科学门失败，不是训练、OOM、checkpoint reload 或 evaluator 数值故障。Search winner 没有迁移到独立 dev：结果支持“当前 search/recipe 的泛化不稳定”，但仅凭聚合指标不能把根因进一步归因到 LR、seed 或 pair composition。

## Stop boundary

Dev 是一次性全局 lease，已经消费。按冻结协议：

- 不重跑 dev，不换 runner-up，不在同一证据上继续调参；
- 不运行 full112、E2B 或 heldout；
- 不把 checkpoint-20 宣称为 qualified/deployment-ready candidate；
- checkpoint-20 保留为失败 finalist 和诊断产物，不作为本协议的晋级结果。

要继续追求合格 candidate，必须使用新目标和新的独立 validation data；不能把已观察的 dev 或尚未授权的 heldout 改作调参集。当前实例已关机，GPU 占用为零。

CPU-only append-only closeout controller 已实现并通过测试；由于远端实例在写入 event 之前关机，goal-0004 的 terminal event 尚待在远端存储重新可访问时同步。该归档缺口不改变上述 checkpoint 或 dev 科学结论。

## 下一轮的最小有效入口（尚未授权训练）

本次失败暴露的是 search winner 到独立 seed/full154/dev 的迁移不稳定，而不是 GPU 容量问题。下一轮在重新开卡前至少需要：

1. 从远端 receipt 做逐 pair/family/length 诊断；在拿到该证据前不预判 LR、seed 或 pair weighting 是根因。
2. 使用已冻结的新 blind validation pool：Day22 的 60 个 `both_fail` MBPP family 从未进入旧 200 个 Day23 preference pairs；它们已按固定 hash 分成 dev30/heldout30。已消费的 dev17 不得再用于选择。
3. 每个 recipe 从 promoted S1 fresh start；失败 finalist checkpoint 不作为 resume 或 parent。
4. 在打开新 dev 前先要求至少两个独立训练 seed 都通过同一 train-only/search 稳定性门；只允许一个预注册 finalist 进入新 dev。
5. 训练拓扑保持已实证的 world2/B8/GA2。它在约一分钟内完成 30 steps 且有充足显存余量；在获得新的科学证据前，不把 batch/topology 与科学 lever 同时改变。

CPU preregistration 已进一步把第 4 项变成可执行硬门：历史
`csearch-v0001` 的 seed `20260819` / `4.3e-6` / step `20` PASS 会在远端
按 campaign、selection、selected-evaluation 原字节重新验证；新 seed
`20260820` 训练后必须先在同一份已观察过、非盲的 search30 上达到
`>=20/30` 且两项 margin 均为正。只有两份独立 seed evidence 都 PASS，
evaluator 才能创建新 blind dev30 的一次性 claim。这样仍只新增一条训练
trajectory，但不会把不稳定 checkpoint 直接送进新的盲集。

因此当前真正缺口是新的盲验证证据和基于逐 pair 诊断选出的单一干预，不是继续占用 GPU 重跑现有 recipe。

新 blind pool 的 manifest 是 [`day23-qwen35-dpo-next-blind-manifest.json`](../data/day23-qwen35-dpo-next-blind-manifest.json)，content SHA-256 为 `762fb8c6aa97fbf91e438ceb5e165cfb661e505c14d20dcb1ed1fe44b41f48e2`。每个 chosen 来自冻结的 MBPP canonical continuation；每个 rejected 都绑定两次稳定 `wrong_answer` sandbox evidence。它与旧 200 个 preference family 的 overlap 为 `0`，当前 `candidate_metrics_computed=false`，且训练使用被禁止。

Pinned Qwen3.5 processor 已在不加载权重的情况下对这 60 对/120 branches 做真实 RLHF 编码：input 长度 `83–319`、response 长度 `14–230`、`max_length=512`，零 truncation、零 prompt-prefix/mask/four-space/ChatML-tail/decode failure。审计见 [`day23-qwen35-dpo-next-blind-processor-audit-summary.json`](../eval/day23-qwen35-dpo-next-blind-processor-audit-summary.json)。

下一候选 CPU contract：[`cpu-contract.json`](../configs/day23-qwen35-dpo-next-candidate/cpu-contract.json)，content SHA-256
`bdd80ee81ec20974bdb2d47df06f2a9239ac4bf0c5a0577c1faab992b5ebf540`。
完整的下一次开卡命令与 stop gates 见
[`day23-next-candidate-runbook.md`](day23-next-candidate-runbook.md)。
