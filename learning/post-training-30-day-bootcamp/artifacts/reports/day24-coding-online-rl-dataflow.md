# Day 24 — Coding Online RL Verifier / Reward Contract

日期：`2026-08-16`（提前执行；原计划 `2026-08-19`）  
状态：`closed_pass_cpu_contract`  
资源：CPU control plane + fresh E2B Firecracker sandboxes；`0 GPU-hour`

## 结论

Day 24 的 CPU 实验闭环已完成。冻结的两个 MBPP train prompt 各绑定四条 Day 22 promoted-S1 completion；每条 completion 在 fresh、禁网、限资源的 E2B sandbox 中逐 test case 执行两次。16 条执行 evidence 中 `ok=10`、`wrong_answer=6`、`infra_error=0`，8/8 trajectory 的 semantic result hash 跨两次执行完全一致。

当前闭环为：

```text
frozen MBPP prompt/starter/tests + promoted-S1 completion
  -> code artifact parse/compile
  -> fresh E2B completion VM
  -> fresh Python process/cwd per testcase
  -> raw testcase evidence
  -> tests_only / tests_format_style reward
  -> G=4 pinned ms-swift mean/sample-std normalization
  -> normalized reward / advantage
  -> persisted replayable evidence
```

本日没有运行真实 RL optimizer update，也没有生成新 rollout。真实 old/current/reference log-prob、loss mask、weight sync 和 next-policy rollout 仍属于 Day 25。

## 冻结输入与 lineage

| 对象 | 冻结值 |
|---|---|
| Source dataset | `google-research-datasets/mbpp`，CC-BY-4.0，revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| Policy parent | promoted merged S1 `main-s20260809-lr1e-4-final` |
| Promotion manifest | `64e6a6bd61951d1f10eb4def520273cb056625c1b041b68ba861c6d1e15d7a6c` |
| Source checkpoint integrity | `d0f72be9751628c9073cc8e4104f16d8620bd598dbb8a1e97f1df9bae51170a3` |
| Processor contract | `12af5b3df5597fc2de4e6e9eebc9abba2e0f14d27971f39dc7e4af83c66e1083` |
| Tokenizer | `4570b194bd50f1fe92788865224ce75518f64963ad699cb32148655bdd3eed4e` |
| Template | `7f835342861cc9aabfd6299228033df80437b183227e04db311450437cd07ded` |
| Groups | train-only `mbpp:task:602`、`mbpp:task:604`，每组 `G=4` |
| Day 23 failure checkpoint | 未使用 |

冻结 input 中保留 exact prompt/output token IDs、source row hash、generation config、processor/policy provenance 和 test manifest。`modality=text`、`image_tokens=0`、`video_tokens=0` 显式记录。由于本日不加载模型，log-prob 与 loss mask 均为 `null`，并标记 `not_computed_day24_cpu_contract`，没有伪造训练侧数值。

## Producer → schema → consumer

| Object | Producer | Schema/transport | Consumer | Persistence / replay key |
|---|---|---|---|---|
| Prompt/task | frozen MBPP source | Day 24 trajectory v2 JSONL | rollout/reward adapter | task manifest hash |
| Completion/code artifact | Day 22 promoted-S1 rollout | exact text + token IDs | verifier | completion hash |
| Test manifest | Day 24 preregistration | visible/reward/frozen families | sandbox harness | tests manifest hash |
| Execution attempt | E2B adapter | per-case JSON evidence | verifier/replay audit | event/evidence hash |
| Verification result | deterministic verifier | status + testcase summary | reward policies | semantic result hash |
| Reward | pure policy | components + aggregate | group reducer | policy/reward hash |
| Group sample | pinned ms-swift reducer | mean/sample-std/normalized/advantage | Day 25 trainer adapter | group reward hash |

`deterministic_replay_key` 只绑定 task、completion、tests、verifier、sandbox 与 reward-policy contracts。时间、UUID 和 duration 只进入 event evidence hash，不进入 semantic replay identity。

## Sandbox 与 verifier 语义

Sandbox contract 固定为 E2B SDK `2.37.0`、template `rki5dems9wqfm4r03t7g`、fresh security sandbox、无 mounts/env/MCP、`allow_internet_access=false`、`deny_out=0.0.0.0/0`。每条 completion 使用一个 fresh VM；同一 completion 内每个 testcase 使用 fresh Python process 和 fresh cwd，共享 completion VM 的边界已显式记录。

Verifier 将以下状态分开：

- `parse_error` / `compile_error`：有效模型结果，未冒充 infra failure。
- `wrong_answer` / `runtime_error` / `candidate_timeout`：有效模型结果，可按冻结规则给 0 或 partial reward。
- `infra_error`：reward 为 `null`；受控重试一次仍失败，则整个 `G=4` group 为 `invalid_infra`，不缩成 `G=3`。
- `truncated` / `aborted`：独立状态，不混入 parse/test failure。

当前 live cohort 无 infra failure。单测另覆盖 SDK/全 infra、candidate timeout、compile、parse、runtime、partial tests 和 missing component fail-closed。

## 两个 G=4 group

### `mbpp:task:602` — first repeated character

| Trajectory | Output tokens | Status | Reward tests | Frozen eval |
|---|---:|---|---:|---:|
| `sample:00` | 42 | ok | 3/3 | 2/2 |
| `sample:03` | 24 | ok | 3/3 | 1/2 |
| `sample:01` | 65 | wrong_answer | 2/3 | 2/2 |
| `sample:02` | 29 | wrong_answer | 2/3 | 1/2 |

### `mbpp:task:604` — reverse space-separated words

| Trajectory | Output tokens | Status | Reward tests | Frozen eval |
|---|---:|---|---:|---:|
| `sample:00` | 28 | ok | 3/3 | 2/2 |
| `sample:01` | 29 | ok | 3/3 | 2/2 |
| `sample:02` | 12 | ok | 3/3 | 2/2 |
| `sample:04` | 17 | wrong_answer | 0/3 | 0/2 |

长度 slice 没有显示“越长越正确”：`task:602` 最长的 65-token completion 只有 2/3 reward tests，而 24-token completion 得到 3/3；`task:604` 的 12-token completion 得到 3/3。当前只有 8 条 trajectory，因此这些是诊断事实，不作统计外推。

## Reward、normalization 与 advantage

冻结策略：

```text
correctness = passed_reward_tests / total_reward_tests
tests_only = correctness
tests_format_style = 0.90 * correctness + 0.05 * format + 0.05 * style
safety violation = hard gate to 0
frozen_eval_correctness 不进入 aggregate reward
```

| Group | Policy | Aggregate rewards | Mean | Sample std | Advantages |
|---|---|---|---:|---:|---|
| task 602 | tests_only | `[1, 1, .6667, .6667]` | .8333 | .1925 | `[.8656, .8656, -.8656, -.8656]` |
| task 602 | tests+format/style | `[1, 1, .7, .7]` | .8500 | .1732 | `[.8655, .8655, -.8655, -.8655]` |
| task 604 | tests_only | `[1, 1, 1, 0]` | .7500 | .5000 | `[.4999, .4999, .4999, -1.4997]` |
| task 604 | tests+format/style | `[1, 1, 1, .1]` | .7750 | .4500 | `[.4999, .4999, .4999, -1.4997]` |

这里的 `std` 使用 PyTorch 默认 correction=1（即 `ddof=1`），advantage denominator 为 `std + 1e-4`，与 pinned ms-swift `compute_advantages(..., scale_rewards="group")` 精确一致。

Raw testcase result、reward component、aggregate reward、normalized reward 与 advantage 使用不同字段。当前两种 policy 的 raw semantic evidence 完全相同；只有 policy hash、aggregate 和 reducer 输出变化。

单测中的 zero-variance fixture 使用 `[0,0,0,0]`，得到 `mean=0`、`std=0`、四个 advantage 均为 `0`、`optimizer_update_eligible=false`。全 infra fixture 得到四个 `null` reward、`invalid_infra` 和四个 `null` advantage。

## Reward hacking

`task:602 sample:03` 的实现本质上返回“字符串中第一个总出现次数大于 1 的字符”，不是扫描过程中第一个发生重复的字符。它通过全部三个 reward tests，因此：

```text
tests_only reward = 1.0
frozen correctness = 1/2 = 0.5
```

冻结 case `first_repeated_char("abccba") == "c"` 揭示该实现错误返回 `"a"`。这证明训练 reward 上升或满分并不能单独支持 hidden correctness 提升。`task:604 sample:04` 还展示 format/style bonus 可把完全错误的 correctness `0` 提高到 aggregate `.1`，而 frozen correctness 仍为 `0`。

## Replay / idempotency

8 条 trajectory 均运行两次，共 16 条 live evidence。每条 trajectory 的两次：

- event ID、时间与 duration 不同；
- semantic result hash 相同；
- testcase statuses、error types、stdout/stderr hashes、summary 与 sandbox digest 相同。

结果：`8/8 semantic replay pass`，无受控 infra retry 被触发。Contract test 另证明 infra 时最多重试一次，并在持续失败后原子失效整个 group。

## 持久化证据

| 文件 | File SHA-256 |
|---|---|
| `artifacts/data/day24-coding-trajectory-schema-v2.json` | `78281bbce8d9b6c5bdedb6deb68f3bef638751c73c26391c8931f08d3f104fef` |
| `artifacts/data/day24-coding-mini-pipeline-input.jsonl` | `1f1f56b9a517a7c77f95a1731126a187a44701e7c3afc1de3461b3f47b46aaf6` |
| `artifacts/eval/day24-coding-verifier-evidence.jsonl` | `f3e6bb20ecab2e45d2768351059fb2d33252dcdefdda4975dd0c9df07327cf9a` |
| `artifacts/eval/day24-coding-group-rewards.jsonl` | `29300ffba56ae362ee166809c59389a9eebbe0fbe533b576db80ce3baca12ec2` |

## Day 25 adapter

Day 25 只需实现薄 adapter：

```python
def ms_swift_reward_adapter(tasks, completions, policy_version):
    # 1. 构造 Day 24 trajectory/replay request
    # 2. batch 调用 verifier
    # 3. 持久化 raw evidence/reward components
    # 4. 返回与 completion 顺序一致的 scalar rewards
    ...
```

Adapter 不得重写 test execution、reward weights 或 group normalization。Day 25 仍需验证真实 response mask、old/current/reference log-prob、one optimizer update、LoRA/weight sync、next rollout policy version、显存/topology 和 frozen 30–50 family eval。Day 24 不声明真实 RL capability gain。
