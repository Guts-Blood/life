# Day 23 Qwen3.5 Coding DPO — GPU Gate Closeout

执行日期：`2026-08-14`（提前执行原计划 Day 23）

状态：`CLOSED_NO_CANDIDATE`（terminal mechanism NO-GO）

轨道：`experimental_ai_assisted`；不宣称 formal human-reviewed readiness

## 结论

Day 23 已按 fail-closed pipeline 结束，**没有可晋级的 DPO checkpoint**。

远端 promoted S1、双卡 DDP、fresh LoRA / frozen reference、一步 optimizer、显存、checkpoint 保存和独立新进程重载均通过。随后两个预注册的 5-step mechanism 尝试都只让 `1/4` pair 的 chosen reward margin 改善，未达到 `>=3/4` 的硬门。因此没有启动 30-step bounded smoke，没有进行 dev selection，也没有读取 preference heldout 或运行 post-selection coding guardrails。

这不是 OOM、runtime、DDP 或冻结范围故障。两次训练均完成 5 steps、LoRA 确实变化、所有数值 finite、冻结参数未变化、显存余量充足。失败发生在训练完成后的逐 pair 方向门。

## Gate 结果

| Gate | 结果 | 退出状态 |
|---|---:|---|
| G0 remote S1 payload / runtime | PASS | promotion、key、merged export、checkpoint、parity 与 runtime hashes 全部匹配 |
| G1/G2 dual-GPU one-step | PASS | 2×GPU、fresh LoRA、disabled-adapter reference、精确 trainable/frozen inventory、一步 finite |
| G2 save/reload | PASS | checkpoint-1 保存；独立新进程逐键重载与 log-prob parity 通过 |
| G3 attempt 1：2×GPU B2 / GA1 | FAIL | mean improvement 为正，但仅 `1/4` pair 改善 |
| G3 attempt 2：2×GPU B1 / GA4 | FAIL | one-shot topology correction；仍仅 `1/4` pair 改善 |
| G4 fresh 30-step smoke | NOT RUN | 被 terminal G3 gate 阻断 |
| G5 dev / heldout / guardrails | NOT CONSUMED | 无 candidate；heldout 保持未读、未消费 |

## Remote parent 与运行身份

- S1 checkpoint ID：`qwen35-4b-s1-main-s20260809-lr1e-4-final`
- S1 downstream key：`s1:qwen35-4b:c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a`
- merged export files digest：`16bc212df51ecac9a5b3b34062e0ccc880c719d63248f04949be8636550e53be`
- ms-swift checkout：`565a1ad586a21d24b23931c52d2c62b49c39bee8`
- GPU：2× NVIDIA RTX PRO 6000 Blackwell Server Edition
- learner runtime：Python `3.12.13`、torch `2.10.0+cu128`、Transformers `5.12.1`、PEFT `0.19.1`、TRL `0.29.1`
- authoritative run root：`/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T094805Z`
- binding content SHA-256：`1582776498fcc490fd8c67dc03778a436028e90f51598c5121802fb55e76f2bd`
- binding file SHA-256：`3fd205b024853192bdd3d3f82ac7db67f7c4f94fea22fbd1e09870d6e8fd2c54`

Policy 与 reference 均来自同一个 merged S1。Policy 是 fresh LoRA；reference 在同一个 PEFT model 上通过 `disable_adapter()` 回到冻结的 merged S1。Base、Day 01–12 权重和 resumable S1 checkpoint 都没有作为 Day 23 model input。

## G1/G2 runtime evidence

- Trainable inventory：`496` LoRA tensors，`16,232,448` params；optimizer inventory 精确相等。
- Frozen inventory：`723` tensors，`4,539,265,536` params；vision、aligner、embedding 与其余 LLM parent 全部版本不变。
- Reference/policy forward 均 finite；reference context 实际进入 disabled-adapter 路径。
- G2 LoRA digest 从 `d3c8ba84...` 变为 `a9cd1f24...`，证明 optimizer step 实际更新。
- G2 最低 observed free fraction：`87.06%`；硬门为 `15%`。
- G2 success receipt content SHA-256：`3e4a0d494f26e09429466964567cfadedeb07cda60d0437d09075fc4a2370066`
- G2 success receipt file SHA-256：`58b982df1d40522bd0f55397f054da22a57ac8c80a1305ac34a1050bcccde42f`
- G2 fresh reload evaluation SHA-256：`c90eb42ed06d34e1e537f3888fd8834dd69d2c8c061611cefbb046576e0be529`
- G2 reload file SHA-256：`d583fdc94675483226c2619262b196b16a4de07f672aff86660e76c7f165d642`

G2 checkpoint 的独立 evaluator 是新进程；它重新加载 merged S1 与保存的 adapter，核对完整 LoRA tensor inventory，并重新计算 policy/reference response-only log-prob。`fresh_process_reload_proven=true` 与 `adapter_reload_proven=true`。

## G3 mechanism 结果

冻结判据同时要求：

1. 四条 pair 的 mean reward-margin improvement `> 0`；
2. 至少 `3/4` pair 的 improvement `> 0`；
3. loss、reward、log-prob、grad 全 finite；LoRA 更新；parent/reference 不变；显存门通过。

两次尝试的逐 pair 结果如下。数值均为 `post_train_margin - pre_train_margin`，`beta=0.1`。

| Pair | Attempt 1：B2 / GA1 | Attempt 2：B1 / GA4 |
|---|---:|---:|
| `mbpp:task:602:s1pair:37673b32b9385aea` | `-0.00338554` | `-0.00447578` |
| `mbpp:task:604:s1pair:388a8daec7e97c5a` | `-0.00648251` | `-0.00099678` |
| `mbpp:task:605:s1pair:9ae278fe596ed4d4` | `-0.00177670` | `-0.00314293` |
| `mbpp:task:610:s1pair:9e57fb63f7544777` | `+0.01838398` | `+0.03027897` |
| Mean | `+0.00168481` | `+0.00541587` |
| Improved | **`1/4`** | **`1/4`** |

Attempt 1 与 attempt 2 都通过 mean 门，但都未通过 `>=3/4` 方向一致性门；两次都是前三条退化、只有 task610 改善。B1 / GA4 的 topology correction 没有修复这个模式，所以结果按预注册解释为 topology-sensitive correction 也失败，而不是把 attempt 1 覆盖掉。

Attempt 2 的其余运行证据正常：

- `global_step=5`，`train_loss=0.69078096`；5/5 steps 完成。
- configured GA 为 4；4-row/world-size-2 下每 rank 每 epoch 实际有两个 microbatches，receipt 记录 `current_gradient_accumulation_steps=2` 和 10 次 training-step call。
- LoRA digest 变化；optimizer 仍精确覆盖 496 tensors。
- `all_versions_unchanged=true`，`version_changes=[]`，`violations=[]`，无 nonfinite。
- recorded rank 的最低 free fraction 为 `87.53%`，peak reserved 约 `10.94 GiB`，显存不是失败原因。
- checkpoint-5 的 11 个文件与失败 receipt 中的 manifest 完全匹配，但 checkpoint 仅是失败附属物，禁止 candidate/resume。

Attempt 2 failure receipt 由 rank 1 在硬门异常后抢先以 O_EXCL 封存；rank 0 随后的重复写入被拒绝。这符合不可覆盖设计，不是证据链损坏。由于异常发生在 all-gather 前，receipt 的 probe 是 rank-1 local evidence；不能把它表述为“两 rank probe 数值完全一致”。全 rank 必须通过，而一个 rank 已明确违反方向门，足以触发 terminal NO-GO。

## Immutable failure chain

Attempt 1（保留，不覆盖）：

- path：`/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T091033Z/evidence/mechanism-5step/failure-receipt.json`
- receipt content SHA-256：`600341cb77ee2bd97b2e4a00ae26a89720a9204b8266b86b6571c6c3181adc20`
- file SHA-256：`cf5def99380c5dde0f095d9b574a054fde1b6bf2805206435fef3de1ff199770`

Attempt 2 one-shot claim：

- path：`/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T091033Z/evidence/mechanism-5step/topology-correction-attempt2-claim.json`
- claim content SHA-256：`7044149a82a1b2db899765cf3448f35b85637aae8ffa3dfe15f43bff3eee0807`
- file SHA-256：`2878ad139415505785f654cdee813a340042f5e18780ad995f1c35fb677ccfb5`

Attempt 2 failure：

- path：`/root/autodl-tmp/runs/day23-qwen35-dpo-20260814T094805Z/evidence/mechanism-5step/failure-receipt.json`
- receipt content SHA-256：`090bc75fbbf5eb1eea905492c328db7dda07d6b5a0c1786b5a79015fea15a3d2`
- file SHA-256：`cf288dd9bc76652a67c1bcb099317ab4c91d2acb4d5c740d8d7d07477058791c`

这些 self-hash、file hash、binding/config/G2/prior-attempt cross-bind 均已独立复算通过。Claim 固定 `one_shot=true`、`third_attempt_forbidden=true`、`interpretation_if_fail=terminal mechanism NO-GO; no third attempt`。

## Claim boundary

- 没有 mechanism success receipt，也不生成 checkpoint-5 integrity receipt。
- 失败 checkpoint 不做 fresh reload；重载只会验证可装载性，不能逆转科学硬门。
- 没有启动 bounded 30-step optimizer。
- dev `17` 条与 heldout `29` 条均未用于训练或选择；preference heldout 未读取、未 claim、未消费。
- 没有 generation、sandbox、general/math/format guardrail 结果。
- 不宣称 preference 泛化提升、coding correctness 提升或可部署 DPO policy。
- promoted S1 本身仍然有效；被拒绝的是这次 Day 23 DPO recipe/candidate。

## 后续边界

本协议内禁止第三次 mechanism retry，也禁止把任一 failure checkpoint 作为 candidate 或 resume source。若以后重新研究 DPO，需要新建 append-only charter，在查看新结果前冻结新的机制样本规模、假设、超参数与停止规则；不得改写本次两次失败，也不得用当前未消费的 heldout 来调参。

Day 24/25 若继续，仍须从 promoted S1 独立启动；本次 DPO failure 不降级 S1，也不自动授权任何新的 DPO 尝试。
