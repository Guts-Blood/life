# Day 10 Eval Protocol

日期：`2026-08-05`

状态：`frozen / base_dev_automated_eval_complete / frozen_test_unconsumed`

## 结论

Day 10 已冻结一套可复现的 160 条 pilot eval：112 条 `dev` 已用于 Base baseline，包含 28 条经 E2B 强隔离执行的 HumanEval code；48 条 `frozen_test` 没有生成、查看或评分。该协议适合后续 Day 12 在同一生成与 code-sandbox 协议下比较 Base、SFT-A 与 SFT-B；它不是人口总体 benchmark，也不证明样本从 Qwen 预训练语料中全局无污染。

Day 09 的上游状态是 `complete_with_gate_c_waiver`。train/eval exact/near-overlap 在冻结的 matcher、threshold 和两份 train manifests 范围内没有未处理命中；Gate C 的 0/60 occurrence-level 人审是用户 waiver，不得表述为人工 QA passed。

## Dataset contract

| 项目 | 冻结值 |
|---|---|
| Manifest version | `day10_frozen_eval_manifest_v3` |
| 总样本 | 160 |
| `dev` | 112；general/math/code/finance 各 28 |
| `frozen_test` | 48；每 slice 12 |
| Manifest semantic hash | `sha256:70749bf91842e2a3f10b66051d0fe3b26a8b6b3225198753f5861e2c39570cc0` |
| Manifest exact file SHA-256 | `be4b79f93d8c850360f3f2d38c223fcd347517796daaa9a1a4d3bc1a5eab7491` |
| Dataset context hash | `sha256:77cb6be790afa4bd97bd190c2d9a9520146b79aefbdb948a3eb6780a3d56ef6b` |
| Rendered inputs hash | `sha256:96e657165705eea82b6c8fb213a1e85f909cab4930a801bf0d9f17489e947129` |
| Eval suite hash | `sha256:41b08752d045d4fb2834a84613d71ce5569513e865e38cdc6b050be46e04d376` |
| Input-token range | 57–2,412；全 manifest 共 39,214 |

`frozen_test` 的访问规则是：Day 12 在 A/B 两条训练轨迹内分别选定 checkpoint 前，不生成、不查看、不评分，也不抽样做人审。manifest 中保存 identity/reference 是为了冻结尺子，不等于消费模型输出。

## Model and rendering contract

| 项目 | 冻结值 |
|---|---|
| Model | `Qwen/Qwen3-0.6B-Base` |
| Revision | `ddc928429ed09d9ad603fd762053d0434c15e865` |
| Observed snapshot hash | `sha256:b4117311d9b057b9df4e91c241cbf15650258d780c4da29731a626848ad9bf8a` |
| `model.safetensors` SHA-256 | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` |
| `config.json` SHA-256 | `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59` |
| Chat-template SHA-256 | `44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5` |
| Rendering | `add_generation_prompt=true`、`enable_thinking=false`、不二次加 special tokens |
| Maximum input | 4,096 tokens |

每个 slice 使用独立、已版本化的 prompt adapter。`rendered_prompt_hash` 和 `input_ids_hash` 都逐样本保存，避免只冻结 raw task、却让模板或 tokenizer 静默漂移。

## Generation and scoring contract

- Greedy decoding：`do_sample=false`、`num_beams=1`、`repetition_penalty=1`、seed `20260805`。
- EOS/stop token：`151645`；pad token：`151643`。
- `max_new_tokens`：general 32；math/code/finance 512。
- Scorer registry：`day10-scorer-registry-v3`。
- Protocol hash：`sha256:4aa329a2566f0a3864d0bf35c158e1b22b08670afb6770b48a25161aed4e7c06`。
- Scorer registry hash：`sha256:593de0e8411ce4b3af85d593c6856afbcd310c256823e3f9e1b5c218575c866d`。
- Aggregation hash：`sha256:67680e523d577037611398dd29f98dd57097f6d9a7f89dccb0e3c9fe3492b357`。

5-sample dry run 先后发现并修正了两个真实 scorer 缺陷：v1 会把 prompt 中的 MMLU 选项文本误当最终答案；v2 会把 TAT-QA 的格式说明误当答案。两份被拒绝的诊断输出被保留，正式 manifest 与 baseline 使用 v3。修 scorer 后重新冻结 manifest，未根据 Base 分数修改数据或选择规则。

HumanEval completion 从未在 macOS 宿主机执行。本机虽有已废弃的 `sandbox-exec`，但没有可靠的硬内存隔离，不能冒充强安全边界。正式 code scorer 重新验证完整 112-row Base run、冻结 extractor、HumanEval source revision/test hash，再把 28 条 dev completion 分别送入新建的 E2B sandbox。结果为 0/28 pass@1；25 条 syntax error、3 条 runtime error、0 条 infrastructure failure。

## Execution contract

Base run 使用专用 Conda 环境 `post_training_lab`：Python 3.11.15、PyTorch 2.9.1、Transformers 4.57.3，在 macOS arm64 CPU 上以 BF16、10 torch threads 和 14 interop threads 运行。

| 项目 | Hash |
|---|---|
| Conda direct contract | `b5a41b3111dcce0a0ddc8e1cf1e8dac3abd7cf1a2e16835948dbe276af77d7e3` |
| Resolved pip snapshot | `40aac26b5f4672db9d1415b816148107dc05563db116d6b5c53bcfabd8336e12` |
| Execution protocol | `sha256:ac58adf3527a8824035c11c92cb0c746f044a5f4cd51a6405ab1ef9705d25bb1` |
| Comparison key | `sha256:b0ecfbf25d6879f05b60c945319685189d1b0737e75d9968de3f50ff6c3a7788` |
| Base run hash | `sha256:49408d403fdc72f9f125c9aeda158e0d6f22d80cbce05ac74f69b9d9b535de2d` |

Code 执行使用独立的 `post_training_sandbox` client 环境，避免给已冻结的生成环境安装 E2B SDK。每条样本均使用 E2B template `rki5dems9wqfm4r03t7g`（Python 3.11.6、envd 0.6.10、2 vCPU、512 MiB），`secure=true`、禁公网出站、禁 public traffic、无 env/mount/MCP；candidate CPU 2 秒、wall 8 秒，SDK outer timeout 15 秒，sandbox TTL 60 秒。live smoke 验证实际 HTTPS content fetch 被阻断，并验证 pass/assertion/CPU-timeout 三条结果路径。

| Code sandbox item | Hash / value |
|---|---|
| Client direct contract | `dc2f0bcab5217f1c57b5b456b191192498f237ee23249cdf70a2967d8220afc5` |
| Client resolved snapshot | `8af157cf4772289a2d898e7f65487d9500f144e3f6aee91ccb18d4d42133bfa7` |
| Sandbox config exact SHA-256 | `d4841e1e0479f6488306e916eb5d664d848716e80cfc7aef9943e6134ecbfc8c` |
| Code execution protocol | `sha256:9571c64301029e230ab2d01e1b8daf8a6dcf7c4a5c8b0696e21d9b073a730c87` |
| Code run hash | `sha256:147616a3cd6f02975a2be509b6420703f4334631db0a1f5d2ca421ef6b51a626` |
| Sidecar exact SHA-256 | `d50be508bf284beac8b8e731fd521a68547dccf928d7206938a64d65b264509c` |
| Complete comparison key | `sha256:850a69854fe0a945d7b86f3c71a59a7977e09dd42a2df96ca7548b6068bd0f14` |

原 `comparison_key` 只覆盖生成侧测量协议；它不包含本次被评测 checkpoint 的权重，权重只进入 `run_hash`。加入 code scorer 后，Day 12 必须额外匹配 `complete_comparison_key`，才能把 Base 与 SFT code pass@1 直接相减。若改在 H100/CUDA 上生成，generation execution protocol 会变化，仍须在同一 GPU 协议下重跑 Base。

## Pre-registered selection contract

Day 12 selection policy 已在训练前冻结：code 是 target；缺少强隔离 sandbox 时 selection 直接判 `blocked/inconclusive`；相对 Base 增加至少 3 个 code correct cases 才算有意义；general/math/finance 任一 slice 相对 Base 退化超过 2 个 correct cases 时 checkpoint 不合格。A/B 在 25%/60%/100% token budget 上做 matched contrast。

Selection policy exact SHA-256：`2ddf00ec35f7e00758c80a72b39edd691fd238f31f72b2b34a44aca2e509ce4d`。由于 Base code 为 0/28，“相对 Base 至少增加 3 个 correct cases”在本协议下等价于候选 checkpoint 至少达到 3/28。

## Source artifacts

- `../eval/day10-frozen-eval-manifest.json`
- `../eval/day10-qwen3-0.6b-base-predictions.jsonl`
- `../eval/day10-qwen3-0.6b-base-code-sandbox-e2b.jsonl`
- `../configs/day12-checkpoint-selection-policy.json`
- `../logs/post-training-lab-environment.md`
- `../logs/post-training-sandbox-environment.md`
- `../../day-10-frozen-eval-baseline/day10_e2b_sandbox_config.json`
- `day10-base-baseline.md`
