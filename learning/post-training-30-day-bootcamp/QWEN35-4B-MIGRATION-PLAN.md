# Qwen3.5-4B 迁移计划（Day 13+）

决策日期：`2026-08-07`
状态：`model_selected / standalone_revision_frozen / sequential_day15_acceptance_pending / no_promoted_s1`
活动模型：[`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

## 决策与边界

Day 13 之后的训练主线改为 `Qwen/Qwen3.5-4B-Base`。Day 01–12 的 `Qwen/Qwen3-0.6B-Base` 数据、token IDs、配置、checkpoint、评测和失败结论全部保留为不可改写的 `v1` 历史证据；它们不再是后续 policy 的权重起点，也不会被全局替换成 4B。

新主线单独编号为 `qwen35-4b-day13-plus-v2`：

```text
Qwen3.5-4B Base (S0)
  -> processor/template/data/eval 迁移验收
  -> coding LoRA SFT anchor (S1)
  -> DPO branch 或 coding GRPO branch (S2)
```

Day 12 的十轮 0.6B recovery 是一次完成的、信息量充分的负结果：`10/10` rollouts、`0` 个合格 checkpoint、frozen test 未消费。它能贡献训练与选择方法，但不能充当 Qwen3.5 的 SFT anchor。任何 DPO/GRPO 都必须从新的、通过晋级 gate 的 `S1` 出发，禁止从 Base 冷启动冒充已对齐 policy。

本次没有选择 teacher。`teacher_model_id` 和 `teacher_revision` 保持 `null`；Teacher/OPD 仅作为另行批准的 deferred extension，不能静默选择 Qwen3.5-9B、Qwen3-8B 或其他模型。

Day 18–20 后续提前执行了三组 standalone experiments。它们共同冻结并使用 revision `1001bb4d826a52d1f399e183466143f4da7b741b`，提供 Megatron compatibility、Full-SFT regression diagnosis 与 LoRA probe evidence；但它们没有完成顺序课程 Day 15 的全部 M2–M5 acceptance，也没有产生通过 Day 21 promotion 的 S1。Standalone evidence 与顺序课程 gate 必须分栏记录。

## 为什么不能只替换 model ID

Qwen3.5-4B-Base 是原生多模态 checkpoint。官方仓库把它标为 `image-text-to-text`，完整仓库约 `4,659,865,088` 参数，语言模型名义为 4B；架构是 `Qwen3_5ForConditionalGeneration`，含 vision encoder、3:1 Gated DeltaNet/full-attention 混合层和 262,144-token native context。官方加载路径是 `AutoProcessor` + `AutoModelForMultimodalLM`，不是现有 Day 08–12 的 `AutoTokenizer` + `AutoModelForCausalLM` 直接替换。

当前任务是纯文本 coding post-training，因此 v2 默认训练范围为：

- 保留完整官方 checkpoint 和 processor，不自行抽取一个未验证的 text-backbone checkpoint；
- 输入中明确 `image_count=0`、`video_count=0`；
- 冻结 vision tower 和 aligner，并在每次 run 前断言实际 trainable module/name/count；
- LoRA target 先枚举后白名单化，禁止 `all-linear` 意外覆盖视觉模块；
- 如以后要训练多模态能力，必须新开 scope decision、数据合同和 parity gate。

## v1 与 v2 可复用规则

| 对象 | v1 状态 | v2 处理 |
|---|---|---|
| Day 08–12 原始样本、source/license/provenance | 可作为候选源材料 | 重新审核并在新 manifest 中显式引用 |
| rendered text、token IDs、label spans、长度、template/hash | 绑定旧 tokenizer | 全部用 Qwen3.5 processor/template 重建 |
| Day 09 mixture 比例与 supervised-token budget | 仅是 0.6B 历史实验 | 重新分词后重算，不能复制旧 token 数和 schedule |
| Day 10 task/scorer 与未消费 raw test IDs | 方法可复用 | 建立 v2 manifest、rendering、comparison key 和 Base baseline |
| Day 10/12 dev | 已支持十轮连续选择 | 降为历史诊断；v2 冻结新的 selection dev 与 confirmation split |
| Day 11 tiny-overfit/resume 结果 | 证明 v1 pipeline 正确 | 在 v2 loader、mask、scope 和 dtype 下重跑 |
| Day 12 checkpoint/export | 无 eligible checkpoint，且模型不同 | 不可作为 v2 parent；只保留证据 |
| eval aggregate score | 绑定旧协议 | 不与 v2 分数作直接纵向结论 |

未消费的 v1 frozen-test raw IDs 只有在不查看内容、通过脚本化 crosswalk 并生成新 v2 manifest 时才可继续使用；旧 rendered prompt、token IDs、comparison key 和 Base score一律不能沿用。

## 迁移 Gate

Gate 按顺序 fail closed；前一项未通过，后续训练日自动变为 blocked，而不是临时降低标准。

### M0 — Lineage boundary

- 冻结 `qwen3-0.6b-day01-12-v1` 为 immutable history。
- 新建 `qwen35-4b-day13-plus-v2`，任何 run manifest 必须带 `lineage_id` 和 parent checkpoint。
- 模型选择已完成；精确 revision 已为 standalone Day 18–20 冻结，但顺序课程 Day 15 acceptance 仍待完成。

### M1 — Revision 与文件合同

- Standalone Day 18 已从官方仓库冻结完整 40-hex commit `1001bb4d826a52d1f399e183466143f4da7b741b`，并记录 config、safetensors shards/index、tokenizer、processor/preprocessor 和 template 文件的 SHA-256；Day 19/20 继承同一模型身份。
- Day 15 顺序课程必须复核上述 revision/file manifest，并完成其余 processor/data/eval/training gates；复核通过前不能把 standalone M1 evidence描述成完整 Day 15 acceptance。
- 固定 Apache-2.0 许可记录、参数口径和本地 snapshot 路径；禁止滚动 `main/latest`。模型卡顶部明确称本仓为 `pre-trained only`，但 Overview 的 `Training Stage` 字段同时写有 `Pre-training & Post-training`；把这项官方文档矛盾连同 card snapshot 一并记录，不把 `-Base` 仓与 instruct checkpoint 混用。

### M2 — 独立环境与 loader smoke

- 保留 Day 01–12 的 Transformers 4.57.3 环境，不原地升级。
- 新建 Python 3.12 的 Qwen3.5 环境。最低功能地板是已含 Qwen3.5 的 Transformers 5.2；Day 15 按当日官方 best-practice 与 ms-swift commit 冻结精确版本（当前官方实践要求更高版本时，以实测 lock 为准）。
- 冻结 `transformers/ms-swift/torch/peft/trl/accelerate/vLLM` 及 `qwen_vl_utils/decord/causal-conv1d/flash-linear-attention/flash-attn` 的精确版本或 commit。
- 通过一条纯文本 forward、greedy generation、BF16 reload 和最终 rollout backend parity；记录 GDN/attention backend，禁止静默 fallback 后仍标记为同一 run。

### M3 — Processor、模板与 loss mask

- 记录 processor/tokenizer class、special-token IDs、chat-template hash、`enable_thinking`、padding、truncation 和 EOS/stop contract。
- 对 system/user/assistant、多轮、代码块、tool 标记、超长截断各保留 golden render、input IDs、labels 和逐 token mask。
- 纯文本 coding 主线固定 `enable_thinking=false`，除非另开单变量实验。
- 验证只学习 assistant body 与每个合法 assistant 结束边界；零监督、截断破坏 final boundary、vision token 泄漏均拒绝。

### M4 — 数据与 eval 重建

- 用新 processor 重算长度、label tokens、truncation、mixture 和 supervised-token schedule；旧 Day 09/12 token budget 作废。
- 建立新的 v2 coding-heavy dev/confirmation manifest；冻结 task IDs、prompt/template、decoder、stop、sandbox/scorer、timeout、seed 和 comparison key。
- 在任何 SFT 候选前，先跑完整 Qwen3.5 Base dev baseline、至少 10 条 deterministic repeatability 和 code sandbox sidecar。

### M5 — 训练链路与显存验收

- 顺序固定为：one forward/backward/optimizer step → 2–4 样本 tiny overfit → save/fresh-process resume → BF16 export/reload parity。
- 记录 finite loss/grad、LoRA coverage、trainable/frozen parameter counts、每卡 `max_memory_allocated`/`max_memory_reserved`、吞吐和完整 step 后余量。
- 显存余量必须至少 10–15%；仅“权重能加载”不算通过。
- 首轮 `packing=false`。只有 unpacked baseline 通过后，Day 16 才允许做受约束的 packing semantic/parity 对照。

### M6 — Coding SFT candidate 与 anchor 晋级

- Day 16 从 Qwen3.5 Base 运行受控 LoRA/QLoRA coding SFT；学习率、adapter coverage、token budget 和 checkpoint 节奏重新预注册，不继承 0.6B 数值，并产生预注册 candidate set。
- Day 17 完成 resume/export 等价性；Day 21 盲化审计候选。只有同时通过 code primary metric、retention guardrails、termination/length、sandbox error rate、resume 与 export gate 的候选，才能登记为正式 `S1`。
- 若 Day 21 无合格 `S1`，Day 23 DPO 和 Day 25 GRPO 均 blocked；禁止用 Base 或“看起来最好”的不合格 checkpoint 绕过。
- Standalone Day 19 的 Full-SFT checkpoints 与 Day 20 的三档 LoRA probes 都是诊断产物：前者未成为合格 anchor，后者没有 passing probe且未启动 main。两者均不能写入 S1 slot。

### M7 — DPO / Coding GRPO ancestry

- DPO 与 GRPO 都从同一个已晋级 `S1` 分叉，记录 parent hash、processor/template、policy/reference/rollout version。
- Coding reward 优先使用 CPU 隔离 sandbox、确定性测试、timeout 和 error taxonomy；LLM judge 是额外 GPU 预算，不默认启用。
- GRPO 显式限制 rollout context/model length；首轮建议 8K，只有显存/吞吐证据通过后才上探 12K，绝不继承 262K native context。
- 先跑 5–10 step smoke，检查 reward variance、KL、长度、stale rollout、mask/logprob 对齐、save/resume，再启动控制实验。

## 资源规划口径

下表是排期 envelope，不是“保证可跑”的显存数字。官方尚未发布 NVIDIA 上 Qwen3.5-4B coding GRPO 的可直接照抄峰值；每种 GPU、sequence、batch、generation count、LoRA target、kernel 和 colocate 方式都必须经过 M5/GRPO preflight。

| 资源 | 允许的首要用途 | 不应承诺 |
|---|---|---|
| 1×24GB | 4-bit QLoRA、512–1K 短序列、one-step/tiny smoke | 正式 coding GRPO 或 8K rollout |
| 1×48GB | BF16 LoRA SFT；GRPO 仅 4K/G4 级谨慎 smoke | 未实测即长跑或 full-parameter |
| 1×80GB | LoRA SFT；colocated 4–8K/G4–8 的候选配置 | full-parameter GRPO 的可靠下限 |
| 2×48GB / 2×80GB | learner 与 rollout 分离，较稳定的 coding GRPO | 自动获得线性显存/吞吐收益 |
| 2×80GB | full-parameter GRPO 的实验下限 | 稳定生产配置 |
| 3×80GB | full-parameter coding GRPO 更稳妥的规划档 | 不经 topology smoke 直接开长跑 |
| 4×80GB+ | PPO/多角色或额外 judge/reference 预算 | 代表本计划必须使用 PPO |

BF16 完整权重约 8.68 GiB，但 optimizer、gradient、activation、logits、KV cache、reference/rollout 副本和视觉塔都会增加峰值。旧 FP32 full-AdamW 静态状态约 69.4 GiB，尚未计 activation，因此旧 20/24GB gate 明确失效。磁盘按 `snapshot + full state × retained checkpoints + exports + rollout logs + 20% margin` 动态计算，不沿用固定 50 GiB。

## 调整后的 30-Day 节点

| Day | 新职责 | 退出条件 |
|---:|---|---|
| 13 | Qwen3 历史证据 × Qwen3.5 迁移阅读 | 完成 lineage/差异 memo |
| 14 | Week 2 历史复盘 + v2 readiness contract | M0 清楚；M1–M6 owner/证据/stop 已列明 |
| 15 | Qwen3.5 onboarding acceptance | M1–M5 全通过，否则 Day 16 blocked |
| 16 | 受控 coding LoRA SFT + packing parity | 冻结 provisional candidate set，或明确 no-candidate |
| 17 | selected config exact resume/repro | 连续与恢复 run 的状态/样本/LR 对齐 |
| 18 | Megatron/TP 最小链 + Qwen3.5 兼容 gate | GDN、loader、checkpoint/export parity 有证据 |
| 19 | optimizer/LR 稳定性 + failure injection | 完成单变量诊断矩阵 |
| 20–22 | 失败阅读、Day 21 最终 S1 promotion、preference 数据 | 正式 S1（或 no-anchor）和 coding pair contract 冻结 |
| 23 | Qwen3.5 DPO smoke | parent=S1；loss/mask/reference/eval gate 通过 |
| 24 | coding online-RL/reward contract | sandbox 与 trajectory schema 可重算 |
| 25 | Qwen3.5 coding GRPO smoke | parent=S1；5–10 step + memory/KL/reward gate 通过 |
| 26–29 | slime compatibility/read/replay/可选闭环 | 只用已验证支持 Qwen3.5 的 release；否则保留 ms-swift 主线并标 blocked |
| 30 | v2 clean reproduction | 从 Base→S1→选定 DPO/GRPO 分支重建；展示 v1/v2 边界 |

## 立即停止条件

- revision、processor/template 或文件 hash 漂移；
- loader/import 只能依赖未记录的 remote code 或 silent kernel fallback；
- LoRA 覆盖到 vision/aligner，或 trainable count 与合同不符；
- v2 baseline、golden token/mask、tiny-overfit/resume 任一缺失；
- 完整 optimizer step 后显存余量低于 10%；
- GRPO rollout 超过显式 8K/12K cap、reward 无方差、policy version stale 或 logprob/mask 错位；
- 未晋级 S1 却尝试 DPO/GRPO；
- 为让 slime 跑起来而静默切换模型、revision 或训练目标。

机器可读合同见 [`artifacts/configs/qwen35-4b-migration-contract.json`](artifacts/configs/qwen35-4b-migration-contract.json)。资源与租卡细节见 [`AUTODL.md`](AUTODL.md)，每日执行入口见 [`README.md`](README.md) 与 [`PROGRESS.md`](PROGRESS.md)。

## 一手资料

- [Qwen3.5-4B-Base 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)
- [ms-swift GRPO 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)
- [TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration)
