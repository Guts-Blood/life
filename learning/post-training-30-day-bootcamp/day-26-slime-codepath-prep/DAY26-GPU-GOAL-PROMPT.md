# Day 26 GPU Plan / Goal Prompt

将以下内容完整发送给开卡后的 Codex：

---

你现在负责执行 Day 26 的 GPU runtime compatibility gate。请先创建 goal，再维护 plan，持续执行直到得到唯一合法终态；不要只给命令或建议。

## Goal

在不更换模型、release、container 或 topology 的前提下，验证固定的 slime `v0.3.1` 是否能让 Day 21 promoted Qwen3.5-4B merged S1 完成：

```text
S0 container/import/config
→ S1 HF→Megatron conversion/load
→ S2 rollout-only
→ S3 Day 24 reward/schema live replay
→ S4 train-only controlled optimizer step
→ S5 full weight sync + next-version rollout
```

最终只允许：

- S0–S5 全部有 runtime evidence：`go_day29`
- 任一 gate 在固定边界内失败：`slime_qwen35_compatibility_blocked`

`pre_gpu_ready`、`static_pass`、CLI 能启动或换模型 demo 都不是完成。

## 必读冻结输入

先读取并验证这些文件，不得跳过：

- `learning/post-training-30-day-bootcamp/day-26-slime-codepath-prep/README.md`
- `learning/post-training-30-day-bootcamp/artifacts/configs/day26-slime-qwen35-runtime.json`
- `learning/post-training-30-day-bootcamp/artifacts/reports/day26-slime-qwen35-compatibility.md`
- `learning/post-training-30-day-bootcamp/artifacts/eval/day26-slime-v031-static-audit.json`
- `learning/post-training-30-day-bootcamp/artifacts/data/day26-slime-qwen35-runtime-prompts.jsonl`
- `learning/post-training-30-day-bootcamp/day-26-slime-codepath-prep/day26_slime_reward.py`
- Day 21 promotion manifest、Day 24 schema、Day 25 GPU report

冻结身份：

```text
slime tag: v0.3.1
slime SHA: a6272da0d4f3d0a08520c99a2f3b4f6c887960dc
image: slimerl/slime:nightly-dev-20260804a
image digest: sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7
platform: linux/amd64
S1 downstream key: s1:qwen35-4b:c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a
S1 merged HF: /root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/exports/main-s20260809-lr1e-4-final-merged
input SHA-256: 93fbe3201ef3b1ba656dd3e131afa74e28f21160a645876879242809a5a9da12
topology: 1 GPU, approximately 98 GiB, actor/rollout colocate, TP=PP=CP=1
GPU time budget: 60 minutes
```

如果实际 GPU 不是约 98 GiB、不是单卡、容器不是该 digest、S1 merged export 缺失，先保存 evidence，然后 blocked。不得自动加卡、换 H100 topology、换 0.8B/Qwen3/其他模型、换 Miles 或 main。

## Plan

### 0. 建立 append-only run root

建立唯一 `DAY26_RUN_ROOT`，所有 config、命令、stdout/stderr、environment inventory、memory CSV、rollout dump、train dump、reward ledger、checkpoint 和 decision 都写入其中。不要覆盖失败 attempt。

将 workspace 中 Day 26/Day 24/Day 25 所需脚本和 artifacts 同步到机器；记录同步前后的 SHA-256。解析真实 workspace 路径，不假设当前目录。

设置并持久化：

```text
DAY26_RUN_ID
DAY26_RUN_ROOT
DAY26_TRAJECTORY_LEDGER
DAY26_POLICY_VERSION=slime-v000000
DAY26_SANDBOX_WORKERS=4
PYTHONPATH=<day26-dir>:/root/slime:/root/Megatron-LM
```

E2B credential 只通过项目已有 Day 22 helper 解析，不打印 secret。

### S0 — container/import/config

1. 记录 `nvidia-smi -q`、GPU UUID/name/memory、driver、CUDA、NCCL、Python、torch、transformers、Ray、SGLang、FLA、FlashQLA、Megatron 和 pip inventory。
2. 核验运行 image digest；若当前环境无法证明该 digest，S0 fail。
3. `/root/slime` 必须 checkout 到 detached `a6272da0…`；即使 image 内原本是 main，也要 fetch tag 后精确 checkout。不能读取 main 生成命令。
4. 运行本地 `audit_day26_slime_static.py` 对 runtime checkout 复验，必须 0 failed checks。
5. source `scripts/models/qwen3.5-4B.sh`，记录展开后的 `MODEL_ARGS`。
6. 用 Day 21 merged S1 实测 `AutoConfig`、`AutoProcessor`/tokenizer：class、revision assets、chat template、`enable_thinking=false`、text-only 两条 prompt token IDs。核对 promotion manifest 和本地 export file inventory/hash。
7. 验证 Ray placement 只能看到 1 GPU；先不启动正式训练。

S0 任一项不成立就停止。

### S1 — conversion/load

使用 exact tag 的官方 conversion：

```bash
cd /root/slime
source scripts/models/qwen3.5-4B.sh
PYTHONPATH=/root/slime:/root/Megatron-LM:${PYTHONPATH:-} \
torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${S1_MERGED}" \
  --save "${DAY26_RUN_ROOT}/s1_torch_dist"
```

保存 conversion log、peak GPU/CPU memory、`latest_checkpointed_iteration.txt`、文件树和逐文件 SHA-256。核对关键参数 inventory：embedding、full-attention Q/K/V/O、linear-attention `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`、MLP、norm、output head；检查 missing/unexpected/duplicate key 为零。

随后分别证明：

- SGLang 能从 merged S1 HF load。
- Megatron actor 能从转换后的 torch_dist load。
- resolved config 与 `qwen3.5-4B.sh` 一致。

不得加载原 LoRA adapter；upstream slime v0.3.1 只走 merged-full path。

### 公共最小参数合同

后续命令必须由 exact checkout 的 `train.py --help` 验证参数名，并冻结 resolved argv。核心值不得改变：

```text
--hf-checkpoint <S1_MERGED>
--ref-load <DAY26_RUN_ROOT>/s1_torch_dist
--prompt-data <bootcamp>/artifacts/data/day26-slime-qwen35-runtime-prompts.jsonl
--input-key input
--metadata-key metadata
--apply-chat-template
--apply-chat-template-kwargs '{"enable_thinking": false}'
--group-rm
--custom-rm-path day26_slime_reward.reward_func
--rollout-batch-size 2
--n-samples-per-prompt 4
--global-batch-size 8
--rollout-max-prompt-len 512
--rollout-max-response-len 512
--rollout-temperature 0.8
--rollout-top-p 0.95
--advantage-estimator grpo
--loss-mask-type qwen3_5
--optimizer adam
--lr 1e-7
--lr-decay-style constant
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--expert-model-parallel-size 1
--expert-tensor-parallel-size 1
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
--use-dynamic-batch-size
--max-tokens-per-gpu 1024
--actor-num-nodes 1
--actor-num-gpus-per-node 1
--rollout-num-gpus 1
--rollout-num-gpus-per-engine 1
--num-gpus-per-node 1
--sglang-mem-fraction-static 0.25
--qwen-gdn-backend fla
--update-weight-mode full
--update-weight-transport nccl
```

不启用 reference/KL、dynamic sampling、async rollout、VLM、LoRA、delta sync 或 reward 修改。

### S2 + S3 — rollout-only 与 live reward/replay

用公共合同加：

```text
--debug-rollout-only
--num-rollout 1
--save-debug-rollout-data <RUN_ROOT>/s2/rollout_{rollout_id}.pt
```

必须得到恰好 2 groups × G=4：

- text-only，无 image/video tokens
- 8 条可解码 response
- prompt/response token IDs、loss mask、rollout logprobs 长度一致
- status 只能是 completed/truncated；aborted/failed 单独解释
- processor/template 与 Day 25 reference 的差异可解释

S3 同批必须由 `day26_slime_reward.reward_func` 调用 live E2B：raw sandbox evidence 可离线重算；persistent infra retry 后整组 abort，不返回 NaN/None/假 0。运行本地 adapter tests，再对 ledger 做 raw replay，核对 8 条 reward、group mean/std/advantage。所有 ledger append-only。

### S4 — train-only controlled step

只加载 S2 固定 dump，用公共合同加：

```text
--debug-train-only
--load-debug-rollout-data <RUN_ROOT>/s2/rollout_{rollout_id}.pt
--num-rollout 1
--save-debug-train-data <RUN_ROOT>/s4/train_{rollout_id}_{rank}.pt
--save <RUN_ROOT>/s4/checkpoints
--save-interval 1
```

验证 train batch 数量守恒、tokens/response lengths/loss mask/reward/advantage/rollout logprob 对齐；完成一个受控 optimizer step，所有 loss/ratio/entropy/grad 指标有限。若两组均零方差，必须如实记录 zero-gradient，不能改 G/reward；但 optimizer path、checkpoint 和 state 仍须完整。

### S5 — full sync + next policy version

运行一次完整 colocated 最小闭环，用公共合同加：

```text
--colocate
--num-rollout 2
--check-weight-update-equal
--save-debug-rollout-data <RUN_ROOT>/s5/rollout_{rollout_id}.pt
--save-debug-train-data <RUN_ROOT>/s5/train_{rollout_id}_{rank}.pt
--save <RUN_ROOT>/s5/checkpoints
--save-interval 1
```

全程采集每秒 GPU allocated/reserved/free memory；最低 free-memory fraction 必须 ≥15%。先证明 initial Megatron→SGLang equality，再证明 rollout 0 后发生 full sync，rollout 1 的 `Sample.weight_versions` 或 engine evidence 是新 version。只有日志中的真实 version 边和 next rollout 证据才算 S5 pass；静态调用栈或手写环境变量不算。

### Stop conditions

以下任一情况立即停止该层并保存 minimal reproducer：

- image/tag/SHA/S1/input hash drift
- model/processor/GDN/config 不匹配
- conversion missing/unexpected keys
- 单卡 OOM、NaN/Inf、free memory <15%
- trajectory/schema/token/mask/logprob 错位
- persistent sandbox infra failure
- train-only replay 不守恒
- weight equality失败或 next version 无法证明
- 需要 patch upstream API、换 release、换模型或加卡才能继续

不因 OOM 静默缩长度、改 batch、改 G、改 reward 或启用不同 backend。若认为一个参数调整仍属于原合同，先把它作为失败后的新 config key记录，但 Day 26 默认结论仍 fail-closed，不能无记录试错。

## Required outputs

更新且不得伪造：

- `artifacts/reports/day26-slime-qwen35-compatibility.md`
- `artifacts/configs/day26-slime-qwen35-runtime.json`
- S0–S5 logs、resolved argv、environment inventory、conversion/hash inventory
- rollout/train dumps、reward ledger/replay、memory CSV、checkpoint manifest
- Day 29 go/no-go record
- Day 26 README 的 Daily Log 和唯一终态

报告必须把 fact、inference、runtime evidence 和 unknown 分开。结束前运行所有 Day 26/Day 25 reward tests、JSON parse、SHA/inventory checks，并给出精确终态与证据路径。

---
