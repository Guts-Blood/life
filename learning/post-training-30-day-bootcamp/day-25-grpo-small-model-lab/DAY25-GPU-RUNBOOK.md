# Day 25 GPU Runbook — Qwen3.5-4B Coding GRPO

状态：`cpu_ready_gpu_pending`  
CPU trust root：`59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c`

本 runbook 只允许使用 Day 21 promoted merged S1。Day 23 DPO checkpoint、Base checkpoint 和旧 0.6B lineage 均不得替代。

## 0. 本地关机前 gate

在 bootcamp 根目录执行：

```bash
python3 day-25-grpo-small-model-lab/run_day25_cpu_preflight.py
```

只接受：

```text
status=pass_cpu_ready_gpu_pending
gates=8
preflight_sha256=8b38235160b9f9561ee2a6cd1e49e0e31acff124dea6c2d5afb7817a2e35ed5b
```

若 contract、argument audit、源码、数据或 pinned ms-swift 漂移，先在 CPU 侧重新审计并产生新 trust root；不得手改 GPU JSON 绕过。

## 1. 目标机与目录

Primary topology 固定为一张 H100 80GB、colocate vLLM、tensor parallel 1。先设置目标机路径；`DAY25_RUN_ROOT` 必须是尚不存在的绝对路径：

```bash
export DAY25_REPO=/path/to/life
export DAY25_BOOTCAMP="$DAY25_REPO/learning/post-training-30-day-bootcamp"
export DAY25_MODEL=/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/exports/main-s20260809-lr1e-4-final-merged
export DAY25_RUN_ROOT=/root/autodl-tmp/runs/day25-qwen35-coding-grpo-YYYYMMDDTHHMMSSZ
export PYTHONPATH="$DAY25_REPO/vendor/ms-swift"
export USE_MCORE_GDN=0
cd "$DAY25_BOOTCAMP"
```

目标 Python 必须满足 binder 的 fail-closed runtime contract：

- `ms-swift==4.5.0.dev0`，且 import 来自 clean commit `565a1ad586a21d24b23931c52d2c62b49c39bee8`
- `torch==2.10.0+cu128`
- `transformers==5.12.1`
- `peft==0.19.1`
- `e2b==2.37.0`
- `vllm>=0.17.0`、`trl>=0.20.0`
- `torch.cuda.device_count()==1`

不要先启动训练来试错；下一步 binder 会一次性核对这些条件、远端 S1 全部文件和四份真实 CLI config。

## 2. E2B credential

将 credential 安全放到 `$DAY25_REPO/.env.e2b`，文件只能包含一行 `E2B_API_KEY=...`，owner 为当前用户且 mode 为 `0600`：

```bash
chmod 600 "$DAY25_REPO/.env.e2b"
```

不得把 key 写入 config、ledger、日志或 Git。Day 22 credential reader 会在每次 live sandbox 调用前校验权限和格式。

## 3. 绑定目标 GPU、runtime 与 promoted S1

```bash
python3 day-25-grpo-small-model-lab/bind_day25_qwen35_gpu.py \
  --model "$DAY25_MODEL" \
  --ms-swift-root "$DAY25_REPO/vendor/ms-swift" \
  --run-root "$DAY25_RUN_ROOT"
```

成功后会原子创建：

```text
$DAY25_RUN_ROOT/binding.json
$DAY25_RUN_ROOT/configs/g1-rollout-only.json
$DAY25_RUN_ROOT/configs/g2-one-update.json
$DAY25_RUN_ROOT/configs/g3-weight-sync-probe.json
$DAY25_RUN_ROOT/configs/g4-bounded-short-run.json
```

此时状态只是 `gpu_config_bound_preflight_pending`：binder 没有宣称模型已加载，也没有宣称显存足够。

## 4. 通用显存采样

每个 stage 启动前创建独立采样文件：

```bash
mkdir -p "$DAY25_RUN_ROOT/logs" "$DAY25_RUN_ROOT/ledgers" "$DAY25_RUN_ROOT/audits"
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits -l 1 \
  > "$DAY25_RUN_ROOT/logs/STAGE-memory.csv" &
export DAY25_MEMORY_MONITOR_PID=$!
```

Stage 结束后：

```bash
kill "$DAY25_MEMORY_MONITOR_PID" || true
wait "$DAY25_MEMORY_MONITOR_PID" 2>/dev/null || true
awk -F',' '{gsub(/ /,"",$4); gsub(/ /,"",$6); if (($6/$4)<0.15) bad=1} END{exit bad}' \
  "$DAY25_RUN_ROOT/logs/STAGE-memory.csv"
```

`awk` 非零即显存余量 gate 失败。不要靠降低 group、completion cap、LoRA rank 或改 `beta` 原地续跑；变更必须产生新 contract key。

## 5. G1 — rollout-only，optimizer 未获授权

G1 使用正常 `max_steps=1` 配置，但 reward plugin 会在 sealed ledger `fsync` 后、scalar 返回 trainer 前抛出明确 sentinel。因此训练进程的非零退出是预期结果。

```bash
export DAY25_RUN_ID=day25-g1-rollout-only
export DAY25_POLICY_VERSION=v0
export DAY25_TRAJECTORY_LEDGER="$DAY25_RUN_ROOT/ledgers/g1-rollout-only.jsonl"
export DAY25_SANDBOX_WORKERS=4
export DAY25_ABORT_AFTER_REWARD_BATCH=1

set +e
CUDA_VISIBLE_DEVICES=0 python3 -m swift.cli.main rlhf \
  "$DAY25_RUN_ROOT/configs/g1-rollout-only.json" \
  > "$DAY25_RUN_ROOT/logs/g1-rollout-only.log" 2>&1
export DAY25_G1_EXIT=$?
set -e
```

必须同时满足：

```bash
test "$DAY25_G1_EXIT" -ne 0
rg -F "intentional G1 rollout-only stop" "$DAY25_RUN_ROOT/logs/g1-rollout-only.log"
test ! -e "$DAY25_RUN_ROOT/outputs/g1_rollout_only/checkpoint-1"
python3 day-25-grpo-small-model-lab/audit_day25_rewards.py \
  "$DAY25_TRAJECTORY_LEDGER" \
  --output "$DAY25_RUN_ROOT/audits/g1-reward-audit.json"
```

Audit 必须为 `pass`、`complete_rollout_only_stop_batches=1`、`complete_groups=2`，且没有 persistent infra。还需通过 15% 显存余量检查。任一项失败，不进入 G2。

## 6. G2 — exactly one optimizer update

G2 必须从 fresh S1 + fresh LoRA 启动，不能 resume G1：

```bash
export DAY25_RUN_ID=day25-g2-one-update
export DAY25_POLICY_VERSION=v0
export DAY25_TRAJECTORY_LEDGER="$DAY25_RUN_ROOT/ledgers/g2-one-update.jsonl"
export DAY25_ABORT_AFTER_REWARD_BATCH=0

CUDA_VISIBLE_DEVICES=0 python3 -m swift.cli.main rlhf \
  "$DAY25_RUN_ROOT/configs/g2-one-update.json" \
  > "$DAY25_RUN_ROOT/logs/g2-one-update.log" 2>&1

python3 day-25-grpo-small-model-lab/audit_day25_rewards.py \
  "$DAY25_TRAJECTORY_LEDGER" \
  --output "$DAY25_RUN_ROOT/audits/g2-reward-audit.json"
test -d "$DAY25_RUN_ROOT/outputs/g2_one_update/checkpoint-1"
```

只有 exit 0、reward audit pass、finite loss/grad norm/entropy/ratio、checkpoint-1 durable 和显存余量通过，才能声明 one-update gate 通过。

## 7. G3 — next rollout 使用更新后 policy

G3 同样从 fresh S1 + fresh LoRA 启动，`max_steps=2`。第一批 reward ledger 应标记 `g3:trainer-step-0`，完成第一次 update 后的第二批应标记 `g3:trainer-step-1`；ms-swift colocate path 应在 global step 变化时重新同步 LoRA 到 vLLM。

```bash
export DAY25_RUN_ID=day25-g3-weight-sync
export DAY25_POLICY_VERSION=g3
export DAY25_TRAJECTORY_LEDGER="$DAY25_RUN_ROOT/ledgers/g3-weight-sync.jsonl"
export DAY25_ABORT_AFTER_REWARD_BATCH=0

CUDA_VISIBLE_DEVICES=0 python3 -m swift.cli.main rlhf \
  "$DAY25_RUN_ROOT/configs/g3-weight-sync-probe.json" \
  > "$DAY25_RUN_ROOT/logs/g3-weight-sync.log" 2>&1

python3 day-25-grpo-small-model-lab/audit_day25_rewards.py \
  "$DAY25_TRAJECTORY_LEDGER" \
  --output "$DAY25_RUN_ROOT/audits/g3-reward-audit.json"
test -d "$DAY25_RUN_ROOT/outputs/g3_weight_sync_probe/checkpoint-2"
```

第二次 optimizer update 只是该两步 probe 的附带结果，不计入 G2 的“one update”声明。若 ledger 没有同时出现 step 0/1、vLLM sync 日志缺失、第二批仍可疑地使用旧 weights，均视为 G3 失败。

## 8. G4 — bounded short run（可选）

只有 G0 runtime、G1、G2、G3 全部通过后，才允许从 fresh S1 + fresh LoRA 启动最多 10 updates：

```bash
export DAY25_RUN_ID=day25-g4-bounded-short-run
export DAY25_POLICY_VERSION=g4
export DAY25_TRAJECTORY_LEDGER="$DAY25_RUN_ROOT/ledgers/g4-bounded-short-run.jsonl"
export DAY25_ABORT_AFTER_REWARD_BATCH=0

CUDA_VISIBLE_DEVICES=0 python3 -m swift.cli.main rlhf \
  "$DAY25_RUN_ROOT/configs/g4-bounded-short-run.json" \
  > "$DAY25_RUN_ROOT/logs/g4-bounded-short-run.log" 2>&1
```

训练 reward 不等于能力提升。G4 结束后必须单独跑 frozen eval40、保存逐题 sandbox evidence，并与同配置的 S1 baseline 比较；在此之前不晋级 GRPO checkpoint。

## 9. 晋级 Eval — search40 → confirmation24

晋级契约已在任何 GPU eval output 产生前冻结，content SHA-256 为：

```text
4a0060486184694f7caae505084d8939cf01eaeb510cdc0069f1b0fe749b8cd9
```

唯一候选是 fresh G4 `checkpoint-10`；G2、G3 和 G4 中间 checkpoint 都没有候选资格。先生成 matched greedy search40：

```bash
mkdir -p "$DAY25_RUN_ROOT/eval/search40" "$DAY25_RUN_ROOT/eval/confirmation24"

python3 day-25-grpo-small-model-lab/generate_day25_promotion_eval.py \
  --binding "$DAY25_RUN_ROOT/binding.json" \
  --suite search40 \
  --model-role s1_parent \
  --model "$DAY25_MODEL" \
  --output "$DAY25_RUN_ROOT/eval/search40/s1-completions.json"

python3 day-25-grpo-small-model-lab/generate_day25_promotion_eval.py \
  --binding "$DAY25_RUN_ROOT/binding.json" \
  --suite search40 \
  --model-role grpo_g4_final \
  --model "$DAY25_MODEL" \
  --adapter "$DAY25_RUN_ROOT/outputs/g4_bounded_short_run/checkpoint-10" \
  --output "$DAY25_RUN_ROOT/eval/search40/grpo-completions.json"
```

分别通过 Day 24 E2B verifier；persistent infra 会使整套结果无效：

```bash
python3 day-25-grpo-small-model-lab/sandbox_day25_promotion_eval.py \
  --suite search40 \
  --completions "$DAY25_RUN_ROOT/eval/search40/s1-completions.json" \
  --evidence-output "$DAY25_RUN_ROOT/eval/search40/s1-evidence.json" \
  --results-output "$DAY25_RUN_ROOT/eval/search40/s1-results.json"

python3 day-25-grpo-small-model-lab/sandbox_day25_promotion_eval.py \
  --suite search40 \
  --completions "$DAY25_RUN_ROOT/eval/search40/grpo-completions.json" \
  --evidence-output "$DAY25_RUN_ROOT/eval/search40/grpo-evidence.json" \
  --results-output "$DAY25_RUN_ROOT/eval/search40/grpo-results.json"

python3 day-25-grpo-small-model-lab/score_day25_promotion_eval.py \
  --suite search40 \
  --baseline-results "$DAY25_RUN_ROOT/eval/search40/s1-results.json" \
  --candidate-results "$DAY25_RUN_ROOT/eval/search40/grpo-results.json" \
  --output "$DAY25_RUN_ROOT/eval/search40/decision.json"
```

Search40 必须同时满足：正确题净增至少 3、paired net wins 至少 3、regressions 至多 2、format-valid count 最多损失 1、truncation 不增加、平均 completion tokens 不超过 S1 的 1.5 倍、两边零 infra。失败即 `closed_no_candidate_confirmation_unopened`。

只有 search decision 为 `search_gate_pass_confirmation_authorized`，才允许打开 confirmation24。两次 generation 都必须添加：

```bash
--suite confirmation24 \
--search-decision "$DAY25_RUN_ROOT/eval/search40/decision.json"
```

Sandbox 命令只需换 suite/path，不接受 `--search-decision`；最终 paired scorer 换成 `--suite confirmation24`，并添加同一个 `--search-decision`。Confirmation24 要求正确题净增至少 2、paired net wins 至少 2、regressions 至多 1，其余 guardrails 与 search40 相同。

两层均通过时，唯一允许的结论为 `directional_code_candidate_eligible`。该 24-row confirmation 是 model-output blind engineering gate，不是全球无污染的大样本统计确认，因此不得写成“confirmed capability gain”或直接晋级 broader S2。

## 10. 立即停止条件

出现以下任一情况立即停止并保留 run root：

- 远端 S1、CPU trust root、pinned checkout 或 package version 漂移；
- Qwen3.5/vLLM/LoRA load 失败，或最低 free VRAM fraction `<0.15`；
- live E2B 不通、persistent infra、G=4 被缩组或 ledger 未落盘；
- reward/advantage 离线重算不一致；
- NaN/Inf loss、grad norm、ratio、entropy，或 checkpoint/optimizer state 不完整；
- G3 无法证明第二批 rollout 位于 trainer step 1；
- 需要修改冻结超参才能继续。

先同步 `binding.json`、configs、logs、ledgers、audits、memory CSV、checkpoint 和 TensorBoard，再关机。
