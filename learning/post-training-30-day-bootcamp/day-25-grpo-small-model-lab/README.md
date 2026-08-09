# Day 25 — Qwen3.5-4B Coding GRPO：One-update、vLLM 与显存 Gate

日期：`2026-08-20`
状态：`not_started`
强度：4–5 小时

## 主要目标

使用 ms-swift 从 Day 21 promoted Qwen3.5 SFT anchor 完成一个可审计的 coding GRPO one-update 与短 run，把 Day 24 trajectory/reward contract 映射到真实 rollout、sandbox、log-prob、advantage、optimizer 和 weight-version evidence。

`Qwen/Qwen3.5-4B-Base`、Day 23 DPO candidate 和任何 Day 01–12 checkpoint 都不是本日默认 parent；除非另有预注册实验，本日 policy parent 只允许 Day 21 SFT anchor。

## Hard Prerequisites

- Day 21 SFT promotion manifest、Day 24 schema/reward/sandbox tests 与 frozen coding prompts 全部通过。
- pinned ms-swift 和 vLLM 版本明确支持 resolved Qwen3.5 model class；以 [ms-swift GRPO docs](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)、[Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)、[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/) 和实际 `--help` 为证据。
- LoRA adapter/full-weight sync 路径、ViT/aligner freeze、policy/reference identity 与 `beta` 已冻结。
- Coding sandbox 与训练进程隔离，reward 能从 raw trajectory 离线重算。

任一 prerequisite 失败即停止；不换 0.6B 或 post-trained fallback 来打勾。

## 理论（45 分钟）

- 一个 prompt 的 `G` 个 completions 如何组成 group；zero-variance group 如何影响 signal。
- Rollout/old/current/reference log-prob、policy ratio、KL 与 LoRA weight sync 的角色。
- Sampling temperature、group size、prompt/completion cap 与 sandbox latency 如何影响 variance、显存和 throughput。
- Colocate 与 server/disaggregated rollout 分别把 learner、rollout weights、KV cache 和 optimizer 放在哪里。

## 显存、Topology 与长度 Preflight（45–60 分钟）

官方资料没有给出 NVIDIA 上 Qwen3.5-4B coding GRPO 的通用峰值；因此今天只接受实测：

- 记录 learner Base/adapter、optimizer、rollout model、KV cache、activations、temporary buffers、reference（若有）的 peak VRAM/CPU RAM。
- 先评估 1×H100 80GB colocate；只有满足预注册余量才运行。否则使用最多 2×H100 的 server topology（learner 与 rollout 分离）并重新 preflight。
- 明确冻结 `beta=0` 无 reference，或非零 beta + frozen reference 的选择；不能在 OOM 后静默改变目标。
- Pilot 限制为 `max_prompt_length <= 2048`、`max_completion_length <= 2048`；`vllm_max_model_len` 设为实际 prompt+completion+template headroom，不按模型原生 262K 建 cache。
- 冻结 group size、generation batch、LoRA sync、sleep/offload、tensor parallel 和 sandbox concurrency；任何变化都产生新 config key。

## Coding（60 分钟）

- 将 Day 24 verifier 接入 pinned ms-swift reward extension，先跑完全相同的 unit cases。
- 添加逐 trajectory/group 日志：IDs、completion/code、status、raw sandbox evidence、reward components、group mean/std、advantage、length/truncation、policy/weight version。
- 写 offline audit：重放 sandbox、重算 reward/group advantage，并核对训练日志与 response mask。
- 保存 resolved model/processor/template、vLLM engine、learner/rollout placement 与 weight-sync mapping。

## 训练 / 实验（120–150 分钟）

- Data：32–128 条可验证的短 coding tasks；独立冻结 30–50 条 v2 eval prompts/test families。
- G0：从 promoted SFT anchor 生成 baseline，保存每条 completion 和 sandbox evidence。
- G1：最小 rollout-only batch，确认 group/status/reward/replay。
- G2：同一配置完成 **1 rollout + 1 optimizer update**，保存 learner/rollout weight versions。
- G3：同步新 LoRA/weights 后生成下一批，证明 next rollout 使用新 policy version。
- G4：G0–G3 全通过后才运行 5–20 updates；比较 reward variance、zero-variance ratio、KL/entropy、length/truncation、grad norm、rollout/train/sandbox time 和 frozen coding pass rate。
- Optional：只改变一个 reward component，最多 5 updates；不以训练 reward 单独宣称成功。

## 资源与租卡

- GPU 上限：经 preflight 通过的 1×H100 colocate，或最多同机 2×H100 server topology；不把规划值写成最低显存结论。
- One-update gate 不通过不扩大。持续 OOM/NaN、reward 无方差、trajectory 未落盘、sandbox evidence 不可重放或 weight version 不可验证时立即停止。
- 所有 trajectories、configs、memory snapshots、metrics、adapter/checkpoint 和 sandbox evidence 同步后关机。

## Evidence-first 产物

- `../artifacts/configs/day25-qwen35-coding-grpo/`
- `../artifacts/logs/day25-qwen35-coding-trajectories.jsonl`
- `../artifacts/reports/day25-qwen35-grpo-memory-topology.md`
- `../artifacts/reports/day25-qwen35-coding-grpo.md`

## 验收

- [ ] Policy parent 仅为 Day 21 promoted SFT anchor，Base/v1/DPO candidate 未混入。
- [ ] 显存/topology/length/group/reference 均由 preflight 与 resolved config 冻结。
- [ ] 真正完成 rollout→sandbox reward→optimizer update→weight sync→next-version rollout。
- [ ] 任一 reward 可从 raw trajectory 与 sandbox evidence 离线重算。
- [ ] 报告包含逐 group variance、KL/entropy/length/truncation、memory/time 与 frozen coding eval。

## Daily Log

### Parent / pinned runtime / placement

### Memory and max-length gate

### One-update / next-version evidence

### Training reward vs frozen coding eval

### Day 26 第一动作
