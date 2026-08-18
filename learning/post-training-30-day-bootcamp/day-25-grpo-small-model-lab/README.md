# Day 25 — Qwen3.5-4B Coding GRPO：One-update、vLLM 与显存 Gate

日期：`2026-08-20`
状态：`closed_no_candidate`（`2026-08-16` 提前实跑；终态 `closed_no_candidate_confirmation_unopened`）
强度：4–5 小时

## 主要目标

使用 ms-swift 从 Day 21 promoted Qwen3.5 SFT anchor 完成一个可审计的 coding GRPO one-update 与短 run，把 Day 24 trajectory/reward contract 映射到真实 rollout、sandbox、log-prob、advantage、optimizer 和 weight-version evidence。

`Qwen/Qwen3.5-4B-Base`、Day 23 DPO candidate 和任何 Day 01–12 checkpoint 都不是本日默认 parent；除非另有预注册实验，本日 policy parent 只允许 Day 21 SFT anchor。

## GPU closeout

CPU control plane、GPU runtime、G1 rollout-only、G2 one-update、G3 next-version vLLM LoRA sync 和 G4 10-step short run 均已通过。实际机器是用户指定的 RTX PRO 6000 Blackwell Server Edition 97887 MiB；原 H100 硬件标识按用户授权覆盖，数据、超参、阈值、候选规则和 gate 顺序未改。

G4 checkpoint-10 的 frozen matched greedy search40 与 S1 均为 `24/40`，paired wins/regressions 均为 `0`，correct gain 与 paired net wins 未达到 `+3`。因此 confirmation24 保持未打开，没有 directional candidate，也不允许 broader S2 promotion。完整 closeout 见 [`GPU report`](../artifacts/reports/day25-qwen35-coding-grpo.md)。

晋级评测在任何 GPU eval output 产生前冻结；唯一候选是 fresh G4 checkpoint-10。search40 已按 fail-closed contract 结束，decision SHA-256 为 `8f3c4b0e3548b1b5e42ddded346917faca2ba76c78f688c4967be4faccb60913`。

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
- 冻结数据的 prompt audit cap 为 `512`，实测最大 `88` tokens；`max_completion_length=512`、trainer `max_length=1024`、`vllm_max_model_len=1152`，不按模型原生长上下文建 cache。
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
- `../artifacts/data/day25-qwen35-coding-grpo-{train,eval}.jsonl`
- `../artifacts/data/day25-qwen35-coding-grpo-confirmation24.jsonl`
- `../artifacts/eval/day25-qwen35-coding-grpo-{processor-audit-summary,argument-audit,cpu-preflight}.json`
- `../artifacts/reports/day25-qwen35-coding-grpo-cpu-preflight.md`
- `../artifacts/reports/day25-qwen35-coding-grpo-promotion-eval-freeze.md`
- `../artifacts/reports/day25-qwen35-coding-grpo-runtime-hardware-override.md`
- `../artifacts/reports/day25-qwen35-coding-grpo.md`
- G1–G4 raw trajectories、metrics、memory snapshots、adapter/checkpoint 和 sandbox evidence 保留在远端运行包；Git 只发布上述冻结合同、审计结果与 closeout 报告，不发布本地 runtime mirror。

## 验收

- [x] CPU contract 的 policy parent 仅为 Day 21 promoted SFT anchor，Base/v1/DPO candidate 未混入。
- [x] Length/group/reference 已冻结；RTX hardware override 已单独记录，全程最低 free-memory fraction 不低于 `0.5917`，通过 `15%` gate。
- [x] Train32/eval40 family disjoint；72/72 processor records 零截断，non-thinking prefix mask 与首个四空格 response token supervision 通过。
- [x] G1 可在 sealed reward batch 后、scalar 返回前有意停止；persistent infra 也会原子中止，不允许缩组或返回 NaN/None。
- [x] 四阶段 JSON 在 pinned ms-swift 中真实解析，dataset/plugin 初始化通过，offline advantage 与 trainer sample-std 结果精确一致。
- [x] Search40→confirmation24 的单次访问、唯一 checkpoint、matched generation、paired metrics、阈值和 fail-closed 决策均已冻结并通过 contract tests。
- [x] 真正完成 rollout→sandbox reward→optimizer update→weight sync→next-version rollout。
- [x] G1/G2/G3/G4 共 `112` 条训练轨迹均保留 raw completion、sandbox evidence 与离线 reward audit；未因零方差降低 G。
- [x] 报告包含逐 group variance、KL/entropy/length/truncation、memory/time 与 frozen coding eval；search40 fail-closed，无 candidate。

## Daily Log

### Parent / pinned runtime / placement

- Parent 固定为 Day 21 promoted merged S1 `main-s20260809-lr1e-4-final`；Day 23 checkpoint 显式未使用。
- Pinned ms-swift commit：`565a1ad586a21d24b23931c52d2c62b49c39bee8`；CPU contract content SHA-256：`59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c`。
- 实际 topology：单张 RTX PRO 6000 Blackwell Server Edition 97887 MiB colocate、fresh LoRA `r=8/alpha=16`、`beta=0` 无 reference、vLLM LoRA sync 开启；hardware override 单独留档。

### Memory and max-length gate

- 72 条 train/eval prompt 的最大 token 数为 `88`，cap `512`；completion cap `512`、vLLM max model len `1152`。
- 实测最低 free-memory fraction：G0 `0.6855`、G1 `0.6301`、G2 `0.5917`、G3/G4 `0.6301`；均高于 `15%`。

### One-update / next-version evidence

- G1 配置以 `DAY25_ABORT_AFTER_REWARD_BATCH=1` 在 reward ledger fsync 后主动抛出 sentinel，确保 optimizer 未获 scalar。
- G2 固定 `max_steps=1`；G3 固定 `max_steps=2`，ledger policy version 绑定 trainer global step，检查 `step-0 → step-1`。
- G2 checkpoint-1 durable；G3 观测器记录 global step 0/1 的两次 vLLM LoRA sync；G4 完成 10/10 steps、80/80 trajectories，10 行指标全有限且 6 行非零梯度。

### Training reward vs frozen coding eval

- Train：32 个 Day 22 train families。Eval：20 dev + 20 heldout families，与 train family 完全不相交。
- Eval40 对 Day 25 optimizer 独立，但并非整个项目历史中的“从未看过”；报告必须保留这一边界。
- Search40：S1 `24/40`、checkpoint-10 `24/40`、wins `0`、regressions `0`、format `40→40`、truncation `0→0`、infra `0→0`；confirmation24 未打开。

### Day 26 第一动作

只有 Day 25 的目标 GPU runtime、G1/G2/G3 全部通过后，才把已验证的 load/rollout/train/weight-sync envelope 带入 Day 26 slime compatibility gate；不以 CPU readiness 替代 runtime pass。
