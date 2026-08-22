# Day 25 — Qwen3.5-4B Coding GRPO GPU closeout

执行时间：`2026-08-16`  
终态：`closed_no_candidate_confirmation_unopened`

## 结论

Day 21 promoted S1 已在冻结的 Day 25 contract 下完成真实 rollout-only、one-update、next-version vLLM LoRA sync probe 与 10-step GRPO short run。所有 runtime/training gates 通过，但 checkpoint-10 在 matched greedy search40 上相对 S1 没有正确题净增，因此不是 directional code candidate；confirmation24 按 fail-closed contract 保持未打开，broader S2 promotion 不获授权。

## Runtime 与谱系

- SSH endpoint：`autodl-36153`；host：`autodl-container-6acf40b03d-62d686fe`。
- 用户授权的 hardware override：物理 GPU 0，UUID `GPU-8ba9b2a0-4467-b92c-4fca-fcad8cc0fc5b`，`NVIDIA RTX PRO 6000 Blackwell Server Edition`，`97887 MiB`；全程 `CUDA_VISIBLE_DEVICES=0`。原冻结 H100 标识有误，只有硬件身份被覆盖，数据、超参、阈值、gate 顺序和候选规则均未修改。
- Parent：Day 21 promoted merged S1 `main-s20260809-lr1e-4-final`；Day 23 DPO checkpoint 未使用。
- Pinned ms-swift commit：`565a1ad586a21d24b23931c52d2c62b49c39bee8`；binding SHA-256：`43f32bade493740383fd8dae2e6e7e6323de57a74952dd3ae2a8958cd4285c96`。
- Runtime：ms-swift `4.5.0.dev0`、torch `2.10.0+cu128`、transformers `5.12.1`、PEFT `0.19.1`、TRL `0.29.1`、vLLM `0.19.1`、E2B `2.37.0`；`pip check` 通过。

## Gate 结果

| Gate | 结果 | 核心证据 |
|---|---|---|
| G0 | pass | Qwen3.5/vLLM/LoRA load 与生成成功；E2B live sandbox 成功；最低 free-memory fraction `0.6855`。 |
| G1 | pass | intentional rollout-only sentinel；`8/8` trajectories、`2` 个 G=4 groups、零 persistent infra、无 checkpoint；最低 free fraction `0.6301`。 |
| G2 | pass | exit 0、`global_step=1`、checkpoint-1 含 adapter/optimizer/scheduler/RNG/trainer state；所有 loss/grad/entropy/ratio 有限；最低 free fraction `0.5917`。两组零方差，loss/grad norm 为 0，但未缩 G、未改参。 |
| G3 | pass | ledger 同时出现 `g3:trainer-step-0` 与 `g3:trainer-step-1`；观测器记录 vLLM LoRA sync 在 global step 0/1 各完成一次。第 1 批零梯度使两次输入摘要相同；第 2 批出现非零优势，step-2 grad norm `0.2578`。 |
| G4 | pass | 10/10 steps、`80/80` trajectories、`20` 个完整 G=4 groups、零 persistent infra；10 行指标全有限，6 行非零梯度；checkpoint-10 durable；最低 free fraction `0.6301`。 |

G2/G3/G4 的训练 reward 只证明数据流与 optimizer 可运行，不作为能力提升结论。冻结数据中的零方差 batch 被如实保留，没有用降低 group size 或改 reward 规则规避。

## Frozen paired Code Eval

Eval contract SHA-256：`4a0060486184694f7caae505084d8939cf01eaeb510cdc0069f1b0fe749b8cd9`。

Search40 结果：

- S1 baseline：`24/40` correct。
- GRPO checkpoint-10：`24/40` correct。
- Paired wins `0`、regressions `0`、net wins `0`、correct gain `0`。
- Format valid：`40 → 40`；truncation：`0 → 0`；infra errors：`0 → 0`。
- Mean completion tokens：`39.0 → 40.825`，倍率 `1.0468`，长度 guardrail 通过。
- 未通过的预注册 gates：correct gain 至少 `3`、paired net wins 至少 `3`。

Search decision SHA-256：`8f3c4b0e3548b1b5e42ddded346917faca2ba76c78f688c4967be4faccb60913`。合法结论仅为 `closed_no_candidate_confirmation_unopened`；不得写成 confirmed capability gain，也不得晋级 broader S2。

## 可重放证据

远端 run root：`/root/autodl-tmp/runs/day25-qwen35-coding-grpo-20260816T064700Z`。本地镜像放在 workspace 的 `tmp/day25-qwen35-coding-grpo-20260816T064700Z`。run root 保留 binding/configs、所有成功与失败 attempt、raw token IDs/completions、E2B evidence、reward ledgers/audits、trainer/TensorBoard logs、memory CSV、checkpoint 文件及最终 tree manifest。

早期两次参数解析失败均发生在模型加载前，已 append-only 保留：G2 attempt 1 误用 pipeline `--config`；G3 instrumentation attempt 2 未先展开 JSON argv。它们没有 optimizer update，也没有覆盖后续成功证据。
