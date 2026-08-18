# Day 26 — slime × Qwen3.5 Compatibility Verification（No Fallback）

日期：`2026-08-21`  
状态：`slime_qwen35_compatibility_blocked`（`2026-08-17` 提前执行；S0 fail，S1–S5 按合同未运行）
强度：4–5 小时

## 主要目标

只回答一个问题：官方 slime release 中是否存在一条能对 Day 21 promoted Qwen3.5 merged SFT anchor 正确完成 model/processor load、rollout schema、full-parameter learner train 与 weight sync 的可复现路径。今天不启动正式 RL；`v0.3.0` 只保留为历史架构阅读基线，不能被默认当作 Qwen3.5 runtime。

若没有兼容路径，结论必须是 `slime_qwen35_compatibility_blocked`，Day 29 随之 blocked。禁止换 0.6B、Qwen3 或其他模型完成一个无关 recipe 后称为 active track 成功。

## Pre-GPU Closeout（2026-08-17）

- 固定正式 release `v0.3.1@a6272da0d4f3d0a08520c99a2f3b4f6c887960dc`；不使用滚动 `main`。
- 固定 `slimerl/slime:nightly-dev-20260804a@sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7`，并记录 CUDA 12.9.1、NCCL 2.27.3、SGLang v0.5.15.post1 与 Megatron commit。
- exact tag 已提供 Qwen3.5-4B config、GDN、HF↔Megatron conversion、Qwen3.5 loss mask、rollout、debug replay、full train/full weight-sync 源码路径；static audit 0 failed checks。
- upstream slime v0.3.1 没有已文档化的 PEFT/LoRA learner 路径。因此冻结 Day 21 merged S1 → full Megatron → full sync；不把 adapter hack 成 slime baseline。
- 已生成两条冻结 prompt，`2 × G=4 = 8` trajectories；Day 24 E2B batch reward adapter 单测 `4/4`，Day 25 reference adapter regression `7/7`。
- 开卡 topology 固定为 Day 25 同级单卡约 98 GiB、colocate、TP/PP/CP=1；GPU gate 60 分钟且最低 free-memory fraction `15%`。未经授权不得加卡。
- 该条是执行前记录；实际 GPU run 已在 S0 触发 fail-closed，最终状态见 Daily Log。

## Release/Version Gate（45 分钟）

1. 查看 [slime Releases](https://github.com/THUDM/slime/releases) 与官方文档，列出可能支持 Qwen3.5 的正式 release/tag；不预先指定 winner。
2. 对每个候选核对 model architecture、Megatron bridge、SGLang/vLLM、Transformers、Ray、container/CUDA/NCCL compatibility。
3. 选择一个候选 checkout，记录 tag/SHA、submodules/dependencies、Docker digest 与官方 example/source evidence。
4. `v0.3.0` 可用于对照 Sample/DataSource/rollout/train 概念，但没有 runtime evidence 时不得进入 Day 29 command。

## Pinned Codepath Audit（90 分钟）

使用 `rg`/symbol search 解析实际 `file:function`，不按记忆写路径：

- Qwen3.5 conditional model/config/processor 与 HF↔Megatron/rollout conversion；
- GDN/attention、text-only/vision freeze 与 LoRA/trainable state；
- Sample/DataSource、rollout/group/status、response/tool/environment masks；
- coding reward/sandbox adapter、buffer/requeue、train batch/advantage/loss；
- learner→rollout weight sync、adapter/full sync 与 next policy-version evidence；
- rollout-only、train-only replay、debug dump 与 failure recovery entrypoints。

每条边记录 `producer -> schema -> transport -> consumer -> observable evidence`。

## Compatibility Gates（120 分钟）

严格按顺序；每项都使用 Day 15 exact Base revision 与 Day 21 promoted SFT anchor：

1. **S0 import/config**：container 可解析 model、processor、runtime 和资源 placement。
2. **S1 conversion/load**：Base+SFT adapter 或 promoted export 按官方路径加载；参数/adapter/hash inventory 一致。
3. **S2 rollout-only**：两条 text-only coding prompts 产生可解码 trajectories，token IDs/template/status 与 ms-swift reference 可解释。
4. **S3 reward/schema**：Day 24 sandbox/reward adapter unit cases 与 raw replay 通过。
5. **S4 train-only dry gate**：从固定 dump 构造 learner batch，验证 loss/mask/advantage 和一个受控 optimizer step 所需路径；不扩展正式训练。
6. **S5 weight sync**：验证框架宣称的 LoRA/full sync 方式，以及 next rollout 如何证明新 version；若只完成静态检查，状态必须标成 `runtime_unverified`。

任一 gate 失败就停在该层，保存 minimal reproducer 和 support boundary。不能通过打补丁猜 API 来绕过正式 model support；需要未发布代码时，记录 blocker，不把本地 hack 冻结为课程基线。

## 资源与租卡

- CPU/container/config audit 为主；只有 S0/S1 通过后才允许使用 Day 25 已验证 topology，GPU gate 总计不超过 1 小时。
- 不做正式 RL、不做大规模权重转换、不测试 8×H100 throughput。
- Day 29 的 model、release、container、topology 和 runbook 只能来自 S0–S5 全部 runtime-verified 的结果。

## Evidence-first 产物

- `../artifacts/reports/day26-slime-qwen35-compatibility.md`
- `../artifacts/configs/day26-slime-qwen35-runtime.json`
- `../artifacts/eval/day26-slime-v031-static-audit.json`
- `../artifacts/data/day26-slime-qwen35-runtime-prompts.jsonl`
- `day26_slime_reward.py` 与 `test_day26_slime_reward.py`
- `DAY26-GPU-GOAL-PROMPT.md`
- S0–S5 logs、conversion/hash inventory、resolved codepath 与 Day 29 go/no-go record
- `../artifacts/eval/day26-slime-qwen35-day29-go-no-go.json`
- 本地未版本化的 `../tmp/day26-slime-qwen35-20260817T141117Z/` 保存 90-file runtime evidence mirror；远端仓库以 compatibility report、runtime config 和 Day 29 go/no-go JSON 作为公开证据入口。

## 验收

- [x] release/tag/SHA/image digest 有官方依据并已在 pre-GPU contract 冻结；container 内无法证明该 digest，S0 已 fail-closed。
- [ ] Qwen3.5 model/processor/GDN 与 Sample→rollout→train→full-sync 每条边有真实 runtime evidence；static code evidence 已完成。
- [ ] Day 24 reward/schema 在 pinned runtime 下 live 重放；本地 adapter contract 已 `4/4`。
- [x] pre-GPU 产物没有使用不同模型 fallback，也没有把 LoRA unsupported 边界隐藏为成功。
- [x] 输出唯一终态 `slime_qwen35_compatibility_blocked`，S1–S5 明确为 `not_run_due_to_s0_fail`。

## Daily Log

### Candidate releases / selected runtime

- Selected：`v0.3.1@a6272da0d4f3d0a08520c99a2f3b4f6c887960dc`。
- Image：`slimerl/slime:nightly-dev-20260804a@sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7`。
- `v0.3.0` 保留为历史 release/change boundary，不进入 active command。

### Qwen3.5 load/conversion evidence

- `scripts/models/qwen3.5-4B.sh` → `slime_plugins.models.qwen3_5:get_qwen3_5_spec`。
- `Qwen3_5GatedDeltaNet` 处理 linear-attention/GDN；full-attention 沿用 Megatron spec。
- `qwen3_5_hf_tensor` 与 `convert_qwen3_5_to_hf` 覆盖 dense attention、linear attention、MLP、norm、embedding/output。
- Day 21 merged S1 是唯一 load input；远端复算 `11` 文件 / `9,098,708,091` bytes，inventory seal `16bc212d…53be` 通过，但因 S0 失败未执行 HF→Megatron conversion。
- upstream LoRA learner 未证实，active path 固定为 merged full model/full sync。

### S0–S5 results

- Pre-GPU：release/source audit pass；reward adapter `4/4`；frozen input ready。
- Runtime root：`/root/autodl-tmp/runs/day26-slime-qwen35-20260817T141117Z`；未版本化的本地镜像 `../tmp/day26-slime-qwen35-20260817T141117Z` 已对 90 项远端证据逐文件回验。
- S0：**fail**。物理 GPU `2×97887 MiB`；目标 OCI digest 无法证明；active CUDA/NCCL 为 `12.4/2.21.5`；Ray/SGLang/Megatron 与 exact source checkouts 缺失。
- `CUDA_VISIBLE_DEVICES=0` 只让 torch 看到一张卡；Ray 因未安装无法完成 placement gate，不能覆盖物理 topology 和其他 blocker。
- S1–S5：`not_run_due_to_s0_fail`；没有 conversion、model load、rollout、live E2B、optimizer、checkpoint 或 weight sync。
- S0 原始 decision：未版本化本地镜像中的 `../tmp/day26-slime-qwen35-20260817T141117Z/decision/s0-decision.json`；公开结论见 compatibility report 与 Day 29 go/no-go JSON。

### Day 29 go/no-go

`slime_qwen35_compatibility_blocked`；`go_day29=false`。记录：`../artifacts/eval/day26-slime-qwen35-day29-go-no-go.json`。Day 29 不得启动替代 release/model/image/topology recipe。
