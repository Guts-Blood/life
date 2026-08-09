# Day 26 — slime × Qwen3.5 Compatibility Verification（No Fallback）

日期：`2026-08-21`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

只回答一个问题：官方 slime release 中是否存在一条能对 Day 21 promoted Qwen3.5 SFT anchor 正确完成 model/processor load、rollout schema、LoRA/learner train 与 weight sync 的可复现路径。今天不启动正式 RL；`v0.3.0` 只保留为历史架构阅读基线，不能被默认当作 Qwen3.5 runtime。

若没有兼容路径，结论必须是 `slime_qwen35_compatibility_blocked`，Day 29 随之 blocked。禁止换 0.6B、Qwen3 或其他模型完成一个无关 recipe 后称为 active track 成功。

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
- S0–S5 logs、conversion/hash inventory、resolved codepath 与 Day 29 go/no-go record

## 验收

- [ ] 实际 runtime release/tag/SHA/image 有官方依据并已冻结，或明确记录 no-supported-release。
- [ ] Qwen3.5 model/processor/GDN/LoRA 与 Sample→rollout→train→sync 每条边有真实 code/runtime evidence。
- [ ] Day 24 reward/schema 在该 runtime 下可重放。
- [ ] 不使用任何不同模型 fallback。
- [ ] 输出唯一 `go_day29` 或 `slime_qwen35_compatibility_blocked` 结论。

## Daily Log

### Candidate releases / selected runtime

### Qwen3.5 load/conversion evidence

### S0–S5 results

### Day 29 go/no-go
