# Day 15 — Qwen3.5-4B Onboarding、Lineage v2 与最小训练 Gate

日期：`2026-08-10`  
状态：`not_started`  
强度：4–5 小时

## 主要目标

为后续 active track 冻结 `Qwen/Qwen3.5-4B-Base` 的可复现入口，并重新验证 processor/template、数据 tokenization、Base dev baseline、tiny overfit、fresh-process resume 与显存边界。今天不做完整 packing ablation；任何 onboarding gate 未通过就停在该层。

Day 01–12 的 Qwen3-0.6B artifacts 全部保持历史只读。新文件使用 v2/model-specific 命名，不覆盖旧 rendered text、token IDs、manifest、prediction 或 checkpoint。

## Gate 0：Revision、环境与模型范围（45 分钟）

- 从 [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) 解析 exact commit；记录 repo ID、revision、config/weight hashes 与下载时间。
- 从 [ms-swift Qwen3.5 Best Practice](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md) 和 [Supported Models](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md) 选择并冻结实际可运行的 ms-swift、Transformers、PyTorch、CUDA 与 kernel 版本；现在不把滚动 `main/latest` 写成 revision。
- 保存 resolved loader/model class、processor/tokenizer class、dtype 与完整 config；确认它不是旧的纯文本 `AutoModelForCausalLM` 直接换 ID。
- Core 范围冻结为 **text-only coding**：训练样本不含 image/video；显式记录 ViT/aligner 的 freeze policy。即使视觉模块冻结，checkpoint 和 processor 仍按多模态模型记录。

Gate 0 产物必须写真实解析值；模板中的 `pending` 只有在失败退出时才允许保留。

## Gate 1：Processor、Template 与 Loss Contract（60 分钟）

- 复用 Day 08 的 raw message fixtures 和 edge-case 意图，使用 Qwen3.5 processor/template 重新 render、tokenize、构造 labels 与 causal shift。
- 重新冻结 chat template、special token IDs、EOS/PAD、assistant span、empty-think 处理、truncation 与 zero-supervision rejection。
- 逐 token 审计至少 10 个正向 case 和 10 个 edge cases；任何旧 Qwen3 token ID/hash 只作历史对照，不能进入 v2 assertion。
- 测试 text-only 输入不会意外插入 image/video tokens，ViT/aligner freeze 与 trainable-parameter inventory 一致。

## Gate 2：Retokenize 与 Manifest v2（45 分钟）

- 复用 Day 09 已审核 raw sample IDs、source split、去重/去污染决定；重新生成 Qwen3.5 rendered/token/label records。
- 重算 encoded、non-padding、supervised 和 truncated token 数；以新 supervised-token 口径生成 coding SFT mixture 与 budget。
- 新 manifest 记录 parent v1 raw-data manifest hash，但拥有独立 processor/template/data hashes。
- Day 10 的 task/scorer schema 可复用，但旧 dev 已支持十轮连续选择，只能作为历史诊断；为 v2 冻结新的 selection dev 与 confirmation raw IDs，并新建 render key、prediction path 和 baseline summary。

## Gate 3：新 Base Dev Baseline（45 分钟）

- 在训练前运行 Qwen3.5 Base 的新 v2 selection **dev** slices，保存逐样本 prompt/render、generation、score、status 与 bad cases。
- 保持 Day 10 scorer 语义；如果模型差异要求 adapter/parser，单独版本化并证明 scorer 语义未变。
- 旧 48 条 frozen test 继续封存且不进入 v2 selection；v2 confirmation 另行冻结，只在候选最终选定后消费一次。
- v1 0.6B 和 v2 4B 的 aggregate 可并列描述，但不把模型规模差异当成训练增益。

## Gate 4：LoRA Tiny-overfit 与 Fresh-process Resume（75–90 分钟）

- 从 v2 数据中冻结 16–32 条短 coding/general 样本，保存 raw/render/token/label hashes。
- 使用 Day 16 计划采用的 LoRA 语言模块范围；ViT/aligner 按 Gate 0 policy 冻结。
- 先完成 1 optimizer-step audit，再在预注册上限内达到 tiny-overfit 阈值；记录 supervised-token-weighted accumulation、loss、grad norm、LR、trainable-parameter checksum 与 generation。
- 运行 uninterrupted 与 `save -> 退出进程 -> resume` 的短对照；恢复 adapter/full trainable state、optimizer、scheduler、RNG 与 sampler cursor。
- Base、early、final 使用完全相同的 v2 processor/template/eval path。

## Gate 5：显存与长度 Preflight（30 分钟）

只做测量，不做完整 packing 实验：

- 分别记录 model load/inference、1-step LoRA train、save/load 和 resume 的 peak allocated/reserved VRAM、CPU RAM、disk 与 wall time。
- 在 512/1024/2048 的候选训练长度中只做 forward/backward smoke，遇到 OOM 即记录边界，不同时改 batch、checkpointing 或 dtype。
- 核对当前 pinned stack 是否声称支持 Qwen3.5 packing/padding-free；今天只跑 boundary contract tests，不据此宣布 throughput winner。
- 基于实测为 Day 16 冻结 microbatch、accumulation、length 候选和硬停止阈值。

## 资源与租卡

- CPU 完成 revision、hash、retokenize 和 manifest；GPU 只用于 Base baseline、tiny-overfit/resume 和显存 preflight。
- Planning 上限为同机 1×H100 80GB；这不是官方最低显存声明。若 1-step LoRA gate 不满足余量，立即停止并把 Day 16 topology 标为待重算，不用 offload 临时掩盖容量问题。
- 所有模型缓存、configs、逐 step 指标、predictions 和 checkpoints 同步后关机。

## Evidence-first 产物

- `../artifacts/configs/day15-qwen35-transition.json`
- `../artifacts/data/day15-qwen35-processor-template-contract.json`
- `../artifacts/data/day15-qwen35-sft-manifest.json`
- `../artifacts/eval/day15-qwen35-base-dev-predictions.jsonl`
- `../artifacts/logs/day15-qwen35-tiny-overfit/`
- `../artifacts/reports/day15-qwen35-onboarding.md`

## 验收

- [ ] exact model revision、weights/config、runtime 与 loader/processor 已解析并冻结。
- [ ] text-only/ViT/aligner policy 和 trainable-parameter inventory 一致。
- [ ] v2 render/tokens/labels/manifests 全部新建，未覆盖任何 Day 01–12 artifact。
- [ ] 新 Base dev baseline 已落盘；v2 confirmation 未消费，旧 v1 frozen test 仍 sealed/unconsumed。
- [ ] LoRA tiny-overfit、fresh-process resume 与 save/reload 全部通过。
- [ ] Day 16 的长度、microbatch、显存余量和停止条件来自实测。

## Daily Log

### v1 → v2 lineage boundary

### Frozen revisions / loader / scope

### Contract and manifest rebuild

### Base baseline / tiny-overfit / resume

### Memory preflight and Day 16 gate
