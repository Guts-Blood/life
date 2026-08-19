# Day 32 — Qwen3.5-4B Single/TP2 Parity 与 Capacity Plan

状态：`deferred_after_architecture_study`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；2×GPU 短 smoke

## 当前状态

Day 27–30 已转向 Megatron/slime architecture study。本 GPU parity/capacity 任务暂停，不因完成 Day 30 自动开卡；只有用户明确重启执行型 Capstone 后才恢复。

## 主要目标

用 Day 31 冻结的 exact `Qwen/Qwen3.5-4B-Base@revision` 建立 single-GPU correctness reference，再验证 TP2 的 processor output、数据、loss、更新和 checkpoint parity；同时分别核算 full-parameter、LoRA、QLoRA 的 capacity。今天输出 evidence，不替用户选择训练 recipe，也不计算或启动未选择的 teacher。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 32 — TP Parity 与 Capacity](../SCALING-BOOK-READING-GUIDE.md#day-32)。

## 实验

1. 从 exact `S0`、同一 8–32 条固定 coding batch、同一 processor/render contract 和同一 global label-token budget 启动 single-GPU reference。
2. 运行 5-step gate，保存 expanded config、sample IDs、initial logits/loss、grad norm 和权重 checksum。
3. 使用 TP2 运行相同 5 steps；必要时扩至 20 steps观察分叉。
4. 保存 TP checkpoint，退出进程，reload/merge 为 eval checkpoint，运行相同 deterministic predictions。
5. 对 S0 分别计算 full-parameter、LoRA、QLoRA 的 weights/gradients/master weights/optimizer/activation/logits/temporary buffers，并单列 rollout model、KV/cache 与 vLLM/CUDA-graph 余量。
6. 比较 single、TP2、FSDP/ZeRO、offload、activation checkpointing 的容量与通信代价；结论保持 `measured options`，recipe 选择留给后续冻结。
7. 验证 text-only coding batch 没有静默训练或丢弃未声明模块；保存 language model、vision tower、projector、MTP、adapter targets 的 module-coverage evidence。

global batch 公式必须明确区分 DP 与 TP；不能因 world size=2 就把 TP2 的 batch 乘二。

## 必查 TP 细节

- full/linear attention heads、KV heads 与 hybrid state 对 TP degree 的整除；
- vocabulary padding/sharding；
- sequence parallel 与 activation layout；
- loss reduction、RNG/dropout、sample order；
- multimodal loader/processor、checkpoint shard metadata 与 HF/eval conversion；
- NCCL topology、collective bytes、通信暴露时间。

## Evidence-first 产物

- `../artifacts/reports/capstone/day32-single-tp2-parity.md`
- `../artifacts/configs/capstone/day32-single/` 与 `day32-tp2/`
- `../artifacts/reports/capstone/qwen35-4b-capacity-topology-plan.md`
- `../artifacts/reports/capstone/qwen35-4b-module-coverage.md`
- per-rank logs、checkpoint manifest、conversion/eval evidence

## 验收

- [ ] single 与 TP2 实际消费相同 sample IDs 和 label-token budget。
- [ ] 报告 initial/stepwise loss、grad norm、parameter diff 的容差与首个分叉。
- [ ] TP checkpoint 能在新进程恢复并转换为可评测权重。
- [ ] full/LoRA/QLoRA 三套状态量、activation/temporary peak 和 rollout 余量分账，不把 LoRA trainable parameters 当总显存。
- [ ] 没有自动选定 full、LoRA 或 QLoRA；后续 recipe 必须引用本日 measured evidence。
- [ ] 未解析、未下载、未训练任何 teacher candidate。
- [ ] 若 TP2 更慢，通信证据能解释；不把速度下降当 correctness failure。

## Stop Conditions

出现 unexplained processor/render、sample-order、loss-reduction、module coverage、checkpoint-conversion 或 rank hang 时停止，不进入 S1 coding SFT 或 S2 direct RL。
