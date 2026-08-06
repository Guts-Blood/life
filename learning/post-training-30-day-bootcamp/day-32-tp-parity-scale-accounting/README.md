# Day 32 — 4B Single/TP2 Parity 与 8B Capacity Plan

状态：`not_started`
日期：`unscheduled_after_day30`
强度：4–5 小时人工工作；2×GPU 短 smoke

## 主要目标

用一个单卡可运行的 <=4B checkpoint 建立 single-GPU correctness reference，再验证 TP2 的数据、loss、更新和 checkpoint parity；同时为 8B full-parameter SFT 选择真实可行的 TP topology。

## 理论 / 定向阅读（45–60 分钟）

精读清单：[Day 32 — TP Parity 与 Capacity](../SCALING-BOOK-READING-GUIDE.md#day-32)。

## 实验

1. 从同一 `S0`、同一 8–32 条固定 batch 和同一 global label-token budget 启动 single-GPU reference。
2. 运行 5-step gate，保存 expanded config、sample IDs、initial logits/loss、grad norm 和权重 checksum。
3. 使用 TP2 运行相同 5 steps；必要时扩至 20 steps观察分叉。
4. 保存 TP checkpoint，退出进程，reload/merge 为 eval checkpoint，运行相同 deterministic predictions。
5. 对 8B `T0` 计算 weights/gradients/master weights/optimizer/activation/temporary buffers，比较 TP2、TP4 和是否启用 distributed optimizer/activation checkpointing。

global batch 公式必须明确区分 DP 与 TP；不能因 world size=2 就把 TP2 的 batch 乘二。

## 必查 TP 细节

- attention heads 与 KV heads 对 TP degree 的整除；
- vocabulary padding/sharding；
- sequence parallel 与 activation layout；
- loss reduction、RNG/dropout、sample order；
- checkpoint shard metadata 与 HF/eval conversion；
- NCCL topology、collective bytes、通信暴露时间。

## Evidence-first 产物

- `../artifacts/reports/capstone/day32-single-tp2-parity.md`
- `../artifacts/configs/capstone/day32-single/` 与 `day32-tp2/`
- `../artifacts/reports/capstone/8b-capacity-topology-plan.md`
- per-rank logs、checkpoint manifest、conversion/eval evidence

## 验收

- [ ] single 与 TP2 实际消费相同 sample IDs 和 label-token budget。
- [ ] 报告 initial/stepwise loss、grad norm、parameter diff 的容差与首个分叉。
- [ ] TP checkpoint 能在新进程恢复并转换为可评测权重。
- [ ] 8B topology 由 measured memory/step smoke 选择，不由“想学 TP”决定。
- [ ] 若 TP2 更慢，通信证据能解释；不把速度下降当 correctness failure。

## Stop Conditions

出现 unexplained sample-order、loss-reduction、checkpoint-conversion 或 rank hang 时停止，不进入 8B SFT。
