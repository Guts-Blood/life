# Day 16 Packing Parity

日期：`2026-08-09`
状态：`not_run_packing_false`
追加 GPU 运行：`0`

## 判定

Day 16 不采用 packing，冻结：

```text
packing=false
padding_free=false
```

这不是 parity `pass`。原计划要求 Day 15 packing contract tests 先通过，再运行 unpacked P0 与 packed P1。Day 15 Close 明确保留了 packing 缺口；实际 Day 20 canonical run 也固定使用 unpacked，并没有创建 packed 对照。

因此当前只有一个可审计结论：packing 未被批准，不能声称吞吐提升、loss parity、position parity 或跨样本 attention isolation 已验证。

## 现有证据

- 本地且不纳入 Git 的 `remote-runs/day20-qwen35-lora-20260809T042005Z/DAY20-MANIFEST.json`：runtime contract 固定 `packing=false`、`padding_free=false`、`max_length=2304`；运行包身份由仓库内 [`run_provenance.json`](../../day-20-qwen35-balanced-lora-sft/reports/day20-probe-analysis-20260809/evidence/run_provenance.json) 记录。
- [`Day 15 Close`](day15-close.md)：将 packing correctness/throughput parity 明确移交 Day 16，没有补写通过结论。
- [`Day 16 Gap Audit`](day16-gap-audit.md)：由于 LR probe 已 fail closed，当前不为一个不可进入 main 的 recipe 追加 packing GPU 对照。

## 未来采用 packing 的进入条件

新 charter 必须用同一批 raw IDs、processor/template、parent S0、data order 与有效 supervised-token budget 对照 P0/P1，并逐样本验证：

- assistant boundary 与 labels 完全一致；
- position IDs 与 cross-sample attention isolation 正确；
- 无跨样本 loss 泄漏；
- finite loss/grad，且数值差异在预注册容忍范围内；
- 有效 label tokens/s、step time 与 peak allocated/reserved memory 有实际收益；
- 任一 correctness 差异都使 packed 路径作废。

在这些证据产生前，唯一允许的值仍是 `packing=false`。
