# Day 25 — Coding GRPO 晋级 Eval Freeze

日期：`2026-08-16`  
状态：`frozen_before_gpu_eval_outputs`  
Eval contract SHA-256：`4a0060486184694f7caae505084d8939cf01eaeb510cdc0069f1b0fe749b8cd9`

## 结论

Day 25 的 code candidate 晋级尺子已在任何 S1/GRPO GPU eval output 产生前冻结。评测分两层：search40 只允许使用一次；通过全部 conjunctive gates 后，才允许打开 model-output blind confirmation24。任何一层失败都得到合法的 no-candidate 结论，不通过改阈值、换中间 checkpoint 或重训来补救。

该 contract 是附加评测 trust root，不修改已冻结的训练 CPU contract `59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c`。

## 比较对象

| Role | 冻结对象 |
|---|---|
| Baseline | Day 21 promoted merged S1 `qwen35-4b-s1-main-s20260809-lr1e-4-final` |
| Candidate | 仅 fresh G4 `g4_bounded_short_run/checkpoint-10` |
| 禁止候选 | G2、G3、G4 checkpoint-1…9、early-stopped/incomplete G4、Day 23 DPO checkpoint |

两边使用同一张 GPU、同一 processor/template、`enable_thinking=false`、non-thinking prefix、greedy pass@1、seed `20260820`、completion cap `512` 和 batch size 1。评测完成前不得按输出挑 checkpoint。

## Eval 数据

| Suite | 规模 | 角色 | 访问条件 |
|---|---:|---|---|
| search40 | 40 | 单次 candidate screen | G0–G3 和完整 G4 checkpoint-10 全部通过 |
| confirmation24 | 24 | 单次 model-output blind confirmation | sealed search40 decision 通过全部 gates |

Confirmation24 从尚未进入 Day25 train32/search40 的 Day 22 seed families 中确定性选择：12 dev + 12 heldout；另有 1 dev + 1 heldout 保持 quarantine，不进入本日决策。Train、search、confirmation 三者 family overlap 均为空。

Confirmation24 file SHA-256：`b3cec2c9ba0aee29d89075ac65f28ec6c286c6cfd83934b684e87269194ceae7`。

## 晋级阈值

所有条件是合取关系，不接受 aggregate score 抵消 guardrail failure。

| Gate | search40 | confirmation24 |
|---|---:|---:|
| Candidate correct gain | `>=3` | `>=2` |
| Paired net wins | `>=3` | `>=2` |
| Regressions | `<=2` | `<=1` |
| Format-valid count loss | `<=1` | `<=1` |
| Truncation count increase | `<=0` | `<=0` |
| Mean completion token multiplier | `<=1.50× S1` | `<=1.50× S1` |
| Infra errors | 两边均为 0 | 两边均为 0 |

逐题以全部冻结 MBPP tests 通过、format valid、非 length finish 为 correct。诊断额外报告 wins/regressions/ties、one-sided exact sign p、固定 seed 的 10,000 次 paired bootstrap 95% interval、status/format/truncation/token counts；统计诊断不能覆盖工程 gate。

## 执行与证据链

```text
frozen suite + frozen model identity
  -> matched greedy completion package
  -> sealed raw text/token/finish evidence
  -> Day 24 fresh E2B verifier
  -> sealed per-task result package
  -> paired S1 vs G4 decision
  -> search pass 才授权 confirmation
```

对应入口：

- `build_day25_promotion_eval.py`：确定性重建数据与 eval contract。
- `generate_day25_promotion_eval.py`：核验 GPU binding、S1/G4 identity 并生成 matched completions。
- `sandbox_day25_promotion_eval.py`：复用 Day 24 verifier；persistent infra 时保留 evidence，但不产出可晋级 result package。
- `score_day25_promotion_eval.py`：纯离线配对统计和 fail-closed decision。
- `test_day25_promotion_eval.py`：selection、threshold、confirmation authorization、tampering 和 verifier reuse tests。

## 决策语义

- Search fail：`closed_no_candidate_confirmation_unopened`
- Search pass、confirmation fail：`closed_no_candidate_confirmation_failed`
- 两层通过：`directional_code_candidate_eligible`

即使两层通过，也只证明当前小型 MBPP engineering gate 上出现稳定方向性 code gain。Confirmation24 不是全球无污染或足以支持强统计声明的大样本，因此 `confirmed capability gain` 和 broader S2 promotion 仍不允许；后者需要新的更大 blind suite 与非 code guardrails。
