# 可复制 Goal Prompt — 继续 Day 25 GPU 工作

复制下面整段到持有 GPU terminal 的新会话：

```text
请创建并持续推进一个 goal，objective 是：

“在当前已经租用的单张 H100 80GB 上，严格按照冻结的 Day 25 Qwen3.5-4B Coding GRPO 契约，从 promoted Day 21 S1 完成 GPU binding、G1 rollout-only、G2 exactly-one-update、G3 next-version weight-sync probe；只有全部 gate 通过才运行 fresh G4 10-step bounded run，并按冻结的 search40→confirmation24 paired Code Eval 做出 directional candidate / no-candidate 的最终结论，保存所有可回放证据。不得修改冻结数据、训练超参、晋级阈值或候选选择规则。”

不要设置 token budget。你已获授权使用当前已打开的 GPU 和任务范围内的 E2B；不要创建或购买额外 GPU，不要销毁实例，结束时提醒我同步证据和关卡。

工作目录是 life repo 下的：
learning/post-training-30-day-bootcamp

首先完整阅读并服从：
1. day-25-grpo-small-model-lab/DAY25-GPU-RUNBOOK.md
2. day-25-grpo-small-model-lab/README.md
3. artifacts/reports/day25-qwen35-coding-grpo-cpu-preflight.md
4. artifacts/reports/day25-qwen35-coding-grpo-promotion-eval-freeze.md
5. artifacts/configs/day25-qwen35-coding-grpo/cpu-run-contract.json
6. artifacts/configs/day25-qwen35-coding-grpo/promotion-eval-contract.json

固定 trust roots：
- Training CPU contract: 59e4a07abbed6581650784b264fb43da81ce913c74c81f1be2b7880504f3ed2c
- Promotion eval contract: 4a0060486184694f7caae505084d8939cf01eaeb510cdc0069f1b0fe749b8cd9
- Pinned ms-swift commit: 565a1ad586a21d24b23931c52d2c62b49c39bee8
- Parent: Day 21 promoted merged S1 main-s20260809-lr1e-4-final
- Day 23 checkpoint is forbidden as parent or candidate.

开始时先检查当前 terminal、GPU、repo 路径和已同步文件，不要假设本地路径等于远端路径。若 GPU 端缺少本轮新增文件，先以不覆盖远端运行证据的方式同步 Day 25 源码和 artifacts。绝不打印、复制到日志或提交 E2B secret。

在消耗训练 GPU 前依次运行：
- 验证已持久化的 artifacts/eval/day25-qwen35-coding-grpo-cpu-preflight.json seal、状态和 trust roots。只有远端也具备本地 parse-only Qwen3.5 snapshot 时才重跑 run_day25_cpu_preflight.py；不要为重复 CPU 审计下载第二份模型。
- python3 day-25-grpo-small-model-lab/build_day25_promotion_eval.py --mode check
- python3 day-25-grpo-small-model-lab/test_day25_promotion_eval.py
- git -C "$DAY25_REPO/vendor/ms-swift" rev-parse HEAD 与 git status --porcelain
- nvidia-smi、目标 Python package/version/import-path 检查

然后严格执行：

G0 / binding
- 只绑定远端 promoted merged S1 和当前单张 H100。
- bind_day25_qwen35_gpu.py 必须成功并创建全新 run root。
- binding 只代表 config/runtime/lineage preflight，不代表模型已成功训练。

G1 / rollout-only
- DAY25_ABORT_AFTER_REWARD_BATCH=1。
- 预期在 sealed reward batch 落盘后出现 intentional G1 sentinel 和非零退出。
- 必须没有 checkpoint-1。
- audit_day25_rewards.py 必须 pass、2 个完整 G=4 groups、零 persistent infra。
- 记录显存；任一采样 free fraction <15% 即停止。

G2 / one update
- fresh S1 + fresh LoRA，不能 resume G1。
- max_steps=1，exit 0，reward audit pass，checkpoint-1 durable。
- loss、grad norm、ratio、entropy、old/current logprob 相关证据必须 finite。
- 不把 G2 checkpoint 当能力候选。

G3 / weight sync
- fresh S1 + fresh LoRA，max_steps=2。
- ledger 必须同时出现 g3:trainer-step-0 和 g3:trainer-step-1。
- 核对 colocate vLLM 在 global step 变化后重载/同步 LoRA 的日志与源码路径。
- checkpoint-2 只用于机制证据，不是能力候选。

G4 / bounded candidate run
- 仅 G0–G3 全通过后运行；否则 goal 以合法 blocked/no-run 证据结束。
- fresh S1 + fresh LoRA，严格使用冻结 config，完整 max_steps=10。
- 唯一有资格进入晋级 eval 的 checkpoint 是 g4_bounded_short_run/checkpoint-10。
- G4 中间 checkpoint、early-stopped 或 incomplete run 都没有候选资格。

晋级 Code Eval
- 先使用 generate_day25_promotion_eval.py 为 S1 与 G4 checkpoint-10 生成 matched greedy search40 completions。
- 使用 sandbox_day25_promotion_eval.py 逐模型复用 Day 24 fresh E2B verifier；persistent infra 时不得生成可晋级 results。
- 使用 score_day25_promotion_eval.py 生成 sealed search decision。
- Search40 失败时立刻结论 closed_no_candidate_confirmation_unopened，绝不打开 confirmation24。
- 只有 search_gate_pass_confirmation_authorized 才能为两模型生成和评分 confirmation24，并把 search decision 传给 generation 和最终 scorer。
- Confirmation 失败：closed_no_candidate_confirmation_failed。
- 两层通过：directional_code_candidate_eligible。
- 即使通过，也不得声明 confirmed capability gain 或 broader S2 promotion。

晋级阈值不得修改：
- search40：correct gain>=3，paired net wins>=3，regressions<=2；format-valid loss<=1；truncation 不增加；mean tokens<=1.5x S1；零 infra。
- confirmation24：correct gain>=2，paired net wins>=2，regressions<=1；其余 guardrails 相同。

全程每个 stage：
- 使用独立 run ID、ledger、output、log 和 memory CSV；不覆盖已有文件。
- 一个 gate 完成后立即离线审计并汇报关键数值，再进入下一 gate。
- 保留 binding、resolved configs、raw completions/token IDs、sandbox evidence、reward ledger、audit、trainer logs、TensorBoard、memory samples、checkpoint 文件 manifest/hash 和 paired decisions。
- 不因 OOM、无方差、infra、NaN 或 gate failure 静默降低 G、length、LoRA rank、改变 beta/topology/reward/threshold。

立即停止条件：lineage/hash/runtime 漂移；Qwen3.5/vLLM/LoRA load 失败；persistent E2B infra；显存余量<15%；G=4 缩组；reward/advantage replay 不一致；NaN/Inf；checkpoint 不完整；G3 无 step-1 rollout；需要修改冻结 config 才能继续。

请保持 goal active 并自主推进，除非需要新的权限、遇到连续不可消解的同一 blocker，或已经到达最终 terminal conclusion。不要只给我命令让我自己跑；在当前 GPU terminal 中实际执行、验证、保存证据。每次更新都说明：当前 gate、已验证证据、GPU/显存状态、下一步和是否仍符合 claim boundary。

最终交付：
1. Day 25 GPU run root 与不可变 manifest/hash；
2. G0–G4 每个 gate 的 pass/fail/未运行状态和证据路径；
3. 若有 candidate，search40 与 confirmation24 的逐题 paired metrics/decision；
4. S1 与 candidate 的 wins/regressions/status/format/truncation/length 对照；
5. 总 GPU 时间、E2B 执行数/infra、峰值显存与最低 free fraction；
6. 最终只允许以下结论之一：runtime_blocked、mechanism_failed、closed_no_candidate_confirmation_unopened、closed_no_candidate_confirmation_failed、directional_code_candidate_eligible。
```
