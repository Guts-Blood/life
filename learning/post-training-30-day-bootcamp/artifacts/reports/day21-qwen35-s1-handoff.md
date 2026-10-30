# Day 21 — Qwen3.5 S1 downstream-ready handoff

日期：`2026-08-13`

状态：`downstream_ready`

## 结论

RSI v0002 operational winner `main-s20260809-lr1e-4-final` 已完成
append-only checkpoint recovery、winner-only merged export、adapter/merged 两个
fresh process 的严格 token-ID parity，以及 self-hashed promotion/downstream key。
最终验证通过；Day 22 rollout 与后续 DPO/GRPO 可以只按下面的 downstream key
消费该 S1，不得用 Base 或未合并 adapter URI 替代。

```text
downstream_key = s1:qwen35-4b:c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a
promotion_manifest_sha256 = 64e6a6bd61951d1f10eb4def520273cb056625c1b041b68ba861c6d1e15d7a6c
inference_export.manifest_sha256 = 660eed4af7f76796631561275f0190c402952520a8ccce358289e6269fb8f8d3
```

远端 merged inference export：

```text
/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/exports/main-s20260809-lr1e-4-final-merged
```

远端 resumable archive 与 rollout contract：

```text
/root/autodl-tmp/s1-checkpoints/qwen35-4b-s1-main-s20260809-lr1e-4-final/checkpoint-1904
/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/handoff/S1-PROMOTION-MANIFEST.json
/root/autodl-tmp/runs/day20-v3-qwen35-lora-20260812T105141Z/handoff/S1-DOWNSTREAM-KEY.json
```

## 冻结身份

- Base：`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`
- Base snapshot：`a176a3c982da5480a0ca98280848474133d105d2e4ab45dc37ce2c68a7c2195b`
- Winner checkpoint integrity：`d0f72be9751628c9073cc8e4104f16d8620bd598dbb8a1e97f1df9bae51170a3`
- Winner checkpoint snapshot：`c169e0bb55b20951d27a889e4ef0deae3a01fe10acabee804b212d3d8170db0a`
- Adapter：`29dc1a7d676b7f2f88675521bc44cf81dd43b700423df3cf4adf8d6a1c4487c3`
- Inference template：显式 `qwen3_5`，`enable_thinking=false`
- Runtime：ms-swift `4.5.0.dev0` / commit
  `565a1ad586a21d24b23931c52d2c62b49c39bee8`，Transformers `5.12.1`，
  Torch `2.10.0+cu128`，PEFT `0.19.1`

## 验收链

1. 独立子进程重建 RSI collector：`44` items、`status=pass`、inventory
   `9d6a78e9954e37e74fae9698ad6729dc253cb3c8156f3008d86ad24ed7ccab5b`。
2. 将 resumable checkpoint 逐文件复制到新 immutable AutoDL archive；11 个文件与
   source byte manifest 完全一致，optimizer/scheduler/RNG/trainer state 均存在。
3. merged export 有 11 个文件、2 个 safetensors shards、processor/tokenizer assets；
   export files hash 为
   `16bc212df51ecac9a5b3b34062e0ccc880c719d63248f04949be8636550e53be`。
4. GPU0 上分别以 PID `7411`（Base + adapter）和 PID `7747`（merged export）
   fresh load；四个固定短 prompt 使用同 input token IDs、greedy generation，四行
   output token IDs 全部 exact，`4 passed / 0 failed`，没有阈值放宽。
5. promotion 与 downstream key 的自哈希复算通过，CLI verify 返回
   `status=promoted`、`downstream_ready=true`、key `status=active`。

## 本地 byte-exact compact evidence

以下 JSON 与远端发布文件逐字节相同；模型 shards 和约 `187 MiB` resumable
checkpoint 没有复制进 Git。

| Artifact | File SHA-256 | Content/self SHA-256 |
|---|---|---|
| [`checkpoint archive evidence`](../checkpoints/day21-qwen35-s1-checkpoint-archive-evidence.json) | `057806704e53ab7d19aef61a2f9a49b2991071d1aafcbba6c004edb9c1e88bd0` | `c781ed2404f7ab8cbb10d51b1faf711358f60c552534038779e5ea528254f88e` |
| [`merged export manifest`](../checkpoints/day21-qwen35-s1-merged-export-manifest.json) | `9c196e43f633116d7fc804870dcfe961db5ae0d216a15f5e21d679d8c1f14b5a` | `660eed4af7f76796631561275f0190c402952520a8ccce358289e6269fb8f8d3` |
| [`adapter capture`](../checkpoints/day21-qwen35-s1-adapter-fresh-capture.json) | `8d5c34b556b05d57f7679b8c067a89050f68620eebf89a6a1ca9222803cc7c24` | `bdcd04e0779869ca8f5b59726e298d7694a83d1631e125f0a8ee640b20d440f4` |
| [`merged capture`](../checkpoints/day21-qwen35-s1-merged-fresh-capture.json) | `db2af58635f35a165d0386677e64751f70da529f04590f00c0f4b230b3b7082a` | `6ded91a67250decf48c5c66aa64bd2069defe62bc5bed09e3f73066399dee3cf` |
| [`exact parity`](../checkpoints/day21-qwen35-s1-adapter-merged-parity.json) | `ded244ef1813d898fb08eb64e773a0390d1c42e252a1fcc70e45a76292ee6bea` | `93299b61322ecc45eba86e995f63216ef76264ac239c4894d374bd9515a07dc7` |
| [`promotion manifest`](../checkpoints/day21-qwen35-s1-promotion-manifest.json) | `40c76f690dcb73805250e0036f8e124ef07b8d2d32b1723637656ae4304d94f8` | `64e6a6bd61951d1f10eb4def520273cb056625c1b041b68ba861c6d1e15d7a6c` |
| [`downstream key`](../checkpoints/day21-qwen35-s1-downstream-key.json) | `979e599aa6c9c05bec9a14144b6ebae689dbdf8315a9112caaec09b5469bd545` | `4a3a467232288feb675eb7d15cc00ff95bc7255a1c4dc2d1debdd55da5831241` |
| [`CLI verify output`](../checkpoints/day21-qwen35-s1-handoff-verify.json) | `249f7f5837477b31059ea1eb56b5620f7ae44ab4ed6cbb11c2f64c7a8fb1d9a6` | verifier recomputed promotion/key hashes |

实现与回归测试见
[`day21_s1_handoff.py`](../../day-21-weekend-eval-reading/day21_s1_handoff.py) 和
[`test_day21_s1_handoff.py`](../../day-21-weekend-eval-reading/test_day21_s1_handoff.py)。

## Claim boundary 与剩余运维风险

- downstream-ready 表示上述 merged export 通过冻结 loader/template 与四个固定 prompt
  的严格 conversion parity；它不是全输入空间等价证明。
- fixed-full112 qualification 仍只支持 operational selection 与 two-training-seed
  same-suite reproducibility，不证明统计唯一最优或独立 held-out 泛化。
- 紧凑证据已本地归档，但 model shards 与 resumable checkpoint 仍在 AutoDL；长期对象存储
  备份仍是运维 durability 工作，不影响当前远端 rollout，但不可把 compact JSON 当作权重备份。
- 该 handoff 只解除 S1 parent gate；formal preference pairs、人工盲审与 DPO 训练各自仍有
  独立 gate。
