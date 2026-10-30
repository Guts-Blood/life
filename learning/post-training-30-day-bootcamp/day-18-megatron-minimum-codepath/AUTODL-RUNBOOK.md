# Day 18 AutoDL Handoff

Current status (`2026-08-10`): **historical handoff; Day 18 is `closed_pass_c0_c5` and must not be automatically rerun.** The actual 2×H800 run passed C0–C5; local closeout revalidated the evidence bundle and tests. See [`day18-close.md`](../artifacts/reports/day18-close.md).

The original pre-run boundary was: CUDA, GDN, TP/DP, checkpoint, and export compatibility remained unverified until the H800/H100 Hopper run; a prepared script was not a passed gate. The remainder of this file preserves that handoff and replay procedure as historical documentation.

## Calendar and machine choice

The curriculum calendar puts Day 14 on `2026-08-09` and explicitly makes it CPU-only. The first Qwen3.5 GPU window is Day 15 on `2026-08-10` (`1x H100 80GB`); the planned Day 18 window is `2026-08-13` (`2x H100 80GB`, same host).

If Day 18 is intentionally pulled forward, use one same-host machine with exactly `2x H800 80GB` or `2x H100 80GB` (Hopper compute capability 9.0), at least 64 GB host RAM (128 GB preferred), and at least 350 GB of data disk (400 GB preferred). Do not split the two GPUs across nodes.

The disk number is deliberately larger than a normal 4B smoke: C2, C3, and C4 each retain a full-parameter Adam distributed checkpoint with model, FP32 main parameters, two FP32 moment tensors, and RNG state. C5 adds one final model-only MCore checkpoint and one HF export; it deliberately does not retain another Adam/RNG state. Together with the uploaded/extracted base, C0 MCore copy, C1 HF round-trip copy, C4/C5 HF exports, package cache, and build temporary files, a 300 GB disk is not a safe retention target. After upload and extraction, the script requires at least 280 GiB free and always retains 15% filesystem headroom.

At the time of this run, Day 15 was still a formal prerequisite in the curriculum and remained `not_started`. Pulling this run forward produced standalone C0–C5 evidence; C5 overlapped the v2 tiny-overfit learnability question but did not by itself backfill Day 15 M1–M5. On `2026-08-09`, the later Day 15 close decision accepted the combined Day 18–20 evidence as superseding close evidence without fabricating the missing ordered artifacts or promoting S1; see [`day15-close.md`](../artifacts/reports/day15-close.md). No additional Day 15 GPU rental is required.

The selected AutoDL boot image does not expose a verifiable OCI digest inside the instance. Record the exact observed boot-runtime fingerprint instead of copying an unrelated registry digest:

```text
autodl-observed-runtime:ubuntu22.04.4-py3.12.3-torch2.5.1+cu124-cuda12.4
```

This reference records what was directly observed on `2026-08-08`: Ubuntu 22.04.4, base Python 3.12.3, base Torch 2.5.1+cu124, and `/usr/local/cuda-12.4`. Training does not use that base Torch. It uses the isolated `/root/autodl-tmp/qwen35-v2/venv` with Python 3.12, Torch 2.10.0+cu126, and the pinned Day 18 packages. Both boot and isolated runtime facts are written into evidence.

The `bootstrap` gate consumes that pre-created isolated prefix, installs only the pinned local ms-swift checkout with `--no-deps`, and audits it with `pip check`, `pip freeze`, and `pip inspect`. It deliberately refuses to create a `--system-site-packages` venv or repeat the CUDA extension install.

## Local deliverables

The model and control package are separate so a control-script revision never requires repacking 9+ GB of weights.

Expected model snapshot:

```text
/Users/jiaweiqian/Desktop/python_projects/life-checkpoints/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b/
```

Expected packages after local verification:

```text
/Users/jiaweiqian/Desktop/python_projects/life/tmp/day18-qwen35-ready/day18-control-1001bb4d826a52d1f399e183466143f4da7b741b.tar
/Users/jiaweiqian/Desktop/python_projects/life-checkpoints/packages/Qwen--Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b.tar
```

Every tar has an adjacent `.sha256`. Never upload a snapshot containing `.incomplete` or lock files.

Create both packages only after the local snapshot verifier passes:

```bash
bash /Users/jiaweiqian/Desktop/python_projects/life/learning/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath/package_qwen35_snapshot.sh
bash /Users/jiaweiqian/Desktop/python_projects/life/learning/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath/build_day18_control_bundle.sh
cd /Users/jiaweiqian/Desktop/python_projects/life-checkpoints/packages
shasum -a 256 -c Qwen--Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b.tar.sha256
cd /Users/jiaweiqian/Desktop/python_projects/life/tmp/day18-qwen35-ready
shasum -a 256 -c day18-control-1001bb4d826a52d1f399e183466143f4da7b741b.tar.sha256
```

## Upload after the instance starts

Do not place credentials in the repository. Export the connection only in the local shell:

```bash
export DAY18_REMOTE_HOST='root@YOUR_AUTODL_HOST'
export DAY18_SSH_PORT='YOUR_PORT'
export DAY18_CONTROL_BUNDLE='/Users/jiaweiqian/Desktop/python_projects/life/tmp/day18-qwen35-ready/day18-control-1001bb4d826a52d1f399e183466143f4da7b741b.tar'
export DAY18_MODEL_BUNDLE='/Users/jiaweiqian/Desktop/python_projects/life-checkpoints/packages/Qwen--Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b.tar'
bash /Users/jiaweiqian/Desktop/python_projects/life/learning/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath/upload_day18_bundles.sh
```

The upload is resumable with `rsync` when both ends provide it. Otherwise it falls back to `scp`. It refuses non-empty extraction targets, then verifies both tar hashes, both internal manifests, and the extracted model snapshot before doing any install.

## Remote tmux and gate order

```bash
tmux new-session -A -s day18
cd /root/autodl-tmp/post-training-30-day-bootcamp/day-18-megatron-minimum-codepath
export DAY18_BOOT_IMAGE_REFERENCE='autodl-observed-runtime:ubuntu22.04.4-py3.12.3-torch2.5.1+cu124-cuda12.4'
```

Run one command at a time. Inspect its JSON evidence before proceeding. The runner also writes pass markers and refuses to skip or reorder prerequisites; `sync-evidence` remains available after a failed gate so blocker evidence can still be recovered.

```bash
bash run_day18_autodl.sh init
bash run_day18_autodl.sh host-inventory
bash run_day18_autodl.sh bootstrap
bash run_day18_autodl.sh model-preflight
bash run_day18_autodl.sh runtime-preflight
bash run_day18_autodl.sh topology-preflight
bash run_day18_autodl.sh c0-c1
bash run_day18_autodl.sh c2-dp2
bash run_day18_autodl.sh c3-tp2
bash run_day18_autodl.sh c4-resume-export
bash run_day18_autodl.sh c5-tiny-overfit
bash run_day18_autodl.sh finalize
bash run_day18_autodl.sh sync-evidence
```

The current run root is recorded in:

```text
/root/autodl-tmp/qwen35-v2/state/current-run-root
```

Each gate refuses to overwrite an existing output directory. C2 is `TP=1, DP=2`; C3 is `TP=2, DP=1`. Both use the same two rows, global batch 2, and one logical microbatch per step. C4 is a new process that loads optimizer and RNG state from iteration 3 and runs to total iteration 5. C5 independently restarts from C0 with the same TP2 training recipe for 150 updates and saves only the final model to control checkpoint I/O and disk use.

Record every environment amendment, observed difference, and runtime defect before `finalize`; use `open` only for an unresolved blocker and `resolved`/`observation` otherwise:

```bash
bash run_day18_autodl.sh record-problem \
  --gate bootstrap --category runtime_amendment --status resolved \
  --summary 'AutoDL booted CUDA 12.4; Day18 ran from the audited isolated CUDA 12.6 prefix.' \
  --evidence evidence/runtime-preflight.json
```

`finalize` refuses to emit the immutable `DAY18-PASS.json` unless all C0–C5 markers and required JSON evidence pass and `problems.jsonl` has no open record. The pass manifest states the bounded claim explicitly and is included in the evidence tar.

## What each gate proves

- `model-preflight`: exact revision, architecture, processor, 738 tensor mappings, file sizes, and all SHA-256 hashes.
- `runtime-preflight`: Python/Torch/direct dependency pins, CUDA/BF16/NCCL imports, exact conditional processor, and an evidenced mcore-bridge GDN source with `USE_MCORE_GDN=1`.
- `topology-preflight`: two unique same-host H800/H100 Hopper GPUs (CC 9.0, at least 80000 MiB each) and a real NCCL all-reduce.
- `c0-c1`: HF to MCore conversion, an exact 15-tensor MTP HF→MCore→HF round-trip, plus both fixed text-only rows' loss-token logits and weighted single-rank HF/MCore main-loss parity.
- `c2-dp2`: three real optimizer steps, finite/nonzero MTP gradients and loss, a distributed checkpoint with optimizer/RNG state, then a fresh-process load-only check of that state.
- `c3-tp2`: the same logical batch through supported TP2 for three steps, compared with C2.
- `c4-resume-export`: fresh-process iteration 3 to 5 resume, TP2 checkpoint to HF reshard/export, reload, and parity.
- `c5-tiny-overfit`: same-entrypoint learnability on the frozen two rows; fresh HF reload must reach ≥95% teacher-forced token accuracy, ≥80% relative loss reduction and loss ≤1.0, with changed main/MTP tensors and bitwise-exact frozen visual/aligner tensors. This is not a generalization or exact-resume claim.

## Hard stops

Stop at the first failed JSON gate. Keep the run directory and write `compatibility_blocked_at_Cx`; do not change thresholds or substitute another Qwen model.

Stop immediately for revision/hash drift, package drift, boot-image reference drift, processor/loader mismatch, missing/randomly initialized MTP tensors, missing/non-finite MTP loss, missing/non-finite/zero MTP gradients, unexpected visual tokens/tensors, trainable or changed ViT/aligner state, unevidenced GDN fallback, non-finite main loss/gradient norm, TP/NCCL hang, missing optimizer/RNG state in C2–C4, resume mismatch, C5 learnability failure, parity failure, disk above 85% used, or post-step VRAM headroom below 10%.

Before shutting down, download the evidence tar printed by `sync-evidence` and verify its adjacent hash locally. The small evidence tar intentionally excludes model/checkpoint tensors; retain or separately transfer the full C4 checkpoint/HF export and C5 final model/HF export until the report is accepted.
