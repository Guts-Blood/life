# Day 11 AutoDL Runbook

This runbook is intentionally operational. The theory and acceptance criteria remain in `README.md`.

## Frozen execution contract

- GPU: one NVIDIA RTX 4090 24GB; do not select V100, a non-CUDA accelerator, or a GPU below 20 GiB.
- Image: Ubuntu with Python 3.10/3.11, CUDA 12.x, and a CUDA-enabled PyTorch 2.4 or newer.
- Model: `Qwen/Qwen3-0.6B-Base` at revision `ddc928429ed09d9ad603fd762053d0434c15e865`.
- Precision: FP32 trainable parameters/gradients/Adam state, BF16 autocast compute, and FP32 cross-entropy. The preflight fails closed if CUDA/BF16/model/data/template/disk checks do not pass.
- Data: 18 frozen records, 2,194 encoded tokens, 904 shifted supervised tokens, no packing.
- Main run: micro batch 1, accumulation 4, AdamW, LR `1e-4`, 5-step linear warmup then constant, clip norm 1.0, at most 150 optimizer steps.
- Success: exact fresh-process resume comparison passes and full tiny-set teacher-forced token accuracy reaches at least 95%.

## Upload and run

Upload `tmp/day11-ready-upload.tar` and its adjacent `.sha256` file from the local repository to `/root/autodl-tmp/`. In the AutoDL terminal run:

```bash
cd /root/autodl-tmp
sha256sum -c day11-ready-upload.tar.sha256
tar -xf day11-ready-upload.tar
cd /root/autodl-tmp/day11-ready
bash run-day11.sh
```

The entrypoint performs, in order:

1. checks the image's CUDA-enabled PyTorch and installs only the four pinned Python packages missing from the base image;
2. verifies GPU BF16 support, at least 20 GiB VRAM, at least 35 GiB free disk, every frozen model/data file hash, template hash, and label count;
3. runs a 6-step uninterrupted reference;
4. runs 3 steps, saves a full checkpoint, starts a fresh process, and resumes to step 6;
5. requires exact sample order, loss, gradient, LR, model, optimizer, scheduler, RNG-dependent trajectory, and sampler cursor agreement;
6. runs Base/early/final evaluation and the 50–150 step main tiny overfit;
7. exits successfully only after writing `DAY11-PASS.json`.

Do not edit the YAML after a failed preflight. Preserve the error and ask for diagnosis; changing batch, dtype, masking, or LR would invalidate the frozen run.

## Expected outputs

Find the completed run with:

```bash
find /root/autodl-tmp/runs -name DAY11-PASS.json -print
```

Before shutting down, download or back up:

- the `DAY11-PASS.json` file;
- the completed run's `main/` directory, including early and final checkpoints;
- `resume-interrupted/checkpoints/checkpoint-step-000003/`;
- `post-training-30-day-bootcamp/artifacts/configs/day11-*`;
- `post-training-30-day-bootcamp/artifacts/data/day11-*`;
- `post-training-30-day-bootcamp/artifacts/logs/day11-*`;
- `post-training-30-day-bootcamp/artifacts/reports/day11-*`.

The deliberately memorized Day 11 final checkpoint is evidence for pipeline correctness. Day 12 must start independently from the same frozen Base revision, not from this checkpoint.
