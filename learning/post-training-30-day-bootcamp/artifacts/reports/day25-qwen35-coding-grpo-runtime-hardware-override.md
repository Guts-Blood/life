# Day 25 runtime hardware override

- Recorded at: `2026-08-16T06:37:46Z` (`2026-08-16T14:37:46+0800`)
- Authority: explicit user correction in the active Day 25 goal thread
- SSH target: `autodl-36153`
- Host: `autodl-container-6acf40b03d-62d686fe`
- Selected device: physical GPU 0, UUID `GPU-8ba9b2a0-4467-b92c-4fca-fcad8cc0fc5b`
- Observed model: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Observed memory: `97887 MiB`
- Excluded device: physical GPU 1, UUID `GPU-d2cd44ac-b11c-6852-02f8-083662f35118`

The runbook's H100 80GB hardware label is incorrect for the already-rented target. The user explicitly authorized proceeding on the selected RTX device and instructed that this mismatch must not block training.

This override is limited to runtime hardware identity and capacity. It does not modify the frozen Day 25 CPU contract, data, GRPO training hyperparameters, gate order, promotion thresholds, confirmation authorization, or candidate-selection rules. Every GPU process must run with `CUDA_VISIBLE_DEVICES=0`, so the trainer sees exactly one CUDA device. The frozen minimum free-memory fraction remains `0.15`, evaluated against the selected RTX device's observed total memory.
