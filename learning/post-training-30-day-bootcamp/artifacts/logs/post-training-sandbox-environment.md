# `post_training_sandbox` environment evidence

Date: `2026-08-05`

Status: `local_client_layer_passed`

## Identity

| Field | Frozen value |
|---|---|
| Conda environment | `post_training_sandbox` |
| Prefix | `/Users/jiaweiqian/miniforge3/envs/post_training_sandbox` |
| Platform | `macOS-26.5.2-arm64-arm-64bit` |
| Python | `3.11.15` |
| E2B SDK | `2.37.0` |
| RFC 8785 | `0.1.4` |
| Direct dependency contract SHA-256 | `dc2f0bcab5217f1c57b5b456b191192498f237ee23249cdf70a2967d8220afc5` |
| Resolved pip snapshot SHA-256 | `8af157cf4772289a2d898e7f65487d9500f144e3f6aee91ccb18d4d42133bfa7` |

## Verification evidence

- Both pinned packages import successfully.
- `python -m pip check`: passed with no broken requirements.
- The credential is loaded only by the local client from root `.env.e2b`.
- The credential file is Git-ignored, mode `0600`, and never sent into a sandbox.
- A create/write/run/kill smoke test passed against E2B template
  `rki5dems9wqfm4r03t7g`: Python `3.11.6`, envd `0.6.10`, 2 vCPU and
  512 MiB RAM.
- Egress was frozen to `allow_internet_access=false` plus explicit
  `deny_out=["0.0.0.0/0"]`; an HTTPS fetch could not retrieve public data and
  ended at its four-second hard timeout.

The smoke test deliberately treats successful content retrieval—not a bare TCP
connect—as the egress check. E2B's network proxy may accept the initial TCP
connection even while the deny-all policy prevents a public HTTPS response.

## Day 10 formal run

| Field | Value |
|---|---|
| Sandbox contract hash | `sha256:9571c64301029e230ab2d01e1b8daf8a6dcf7c4a5c8b0696e21d9b073a730c87` |
| Complete comparison key | `sha256:850a69854fe0a945d7b86f3c71a59a7977e09dd42a2df96ca7548b6068bd0f14` |
| Code run hash | `sha256:147616a3cd6f02975a2be509b6420703f4334631db0a1f5d2ca421ef6b51a626` |
| Results exact SHA-256 | `d50be508bf284beac8b8e731fd521a68547dccf928d7206938a64d65b264509c` |
| Results | 28 valid scores; 0 infrastructure failures |

The formal runner also passed live pass, assertion-failure, and CPU-timeout
classification smokes before receiving the frozen Base completions. A real
CPU-bound candidate exited with status `152` (`SIGXCPU`); the frozen scorer
therefore classifies that status as a valid candidate timeout rather than an
E2B infrastructure failure.

## Isolation boundary

This environment is only the E2B control plane. It does not generate model
outputs and does not mutate the frozen `post_training_lab` environment used for
the Base run. Each HumanEval completion is executed in a fresh remote sandbox
under the separately hashed Day 10 sandbox contract.
