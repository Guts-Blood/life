#!/usr/bin/env python3
"""Fail-closed static audit for the pinned Day 26 slime release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SLIME_TAG = "v0.3.1"
SLIME_SHA = "a6272da0d4f3d0a08520c99a2f3b4f6c887960dc"
DOCKER_TAG = "nightly-dev-20260804a"
DOCKER_DIGEST = "sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7"

EXPECTED_FILES = {
    "docker/version.txt": "34ee9066e159add2be4bc616ca9b76d940d63008db8b0cf1dfaf0d0a91a240ca",
    "docker/Dockerfile": "f5aa2b74b4dd22d9a3014d956c23e18c1d715be24702b19a3ec789ef965d8d14",
    "requirements.txt": "860590f9a7272889465ea157bda7025b1f28932a0dbcd05dd2eb2be2a9dd292f",
    "scripts/models/qwen3.5-4B.sh": "b0f36a96bb99a7bcacee3076e316d487c100f100963cc03cd8a36226085e4f39",
    "slime_plugins/models/qwen3_5.py": "905c78ee17e70f979a23523a0d29cb60f4a55af909d673ab61b8909c6a324b9a",
    "slime/backends/megatron_utils/hf_to_megatron/qwen3_5.py": "7a808f639f473874dd40cfe3283642fb159d7076d2a1a492c16e540cdc882a9a",
    "slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py": "c1a786d5d54c7f27bd84394a6ebfea755bc2694b2aabd7549633a93939b6a704",
    "slime/utils/types.py": "0e8aca6bc49dcba30975a5a325e210cd450fbb25b3c0c79c107a27986692816b",
    "slime/rollout/data_source.py": "446595578aff88fac1637b8956a245853db1167e7964896bbab2fff61b1db616",
    "slime/rollout/sglang_rollout.py": "30bef16ad975d46ed288f7c026111833159730c5f58b627b75fc0ee99ad20c27",
    "slime/ray/rollout.py": "acec11f7c95010f8cb03e1a394d5cc936385701770695e81eafdf49b4b725519",
    "slime/ray/actor_group.py": "0f8a76316f4c1f2618f23dafd506e41ff7511e1b8d1cd3dc76d3ee24e27e2170",
    "train.py": "8b1e231caaa59ac5970fdbcd1a1cd5d10bae6a293f27072781ce800df2baa0b0",
    "tests/test_qwen3.5_0.8B_gsm8k_short.py": "77527d49770611dee469bbbf39c7643b3348abbf6231328a5fa6eee3f89751e6",
    "tests/test_full_disk_weight_update.py": "473023cbad89a34befec71448c37b30776553898b208cc9978841b92953a0bcb",
    "tests/utils/test_loss_mask_type_qwen35.py": "1e7d48b35d6c6f45ceb161cd47041bd74c0553532d85528bf1e7bfd3e82a9295",
}

REQUIRED_TEXT = {
    "docker/version.txt": [DOCKER_TAG],
    "docker/Dockerfile": [
        "v0.5.15.post1-cu129",
        "MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087",
        "flash-linear-attention==0.4.2",
        "QwenLM/FlashQLA.git",
    ],
    "scripts/models/qwen3.5-4B.sh": [
        '"slime_plugins.models.qwen3_5" "get_qwen3_5_spec"',
        "--num-layers 32",
        "--hidden-size 2560",
        "--attention-output-gate",
    ],
    "slime_plugins/models/qwen3_5.py": [
        "class Qwen3_5GatedDeltaNet",
        "def get_qwen3_5_spec",
        'self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")',
    ],
    "slime/backends/megatron_utils/hf_to_megatron/qwen3_5.py": [
        "def qwen3_5_hf_tensor",
        'rest.startswith("self_attention.linear_attn.")',
    ],
    "slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py": [
        "def convert_qwen3_5_to_hf",
        '"linear_attn.in_proj_z.weight"',
    ],
    "slime/rollout/sglang_rollout.py": [
        "async def generate_rollout_async",
        "rewards = await batched_async_rm",
    ],
    "slime/ray/rollout.py": [
        "def _convert_samples_to_train_data",
        "def _save_debug_rollout_data",
        "load_debug_rollout_data",
    ],
    "slime/ray/actor_group.py": [
        "def async_train",
        "def update_weights",
        "def _reload_rollout_weights_from_disk",
    ],
    "train.py": [
        "rollout_manager.generate.remote",
        "actor_model.async_train",
        "actor_model.update_weights",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slime-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.slime_repo.resolve()

    checks: list[dict[str, str]] = []

    def check(name: str, ok: bool, evidence: str) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "evidence": evidence})

    head = git(repo, "rev-parse", "HEAD")
    tag = git(repo, "describe", "--tags", "--exact-match")
    check("release_sha", head == SLIME_SHA, head)
    check("release_tag", tag == SLIME_TAG, tag)

    file_hashes = {}
    for relative, expected in EXPECTED_FILES.items():
        file_path = repo / relative
        actual = sha256(file_path) if file_path.is_file() else "missing"
        file_hashes[relative] = actual
        check(f"sha256:{relative}", actual == expected, actual)

    for relative, needles in REQUIRED_TEXT.items():
        text = (repo / relative).read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in text]
        check(f"symbols:{relative}", not missing, "all required symbols present" if not missing else repr(missing))

    arguments_text = (repo / "slime/utils/arguments.py").read_text(encoding="utf-8")
    check(
        "upstream_lora_training_not_documented",
        "--lora-rank" not in arguments_text and "--tuner-type" not in arguments_text,
        "no PEFT/LoRA learner arguments in slime v0.3.1; use merged S1 plus full-weight path",
    )

    failed = [item["name"] for item in checks if item["status"] != "pass"]
    result = {
        "schema_name": "day26.slime_static_audit",
        "schema_version": 1,
        "audit_date": "2026-08-17",
        "status": "pass" if not failed else "fail",
        "release": {"repository": "https://github.com/THUDM/slime", "tag": SLIME_TAG, "sha": SLIME_SHA},
        "container": {
            "image": f"slimerl/slime:{DOCKER_TAG}",
            "digest": DOCKER_DIGEST,
            "platform": "linux/amd64",
            "runtime_verification": "pending_gpu_container",
        },
        "support_boundary": {
            "qwen35_4b_model_config": "static_pass",
            "gdn_model_spec": "static_pass",
            "hf_megatron_conversion": "static_pass",
            "sample_rollout_reward_train_full_sync": "static_pass",
            "lora_learner_and_adapter_sync": "unsupported_or_undocumented_upstream",
            "selected_day26_path": "Day 21 merged S1 -> full Megatron training -> full weight sync",
        },
        "claim_boundary": "Static source and release audit only; S0 container import and S1-S5 require the rented GPU runtime.",
        "failed_checks": failed,
        "checks": checks,
        "file_sha256": file_hashes,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
