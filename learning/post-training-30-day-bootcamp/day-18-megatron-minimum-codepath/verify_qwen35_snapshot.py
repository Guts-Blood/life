#!/usr/bin/env python3
"""Fail-closed integrity check for the pinned Qwen3.5-4B-Base snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


MODEL_ID = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
EXPECTED_TOTAL_SIZE = 9_319_737_856
EXPECTED_SPECIAL_TOKEN_IDS = {
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
EXPECTED_MTP_TENSORS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}
EXPECTED_REPOSITORY_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "merges.txt",
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
}
ALLOWED_AUXILIARY_FILES = {"snapshot-manifest.json", "MANIFEST.sha256"}
EXPECTED_CRITICAL_FILES = {
    "model.safetensors-00001-of-00002.safetensors": {
        "size": 5_329_398_712,
        "sha256": "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712",
    },
    "model.safetensors-00002-of-00002.safetensors": {
        "size": 3_990_429_344,
        "sha256": "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef",
    },
    "tokenizer.json": {
        "size": 12_807_196,
        "sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    },
}
EXPECTED_FILE_HASHES = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    "README.md": "76ead24ea3de5a52610c755ffa0b6e7cb51e3c413f124a3776bc6f16b694474e",
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "model.safetensors-00001-of-00002.safetensors": "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712",
    "model.safetensors-00002-of-00002.safetensors": "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef",
    "model.safetensors.index.json": "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "video_preprocessor_config.json": "d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def top_level_files(model_dir: Path) -> Iterable[Path]:
    return sorted(path for path in model_dir.iterdir() if path.is_file())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify(model_dir: Path, hash_all: bool) -> Dict[str, Any]:
    require(model_dir.is_dir(), f"missing model directory: {model_dir}")
    require(model_dir.name == REVISION, f"directory is not revision-scoped to {REVISION}")

    partials = sorted(
        str(path.relative_to(model_dir))
        for path in model_dir.rglob("*")
        if path.is_file() and path.name.endswith(".incomplete")
    )
    # huggingface_hub's FileLock leaves released zero-byte .lock sentinels in
    # local-dir metadata. They are not payload and are excluded from the tar.
    require(not partials, f"snapshot still has partial files: {partials[:5]}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    preprocessor_path = model_dir / "preprocessor_config.json"
    for path in (config_path, index_path, preprocessor_path):
        require(path.is_file(), f"missing required file: {path.name}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    require(
        "Qwen3_5ForConditionalGeneration" in architectures,
        f"unexpected architectures: {architectures}",
    )
    require(config.get("model_type") == "qwen3_5", f"unexpected model_type: {config.get('model_type')}")
    require(isinstance(config.get("vision_config"), dict), "full VLM snapshot must contain vision_config")
    require(
        config.get("text_config", {}).get("mtp_num_hidden_layers") == 1,
        f"unexpected text_config.mtp_num_hidden_layers: {config.get('text_config', {}).get('mtp_num_hidden_layers')}",
    )
    for key, expected in EXPECTED_SPECIAL_TOKEN_IDS.items():
        require(config.get(key) == expected, f"unexpected {key}: {config.get(key)}")

    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    processor_class = preprocessor.get("processor_class")
    require(processor_class == "Qwen3VLProcessor", f"unexpected processor_class: {processor_class}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    targets = sorted(set(weight_map.values()))
    require(len(weight_map) == 738, f"expected 738 tensor mappings, got {len(weight_map)}")
    mtp_tensors = {name for name in weight_map if name.startswith("mtp.")}
    require(mtp_tensors == EXPECTED_MTP_TENSORS, f"unexpected MTP tensor set: {sorted(mtp_tensors)}")
    require(
        targets == [
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        ],
        f"unexpected weight shard targets: {targets}",
    )
    require(
        index.get("metadata", {}).get("total_size") == EXPECTED_TOTAL_SIZE,
        f"unexpected total_size: {index.get('metadata', {}).get('total_size')}",
    )
    for target in targets:
        require((model_dir / target).is_file(), f"index target is missing: {target}")

    present_top_level = {path.name for path in top_level_files(model_dir)}
    missing_repository_files = sorted(EXPECTED_REPOSITORY_FILES - present_top_level)
    require(not missing_repository_files, f"snapshot is missing repository files: {missing_repository_files}")
    unexpected_files = sorted(present_top_level - EXPECTED_REPOSITORY_FILES - ALLOWED_AUXILIARY_FILES)
    require(not unexpected_files, f"snapshot contains unexpected top-level files: {unexpected_files}")

    files: Dict[str, Dict[str, Any]] = {}
    for path in top_level_files(model_dir):
        stat = path.stat()
        entry: Dict[str, Any] = {"size": stat.st_size}
        expected = EXPECTED_CRITICAL_FILES.get(path.name)
        expected_sha256 = EXPECTED_FILE_HASHES.get(path.name)
        if expected is not None:
            require(stat.st_size == expected["size"], f"size mismatch for {path.name}: {stat.st_size}")
        if hash_all or expected_sha256 is not None:
            entry["sha256"] = sha256_file(path)
        if expected_sha256 is not None:
            require(entry["sha256"] == expected_sha256, f"SHA-256 mismatch for {path.name}")
        files[path.name] = entry

    return {
        "schema_version": 1,
        "status": "downloaded_verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/{MODEL_ID}/commit/{REVISION}",
        "local_path": str(model_dir.resolve()),
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "processor_class": processor_class,
        "special_token_ids": EXPECTED_SPECIAL_TOKEN_IDS,
        "mtp_num_hidden_layers": 1,
        "mtp_tensor_count": len(mtp_tensors),
        "tensor_count": len(weight_map),
        "weight_total_size": EXPECTED_TOTAL_SIZE,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-all", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.model_dir, hash_all=args.hash_all)
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        print(f"snapshot verification failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
