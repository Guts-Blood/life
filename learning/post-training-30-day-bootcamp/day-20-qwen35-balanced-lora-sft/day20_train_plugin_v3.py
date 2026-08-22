#!/usr/bin/env python3
"""Day 20 v3 external plugin and target-binding evidence callback.

When loaded by ms-swift, registration is deliberately ordered:

1. register the training-only target-template alias;
2. import/reuse the frozen v2 evidence callback;
3. register the v3 target-binding callback.

Normal CLI import remains offline and does not import ms-swift.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from day20_train_runtime_v3 import (
    Day20TrainingRuntimeV3Error,
    build_attestation,
    build_binding_config,
    build_checkpoint_envelope,
    file_sha256,
    load_json,
    verify_attestation,
    verify_binding_config,
    verify_checkpoint_envelope,
    write_json_new,
)


TARGET_BINDING_CALLBACK_NAME = "day20_v3_target_binding"


class Day20TrainPluginV3Error(ValueError):
    """The v3 external plugin or target-binding evidence drifted."""


def _live_template_alias(template: Any) -> str:
    meta = getattr(template, "template_meta", None)
    candidates = (
        getattr(meta, "template_type", None),
        getattr(template, "template_type", None),
        getattr(meta, "name", None),
    )
    aliases = [value for value in candidates if isinstance(value, str) and value]
    if not aliases:
        raise Day20TrainPluginV3Error("cannot determine live ms-swift template alias")
    if len(set(aliases)) != 1:
        raise Day20TrainPluginV3Error(f"live template aliases disagree: {aliases}")
    return aliases[0]


def _json_argument(value: Any) -> Any:
    raw = getattr(value, "value", value)
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw
    return str(raw)


def _trainer_arguments(args: Any) -> dict[str, Any]:
    names = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "learning_rate",
        "num_train_epochs",
        "seed",
        "data_seed",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "save_only_model",
        "save_total_limit",
        "save_strategy",
        "train_dataloader_shuffle",
    )
    missing = [name for name in names if not hasattr(args, name)]
    if missing:
        raise Day20TrainPluginV3Error(
            "live TrainingArguments fields are missing: " + ", ".join(missing)
        )
    return {name: _json_argument(getattr(args, name)) for name in names}


def _expected_trainer_arguments(binding: Mapping[str, Any]) -> dict[str, Any]:
    try:
        config = load_json(Path(binding["training_config"]["path"]))
        training = config["training"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise Day20TrainPluginV3Error(
            "cannot load bound TrainingArguments contract"
        ) from error
    if not isinstance(training, Mapping):
        raise Day20TrainPluginV3Error("bound training config is not an object")
    required = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "num_train_epochs",
        "seed",
        "data_seed",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "save_only_model",
        "save_total_limit",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise Day20TrainPluginV3Error(
            "bound training config fields are missing: " + ", ".join(missing)
        )
    return {
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "bf16": True,
        "learning_rate": float(training["learning_rate"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "seed": training["seed"],
        "data_seed": training["data_seed"],
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "adam_beta1": float(training["adam_beta1"]),
        "adam_beta2": float(training["adam_beta2"]),
        "adam_epsilon": float(training["adam_epsilon"]),
        "save_only_model": training["save_only_model"],
        "save_total_limit": training["save_total_limit"],
        "save_strategy": "no",
        "train_dataloader_shuffle": False,
    }


def _validate_live_trainer_arguments(
    values: Mapping[str, Any], *, expected: Mapping[str, Any]
) -> None:
    drift = {
        key: {"actual": values.get(key), "expected": value}
        for key, value in expected.items()
        if values.get(key) != value
    }
    if drift:
        raise Day20TrainPluginV3Error(
            "live target trainer arguments drifted: "
            + json.dumps(drift, ensure_ascii=False, sort_keys=True)
        )


def _validate_live_target_template(
    template: Any, *, binding: Mapping[str, Any]
) -> None:
    import day20_target_encoding_v3 as target_encoding

    try:
        target_identity = binding["target_module"]
        target_path = Path(target_identity["path"]).resolve()
        config = load_json(Path(binding["training_config"]["path"]))
        bound_template = config["template"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise Day20TrainPluginV3Error(
            "cannot load bound target-template contract"
        ) from error
    if (
        Path(target_encoding.__file__).resolve() != target_path
        or target_encoding.TARGET_TEMPLATE_ALIAS
        != target_identity.get("template_alias")
        or target_encoding.TARGET_ENCODING_CONTRACT_VERSION
        != target_identity.get("contract_version")
        or target_encoding.TARGET_ENCODING_CONTRACT_SHA256
        != target_identity.get("contract_sha256")
    ):
        raise Day20TrainPluginV3Error("loaded target-template module identity drifted")
    expected = {
        "model_type": "qwen3_5",
        "template": target_encoding.TARGET_TEMPLATE_ALIAS,
        "enable_thinking": False,
        "add_non_thinking_prefix": True,
        "loss_scale": "default+ignore_empty_think",
        "max_length": 2304,
        # ms-swift maps the SFT CLI value `delete` to the live Template value `raise`.
        "truncation_strategy": "raise",
        "packing": False,
        "padding_free": False,
    }
    if not isinstance(bound_template, Mapping) or dict(bound_template) != expected:
        raise Day20TrainPluginV3Error("bound target-template config drifted")
    actual = {
        "model_type": getattr(getattr(template, "model_info", None), "model_type", None),
        "template": _live_template_alias(template),
        "enable_thinking": getattr(template, "enable_thinking", None),
        "add_non_thinking_prefix": getattr(
            template, "add_non_thinking_prefix", None
        ),
        "loss_scale": getattr(template, "_loss_scale", None),
        "max_length": getattr(template, "max_length", None),
        "truncation_strategy": getattr(template, "truncation_strategy", None),
        "packing": getattr(template, "packing", None),
        "padding_free": getattr(template, "padding_free", None),
    }
    if actual != expected:
        raise Day20TrainPluginV3Error(
            "live target template drifted: "
            + json.dumps(
                {"actual": actual, "expected": expected},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if (
        getattr(
            type(template), "day20_target_encoding_contract_sha256", None
        )
        != target_encoding.TARGET_ENCODING_CONTRACT_SHA256
    ):
        raise Day20TrainPluginV3Error("live target-template class contract drifted")
    try:
        target_encoding._assert_template_contract(template)
    except Exception as error:
        raise Day20TrainPluginV3Error(
            f"live target-template token contract failed: {type(error).__name__}: {error}"
        ) from error


def _register_target_binding_callback() -> None:
    from swift.callbacks import TrainerCallback, callbacks_map

    class Day20V3TargetBindingCallback(TrainerCallback):
        def __init__(self, args: Any, trainer: Any) -> None:
            super().__init__(args, trainer)
            config_path = os.environ.get("DAY20_V3_BINDING_CONFIG")
            if not config_path:
                raise RuntimeError("DAY20_V3_BINDING_CONFIG is required")
            self.binding_path = Path(config_path).resolve()
            self.binding = verify_binding_config(self.binding_path)
            self.evidence_dir = Path(self.binding["evidence_dir"])
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            self.attestation_path = (
                self.evidence_dir / "target-binding-attestation.json"
            )

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if verify_binding_config(self.binding_path) != self.binding:
                raise RuntimeError("v3 binding config changed before training")
            template = self.trainer.template
            alias = _live_template_alias(template)
            _validate_live_target_template(template, binding=self.binding)
            trainer_arguments = _trainer_arguments(args)
            _validate_live_trainer_arguments(
                trainer_arguments,
                expected=_expected_trainer_arguments(self.binding),
            )
            attestation = build_attestation(
                self.binding_path,
                actual_template_alias=alias,
                template_class=(
                    f"{type(self.trainer.template).__module__}."
                    f"{type(self.trainer.template).__qualname__}"
                ),
                trainer_arguments=trainer_arguments,
            )
            if self.attestation_path.exists():
                existing = verify_attestation(
                    self.attestation_path, binding_config_path=self.binding_path
                )
                if existing != attestation:
                    raise RuntimeError("v3 target attestation changed across resume")
            else:
                write_json_new(self.attestation_path, attestation)
            return control

        def on_train_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            verify_attestation(
                self.attestation_path, binding_config_path=self.binding_path
            )
            return control

    callbacks_map[TARGET_BINDING_CALLBACK_NAME] = Day20V3TargetBindingCallback


def _register_external_plugin() -> None:
    # Do not move either import above this call: the alias must exist before
    # ms-swift resolves CLI template arguments or initializes callbacks.
    from day20_target_encoding_v3 import register_target_template_v3

    register_target_template_v3()

    if os.environ.get("DAY20_V2_REGISTER_SWIFT_CALLBACK") == "1":
        # Importing the frozen v2 plugin under this environment flag registers
        # day20_v2_evidence.  V3 adds evidence; it does not replace v2 evidence.
        import day20_train_plugin_v2  # noqa: F401

        _register_target_binding_callback()


def seal_summary(
    *,
    summary_path: Path,
    run_root: Path,
    binding_config_path: Path,
    attestation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from day20_train_plugin_v2 import verify_training_summary

    summary = verify_training_summary(summary_path.resolve(), run_root=run_root.resolve())
    output = output_dir.resolve()
    if output == run_root.resolve() or run_root.resolve() not in output.parents:
        raise Day20TrainPluginV3Error("checkpoint envelope directory is outside run_root")
    if any(
        part.startswith("checkpoint-") and part.removeprefix("checkpoint-").isdigit()
        for part in output.parts
    ):
        raise Day20TrainPluginV3Error("checkpoint envelopes must be outside checkpoints")
    envelopes: dict[str, Any] = {}
    for label, candidate in summary["candidate_ids"].items():
        destination = output / f"{candidate}.target-envelope-v3.json"
        expected = build_checkpoint_envelope(
            checkpoint=Path(summary["checkpoints"][label]),
            run_root=run_root,
            training_summary_path=summary_path,
            binding_config_path=binding_config_path,
            attestation_path=attestation_path,
            candidate=candidate,
        )
        if destination.exists():
            actual = verify_checkpoint_envelope(destination, run_root=run_root)
            if actual != expected:
                raise Day20TrainPluginV3Error(
                    f"existing checkpoint envelope drifted: {destination}"
                )
        else:
            write_json_new(destination, expected)
        envelopes[label] = {
            "candidate": candidate,
            "path": str(destination),
            "file_sha256": file_sha256(destination),
            "content_sha256": expected["envelope_sha256"],
        }
    return {"status": "pass", "envelopes": envelopes}


def verify_sealed_summary(
    *,
    summary_path: Path,
    run_root: Path,
    binding_config_path: Path,
    attestation_path: Path,
    envelope_dir: Path,
) -> dict[str, Any]:
    """Verify that every v2 candidate has exactly one valid external envelope."""
    from day20_train_plugin_v2 import verify_training_summary

    root = run_root.resolve()
    summary = verify_training_summary(summary_path.resolve(), run_root=root)
    verify_binding_config(binding_config_path.resolve())
    verify_attestation(
        attestation_path.resolve(), binding_config_path=binding_config_path.resolve()
    )
    directory = envelope_dir.resolve()
    inside_checkpoint = any(
        part.startswith("checkpoint-") and part.removeprefix("checkpoint-").isdigit()
        for part in directory.parts
    )
    if directory == root or root not in directory.parents or inside_checkpoint:
        raise Day20TrainPluginV3Error("checkpoint envelope directory is unsafe")
    if not directory.is_dir() or directory.is_symlink():
        raise Day20TrainPluginV3Error("checkpoint envelope directory is missing")
    expected = set(summary["candidate_ids"].values())
    paths = sorted(directory.glob("*.target-envelope-v3.json"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise Day20TrainPluginV3Error("checkpoint envelope must be a regular file")
    actual: dict[str, dict[str, Any]] = {}
    for path in paths:
        envelope = verify_checkpoint_envelope(path, run_root=root)
        candidate = envelope.get("candidate")
        if not isinstance(candidate, str) or candidate in actual:
            raise Day20TrainPluginV3Error("checkpoint envelope candidate is invalid")
        if path.name != f"{candidate}.target-envelope-v3.json":
            raise Day20TrainPluginV3Error("checkpoint envelope filename drifted")
        actual[candidate] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "content_sha256": envelope["envelope_sha256"],
        }
    if set(actual) != expected:
        raise Day20TrainPluginV3Error(
            "checkpoint envelope cohort is incomplete or contains extras"
        )
    return {"status": "pass", "candidates": actual}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-binding-config")
    build.add_argument("--outer-manifest", required=True, type=Path)
    build.add_argument("--core-manifest", required=True, type=Path)
    build.add_argument("--target-audit", required=True, type=Path)
    build.add_argument("--target-module", required=True, type=Path)
    build.add_argument("--dataset", required=True, type=Path)
    build.add_argument("--training-config", required=True, type=Path)
    build.add_argument("--v2-callback-config", required=True, type=Path)
    build.add_argument("--evidence-dir", required=True, type=Path)
    build.add_argument("--run-kind", required=True, choices=("probe", "main"))
    build.add_argument("--seed", required=True, type=int)
    build.add_argument("--learning-rate", required=True)
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify-binding-config")
    verify.add_argument("--binding-config", required=True, type=Path)
    attest = commands.add_parser("verify-attestation")
    attest.add_argument("--attestation", required=True, type=Path)
    attest.add_argument("--binding-config", required=True, type=Path)
    seal = commands.add_parser("seal-summary")
    seal.add_argument("--summary", required=True, type=Path)
    seal.add_argument("--run-root", required=True, type=Path)
    seal.add_argument("--binding-config", required=True, type=Path)
    seal.add_argument("--attestation", required=True, type=Path)
    seal.add_argument("--output-dir", required=True, type=Path)
    verify_sealed = commands.add_parser("verify-sealed-summary")
    verify_sealed.add_argument("--summary", required=True, type=Path)
    verify_sealed.add_argument("--run-root", required=True, type=Path)
    verify_sealed.add_argument("--binding-config", required=True, type=Path)
    verify_sealed.add_argument("--attestation", required=True, type=Path)
    verify_sealed.add_argument("--envelope-dir", required=True, type=Path)
    envelope = commands.add_parser("verify-envelope")
    envelope.add_argument("--envelope", required=True, type=Path)
    envelope.add_argument("--run-root", required=True, type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "build-binding-config":
            result: Any = build_binding_config(
                outer_manifest_path=args.outer_manifest,
                core_manifest_path=args.core_manifest,
                audit_path=args.target_audit,
                target_module_path=args.target_module,
                dataset_path=args.dataset,
                training_config_path=args.training_config,
                v2_callback_config_path=args.v2_callback_config,
                evidence_dir=args.evidence_dir,
                run_kind=args.run_kind,
                seed=args.seed,
                learning_rate=args.learning_rate,
            )
            write_json_new(args.output.resolve(), result)
        elif args.command == "verify-binding-config":
            result = verify_binding_config(args.binding_config)
        elif args.command == "verify-attestation":
            result = verify_attestation(
                args.attestation, binding_config_path=args.binding_config
            )
        elif args.command == "seal-summary":
            result = seal_summary(
                summary_path=args.summary,
                run_root=args.run_root,
                binding_config_path=args.binding_config,
                attestation_path=args.attestation,
                output_dir=args.output_dir,
            )
        elif args.command == "verify-sealed-summary":
            result = verify_sealed_summary(
                summary_path=args.summary,
                run_root=args.run_root,
                binding_config_path=args.binding_config,
                attestation_path=args.attestation,
                envelope_dir=args.envelope_dir,
            )
        else:
            result = verify_checkpoint_envelope(args.envelope, run_root=args.run_root)
    except (Day20TrainingRuntimeV3Error, Day20TrainPluginV3Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if os.environ.get("DAY20_V3_REGISTER_SWIFT_PLUGIN") == "1":
    _register_external_plugin()


if __name__ == "__main__":
    main()
