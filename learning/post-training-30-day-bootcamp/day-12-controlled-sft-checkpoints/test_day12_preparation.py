from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "prepare_day12_experiment.py"
SPEC = importlib.util.spec_from_file_location("prepare_day12_experiment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
day12 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(day12)

MIX_C_MODULE_PATH = HERE / "prepare_day12_mix_c.py"
MIX_C_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_mix_c", MIX_C_MODULE_PATH
)
assert MIX_C_SPEC is not None and MIX_C_SPEC.loader is not None
mix_c = importlib.util.module_from_spec(MIX_C_SPEC)
MIX_C_SPEC.loader.exec_module(mix_c)

RUN_D_MODULE_PATH = HERE / "prepare_day12_run_d.py"
RUN_D_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_d", RUN_D_MODULE_PATH
)
assert RUN_D_SPEC is not None and RUN_D_SPEC.loader is not None
run_d = importlib.util.module_from_spec(RUN_D_SPEC)
RUN_D_SPEC.loader.exec_module(run_d)

RUN_E_MODULE_PATH = HERE / "prepare_day12_run_e.py"
RUN_E_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_e", RUN_E_MODULE_PATH
)
assert RUN_E_SPEC is not None and RUN_E_SPEC.loader is not None
run_e = importlib.util.module_from_spec(RUN_E_SPEC)
RUN_E_SPEC.loader.exec_module(run_e)

RUN_G_MODULE_PATH = HERE / "prepare_day12_run_g.py"
RUN_G_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_g", RUN_G_MODULE_PATH
)
assert RUN_G_SPEC is not None and RUN_G_SPEC.loader is not None
run_g = importlib.util.module_from_spec(RUN_G_SPEC)
RUN_G_SPEC.loader.exec_module(run_g)

RUN_I_MODULE_PATH = HERE / "prepare_day12_run_i.py"
RUN_I_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_i", RUN_I_MODULE_PATH
)
assert RUN_I_SPEC is not None and RUN_I_SPEC.loader is not None
run_i = importlib.util.module_from_spec(RUN_I_SPEC)
RUN_I_SPEC.loader.exec_module(run_i)

RUN_K_MODULE_PATH = HERE / "prepare_day12_run_k.py"
RUN_K_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_k", RUN_K_MODULE_PATH
)
assert RUN_K_SPEC is not None and RUN_K_SPEC.loader is not None
run_k = importlib.util.module_from_spec(RUN_K_SPEC)
RUN_K_SPEC.loader.exec_module(run_k)

RUN_L_MODULE_PATH = HERE / "prepare_day12_run_l.py"
RUN_L_SPEC = importlib.util.spec_from_file_location(
    "prepare_day12_run_l", RUN_L_MODULE_PATH
)
assert RUN_L_SPEC is not None and RUN_L_SPEC.loader is not None
run_l = importlib.util.module_from_spec(RUN_L_SPEC)
RUN_L_SPEC.loader.exec_module(run_l)


def test_exact_subset_indices_is_exact_and_deterministic() -> None:
    weights = [3, 5, 7, 11, 13]
    first = day12.exact_subset_indices(weights, 18)
    second = day12.exact_subset_indices(weights, 18)
    assert first == second
    assert sum(weights[index] for index in first) == 18


def test_exact_subset_indices_rejects_impossible_target() -> None:
    with pytest.raises(day12.Day12PreparationError, match="cannot construct"):
        day12.exact_subset_indices([4, 8, 12], 5)


def test_real_ab_contract_and_schedules(tmp_path: Path) -> None:
    schedule_a = tmp_path / "schedule-a.json"
    schedule_b = tmp_path / "schedule-b.json"
    report = tmp_path / "diff.md"
    result = day12.prepare(
        day12.DEFAULT_A_CONFIG,
        day12.DEFAULT_B_CONFIG,
        schedule_a,
        schedule_b,
        report,
    )
    assert result["status"] == "day12_preparation_pass"
    assert set(result["authorized_differences"]) == day12.AUTHORIZED_DIFFS

    for path in (schedule_a, schedule_b):
        schedule = day12.load_json(path)
        day12.verify_schedule(schedule)
        segments = schedule["segments"]
        assert segments["25_percent"]["cumulative_supervised_tokens"] == 61734
        assert segments["60_percent"]["cumulative_supervised_tokens"] == 148162
        assert segments["100_percent"]["cumulative_supervised_tokens"] == 246936
        occurrence_ids = [
            occurrence_id
            for name in schedule["header"]["checkpoint_order"]
            for occurrence_id in segments[name]["occurrence_ids"]
        ]
        assert len(occurrence_ids) == len(set(occurrence_ids))
    assert "Status: `pass`" in report.read_text(encoding="utf-8")


def test_unauthorized_diff_is_rejected() -> None:
    config_a = day12.load_resolved_config(day12.DEFAULT_A_CONFIG)
    config_b = day12.load_resolved_config(day12.DEFAULT_B_CONFIG)
    config_b["optimizer"]["learning_rate"] = 9e-5
    with pytest.raises(day12.Day12PreparationError, match="unauthorized"):
        day12.verify_ab_diff(config_a, config_b)


def test_resolve_project_path_supports_stripped_remote_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo_root = tmp_path / "autodl-tmp"
    remote_bootcamp = (
        remote_repo_root / "day11-ready" / "post-training-30-day-bootcamp"
    )
    expected = remote_bootcamp / "artifacts" / "data" / "mixture.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(day12, "REPO_ROOT", remote_repo_root)
    monkeypatch.setattr(day12, "BOOTCAMP_ROOT", remote_bootcamp)

    configured = "learning/post-training-30-day-bootcamp/artifacts/data/mixture.json"
    assert day12.resolve_project_path(configured) == expected.resolve()
    assert day12.canonical_project_path(expected) == configured


def test_real_mix_c_contract_and_schedule() -> None:
    summary = mix_c.prepare()
    assert summary["status"] == "day12_mix_c_preparation_pass"
    assert summary["total_supervised_tokens"] == 246936
    assert summary["target_ratios"] == {
        "general": "7/24",
        "math": "7/24",
        "code": "7/24",
        "finance": "1/8",
    }
    distribution = summary["distribution_by_skill"]
    assert distribution["general"]["supervised_tokens"] == 72023
    assert distribution["math"]["supervised_tokens"] == 72023
    assert distribution["code"]["supervised_tokens"] == 72023
    assert distribution["finance"]["supervised_tokens"] == 30867
    assert all(summary["validation_checks"].values())

    schedule = mix_c.DAY12.load_json(mix_c.DEFAULT_SCHEDULE)
    mix_c.DAY12.verify_schedule(schedule)
    assert schedule["segments"]["25_percent"]["cumulative_supervised_tokens"] == 61734
    assert schedule["segments"]["60_percent"]["cumulative_supervised_tokens"] == 148162
    assert schedule["segments"]["100_percent"]["cumulative_supervised_tokens"] == 246936


def test_mix_c_paths_support_stripped_remote_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote_repo_root = tmp_path / "autodl-tmp"
    remote_bootcamp = (
        remote_repo_root / "day11-ready" / "post-training-30-day-bootcamp"
    )
    expected = remote_bootcamp / "artifacts" / "data" / "mixture.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(mix_c.DAY12, "REPO_ROOT", remote_repo_root)
    monkeypatch.setattr(mix_c.DAY12, "BOOTCAMP_ROOT", remote_bootcamp)

    configured = "learning/post-training-30-day-bootcamp/artifacts/data/mixture.json"
    assert mix_c.resolve_repo_path(configured) == expected.resolve()
    assert mix_c.configured_path(expected) == configured
    mix_c.configure_day09_canonical_paths()
    assert mix_c.DAY09_MIX.relative_path(expected) == configured


def test_run_d_rewrites_finance_program_as_natural_steps() -> None:
    response = run_d.rewrite_finance_response(
        "Calculation: subtract(152, 102), divide(#0, 102)\nFinal answer: 49%"
    )
    assert response == (
        "Reasoning:\n"
        "1. Subtract 102 from 152.\n"
        "2. Divide the result from step 1 by 102.\n"
        "Final answer: 49%"
    )
    assert "Calculation:" not in response
    assert "divide(" not in response


def test_real_run_d_format_contract_and_schedule() -> None:
    summary = run_d.prepare()
    assert summary["status"] == "day12_run_d_preparation_pass"
    assert summary["total_supervised_tokens"] == 246936
    assert summary["target_ratios"] == {
        "general": "7/24",
        "math": "7/24",
        "code": "7/24",
        "finance": "1/8",
    }
    assert summary["repaired_finance_occurrences"] < summary[
        "parent_finance_occurrences"
    ]
    assert summary["maximum_input_tokens"] <= 2048
    assert all(summary["validation_checks"].values())

    schedule = run_d.DAY12.load_json(run_d.DEFAULT_SCHEDULE)
    run_d.DAY12.verify_schedule(schedule)
    assert schedule["segments"]["25_percent"]["cumulative_supervised_tokens"] == 61734
    assert schedule["segments"]["60_percent"]["cumulative_supervised_tokens"] == 148162
    assert schedule["segments"]["100_percent"]["cumulative_supervised_tokens"] == 246936


def test_real_run_e_changes_only_learning_rate_and_identity() -> None:
    summary = run_e.prepare()
    assert summary["status"] == "day12_run_e_preparation_pass"
    assert summary["authorized_differences"] == {
        "optimizer.learning_rate": {"parent": 2e-5, "rollout": 1e-5},
        "run.id": {"parent": "D", "rollout": "E"},
    }
    assert all(summary["validation_checks"].values())
    schedule = run_e.DAY12.load_json(run_e.DEFAULT_SCHEDULE)
    run_e.DAY12.verify_schedule(schedule)
    assert schedule["segments"]["25_percent"]["cumulative_supervised_tokens"] == 61734


def test_run_f_continues_the_single_variable_lr_curve() -> None:
    root = HERE.parent
    summary = run_e.prepare(
        spec_path=root / "artifacts/configs/day12-rollout-F-spec.json",
        parent_config_path=root / "artifacts/configs/day12-controlled-sft-E.yaml",
        config_path=root / "artifacts/configs/day12-controlled-sft-F.yaml",
        schedule_path=root / "artifacts/data/day12-training-schedule-F.json",
        summary_path=root / "artifacts/reports/day12-rollout-F-preparation.json",
    )
    assert summary["status"] == "day12_run_f_preparation_pass"
    assert summary["authorized_differences"] == {
        "optimizer.learning_rate": {"parent": 1e-5, "rollout": 5e-6},
        "run.id": {"parent": "E", "rollout": "F"},
    }
    assert all(summary["validation_checks"].values())


def test_run_g_changes_only_parameterization_and_identity() -> None:
    summary = run_g.prepare()
    assert summary["status"] == "day12_run_g_preparation_pass"
    assert set(summary["authorized_differences"]) == run_g.AUTHORIZED_DIFFERENCES
    assert summary["authorized_differences"]["run.id"] == {
        "parent": "E",
        "rollout": "G",
    }
    assert summary["primary_change"]["parent_value"] == "full_parameters"
    assert summary["primary_change"]["rollout_value"] == "lora"
    assert all(summary["validation_checks"].values())
    schedule = run_g.DAY12.load_json(run_g.DEFAULT_SCHEDULE)
    run_g.DAY12.verify_schedule(schedule)
    assert schedule["segments"]["25_percent"]["cumulative_supervised_tokens"] == 61734


def test_run_h_changes_only_lora_learning_rate_and_identity() -> None:
    root = HERE.parent
    summary = run_e.prepare(
        spec_path=root / "artifacts/configs/day12-rollout-H-spec.json",
        parent_config_path=root / "artifacts/configs/day12-controlled-sft-G.yaml",
        config_path=root / "artifacts/configs/day12-controlled-sft-H.yaml",
        schedule_path=root / "artifacts/data/day12-training-schedule-H.json",
        summary_path=root / "artifacts/reports/day12-rollout-H-preparation.json",
    )
    assert summary["status"] == "day12_run_h_preparation_pass"
    assert summary["authorized_differences"] == {
        "optimizer.learning_rate": {"parent": 1e-5, "rollout": 5e-5},
        "run.id": {"parent": "G", "rollout": "H"},
    }
    assert all(summary["validation_checks"].values())


def test_run_i_rebalances_only_data_toward_math_and_code() -> None:
    summary = run_i.prepare()
    assert summary["status"] == "day12_run_i_preparation_pass"
    assert summary["target_ratios"] == {
        "general": "1/8",
        "math": "5/12",
        "code": "5/12",
        "finance": "1/24",
    }
    assert summary["target_supervised_tokens"] == {
        "general": 30867,
        "math": 102890,
        "code": 102890,
        "finance": 10289,
    }
    assert summary["total_supervised_tokens"] == 246936
    assert summary["maximum_input_tokens"] <= 2048
    assert all(summary["validation_checks"].values())


def test_run_j_changes_only_run_i_lora_learning_rate_and_identity() -> None:
    root = HERE.parent
    summary = run_e.prepare(
        spec_path=root / "artifacts/configs/day12-rollout-J-spec.json",
        parent_config_path=root / "artifacts/configs/day12-controlled-sft-I.yaml",
        config_path=root / "artifacts/configs/day12-controlled-sft-J.yaml",
        schedule_path=root / "artifacts/data/day12-training-schedule-J.json",
        summary_path=root / "artifacts/reports/day12-rollout-J-preparation.json",
    )
    assert summary["status"] == "day12_run_j_preparation_pass"
    assert summary["authorized_differences"] == {
        "optimizer.learning_rate": {"parent": 5e-5, "rollout": 3e-5},
        "run.id": {"parent": "I", "rollout": "J"},
    }
    assert all(summary["validation_checks"].values())


def test_run_k_adds_only_output_projection_lora_coverage() -> None:
    summary = run_k.prepare()
    assert summary["status"] == "day12_run_k_preparation_pass"
    assert set(summary["authorized_differences"]) == {
        "run.id",
        "adapter.target_modules",
    }
    assert summary["authorized_differences"]["adapter.target_modules"] == {
        "parent": "all-linear",
        "rollout": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ],
    }
    assert all(summary["validation_checks"].values())


def test_run_l_changes_only_rank_from_best_code_lora_parent() -> None:
    summary = run_l.prepare()
    assert summary["status"] == "day12_run_l_preparation_pass"
    assert summary["authorized_differences"] == {
        "adapter.rank": {"parent": 16, "rollout": 32},
        "run.id": {"parent": "I", "rollout": "L"},
    }
    assert summary["derived_scaling_change"] == {
        "parent_alpha_over_rank": 2.0,
        "rollout_alpha_over_rank": 1.0,
    }
    assert all(summary["validation_checks"].values())
