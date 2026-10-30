import copy
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_day09_mixtures.py")
SPEC = importlib.util.spec_from_file_location("build_day09_mixtures", MODULE_PATH)
assert SPEC and SPEC.loader
MIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIX)


def parent_row(skill: str, suffix: str, supervised: int = 1) -> dict:
    sample_id = f"{skill}:{suffix}"
    return {
        "sample_id": sample_id,
        "skill": skill,
        "raw_token_count": 2,
        "input_token_count": 3,
        "supervised_token_count": supervised,
    }


def proof_for_rows(rows: list[dict]) -> dict:
    slices = {}
    for skill in MIX.SKILLS:
        row = next(item for item in rows if item["skill"] == skill)
        selection = [
            {
                "sample_id": row["sample_id"],
                "supervised_token_count": row["supervised_token_count"],
                "occurrence_count": 1,
            }
        ]
        slices[skill] = {
            "selection": selection,
            "selection_sample_ids_sha256": MIX.object_sha256([row["sample_id"]]),
            "actual_supervised_tokens": row["supervised_token_count"],
            "total_occurrences": 1,
        }
    return {"slices": slices}


def valid_manifest(mix_name: str, counts: dict[str, int], ratios: dict[str, str]) -> dict:
    records = []
    for skill in MIX.SKILLS:
        for index in range(counts[skill]):
            sample_id = f"{skill}:{mix_name}:{index}"
            records.append(
                {
                    "sample_id": sample_id,
                    "canonical_sample_id": sample_id,
                    "occurrence_id": f"{mix_name}|{sample_id}|occurrence=1",
                    "occurrence_index": 1,
                    "sampling_count": 1,
                    "supervised_tokens_per_occurrence": 1,
                    "skill": skill,
                }
            )
    total = len(records)
    config_hashes = {
        "preprocessing_config_sha256": "pre",
        "quality_filter_config_sha256": "filter",
        "decontamination_config_sha256": "decontamination",
    }
    header = {
        "common_invariants": {
            "parent_pool_hash": "pool",
            "config_hashes": config_hashes,
        },
        "target_ratios": ratios,
        "totals": {"supervised_tokens": total},
        "distribution_by_skill": {
            skill: {"supervised_tokens": counts[skill]} for skill in MIX.SKILLS
        },
        "occurrence_ids_sha256": MIX.object_sha256(
            [row["occurrence_id"] for row in records]
        ),
    }
    manifest = {"header": header, "records": records}
    header["manifest_hash"] = MIX.object_sha256(manifest)
    return manifest


class Day09MixtureTest(unittest.TestCase):
    def test_occurrences_preserve_canonical_identity_and_are_deterministic(self):
        rows = [parent_row(skill, "one") for skill in MIX.SKILLS]
        proof = proof_for_rows(rows)
        parent_by_id = {row["sample_id"]: row for row in rows}
        first = MIX.build_occurrences("mix_A_balanced", proof, parent_by_id, 7)
        second = MIX.build_occurrences("mix_A_balanced", proof, parent_by_id, 7)
        self.assertEqual(first, second)
        self.assertTrue(
            all(row["sample_id"] == row["canonical_sample_id"] for row in first)
        )
        self.assertTrue(all("|occurrence=1" in row["occurrence_id"] for row in first))

    def test_occurrence_build_rejects_token_count_tampering(self):
        rows = [parent_row(skill, "one") for skill in MIX.SKILLS]
        proof = proof_for_rows(rows)
        proof["slices"]["general"]["selection"][0]["supervised_token_count"] = 2
        with self.assertRaises(MIX.MixtureBuildError):
            MIX.build_occurrences(
                "mix_A_balanced",
                proof,
                {row["sample_id"]: row for row in rows},
                7,
            )

    def test_pair_validation_accepts_only_the_intended_ratio_difference(self):
        mix_a = valid_manifest(
            "mix_A_balanced",
            {skill: 2 for skill in MIX.SKILLS},
            {skill: "1/4" for skill in MIX.SKILLS},
        )
        mix_b = valid_manifest(
            "mix_B_targeted",
            {"general": 1, "math": 1, "code": 4, "finance": 2},
            {"general": "1/8", "math": "1/8", "code": "1/2", "finance": "1/4"},
        )
        checks = MIX.validate_pair(mix_a, mix_b)
        self.assertTrue(all(checks.values()))

    def test_pair_validation_rejects_parent_pool_mismatch(self):
        mix_a = valid_manifest(
            "mix_A_balanced",
            {skill: 2 for skill in MIX.SKILLS},
            {skill: "1/4" for skill in MIX.SKILLS},
        )
        mix_b = valid_manifest(
            "mix_B_targeted",
            {"general": 1, "math": 1, "code": 4, "finance": 2},
            {"general": "1/8", "math": "1/8", "code": "1/2", "finance": "1/4"},
        )
        mix_b = copy.deepcopy(mix_b)
        mix_b["header"]["common_invariants"]["parent_pool_hash"] = "other"
        mix_b["header"]["manifest_hash"] = MIX.object_sha256(
            {
                "header": {
                    key: value
                    for key, value in mix_b["header"].items()
                    if key != "manifest_hash"
                },
                "records": mix_b["records"],
            }
        )
        with self.assertRaises(MIX.MixtureBuildError):
            MIX.validate_pair(mix_a, mix_b)


if __name__ == "__main__":
    unittest.main()
