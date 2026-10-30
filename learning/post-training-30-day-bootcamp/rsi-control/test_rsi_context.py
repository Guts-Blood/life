#!/usr/bin/env python3

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rsi_context


class RSIContextGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = rsi_context.build_graph()
        cls.nodes = {node["id"]: node for node in cls.graph["nodes"]}

    def test_graph_is_deterministic_and_self_identifying(self) -> None:
        rebuilt = rsi_context.build_graph()
        self.assertEqual(self.graph, rebuilt)
        unsigned = copy.deepcopy(self.graph)
        graph_sha256 = unsigned.pop("graph_sha256")
        self.assertEqual(graph_sha256, rsi_context.object_sha256(unsigned))

    def test_cli_graph_builder_brackets_compilation_with_validation(self) -> None:
        validated = rsi_context.build_validated_graph()
        self.assertEqual(validated["graph_sha256"], self.graph["graph_sha256"])

    def test_graph_has_no_dangling_edges(self) -> None:
        self.assertGreater(len(self.nodes), 600)
        source_paths = {source["path"] for source in self.graph["source_files"]}
        for node in self.graph["nodes"]:
            self.assertIn(node["source"]["path"], source_paths)
        for edge in self.graph["edges"]:
            self.assertIn(edge["source"], self.nodes)
            self.assertIn(edge["target"], self.nodes)
            self.assertIn(edge["relation"], rsi_context.RELATION_COSTS)

    def test_nodes_retain_precise_source_locators(self) -> None:
        failure = self.nodes["failure.data.supervision.template_rendering"]
        self.assertEqual(failure["source"]["path"], "failure-taxonomy.json")
        self.assertRegex(failure["source"]["pointer"], r"^/failure_modes/\d+$")

        evidence_id = (
            "evidence:versions/rsi-v0001/runs/run-001-probe-primary/"
            "attempts/attempt-001/evidence/stage-a-root-cause-diagnosis.json"
        )
        evidence = self.nodes[evidence_id]
        self.assertRegex(evidence["attributes"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(evidence["attributes"]["bytes"], 0)

    def test_intervention_topic_may_also_be_a_canonical_leaf(self) -> None:
        node = self.nodes["model.trainable_scope.lora_rank"]
        self.assertEqual(node["kind"], "intervention_leaf")
        self.assertTrue(node["attributes"]["also_topic"])

    def test_future_diagnosis_evidence_attaches_to_atomic_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rsi-control"
            shutil.copytree(rsi_context.ROOT, root)
            version_id = "rsi-v0003"
            version_dir = root / "versions" / version_id
            evidence_dir = version_dir / "evidence"
            evidence_dir.mkdir(parents=True)
            shared_ref = f"versions/{version_id}/evidence/shared.json"
            (root / shared_ref).write_text('{"result":"verified"}\n', encoding="utf-8")
            lever = "model.target.representation.boundary_tokens"
            dependent = "model.target.masking.token_eligibility"
            version = {
                "version_id": version_id,
                "kind": "causal_intervention",
                "status": "planned",
                "intervention": {
                    "primary_lever": lever,
                    "changed_levers": [lever, dependent],
                    "dependent_lever_reason": "the retained token must enter loss",
                },
                "diagnosis": {
                    "diagnosis_id": "diagnosis-0003",
                    "observed_symptoms": [
                        {
                            "id": "symptom.output.boundary_mismatch",
                            "scope": "probe/code",
                            "evidence_refs": [shared_ref],
                        }
                    ],
                    "primary_failure": {
                        "id": "failure.data.supervision.template_rendering",
                        "first_broken_invariant": "rendering must preserve bytes",
                        "pipeline_boundary": "canonical target -> rendered label",
                        "causal_status": "supported",
                        "evidence_refs": [shared_ref],
                    },
                    "contributing_failures": [],
                    "excluded_alternatives": [
                        {
                            "id": "failure.eval.scorer.metric_semantics",
                            "reason": "frozen scorer replay matched",
                            "evidence_refs": [shared_ref],
                        }
                    ],
                    "intervention": {
                        "primary_lever": lever,
                        "dependent_lever": dependent,
                        "dependent_lever_reason": "the retained token must enter loss",
                        "expected_mechanism": "preserve and supervise the boundary",
                    },
                    "prediction": "eligibility recovers",
                    "falsifier": "boundary remains absent",
                    "frozen_invariants": ["source rows"],
                    "guardrails": ["frozen eval"],
                },
            }
            (version_dir / "version.json").write_text(
                json.dumps(version), encoding="utf-8"
            )
            state_path = root / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["current_version"] = version_id
            state["versions"].append(
                {
                    "version_id": version_id,
                    "status": "planned",
                    "confirmed_qualified": False,
                }
            )
            second_version_id = "rsi-v0004"
            second_version_dir = root / "versions" / second_version_id
            second_version_dir.mkdir()
            second_version = copy.deepcopy(version)
            second_version["version_id"] = second_version_id
            (second_version_dir / "version.json").write_text(
                json.dumps(second_version), encoding="utf-8"
            )
            state["versions"].append(
                {
                    "version_id": second_version_id,
                    "status": "planned",
                    "confirmed_qualified": False,
                }
            )
            state_path.write_text(json.dumps(state), encoding="utf-8")

            graph = rsi_context.build_graph(root)
            nodes = {node["id"] for node in graph["nodes"]}
            diagnosis_id = "diagnosis:rsi-v0003:diagnosis-0003"
            self.assertIn("diagnosis:rsi-v0004:diagnosis-0003", nodes)
            self.assertIn(f"{diagnosis_id}:observation:00", nodes)
            self.assertIn(f"{diagnosis_id}:primary-failure", nodes)
            self.assertIn(f"{diagnosis_id}:excluded:00", nodes)
            evidence_id = f"evidence:{shared_ref}"
            supporters = {
                edge["source"]
                for edge in graph["edges"]
                if edge["relation"] == "supported_by"
                and edge["target"] == evidence_id
            }
            self.assertTrue(
                {
                    f"{diagnosis_id}:observation:00",
                    f"{diagnosis_id}:primary-failure",
                    f"{diagnosis_id}:excluded:00",
                }.issubset(supporters)
            )
            self.assertNotIn(diagnosis_id, supporters)
            self.assertNotIn("diagnosis:rsi-v0004:diagnosis-0003", supporters)
            relations = {
                (edge["source"], edge["relation"], edge["target"])
                for edge in graph["edges"]
            }
            self.assertIn(
                (diagnosis_id, "primary_intervention", lever), relations
            )
            self.assertIn(
                (diagnosis_id, "dependent_intervention", dependent),
                relations,
            )
            self.assertIn((lever, "requires", dependent), relations)

    def test_search_localizes_a_specific_failure(self) -> None:
        matches = rsi_context.search_graph(self.graph, "catastrophic forgetting")
        self.assertEqual(
            matches[0]["id"],
            "failure.training.transfer.catastrophic_forgetting",
        )
        self.assertEqual(
            rsi_context.search_graph(self.graph, "灾难性遗忘")[0]["id"],
            "failure.training.transfer.catastrophic_forgetting",
        )
        self.assertEqual(
            rsi_context.search_graph(self.graph, "loss function")[0]["id"],
            "failure.model.objective.loss_formulation",
        )

    def test_case_preserves_primary_and_dependent_intervention_roles(self) -> None:
        case_id = "case-rsi-v0001-v0002-leading-four-spaces"
        relations = {
            (edge["relation"], edge["target"])
            for edge in self.graph["edges"]
            if edge["source"] == case_id
        }
        self.assertIn(
            (
                "prospective_primary_intervention",
                "model.target.representation.boundary_tokens",
            ),
            relations,
        )
        self.assertIn(
            (
                "prospective_dependent_intervention",
                "model.target.masking.token_eligibility",
            ),
            relations,
        )


class RSIContextPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = rsi_context.build_graph()

    def test_case_packet_contains_causal_chain_and_evidence(self) -> None:
        seed = "case-rsi-v0001-v0002-leading-four-spaces"
        packet = rsi_context.build_packet(
            self.graph,
            [seed],
            max_nodes=18,
            max_bytes=14_000,
        )
        node_ids = {node["id"] for node in packet["nodes"]}
        self.assertIn(seed, node_ids)
        self.assertIn("failure.data.supervision.template_rendering", node_ids)
        self.assertIn("failure.data.supervision.mask_materialization", node_ids)
        self.assertIn("model.target.representation.boundary_tokens", node_ids)
        self.assertIn("model.target.masking.token_eligibility", node_ids)
        self.assertTrue(any(node_id.startswith("evidence:") for node_id in node_ids))
        self.assertNotIn("failure.model.objective.loss_formulation", node_ids)
        case = next(node for node in packet["nodes"] if node["id"] == seed)
        self.assertEqual(
            case["attributes"]["observed"][
                "code_source_targets_start_exact_four_spaces"
            ],
            105,
        )
        lever = next(
            node
            for node in packet["nodes"]
            if node["id"] == "model.target.representation.boundary_tokens"
        )
        self.assertEqual(lever["current_goal_authorization"], "allowed")
        self.assertLessEqual(rsi_context.emitted_json_bytes(packet), 14_000)

        unsigned = copy.deepcopy(packet)
        packet_sha256 = unsigned.pop("packet_sha256")
        self.assertEqual(packet_sha256, rsi_context.object_sha256(unsigned))

    def test_packet_budget_is_deterministic_and_never_drops_seed(self) -> None:
        seed = "failure.training.transfer.catastrophic_forgetting"
        left = rsi_context.build_packet(
            self.graph, [seed], max_nodes=10, max_bytes=8_000
        )
        right = rsi_context.build_packet(
            self.graph, [seed], max_nodes=10, max_bytes=8_000
        )
        self.assertEqual(left, right)
        self.assertIn(seed, {node["id"] for node in left["nodes"]})
        self.assertNotIn(
            "failure.model.objective.loss_formulation",
            {node["id"] for node in left["nodes"]},
        )
        with self.assertRaisesRegex(rsi_context.RSIContextError, "exceed max_bytes"):
            rsi_context.build_packet(
                self.graph, [seed], max_nodes=2, max_bytes=100
            )

    def test_unknown_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(rsi_context.RSIContextError, "unknown context"):
            rsi_context.build_packet(self.graph, ["failure.not.real"])

    def test_tampered_graph_is_rejected_before_packet_build(self) -> None:
        graph = copy.deepcopy(self.graph)
        graph["nodes"][0]["summary"] = "TAMPERED"
        with self.assertRaisesRegex(rsi_context.RSIContextError, "graph hash"):
            rsi_context.build_packet(graph, [graph["nodes"][0]["id"]])

    def test_packet_hashes_edge_locator_sources(self) -> None:
        packet = rsi_context.build_packet(
            self.graph,
            ["model.target.representation.boundary_tokens"],
            max_cost=1,
            max_nodes=8,
            max_bytes=8_000,
        )
        source_paths = {source["path"] for source in packet["source_files"]}
        self.assertIn("failure-taxonomy.json", source_paths)
        self.assertTrue(
            any(edge["relation"] == "requires" for edge in packet["edges"])
        )

    def test_packet_marks_levers_outside_the_current_goal(self) -> None:
        seed = "model.architecture.capacity.parameter_scale"
        packet = rsi_context.build_packet(
            self.graph, [seed], max_nodes=8, max_bytes=8_000
        )
        lever = next(node for node in packet["nodes"] if node["id"] == seed)
        self.assertEqual(
            lever["current_goal_authorization"], "outside_current_goal"
        )


class RSIContextPathTests(unittest.TestCase):
    def test_relative_paths_have_one_canonical_identity(self) -> None:
        self.assertEqual(rsi_context._canonical_relative("./evidence/x.json"), "evidence/x.json")
        self.assertEqual(rsi_context._canonical_relative("evidence//x.json"), "evidence/x.json")
        with self.assertRaisesRegex(rsi_context.RSIContextError, "unsafe"):
            rsi_context._canonical_relative("generated/evidence.json")
        with self.assertRaisesRegex(rsi_context.RSIContextError, "unsafe"):
            rsi_context._canonical_relative("Generated/evidence.json")

    def test_absolute_parent_and_symlink_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            outside = root.parent / f"{root.name}-outside.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                with self.assertRaisesRegex(rsi_context.RSIContextError, "unsafe"):
                    rsi_context._safe_file(root, str(source.resolve()))
                with self.assertRaisesRegex(rsi_context.RSIContextError, "unsafe"):
                    rsi_context._safe_file(root, "../outside.json")
                os.symlink(source, root / "source-link.json")
                with self.assertRaisesRegex(rsi_context.RSIContextError, "symlink"):
                    rsi_context._safe_file(root, "source-link.json")
            finally:
                outside.unlink(missing_ok=True)

    def test_source_swap_to_external_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            source = root / "source.json"
            source.write_text('{"inside":true}', encoding="utf-8")
            outside = base / "outside.json"
            outside.write_text('{"outside":true}', encoding="utf-8")
            original_safe_file = rsi_context._safe_file

            def swap_after_check(root_path: Path, relative: str) -> Path:
                checked = original_safe_file(root_path, relative)
                source.unlink()
                os.symlink(outside, source)
                return checked

            with mock.patch.object(
                rsi_context, "_safe_file", side_effect=swap_after_check
            ):
                with self.assertRaisesRegex(
                    rsi_context.RSIContextError, "securely open"
                ):
                    rsi_context._read_source(root, "source.json", {})

    def test_physical_file_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            alias = root / "SOURCE.JSON"
            if not alias.exists() or not os.path.samefile(source, alias):
                self.skipTest("filesystem is case-sensitive")
            sources: dict[str, dict] = {}
            rsi_context._read_source(root, "source.json", sources)
            with self.assertRaisesRegex(rsi_context.RSIContextError, "aliases"):
                rsi_context._read_source(root, "SOURCE.JSON", sources)

    def test_output_cannot_overwrite_canonical_or_history_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rsi-control"
            root.mkdir()
            with self.assertRaisesRegex(rsi_context.RSIContextError, "generated"):
                rsi_context._validated_output_path(
                    root / "versions/rsi-v0001/version.json", root
                )
            self.assertEqual(
                rsi_context._validated_output_path(root / "generated/graph.json", root),
                (root / "generated/graph.json").resolve(),
            )
            self.assertEqual(
                rsi_context._validated_output_path(root.parent / "report.json", root),
                (root.parent / "report.json").resolve(),
            )
            with self.assertRaisesRegex(rsi_context.RSIContextError, "unsafe"):
                rsi_context._validate_output_source_disjoint(
                    {"source_files": [{"path": "generated/evidence.json"}]},
                    (root / "generated/evidence.json").resolve(),
                    root,
                )

    def test_output_source_disjoint_uses_physical_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}", encoding="utf-8")
            alias = root / "SOURCE.JSON"
            if not alias.exists() or not os.path.samefile(source, alias):
                self.skipTest("filesystem is case-sensitive")
            with self.assertRaisesRegex(rsi_context.RSIContextError, "overlaps"):
                rsi_context._validate_output_source_disjoint(
                    {"source_files": [{"path": "source.json"}]},
                    alias.resolve(),
                    root,
                )


if __name__ == "__main__":
    unittest.main()
