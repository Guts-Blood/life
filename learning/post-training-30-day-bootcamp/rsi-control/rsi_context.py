#!/usr/bin/env python3
"""Build a deterministic RSI knowledge graph and compact context packets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import rsi_control


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 4 * 1024 * 1024

SEARCH_ALIASES = {
    "loss function": "loss formulation",
    "loss functions": "loss formulation",
    "灾难性遗忘": "catastrophic forgetting",
    "损失函数": "loss formulation",
    "数据处理": "data supervision",
    "模板渲染": "template rendering",
    "模型架构": "model architecture",
    "训练方案": "training schedule",
}

RELATION_COSTS: dict[str, tuple[int, int]] = {
    # Forward and reverse traversal costs.  Taxonomy ancestors are cheap;
    # expanding from a broad parent into every child is deliberately expensive.
    "parent": (1, 3),
    "decomposes_into": (1, 3),
    "maps_to": (1, 1),
    "observed_as": (1, 1),
    "primary_failure": (1, 1),
    "contributing_failure": (1, 1),
    "excluded_failure": (2, 2),
    # These relations are intentionally directional.  A failure may expand to
    # candidate levers, but a shared lever must not pull unrelated failures,
    # versions, or cases back into the same packet.
    "candidate_intervention": (1, 4),
    "historical_primary_intervention": (2, 4),
    "historical_dependent_intervention": (2, 4),
    "prospective_primary_intervention": (1, 4),
    "prospective_dependent_intervention": (1, 4),
    "primary_intervention": (1, 4),
    "dependent_intervention": (1, 4),
    "requires": (1, 4),
    "changed_lever": (1, 4),
    "diagnosed_in": (2, 2),
    "remediated_in": (2, 2),
    "supported_by": (1, 1),
    "has_diagnosis": (1, 1),
    "current_goal": (1, 1),
    "current_version": (1, 1),
    "allows_prefix": (2, 2),
    "prioritizes": (3, 3),
}


class RSIContextError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def emitted_json_bytes(value: Any) -> int:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return len(payload.encode("utf-8"))


def _canonical_relative(relative: str) -> str:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RSIContextError(f"unsafe relative path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RSIContextError(f"unsafe relative path: {relative!r}")
    canonical = path.as_posix()
    if canonical in {"", "."} or path.parts[0].casefold() == "generated":
        raise RSIContextError(f"unsafe relative path: {relative!r}")
    return canonical


def _safe_file(root: Path, relative: str) -> Path:
    relative = _canonical_relative(relative)
    path = Path(relative)
    root_resolved = root.resolve()
    candidate = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise RSIContextError(f"symlink is not a trusted graph source: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RSIContextError(f"missing graph source: {relative}") from error
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise RSIContextError(f"graph source escapes its root: {relative}") from error
    generated_root = (root_resolved / "generated").resolve()
    try:
        resolved.relative_to(generated_root)
    except ValueError:
        pass
    else:
        raise RSIContextError(f"generated output cannot be a graph source: {relative}")
    if not resolved.is_file():
        raise RSIContextError(f"graph source is not a regular file: {relative}")
    if resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise RSIContextError(f"graph source exceeds {MAX_SOURCE_BYTES} bytes: {relative}")
    return resolved


def _read_source(root: Path, relative: str, sources: dict[str, dict[str, Any]]) -> bytes:
    relative = _canonical_relative(relative)
    path = _safe_file(root, relative)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RSIContextError("secure graph reads require O_NOFOLLOW support")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RSIContextError(f"cannot securely open graph source: {relative}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RSIContextError(f"graph source is not a regular file: {relative}")
        try:
            opened_path = path.resolve(strict=True)
            opened_path.relative_to(root.resolve())
            current_stat = os.stat(path, follow_symlinks=False)
        except (OSError, ValueError) as error:
            raise RSIContextError(
                f"graph source escaped or changed during secure open: {relative}"
            ) from error
        if (current_stat.st_dev, current_stat.st_ino) != (
            file_stat.st_dev,
            file_stat.st_ino,
        ):
            raise RSIContextError(
                f"graph source changed during secure open: {relative}"
            )
        if file_stat.st_size > MAX_SOURCE_BYTES:
            raise RSIContextError(
                f"graph source exceeds {MAX_SOURCE_BYTES} bytes: {relative}"
            )
        chunks: list[bytes] = []
        remaining = MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_SOURCE_BYTES:
            raise RSIContextError(
                f"graph source exceeds {MAX_SOURCE_BYTES} bytes: {relative}"
            )
    finally:
        os.close(descriptor)
    physical_identity = [file_stat.st_dev, file_stat.st_ino]
    for known_path, known_source in sources.items():
        if (
            known_path != relative
            and known_source.get("_physical_identity") == physical_identity
        ):
            raise RSIContextError(
                f"graph source aliases one physical file: {known_path} / {relative}"
            )
    metadata = {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "_physical_identity": physical_identity,
    }
    if relative in sources and sources[relative] != metadata:
        raise RSIContextError(f"graph source changed during build: {relative}")
    sources[relative] = metadata
    return payload


def _read_json(root: Path, relative: str, sources: dict[str, dict[str, Any]]) -> Any:
    payload = _read_source(root, relative, sources)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RSIContextError(f"invalid JSON graph source: {relative}") from error


def _public_source(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": metadata["path"],
        "sha256": metadata["sha256"],
        "bytes": metadata["bytes"],
    }


def _pointer(section: str, index: int | None = None) -> str:
    return f"/{section}" if index is None else f"/{section}/{index}"


def build_graph(root: Path = ROOT) -> dict[str, Any]:
    """Compile canonical RSI files into a normalized, read-only graph."""

    root = root.resolve()
    sources: dict[str, dict[str, Any]] = {}
    _read_source(root, "rsi_context.py", sources)
    failure_taxonomy = _read_json(root, "failure-taxonomy.json", sources)
    intervention_taxonomy = _read_json(root, "taxonomy.json", sources)
    goal = _read_json(root, "goal.json", sources)
    state = _read_json(root, "state.json", sources)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(
        node_id: str,
        kind: str,
        summary: str,
        source_path: str,
        pointer: str | None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if node_id in nodes:
            raise RSIContextError(f"duplicate graph node: {node_id}")
        node: dict[str, Any] = {
            "id": node_id,
            "kind": kind,
            "summary": summary,
            "source": {"path": source_path, "pointer": pointer},
        }
        if attributes:
            node["attributes"] = dict(attributes)
        nodes[node_id] = node

    def add_edge(source: str, relation: str, target: str, locator: str) -> None:
        if relation not in RELATION_COSTS:
            raise RSIContextError(f"unknown graph relation: {relation}")
        key = (source, relation, target)
        if key in edges:
            if locator not in edges[key]["locators"]:
                edges[key]["locators"].append(locator)
                edges[key]["locators"].sort()
            return
        edges[key] = {
            "source": source,
            "relation": relation,
            "target": target,
            "locators": [locator],
        }

    failure_sections = (
        ("failure_families", "failure_family"),
        ("failure_topics", "failure_topic"),
        ("failure_modes", "failure_mode"),
        ("symptoms", "symptom"),
    )
    for section, kind in failure_sections:
        for index, record in enumerate(failure_taxonomy[section]):
            node_id = record["id"]
            pointer = _pointer(section, index)
            add_node(
                node_id,
                kind,
                record["definition"],
                "failure-taxonomy.json",
                pointer,
            )
            if "parent" in record:
                add_edge(node_id, "parent", record["parent"], f"failure-taxonomy.json#{pointer}/parent")

    intervention_groups: set[str] = set()
    for index, record in enumerate(intervention_taxonomy["levers"]):
        family = record["id"]
        pointer = _pointer("levers", index)
        add_node(
            family,
            "intervention_family",
            f"Intervention family {family}.",
            "taxonomy.json",
            pointer,
            {"execution_only": bool(record.get("execution_only", False))},
        )
        intervention_groups.add(family)
        for child_index, child in enumerate(record["children"]):
            child_id = f"{family}.{child}"
            add_node(
                child_id,
                "intervention_topic",
                f"Intervention topic {child_id}.",
                "taxonomy.json",
                f"{pointer}/children/{child_index}",
            )
            add_edge(
                child_id,
                "parent",
                family,
                f"taxonomy.json#{pointer}/children/{child_index}",
            )
            intervention_groups.add(child_id)

    for index, record in enumerate(intervention_taxonomy["detailed_levers"]):
        node_id = record["id"]
        pointer = _pointer("detailed_levers", index)
        if node_id in nodes:
            if nodes[node_id]["kind"] != "intervention_topic":
                raise RSIContextError(f"intervention leaf collides with a node: {node_id}")
            nodes[node_id].update(
                {
                    "kind": "intervention_leaf",
                    "summary": record["description"],
                    "source": {"path": "taxonomy.json", "pointer": pointer},
                    "attributes": {"family": record["family"], "also_topic": True},
                }
            )
            continue
        add_node(
            node_id,
            "intervention_leaf",
            record["description"],
            "taxonomy.json",
            pointer,
            {"family": record["family"]},
        )
        parents = [group for group in intervention_groups if node_id.startswith(group + ".")]
        if not parents:
            raise RSIContextError(f"intervention leaf has no graph parent: {node_id}")
        parent = max(parents, key=len)
        add_edge(node_id, "parent", parent, f"taxonomy.json#{pointer}/family")

    for index, alias in enumerate(failure_taxonomy["legacy_aliases"]):
        node_id = f"alias:{alias['alias_type']}:{alias['alias']}"
        pointer = _pointer("legacy_aliases", index)
        add_node(
            node_id,
            "legacy_alias",
            alias.get("note", f"Legacy alias {alias['alias']}."),
            "failure-taxonomy.json",
            pointer,
            {"alias": alias["alias"], "alias_type": alias["alias_type"]},
        )
        add_edge(node_id, "maps_to", alias["maps_to"], f"failure-taxonomy.json#{pointer}/maps_to")

    for index, example in enumerate(
        failure_taxonomy["intervention_crosswalk_examples"]
    ):
        pointer = _pointer("intervention_crosswalk_examples", index)
        for lever_index, lever in enumerate(example["candidate_levers"]):
            add_edge(
                example["failure_mode"],
                "candidate_intervention",
                lever,
                f"failure-taxonomy.json#{pointer}/candidate_levers/{lever_index}",
            )

    def add_evidence(owner: str, refs: list[str], locator_prefix: str) -> None:
        for index, raw_relative in enumerate(refs):
            relative = _canonical_relative(raw_relative)
            if relative not in sources:
                _read_source(root, relative, sources)
            evidence_id = f"evidence:{relative}"
            if evidence_id not in nodes:
                source = sources[relative]
                add_node(
                    evidence_id,
                    "evidence",
                    f"Immutable evidence locator for {relative}.",
                    relative,
                    None,
                    {"sha256": source["sha256"], "bytes": source["bytes"]},
                )
            add_edge(owner, "supported_by", evidence_id, f"{locator_prefix}/{index}")

    for index, case in enumerate(failure_taxonomy["case_examples"]):
        case_id = case["id"]
        pointer = _pointer("case_examples", index)
        add_node(
            case_id,
            "case",
            case["title"],
            "failure-taxonomy.json",
            pointer,
            {
                "confidence": case["confidence"],
                "last_good_stage": case["last_good_stage"],
                "first_bad_stage": case["first_bad_stage"],
                "broken_invariant": case["broken_invariant"],
                "observed": case["observed"],
                "historical_intervention": case["historical_intervention"],
                "prospective_canonical_equivalent": case[
                    "prospective_canonical_equivalent"
                ],
            },
        )
        for symptom_index, symptom in enumerate(case["symptom_ids"]):
            add_edge(case_id, "observed_as", symptom, f"failure-taxonomy.json#{pointer}/symptom_ids/{symptom_index}")
        add_edge(case_id, "primary_failure", case["primary_failure_mode"], f"failure-taxonomy.json#{pointer}/primary_failure_mode")
        for failure_index, failure in enumerate(case["contributing_failure_modes"]):
            add_edge(case_id, "contributing_failure", failure, f"failure-taxonomy.json#{pointer}/contributing_failure_modes/{failure_index}")
        for failure_index, failure in enumerate(case["excluded_failure_modes"]):
            add_edge(case_id, "excluded_failure", failure, f"failure-taxonomy.json#{pointer}/excluded_failure_modes/{failure_index}")
        for field, relation_prefix in (
            ("historical_intervention", "historical"),
            ("prospective_canonical_equivalent", "prospective"),
        ):
            case_intervention = case[field]
            primary = case_intervention["primary_lever"]
            dependent = case_intervention.get("dependent_lever")
            add_edge(
                case_id,
                f"{relation_prefix}_primary_intervention",
                primary,
                f"failure-taxonomy.json#{pointer}/{field}/primary_lever",
            )
            if dependent is not None:
                add_edge(
                    case_id,
                    f"{relation_prefix}_dependent_intervention",
                    dependent,
                    f"failure-taxonomy.json#{pointer}/{field}/dependent_lever",
                )
                add_edge(
                    primary,
                    "requires",
                    dependent,
                    f"failure-taxonomy.json#{pointer}/{field}/dependent_lever",
                )
        add_edge(case_id, "diagnosed_in", case["diagnosed_in_version"], f"failure-taxonomy.json#{pointer}/diagnosed_in_version")
        add_edge(case_id, "remediated_in", case["remediated_in_version"], f"failure-taxonomy.json#{pointer}/remediated_in_version")
        add_evidence(
            case_id,
            case["evidence_refs"],
            f"failure-taxonomy.json#{pointer}/evidence_refs",
        )

    for version_entry in state["versions"]:
        version_id = version_entry["version_id"]
        relative = f"versions/{version_id}/version.json"
        version = _read_json(root, relative, sources)
        add_node(
            version_id,
            "version",
            f"{version['kind']} version {version_id} ({version['status']}).",
            relative,
            "",
            {"kind": version["kind"], "status": version["status"]},
        )
        intervention = version["intervention"]
        primary = intervention.get("primary_lever")
        if primary is not None:
            add_edge(version_id, "primary_intervention", primary, f"{relative}#/intervention/primary_lever")
        for lever_index, lever in enumerate(intervention["changed_levers"]):
            add_edge(version_id, "changed_lever", lever, f"{relative}#/intervention/changed_levers/{lever_index}")

        diagnosis = version.get("diagnosis")
        if diagnosis:
            raw_diagnosis_id = diagnosis["diagnosis_id"]
            diagnosis_id = f"diagnosis:{version_id}:{raw_diagnosis_id}"
            add_node(
                diagnosis_id,
                "diagnosis",
                f"Structured diagnosis for {version_id}.",
                relative,
                "/diagnosis",
                {
                    "diagnosis_id": raw_diagnosis_id,
                    "expected_mechanism": diagnosis["intervention"][
                        "expected_mechanism"
                    ],
                    "dependent_lever_reason": diagnosis["intervention"].get(
                        "dependent_lever_reason"
                    ),
                },
            )
            add_edge(version_id, "has_diagnosis", diagnosis_id, f"{relative}#/diagnosis")
            for symptom_index, symptom in enumerate(diagnosis["observed_symptoms"]):
                claim_id = f"{diagnosis_id}:observation:{symptom_index:02d}"
                pointer = f"/diagnosis/observed_symptoms/{symptom_index}"
                add_node(
                    claim_id,
                    "observation",
                    f"{symptom['scope']}: {symptom['id']}",
                    relative,
                    pointer,
                    {"scope": symptom["scope"]},
                )
                add_edge(
                    diagnosis_id,
                    "decomposes_into",
                    claim_id,
                    f"{relative}#{pointer}",
                )
                add_edge(
                    claim_id,
                    "observed_as",
                    symptom["id"],
                    f"{relative}#{pointer}/id",
                )
                add_evidence(
                    claim_id,
                    symptom["evidence_refs"],
                    f"{relative}#{pointer}/evidence_refs",
                )
            primary_failure = diagnosis["primary_failure"]
            primary_claim_id = f"{diagnosis_id}:primary-failure"
            add_node(
                primary_claim_id,
                "failure_hypothesis",
                primary_failure["first_broken_invariant"],
                relative,
                "/diagnosis/primary_failure",
                {
                    "pipeline_boundary": primary_failure["pipeline_boundary"],
                    "causal_status": primary_failure["causal_status"],
                },
            )
            add_edge(
                diagnosis_id,
                "decomposes_into",
                primary_claim_id,
                f"{relative}#/diagnosis/primary_failure",
            )
            add_edge(
                primary_claim_id,
                "primary_failure",
                primary_failure["id"],
                f"{relative}#/diagnosis/primary_failure/id",
            )
            add_evidence(
                primary_claim_id,
                primary_failure["evidence_refs"],
                f"{relative}#/diagnosis/primary_failure/evidence_refs",
            )
            for failure_index, failure in enumerate(diagnosis["contributing_failures"]):
                add_edge(diagnosis_id, "contributing_failure", failure, f"{relative}#/diagnosis/contributing_failures/{failure_index}")
            for failure_index, failure in enumerate(diagnosis["excluded_alternatives"]):
                claim_id = f"{diagnosis_id}:excluded:{failure_index:02d}"
                pointer = f"/diagnosis/excluded_alternatives/{failure_index}"
                add_node(
                    claim_id,
                    "excluded_hypothesis",
                    failure["reason"],
                    relative,
                    pointer,
                )
                add_edge(
                    diagnosis_id,
                    "decomposes_into",
                    claim_id,
                    f"{relative}#{pointer}",
                )
                add_edge(
                    claim_id,
                    "excluded_failure",
                    failure["id"],
                    f"{relative}#{pointer}/id",
                )
                add_evidence(
                    claim_id,
                    failure["evidence_refs"],
                    f"{relative}#{pointer}/evidence_refs",
                )

            for kind, field in (("prediction", "prediction"), ("falsifier", "falsifier")):
                claim_id = f"{diagnosis_id}:{kind}"
                add_node(
                    claim_id,
                    kind,
                    diagnosis[field],
                    relative,
                    f"/diagnosis/{field}",
                )
                add_edge(
                    diagnosis_id,
                    "decomposes_into",
                    claim_id,
                    f"{relative}#/diagnosis/{field}",
                )
            for field, kind in (
                ("frozen_invariants", "frozen_invariant"),
                ("guardrails", "guardrail"),
            ):
                for claim_index, statement in enumerate(diagnosis[field]):
                    claim_id = f"{diagnosis_id}:{kind}:{claim_index:02d}"
                    pointer = f"/diagnosis/{field}/{claim_index}"
                    add_node(claim_id, kind, statement, relative, pointer)
                    add_edge(
                        diagnosis_id,
                        "decomposes_into",
                        claim_id,
                        f"{relative}#{pointer}",
                    )

            diagnosed_intervention = diagnosis["intervention"]
            diagnosed_primary = diagnosed_intervention["primary_lever"]
            diagnosed_dependent = diagnosed_intervention.get("dependent_lever")
            add_edge(
                diagnosis_id,
                "primary_intervention",
                diagnosed_primary,
                f"{relative}#/diagnosis/intervention/primary_lever",
            )
            if diagnosed_dependent is not None:
                add_edge(
                    diagnosis_id,
                    "dependent_intervention",
                    diagnosed_dependent,
                    f"{relative}#/diagnosis/intervention/dependent_lever",
                )
                add_edge(
                    diagnosed_primary,
                    "requires",
                    diagnosed_dependent,
                    f"{relative}#/diagnosis/intervention/dependent_lever_reason",
                )

    goal_id = f"goal:{goal['goal_id']}"
    add_node(
        goal_id,
        "goal",
        goal["objective"],
        "goal.json",
        "",
        {
            "claim_boundary": goal["claim_boundary"],
            "allowed_lever_prefixes": goal["allowed_lever_prefixes"],
            "forbidden_actions": goal["forbidden"],
        },
    )
    for index, prefix in enumerate(goal["allowed_lever_prefixes"]):
        add_edge(goal_id, "allows_prefix", prefix, f"goal.json#/allowed_lever_prefixes/{index}")
    for index, priority in enumerate(intervention_taxonomy["current_sft_priority"]):
        add_edge(goal_id, "prioritizes", priority, f"taxonomy.json#/current_sft_priority/{index}")

    state_id = "state:current"
    add_node(
        state_id,
        "state",
        state["next_action"],
        "state.json",
        "",
        {"goal_status": state["goal_status"]},
    )
    add_edge(state_id, "current_goal", goal_id, "state.json#/current_goal")
    add_edge(state_id, "current_version", state["current_version"], "state.json#/current_version")

    for edge in edges.values():
        for endpoint in (edge["source"], edge["target"]):
            if endpoint not in nodes:
                raise RSIContextError(
                    f"dangling graph edge {edge['relation']}: {endpoint}"
                )

    # Re-read the explicit source set before finalizing so a concurrent edit
    # cannot silently produce a graph assembled from two filesystem snapshots.
    for relative in sorted(sources):
        _read_source(root, relative, sources)
    sorted_sources = [_public_source(sources[key]) for key in sorted(sources)]
    source_set_sha256 = object_sha256(sorted_sources)
    graph: dict[str, Any] = {
        "schema_name": "rsi.context_graph",
        "schema_version": SCHEMA_VERSION,
        "generator": "rsi_context.py",
        "generator_sha256": sources["rsi_context.py"]["sha256"],
        "source_set_sha256": source_set_sha256,
        "source_files": sorted_sources,
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
    }
    graph["graph_sha256"] = object_sha256(graph)
    return graph


def _ranked_nodes(
    graph: Mapping[str, Any], seeds: list[str], max_cost: int
) -> tuple[list[tuple[int, str]], int]:
    nodes = {node["id"] for node in graph["nodes"]}
    unknown = [seed for seed in seeds if seed not in nodes]
    if unknown:
        raise RSIContextError(f"unknown context seed(s): {', '.join(unknown)}")
    adjacency: dict[str, list[tuple[str, int]]] = {node_id: [] for node_id in nodes}
    for edge in graph["edges"]:
        forward, reverse = RELATION_COSTS[edge["relation"]]
        adjacency[edge["source"]].append((edge["target"], forward))
        adjacency[edge["target"]].append((edge["source"], reverse))
    distances = {seed: 0 for seed in seeds}
    queue = [(0, seed) for seed in seeds]
    heapq.heapify(queue)
    while queue:
        cost, node_id = heapq.heappop(queue)
        if cost != distances[node_id] or cost >= max_cost:
            continue
        for neighbor, edge_cost in sorted(adjacency[node_id]):
            new_cost = cost + edge_cost
            if new_cost <= max_cost and new_cost < distances.get(neighbor, max_cost + 1):
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
    return sorted((cost, node_id) for node_id, cost in distances.items()), len(nodes)


def validate_graph_identity(graph: Mapping[str, Any]) -> None:
    supplied_graph_sha256 = graph.get("graph_sha256")
    unsigned = dict(graph)
    unsigned.pop("graph_sha256", None)
    if supplied_graph_sha256 != object_sha256(unsigned):
        raise RSIContextError("context graph hash mismatch")
    if graph.get("source_set_sha256") != object_sha256(graph.get("source_files")):
        raise RSIContextError("context graph source-set hash mismatch")


def build_validated_graph(root: Path = ROOT) -> dict[str, Any]:
    """Bracket graph compilation with controller and source-snapshot checks."""

    if root.resolve() != rsi_control.ROOT.resolve():
        raise RSIContextError("validated graph root differs from controller root")
    rsi_control.validate_control()
    graph = build_graph(root)
    rsi_control.validate_control()
    current_sources: dict[str, dict[str, Any]] = {}
    for source in graph["source_files"]:
        _read_source(root, source["path"], current_sources)
    if [
        _public_source(current_sources[path]) for path in sorted(current_sources)
    ] != graph["source_files"]:
        raise RSIContextError("graph sources changed across validation and compilation")
    validate_graph_identity(graph)
    return graph


def build_packet(
    graph: Mapping[str, Any],
    seeds: list[str],
    *,
    max_cost: int = 3,
    max_nodes: int = 24,
    max_bytes: int = 16_000,
) -> dict[str, Any]:
    """Select a deterministic, bounded neighborhood around one or more nodes."""

    validate_graph_identity(graph)
    if not seeds or len(set(seeds)) != len(seeds):
        raise RSIContextError("context seeds must be nonempty and unique")
    if max_cost < 0 or max_bytes < 1:
        raise RSIContextError("invalid context packet budget")

    ranked, graph_node_count = _ranked_nodes(graph, seeds, max_cost)
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    source_by_path = {source["path"]: source for source in graph["source_files"]}
    distance_by_id = {node_id: cost for cost, node_id in ranked}
    goal_ids = sorted(
        node_id for node_id, node in node_by_id.items() if node["kind"] == "goal"
    )
    if len(goal_ids) != 1:
        raise RSIContextError("context graph must contain exactly one current goal")
    goal_id = goal_ids[0]
    allowed_prefixes = node_by_id[goal_id]["attributes"]["allowed_lever_prefixes"]
    distance_by_id[goal_id] = 0
    mandatory = set(seeds) | set(goal_ids)
    if max_nodes < len(mandatory):
        raise RSIContextError("max_nodes cannot contain seeds and current goal")
    candidates = sorted(
        ((cost, node_id) for node_id, cost in distance_by_id.items()),
        key=lambda item: (item[0], item[1]),
    )
    selected = [node_id for _, node_id in candidates[:max_nodes]]

    def materialize(selected_ids: list[str]) -> dict[str, Any]:
        selected_set = set(selected_ids)
        packet_nodes = []
        source_paths: set[str] = set()
        for node_id in sorted(selected_set, key=lambda value: (distance_by_id[value], value)):
            node = dict(node_by_id[node_id])
            node["distance"] = distance_by_id[node_id]
            if node["kind"].startswith("intervention_"):
                if any(
                    rsi_control.segment_prefix(node_id, prefix)
                    for prefix in allowed_prefixes
                ):
                    authorization = "allowed"
                elif any(
                    rsi_control.segment_prefix(prefix, node_id)
                    for prefix in allowed_prefixes
                ):
                    authorization = "partially_allowed"
                else:
                    authorization = "outside_current_goal"
                node["current_goal_authorization"] = authorization
            packet_nodes.append(node)
            source_paths.add(node["source"]["path"])
        packet_edges = [
            edge
            for edge in graph["edges"]
            if edge["source"] in selected_set and edge["target"] in selected_set
        ]
        for edge in packet_edges:
            for locator in edge["locators"]:
                source_path = locator.partition("#")[0]
                if source_path not in source_by_path:
                    raise RSIContextError(
                        f"edge locator has no hashed source: {locator}"
                    )
                source_paths.add(source_path)
        packet: dict[str, Any] = {
            "schema_name": "rsi.context_packet",
            "schema_version": SCHEMA_VERSION,
            "generator": graph["generator"],
            "generator_sha256": graph["generator_sha256"],
            "graph_sha256": graph["graph_sha256"],
            "source_set_sha256": graph["source_set_sha256"],
            "request": {
                "seeds": seeds,
                "max_cost": max_cost,
                "max_nodes": max_nodes,
                "max_bytes": max_bytes,
            },
            "selection": {
                "included_nodes": len(packet_nodes),
                "candidate_nodes": len(candidates),
                "graph_nodes": graph_node_count,
                "omitted_candidate_nodes": len(candidates) - len(packet_nodes),
                "omission_reason": (
                    "max_nodes_or_max_bytes"
                    if len(packet_nodes) < len(candidates)
                    else None
                ),
            },
            "source_files": [source_by_path[path] for path in sorted(source_paths)],
            "nodes": packet_nodes,
            "edges": packet_edges,
        }
        packet["packet_sha256"] = object_sha256(packet)
        return packet

    while True:
        packet = materialize(selected)
        if emitted_json_bytes(packet) <= max_bytes:
            return packet
        removable = [node_id for node_id in reversed(selected) if node_id not in mandatory]
        if not removable:
            raise RSIContextError(
                "mandatory context seeds and provenance exceed max_bytes"
            )
        selected.remove(removable[0])


def search_graph(graph: Mapping[str, Any], query: str, limit: int = 20) -> list[dict[str, Any]]:
    normalized = query.casefold().strip()
    for alias, replacement in sorted(
        SEARCH_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        normalized = normalized.replace(alias, f" {replacement} ")
    terms = [term for term in re.findall(r"\w+", normalized) if term]
    if not terms or limit < 1:
        raise RSIContextError("search query and limit must be nonempty")
    matches: list[tuple[int, str, Mapping[str, Any]]] = []
    for node in graph["nodes"]:
        node_id = node["id"].casefold()
        attributes = json.dumps(
            node.get("attributes", {}), ensure_ascii=False, sort_keys=True
        ).casefold()
        text = f"{node_id} {node['summary'].casefold()} {attributes}"
        matched = [term for term in terms if term in text]
        if not matched:
            continue
        score = sum(8 if term in node_id else 2 for term in matched)
        if len(matched) == len(terms):
            score += 4
        if normalized == node_id:
            score += 100
        matches.append((-score, node["id"], node))
    return [
        {
            "id": node["id"],
            "kind": node["kind"],
            "summary": node["summary"],
            "source": node["source"],
        }
        for _, _, node in sorted(matches)[:limit]
    ]


def _validated_output_path(output: Path, root: Path = ROOT) -> Path:
    if not output.is_absolute():
        output = Path.cwd() / output
    resolved = output.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] != "generated":
        raise RSIContextError(
            "output inside rsi-control is allowed only under generated/"
        )
    return resolved


def _validate_output_source_disjoint(
    value: Mapping[str, Any] | list[dict[str, Any]],
    output: Path,
    root: Path = ROOT,
) -> None:
    if not isinstance(value, Mapping):
        return
    source_files = value.get("source_files", [])
    if not isinstance(source_files, list):
        raise RSIContextError("output source manifest is invalid")
    for source in source_files:
        if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
            raise RSIContextError("output source manifest is invalid")
        source_path = (root / _canonical_relative(source["path"])).resolve()
        same_file = False
        if output.exists() and source_path.exists():
            try:
                same_file = os.path.samefile(output, source_path)
            except OSError:
                same_file = False
        same_parent_alias = False
        if output.parent.exists() and source_path.parent.exists():
            try:
                same_parent = os.path.samefile(output.parent, source_path.parent)
            except OSError:
                same_parent = False
            output_name = unicodedata.normalize("NFC", output.name).casefold()
            source_name = unicodedata.normalize("NFC", source_path.name).casefold()
            same_parent_alias = same_parent and output_name == source_name
        if output == source_path or same_file or same_parent_alias:
            raise RSIContextError("output path overlaps a graph or packet source")


def _emit(value: Mapping[str, Any] | list[dict[str, Any]], output: Path | None) -> None:
    if output is None:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return
    output = _validated_output_path(output)
    _validate_output_source_disjoint(value, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rsi_control.atomic_write_json(output, value)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    graph_parser = subparsers.add_parser("graph", help="emit the complete graph")
    graph_parser.add_argument("--output", type=Path)
    search_parser = subparsers.add_parser("search", help="find graph node IDs")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--output", type=Path)
    pack_parser = subparsers.add_parser("pack", help="emit a bounded context packet")
    pack_parser.add_argument("--seed", action="append", required=True)
    pack_parser.add_argument("--max-cost", type=int, default=3)
    pack_parser.add_argument("--max-nodes", type=int, default=24)
    pack_parser.add_argument("--max-bytes", type=int, default=16_000)
    pack_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        graph = build_validated_graph()
        if args.command == "graph":
            result: Mapping[str, Any] | list[dict[str, Any]] = graph
        elif args.command == "search":
            result = search_graph(graph, args.query, args.limit)
        else:
            result = build_packet(
                graph,
                args.seed,
                max_cost=args.max_cost,
                max_nodes=args.max_nodes,
                max_bytes=args.max_bytes,
            )
        _emit(result, args.output)
    except (RSIContextError, rsi_control.RSIControlError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
