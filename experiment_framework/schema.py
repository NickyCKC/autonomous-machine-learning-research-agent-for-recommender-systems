"""Strict schema and validation for an experiment node."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping


NODE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ALLOWED_NODE_KEYS = {
    "schema_version",
    "node_id",
    "experiment_id",
    "template",
    "hypothesis",
    "seed",
    "parameters",
    "depends_on",
    "objective",
    "allow_failure",
}


class SchemaError(ValueError):
    """Raised when an experiment plan is invalid."""


@dataclass(frozen=True)
class Objective:
    metric: str
    direction: str = "maximize"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Objective":
        if not isinstance(value, Mapping):
            raise SchemaError("objective must be an object")
        unknown = set(value) - {"metric", "direction"}
        if unknown:
            raise SchemaError(f"unknown objective fields: {sorted(unknown)}")
        metric = value.get("metric")
        direction = value.get("direction", "maximize")
        if not isinstance(metric, str) or not metric.strip():
            raise SchemaError("objective.metric must be a non-empty string")
        if direction not in {"maximize", "minimize"}:
            raise SchemaError("objective.direction must be maximize or minimize")
        return cls(metric=metric, direction=direction)

    def to_dict(self) -> dict[str, str]:
        return {"metric": self.metric, "direction": self.direction}


@dataclass(frozen=True)
class ExperimentNode:
    node_id: str
    experiment_id: str
    template: str
    hypothesis: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    objective: Objective = field(
        default_factory=lambda: Objective("validation.synthetic_proxy")
    )
    allow_failure: bool = False
    schema_version: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentNode":
        if not isinstance(value, Mapping):
            raise SchemaError("each node must be an object")
        unknown = set(value) - ALLOWED_NODE_KEYS
        if unknown:
            raise SchemaError(f"unknown node fields: {sorted(unknown)}")
        missing = {
            "node_id",
            "experiment_id",
            "template",
            "hypothesis",
            "seed",
            "objective",
        } - set(value)
        if missing:
            raise SchemaError(f"missing node fields: {sorted(missing)}")
        if value.get("schema_version", 1) != 1:
            raise SchemaError("only node schema_version 1 is supported")

        node_id = value["node_id"]
        if not isinstance(node_id, str) or not NODE_ID_PATTERN.fullmatch(node_id):
            raise SchemaError(
                "node_id must match ^[a-z0-9][a-z0-9_-]{0,63}$"
            )
        for key in ("experiment_id", "template", "hypothesis"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise SchemaError(f"{key} must be a non-empty string")

        seed = value["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SchemaError("seed must be a non-negative integer")
        parameters = value.get("parameters", {})
        if not isinstance(parameters, dict):
            raise SchemaError("parameters must be an object")
        try:
            json.dumps(parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaError("parameters must contain finite JSON values") from exc

        depends_on = value.get("depends_on", [])
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) and NODE_ID_PATTERN.fullmatch(item)
            for item in depends_on
        ):
            raise SchemaError("depends_on must be a list of valid node IDs")
        if len(depends_on) != len(set(depends_on)):
            raise SchemaError("depends_on cannot contain duplicates")
        if node_id in depends_on:
            raise SchemaError("a node cannot depend on itself")

        allow_failure = value.get("allow_failure", False)
        if not isinstance(allow_failure, bool):
            raise SchemaError("allow_failure must be a boolean")

        return cls(
            node_id=node_id,
            experiment_id=value["experiment_id"],
            template=value["template"],
            hypothesis=value["hypothesis"],
            seed=seed,
            parameters=dict(parameters),
            depends_on=tuple(depends_on),
            objective=Objective.from_mapping(value["objective"]),
            allow_failure=allow_failure,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "experiment_id": self.experiment_id,
            "template": self.template,
            "hypothesis": self.hypothesis,
            "seed": self.seed,
            "parameters": self.parameters,
            "depends_on": list(self.depends_on),
            "objective": self.objective.to_dict(),
            "allow_failure": self.allow_failure,
        }


def load_plan(path: str) -> list[ExperimentNode]:
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or set(document) != {"schema_version", "nodes"}:
        raise SchemaError("plan must contain exactly schema_version and nodes")
    if document["schema_version"] != 1:
        raise SchemaError("only plan schema_version 1 is supported")
    if not isinstance(document["nodes"], list) or not document["nodes"]:
        raise SchemaError("plan.nodes must be a non-empty list")
    nodes = [ExperimentNode.from_mapping(value) for value in document["nodes"]]
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise SchemaError("node IDs must be unique")
    positions = {node_id: index for index, node_id in enumerate(node_ids)}
    for index, node in enumerate(nodes):
        unknown = set(node.depends_on) - set(node_ids)
        if unknown:
            raise SchemaError(
                f"node {node.node_id} has unknown dependencies: {sorted(unknown)}"
            )
        if any(positions[parent] >= index for parent in node.depends_on):
            raise SchemaError(
                f"node {node.node_id} dependencies must appear earlier in the plan"
            )
    return nodes
