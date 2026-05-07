from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


REQUIRED_VALIDATION_MODEL = (
    "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1"
)
REQUIRED_SCHEMA_VERSION = "bumkc.optimization_playground.v1"


class BumkcArtifactError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class BumkcRuntimeSummary:
    task_count: int
    conventional_launch_count: int
    persistent_launch_count: int
    jit_task_count: int
    aot_task_count: int
    queue_capacity: int
    task_instance_capacity: int
    device_global_binding_count: int
    kv_cache_binding_count: int

    def as_log_dict(self) -> dict[str, int]:
        return {
            "task_count": self.task_count,
            "conventional_launch_count": self.conventional_launch_count,
            "persistent_launch_count": self.persistent_launch_count,
            "jit_task_count": self.jit_task_count,
            "aot_task_count": self.aot_task_count,
            "queue_capacity": self.queue_capacity,
            "task_instance_capacity": self.task_instance_capacity,
            "device_global_binding_count": self.device_global_binding_count,
            "kv_cache_binding_count": self.kv_cache_binding_count,
        }


@dataclasses.dataclass(frozen=True)
class BumkcArtifactSummary:
    root: Path
    plan_id: str
    program_id: str
    model: str
    gpu_count: int
    fallback_mode: str
    runtime_executable: bool
    runtime_entrypoints: tuple[str, ...]
    runtime_summary: BumkcRuntimeSummary
    task_count: int
    tensor_smoke_enabled: bool
    required_validation_model: str

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "plan_id": self.plan_id,
            "program_id": self.program_id,
            "model": self.model,
            "gpu_count": self.gpu_count,
            "fallback_mode": self.fallback_mode,
            "runtime_executable": self.runtime_executable,
            "runtime_entrypoints": list(self.runtime_entrypoints),
            "runtime_summary": self.runtime_summary.as_log_dict(),
            "task_count": self.task_count,
            "tensor_smoke_enabled": self.tensor_smoke_enabled,
            "required_validation_model": self.required_validation_model,
        }


def load_bumkc_artifact(
    path: str | Path, *, require_executable: bool = False
) -> BumkcArtifactSummary:
    root = _resolve_plan_dir(Path(path))
    manifest = _read_json(root / "manifest.json")
    runtime = _read_json(root / "runtime" / "plan.json")
    engine = _read_json(root / "engine" / "optimization-playground.json")
    tensor_smoke = _read_json(root / "generated" / "tensor-smoke.json")

    _validate_identity(manifest, runtime, engine, tensor_smoke)
    if engine.get("schema_version") != REQUIRED_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported engine schema")
    if engine.get("engine") != "sglang":
        raise BumkcArtifactError("BUMKC artifact is not targeted at the SGLang engine")
    if engine.get("engine_profile") != "optimization_playground":
        raise BumkcArtifactError("BUMKC artifact is not for optimization-playground")
    if engine.get("fallback_mode") != "checked":
        raise BumkcArtifactError("BUMKC artifact must use checked fallback mode")
    if not engine.get("preserve_custom_optimizations"):
        raise BumkcArtifactError("BUMKC artifact does not preserve custom optimizations")
    if engine.get("required_validation_model") != REQUIRED_VALIDATION_MODEL:
        raise BumkcArtifactError("BUMKC artifact validation model is not the REAP target")
    if engine.get("runtime_executable") != runtime.get("executable"):
        raise BumkcArtifactError("BUMKC engine and runtime executable flags disagree")
    if require_executable and not runtime.get("executable"):
        raise BumkcArtifactError("BUMKC artifact is not executable")

    runtime_summary = _load_runtime_summary(engine, runtime)
    entrypoints = tuple(
        entrypoint.get("name", "") for entrypoint in runtime.get("entrypoints", [])
    )
    return BumkcArtifactSummary(
        root=root,
        plan_id=engine["plan_id"],
        program_id=engine["program_id"],
        model=engine["model"],
        gpu_count=int(engine["gpu_count"]),
        fallback_mode=engine["fallback_mode"],
        runtime_executable=bool(runtime["executable"]),
        runtime_entrypoints=entrypoints,
        runtime_summary=runtime_summary,
        task_count=runtime_summary.task_count,
        tensor_smoke_enabled=bool(tensor_smoke["enabled"]),
        required_validation_model=engine["required_validation_model"],
    )


def _resolve_plan_dir(path: Path) -> Path:
    if (path / "manifest.json").is_file():
        return path

    manifests = sorted(path.glob("*/manifest.json"))
    if len(manifests) == 1:
        return manifests[0].parent
    if not manifests:
        raise BumkcArtifactError(f"no BUMKC manifest found below {path}")
    raise BumkcArtifactError(f"multiple BUMKC plans found below {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise BumkcArtifactError(f"missing BUMKC artifact file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BumkcArtifactError(f"invalid BUMKC artifact JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BumkcArtifactError(f"BUMKC artifact file is not a JSON object: {path}")
    return value


def _validate_identity(*objects: dict[str, Any]) -> None:
    plan_ids = {obj.get("plan_id") for obj in objects}
    program_ids = {obj.get("program_id") for obj in objects}
    if len(plan_ids) != 1 or None in plan_ids:
        raise BumkcArtifactError("BUMKC artifact plan IDs do not match")
    if len(program_ids) != 1 or None in program_ids:
        raise BumkcArtifactError("BUMKC artifact program IDs do not match")


def _load_runtime_summary(
    engine: dict[str, Any], runtime: dict[str, Any]
) -> BumkcRuntimeSummary:
    summary = engine.get("runtime_summary")
    if not isinstance(summary, dict):
        raise BumkcArtifactError("BUMKC engine runtime summary is missing")

    execution_model = _read_object(runtime, "execution_model")
    queue_plan = _read_object(runtime, "queue_plan")
    memory_plan = _read_object(runtime, "memory_plan")
    expected = {
        "task_count": runtime.get("task_count"),
        "conventional_launch_count": execution_model.get("conventional_launch_count"),
        "persistent_launch_count": execution_model.get("persistent_launch_count"),
        "jit_task_count": queue_plan.get("jit_task_count"),
        "aot_task_count": queue_plan.get("aot_task_count"),
        "queue_capacity": queue_plan.get("queue_capacity"),
        "task_instance_capacity": queue_plan.get("task_instance_capacity"),
        "device_global_binding_count": memory_plan.get("device_global_binding_count"),
        "kv_cache_binding_count": memory_plan.get("kv_cache_binding_count"),
    }
    for key, value in expected.items():
        if summary.get(key) is None or value is None or summary.get(key) != value:
            raise BumkcArtifactError(f"BUMKC engine runtime summary mismatch: {key}")

    return BumkcRuntimeSummary(**{key: int(value) for key, value in expected.items()})


def _read_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise BumkcArtifactError(f"BUMKC runtime {key} object is missing")
    return value
