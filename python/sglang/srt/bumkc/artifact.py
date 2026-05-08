from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT_DIGEST_PATH = Path("reports/artifact-digests.json")
REQUIRED_DIGEST_SCHEMA_VERSION = "bumkc.artifact_digests.v0"
REQUIRED_VALIDATION_MODEL = (
    "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1"
)
REQUIRED_SCHEMA_VERSION = "bumkc.optimization_playground.v8"


class BumkcArtifactError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class BumkcRuntimeShapeSymbolBinding:
    symbol: str
    min: int
    max: int
    bucket: int
    default_value: int

    def bucketed_value(self, value: int) -> int:
        return ((value + self.bucket - 1) // self.bucket) * self.bucket


@dataclasses.dataclass(frozen=True)
class BumkcRuntimeServingStateBinding:
    kind: str
    symbol: str | None
    required: bool

    def key(self) -> tuple[str, str | None]:
        return (self.kind, self.symbol)


@dataclasses.dataclass(frozen=True)
class BumkcRuntimeShapeSymbolValue:
    symbol: str
    value: int
    bucketed_value: int


@dataclasses.dataclass(frozen=True)
class BumkcRuntimeLaunchPlan:
    shape_symbols: tuple[BumkcRuntimeShapeSymbolValue, ...]
    serving_state: tuple[BumkcRuntimeServingStateBinding, ...]


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
    rank_count: int
    task_rank_group_count: int
    task_rank_reference_count: int
    rank_id_sum: int
    task_dependency_count: int
    tile_overlap_dependency_count: int
    whole_producer_dependency_count: int
    dependency_scope_code_sum: int
    collective_task_count: int
    collective_group_size_sum: int
    collective_kind_code_sum: int
    serving_task_count: int
    serving_dependency_count: int
    serving_kind_code_sum: int
    serving_symbol_count: int
    substitution_shape_symbol_count: int
    substitution_serving_binding_count: int
    substitution_symbol_max_sum: int
    substitution_symbol_bucket_sum: int

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
            "rank_count": self.rank_count,
            "task_rank_group_count": self.task_rank_group_count,
            "task_rank_reference_count": self.task_rank_reference_count,
            "rank_id_sum": self.rank_id_sum,
            "task_dependency_count": self.task_dependency_count,
            "tile_overlap_dependency_count": self.tile_overlap_dependency_count,
            "whole_producer_dependency_count": self.whole_producer_dependency_count,
            "dependency_scope_code_sum": self.dependency_scope_code_sum,
            "collective_task_count": self.collective_task_count,
            "collective_group_size_sum": self.collective_group_size_sum,
            "collective_kind_code_sum": self.collective_kind_code_sum,
            "serving_task_count": self.serving_task_count,
            "serving_dependency_count": self.serving_dependency_count,
            "serving_kind_code_sum": self.serving_kind_code_sum,
            "serving_symbol_count": self.serving_symbol_count,
            "substitution_shape_symbol_count": self.substitution_shape_symbol_count,
            "substitution_serving_binding_count": (
                self.substitution_serving_binding_count
            ),
            "substitution_symbol_max_sum": self.substitution_symbol_max_sum,
            "substitution_symbol_bucket_sum": self.substitution_symbol_bucket_sum,
        }


@dataclasses.dataclass(frozen=True)
class BumkcArtifactSummary:
    root: Path
    plan_id: str
    program_id: str
    model: str
    gpu_count: int
    target_arch: str | None
    fallback_mode: str
    runtime_executable: bool
    runtime_entrypoints: tuple[str, ...]
    runtime_summary: BumkcRuntimeSummary
    runtime_shape_symbols: tuple[BumkcRuntimeShapeSymbolBinding, ...]
    runtime_serving_state: tuple[BumkcRuntimeServingStateBinding, ...]
    task_count: int
    tensor_smoke_enabled: bool
    artifact_digest_count: int
    required_validation_model: str

    def validate_runtime_launch(
        self,
        *,
        shape_symbols: Mapping[str, int],
        serving_state: Iterable[tuple[str, str | None]],
    ) -> BumkcRuntimeLaunchPlan:
        expected_symbols = {binding.symbol for binding in self.runtime_shape_symbols}
        extra_symbols = sorted(set(shape_symbols) - expected_symbols)
        if extra_symbols:
            raise BumkcArtifactError(
                f"unknown BUMKC runtime shape symbol: {extra_symbols[0]}"
            )

        validated_shape_symbols = []
        for binding in self.runtime_shape_symbols:
            if binding.symbol not in shape_symbols:
                raise BumkcArtifactError(
                    f"missing BUMKC runtime shape symbol: {binding.symbol}"
                )
            value = shape_symbols[binding.symbol]
            if not _is_strict_int(value):
                raise BumkcArtifactError(
                    f"BUMKC runtime shape symbol {binding.symbol} is not an integer"
                )
            if value < binding.min or value > binding.max:
                raise BumkcArtifactError(
                    "BUMKC runtime shape symbol "
                    f"{binding.symbol}={value} is outside "
                    f"{binding.min}..={binding.max}"
                )
            bucketed_value = binding.bucketed_value(value)
            if bucketed_value > binding.max:
                raise BumkcArtifactError(
                    "BUMKC runtime shape symbol "
                    f"{binding.symbol}={value} buckets to "
                    f"{bucketed_value} beyond max {binding.max}"
                )
            validated_shape_symbols.append(
                BumkcRuntimeShapeSymbolValue(
                    symbol=binding.symbol,
                    value=value,
                    bucketed_value=bucketed_value,
                )
            )

        provided_serving_state = _normalize_serving_state(serving_state)
        expected_serving_state = {
            binding.key() for binding in self.runtime_serving_state
        }
        extra_serving_state = sorted(
            provided_serving_state - expected_serving_state,
            key=lambda item: (item[0], "" if item[1] is None else item[1]),
        )
        if extra_serving_state:
            kind, symbol = extra_serving_state[0]
            raise BumkcArtifactError(
                f"unknown BUMKC serving-state binding: {kind}:{symbol}"
            )
        for binding in self.runtime_serving_state:
            if binding.required and binding.key() not in provided_serving_state:
                raise BumkcArtifactError(
                    "missing BUMKC serving-state binding: "
                    f"{binding.kind}:{binding.symbol}"
                )

        return BumkcRuntimeLaunchPlan(
            shape_symbols=tuple(validated_shape_symbols),
            serving_state=tuple(self.runtime_serving_state),
        )

    def validate_default_runtime_launch(self) -> BumkcRuntimeLaunchPlan:
        return self.validate_runtime_launch(
            shape_symbols={
                binding.symbol: binding.default_value
                for binding in self.runtime_shape_symbols
            },
            serving_state=[
                binding.key()
                for binding in self.runtime_serving_state
                if binding.required
            ],
        )

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "plan_id": self.plan_id,
            "program_id": self.program_id,
            "model": self.model,
            "gpu_count": self.gpu_count,
            "target_arch": self.target_arch,
            "fallback_mode": self.fallback_mode,
            "runtime_executable": self.runtime_executable,
            "runtime_entrypoints": list(self.runtime_entrypoints),
            "runtime_summary": self.runtime_summary.as_log_dict(),
            "runtime_shape_symbols": [
                dataclasses.asdict(binding)
                for binding in self.runtime_shape_symbols
            ],
            "runtime_serving_state": [
                dataclasses.asdict(binding) for binding in self.runtime_serving_state
            ],
            "task_count": self.task_count,
            "tensor_smoke_enabled": self.tensor_smoke_enabled,
            "artifact_digest_count": self.artifact_digest_count,
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
    artifact_paths = _read_object(engine, "artifact_paths")
    if artifact_paths.get("artifact_digests") != ARTIFACT_DIGEST_PATH.as_posix():
        raise BumkcArtifactError("BUMKC artifact digest path is not canonical")
    artifact_digest_count = _validate_artifact_digests(root, manifest)
    if engine.get("target_arch") != manifest.get("target_arch"):
        raise BumkcArtifactError("BUMKC engine target architecture does not match manifest")
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
    runtime_shape_symbols = _load_shape_symbol_bindings(runtime)
    runtime_serving_state = _load_serving_state_bindings(runtime)
    entrypoints = tuple(
        entrypoint.get("name", "") for entrypoint in runtime.get("entrypoints", [])
    )
    return BumkcArtifactSummary(
        root=root,
        plan_id=engine["plan_id"],
        program_id=engine["program_id"],
        model=engine["model"],
        gpu_count=int(engine["gpu_count"]),
        target_arch=engine.get("target_arch"),
        fallback_mode=engine["fallback_mode"],
        runtime_executable=bool(runtime["executable"]),
        runtime_entrypoints=entrypoints,
        runtime_summary=runtime_summary,
        runtime_shape_symbols=runtime_shape_symbols,
        runtime_serving_state=runtime_serving_state,
        task_count=runtime_summary.task_count,
        tensor_smoke_enabled=bool(tensor_smoke["enabled"]),
        artifact_digest_count=artifact_digest_count,
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


def _validate_artifact_digests(root: Path, manifest: dict[str, Any]) -> int:
    digest = _read_json(root / ARTIFACT_DIGEST_PATH)
    if digest.get("schema_version") != REQUIRED_DIGEST_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC artifact digest schema is unsupported")
    if digest.get("plan_id") != manifest.get("plan_id"):
        raise BumkcArtifactError("BUMKC artifact digest plan ID does not match")

    files = _read_list(digest, "files")
    expected_paths = []
    for entry in files:
        relative_path = _read_digest_path(entry, "path")
        if relative_path == ARTIFACT_DIGEST_PATH.as_posix():
            raise BumkcArtifactError("BUMKC artifact digest cannot contain itself")
        expected_paths.append(relative_path)
    if len(set(expected_paths)) != len(expected_paths):
        raise BumkcArtifactError("BUMKC artifact digest contains duplicate paths")

    actual_paths = _artifact_file_paths(root)
    if sorted(expected_paths) != actual_paths:
        raise BumkcArtifactError("BUMKC artifact digest file list does not match")

    for entry in files:
        relative_path = _read_digest_path(entry, "path")
        expected_bytes = _read_digest_int(entry, "bytes")
        expected_sha256 = _read_digest_sha256(entry, "sha256")
        path = root / relative_path
        if path.is_symlink():
            raise BumkcArtifactError("BUMKC artifact digest rejects symlinks")
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise BumkcArtifactError(
                f"missing BUMKC artifact digest file: {relative_path}"
            ) from exc
        if len(data) != expected_bytes:
            raise BumkcArtifactError(
                f"BUMKC artifact digest byte mismatch: {relative_path}"
            )
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise BumkcArtifactError(
                f"BUMKC artifact digest SHA-256 mismatch: {relative_path}"
            )
    return len(files)


def _artifact_file_paths(root: Path) -> list[str]:
    paths = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise BumkcArtifactError("BUMKC artifact digest rejects symlinks")
        relative_path = path.relative_to(root).as_posix()
        if relative_path != ARTIFACT_DIGEST_PATH.as_posix():
            paths.append(relative_path)
    return sorted(paths)


def _read_digest_path(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise BumkcArtifactError(f"BUMKC artifact digest {key} string is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise BumkcArtifactError(f"BUMKC artifact digest {key} path is invalid")
    return path.as_posix()


def _read_digest_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not _is_strict_int(value) or value < 0:
        raise BumkcArtifactError(f"BUMKC artifact digest {key} integer is missing")
    return value


def _read_digest_sha256(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise BumkcArtifactError(f"BUMKC artifact digest {key} SHA-256 is missing")
    if any(char not in "0123456789abcdef" for char in value):
        raise BumkcArtifactError(f"BUMKC artifact digest {key} SHA-256 is invalid")
    return value


def _load_runtime_summary(
    engine: dict[str, Any], runtime: dict[str, Any]
) -> BumkcRuntimeSummary:
    summary = engine.get("runtime_summary")
    if not isinstance(summary, dict):
        raise BumkcArtifactError("BUMKC engine runtime summary is missing")

    execution_model = _read_object(runtime, "execution_model")
    queue_plan = _read_object(runtime, "queue_plan")
    memory_plan = _read_object(runtime, "memory_plan")
    scale_up_plan = _read_object(runtime, "scale_up_plan")
    communication_plan = _read_object(runtime, "communication_plan")
    serving_state_plan = _read_object(runtime, "serving_state_plan")
    substitution_plan = _read_object(runtime, "substitution_plan")
    substitution_shape_symbols = _read_list(substitution_plan, "shape_symbols")
    substitution_serving_state = _read_list(substitution_plan, "serving_state")
    dependency_plan = _read_object(runtime, "dependency_plan")
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
        "rank_count": scale_up_plan.get("rank_count"),
        "task_rank_group_count": scale_up_plan.get("task_rank_group_count"),
        "task_rank_reference_count": scale_up_plan.get("task_rank_reference_count"),
        "rank_id_sum": scale_up_plan.get("rank_id_sum"),
        "task_dependency_count": dependency_plan.get("task_dependency_count"),
        "tile_overlap_dependency_count": dependency_plan.get(
            "tile_overlap_dependency_count"
        ),
        "whole_producer_dependency_count": dependency_plan.get(
            "whole_producer_dependency_count"
        ),
        "dependency_scope_code_sum": dependency_plan.get("dependency_scope_code_sum"),
        "collective_task_count": communication_plan.get("collective_task_count"),
        "collective_group_size_sum": communication_plan.get(
            "collective_group_size_sum"
        ),
        "collective_kind_code_sum": communication_plan.get(
            "collective_kind_code_sum"
        ),
        "serving_task_count": serving_state_plan.get("serving_task_count"),
        "serving_dependency_count": serving_state_plan.get(
            "serving_dependency_count"
        ),
        "serving_kind_code_sum": serving_state_plan.get("serving_kind_code_sum"),
        "serving_symbol_count": serving_state_plan.get("serving_symbol_count"),
        "substitution_shape_symbol_count": len(substitution_shape_symbols),
        "substitution_serving_binding_count": len(substitution_serving_state),
        "substitution_symbol_max_sum": sum(
            _read_int(symbol, "max") for symbol in substitution_shape_symbols
        ),
        "substitution_symbol_bucket_sum": sum(
            _read_int(symbol, "bucket") for symbol in substitution_shape_symbols
        ),
    }
    for key, value in expected.items():
        if summary.get(key) is None or value is None or summary.get(key) != value:
            raise BumkcArtifactError(f"BUMKC engine runtime summary mismatch: {key}")

    return BumkcRuntimeSummary(**{key: int(value) for key, value in expected.items()})


def _load_shape_symbol_bindings(
    runtime: dict[str, Any],
) -> tuple[BumkcRuntimeShapeSymbolBinding, ...]:
    substitution_plan = _read_object(runtime, "substitution_plan")
    bindings = []
    seen_symbols = set()
    for entry in _read_list(substitution_plan, "shape_symbols"):
        binding = BumkcRuntimeShapeSymbolBinding(
            symbol=_read_str(entry, "symbol"),
            min=_read_int(entry, "min"),
            max=_read_int(entry, "max"),
            bucket=_read_int(entry, "bucket"),
            default_value=_read_int(entry, "default_value"),
        )
        if binding.min < 1 or binding.max < binding.min:
            raise BumkcArtifactError(
                f"BUMKC runtime shape symbol {binding.symbol} has invalid bounds"
            )
        if binding.bucket < 1 or binding.bucket > binding.max:
            raise BumkcArtifactError(
                f"BUMKC runtime shape symbol {binding.symbol} has invalid bucket"
            )
        if binding.default_value < binding.min or binding.default_value > binding.max:
            raise BumkcArtifactError(
                f"BUMKC runtime shape symbol {binding.symbol} has invalid default"
            )
        if binding.symbol in seen_symbols:
            raise BumkcArtifactError(
                f"BUMKC runtime shape symbol {binding.symbol} is duplicated"
            )
        seen_symbols.add(binding.symbol)
        bindings.append(binding)
    return tuple(bindings)


def _load_serving_state_bindings(
    runtime: dict[str, Any],
) -> tuple[BumkcRuntimeServingStateBinding, ...]:
    substitution_plan = _read_object(runtime, "substitution_plan")
    bindings = []
    seen_bindings = set()
    for entry in _read_list(substitution_plan, "serving_state"):
        binding = BumkcRuntimeServingStateBinding(
            kind=_read_str(entry, "kind"),
            symbol=_read_optional_str(entry, "symbol"),
            required=_read_bool(entry, "required"),
        )
        if binding.key() in seen_bindings:
            raise BumkcArtifactError(
                "BUMKC runtime serving-state binding "
                f"{binding.kind}:{binding.symbol} is duplicated"
            )
        seen_bindings.add(binding.key())
        bindings.append(binding)
    return tuple(bindings)


def _read_object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise BumkcArtifactError(f"BUMKC runtime {key} object is missing")
    return value


def _read_list(parent: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise BumkcArtifactError(f"BUMKC runtime {key} list is missing")
    if not all(isinstance(entry, dict) for entry in value):
        raise BumkcArtifactError(f"BUMKC runtime {key} list has non-object entries")
    return value


def _read_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not _is_strict_int(value):
        raise BumkcArtifactError(f"BUMKC runtime {key} integer is missing")
    return value


def _read_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise BumkcArtifactError(f"BUMKC runtime {key} string is missing")
    return value


def _read_optional_str(parent: dict[str, Any], key: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BumkcArtifactError(f"BUMKC runtime {key} string is invalid")
    return value


def _read_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise BumkcArtifactError(f"BUMKC runtime {key} boolean is missing")
    return value


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_serving_state(
    serving_state: Iterable[tuple[str, str | None]],
) -> set[tuple[str, str | None]]:
    normalized = set()
    for entry in serving_state:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise BumkcArtifactError("BUMKC serving-state binding is malformed")
        kind, symbol = entry
        if not isinstance(kind, str) or not kind:
            raise BumkcArtifactError("BUMKC serving-state kind is malformed")
        if symbol is not None and (not isinstance(symbol, str) or not symbol):
            raise BumkcArtifactError("BUMKC serving-state symbol is malformed")
        key = (kind, symbol)
        if key in normalized:
            raise BumkcArtifactError(
                f"BUMKC serving-state binding is duplicated: {kind}:{symbol}"
            )
        normalized.add(key)
    return normalized
