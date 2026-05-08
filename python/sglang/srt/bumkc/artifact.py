from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


ARTIFACT_DIGEST_PATH = Path("reports/artifact-digests.json")
REQUIRED_DIGEST_SCHEMA_VERSION = "bumkc.artifact_digests.v0"
REQUIRED_CLI_FLAGS = (
    "--enable-bumkc",
    "--bumkc-plan-path <artifact-root>/<plan-id>",
    "--bumkc-fallback-mode checked",
)
REQUIRED_VALIDATION_MODEL = (
    "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1"
)
REQUIRED_PLAN_SCHEMA_VERSION = "bumkc.plan.v1"
REQUIRED_CAPABILITY_LEVEL = "hvm_rooted_runtime_descriptor"
REQUIRED_SCHEMA_VERSION = "bumkc.optimization_playground.v16"
REQUIRED_SOURCE_SCHEMA_VERSION = "bumkc.source.v8"
REQUIRED_RUNTIME_ABI_VERSION = "bumkc.runtime.v1"
REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION = "bumkc.cuda_smoke.v11"
_CONTRACT_HASH_OFFSET = 0xCBF29CE484222325
_CONTRACT_HASH_PRIME = 0x00000100000001B3
_U64_MASK = (1 << 64) - 1
_DESCRIPTOR_CONTRACT_DOMAIN = "bumkc.cuda_smoke.descriptors.v1"
_SOURCE_CONTRACT_DOMAIN = "bumkc.cuda_smoke.source.v1"
_SOURCE_CONTRACT_KEYS = (
    "expected_task_count",
    "expected_conventional_launch_count",
    "expected_persistent_launch_count",
    "expected_launch_reduction_per_mille",
    "expected_jit_task_count",
    "expected_aot_task_count",
    "expected_queue_capacity",
    "expected_task_instance_capacity",
    "expected_predecessor_event_count_sum",
    "expected_dependency_edge_count",
    "expected_dependency_tensor_count",
    "expected_dependency_scope_code_sum",
    "expected_launch_domain_rank_sum",
    "expected_launch_domain_element_sum",
    "expected_operator_code_sum",
    "expected_kv_cache_binding_count",
    "expected_communication_task_count",
    "expected_communication_group_size_sum",
    "expected_communication_kind_code_sum",
    "expected_side_effecting_task_count",
    "expected_task_side_effect_count",
    "expected_task_side_effect_code_sum",
    "expected_serving_task_count",
    "expected_serving_dependency_count",
    "expected_serving_kind_code_sum",
    "expected_serving_symbol_count",
    "expected_substitution_shape_symbol_count",
    "expected_substitution_serving_binding_count",
    "expected_substitution_symbol_max_sum",
    "expected_substitution_symbol_bucket_sum",
    "expected_rank_group_size_sum",
    "expected_rank_id_sum",
    "expected_event_tensor_count",
    "expected_event_predecessor_edge_count",
    "expected_event_successor_edge_count",
    "expected_event_notification_count",
    "expected_event_execution_count",
    "expected_event_simulation_violation_count",
)
_SIDE_EFFECT_CODES = {
    "kv_cache_read": 1,
    "kv_cache_write": 2,
    "collective": 3,
    "host_visible_output": 4,
    "profiling_write": 5,
}
_OPERATOR_CODES = {
    "matmul": 1,
    "attention": 2,
    "rms_norm": 3,
    "rope": 4,
    "mlp": 5,
    "moe_dispatch": 6,
    "kv_cache_read": 7,
    "kv_cache_write": 8,
    "collective": 9,
    "unknown": 0,
}
_SCHEDULING_POLICY_CODES = {
    "static_aot": 0,
    "dynamic_jit": 1,
}
_DEPENDENCY_SCOPE_CODES = {
    "tile_overlap": 1,
    "whole_producer": 2,
}
_SERVING_STATE_KIND_CODES = {
    "batch": 1,
    "sequence": 2,
    "decode_step": 3,
    "token_ids": 4,
    "kv_cache_pages": 5,
}
_DEPENDENCY_HASH_OFFSET = 14695981039346656037
_DEPENDENCY_HASH_PRIME = 1099511628211


def _contract_hash_str(value: str) -> int:
    return _mix_contract_str(_CONTRACT_HASH_OFFSET, value)


def _mix_contract_str(hash_value: int, value: str) -> int:
    hash_value = _mix_contract_u64(hash_value, len(value.encode("utf-8")))
    for byte in value.encode("utf-8"):
        hash_value = _mix_contract_byte(hash_value, byte)
    return hash_value


def _mix_contract_u64(hash_value: int, value: int) -> int:
    if value < 0 or value > _U64_MASK:
        raise BumkcArtifactError("BUMKC runtime smoke contract value is out of range")
    for byte in value.to_bytes(8, byteorder="little", signed=False):
        hash_value = _mix_contract_byte(hash_value, byte)
    return hash_value


def _mix_contract_byte(hash_value: int, byte: int) -> int:
    return ((hash_value ^ byte) * _CONTRACT_HASH_PRIME) & _U64_MASK


def _dependency_descriptor_hash(descriptors: list[dict[str, Any]]) -> int:
    hash_value = _DEPENDENCY_HASH_OFFSET
    for descriptor in descriptors:
        hash_value = _mix_dependency_hash_str(hash_value, _read_str(descriptor, "task"))
        hash_value = _mix_dependency_hash_str(
            hash_value, _read_str(descriptor, "consumer_event")
        )
        hash_value = _mix_dependency_hash_str(
            hash_value, _read_str(descriptor, "predecessor_event")
        )
        hash_value = _mix_dependency_hash_u64(
            hash_value, _read_int(descriptor, "dependency_ordinal")
        )
        tensors = _read_any_list(descriptor, "tensors")
        hash_value = _mix_dependency_hash_u64(hash_value, len(tensors))
        for tensor in tensors:
            if not isinstance(tensor, str) or not tensor:
                raise BumkcArtifactError(
                    "BUMKC dependency descriptor tensor is malformed"
                )
            hash_value = _mix_dependency_hash_str(hash_value, tensor)
        scope = _read_str(descriptor, "scope")
        if scope not in _DEPENDENCY_SCOPE_CODES:
            raise BumkcArtifactError(f"BUMKC dependency scope is unsupported: {scope}")
        hash_value = _mix_dependency_hash_u64(
            hash_value, _DEPENDENCY_SCOPE_CODES[scope]
        )
        hash_value = _mix_dependency_hash_str(
            hash_value, _read_str(descriptor, "wait_expression")
        )
    return hash_value


def _mix_dependency_hash_str(hash_value: int, value: str) -> int:
    encoded = value.encode("utf-8")
    hash_value = _mix_dependency_hash_u64(hash_value, len(encoded))
    for byte in encoded:
        hash_value = _mix_dependency_hash_u64(hash_value, byte)
    return hash_value


def _mix_dependency_hash_u64(hash_value: int, value: int) -> int:
    if value < 0 or value > _U64_MASK:
        raise BumkcArtifactError("BUMKC dependency descriptor value is out of range")
    return ((hash_value * _DEPENDENCY_HASH_PRIME) + value) & _U64_MASK


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
class BumkcCompilerSummary:
    tensor_island_count: int
    native_tensor_island_count: int
    fallback_tensor_island_count: int
    side_effecting_tensor_island_count: int
    tensor_island_side_effect_count: int
    tensor_island_side_effect_code_sum: int
    fallback_bridge_count: int
    moe_dispatch_tensor_island_count: int
    block_op_count: int
    moe_dispatch_block_op_count: int
    side_effecting_block_op_count: int
    block_side_effect_count: int
    block_side_effect_code_sum: int
    event_tensor_count: int
    moe_dispatch_event_tensor_count: int
    side_effecting_event_tensor_count: int
    event_side_effect_count: int
    event_side_effect_code_sum: int
    event_predecessor_edge_count: int
    event_successor_edge_count: int
    event_dependency_tensor_count: int
    event_notification_count: int
    event_execution_count: int
    event_simulation_violation_count: int

    def as_log_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


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
    dependency_tensor_count: int
    dependency_scope_code_sum: int
    dependency_descriptor_count: int
    dependency_descriptor_hash: int
    collective_task_count: int
    collective_group_size_sum: int
    collective_kind_code_sum: int
    side_effecting_task_count: int
    task_side_effect_count: int
    task_side_effect_code_sum: int
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
            "dependency_tensor_count": self.dependency_tensor_count,
            "dependency_scope_code_sum": self.dependency_scope_code_sum,
            "dependency_descriptor_count": self.dependency_descriptor_count,
            "dependency_descriptor_hash": self.dependency_descriptor_hash,
            "collective_task_count": self.collective_task_count,
            "collective_group_size_sum": self.collective_group_size_sum,
            "collective_kind_code_sum": self.collective_kind_code_sum,
            "side_effecting_task_count": self.side_effecting_task_count,
            "task_side_effect_count": self.task_side_effect_count,
            "task_side_effect_code_sum": self.task_side_effect_code_sum,
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
    plan_schema_version: str
    capability_level: str
    engine_schema_version: str
    fallback_mode: str
    runtime_executable: bool
    runtime_entrypoints: tuple[str, ...]
    compiler_summary: BumkcCompilerSummary
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
            "plan_schema_version": self.plan_schema_version,
            "capability_level": self.capability_level,
            "engine_schema_version": self.engine_schema_version,
            "fallback_mode": self.fallback_mode,
            "runtime_executable": self.runtime_executable,
            "runtime_entrypoints": list(self.runtime_entrypoints),
            "compiler_summary": self.compiler_summary.as_log_dict(),
            "runtime_summary": self.runtime_summary.as_log_dict(),
            "runtime_shape_symbols": [
                dataclasses.asdict(binding) for binding in self.runtime_shape_symbols
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
    model_source = _read_json(root / "source" / "model-source.json")
    tensor_islands = _read_json(root / "ir" / "hvm-tensor-islands.json")
    block_plan = _read_json(root / "ir" / "hvm-block-role-pipelines.json")
    event_tensors = _read_json(root / "ir" / "hvm-event-tensors.json")
    simulation = _read_json(root / "reports" / "simulation.json")
    runtime = _read_json(root / "runtime" / "plan.json")
    engine = _read_json(root / "engine" / "optimization-playground.json")
    runtime_smoke = _read_json(root / "generated" / "runtime-smoke.json")
    tensor_smoke = _read_json(root / "generated" / "tensor-smoke.json")

    _validate_identity(
        manifest,
        model_source,
        tensor_islands,
        block_plan,
        event_tensors,
        simulation,
        runtime,
        engine,
        runtime_smoke,
        tensor_smoke,
    )
    if manifest.get("schema_version") != REQUIRED_PLAN_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported plan schema")
    if manifest.get("capability_level") != REQUIRED_CAPABILITY_LEVEL:
        raise BumkcArtifactError("BUMKC artifact capability is unsupported")
    if engine.get("schema_version") != REQUIRED_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported engine schema")
    if engine.get("manifest_schema_version") != manifest.get("schema_version"):
        raise BumkcArtifactError("BUMKC engine manifest schema does not match manifest")
    if engine.get("manifest_capability_level") != manifest.get("capability_level"):
        raise BumkcArtifactError(
            "BUMKC engine manifest capability does not match manifest"
        )
    if engine.get("engine") != "sglang":
        raise BumkcArtifactError("BUMKC artifact is not targeted at the SGLang engine")
    if engine.get("engine_profile") != "optimization_playground":
        raise BumkcArtifactError("BUMKC artifact is not for optimization-playground")
    if tuple(engine.get("cli_flags", ())) != REQUIRED_CLI_FLAGS:
        raise BumkcArtifactError("BUMKC artifact serving CLI flags are unsupported")
    artifact_paths = _read_object(engine, "artifact_paths")
    _validate_artifact_paths(root, artifact_paths)
    if artifact_paths.get("artifact_digests") != ARTIFACT_DIGEST_PATH.as_posix():
        raise BumkcArtifactError("BUMKC artifact digest path is not canonical")
    artifact_digest_count = _validate_artifact_digests(root, manifest)
    if engine.get("target_arch") != manifest.get("target_arch"):
        raise BumkcArtifactError(
            "BUMKC engine target architecture does not match manifest"
        )
    if engine.get("fallback_mode") != "checked":
        raise BumkcArtifactError("BUMKC artifact must use checked fallback mode")
    if not engine.get("preserve_custom_optimizations"):
        raise BumkcArtifactError(
            "BUMKC artifact does not preserve custom optimizations"
        )
    if engine.get("required_validation_model") != REQUIRED_VALIDATION_MODEL:
        raise BumkcArtifactError(
            "BUMKC artifact validation model is not the REAP target"
        )
    _validate_source_artifact(model_source, manifest, engine, tensor_islands)
    if engine.get("runtime_executable") != runtime.get("executable"):
        raise BumkcArtifactError("BUMKC engine and runtime executable flags disagree")
    if runtime.get("runtime_abi_version") != REQUIRED_RUNTIME_ABI_VERSION:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported runtime ABI")
    if require_executable and not runtime.get("executable"):
        raise BumkcArtifactError("BUMKC artifact is not executable")

    compiler_summary = _load_compiler_summary(
        engine,
        tensor_islands,
        block_plan,
        event_tensors,
        simulation,
    )
    runtime_summary = _load_runtime_summary(engine, runtime)
    _validate_runtime_smoke_plan(
        runtime_smoke,
        runtime,
        runtime_summary,
        compiler_summary,
    )
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
        plan_schema_version=manifest["schema_version"],
        capability_level=manifest["capability_level"],
        engine_schema_version=engine["schema_version"],
        fallback_mode=engine["fallback_mode"],
        runtime_executable=bool(runtime["executable"]),
        runtime_entrypoints=entrypoints,
        compiler_summary=compiler_summary,
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


def _validate_artifact_paths(root: Path, artifact_paths: dict[str, Any]) -> None:
    expected = {
        "artifact_digests": "reports/artifact-digests.json",
        "cpu_reference_report": "reports/cpu-reference.json",
        "cuda_smoke_plan": "generated/runtime-smoke.json",
        "cuda_smoke_source": "generated/runtime_smoke.cu",
        "cuda_tensor_smoke_plan": "generated/tensor-smoke.json",
        "cuda_tensor_smoke_source": "generated/tensor_smoke.cu",
        "hvm_block_role_pipelines": "ir/hvm-block-role-pipelines.json",
        "hvm_core_book": "ir/hvm-core-book.json",
        "hvm_core_book_source": "source/hvm-core-book.hvm",
        "hvm_event_tensors": "ir/hvm-event-tensors.json",
        "hvm_sm_task_runtime": "ir/hvm-sm-task-runtime.json",
        "hvm_tensor_islands": "ir/hvm-tensor-islands.json",
        "manifest": "manifest.json",
        "model_source": "source/model-source.json",
        "runtime_plan": "runtime/plan.json",
    }
    for key, value in expected.items():
        if artifact_paths.get(key) != value:
            raise BumkcArtifactError(f"BUMKC artifact path mismatch: {key}")
        if not (root / value).is_file():
            raise BumkcArtifactError(f"BUMKC artifact path is missing: {key}")


def _validate_source_artifact(
    model_source: dict[str, Any],
    manifest: dict[str, Any],
    engine: dict[str, Any],
    tensor_islands: dict[str, Any],
) -> None:
    if model_source.get("schema_version") != REQUIRED_SOURCE_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC model source schema is unsupported")
    if _read_object(model_source, "source") != _read_object(manifest, "source"):
        raise BumkcArtifactError("BUMKC model source provenance mismatches manifest")
    if model_source["source"].get("model") != engine.get("model"):
        raise BumkcArtifactError("BUMKC model source provenance mismatches engine")

    expected_manifest_fields = {
        "gpu_count": manifest.get("gpu_count"),
        "target_arch": manifest.get("target_arch"),
        "engine": manifest.get("engine"),
        "engine_profile": manifest.get("engine_profile"),
        "fallback_mode": manifest.get("fallback_mode"),
        "runtime_mode": manifest.get("runtime_mode"),
    }
    for key, value in expected_manifest_fields.items():
        if model_source.get(key) != value:
            raise BumkcArtifactError(f"BUMKC model source manifest mismatch: {key}")
    if model_source.get("gpu_count") != engine.get("gpu_count"):
        raise BumkcArtifactError("BUMKC model source GPU count mismatches engine")
    if model_source.get("target_arch") != engine.get("target_arch"):
        raise BumkcArtifactError(
            "BUMKC model source target architecture mismatches engine"
        )
    if model_source.get("fallback_mode") != engine.get("fallback_mode"):
        raise BumkcArtifactError("BUMKC model source fallback mode mismatches engine")
    if model_source.get("hvm_core_book_source_path") != "source/hvm-core-book.hvm":
        raise BumkcArtifactError("BUMKC model source HVM path is not canonical")
    _source_coverage_status(model_source, "hvm_capture_status")

    islands = _read_list(tensor_islands, "islands")
    fallback_bridges = _read_list(tensor_islands, "fallback_bridges")
    shape_symbols = _read_list(tensor_islands, "shape_symbols")
    island_side_effects = [
        _read_side_effect_list(island, "side_effects") for island in islands
    ]
    island_serving_state = [_read_list(island, "serving_state") for island in islands]
    coverage_statuses = [
        _source_coverage_status(island, "coverage_status") for island in islands
    ]
    expected = {
        "tensor_island_count": len(islands),
        "native_eligible_island_count": sum(
            1 for status in coverage_statuses if status == "native_eligible"
        ),
        "fallback_island_count": sum(
            1 for status in coverage_statuses if status == "fallback_only"
        ),
        "fallback_bridge_count": len(fallback_bridges),
        "side_effecting_island_count": sum(
            1 for side_effects in island_side_effects if side_effects
        ),
        "collective_island_count": sum(
            1 for island in islands if _read_optional_object(island, "communication")
        ),
        "moe_dispatch_island_count": sum(
            1 for island in islands if _read_str(island, "operator") == "moe_dispatch"
        ),
        "serving_state_island_count": sum(
            1 for serving_state in island_serving_state if serving_state
        ),
        "serving_state_dependency_count": sum(
            len(serving_state) for serving_state in island_serving_state
        ),
        "serving_state_kind_code_sum": _serving_state_kind_code_sum(
            island_serving_state
        ),
        "serving_state_symbol_count": sum(
            1
            for serving_state in island_serving_state
            for dependency in serving_state
            if _read_optional_str(dependency, "symbol") is not None
        ),
        "shape_symbol_count": len(shape_symbols),
        "shape_symbol_max_sum": sum(
            _read_int(symbol, "max") for symbol in shape_symbols
        ),
        "shape_symbol_bucket_sum": sum(
            _read_int(symbol, "bucket") for symbol in shape_symbols
        ),
    }
    for key, value in expected.items():
        if _read_int(model_source, key) != value:
            raise BumkcArtifactError(f"BUMKC model source summary mismatch: {key}")


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
    side_effect_plan = _read_object(runtime, "side_effect_plan")
    serving_state_plan = _read_object(runtime, "serving_state_plan")
    substitution_plan = _read_object(runtime, "substitution_plan")
    substitution_shape_symbols = _read_list(substitution_plan, "shape_symbols")
    substitution_serving_state = _read_list(substitution_plan, "serving_state")
    dependency_plan = _read_object(runtime, "dependency_plan")
    dependency_descriptors = _read_list(dependency_plan, "dependency_descriptors")
    dependency_descriptor_hash = _dependency_descriptor_hash(dependency_descriptors)
    if _read_int(dependency_plan, "dependency_descriptor_count") != len(
        dependency_descriptors
    ):
        raise BumkcArtifactError("BUMKC runtime dependency descriptor count mismatch")
    if (
        _read_int(dependency_plan, "dependency_descriptor_hash")
        != dependency_descriptor_hash
    ):
        raise BumkcArtifactError("BUMKC runtime dependency descriptor hash mismatch")
    dependency_tensor_count = sum(
        len(_read_any_list(descriptor, "tensors"))
        for descriptor in dependency_descriptors
    )
    if _read_int(dependency_plan, "dependency_tensor_count") != dependency_tensor_count:
        raise BumkcArtifactError("BUMKC runtime dependency tensor count mismatch")
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
        "dependency_tensor_count": dependency_tensor_count,
        "dependency_scope_code_sum": dependency_plan.get("dependency_scope_code_sum"),
        "dependency_descriptor_count": len(dependency_descriptors),
        "dependency_descriptor_hash": dependency_descriptor_hash,
        "collective_task_count": communication_plan.get("collective_task_count"),
        "collective_group_size_sum": communication_plan.get(
            "collective_group_size_sum"
        ),
        "collective_kind_code_sum": communication_plan.get("collective_kind_code_sum"),
        "side_effecting_task_count": side_effect_plan.get("side_effecting_task_count"),
        "task_side_effect_count": side_effect_plan.get("task_side_effect_count"),
        "task_side_effect_code_sum": side_effect_plan.get("task_side_effect_code_sum"),
        "serving_task_count": serving_state_plan.get("serving_task_count"),
        "serving_dependency_count": serving_state_plan.get("serving_dependency_count"),
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
        if value is None or _read_summary_int(summary, key) != value:
            raise BumkcArtifactError(f"BUMKC engine runtime summary mismatch: {key}")

    return BumkcRuntimeSummary(**{key: int(value) for key, value in expected.items()})


def _load_compiler_summary(
    engine: dict[str, Any],
    tensor_islands: dict[str, Any],
    block_plan: dict[str, Any],
    event_tensors: dict[str, Any],
    simulation: dict[str, Any],
) -> BumkcCompilerSummary:
    summary = engine.get("compiler_summary")
    if not isinstance(summary, dict):
        raise BumkcArtifactError("BUMKC engine compiler summary is missing")

    islands = _read_list(tensor_islands, "islands")
    fallback_bridges = _read_list(tensor_islands, "fallback_bridges")
    block_ops = _read_list(block_plan, "block_ops")
    events = _read_list(event_tensors, "event_tensors")
    execution_order = _read_any_list(simulation, "execution_order")
    violations = _read_list(simulation, "violations")
    island_side_effects = [
        _read_side_effect_list(island, "side_effects") for island in islands
    ]
    block_side_effects = [
        _read_side_effect_list(block, "side_effects") for block in block_ops
    ]
    event_side_effects = [
        _read_side_effect_list(event, "side_effects") for event in events
    ]
    native_island_count = 0
    fallback_island_count = 0
    for island in islands:
        coverage_status = _read_str(island, "coverage_status")
        if coverage_status in ("native_eligible", "native_compiled"):
            native_island_count += 1
        elif coverage_status == "fallback_only":
            fallback_island_count += 1
        else:
            raise BumkcArtifactError(
                f"BUMKC tensor island coverage status is unsupported: {coverage_status}"
            )

    expected = {
        "tensor_island_count": len(islands),
        "native_tensor_island_count": native_island_count,
        "fallback_tensor_island_count": fallback_island_count,
        "side_effecting_tensor_island_count": sum(
            1 for side_effects in island_side_effects if side_effects
        ),
        "tensor_island_side_effect_count": sum(
            len(side_effects) for side_effects in island_side_effects
        ),
        "tensor_island_side_effect_code_sum": _side_effect_code_sum(
            island_side_effects
        ),
        "fallback_bridge_count": len(fallback_bridges),
        "moe_dispatch_tensor_island_count": sum(
            1 for island in islands if _read_str(island, "operator") == "moe_dispatch"
        ),
        "block_op_count": len(block_ops),
        "moe_dispatch_block_op_count": sum(
            1 for block in block_ops if _read_str(block, "operator") == "moe_dispatch"
        ),
        "side_effecting_block_op_count": sum(
            1 for side_effects in block_side_effects if side_effects
        ),
        "block_side_effect_count": sum(
            len(side_effects) for side_effects in block_side_effects
        ),
        "block_side_effect_code_sum": _side_effect_code_sum(block_side_effects),
        "event_tensor_count": len(events),
        "side_effecting_event_tensor_count": sum(
            1 for side_effects in event_side_effects if side_effects
        ),
        "event_side_effect_count": sum(
            len(side_effects) for side_effects in event_side_effects
        ),
        "event_side_effect_code_sum": _side_effect_code_sum(event_side_effects),
        "moe_dispatch_event_tensor_count": sum(
            1 for event in events if _read_str(event, "operator") == "moe_dispatch"
        ),
        "event_predecessor_edge_count": sum(
            len(_read_any_list(event, "predecessor_events")) for event in events
        ),
        "event_successor_edge_count": sum(
            len(_read_any_list(event, "successor_events")) for event in events
        ),
        "event_dependency_tensor_count": _event_dependency_tensor_count(events),
        "event_notification_count": _read_int(simulation, "notification_count"),
        "event_execution_count": len(execution_order),
        "event_simulation_violation_count": len(violations),
    }
    for key, value in expected.items():
        if _read_summary_int(summary, key) != value:
            raise BumkcArtifactError(f"BUMKC engine compiler summary mismatch: {key}")

    return BumkcCompilerSummary(**{key: int(value) for key, value in expected.items()})


def _validate_runtime_smoke_plan(
    runtime_smoke: dict[str, Any],
    runtime: dict[str, Any],
    runtime_summary: BumkcRuntimeSummary,
    compiler_summary: BumkcCompilerSummary,
) -> None:
    if runtime_smoke.get("schema_version") != REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC runtime smoke schema is unsupported")
    if runtime_smoke.get("runtime_abi_version") != runtime.get("runtime_abi_version"):
        raise BumkcArtifactError("BUMKC runtime smoke ABI does not match runtime")
    if runtime_smoke.get("source_path") != "generated/runtime_smoke.cu":
        raise BumkcArtifactError("BUMKC runtime smoke source path is not canonical")
    if runtime_smoke.get("binary_name") != "runtime_smoke":
        raise BumkcArtifactError("BUMKC runtime smoke binary name is not canonical")

    execution_model = _read_object(runtime, "execution_model")
    expected = {
        "expected_task_count": runtime_summary.task_count,
        "expected_conventional_launch_count": (
            runtime_summary.conventional_launch_count
        ),
        "expected_persistent_launch_count": runtime_summary.persistent_launch_count,
        "expected_launch_reduction_per_mille": _read_int(
            execution_model, "launch_reduction_per_mille"
        ),
        "expected_jit_task_count": runtime_summary.jit_task_count,
        "expected_aot_task_count": runtime_summary.aot_task_count,
        "expected_queue_capacity": runtime_summary.queue_capacity,
        "expected_task_instance_capacity": runtime_summary.task_instance_capacity,
        "expected_dependency_edge_count": runtime_summary.task_dependency_count,
        "expected_dependency_tensor_count": runtime_summary.dependency_tensor_count,
        "expected_dependency_scope_code_sum": (
            runtime_summary.dependency_scope_code_sum
        ),
        "expected_kv_cache_binding_count": runtime_summary.kv_cache_binding_count,
        "expected_communication_task_count": runtime_summary.collective_task_count,
        "expected_communication_group_size_sum": (
            runtime_summary.collective_group_size_sum
        ),
        "expected_communication_kind_code_sum": (
            runtime_summary.collective_kind_code_sum
        ),
        "expected_side_effecting_task_count": (
            runtime_summary.side_effecting_task_count
        ),
        "expected_task_side_effect_count": runtime_summary.task_side_effect_count,
        "expected_task_side_effect_code_sum": (
            runtime_summary.task_side_effect_code_sum
        ),
        "expected_serving_task_count": runtime_summary.serving_task_count,
        "expected_serving_dependency_count": runtime_summary.serving_dependency_count,
        "expected_serving_kind_code_sum": runtime_summary.serving_kind_code_sum,
        "expected_serving_symbol_count": runtime_summary.serving_symbol_count,
        "expected_substitution_shape_symbol_count": (
            runtime_summary.substitution_shape_symbol_count
        ),
        "expected_substitution_serving_binding_count": (
            runtime_summary.substitution_serving_binding_count
        ),
        "expected_substitution_symbol_max_sum": (
            runtime_summary.substitution_symbol_max_sum
        ),
        "expected_substitution_symbol_bucket_sum": (
            runtime_summary.substitution_symbol_bucket_sum
        ),
        "expected_rank_group_size_sum": runtime_summary.task_rank_reference_count,
        "expected_rank_id_sum": runtime_summary.rank_id_sum,
        "expected_event_tensor_count": compiler_summary.event_tensor_count,
        "expected_event_predecessor_edge_count": (
            compiler_summary.event_predecessor_edge_count
        ),
        "expected_event_successor_edge_count": (
            compiler_summary.event_successor_edge_count
        ),
        "expected_event_notification_count": (
            compiler_summary.event_notification_count
        ),
        "expected_event_execution_count": compiler_summary.event_execution_count,
        "expected_event_simulation_violation_count": (
            compiler_summary.event_simulation_violation_count
        ),
    }
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke mismatch: {key}")

    event_descriptors = _read_list(runtime_smoke, "event_descriptors")
    task_descriptors = _read_list(runtime_smoke, "task_descriptors")
    if len(event_descriptors) != compiler_summary.event_tensor_count:
        raise BumkcArtifactError(
            "BUMKC runtime smoke event descriptor count mismatches"
        )
    if len(task_descriptors) != runtime_summary.task_count:
        raise BumkcArtifactError("BUMKC runtime smoke task descriptor count mismatches")
    _validate_runtime_smoke_event_descriptors(
        runtime_smoke,
        event_descriptors,
    )
    _validate_runtime_smoke_task_descriptors(
        runtime_smoke,
        task_descriptors,
    )
    _validate_runtime_smoke_contracts(
        runtime_smoke,
        event_descriptors,
        task_descriptors,
    )


def _validate_runtime_smoke_event_descriptors(
    runtime_smoke: dict[str, Any],
    event_descriptors: list[dict[str, Any]],
) -> None:
    expected = {
        "expected_event_predecessor_edge_count": 0,
        "expected_event_successor_edge_count": 0,
    }
    for ordinal, event in enumerate(event_descriptors):
        if _read_int(event, "ordinal") != ordinal:
            raise BumkcArtifactError("BUMKC runtime smoke event ordinal mismatch")
        _read_str(event, "event_id")
        expected["expected_event_predecessor_edge_count"] += _read_int(
            event, "predecessor_event_count"
        )
        expected["expected_event_successor_edge_count"] += _read_int(
            event, "successor_event_count"
        )
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke descriptor mismatch: {key}")


def _validate_runtime_smoke_task_descriptors(
    runtime_smoke: dict[str, Any],
    task_descriptors: list[dict[str, Any]],
) -> None:
    expected = {
        "expected_jit_task_count": 0,
        "expected_aot_task_count": 0,
        "expected_task_instance_capacity": 0,
        "expected_predecessor_event_count_sum": 0,
        "expected_dependency_edge_count": 0,
        "expected_dependency_tensor_count": 0,
        "expected_dependency_scope_code_sum": 0,
        "expected_launch_domain_rank_sum": 0,
        "expected_launch_domain_element_sum": 0,
        "expected_operator_code_sum": 0,
        "expected_kv_cache_binding_count": 0,
        "expected_communication_task_count": 0,
        "expected_communication_group_size_sum": 0,
        "expected_communication_kind_code_sum": 0,
        "expected_side_effecting_task_count": 0,
        "expected_task_side_effect_count": 0,
        "expected_task_side_effect_code_sum": 0,
        "expected_serving_task_count": 0,
        "expected_serving_dependency_count": 0,
        "expected_serving_kind_code_sum": 0,
        "expected_serving_symbol_count": 0,
        "expected_rank_group_size_sum": 0,
        "expected_rank_id_sum": 0,
    }
    for ordinal, task in enumerate(task_descriptors):
        if _read_int(task, "ordinal") != ordinal:
            raise BumkcArtifactError("BUMKC runtime smoke task ordinal mismatch")
        _read_str(task, "task_id")
        _read_str(task, "source_event_tensor")
        policy = _read_str(task, "scheduling_policy")
        if policy not in _SCHEDULING_POLICY_CODES:
            raise BumkcArtifactError(
                f"BUMKC runtime smoke scheduling policy is unsupported: {policy}"
            )
        expected["expected_jit_task_count"] += int(policy == "dynamic_jit")
        expected["expected_aot_task_count"] += int(policy == "static_aot")

        operator = _read_str(task, "operator")
        if operator not in _OPERATOR_CODES:
            raise BumkcArtifactError(
                f"BUMKC runtime smoke operator is unsupported: {operator}"
            )
        expected["expected_operator_code_sum"] += _OPERATOR_CODES[operator]

        expected["expected_predecessor_event_count_sum"] += _read_int(
            task, "predecessor_event_count"
        )
        expected["expected_dependency_edge_count"] += _read_int(
            task, "dependency_edge_count"
        )
        expected["expected_dependency_tensor_count"] += _read_int(
            task, "dependency_tensor_count"
        )
        expected["expected_dependency_scope_code_sum"] += _read_int(
            task, "dependency_scope_code_sum"
        )
        expected["expected_launch_domain_rank_sum"] += _read_int(
            task, "launch_domain_rank"
        )
        launch_domain_elements = _read_int(task, "launch_domain_elements")
        expected["expected_launch_domain_element_sum"] += launch_domain_elements
        expected["expected_task_instance_capacity"] += launch_domain_elements
        expected["expected_kv_cache_binding_count"] += _read_int(
            task, "kv_cache_binding_count"
        )
        communication_kind_code = _read_int(task, "communication_kind_code")
        expected["expected_communication_task_count"] += int(
            communication_kind_code != 0
        )
        expected["expected_communication_group_size_sum"] += _read_int(
            task, "communication_group_size"
        )
        expected["expected_communication_kind_code_sum"] += communication_kind_code
        side_effect_count = _read_int(task, "side_effect_count")
        expected["expected_side_effecting_task_count"] += int(side_effect_count != 0)
        expected["expected_task_side_effect_count"] += side_effect_count
        expected["expected_task_side_effect_code_sum"] += _read_int(
            task, "side_effect_code_sum"
        )
        serving_dependency_count = _read_int(task, "serving_dependency_count")
        expected["expected_serving_task_count"] += int(serving_dependency_count != 0)
        expected["expected_serving_dependency_count"] += serving_dependency_count
        expected["expected_serving_kind_code_sum"] += _read_int(
            task, "serving_kind_code_sum"
        )
        expected["expected_serving_symbol_count"] += _read_int(
            task, "serving_symbol_count"
        )
        expected["expected_rank_group_size_sum"] += _read_int(task, "rank_group_size")
        expected["expected_rank_id_sum"] += _read_int(task, "rank_id_sum")
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke descriptor mismatch: {key}")


def _validate_runtime_smoke_contracts(
    runtime_smoke: dict[str, Any],
    event_descriptors: list[dict[str, Any]],
    task_descriptors: list[dict[str, Any]],
) -> None:
    descriptor_hash = _runtime_smoke_descriptor_contract_hash(
        event_descriptors,
        task_descriptors,
    )
    expected = {
        "expected_schema_hash": _contract_hash_str(
            REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION
        ),
        "expected_runtime_abi_hash": _contract_hash_str(
            _read_str(runtime_smoke, "runtime_abi_version")
        ),
        "expected_plan_id_hash": _contract_hash_str(
            _read_str(runtime_smoke, "plan_id")
        ),
        "expected_program_id_hash": _contract_hash_str(
            _read_str(runtime_smoke, "program_id")
        ),
        "expected_descriptor_contract_hash": descriptor_hash,
        "expected_source_contract_hash": _runtime_smoke_source_contract_hash(
            runtime_smoke,
            descriptor_hash,
        ),
    }
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke contract mismatch: {key}")


def _runtime_smoke_descriptor_contract_hash(
    event_descriptors: list[dict[str, Any]],
    task_descriptors: list[dict[str, Any]],
) -> int:
    hash_value = _mix_contract_str(_CONTRACT_HASH_OFFSET, _DESCRIPTOR_CONTRACT_DOMAIN)
    hash_value = _mix_contract_u64(hash_value, len(event_descriptors))
    hash_value = _mix_contract_u64(hash_value, len(task_descriptors))
    for ordinal, event in enumerate(event_descriptors):
        hash_value = _mix_contract_u64(hash_value, ordinal)
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event, "predecessor_event_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event, "successor_event_count"),
        )
    for ordinal, task in enumerate(task_descriptors):
        operator = _read_str(task, "operator")
        policy = _read_str(task, "scheduling_policy")
        hash_value = _mix_contract_u64(hash_value, ordinal)
        hash_value = _mix_contract_u64(hash_value, _OPERATOR_CODES[operator])
        hash_value = _mix_contract_u64(hash_value, _SCHEDULING_POLICY_CODES[policy])
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "predecessor_event_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "dependency_edge_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "dependency_tensor_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "dependency_scope_code_sum"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "launch_domain_rank"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "launch_domain_elements"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "kv_cache_binding_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "communication_kind_code"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "communication_group_size"),
        )
        hash_value = _mix_contract_u64(hash_value, _read_int(task, "side_effect_count"))
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "side_effect_code_sum"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "serving_dependency_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "serving_kind_code_sum"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(task, "serving_symbol_count"),
        )
        hash_value = _mix_contract_u64(hash_value, _read_int(task, "rank_group_size"))
        hash_value = _mix_contract_u64(hash_value, _read_int(task, "rank_id_sum"))
    return hash_value


def _runtime_smoke_source_contract_hash(
    runtime_smoke: dict[str, Any],
    descriptor_hash: int,
) -> int:
    hash_value = _mix_contract_str(_CONTRACT_HASH_OFFSET, _SOURCE_CONTRACT_DOMAIN)
    hash_value = _mix_contract_u64(
        hash_value,
        _read_int(runtime_smoke, "expected_schema_hash"),
    )
    hash_value = _mix_contract_u64(
        hash_value,
        _read_int(runtime_smoke, "expected_runtime_abi_hash"),
    )
    hash_value = _mix_contract_u64(
        hash_value,
        _read_int(runtime_smoke, "expected_plan_id_hash"),
    )
    hash_value = _mix_contract_u64(
        hash_value,
        _read_int(runtime_smoke, "expected_program_id_hash"),
    )
    hash_value = _mix_contract_u64(hash_value, descriptor_hash)
    for key in _SOURCE_CONTRACT_KEYS:
        hash_value = _mix_contract_u64(hash_value, _read_int(runtime_smoke, key))
    return hash_value


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


def _read_optional_object(parent: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BumkcArtifactError(f"BUMKC runtime {key} object is invalid")
    return value


def _read_list(parent: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise BumkcArtifactError(f"BUMKC runtime {key} list is missing")
    if not all(isinstance(entry, dict) for entry in value):
        raise BumkcArtifactError(f"BUMKC runtime {key} list has non-object entries")
    return value


def _read_any_list(parent: dict[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise BumkcArtifactError(f"BUMKC runtime {key} list is missing")
    return value


def _read_side_effect_list(parent: dict[str, Any], key: str) -> list[str]:
    values = _read_any_list(parent, key)
    for value in values:
        if value not in _SIDE_EFFECT_CODES:
            raise BumkcArtifactError(
                f"BUMKC side-effect marker is unsupported: {value}"
            )
    return values


def _side_effect_code_sum(side_effect_lists: Iterable[list[str]]) -> int:
    return sum(
        _SIDE_EFFECT_CODES[side_effect]
        for side_effects in side_effect_lists
        for side_effect in side_effects
    )


def _source_coverage_status(parent: dict[str, Any], key: str) -> str:
    status = _read_str(parent, key)
    if status not in ("fallback_only", "native_eligible", "native_compiled"):
        raise BumkcArtifactError(
            f"BUMKC model source coverage status is unsupported: {status}"
        )
    return status


def _serving_state_kind_code_sum(
    serving_state_lists: Iterable[list[dict[str, Any]]],
) -> int:
    code_sum = 0
    for serving_state in serving_state_lists:
        for dependency in serving_state:
            kind = _read_str(dependency, "kind")
            if kind not in _SERVING_STATE_KIND_CODES:
                raise BumkcArtifactError(
                    f"BUMKC serving-state kind is unsupported: {kind}"
                )
            code_sum += _SERVING_STATE_KIND_CODES[kind]
    return code_sum


def _event_dependency_tensor_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        edges = _read_list(event, "predecessor_edges")
        if len(edges) != len(_read_any_list(event, "predecessor_events")):
            raise BumkcArtifactError("BUMKC Event Tensor edge count mismatch")
        for edge in edges:
            tensors = _read_any_list(edge, "tensors")
            for tensor in tensors:
                if not isinstance(tensor, str) or not tensor:
                    raise BumkcArtifactError(
                        "BUMKC Event Tensor dependency tensor is malformed"
                    )
            count += len(tensors)
    return count


def _read_summary_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not _is_strict_int(value):
        raise BumkcArtifactError(f"BUMKC engine summary {key} integer is missing")
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
