from __future__ import annotations

from collections.abc import Iterable, Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

ARTIFACT_DIGEST_PATH = Path("reports/artifact-digests.json")
VERIFY_REPORT_PATH = Path("reports/verify.json")
REQUIRED_DIGEST_SCHEMA_VERSION = "bumkc.artifact_digests.v1"
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
LEGACY_SCHEMA_VERSION = "bumkc.optimization_playground.v20"
PREVIOUS_SCHEMA_VERSION = "bumkc.optimization_playground.v21"
SCALE_UP_SCHEMA_VERSION = "bumkc.optimization_playground.v22"
SERVING_HINTS_SCHEMA_VERSION = "bumkc.optimization_playground.v23"
LAUNCH_SCHEMA_VERSION = "bumkc.optimization_playground.v24"
REQUIRED_SCHEMA_VERSION = "bumkc.optimization_playground.v25"
SUPPORTED_SCHEMA_VERSIONS = (
    LEGACY_SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    SCALE_UP_SCHEMA_VERSION,
    SERVING_HINTS_SCHEMA_VERSION,
    LAUNCH_SCHEMA_VERSION,
    REQUIRED_SCHEMA_VERSION,
)
REQUIRED_SOURCE_SCHEMA_VERSION = "bumkc.source.v11"
REQUIRED_RUNTIME_ABI_VERSION = "bumkc.runtime.v1"
REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION = "bumkc.cuda_smoke.v14"
SUPPORTED_RUNTIME_MODES = ("debug", "trace", "profile", "production")
RUNTIME_SMOKE_BENCHMARK_ITERATIONS = 8
RUNTIME_SMOKE_BENCHMARK_LAUNCH_CAP = 128
_CONTRACT_HASH_OFFSET = 0xCBF29CE484222325
_CONTRACT_HASH_PRIME = 0x00000100000001B3
_U64_MASK = (1 << 64) - 1
_DESCRIPTOR_CONTRACT_DOMAIN = "bumkc.cuda_smoke.descriptors.v2"
_SOURCE_CONTRACT_DOMAIN = "bumkc.cuda_smoke.source.v1"
_SOURCE_CONTRACT_KEYS = (
    "expected_task_count",
    "expected_conventional_launch_count",
    "expected_persistent_launch_count",
    "expected_launch_reduction_per_mille",
    "benchmark_enabled",
    "benchmark_iterations",
    "benchmark_conventional_launch_count",
    "benchmark_persistent_launch_count",
    "benchmark_total_conventional_launches",
    "benchmark_total_persistent_launches",
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
    "expected_event_dispatch_range_count",
    "expected_event_dispatch_compact_range_count",
    "expected_event_dispatch_indexed_range_count",
    "expected_event_dispatch_task_count_sum",
    "expected_event_dispatch_trigger_count_sum",
    "expected_event_dispatch_trigger_match_count",
    "expected_event_dispatch_owner_scheduler_sum",
    "expected_event_dispatch_max_task_range_len",
    "expected_event_dispatch_max_trigger_count",
    "expected_event_dispatch_range_kind_code_sum",
    "expected_event_dispatch_invalid_range_count",
    "expected_diagnostic_heartbeat_slot_count",
    "expected_diagnostic_queue_snapshot_slot_count",
    "expected_diagnostic_event_counter_snapshot_count",
    "expected_diagnostic_last_completed_task_slot_count",
    "expected_diagnostic_blocked_event_slot_count",
    "expected_watchdog_poll_interval_us",
    "expected_watchdog_timeout_us",
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
_DEPENDENCY_WAIT_EXPRESSIONS = {
    "tile_overlap": "same_logical_tile_ready",
    "whole_producer": "all_producer_tiles_complete",
}
_SERVING_STATE_KIND_CODES = {
    "batch": 1,
    "sequence": 2,
    "decode_step": 3,
    "token_ids": 4,
    "kv_cache_pages": 5,
}
_QUANTIZATION_FORMAT_CODES = {
    "nv_fp4": 1,
    "fp8_e4m3": 2,
    "fp8_e5m2": 3,
    "int4": 4,
}
_QUANTIZATION_SCALE_LAYOUT_CODES = {
    "tensor": 1,
    "tensor_group": 2,
    "channel": 3,
}
_DTYPE_CODES = {
    "bool": 1,
    "i32": 2,
    "i64": 3,
    "f16": 4,
    "bf16": 5,
    "f32": 6,
    "fp8_e4m3": 7,
    "fp8_e5m2": 8,
    "nv_fp4": 9,
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


def _stable_name_hash(value: str | None) -> int:
    if value is None:
        return 0
    hash_value = _DEPENDENCY_HASH_OFFSET
    for byte in value.encode("utf-8"):
        hash_value = ((hash_value ^ byte) * _DEPENDENCY_HASH_PRIME) & _U64_MASK
    return hash_value


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
    diagnostic_heartbeat_slot_count: int
    diagnostic_queue_snapshot_slot_count: int
    diagnostic_event_counter_snapshot_count: int
    diagnostic_last_completed_task_slot_count: int
    diagnostic_blocked_event_slot_count: int
    watchdog_poll_interval_us: int
    watchdog_timeout_us: int

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
            "diagnostic_heartbeat_slot_count": (self.diagnostic_heartbeat_slot_count),
            "diagnostic_queue_snapshot_slot_count": (
                self.diagnostic_queue_snapshot_slot_count
            ),
            "diagnostic_event_counter_snapshot_count": (
                self.diagnostic_event_counter_snapshot_count
            ),
            "diagnostic_last_completed_task_slot_count": (
                self.diagnostic_last_completed_task_slot_count
            ),
            "diagnostic_blocked_event_slot_count": (
                self.diagnostic_blocked_event_slot_count
            ),
            "watchdog_poll_interval_us": self.watchdog_poll_interval_us,
            "watchdog_timeout_us": self.watchdog_timeout_us,
        }


@dataclasses.dataclass(frozen=True)
class BumkcScaleUpSummary:
    compile_gpu_count: int
    runtime_target_gpu_count: int
    runtime_rank_count: int
    target_arch: str | None
    target_arch_code: int
    worker_count: int
    scheduler_count: int
    task_rank_group_count: int
    task_rank_reference_count: int
    rank_id_sum: int
    collective_task_count: int
    collective_group_size_sum: int
    collective_kind_code_sum: int

    def as_log_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BumkcLaunchSummary:
    shape_symbol_count: int
    shape_symbol_min_sum: int
    shape_symbol_max_sum: int
    shape_symbol_bucket_sum: int
    default_shape_value_sum: int
    default_bucketed_shape_value_sum: int
    serving_binding_count: int
    required_serving_binding_count: int
    optional_serving_binding_count: int
    serving_kind_code_sum: int
    serving_symbol_count: int

    def as_log_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BumkcQuantizationSummary:
    scheme: str | None
    scheme_hash: int
    weight_format: str | None
    weight_format_code: int
    weight_bits: int | None
    weight_scale_layout: str | None
    weight_scale_layout_code: int
    weight_group_size: int | None
    weight_scale_dtype: str | None
    weight_scale_dtype_code: int
    weight_symmetric: bool | None
    weight_symmetric_code: int
    weight_zero_point: bool
    activation_bits: int | None
    kv_bits: int | None
    kv_format: str | None
    kv_format_code: int
    indexer_k_bits: int | None
    indexer_k_format: str | None
    indexer_k_format_code: int
    gated_norm: bool
    spinquant: bool
    ignored_module_count: int

    def as_log_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BumkcServingHints:
    quantization: str | None
    moe_runner_backend: str | None

    def as_log_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
    source_schema_version: str
    model_source_frontend: str
    hvm_capture_status: str
    engine_schema_version: str
    fallback_mode: str
    runtime_mode: str
    runtime_executable: bool
    runtime_entrypoints: tuple[str, ...]
    compiler_summary: BumkcCompilerSummary
    runtime_summary: BumkcRuntimeSummary
    scale_up_summary: BumkcScaleUpSummary
    launch_summary: BumkcLaunchSummary
    quantization_summary: BumkcQuantizationSummary
    serving_hints: BumkcServingHints
    runtime_shape_symbols: tuple[BumkcRuntimeShapeSymbolBinding, ...]
    runtime_serving_state: tuple[BumkcRuntimeServingStateBinding, ...]
    task_count: int
    tensor_smoke_enabled: bool
    artifact_digest_count: int
    required_validation_model: str

    def validate_scale_up_domain(self, *, gpu_count: int) -> None:
        if not _is_strict_int(gpu_count) or gpu_count < 1:
            raise BumkcArtifactError("BUMKC serving GPU count is invalid")
        if gpu_count != self.gpu_count:
            raise BumkcArtifactError(
                "BUMKC artifact GPU count does not match serving domain: "
                f"artifact={self.gpu_count}, serving={gpu_count}"
            )
        if gpu_count != self.runtime_summary.rank_count:
            raise BumkcArtifactError(
                "BUMKC runtime rank count does not match serving domain: "
                f"rank_count={self.runtime_summary.rank_count}, serving={gpu_count}"
            )

    def validate_target_architecture(self, *, target_arch: str | None) -> None:
        if self.target_arch != target_arch:
            raise BumkcArtifactError(
                "BUMKC artifact target architecture does not match serving domain: "
                f"artifact={self.target_arch}, serving={target_arch}"
            )

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
            "source_schema_version": self.source_schema_version,
            "model_source_frontend": self.model_source_frontend,
            "hvm_capture_status": self.hvm_capture_status,
            "engine_schema_version": self.engine_schema_version,
            "fallback_mode": self.fallback_mode,
            "runtime_mode": self.runtime_mode,
            "runtime_executable": self.runtime_executable,
            "runtime_entrypoints": list(self.runtime_entrypoints),
            "compiler_summary": self.compiler_summary.as_log_dict(),
            "runtime_summary": self.runtime_summary.as_log_dict(),
            "scale_up_summary": self.scale_up_summary.as_log_dict(),
            "launch_summary": self.launch_summary.as_log_dict(),
            "quantization_summary": self.quantization_summary.as_log_dict(),
            "serving_hints": self.serving_hints.as_log_dict(),
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
    hvm_core_book = _read_json(root / "ir" / "hvm-core-book.json")
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
        hvm_core_book,
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
    if engine.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported engine schema")
    if engine.get("manifest_schema_version") != manifest.get("schema_version"):
        raise BumkcArtifactError("BUMKC engine manifest schema does not match manifest")
    if engine.get("manifest_capability_level") != manifest.get("capability_level"):
        raise BumkcArtifactError(
            "BUMKC engine manifest capability does not match manifest"
        )
    if engine.get("source_schema_version") != REQUIRED_SOURCE_SCHEMA_VERSION:
        raise BumkcArtifactError("BUMKC engine source schema is unsupported")
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
    hvm_core_source = _read_text(root / "source" / "hvm-core-book.hvm")
    _validate_hvm_core_book(
        hvm_core_book,
        hvm_core_source,
        model_source,
        manifest,
        tensor_islands,
        engine,
    )
    if engine.get("target_arch") != manifest.get("target_arch"):
        raise BumkcArtifactError(
            "BUMKC engine target architecture does not match manifest"
        )
    if engine.get("fallback_mode") != "checked":
        raise BumkcArtifactError("BUMKC artifact must use checked fallback mode")
    runtime_mode = _read_str(manifest, "runtime_mode")
    if runtime_mode not in SUPPORTED_RUNTIME_MODES:
        raise BumkcArtifactError("BUMKC artifact runtime mode is unsupported")
    if engine.get("schema_version") == REQUIRED_SCHEMA_VERSION:
        if engine.get("runtime_mode") != runtime_mode:
            raise BumkcArtifactError(
                "BUMKC engine runtime mode does not match manifest"
            )
    elif "runtime_mode" in engine and engine.get("runtime_mode") != runtime_mode:
        raise BumkcArtifactError("BUMKC engine runtime mode does not match manifest")
    if not engine.get("preserve_custom_optimizations"):
        raise BumkcArtifactError(
            "BUMKC artifact does not preserve custom optimizations"
        )
    if engine.get("required_validation_model") != REQUIRED_VALIDATION_MODEL:
        raise BumkcArtifactError(
            "BUMKC artifact validation model is not the REAP target"
        )
    _validate_source_artifact(
        model_source, manifest, engine, hvm_core_book, tensor_islands
    )
    quantization_summary = _load_quantization_summary(engine, tensor_islands)
    if engine.get("runtime_executable") != runtime.get("executable"):
        raise BumkcArtifactError("BUMKC engine and runtime executable flags disagree")
    if runtime.get("runtime_abi_version") != REQUIRED_RUNTIME_ABI_VERSION:
        raise BumkcArtifactError("BUMKC artifact uses an unsupported runtime ABI")
    if runtime_mode == "production" and not runtime.get("executable"):
        raise BumkcArtifactError(
            "BUMKC production runtime mode requires executable artifact"
        )
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
    scale_up_summary = _load_scale_up_summary(engine, runtime)
    launch_summary = _load_launch_summary(engine, runtime)
    _validate_runtime_smoke_plan(
        runtime_smoke,
        runtime,
        runtime_summary,
        compiler_summary,
    )
    runtime_shape_symbols = _load_shape_symbol_bindings(runtime)
    runtime_serving_state = _load_serving_state_bindings(
        runtime,
        runtime_shape_symbols,
    )
    entrypoints = tuple(
        entrypoint.get("name", "") for entrypoint in runtime.get("entrypoints", [])
    )
    model_source_record = _read_object(model_source, "source")
    return BumkcArtifactSummary(
        root=root,
        plan_id=engine["plan_id"],
        program_id=engine["program_id"],
        model=engine["model"],
        gpu_count=int(engine["gpu_count"]),
        target_arch=engine.get("target_arch"),
        plan_schema_version=manifest["schema_version"],
        capability_level=manifest["capability_level"],
        source_schema_version=model_source["schema_version"],
        model_source_frontend=_read_str(model_source_record, "frontend"),
        hvm_capture_status=_read_str(model_source, "hvm_capture_status"),
        engine_schema_version=engine["schema_version"],
        fallback_mode=engine["fallback_mode"],
        runtime_mode=runtime_mode,
        runtime_executable=bool(runtime["executable"]),
        runtime_entrypoints=entrypoints,
        compiler_summary=compiler_summary,
        runtime_summary=runtime_summary,
        scale_up_summary=scale_up_summary,
        launch_summary=launch_summary,
        quantization_summary=quantization_summary,
        serving_hints=_load_serving_hints(engine, quantization_summary),
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


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BumkcArtifactError(f"missing BUMKC artifact file: {path}") from exc
    except UnicodeDecodeError as exc:
        raise BumkcArtifactError(f"invalid BUMKC artifact text: {path}") from exc


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
        if relative_path == VERIFY_REPORT_PATH.as_posix():
            raise BumkcArtifactError(
                "BUMKC artifact digest cannot contain the writer verify report"
            )
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


def _hvm_core_book_counts(hvm_core_book: dict[str, Any]) -> dict[str, int]:
    region_count = 0
    node_count = 0
    model_entry_node_count = 0
    tensor_island_node_count = 0
    fallback_boundary_node_count = 0
    for region in _read_list(hvm_core_book, "regions"):
        region_count += 1
        for node in _read_list(region, "nodes"):
            node_count += 1
            kind = _read_object(node, "kind")
            if len(kind) != 1:
                raise BumkcArtifactError("BUMKC HVM Core node kind is malformed")
            tag = next(iter(kind))
            if tag == "model_entry":
                model_entry_node_count += 1
            elif tag == "tensor_island":
                tensor_island_node_count += 1
            elif tag == "fallback_boundary":
                fallback_boundary_node_count += 1
            else:
                raise BumkcArtifactError(
                    f"BUMKC HVM Core node kind is unsupported: {tag}"
                )
    return {
        "region_count": region_count,
        "node_count": node_count,
        "model_entry_node_count": model_entry_node_count,
        "tensor_island_node_count": tensor_island_node_count,
        "fallback_boundary_node_count": fallback_boundary_node_count,
    }


def _validate_source_artifact(
    model_source: dict[str, Any],
    manifest: dict[str, Any],
    engine: dict[str, Any],
    hvm_core_book: dict[str, Any],
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
    quantization = _read_object(tensor_islands, "quantization")
    island_side_effects = [
        _read_side_effect_list(island, "side_effects") for island in islands
    ]
    island_serving_state = [_read_list(island, "serving_state") for island in islands]
    coverage_statuses = [
        _source_coverage_status(island, "coverage_status") for island in islands
    ]
    hvm_counts = _hvm_core_book_counts(hvm_core_book)
    expected = {
        "hvm_region_count": hvm_counts["region_count"],
        "hvm_node_count": hvm_counts["node_count"],
        "hvm_model_entry_node_count": hvm_counts["model_entry_node_count"],
        "hvm_tensor_island_node_count": hvm_counts["tensor_island_node_count"],
        "hvm_fallback_boundary_node_count": hvm_counts["fallback_boundary_node_count"],
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
        "quantization_scheme_hash": _stable_name_hash(
            _read_optional_str(quantization, "scheme")
        ),
        "quantization_weight_format_code": _quantization_format_code(
            quantization, "weight_format"
        ),
        "quantization_weight_bits": _read_optional_int(quantization, "weight_bits"),
        "quantization_weight_scale_layout_code": _quantization_scale_layout_code(
            quantization, "weight_scale_layout"
        ),
        "quantization_weight_group_size": _read_optional_int(
            quantization, "weight_group_size"
        ),
        "quantization_weight_scale_dtype_code": _dtype_code(
            quantization, "weight_scale_dtype"
        ),
        "quantization_weight_symmetric_code": _optional_bool_code(
            quantization, "weight_symmetric"
        ),
        "quantization_activation_bits": _read_optional_int(
            quantization, "activation_bits"
        ),
        "quantization_kv_bits": _read_optional_int(quantization, "kv_bits"),
        "quantization_kv_format_code": _quantization_format_code(
            quantization, "kv_format"
        ),
        "quantization_indexer_k_bits": _read_optional_int(
            quantization, "indexer_k_bits"
        ),
        "quantization_indexer_k_format_code": _quantization_format_code(
            quantization, "indexer_k_format"
        ),
        "quantization_ignored_module_count": _read_int(
            quantization, "ignored_module_count"
        ),
    }
    for key, value in expected.items():
        if _read_int(model_source, key) != value:
            raise BumkcArtifactError(f"BUMKC model source summary mismatch: {key}")
    for key in ("gated_norm_enabled", "spinquant_enabled"):
        if _read_bool(model_source, key) != _read_bool(
            quantization, key.removesuffix("_enabled")
        ):
            raise BumkcArtifactError(f"BUMKC model source summary mismatch: {key}")
    if _read_bool(model_source, "quantization_weight_zero_point_enabled") != _read_bool(
        quantization, "weight_zero_point"
    ):
        raise BumkcArtifactError(
            "BUMKC model source summary mismatch: "
            "quantization_weight_zero_point_enabled"
        )


def _load_quantization_summary(
    engine: dict[str, Any],
    tensor_islands: dict[str, Any],
) -> BumkcQuantizationSummary:
    expected = _expected_quantization_summary(
        _read_object(tensor_islands, "quantization")
    )
    summary_record = _read_optional_object(engine, "quantization_summary")
    if engine.get("schema_version") == REQUIRED_SCHEMA_VERSION:
        if summary_record is None:
            raise BumkcArtifactError("BUMKC engine quantization summary is missing")
        summary = _read_quantization_summary(summary_record)
        _validate_quantization_summary(summary, expected)
        return summary
    if summary_record is not None:
        summary = _read_quantization_summary(summary_record)
        _validate_quantization_summary(summary, expected)
    return expected


def _validate_quantization_summary(
    summary: BumkcQuantizationSummary,
    expected: BumkcQuantizationSummary,
) -> None:
    for field in dataclasses.fields(BumkcQuantizationSummary):
        name = field.name
        if getattr(summary, name) != getattr(expected, name):
            raise BumkcArtifactError(
                f"BUMKC engine quantization summary mismatch: {name}"
            )


def _expected_quantization_summary(
    quantization: dict[str, Any],
) -> BumkcQuantizationSummary:
    return BumkcQuantizationSummary(
        scheme=_read_optional_str(quantization, "scheme"),
        scheme_hash=_stable_name_hash(_read_optional_str(quantization, "scheme")),
        weight_format=_read_optional_str(quantization, "weight_format"),
        weight_format_code=_quantization_format_code(quantization, "weight_format"),
        weight_bits=_read_nullable_int(quantization, "weight_bits"),
        weight_scale_layout=_read_optional_str(quantization, "weight_scale_layout"),
        weight_scale_layout_code=_quantization_scale_layout_code(
            quantization, "weight_scale_layout"
        ),
        weight_group_size=_read_nullable_int(quantization, "weight_group_size"),
        weight_scale_dtype=_read_optional_str(quantization, "weight_scale_dtype"),
        weight_scale_dtype_code=_dtype_code(quantization, "weight_scale_dtype"),
        weight_symmetric=_read_nullable_bool(quantization, "weight_symmetric"),
        weight_symmetric_code=_optional_bool_code(quantization, "weight_symmetric"),
        weight_zero_point=_read_bool(quantization, "weight_zero_point"),
        activation_bits=_read_nullable_int(quantization, "activation_bits"),
        kv_bits=_read_nullable_int(quantization, "kv_bits"),
        kv_format=_read_optional_str(quantization, "kv_format"),
        kv_format_code=_quantization_format_code(quantization, "kv_format"),
        indexer_k_bits=_read_nullable_int(quantization, "indexer_k_bits"),
        indexer_k_format=_read_optional_str(quantization, "indexer_k_format"),
        indexer_k_format_code=_quantization_format_code(
            quantization, "indexer_k_format"
        ),
        gated_norm=_read_bool(quantization, "gated_norm"),
        spinquant=_read_bool(quantization, "spinquant"),
        ignored_module_count=_read_int(quantization, "ignored_module_count"),
    )


def _read_quantization_summary(
    summary: dict[str, Any],
) -> BumkcQuantizationSummary:
    return BumkcQuantizationSummary(
        scheme=_read_optional_str(summary, "scheme"),
        scheme_hash=_read_int(summary, "scheme_hash"),
        weight_format=_read_optional_str(summary, "weight_format"),
        weight_format_code=_read_int(summary, "weight_format_code"),
        weight_bits=_read_nullable_int(summary, "weight_bits"),
        weight_scale_layout=_read_optional_str(summary, "weight_scale_layout"),
        weight_scale_layout_code=_read_int(summary, "weight_scale_layout_code"),
        weight_group_size=_read_nullable_int(summary, "weight_group_size"),
        weight_scale_dtype=_read_optional_str(summary, "weight_scale_dtype"),
        weight_scale_dtype_code=_read_int(summary, "weight_scale_dtype_code"),
        weight_symmetric=_read_nullable_bool(summary, "weight_symmetric"),
        weight_symmetric_code=_read_int(summary, "weight_symmetric_code"),
        weight_zero_point=_read_bool(summary, "weight_zero_point"),
        activation_bits=_read_nullable_int(summary, "activation_bits"),
        kv_bits=_read_nullable_int(summary, "kv_bits"),
        kv_format=_read_optional_str(summary, "kv_format"),
        kv_format_code=_read_int(summary, "kv_format_code"),
        indexer_k_bits=_read_nullable_int(summary, "indexer_k_bits"),
        indexer_k_format=_read_optional_str(summary, "indexer_k_format"),
        indexer_k_format_code=_read_int(summary, "indexer_k_format_code"),
        gated_norm=_read_bool(summary, "gated_norm"),
        spinquant=_read_bool(summary, "spinquant"),
        ignored_module_count=_read_int(summary, "ignored_module_count"),
    )


def _build_serving_hints(
    quantization: BumkcQuantizationSummary,
) -> BumkcServingHints:
    quantization_method = None
    if quantization.weight_format == "nv_fp4":
        quantization_method = "modelopt_fp4"
    elif quantization.weight_format in ("fp8_e4m3", "fp8_e5m2"):
        quantization_method = "fp8"

    moe_runner_backend = None
    if quantization_method in ("fp8", "modelopt_fp4"):
        moe_runner_backend = "flashinfer_trtllm"

    return BumkcServingHints(
        quantization=quantization_method,
        moe_runner_backend=moe_runner_backend,
    )


def _load_serving_hints(
    engine: dict[str, Any],
    quantization: BumkcQuantizationSummary,
) -> BumkcServingHints:
    expected = _build_serving_hints(quantization)
    summary_record = _read_optional_object(engine, "serving_hints")
    if engine.get("schema_version") == REQUIRED_SCHEMA_VERSION:
        if summary_record is None:
            raise BumkcArtifactError("BUMKC engine serving hints are missing")
        summary = _read_serving_hints(summary_record)
        _validate_serving_hints(summary, expected)
        return summary
    if summary_record is not None:
        summary = _read_serving_hints(summary_record)
        _validate_serving_hints(summary, expected)
    return expected


def _read_serving_hints(
    summary: dict[str, Any],
) -> BumkcServingHints:
    return BumkcServingHints(
        quantization=_read_optional_str(summary, "quantization"),
        moe_runner_backend=_read_optional_str(summary, "moe_runner_backend"),
    )


def _validate_serving_hints(
    summary: BumkcServingHints,
    expected: BumkcServingHints,
) -> None:
    for field in dataclasses.fields(BumkcServingHints):
        name = field.name
        if getattr(summary, name) != getattr(expected, name):
            raise BumkcArtifactError(f"BUMKC engine serving hint mismatch: {name}")


def _validate_hvm_core_book(
    hvm_core_book: dict[str, Any],
    hvm_core_source: str,
    model_source: dict[str, Any],
    manifest: dict[str, Any],
    tensor_islands: dict[str, Any],
    engine: dict[str, Any],
) -> None:
    if _read_object(hvm_core_book, "source") != _read_object(manifest, "source"):
        raise BumkcArtifactError("BUMKC HVM Core Book source mismatches manifest")
    if hvm_core_book.get("coverage_status") != model_source.get("hvm_capture_status"):
        raise BumkcArtifactError("BUMKC HVM Core Book coverage mismatches source")

    program_id = _read_str(engine, "program_id")
    program_ref = _hvm_symbol("bumkc_program", program_id)
    expected_main = f"@main = @{program_ref}\n"
    if not hvm_core_source.startswith(expected_main):
        raise BumkcArtifactError("BUMKC HVM Core source main is not canonical")

    required_refs = (
        program_ref,
        _hvm_symbol("bumkc_regions", program_id),
        _hvm_symbol("bumkc_tensor_islands", program_id),
        _hvm_symbol("bumkc_shape_symbols", program_id),
        _hvm_symbol("bumkc_fallback_bridges", program_id),
        _hvm_symbol("bumkc_tensor_descriptors", program_id),
        _hvm_symbol("bumkc_quantization", program_id),
    )
    if any(f"@{reference}" not in hvm_core_source for reference in required_refs):
        raise BumkcArtifactError("BUMKC HVM Core source missing canonical references")
    if "#BumkcProgram{" not in hvm_core_source:
        raise BumkcArtifactError("BUMKC HVM Core source missing program term")
    if hvm_core_source.count("#BumkcTensorIsland{") != _read_int(
        model_source, "tensor_island_count"
    ):
        raise BumkcArtifactError("BUMKC HVM Core source tensor island count mismatch")
    if hvm_core_source.count("#BumkcShapeSymbol{") != _read_int(
        model_source, "shape_symbol_count"
    ):
        raise BumkcArtifactError("BUMKC HVM Core source shape symbol count mismatch")
    if hvm_core_source.count("#BumkcFallbackBridge{") != _read_int(
        model_source, "fallback_bridge_count"
    ):
        raise BumkcArtifactError("BUMKC HVM Core source fallback bridge count mismatch")

    root_region = _read_str(hvm_core_book, "root_region")
    regions = _read_list(hvm_core_book, "regions")
    region_ids = {_read_str(region, "id") for region in regions}
    if root_region not in region_ids:
        raise BumkcArtifactError("BUMKC HVM Core Book root region is missing")

    known_islands = {
        _read_str(island, "id") for island in _read_list(tensor_islands, "islands")
    }
    manifest_source = _read_object(manifest, "source")
    model_entry_count = 0
    tensor_node_islands = set()
    for region in regions:
        _read_str(region, "kind")
        for node in _read_list(region, "nodes"):
            kind = _read_object(node, "kind")
            if len(kind) != 1:
                raise BumkcArtifactError("BUMKC HVM Core node kind is malformed")
            tag, payload = next(iter(kind.items()))
            if not isinstance(payload, dict):
                raise BumkcArtifactError("BUMKC HVM Core node payload is malformed")
            if tag == "model_entry":
                if payload.get("model") != manifest_source.get("model"):
                    raise BumkcArtifactError("BUMKC HVM Core model entry mismatches")
                model_entry_count += 1
            elif tag == "tensor_island":
                island = _read_str(payload, "island")
                if island not in known_islands:
                    raise BumkcArtifactError("BUMKC HVM Core tensor island is unknown")
                tensor_node_islands.add(island)
            elif tag == "fallback_boundary":
                if not _read_str(payload, "reason"):
                    raise BumkcArtifactError("BUMKC HVM Core fallback reason is empty")
            else:
                raise BumkcArtifactError(
                    f"BUMKC HVM Core node kind is unsupported: {tag}"
                )
    if model_entry_count != 1:
        raise BumkcArtifactError("BUMKC HVM Core must contain one model entry")
    if tensor_node_islands != known_islands:
        raise BumkcArtifactError("BUMKC HVM Core does not cover every tensor island")


def _artifact_file_paths(root: Path) -> list[str]:
    paths = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise BumkcArtifactError("BUMKC artifact digest rejects symlinks")
        relative_path = path.relative_to(root).as_posix()
        if relative_path not in (
            ARTIFACT_DIGEST_PATH.as_posix(),
            VERIFY_REPORT_PATH.as_posix(),
        ):
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

    target_plan = _read_object(runtime, "target_plan")
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
    diagnostics_plan = _read_object(runtime, "diagnostics_plan")
    worker_count = _read_int(target_plan, "worker_count")
    scheduler_count = _read_int(target_plan, "scheduler_count")
    engine_gpu_count = _read_int(engine, "gpu_count")
    runtime_gpu_count = _read_int(target_plan, "gpu_count")
    rank_count = _read_int(scale_up_plan, "rank_count")
    if runtime_gpu_count != engine_gpu_count:
        raise BumkcArtifactError("BUMKC runtime target GPU count mismatches engine")
    if rank_count != engine_gpu_count:
        raise BumkcArtifactError("BUMKC runtime scale-up rank count mismatches engine")
    if target_plan.get("target_arch") != engine.get("target_arch"):
        raise BumkcArtifactError("BUMKC runtime target architecture mismatches engine")
    if _read_int(diagnostics_plan, "heartbeat_slot_count") != (
        worker_count + scheduler_count
    ):
        raise BumkcArtifactError("BUMKC runtime diagnostic heartbeat count mismatch")
    if _read_int(diagnostics_plan, "worker_heartbeat_slot_count") != worker_count:
        raise BumkcArtifactError("BUMKC runtime worker heartbeat count mismatch")
    if _read_int(diagnostics_plan, "scheduler_heartbeat_slot_count") != scheduler_count:
        raise BumkcArtifactError("BUMKC runtime scheduler heartbeat count mismatch")
    if _read_int(diagnostics_plan, "queue_snapshot_slot_count") != (
        _read_int(queue_plan, "worker_queue_count")
        + _read_int(queue_plan, "scheduler_queue_count")
    ):
        raise BumkcArtifactError("BUMKC runtime queue snapshot count mismatch")
    if _read_int(diagnostics_plan, "event_counter_snapshot_count") != _read_int(
        queue_plan, "event_counter_count"
    ):
        raise BumkcArtifactError("BUMKC runtime event counter snapshot count mismatch")
    if _read_int(diagnostics_plan, "last_completed_task_slot_count") != worker_count:
        raise BumkcArtifactError("BUMKC runtime last-completed task count mismatch")
    if _read_int(diagnostics_plan, "blocked_event_slot_count") != scheduler_count:
        raise BumkcArtifactError("BUMKC runtime blocked-event count mismatch")
    if _read_int(diagnostics_plan, "watchdog_timeout_us") <= _read_int(
        diagnostics_plan, "watchdog_poll_interval_us"
    ):
        raise BumkcArtifactError("BUMKC runtime watchdog timing is invalid")
    dependency_descriptors = _read_list(dependency_plan, "dependency_descriptors")
    _validate_dependency_descriptors(dependency_descriptors)
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
        "rank_count": rank_count,
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
        "diagnostic_heartbeat_slot_count": diagnostics_plan.get("heartbeat_slot_count"),
        "diagnostic_queue_snapshot_slot_count": diagnostics_plan.get(
            "queue_snapshot_slot_count"
        ),
        "diagnostic_event_counter_snapshot_count": diagnostics_plan.get(
            "event_counter_snapshot_count"
        ),
        "diagnostic_last_completed_task_slot_count": diagnostics_plan.get(
            "last_completed_task_slot_count"
        ),
        "diagnostic_blocked_event_slot_count": diagnostics_plan.get(
            "blocked_event_slot_count"
        ),
        "watchdog_poll_interval_us": diagnostics_plan.get("watchdog_poll_interval_us"),
        "watchdog_timeout_us": diagnostics_plan.get("watchdog_timeout_us"),
    }
    for key, value in expected.items():
        if value is None or _read_summary_int(summary, key) != value:
            raise BumkcArtifactError(f"BUMKC engine runtime summary mismatch: {key}")

    return BumkcRuntimeSummary(**{key: int(value) for key, value in expected.items()})


def _load_scale_up_summary(
    engine: dict[str, Any],
    runtime: dict[str, Any],
) -> BumkcScaleUpSummary:
    expected = _expected_scale_up_summary(engine, runtime)
    summary_record = _read_optional_object(engine, "scale_up_summary")
    if engine.get("schema_version") == REQUIRED_SCHEMA_VERSION:
        if summary_record is None:
            raise BumkcArtifactError("BUMKC engine scale-up summary is missing")
        summary = _read_scale_up_summary(summary_record)
        _validate_scale_up_summary(summary, expected)
        return summary
    if summary_record is not None:
        summary = _read_scale_up_summary(summary_record)
        _validate_scale_up_summary(summary, expected)
    return expected


def _expected_scale_up_summary(
    engine: dict[str, Any],
    runtime: dict[str, Any],
) -> BumkcScaleUpSummary:
    target_plan = _read_object(runtime, "target_plan")
    scale_up_plan = _read_object(runtime, "scale_up_plan")
    communication_plan = _read_object(runtime, "communication_plan")
    target_arch = _read_optional_str(target_plan, "target_arch")
    return BumkcScaleUpSummary(
        compile_gpu_count=_read_int(engine, "gpu_count"),
        runtime_target_gpu_count=_read_int(target_plan, "gpu_count"),
        runtime_rank_count=_read_int(scale_up_plan, "rank_count"),
        target_arch=target_arch,
        target_arch_code=_target_arch_code(target_arch),
        worker_count=_read_int(target_plan, "worker_count"),
        scheduler_count=_read_int(target_plan, "scheduler_count"),
        task_rank_group_count=_read_int(scale_up_plan, "task_rank_group_count"),
        task_rank_reference_count=_read_int(scale_up_plan, "task_rank_reference_count"),
        rank_id_sum=_read_int(scale_up_plan, "rank_id_sum"),
        collective_task_count=_read_int(communication_plan, "collective_task_count"),
        collective_group_size_sum=_read_int(
            communication_plan, "collective_group_size_sum"
        ),
        collective_kind_code_sum=_read_int(
            communication_plan, "collective_kind_code_sum"
        ),
    )


def _read_scale_up_summary(
    summary: dict[str, Any],
) -> BumkcScaleUpSummary:
    return BumkcScaleUpSummary(
        compile_gpu_count=_read_int(summary, "compile_gpu_count"),
        runtime_target_gpu_count=_read_int(summary, "runtime_target_gpu_count"),
        runtime_rank_count=_read_int(summary, "runtime_rank_count"),
        target_arch=_read_optional_str(summary, "target_arch"),
        target_arch_code=_read_int(summary, "target_arch_code"),
        worker_count=_read_int(summary, "worker_count"),
        scheduler_count=_read_int(summary, "scheduler_count"),
        task_rank_group_count=_read_int(summary, "task_rank_group_count"),
        task_rank_reference_count=_read_int(summary, "task_rank_reference_count"),
        rank_id_sum=_read_int(summary, "rank_id_sum"),
        collective_task_count=_read_int(summary, "collective_task_count"),
        collective_group_size_sum=_read_int(summary, "collective_group_size_sum"),
        collective_kind_code_sum=_read_int(summary, "collective_kind_code_sum"),
    )


def _validate_scale_up_summary(
    summary: BumkcScaleUpSummary,
    expected: BumkcScaleUpSummary,
) -> None:
    for field in dataclasses.fields(BumkcScaleUpSummary):
        name = field.name
        if getattr(summary, name) != getattr(expected, name):
            raise BumkcArtifactError(f"BUMKC engine scale-up summary mismatch: {name}")


def _load_launch_summary(
    engine: dict[str, Any],
    runtime: dict[str, Any],
) -> BumkcLaunchSummary:
    expected = _expected_launch_summary(runtime)
    summary_record = _read_optional_object(engine, "launch_summary")
    if engine.get("schema_version") == REQUIRED_SCHEMA_VERSION:
        if summary_record is None:
            raise BumkcArtifactError("BUMKC engine launch summary is missing")
        summary = _read_launch_summary(summary_record)
        _validate_launch_summary(summary, expected)
        return summary
    if summary_record is not None:
        summary = _read_launch_summary(summary_record)
        _validate_launch_summary(summary, expected)
    return expected


def _expected_launch_summary(runtime: dict[str, Any]) -> BumkcLaunchSummary:
    substitution_plan = _read_object(runtime, "substitution_plan")
    shape_symbols = _read_list(substitution_plan, "shape_symbols")
    serving_state = _read_list(substitution_plan, "serving_state")
    serving_kind_code_sum = 0
    serving_symbol_count = 0
    for binding in serving_state:
        kind = _read_str(binding, "kind")
        if kind not in _SERVING_STATE_KIND_CODES:
            raise BumkcArtifactError(
                f"BUMKC runtime serving-state kind is unsupported: {kind}"
            )
        serving_kind_code_sum += _SERVING_STATE_KIND_CODES[kind]
        if _read_optional_str(binding, "symbol") is not None:
            serving_symbol_count += 1

    return BumkcLaunchSummary(
        shape_symbol_count=len(shape_symbols),
        shape_symbol_min_sum=sum(_read_int(symbol, "min") for symbol in shape_symbols),
        shape_symbol_max_sum=sum(_read_int(symbol, "max") for symbol in shape_symbols),
        shape_symbol_bucket_sum=sum(
            _read_int(symbol, "bucket") for symbol in shape_symbols
        ),
        default_shape_value_sum=sum(
            _read_int(symbol, "default_value") for symbol in shape_symbols
        ),
        default_bucketed_shape_value_sum=sum(
            _bucket_shape_value(
                _read_int(symbol, "default_value"),
                _read_int(symbol, "bucket"),
            )
            for symbol in shape_symbols
        ),
        serving_binding_count=len(serving_state),
        required_serving_binding_count=sum(
            1 for binding in serving_state if _read_bool(binding, "required")
        ),
        optional_serving_binding_count=sum(
            1 for binding in serving_state if not _read_bool(binding, "required")
        ),
        serving_kind_code_sum=serving_kind_code_sum,
        serving_symbol_count=serving_symbol_count,
    )


def _read_launch_summary(
    summary: dict[str, Any],
) -> BumkcLaunchSummary:
    return BumkcLaunchSummary(
        shape_symbol_count=_read_int(summary, "shape_symbol_count"),
        shape_symbol_min_sum=_read_int(summary, "shape_symbol_min_sum"),
        shape_symbol_max_sum=_read_int(summary, "shape_symbol_max_sum"),
        shape_symbol_bucket_sum=_read_int(summary, "shape_symbol_bucket_sum"),
        default_shape_value_sum=_read_int(summary, "default_shape_value_sum"),
        default_bucketed_shape_value_sum=_read_int(
            summary, "default_bucketed_shape_value_sum"
        ),
        serving_binding_count=_read_int(summary, "serving_binding_count"),
        required_serving_binding_count=_read_int(
            summary, "required_serving_binding_count"
        ),
        optional_serving_binding_count=_read_int(
            summary, "optional_serving_binding_count"
        ),
        serving_kind_code_sum=_read_int(summary, "serving_kind_code_sum"),
        serving_symbol_count=_read_int(summary, "serving_symbol_count"),
    )


def _validate_launch_summary(
    summary: BumkcLaunchSummary,
    expected: BumkcLaunchSummary,
) -> None:
    for field in dataclasses.fields(BumkcLaunchSummary):
        name = field.name
        if getattr(summary, name) != getattr(expected, name):
            raise BumkcArtifactError(f"BUMKC engine launch summary mismatch: {name}")


def _validate_dependency_descriptors(descriptors: list[dict[str, Any]]) -> None:
    for descriptor in descriptors:
        scope = _read_str(descriptor, "scope")
        wait_expression = _read_str(descriptor, "wait_expression")
        expected_wait_expression = _DEPENDENCY_WAIT_EXPRESSIONS.get(scope)
        if expected_wait_expression is None:
            raise BumkcArtifactError(f"BUMKC dependency scope is unsupported: {scope}")
        if wait_expression != expected_wait_expression:
            raise BumkcArtifactError(
                "BUMKC runtime dependency wait expression mismatch"
            )
        if scope == "tile_overlap" and not _read_any_list(descriptor, "tensors"):
            raise BumkcArtifactError(
                "BUMKC tile-overlap dependency requires tensor flow"
            )


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
    benchmark_conventional_launch_count = min(
        runtime_summary.conventional_launch_count,
        RUNTIME_SMOKE_BENCHMARK_LAUNCH_CAP,
    )
    benchmark_persistent_launch_count = runtime_summary.persistent_launch_count
    benchmark_enabled = (
        benchmark_conventional_launch_count != 0
        and benchmark_persistent_launch_count != 0
    )
    benchmark_iterations = (
        RUNTIME_SMOKE_BENCHMARK_ITERATIONS if benchmark_enabled else 0
    )
    expected = {
        "expected_task_count": runtime_summary.task_count,
        "expected_conventional_launch_count": (
            runtime_summary.conventional_launch_count
        ),
        "expected_persistent_launch_count": runtime_summary.persistent_launch_count,
        "expected_launch_reduction_per_mille": _read_int(
            execution_model, "launch_reduction_per_mille"
        ),
        "benchmark_enabled": int(benchmark_enabled),
        "benchmark_iterations": benchmark_iterations,
        "benchmark_conventional_launch_count": benchmark_conventional_launch_count,
        "benchmark_persistent_launch_count": benchmark_persistent_launch_count,
        "benchmark_total_conventional_launches": (
            benchmark_iterations * benchmark_conventional_launch_count
        ),
        "benchmark_total_persistent_launches": (
            benchmark_iterations * benchmark_persistent_launch_count
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
        "expected_diagnostic_heartbeat_slot_count": (
            runtime_summary.diagnostic_heartbeat_slot_count
        ),
        "expected_diagnostic_queue_snapshot_slot_count": (
            runtime_summary.diagnostic_queue_snapshot_slot_count
        ),
        "expected_diagnostic_event_counter_snapshot_count": (
            runtime_summary.diagnostic_event_counter_snapshot_count
        ),
        "expected_diagnostic_last_completed_task_slot_count": (
            runtime_summary.diagnostic_last_completed_task_slot_count
        ),
        "expected_diagnostic_blocked_event_slot_count": (
            runtime_summary.diagnostic_blocked_event_slot_count
        ),
        "expected_watchdog_poll_interval_us": (
            runtime_summary.watchdog_poll_interval_us
        ),
        "expected_watchdog_timeout_us": runtime_summary.watchdog_timeout_us,
    }
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke mismatch: {key}")

    event_descriptors = _read_list(runtime_smoke, "event_descriptors")
    event_dispatch_descriptors = _read_list(runtime_smoke, "event_dispatch_descriptors")
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
    _validate_runtime_smoke_event_dispatch_descriptors(
        runtime_smoke,
        event_dispatch_descriptors,
        task_descriptors,
    )
    _validate_runtime_smoke_task_descriptors(
        runtime_smoke,
        task_descriptors,
    )
    _validate_runtime_smoke_contracts(
        runtime_smoke,
        event_descriptors,
        event_dispatch_descriptors,
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


def _validate_runtime_smoke_event_dispatch_descriptors(
    runtime_smoke: dict[str, Any],
    event_dispatch_descriptors: list[dict[str, Any]],
    task_descriptors: list[dict[str, Any]],
) -> None:
    expected = {
        "expected_event_dispatch_range_count": len(event_dispatch_descriptors),
        "expected_event_dispatch_compact_range_count": 0,
        "expected_event_dispatch_indexed_range_count": 0,
        "expected_event_dispatch_task_count_sum": 0,
        "expected_event_dispatch_trigger_count_sum": 0,
        "expected_event_dispatch_trigger_match_count": 0,
        "expected_event_dispatch_owner_scheduler_sum": 0,
        "expected_event_dispatch_max_task_range_len": 0,
        "expected_event_dispatch_max_trigger_count": 0,
        "expected_event_dispatch_range_kind_code_sum": 0,
        "expected_event_dispatch_invalid_range_count": 0,
    }
    task_count = len(task_descriptors)
    for ordinal, event_range in enumerate(event_dispatch_descriptors):
        if _read_int(event_range, "ordinal") != ordinal:
            raise BumkcArtifactError(
                "BUMKC runtime smoke event dispatch ordinal mismatch"
            )
        first_task_index = _read_int(event_range, "first_task_index")
        range_task_count = _read_int(event_range, "task_count")
        trigger_count = _read_int(event_range, "trigger_count")
        owner_scheduler = _read_int(event_range, "owner_scheduler")
        range_kind_code = _read_int(event_range, "range_kind_code")
        expected["expected_event_dispatch_task_count_sum"] += range_task_count
        expected["expected_event_dispatch_trigger_count_sum"] += trigger_count
        expected["expected_event_dispatch_owner_scheduler_sum"] += owner_scheduler
        expected["expected_event_dispatch_range_kind_code_sum"] += range_kind_code
        expected["expected_event_dispatch_compact_range_count"] += int(
            range_kind_code == 1
        )
        expected["expected_event_dispatch_indexed_range_count"] += int(
            range_kind_code == 2
        )
        expected["expected_event_dispatch_max_task_range_len"] = max(
            expected["expected_event_dispatch_max_task_range_len"],
            range_task_count,
        )
        expected["expected_event_dispatch_max_trigger_count"] = max(
            expected["expected_event_dispatch_max_trigger_count"],
            trigger_count,
        )
        range_end = first_task_index + range_task_count
        invalid_range = (
            range_task_count == 0
            or first_task_index > task_count
            or range_end > task_count
            or range_end < first_task_index
        )
        expected["expected_event_dispatch_invalid_range_count"] += int(invalid_range)
        if not invalid_range:
            matching_task_count = 0
            for task in task_descriptors[first_task_index:range_end]:
                matching_task_count += int(
                    _read_int(task, "predecessor_event_count") == trigger_count
                )
            expected["expected_event_dispatch_trigger_match_count"] += int(
                matching_task_count == range_task_count
            )
    for key, value in expected.items():
        if _read_int(runtime_smoke, key) != value:
            raise BumkcArtifactError(f"BUMKC runtime smoke descriptor mismatch: {key}")


def _validate_runtime_smoke_contracts(
    runtime_smoke: dict[str, Any],
    event_descriptors: list[dict[str, Any]],
    event_dispatch_descriptors: list[dict[str, Any]],
    task_descriptors: list[dict[str, Any]],
) -> None:
    descriptor_hash = _runtime_smoke_descriptor_contract_hash(
        event_descriptors,
        event_dispatch_descriptors,
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
    event_dispatch_descriptors: list[dict[str, Any]],
    task_descriptors: list[dict[str, Any]],
) -> int:
    hash_value = _mix_contract_str(_CONTRACT_HASH_OFFSET, _DESCRIPTOR_CONTRACT_DOMAIN)
    hash_value = _mix_contract_u64(hash_value, len(event_descriptors))
    hash_value = _mix_contract_u64(hash_value, len(event_dispatch_descriptors))
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
    for ordinal, event_range in enumerate(event_dispatch_descriptors):
        hash_value = _mix_contract_u64(hash_value, ordinal)
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event_range, "first_task_index"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event_range, "task_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event_range, "trigger_count"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event_range, "owner_scheduler"),
        )
        hash_value = _mix_contract_u64(
            hash_value,
            _read_int(event_range, "range_kind_code"),
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
        if binding.bucketed_value(binding.default_value) > binding.max:
            raise BumkcArtifactError(
                "BUMKC runtime shape symbol "
                f"{binding.symbol} has invalid default bucket"
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
    shape_symbols: tuple[BumkcRuntimeShapeSymbolBinding, ...],
) -> tuple[BumkcRuntimeServingStateBinding, ...]:
    substitution_plan = _read_object(runtime, "substitution_plan")
    bindings = []
    seen_bindings = set()
    known_shape_symbols = {binding.symbol for binding in shape_symbols}
    for entry in _read_list(substitution_plan, "serving_state"):
        binding = BumkcRuntimeServingStateBinding(
            kind=_read_str(entry, "kind"),
            symbol=_read_optional_str(entry, "symbol"),
            required=_read_bool(entry, "required"),
        )
        if binding.kind not in _SERVING_STATE_KIND_CODES:
            raise BumkcArtifactError(
                f"BUMKC runtime serving-state kind is unsupported: {binding.kind}"
            )
        if binding.symbol is not None and binding.symbol not in known_shape_symbols:
            raise BumkcArtifactError(
                f"BUMKC runtime serving-state symbol is not declared: {binding.symbol}"
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


def _quantization_format_code(parent: dict[str, Any], key: str) -> int:
    value = _read_optional_str(parent, key)
    if value is None:
        return 0
    if value not in _QUANTIZATION_FORMAT_CODES:
        raise BumkcArtifactError(f"BUMKC quantization format is unsupported: {value}")
    return _QUANTIZATION_FORMAT_CODES[value]


def _quantization_scale_layout_code(parent: dict[str, Any], key: str) -> int:
    value = _read_optional_str(parent, key)
    if value is None:
        return 0
    if value not in _QUANTIZATION_SCALE_LAYOUT_CODES:
        raise BumkcArtifactError(
            f"BUMKC quantization scale layout is unsupported: {value}"
        )
    return _QUANTIZATION_SCALE_LAYOUT_CODES[value]


def _dtype_code(parent: dict[str, Any], key: str) -> int:
    value = _read_optional_str(parent, key)
    if value is None:
        return 0
    if value not in _DTYPE_CODES:
        raise BumkcArtifactError(f"BUMKC dtype is unsupported: {value}")
    return _DTYPE_CODES[value]


def _target_arch_code(value: str | None) -> int:
    if value is None:
        return 0
    if value == "sm80":
        return 80
    if value == "sm90":
        return 90
    if value == "sm100":
        return 100
    raise BumkcArtifactError(f"BUMKC target architecture is unsupported: {value}")


def _bucket_shape_value(value: int, bucket: int) -> int:
    return ((value + bucket - 1) // bucket) * bucket


def _hvm_symbol(prefix: str, value: str) -> str:
    return f"{prefix}_{_hvm_name(value)}"


def _hvm_name(value: str) -> str:
    name = []
    for char in value:
        if char.isascii() and (char.isalnum() or char == "_"):
            name.append(char)
        elif char == "-":
            name.append("_dash_")
        elif char == ".":
            name.append("_dot_")
        else:
            name.append("_")
    return "".join(name)


def _optional_bool_code(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if value is None:
        return 0
    if not isinstance(value, bool):
        raise BumkcArtifactError(f"BUMKC runtime {key} boolean is invalid")
    return 1 if value else 2


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


def _read_optional_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if value is None:
        return 0
    if not _is_strict_int(value):
        raise BumkcArtifactError(f"BUMKC runtime {key} integer is invalid")
    return value


def _read_nullable_int(parent: dict[str, Any], key: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if not _is_strict_int(value):
        raise BumkcArtifactError(f"BUMKC runtime {key} integer is invalid")
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


def _read_nullable_bool(parent: dict[str, Any], key: str) -> bool | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise BumkcArtifactError(f"BUMKC runtime {key} boolean is invalid")
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
