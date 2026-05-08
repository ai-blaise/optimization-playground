import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "python/sglang/srt/bumkc/artifact.py"
SERVER_ARGS_PATH = Path(__file__).parents[2] / "python/sglang/srt/server_args.py"
SPEC = importlib.util.spec_from_file_location("bumkc_artifact_under_test", MODULE_PATH)
bumkc_artifact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bumkc_artifact
SPEC.loader.exec_module(bumkc_artifact)

REQUIRED_VALIDATION_MODEL = bumkc_artifact.REQUIRED_VALIDATION_MODEL
REQUIRED_PLAN_SCHEMA_VERSION = bumkc_artifact.REQUIRED_PLAN_SCHEMA_VERSION
REQUIRED_CAPABILITY_LEVEL = bumkc_artifact.REQUIRED_CAPABILITY_LEVEL
REQUIRED_SCHEMA_VERSION = bumkc_artifact.REQUIRED_SCHEMA_VERSION
REQUIRED_SOURCE_SCHEMA_VERSION = bumkc_artifact.REQUIRED_SOURCE_SCHEMA_VERSION
REQUIRED_RUNTIME_ABI_VERSION = bumkc_artifact.REQUIRED_RUNTIME_ABI_VERSION
REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION = (
    bumkc_artifact.REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION
)
BumkcArtifactError = bumkc_artifact.BumkcArtifactError
load_bumkc_artifact = bumkc_artifact.load_bumkc_artifact


def test_server_args_exposes_checked_bumkc_fallback_mode():
    source = SERVER_ARGS_PATH.read_text(encoding="utf-8")

    assert 'bumkc_fallback_mode: str = "checked"' in source
    assert '"--bumkc-fallback-mode"' in source
    assert 'choices=["checked"]' in source
    assert '"--bumkc-fallback-mode must be checked"' in source
    assert "self._bumkc_artifact_summary.serving_hints" in source
    assert "validate_scale_up_domain" in source
    assert "validate_target_architecture" in source
    assert "gpu_count=self.tp_size * self.pp_size" in source
    assert 'serving_target_arch = f"sm{device_sm}"' in source
    assert "self.quantization = serving_hints.quantization" in source
    assert "self.moe_runner_backend = serving_hints.moe_runner_backend" in source


def test_loads_executable_bumkc_artifact(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)

    summary = load_bumkc_artifact(tmp_path, require_executable=True)

    assert summary.root == plan_dir
    assert summary.runtime_executable
    assert summary.runtime_entrypoints == ("cuda_tensor_smoke",)
    assert summary.compiler_summary.tensor_island_count == 3
    assert summary.compiler_summary.native_tensor_island_count == 2
    assert summary.compiler_summary.fallback_tensor_island_count == 1
    assert summary.compiler_summary.side_effecting_tensor_island_count == 1
    assert summary.compiler_summary.tensor_island_side_effect_count == 1
    assert summary.compiler_summary.tensor_island_side_effect_code_sum == 3
    assert summary.compiler_summary.fallback_bridge_count == 1
    assert summary.compiler_summary.moe_dispatch_tensor_island_count == 0
    assert summary.compiler_summary.block_op_count == 2
    assert summary.compiler_summary.moe_dispatch_block_op_count == 0
    assert summary.compiler_summary.side_effecting_block_op_count == 1
    assert summary.compiler_summary.block_side_effect_count == 1
    assert summary.compiler_summary.block_side_effect_code_sum == 3
    assert summary.compiler_summary.event_tensor_count == 2
    assert summary.compiler_summary.side_effecting_event_tensor_count == 1
    assert summary.compiler_summary.event_side_effect_count == 1
    assert summary.compiler_summary.event_side_effect_code_sum == 3
    assert summary.compiler_summary.moe_dispatch_event_tensor_count == 0
    assert summary.compiler_summary.event_predecessor_edge_count == 1
    assert summary.compiler_summary.event_successor_edge_count == 1
    assert summary.compiler_summary.event_dependency_tensor_count == 1
    assert summary.compiler_summary.event_notification_count == 1
    assert summary.compiler_summary.event_execution_count == 2
    assert summary.compiler_summary.event_simulation_violation_count == 0
    assert summary.runtime_summary.task_instance_capacity == 2
    assert summary.runtime_summary.kv_cache_binding_count == 1
    assert summary.runtime_summary.rank_count == 8
    assert summary.runtime_summary.task_rank_reference_count == 16
    assert summary.runtime_summary.task_dependency_count == 1
    assert summary.runtime_summary.dependency_tensor_count == 1
    assert summary.runtime_summary.dependency_scope_code_sum == 1
    assert summary.runtime_summary.dependency_descriptor_count == 1
    assert summary.runtime_summary.dependency_descriptor_hash != 0
    assert summary.runtime_summary.collective_task_count == 1
    assert summary.runtime_summary.collective_group_size_sum == 8
    assert summary.runtime_summary.collective_kind_code_sum == 1
    assert summary.runtime_summary.side_effecting_task_count == 1
    assert summary.runtime_summary.task_side_effect_count == 1
    assert summary.runtime_summary.task_side_effect_code_sum == 3
    assert summary.runtime_summary.serving_task_count == 1
    assert summary.runtime_summary.serving_dependency_count == 2
    assert summary.runtime_summary.serving_kind_code_sum == 5
    assert summary.runtime_summary.serving_symbol_count == 1
    assert summary.runtime_summary.substitution_shape_symbol_count == 1
    assert summary.runtime_summary.substitution_serving_binding_count == 2
    assert summary.runtime_summary.substitution_symbol_max_sum == 4096
    assert summary.runtime_summary.substitution_symbol_bucket_sum == 16
    assert summary.runtime_summary.diagnostic_heartbeat_slot_count == 72
    assert summary.runtime_summary.diagnostic_queue_snapshot_slot_count == 72
    assert summary.runtime_summary.diagnostic_event_counter_snapshot_count == 2
    assert summary.runtime_summary.diagnostic_last_completed_task_slot_count == 64
    assert summary.runtime_summary.diagnostic_blocked_event_slot_count == 8
    assert summary.runtime_summary.watchdog_poll_interval_us == 1000
    assert summary.runtime_summary.watchdog_timeout_us == 30_000_000
    assert summary.scale_up_summary.compile_gpu_count == 8
    assert summary.scale_up_summary.runtime_target_gpu_count == 8
    assert summary.scale_up_summary.runtime_rank_count == 8
    assert summary.scale_up_summary.target_arch == "sm90"
    assert summary.scale_up_summary.target_arch_code == 90
    assert summary.scale_up_summary.worker_count == 64
    assert summary.scale_up_summary.scheduler_count == 8
    assert summary.scale_up_summary.task_rank_group_count == 2
    assert summary.scale_up_summary.task_rank_reference_count == 16
    assert summary.scale_up_summary.rank_id_sum == 56
    assert summary.scale_up_summary.collective_task_count == 1
    assert summary.scale_up_summary.collective_group_size_sum == 8
    assert summary.scale_up_summary.collective_kind_code_sum == 1
    assert summary.launch_summary.shape_symbol_count == 1
    assert summary.launch_summary.shape_symbol_min_sum == 1
    assert summary.launch_summary.shape_symbol_max_sum == 4096
    assert summary.launch_summary.shape_symbol_bucket_sum == 16
    assert summary.launch_summary.default_shape_value_sum == 1
    assert summary.launch_summary.default_bucketed_shape_value_sum == 16
    assert summary.launch_summary.serving_binding_count == 2
    assert summary.launch_summary.required_serving_binding_count == 2
    assert summary.launch_summary.optional_serving_binding_count == 0
    assert summary.launch_summary.serving_kind_code_sum == 5
    assert summary.launch_summary.serving_symbol_count == 1
    assert summary.quantization_summary.scheme == "W4A4KV4+IndexerK8"
    assert summary.quantization_summary.scheme_hash != 0
    assert summary.quantization_summary.weight_format == "nv_fp4"
    assert summary.quantization_summary.weight_format_code == 1
    assert summary.quantization_summary.weight_bits == 4
    assert summary.quantization_summary.weight_scale_layout == "tensor_group"
    assert summary.quantization_summary.weight_scale_layout_code == 2
    assert summary.quantization_summary.weight_group_size == 16
    assert summary.quantization_summary.weight_scale_dtype == "fp8_e4m3"
    assert summary.quantization_summary.weight_scale_dtype_code == 7
    assert summary.quantization_summary.weight_symmetric is True
    assert summary.quantization_summary.weight_symmetric_code == 1
    assert not summary.quantization_summary.weight_zero_point
    assert summary.quantization_summary.activation_bits == 4
    assert summary.quantization_summary.kv_bits == 4
    assert summary.quantization_summary.kv_format == "nv_fp4"
    assert summary.quantization_summary.kv_format_code == 1
    assert summary.quantization_summary.indexer_k_bits == 8
    assert summary.quantization_summary.indexer_k_format == "fp8_e4m3"
    assert summary.quantization_summary.indexer_k_format_code == 2
    assert summary.quantization_summary.gated_norm is True
    assert summary.quantization_summary.spinquant is True
    assert summary.quantization_summary.ignored_module_count == 2
    assert summary.serving_hints.quantization == "modelopt_fp4"
    assert summary.serving_hints.moe_runner_backend == "flashinfer_trtllm"
    assert summary.runtime_shape_symbols[0].symbol == "sequence"
    assert summary.runtime_serving_state[0].key() == ("sequence", "sequence")
    assert summary.target_arch == "sm90"
    assert summary.plan_schema_version == REQUIRED_PLAN_SCHEMA_VERSION
    assert summary.capability_level == REQUIRED_CAPABILITY_LEVEL
    assert summary.source_schema_version == REQUIRED_SOURCE_SCHEMA_VERSION
    assert summary.model_source_frontend == "internal_test"
    assert summary.hvm_capture_status == "native_eligible"
    assert summary.engine_schema_version == REQUIRED_SCHEMA_VERSION
    assert summary.runtime_mode == "debug"
    assert summary.task_count == summary.runtime_summary.task_count
    assert summary.tensor_smoke_enabled
    assert summary.artifact_digest_count == 16
    assert summary.fallback_mode == "checked"
    log_dict = summary.as_log_dict()
    assert log_dict["plan_schema_version"] == REQUIRED_PLAN_SCHEMA_VERSION
    assert log_dict["capability_level"] == REQUIRED_CAPABILITY_LEVEL
    assert log_dict["source_schema_version"] == REQUIRED_SOURCE_SCHEMA_VERSION
    assert log_dict["model_source_frontend"] == "internal_test"
    assert log_dict["hvm_capture_status"] == "native_eligible"
    assert log_dict["engine_schema_version"] == REQUIRED_SCHEMA_VERSION
    assert log_dict["runtime_mode"] == "debug"
    assert log_dict["runtime_summary"]["diagnostic_heartbeat_slot_count"] == 72
    assert log_dict["runtime_summary"]["watchdog_timeout_us"] == 30_000_000
    assert log_dict["scale_up_summary"]["runtime_rank_count"] == 8
    assert log_dict["scale_up_summary"]["target_arch_code"] == 90
    assert log_dict["launch_summary"]["default_bucketed_shape_value_sum"] == 16
    assert log_dict["launch_summary"]["serving_kind_code_sum"] == 5
    assert log_dict["quantization_summary"]["indexer_k_bits"] == 8
    assert log_dict["quantization_summary"]["gated_norm"]
    assert log_dict["serving_hints"]["quantization"] == "modelopt_fp4"
    assert log_dict["serving_hints"]["moe_runner_backend"] == "flashinfer_trtllm"
    summary.validate_scale_up_domain(gpu_count=8)

    launch_plan = summary.validate_runtime_launch(
        shape_symbols={"sequence": 17},
        serving_state=[("sequence", "sequence"), ("decode_step", None)],
    )
    assert launch_plan.shape_symbols[0].bucketed_value == 32
    default_launch_plan = summary.validate_default_runtime_launch()
    assert default_launch_plan.shape_symbols[0].value == 1


def test_rejects_required_non_executable_bumkc_artifact(tmp_path):
    write_bumkc_artifact(tmp_path, executable=False)

    with pytest.raises(BumkcArtifactError, match="not executable"):
        load_bumkc_artifact(tmp_path, require_executable=True)


def test_loads_executable_production_bumkc_artifact(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    set_bumkc_runtime_mode(plan_dir, "production")

    summary = load_bumkc_artifact(plan_dir)

    assert summary.runtime_mode == "production"
    assert summary.runtime_executable


def test_rejects_production_non_executable_bumkc_artifact(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=False)
    set_bumkc_runtime_mode(plan_dir, "production")

    with pytest.raises(BumkcArtifactError, match="production runtime mode"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unknown_bumkc_runtime_mode(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    set_bumkc_runtime_mode(plan_dir, "minimum_viable")

    with pytest.raises(BumkcArtifactError, match="runtime mode"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_serving_gpu_count_mismatch(tmp_path):
    write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(tmp_path)

    with pytest.raises(BumkcArtifactError, match="GPU count"):
        summary.validate_scale_up_domain(gpu_count=4)


def test_rejects_invalid_bumkc_serving_gpu_count(tmp_path):
    write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(tmp_path)

    with pytest.raises(BumkcArtifactError, match="GPU count is invalid"):
        summary.validate_scale_up_domain(gpu_count=0)


def test_rejects_bumkc_target_architecture_mismatch(tmp_path):
    write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(tmp_path)

    summary.validate_target_architecture(target_arch="sm90")
    with pytest.raises(BumkcArtifactError, match="target architecture"):
        summary.validate_target_architecture(target_arch="sm80")


def test_derives_bumkc_fp8_serving_hints():
    quantization = bumkc_artifact.BumkcQuantizationSummary(
        scheme="FP8",
        scheme_hash=bumkc_artifact._stable_name_hash("FP8"),
        weight_format="fp8_e4m3",
        weight_format_code=2,
        weight_bits=8,
        weight_scale_layout=None,
        weight_scale_layout_code=0,
        weight_group_size=None,
        weight_scale_dtype=None,
        weight_scale_dtype_code=0,
        weight_symmetric=None,
        weight_symmetric_code=0,
        weight_zero_point=False,
        activation_bits=None,
        kv_bits=None,
        kv_format=None,
        kv_format_code=0,
        indexer_k_bits=None,
        indexer_k_format=None,
        indexer_k_format_code=0,
        gated_norm=False,
        spinquant=False,
        ignored_module_count=0,
    )

    hints = bumkc_artifact._build_serving_hints(quantization)

    assert hints.quantization == "fp8"
    assert hints.moe_runner_backend == "flashinfer_trtllm"


def test_leaves_unknown_bumkc_serving_hints_unset():
    quantization = bumkc_artifact.BumkcQuantizationSummary(
        scheme="INT4",
        scheme_hash=bumkc_artifact._stable_name_hash("INT4"),
        weight_format="int4",
        weight_format_code=4,
        weight_bits=4,
        weight_scale_layout=None,
        weight_scale_layout_code=0,
        weight_group_size=None,
        weight_scale_dtype=None,
        weight_scale_dtype_code=0,
        weight_symmetric=None,
        weight_symmetric_code=0,
        weight_zero_point=False,
        activation_bits=None,
        kv_bits=None,
        kv_format=None,
        kv_format_code=0,
        indexer_k_bits=None,
        indexer_k_format=None,
        indexer_k_format_code=0,
        gated_norm=False,
        spinquant=False,
        ignored_module_count=0,
    )

    hints = bumkc_artifact._build_serving_hints(quantization)

    assert hints.quantization is None
    assert hints.moe_runner_backend is None


def test_rejects_bumkc_identity_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["program_id"] = "program_other"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="program IDs"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_artifact_with_unchecked_fallback(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["fallback_mode"] = "disabled"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="checked fallback"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_cli_flag_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["cli_flags"] = ["--enable-bumkc"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="CLI flags"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unsupported_bumkc_schema(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = "bumkc.optimization_playground.v4"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="unsupported engine schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unsupported_bumkc_model_source_schema(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    source_path = plan_dir / "source" / "model-source.json"
    model_source = json.loads(source_path.read_text(encoding="utf-8"))
    model_source["schema_version"] = "bumkc.source.v0"
    source_path.write_text(json.dumps(model_source), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="model source schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_hvm_source_without_canonical_main(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    write_text(plan_dir / "source" / "hvm-core-book.hvm", "%main = 0\n")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="HVM Core source main"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_hvm_source_tensor_island_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    source_path = plan_dir / "source" / "hvm-core-book.hvm"
    source = source_path.read_text(encoding="utf-8")
    source_path.write_text(source.replace("#BumkcTensorIsland{}", "#Nil{}", 1))
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="tensor island count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_hvm_core_book_unknown_tensor_island(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    hvm_path = plan_dir / "ir" / "hvm-core-book.json"
    hvm_core_book = json.loads(hvm_path.read_text(encoding="utf-8"))
    hvm_core_book["regions"][0]["nodes"][1]["kind"]["tensor_island"]["island"] = (
        "island_missing"
    )
    hvm_path.write_text(json.dumps(hvm_core_book), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="tensor island is unknown"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_hvm_core_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    source_path = plan_dir / "source" / "model-source.json"
    model_source = json.loads(source_path.read_text(encoding="utf-8"))
    model_source["hvm_node_count"] = 3
    source_path.write_text(json.dumps(model_source), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="hvm_node_count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unsupported_bumkc_plan_schema(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    manifest_path = plan_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "bumkc.plan.v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="unsupported plan schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unsupported_bumkc_capability(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    manifest_path = plan_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capability_level"] = "scaffold"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="capability"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_manifest_schema_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["manifest_schema_version"] = "bumkc.plan.v0"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="manifest schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_manifest_capability_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["manifest_capability_level"] = "scaffold"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="manifest capability"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_source_schema_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["source_schema_version"] = "bumkc.source.v0"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="engine source schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_target_arch_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["target_arch"] = "sm80"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="target architecture"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_runtime_mode_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_mode"] = "production"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime mode"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_target_gpu_count_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["target_plan"]["gpu_count"] = 4
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="target GPU count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_scale_up_rank_count_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["scale_up_plan"]["rank_count"] = 4
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="scale-up rank count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_target_arch_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["target_plan"]["target_arch"] = "sm80"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="target architecture"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["kv_cache_binding_count"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_diagnostic_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["diagnostics_plan"]["heartbeat_slot_count"] = 1
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="diagnostic heartbeat"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_dependency_descriptor_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["dependency_plan"]["dependency_descriptors"][0]["task"] = "task_other"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="dependency descriptor hash"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_dependency_wait_scope_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    descriptor = runtime["dependency_plan"]["dependency_descriptors"][0]
    descriptor["wait_expression"] = "all_producer_tiles_complete"
    dependency_hash = bumkc_artifact._dependency_descriptor_hash(
        runtime["dependency_plan"]["dependency_descriptors"]
    )
    runtime["dependency_plan"]["dependency_descriptor_hash"] = dependency_hash
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["dependency_descriptor_hash"] = dependency_hash
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="wait expression"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_dependency_tensor_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["dependency_plan"]["dependency_tensor_count"] = 0
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="dependency tensor count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_dependency_tensor_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["task_descriptors"][1]["dependency_tensor_count"] = 0
    smoke["expected_dependency_tensor_count"] = 0
    populate_runtime_smoke_contracts(smoke)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime smoke mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_abi_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["runtime_abi_version"] = "bumkc.runtime.v0"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["runtime_abi_version"] = "bumkc.runtime.v0"
    populate_runtime_smoke_contracts(smoke)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime ABI"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_compiler_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["compiler_summary"]["event_notification_count"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="compiler summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_model_source_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    source_path = plan_dir / "source" / "model-source.json"
    model_source = json.loads(source_path.read_text(encoding="utf-8"))
    model_source["collective_island_count"] = 0
    source_path.write_text(json.dumps(model_source), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="model source summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_quantization_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    tensor_path = plan_dir / "ir" / "hvm-tensor-islands.json"
    tensor_islands = json.loads(tensor_path.read_text(encoding="utf-8"))
    tensor_islands["quantization"]["indexer_k_bits"] = 4
    tensor_path.write_text(json.dumps(tensor_islands), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="quantization_indexer_k_bits"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_quantization_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["quantization_summary"]["indexer_k_bits"] = 4
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(
        BumkcArtifactError,
        match="engine quantization summary mismatch: indexer_k_bits",
    ):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_engine_scale_up_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["scale_up_summary"]["runtime_rank_count"] = 4
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(
        BumkcArtifactError,
        match="engine scale-up summary mismatch: runtime_rank_count",
    ):
        load_bumkc_artifact(plan_dir)


def test_rejects_required_bumkc_engine_without_scale_up_summary(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    del engine["scale_up_summary"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="scale-up summary is missing"):
        load_bumkc_artifact(plan_dir)


def test_accepts_previous_bumkc_engine_without_scale_up_summary(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = bumkc_artifact.PREVIOUS_SCHEMA_VERSION
    del engine["scale_up_summary"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    summary = load_bumkc_artifact(plan_dir)

    assert summary.engine_schema_version == bumkc_artifact.PREVIOUS_SCHEMA_VERSION
    assert summary.scale_up_summary.runtime_rank_count == 8


def test_rejects_bumkc_engine_launch_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["launch_summary"]["default_bucketed_shape_value_sum"] = 1
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(
        BumkcArtifactError,
        match="engine launch summary mismatch: default_bucketed_shape_value_sum",
    ):
        load_bumkc_artifact(plan_dir)


def test_rejects_required_bumkc_engine_without_launch_summary(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    del engine["launch_summary"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="launch summary is missing"):
        load_bumkc_artifact(plan_dir)


def test_accepts_serving_hints_schema_bumkc_engine_without_launch_summary(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = bumkc_artifact.SERVING_HINTS_SCHEMA_VERSION
    del engine["launch_summary"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    summary = load_bumkc_artifact(plan_dir)

    assert summary.engine_schema_version == bumkc_artifact.SERVING_HINTS_SCHEMA_VERSION
    assert summary.launch_summary.default_bucketed_shape_value_sum == 16


def test_rejects_bumkc_engine_serving_hint_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["serving_hints"]["quantization"] = "fp8"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(
        BumkcArtifactError,
        match="engine serving hint mismatch: quantization",
    ):
        load_bumkc_artifact(plan_dir)


def test_rejects_required_bumkc_engine_without_serving_hints(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    del engine["serving_hints"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="serving hints are missing"):
        load_bumkc_artifact(plan_dir)


def test_accepts_scale_up_schema_bumkc_engine_without_serving_hints(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = bumkc_artifact.SCALE_UP_SCHEMA_VERSION
    del engine["serving_hints"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    summary = load_bumkc_artifact(plan_dir)

    assert summary.engine_schema_version == bumkc_artifact.SCALE_UP_SCHEMA_VERSION
    assert summary.serving_hints.quantization == "modelopt_fp4"


def test_accepts_legacy_bumkc_engine_without_quantization_summary(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = bumkc_artifact.LEGACY_SCHEMA_VERSION
    del engine["quantization_summary"]
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    summary = load_bumkc_artifact(plan_dir)

    assert summary.engine_schema_version == bumkc_artifact.LEGACY_SCHEMA_VERSION
    assert summary.quantization_summary.indexer_k_bits == 8


def test_rejects_bumkc_compiler_summary_non_integer(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["compiler_summary"]["event_notification_count"] = True
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="engine summary"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_side_effect_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    event_path = plan_dir / "ir" / "hvm-event-tensors.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    events["event_tensors"][1]["side_effects"] = ["kv_cache_write"]
    event_path.write_text(json.dumps(events), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="compiler summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_event_dependency_edge_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    event_path = plan_dir / "ir" / "hvm-event-tensors.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    events["event_tensors"][1]["predecessor_edges"] = []
    event_path.write_text(json.dumps(events), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="Event Tensor edge count"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_unknown_side_effect(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    island_path = plan_dir / "ir" / "hvm-tensor-islands.json"
    islands = json.loads(island_path.read_text(encoding="utf-8"))
    islands["islands"][1]["side_effects"] = ["eventual_consistency"]
    island_path.write_text(json.dumps(islands), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="side-effect marker"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_smoke_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["expected_event_execution_count"] = 7
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime smoke mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_smoke_benchmark_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["benchmark_iterations"] = 7
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime smoke mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_smoke_descriptor_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["task_descriptors"][1]["side_effect_code_sum"] = 4
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime smoke descriptor mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_smoke_contract_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["expected_source_contract_hash"] += 1
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime smoke contract mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unknown_bumkc_tensor_island_coverage(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    island_path = plan_dir / "ir" / "hvm-tensor-islands.json"
    islands = json.loads(island_path.read_text(encoding="utf-8"))
    islands["islands"][0]["coverage_status"] = "maybe_native"
    island_path.write_text(json.dumps(islands), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="coverage status"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_serving_state_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["serving_dependency_count"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_substitution_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["substitution_symbol_max_sum"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_digest_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["task_count"] = 3
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="digest"):
        load_bumkc_artifact(plan_dir)


def test_ignores_bumkc_writer_verify_report_drift(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    verify_path = plan_dir / "reports" / "verify.json"
    write_json(verify_path, {"passed": False})

    summary = load_bumkc_artifact(plan_dir)

    assert summary.plan_id == "plan_test"


def test_rejects_bumkc_runtime_launch_shape_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(plan_dir)

    with pytest.raises(BumkcArtifactError, match="outside"):
        summary.validate_runtime_launch(
            shape_symbols={"sequence": 8192},
            serving_state=[("sequence", "sequence"), ("decode_step", None)],
        )


def test_rejects_bumkc_runtime_launch_bucket_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["substitution_plan"]["shape_symbols"][0]["max"] = 4097
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["substitution_symbol_max_sum"] = 4097
    engine["launch_summary"]["shape_symbol_max_sum"] = 4097
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["expected_substitution_symbol_max_sum"] = 4097
    populate_runtime_smoke_contracts(smoke)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)
    summary = load_bumkc_artifact(plan_dir)

    with pytest.raises(BumkcArtifactError, match="buckets"):
        summary.validate_runtime_launch(
            shape_symbols={"sequence": 4097},
            serving_state=[("sequence", "sequence"), ("decode_step", None)],
        )


def test_rejects_bumkc_runtime_default_bucket_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["substitution_plan"]["shape_symbols"][0]["max"] = 4097
    runtime["substitution_plan"]["shape_symbols"][0]["default_value"] = 4097
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["substitution_symbol_max_sum"] = 4097
    engine["launch_summary"]["shape_symbol_max_sum"] = 4097
    engine["launch_summary"]["default_shape_value_sum"] = 4097
    engine["launch_summary"]["default_bucketed_shape_value_sum"] = 4112
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
    smoke_path = plan_dir / "generated" / "runtime-smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["expected_substitution_symbol_max_sum"] = 4097
    populate_runtime_smoke_contracts(smoke)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="default bucket"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_substitution_unknown_serving_kind(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["substitution_plan"]["serving_state"][0]["kind"] = "prefill_window"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="serving-state kind"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_substitution_unknown_symbol(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["substitution_plan"]["serving_state"][0]["symbol"] = "missing"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="serving-state symbol"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_launch_serving_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(plan_dir)

    with pytest.raises(BumkcArtifactError, match="missing BUMKC serving-state"):
        summary.validate_runtime_launch(
            shape_symbols={"sequence": 17},
            serving_state=[("sequence", "sequence")],
        )


def test_rejects_bumkc_runtime_launch_duplicate_serving_state(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    summary = load_bumkc_artifact(plan_dir)

    with pytest.raises(BumkcArtifactError, match="duplicated"):
        summary.validate_runtime_launch(
            shape_symbols={"sequence": 17},
            serving_state=[
                ("sequence", "sequence"),
                ("sequence", "sequence"),
                ("decode_step", None),
            ],
        )


def write_bumkc_artifact(tmp_path, *, executable):
    plan_id = "plan_test"
    program_id = "program_test"
    plan_dir = tmp_path / plan_id
    (plan_dir / "ir").mkdir(parents=True)
    (plan_dir / "runtime").mkdir(parents=True)
    (plan_dir / "engine").mkdir()
    (plan_dir / "generated").mkdir()
    (plan_dir / "reports").mkdir()
    (plan_dir / "source").mkdir()
    dependency_descriptors = [
        {
            "task": "task_second",
            "consumer_event": "event_second",
            "predecessor_event": "event_first",
            "dependency_ordinal": 0,
            "tensors": ["tensor_hidden"],
            "scope": "tile_overlap",
            "wait_expression": "same_logical_tile_ready",
        }
    ]
    dependency_descriptor_hash = bumkc_artifact._dependency_descriptor_hash(
        dependency_descriptors
    )
    quantization = {
        "scheme": "W4A4KV4+IndexerK8",
        "weight_format": "nv_fp4",
        "weight_bits": 4,
        "weight_scale_layout": "tensor_group",
        "weight_group_size": 16,
        "weight_scale_dtype": "fp8_e4m3",
        "weight_symmetric": True,
        "weight_zero_point": False,
        "activation_bits": 4,
        "kv_bits": 4,
        "kv_format": "nv_fp4",
        "indexer_k_bits": 8,
        "indexer_k_format": "fp8_e4m3",
        "gated_norm": True,
        "spinquant": True,
        "ignored_module_count": 2,
    }
    quantization_scheme_hash = bumkc_artifact._stable_name_hash(quantization["scheme"])

    write_json(
        plan_dir / "manifest.json",
        {
            "schema_version": REQUIRED_PLAN_SCHEMA_VERSION,
            "capability_level": REQUIRED_CAPABILITY_LEVEL,
            "plan_id": plan_id,
            "program_id": program_id,
            "source": {
                "frontend": "internal_test",
                "model": "matmul-chain",
            },
            "gpu_count": 8,
            "target_arch": "sm90",
            "engine": "sglang",
            "engine_profile": "optimization_playground",
            "fallback_mode": "checked",
            "runtime_mode": "debug",
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-tensor-islands.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "quantization": quantization,
            "shape_symbols": [
                {
                    "id": "sequence",
                    "min": 1,
                    "max": 4096,
                    "bucket": 16,
                }
            ],
            "islands": [
                {
                    "id": "island_native",
                    "operator": "matmul",
                    "coverage_status": "native_eligible",
                    "communication": None,
                    "serving_state": [],
                    "side_effects": [],
                },
                {
                    "id": "island_collective",
                    "operator": "collective",
                    "coverage_status": "native_eligible",
                    "communication": {
                        "kind": "all_reduce",
                        "group_size": 8,
                    },
                    "serving_state": [
                        {
                            "kind": "sequence",
                            "symbol": "sequence",
                        },
                        {
                            "kind": "decode_step",
                        },
                    ],
                    "side_effects": ["collective"],
                },
                {
                    "id": "island_fallback",
                    "operator": "unknown",
                    "coverage_status": "fallback_only",
                    "communication": None,
                    "serving_state": [],
                    "side_effects": [],
                },
            ],
            "fallback_bridges": [
                {
                    "tensor": "tensor_hidden",
                    "producer_island": "island_fallback",
                    "consumer_island": "island_native",
                    "direction": "fallback_to_native",
                }
            ],
        },
    )
    write_text(
        plan_dir / "source" / "hvm-core-book.hvm",
        "\n".join(
            [
                "@main = @bumkc_program_program_test",
                "",
                (
                    "@bumkc_program_program_test = "
                    "#BumkcProgram{@bumkc_region_region_test_root, "
                    "@bumkc_regions_program_test, "
                    "@bumkc_tensor_islands_program_test, "
                    "@bumkc_shape_symbols_program_test, "
                    "@bumkc_fallback_bridges_program_test, "
                    "@bumkc_tensor_descriptors_program_test, "
                    "@bumkc_quantization_program_test}"
                ),
                "@bumkc_tensor_island_island_native = #BumkcTensorIsland{}",
                "@bumkc_tensor_island_island_collective = #BumkcTensorIsland{}",
                "@bumkc_tensor_island_island_fallback = #BumkcTensorIsland{}",
                "@bumkc_shape_symbol_sequence = #BumkcShapeSymbol{1, 4096, 16}",
                (
                    "@bumkc_fallback_bridge_island_fallback_island_native_tensor_hidden "
                    "= #BumkcFallbackBridge{}"
                ),
                "@bumkc_quantization_program_test = #BumkcQuantization{}",
                "",
            ]
        ),
    )
    write_json(
        plan_dir / "source" / "model-source.json",
        {
            "schema_version": REQUIRED_SOURCE_SCHEMA_VERSION,
            "plan_id": plan_id,
            "program_id": program_id,
            "source": {
                "frontend": "internal_test",
                "model": "matmul-chain",
            },
            "gpu_count": 8,
            "target_arch": "sm90",
            "engine": "sglang",
            "engine_profile": "optimization_playground",
            "fallback_mode": "checked",
            "runtime_mode": "debug",
            "hvm_core_book_source_path": "source/hvm-core-book.hvm",
            "hvm_capture_status": "native_eligible",
            "hvm_region_count": 1,
            "hvm_node_count": 4,
            "hvm_model_entry_node_count": 1,
            "hvm_tensor_island_node_count": 3,
            "hvm_fallback_boundary_node_count": 0,
            "tensor_island_count": 3,
            "native_eligible_island_count": 2,
            "fallback_island_count": 1,
            "fallback_bridge_count": 1,
            "side_effecting_island_count": 1,
            "collective_island_count": 1,
            "moe_dispatch_island_count": 0,
            "serving_state_island_count": 1,
            "serving_state_dependency_count": 2,
            "serving_state_kind_code_sum": 5,
            "serving_state_symbol_count": 1,
            "shape_symbol_count": 1,
            "shape_symbol_max_sum": 4096,
            "shape_symbol_bucket_sum": 16,
            "quantization_scheme_hash": quantization_scheme_hash,
            "quantization_weight_format_code": 1,
            "quantization_weight_bits": 4,
            "quantization_weight_scale_layout_code": 2,
            "quantization_weight_group_size": 16,
            "quantization_weight_scale_dtype_code": 7,
            "quantization_weight_symmetric_code": 1,
            "quantization_weight_zero_point_enabled": False,
            "quantization_activation_bits": 4,
            "quantization_kv_bits": 4,
            "quantization_kv_format_code": 1,
            "quantization_indexer_k_bits": 8,
            "quantization_indexer_k_format_code": 2,
            "gated_norm_enabled": True,
            "spinquant_enabled": True,
            "quantization_ignored_module_count": 2,
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-core-book.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "source": {
                "frontend": "internal_test",
                "model": "matmul-chain",
            },
            "root_region": "region_test_root",
            "regions": [
                {
                    "id": "region_test_root",
                    "kind": "root",
                    "nodes": [
                        {
                            "id": "node_model_entry",
                            "kind": {
                                "model_entry": {
                                    "model": "matmul-chain",
                                }
                            },
                        },
                        {
                            "id": "node_native",
                            "kind": {
                                "tensor_island": {
                                    "island": "island_native",
                                }
                            },
                        },
                        {
                            "id": "node_collective",
                            "kind": {
                                "tensor_island": {
                                    "island": "island_collective",
                                }
                            },
                        },
                        {
                            "id": "node_fallback",
                            "kind": {
                                "tensor_island": {
                                    "island": "island_fallback",
                                }
                            },
                        },
                    ],
                }
            ],
            "coverage_status": "native_eligible",
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-block-role-pipelines.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "block_ops": [
                {"operator": "matmul", "side_effects": []},
                {"operator": "collective", "side_effects": ["collective"]},
            ],
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-event-tensors.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "event_tensors": [
                {
                    "operator": "matmul",
                    "side_effects": [],
                    "predecessor_events": [],
                    "predecessor_edges": [],
                    "successor_events": ["event_second"],
                },
                {
                    "operator": "collective",
                    "side_effects": ["collective"],
                    "predecessor_events": ["event_first"],
                    "predecessor_edges": [
                        {"tensors": ["tensor_hidden"]},
                    ],
                    "successor_events": [],
                },
            ],
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-sm-task-runtime.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
        },
    )
    write_json(
        plan_dir / "reports" / "simulation.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "notification_count": 1,
            "execution_order": ["event_first", "event_second"],
            "violations": [],
        },
    )
    write_json(
        plan_dir / "reports" / "cpu-reference.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
        },
    )
    write_json(
        plan_dir / "runtime" / "plan.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "runtime_abi_version": REQUIRED_RUNTIME_ABI_VERSION,
            "executable": executable,
            "entrypoints": (
                [{"name": "cuda_tensor_smoke", "symbol": "bumkc_tensor_smoke"}]
                if executable
                else []
            ),
            "task_count": 2,
            "target_plan": {
                "gpu_count": 8,
                "target_arch": "sm90",
                "worker_count": 64,
                "scheduler_count": 8,
            },
            "execution_model": {
                "conventional_launch_count": 2,
                "persistent_launch_count": 1,
                "launch_reduction_per_mille": 2000,
            },
            "queue_plan": {
                "worker_queue_count": 64,
                "scheduler_queue_count": 8,
                "jit_task_count": 0,
                "aot_task_count": 2,
                "queue_capacity": 64,
                "task_instance_capacity": 2,
                "event_counter_count": 2,
            },
            "memory_plan": {
                "device_global_binding_count": 4,
                "kv_cache_binding_count": 1,
            },
            "scale_up_plan": {
                "rank_count": 8,
                "task_rank_group_count": 2,
                "task_rank_reference_count": 16,
                "rank_id_sum": 56,
            },
            "dependency_plan": {
                "task_dependency_count": 1,
                "dependency_tensor_count": 1,
                "tile_overlap_dependency_count": 1,
                "whole_producer_dependency_count": 0,
                "dependency_scope_code_sum": 1,
                "dependency_descriptor_count": len(dependency_descriptors),
                "dependency_descriptor_hash": dependency_descriptor_hash,
                "dependency_descriptors": dependency_descriptors,
            },
            "communication_plan": {
                "collective_task_count": 1,
                "collective_group_size_sum": 8,
                "collective_kind_code_sum": 1,
            },
            "side_effect_plan": {
                "side_effecting_task_count": 1,
                "task_side_effect_count": 1,
                "task_side_effect_code_sum": 3,
            },
            "serving_state_plan": {
                "serving_task_count": 1,
                "serving_dependency_count": 2,
                "serving_kind_code_sum": 5,
                "serving_symbol_count": 1,
            },
            "substitution_plan": {
                "shape_symbols": [
                    {
                        "symbol": "sequence",
                        "min": 1,
                        "max": 4096,
                        "bucket": 16,
                        "default_value": 1,
                    }
                ],
                "serving_state": [
                    {
                        "kind": "sequence",
                        "symbol": "sequence",
                        "required": True,
                    },
                    {
                        "kind": "decode_step",
                        "required": True,
                    },
                ],
            },
            "diagnostics_plan": {
                "heartbeat_slot_count": 72,
                "worker_heartbeat_slot_count": 64,
                "scheduler_heartbeat_slot_count": 8,
                "queue_snapshot_slot_count": 72,
                "event_counter_snapshot_count": 2,
                "last_completed_task_slot_count": 64,
                "blocked_event_slot_count": 8,
                "watchdog_poll_interval_us": 1000,
                "watchdog_timeout_us": 30_000_000,
            },
        },
    )
    write_json(
        plan_dir / "engine" / "optimization-playground.json",
        {
            "schema_version": REQUIRED_SCHEMA_VERSION,
            "engine": "sglang",
            "engine_profile": "optimization_playground",
            "integration_branch": "bumkc/serving-integration",
            "plan_id": plan_id,
            "program_id": program_id,
            "manifest_schema_version": REQUIRED_PLAN_SCHEMA_VERSION,
            "manifest_capability_level": REQUIRED_CAPABILITY_LEVEL,
            "source_schema_version": REQUIRED_SOURCE_SCHEMA_VERSION,
            "model": "matmul-chain",
            "gpu_count": 8,
            "target_arch": "sm90",
            "fallback_mode": "checked",
            "runtime_mode": "debug",
            "runtime_executable": executable,
            "cli_flags": [
                "--enable-bumkc",
                "--bumkc-plan-path <artifact-root>/<plan-id>",
                "--bumkc-fallback-mode checked",
            ],
            "compiler_summary": {
                "tensor_island_count": 3,
                "native_tensor_island_count": 2,
                "fallback_tensor_island_count": 1,
                "side_effecting_tensor_island_count": 1,
                "tensor_island_side_effect_count": 1,
                "tensor_island_side_effect_code_sum": 3,
                "fallback_bridge_count": 1,
                "moe_dispatch_tensor_island_count": 0,
                "block_op_count": 2,
                "moe_dispatch_block_op_count": 0,
                "side_effecting_block_op_count": 1,
                "block_side_effect_count": 1,
                "block_side_effect_code_sum": 3,
                "event_tensor_count": 2,
                "side_effecting_event_tensor_count": 1,
                "event_side_effect_count": 1,
                "event_side_effect_code_sum": 3,
                "moe_dispatch_event_tensor_count": 0,
                "event_predecessor_edge_count": 1,
                "event_successor_edge_count": 1,
                "event_dependency_tensor_count": 1,
                "event_notification_count": 1,
                "event_execution_count": 2,
                "event_simulation_violation_count": 0,
            },
            "runtime_summary": {
                "task_count": 2,
                "conventional_launch_count": 2,
                "persistent_launch_count": 1,
                "jit_task_count": 0,
                "aot_task_count": 2,
                "queue_capacity": 64,
                "task_instance_capacity": 2,
                "device_global_binding_count": 4,
                "kv_cache_binding_count": 1,
                "rank_count": 8,
                "task_rank_group_count": 2,
                "task_rank_reference_count": 16,
                "rank_id_sum": 56,
                "task_dependency_count": 1,
                "dependency_tensor_count": 1,
                "tile_overlap_dependency_count": 1,
                "whole_producer_dependency_count": 0,
                "dependency_scope_code_sum": 1,
                "dependency_descriptor_count": len(dependency_descriptors),
                "dependency_descriptor_hash": dependency_descriptor_hash,
                "collective_task_count": 1,
                "collective_group_size_sum": 8,
                "collective_kind_code_sum": 1,
                "side_effecting_task_count": 1,
                "task_side_effect_count": 1,
                "task_side_effect_code_sum": 3,
                "serving_task_count": 1,
                "serving_dependency_count": 2,
                "serving_kind_code_sum": 5,
                "serving_symbol_count": 1,
                "substitution_shape_symbol_count": 1,
                "substitution_serving_binding_count": 2,
                "substitution_symbol_max_sum": 4096,
                "substitution_symbol_bucket_sum": 16,
                "diagnostic_heartbeat_slot_count": 72,
                "diagnostic_queue_snapshot_slot_count": 72,
                "diagnostic_event_counter_snapshot_count": 2,
                "diagnostic_last_completed_task_slot_count": 64,
                "diagnostic_blocked_event_slot_count": 8,
                "watchdog_poll_interval_us": 1000,
                "watchdog_timeout_us": 30_000_000,
            },
            "scale_up_summary": {
                "compile_gpu_count": 8,
                "runtime_target_gpu_count": 8,
                "runtime_rank_count": 8,
                "target_arch": "sm90",
                "target_arch_code": 90,
                "worker_count": 64,
                "scheduler_count": 8,
                "task_rank_group_count": 2,
                "task_rank_reference_count": 16,
                "rank_id_sum": 56,
                "collective_task_count": 1,
                "collective_group_size_sum": 8,
                "collective_kind_code_sum": 1,
            },
            "launch_summary": {
                "shape_symbol_count": 1,
                "shape_symbol_min_sum": 1,
                "shape_symbol_max_sum": 4096,
                "shape_symbol_bucket_sum": 16,
                "default_shape_value_sum": 1,
                "default_bucketed_shape_value_sum": 16,
                "serving_binding_count": 2,
                "required_serving_binding_count": 2,
                "optional_serving_binding_count": 0,
                "serving_kind_code_sum": 5,
                "serving_symbol_count": 1,
            },
            "quantization_summary": {
                "scheme": "W4A4KV4+IndexerK8",
                "scheme_hash": quantization_scheme_hash,
                "weight_format": "nv_fp4",
                "weight_format_code": 1,
                "weight_bits": 4,
                "weight_scale_layout": "tensor_group",
                "weight_scale_layout_code": 2,
                "weight_group_size": 16,
                "weight_scale_dtype": "fp8_e4m3",
                "weight_scale_dtype_code": 7,
                "weight_symmetric": True,
                "weight_symmetric_code": 1,
                "weight_zero_point": False,
                "activation_bits": 4,
                "kv_bits": 4,
                "kv_format": "nv_fp4",
                "kv_format_code": 1,
                "indexer_k_bits": 8,
                "indexer_k_format": "fp8_e4m3",
                "indexer_k_format_code": 2,
                "gated_norm": True,
                "spinquant": True,
                "ignored_module_count": 2,
            },
            "serving_hints": {
                "quantization": "modelopt_fp4",
                "moe_runner_backend": "flashinfer_trtllm",
            },
            "preserve_custom_optimizations": True,
            "required_validation_model": REQUIRED_VALIDATION_MODEL,
            "artifact_paths": {
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
            },
        },
    )
    runtime_smoke = {
        "schema_version": REQUIRED_RUNTIME_SMOKE_SCHEMA_VERSION,
        "plan_id": plan_id,
        "program_id": program_id,
        "runtime_abi_version": REQUIRED_RUNTIME_ABI_VERSION,
        "source_path": "generated/runtime_smoke.cu",
        "binary_name": "runtime_smoke",
        "expected_task_count": 2,
        "expected_conventional_launch_count": 2,
        "expected_persistent_launch_count": 1,
        "expected_launch_reduction_per_mille": 2000,
        "benchmark_enabled": 1,
        "benchmark_iterations": 8,
        "benchmark_conventional_launch_count": 2,
        "benchmark_persistent_launch_count": 1,
        "benchmark_total_conventional_launches": 16,
        "benchmark_total_persistent_launches": 8,
        "expected_jit_task_count": 0,
        "expected_aot_task_count": 2,
        "expected_queue_capacity": 64,
        "expected_task_instance_capacity": 2,
        "expected_predecessor_event_count_sum": 1,
        "expected_dependency_edge_count": 1,
        "expected_dependency_tensor_count": 1,
        "expected_dependency_scope_code_sum": 1,
        "expected_launch_domain_rank_sum": 4,
        "expected_launch_domain_element_sum": 2,
        "expected_operator_code_sum": 10,
        "expected_kv_cache_binding_count": 1,
        "expected_communication_task_count": 1,
        "expected_communication_group_size_sum": 8,
        "expected_communication_kind_code_sum": 1,
        "expected_side_effecting_task_count": 1,
        "expected_task_side_effect_count": 1,
        "expected_task_side_effect_code_sum": 3,
        "expected_serving_task_count": 1,
        "expected_serving_dependency_count": 2,
        "expected_serving_kind_code_sum": 5,
        "expected_serving_symbol_count": 1,
        "expected_substitution_shape_symbol_count": 1,
        "expected_substitution_serving_binding_count": 2,
        "expected_substitution_symbol_max_sum": 4096,
        "expected_substitution_symbol_bucket_sum": 16,
        "expected_rank_group_size_sum": 16,
        "expected_rank_id_sum": 56,
        "expected_event_tensor_count": 2,
        "expected_event_predecessor_edge_count": 1,
        "expected_event_successor_edge_count": 1,
        "expected_event_notification_count": 1,
        "expected_event_execution_count": 2,
        "expected_event_simulation_violation_count": 0,
        "expected_diagnostic_heartbeat_slot_count": 72,
        "expected_diagnostic_queue_snapshot_slot_count": 72,
        "expected_diagnostic_event_counter_snapshot_count": 2,
        "expected_diagnostic_last_completed_task_slot_count": 64,
        "expected_diagnostic_blocked_event_slot_count": 8,
        "expected_watchdog_poll_interval_us": 1000,
        "expected_watchdog_timeout_us": 30_000_000,
        "event_descriptors": [
            {
                "ordinal": 0,
                "event_id": "event_first",
                "predecessor_event_count": 0,
                "successor_event_count": 1,
            },
            {
                "ordinal": 1,
                "event_id": "event_second",
                "predecessor_event_count": 1,
                "successor_event_count": 0,
            },
        ],
        "task_descriptors": [
            {
                "ordinal": 0,
                "task_id": "task_first",
                "source_event_tensor": "event_first",
                "operator": "matmul",
                "scheduling_policy": "static_aot",
                "predecessor_event_count": 0,
                "dependency_edge_count": 0,
                "dependency_tensor_count": 0,
                "dependency_scope_code_sum": 0,
                "launch_domain_rank": 2,
                "launch_domain_elements": 1,
                "kv_cache_binding_count": 0,
                "communication_kind_code": 0,
                "communication_group_size": 0,
                "side_effect_count": 0,
                "side_effect_code_sum": 0,
                "serving_dependency_count": 0,
                "serving_kind_code_sum": 0,
                "serving_symbol_count": 0,
                "rank_group_size": 8,
                "rank_id_sum": 28,
            },
            {
                "ordinal": 1,
                "task_id": "task_second",
                "source_event_tensor": "event_second",
                "operator": "collective",
                "scheduling_policy": "static_aot",
                "predecessor_event_count": 1,
                "dependency_edge_count": 1,
                "dependency_tensor_count": 1,
                "dependency_scope_code_sum": 1,
                "launch_domain_rank": 2,
                "launch_domain_elements": 1,
                "kv_cache_binding_count": 1,
                "communication_kind_code": 1,
                "communication_group_size": 8,
                "side_effect_count": 1,
                "side_effect_code_sum": 3,
                "serving_dependency_count": 2,
                "serving_kind_code_sum": 5,
                "serving_symbol_count": 1,
                "rank_group_size": 8,
                "rank_id_sum": 28,
            },
        ],
    }
    populate_runtime_smoke_contracts(runtime_smoke)
    write_json(plan_dir / "generated" / "runtime-smoke.json", runtime_smoke)
    write_text(
        plan_dir / "generated" / "runtime_smoke.cu",
        "int main() { return 0; }\n",
    )
    write_json(
        plan_dir / "generated" / "tensor-smoke.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "enabled": executable,
        },
    )
    write_text(plan_dir / "generated" / "tensor_smoke.cu", "int main() { return 0; }\n")
    refresh_bumkc_digests(plan_dir)
    return plan_dir


def write_text(path, value):
    path.write_text(value, encoding="utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def set_bumkc_runtime_mode(plan_dir, runtime_mode):
    for path in (
        plan_dir / "manifest.json",
        plan_dir / "source" / "model-source.json",
        plan_dir / "engine" / "optimization-playground.json",
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["runtime_mode"] = runtime_mode
        write_json(path, data)
    refresh_bumkc_digests(plan_dir)


def populate_runtime_smoke_contracts(smoke):
    descriptor_hash = bumkc_artifact._runtime_smoke_descriptor_contract_hash(
        smoke["event_descriptors"],
        smoke["task_descriptors"],
    )
    smoke["expected_schema_hash"] = bumkc_artifact._contract_hash_str(
        smoke["schema_version"]
    )
    smoke["expected_runtime_abi_hash"] = bumkc_artifact._contract_hash_str(
        smoke["runtime_abi_version"]
    )
    smoke["expected_plan_id_hash"] = bumkc_artifact._contract_hash_str(smoke["plan_id"])
    smoke["expected_program_id_hash"] = bumkc_artifact._contract_hash_str(
        smoke["program_id"]
    )
    smoke["expected_descriptor_contract_hash"] = descriptor_hash
    smoke["expected_source_contract_hash"] = (
        bumkc_artifact._runtime_smoke_source_contract_hash(
            smoke,
            descriptor_hash,
        )
    )


def refresh_bumkc_digests(plan_dir):
    files = []
    digest_path = plan_dir / "reports" / "artifact-digests.json"
    verify_path = plan_dir / "reports" / "verify.json"
    for path in sorted(plan_dir.rglob("*")):
        if path.is_dir() or path in (digest_path, verify_path):
            continue
        contents = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(plan_dir).as_posix(),
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    write_json(
        digest_path,
        {
            "schema_version": "bumkc.artifact_digests.v1",
            "plan_id": "plan_test",
            "files": files,
        },
    )
