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
    assert summary.compiler_summary.block_op_count == 2
    assert summary.compiler_summary.side_effecting_block_op_count == 1
    assert summary.compiler_summary.block_side_effect_count == 1
    assert summary.compiler_summary.block_side_effect_code_sum == 3
    assert summary.compiler_summary.event_tensor_count == 2
    assert summary.compiler_summary.side_effecting_event_tensor_count == 1
    assert summary.compiler_summary.event_side_effect_count == 1
    assert summary.compiler_summary.event_side_effect_code_sum == 3
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
    assert summary.runtime_shape_symbols[0].symbol == "sequence"
    assert summary.runtime_serving_state[0].key() == ("sequence", "sequence")
    assert summary.target_arch == "sm90"
    assert summary.plan_schema_version == REQUIRED_PLAN_SCHEMA_VERSION
    assert summary.capability_level == REQUIRED_CAPABILITY_LEVEL
    assert summary.engine_schema_version == REQUIRED_SCHEMA_VERSION
    assert summary.task_count == summary.runtime_summary.task_count
    assert summary.tensor_smoke_enabled
    assert summary.artifact_digest_count == 15
    assert summary.fallback_mode == "checked"
    log_dict = summary.as_log_dict()
    assert log_dict["plan_schema_version"] == REQUIRED_PLAN_SCHEMA_VERSION
    assert log_dict["capability_level"] == REQUIRED_CAPABILITY_LEVEL
    assert log_dict["engine_schema_version"] == REQUIRED_SCHEMA_VERSION

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


def test_rejects_bumkc_target_arch_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["target_arch"] = "sm80"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")
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


def test_rejects_bumkc_runtime_dependency_descriptor_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    runtime_path = plan_dir / "runtime" / "plan.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["dependency_plan"]["dependency_descriptors"][0][
        "wait_expression"
    ] = "bad_wait_expression"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    refresh_bumkc_digests(plan_dir)

    with pytest.raises(BumkcArtifactError, match="dependency descriptor hash"):
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

    write_json(
        plan_dir / "manifest.json",
        {
            "schema_version": REQUIRED_PLAN_SCHEMA_VERSION,
            "capability_level": REQUIRED_CAPABILITY_LEVEL,
            "plan_id": plan_id,
            "program_id": program_id,
            "target_arch": "sm90",
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-tensor-islands.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "islands": [
                {"coverage_status": "native_eligible", "side_effects": []},
                {"coverage_status": "native_eligible", "side_effects": ["collective"]},
                {"coverage_status": "fallback_only", "side_effects": []},
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
    write_text(plan_dir / "source" / "hvm-core-book.hvm", "%main = 0\n")
    write_json(
        plan_dir / "ir" / "hvm-core-book.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-block-role-pipelines.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "block_ops": [{"side_effects": []}, {"side_effects": ["collective"]}],
        },
    )
    write_json(
        plan_dir / "ir" / "hvm-event-tensors.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "event_tensors": [
                {
                    "side_effects": [],
                    "predecessor_events": [],
                    "predecessor_edges": [],
                    "successor_events": ["event_second"],
                },
                {
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
            "execution_model": {
                "conventional_launch_count": 2,
                "persistent_launch_count": 1,
                "launch_reduction_per_mille": 2000,
            },
            "queue_plan": {
                "jit_task_count": 0,
                "aot_task_count": 2,
                "queue_capacity": 64,
                "task_instance_capacity": 2,
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
            "model": "matmul-chain",
            "gpu_count": 8,
            "target_arch": "sm90",
            "fallback_mode": "checked",
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
                "block_op_count": 2,
                "side_effecting_block_op_count": 1,
                "block_side_effect_count": 1,
                "block_side_effect_code_sum": 3,
                "event_tensor_count": 2,
                "side_effecting_event_tensor_count": 1,
                "event_side_effect_count": 1,
                "event_side_effect_code_sum": 3,
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
    smoke["expected_plan_id_hash"] = bumkc_artifact._contract_hash_str(
        smoke["plan_id"]
    )
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
    for path in sorted(plan_dir.rglob("*")):
        if path.is_dir() or path == digest_path:
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
            "schema_version": "bumkc.artifact_digests.v0",
            "plan_id": "plan_test",
            "files": files,
        },
    )
