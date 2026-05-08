import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[2] / "python/sglang/srt/bumkc/artifact.py"
SPEC = importlib.util.spec_from_file_location("bumkc_artifact_under_test", MODULE_PATH)
bumkc_artifact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bumkc_artifact
SPEC.loader.exec_module(bumkc_artifact)

REQUIRED_VALIDATION_MODEL = bumkc_artifact.REQUIRED_VALIDATION_MODEL
REQUIRED_SCHEMA_VERSION = bumkc_artifact.REQUIRED_SCHEMA_VERSION
BumkcArtifactError = bumkc_artifact.BumkcArtifactError
load_bumkc_artifact = bumkc_artifact.load_bumkc_artifact


def test_loads_executable_bumkc_artifact(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)

    summary = load_bumkc_artifact(tmp_path, require_executable=True)

    assert summary.root == plan_dir
    assert summary.runtime_executable
    assert summary.runtime_entrypoints == ("cuda_tensor_smoke",)
    assert summary.runtime_summary.task_instance_capacity == 2
    assert summary.runtime_summary.kv_cache_binding_count == 1
    assert summary.runtime_summary.rank_count == 8
    assert summary.runtime_summary.task_rank_reference_count == 16
    assert summary.runtime_summary.task_dependency_count == 1
    assert summary.runtime_summary.dependency_scope_code_sum == 1
    assert summary.runtime_summary.collective_task_count == 1
    assert summary.runtime_summary.collective_group_size_sum == 8
    assert summary.runtime_summary.collective_kind_code_sum == 1
    assert summary.runtime_summary.serving_task_count == 1
    assert summary.runtime_summary.serving_dependency_count == 2
    assert summary.runtime_summary.serving_kind_code_sum == 5
    assert summary.runtime_summary.serving_symbol_count == 1
    assert summary.runtime_summary.substitution_shape_symbol_count == 1
    assert summary.runtime_summary.substitution_serving_binding_count == 2
    assert summary.runtime_summary.substitution_symbol_max_sum == 4096
    assert summary.runtime_summary.substitution_symbol_bucket_sum == 16
    assert summary.target_arch == "sm90"
    assert summary.task_count == summary.runtime_summary.task_count
    assert summary.tensor_smoke_enabled
    assert summary.fallback_mode == "checked"


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

    with pytest.raises(BumkcArtifactError, match="program IDs"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_artifact_with_unchecked_fallback(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["fallback_mode"] = "disabled"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="checked fallback"):
        load_bumkc_artifact(plan_dir)


def test_rejects_unsupported_bumkc_schema(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["schema_version"] = "bumkc.optimization_playground.v4"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="unsupported engine schema"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_target_arch_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["target_arch"] = "sm80"
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="target architecture"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_runtime_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["kv_cache_binding_count"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_serving_state_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["serving_dependency_count"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def test_rejects_bumkc_substitution_summary_mismatch(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)
    engine_path = plan_dir / "engine" / "optimization-playground.json"
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    engine["runtime_summary"]["substitution_symbol_max_sum"] = 7
    engine_path.write_text(json.dumps(engine), encoding="utf-8")

    with pytest.raises(BumkcArtifactError, match="runtime summary mismatch"):
        load_bumkc_artifact(plan_dir)


def write_bumkc_artifact(tmp_path, *, executable):
    plan_id = "plan_test"
    program_id = "program_test"
    plan_dir = tmp_path / plan_id
    (plan_dir / "runtime").mkdir(parents=True)
    (plan_dir / "engine").mkdir()
    (plan_dir / "generated").mkdir()

    write_json(
        plan_dir / "manifest.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "target_arch": "sm90",
        },
    )
    write_json(
        plan_dir / "runtime" / "plan.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
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
                "tile_overlap_dependency_count": 1,
                "whole_producer_dependency_count": 0,
                "dependency_scope_code_sum": 1,
            },
            "communication_plan": {
                "collective_task_count": 1,
                "collective_group_size_sum": 8,
                "collective_kind_code_sum": 1,
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
            "model": "matmul-chain",
            "gpu_count": 8,
            "target_arch": "sm90",
            "fallback_mode": "checked",
            "runtime_executable": executable,
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
                "tile_overlap_dependency_count": 1,
                "whole_producer_dependency_count": 0,
                "dependency_scope_code_sum": 1,
                "collective_task_count": 1,
                "collective_group_size_sum": 8,
                "collective_kind_code_sum": 1,
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
        },
    )
    write_json(
        plan_dir / "generated" / "tensor-smoke.json",
        {
            "plan_id": plan_id,
            "program_id": program_id,
            "enabled": executable,
        },
    )
    return plan_dir


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
