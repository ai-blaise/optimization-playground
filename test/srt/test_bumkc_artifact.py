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
REQUIRED_SCHEMA_VERSION = bumkc_artifact.REQUIRED_SCHEMA_VERSION
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
    assert summary.runtime_shape_symbols[0].symbol == "sequence"
    assert summary.runtime_serving_state[0].key() == ("sequence", "sequence")
    assert summary.target_arch == "sm90"
    assert summary.task_count == summary.runtime_summary.task_count
    assert summary.tensor_smoke_enabled
    assert summary.artifact_digest_count == 4
    assert summary.fallback_mode == "checked"

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
    (plan_dir / "runtime").mkdir(parents=True)
    (plan_dir / "engine").mkdir()
    (plan_dir / "generated").mkdir()
    (plan_dir / "reports").mkdir()

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
            "cli_flags": [
                "--enable-bumkc",
                "--bumkc-plan-path <artifact-root>/<plan-id>",
                "--bumkc-fallback-mode checked",
            ],
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
            "artifact_paths": {
                "artifact_digests": "reports/artifact-digests.json",
            },
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
    refresh_bumkc_digests(plan_dir)
    return plan_dir


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


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
