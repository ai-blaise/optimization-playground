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
BumkcArtifactError = bumkc_artifact.BumkcArtifactError
load_bumkc_artifact = bumkc_artifact.load_bumkc_artifact


def test_loads_executable_bumkc_artifact(tmp_path):
    plan_dir = write_bumkc_artifact(tmp_path, executable=True)

    summary = load_bumkc_artifact(tmp_path, require_executable=True)

    assert summary.root == plan_dir
    assert summary.runtime_executable
    assert summary.runtime_entrypoints == ("cuda_tensor_smoke",)
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
        },
    )
    write_json(
        plan_dir / "engine" / "optimization-playground.json",
        {
            "schema_version": "bumkc.optimization_playground.v0",
            "engine": "sglang",
            "engine_profile": "optimization_playground",
            "integration_branch": "bumkc/serving-integration",
            "plan_id": plan_id,
            "program_id": program_id,
            "model": "matmul-chain",
            "gpu_count": 8,
            "fallback_mode": "checked",
            "runtime_executable": executable,
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
