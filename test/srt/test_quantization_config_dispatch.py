"""Tests for ``apply_quantization_config_dispatch``.

These tests exercise the config-side dispatcher in isolation by loading
``quantization_config_dispatch.py`` directly. They do not require a GPU
or a running SGLang server; the dispatcher only mutates a fake
``server_args`` object based on a fake ``hf_config``.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Stub out parent packages and the (heavy) ``turboquant_dense_kv`` module so
# we can load only the dispatcher without pulling in torch / CUDA.
for module_name in [
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.quantization",
]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

_stub = types.ModuleType("sglang.srt.layers.quantization.turboquant_dense_kv")
_stub.TURBOQUANT_DENSE_KV_PRESETS = {
    "latent_k8": {"bits": 8, "norm_correction": False},
    "latent_4bit_nc": {"bits": 4, "norm_correction": True},
    "latent_k3_nc": {"bits": 3, "norm_correction": True},
    "latent_2p5bit_nc": {"bits": 2.5, "norm_correction": True},
}
sys.modules["sglang.srt.layers.quantization.turboquant_dense_kv"] = _stub

dispatcher = _load_module(
    "sglang.srt.layers.quantization.quantization_config_dispatch",
    ROOT / "python/sglang/srt/layers/quantization/quantization_config_dispatch.py",
)


@dataclass
class FakeServerArgs:
    enable_turboquant_dense_kv_cache: bool = False
    turboquant_dense_kv_preset: str = "latent_2p5bit_nc"
    indexer_quantization_declared: Optional[Dict[str, Any]] = None


@dataclass
class FakeHfConfig:
    quantization_config: Optional[Dict[str, Any]] = field(default=None)


def test_no_quantization_config_is_noop():
    server_args = FakeServerArgs()
    dispatcher.apply_quantization_config_dispatch(server_args, FakeHfConfig())
    assert server_args.enable_turboquant_dense_kv_cache is False
    assert server_args.turboquant_dense_kv_preset == "latent_2p5bit_nc"
    assert server_args.indexer_quantization_declared is None


def test_turboquant_dense_kv_cache_scheme_enables_flag_and_preset():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_4bit_nc",
                "slot_bytes": 274,
                "packed_bits": 4,
                "kv_dim": 576,
                "latent_dim": 512,
                "rope_dim": 64,
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is True
    assert server_args.turboquant_dense_kv_preset == "latent_4bit_nc"


def test_turboquant_dense_default_preset_keeps_default():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_2p5bit_nc",
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is True
    assert server_args.turboquant_dense_kv_preset == "latent_2p5bit_nc"


def test_indexer_quantization_fp8_records_declaration():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "fp8_e4m3",
                "scale_strategy": "per_token",
                "scale_bytes": 4,
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared == {
        "quant_method": "fp8_e4m3",
        "scale_strategy": "per_token",
        "scale_bytes": 4,
    }


def test_cli_flag_takes_precedence_over_config():
    server_args = FakeServerArgs(
        enable_turboquant_dense_kv_cache=True,
        turboquant_dense_kv_preset="latent_k3_nc",
    )
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_4bit_nc",
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    # The CLI-supplied preset must be preserved when it differs from default.
    assert server_args.enable_turboquant_dense_kv_cache is True
    assert server_args.turboquant_dense_kv_preset == "latent_k3_nc"


def test_unknown_kv_cache_scheme_method_is_ignored():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {"type": "float", "num_bits": 8},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is False
    assert server_args.indexer_quantization_declared is None


def test_unknown_turboquant_preset_is_ignored():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_42bit_nc",
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is False
    assert server_args.turboquant_dense_kv_preset == "latent_2p5bit_nc"


def test_unknown_indexer_quant_method_is_ignored():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {"quant_method": "int4"},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared is None


def test_quantization_config_object_with_to_dict():
    class WrappedQuantConfig:
        def to_dict(self):
            return {
                "kv_cache_scheme": {
                    "quant_method": "turboquant_dense",
                    "preset": "latent_4bit_nc",
                },
                "indexer_quantization": {"quant_method": "fp8_e4m3"},
            }

    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(quantization_config=WrappedQuantConfig())
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is True
    assert server_args.turboquant_dense_kv_preset == "latent_4bit_nc"
    assert server_args.indexer_quantization_declared == {"quant_method": "fp8_e4m3"}


def test_both_fields_compose():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_2p5bit_nc",
            },
            "indexer_quantization": {
                "quant_method": "fp8_e4m3",
                "scale_strategy": "per_token",
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_turboquant_dense_kv_cache is True
    assert server_args.turboquant_dense_kv_preset == "latent_2p5bit_nc"
    assert server_args.indexer_quantization_declared == {
        "quant_method": "fp8_e4m3",
        "scale_strategy": "per_token",
    }


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
