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
# we can load only the dispatcher without pulling in torch / CUDA. Keep package
# paths intact so later tests in the same process can still import real SGLang
# modules from disk.
_PACKAGE_PATHS = {
    "sglang": ROOT / "python/sglang",
    "sglang.srt": ROOT / "python/sglang/srt",
    "sglang.srt.layers": ROOT / "python/sglang/srt/layers",
    "sglang.srt.layers.attention": ROOT / "python/sglang/srt/layers/attention",
    "sglang.srt.layers.attention.nsa": ROOT
    / "python/sglang/srt/layers/attention/nsa",
    "sglang.srt.layers.quantization": ROOT / "python/sglang/srt/layers/quantization",
}
for module_name, package_path in _PACKAGE_PATHS.items():
    module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
    module.__path__ = [str(package_path)]

_load_module(
    "sglang.srt.layers.attention.nsa.indexer_quantization",
    ROOT / "python/sglang/srt/layers/attention/nsa/indexer_quantization.py",
)

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
    enable_higgs_dense_2bit_kv_cache: bool = False
    nsa_indexer_mode: str = "vanilla"
    nsa_indexer_quantization: str = "auto"
    nsa_indexcache_freq: int = 4
    nsa_indexcache_pattern: Optional[str] = None
    hisa_block_size: int = 128
    hisa_block_topk: int = 64
    hisa_compression_ratio: float = 4.0
    hisa_min_seq_len: int = 65536
    hisa_execution_mode: str = "optimized"
    enable_nsa_nvfp4_hisa: bool = False
    indexer_quantization_declared: Optional[Dict[str, Any]] = None
    moe_runner_backend: str = "auto"


@dataclass
class FakeHfConfig:
    quantization_config: Optional[Dict[str, Any]] = field(default=None)


def test_no_quantization_config_is_noop():
    server_args = FakeServerArgs()
    dispatcher.apply_quantization_config_dispatch(server_args, FakeHfConfig())
    assert server_args.enable_turboquant_dense_kv_cache is False
    assert server_args.turboquant_dense_kv_preset == "latent_2p5bit_nc"
    assert server_args.nsa_indexer_mode == "vanilla"
    assert server_args.enable_nsa_nvfp4_hisa is False
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


# ---------------------------------------------------------------------------
# HIGGS 2-bit dense MLA KV (alternative to TurboQuant 2.5-bit). Mirrors the
# TurboQuant dispatcher tests above; the two paths are mutually exclusive.
# ---------------------------------------------------------------------------


def test_higgs_dense_kv_cache_scheme_enables_flag():
    """``quant_method=higgs_dense_2bit`` sets the corresponding bool."""
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "higgs_dense_2bit",
                "preset": "eden2_16",
                "slot_bytes": 258,
                "packed_bits": 2,
                "kv_dim": 576,
                "latent_dim": 512,
                "rope_dim": 64,
                "hadamard_groupsize": 512,
                "codebook": "eden2_16",
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_higgs_dense_2bit_kv_cache is True
    # The TurboQuant flag must remain off — the two paths are mutually exclusive.
    assert server_args.enable_turboquant_dense_kv_cache is False


def test_higgs_dense_minimal_scheme_enables_flag():
    """A bare ``quant_method=higgs_dense_2bit`` (no layout fields) still enables."""
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {"quant_method": "higgs_dense_2bit"},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_higgs_dense_2bit_kv_cache is True
    assert server_args.enable_turboquant_dense_kv_cache is False


def test_higgs_cli_flag_already_set_is_noop():
    """When ``--enable-higgs-dense-2bit-kv-cache`` is already True, dispatcher no-ops."""
    server_args = FakeServerArgs(enable_higgs_dense_2bit_kv_cache=True)
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {"quant_method": "higgs_dense_2bit"},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_higgs_dense_2bit_kv_cache is True


def test_higgs_unknown_kv_cache_scheme_method_is_ignored():
    """An unrecognized ``quant_method`` leaves both KV bools off."""
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {"quant_method": "future_kv_v3"},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_higgs_dense_2bit_kv_cache is False
    assert server_args.enable_turboquant_dense_kv_cache is False


def test_higgs_quantization_config_object_with_to_dict():
    """A wrapper with a ``to_dict()`` method is coerced and applied."""

    class WrappedQuantConfig:
        def to_dict(self):
            return {
                "kv_cache_scheme": {
                    "quant_method": "higgs_dense_2bit",
                    "slot_bytes": 258,
                },
            }

    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(quantization_config=WrappedQuantConfig())
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_higgs_dense_2bit_kv_cache is True
    assert server_args.enable_turboquant_dense_kv_cache is False


def test_higgs_declared_blocks_turboquant_cli():
    """Config declares HIGGS while CLI already enabled TurboQuant → ``ValueError``."""
    server_args = FakeServerArgs(enable_turboquant_dense_kv_cache=True)
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {"quant_method": "higgs_dense_2bit"},
        }
    )
    try:
        dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    except ValueError as exc:
        message = str(exc)
        assert "higgs_dense_2bit" in message
        assert "turboquant" in message.lower()
    else:
        raise AssertionError(
            "Declaring higgs_dense_2bit while TurboQuant is enabled on the CLI "
            "must raise ValueError; the two paths are mutually exclusive."
        )


def test_turboquant_declared_blocks_higgs_cli():
    """Config declares TurboQuant while CLI already enabled HIGGS → ``ValueError``."""
    server_args = FakeServerArgs(enable_higgs_dense_2bit_kv_cache=True)
    hf_config = FakeHfConfig(
        quantization_config={
            "kv_cache_scheme": {
                "quant_method": "turboquant_dense",
                "preset": "latent_2p5bit_nc",
            },
        }
    )
    try:
        dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    except ValueError as exc:
        message = str(exc)
        assert "turboquant_dense" in message
        assert "higgs" in message.lower()
    else:
        raise AssertionError(
            "Declaring turboquant_dense while HIGGS is enabled on the CLI must "
            "raise ValueError; the two paths are mutually exclusive."
        )


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


def test_indexer_quantization_nvfp4_records_declaration():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "value_format": "e2m1",
                "scale_format": "ue8m0",
                "scale_block_size": 32,
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared == {
        "quant_method": "nvfp4_e2m1_ue8m0",
        "value_format": "e2m1",
        "scale_format": "ue8m0",
        "scale_block_size": 32,
    }
    assert server_args.enable_nsa_nvfp4_hisa is False


def test_indexer_quantization_indexcache_enables_config_surface():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "indexcache": {
                    "enabled": True,
                    "freq": 2,
                    "pattern": "FSFS",
                },
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared == {
        "quant_method": "nvfp4_e2m1_ue8m0",
        "indexcache": {
            "enabled": True,
            "freq": 2,
            "pattern": "FSFS",
        },
    }
    assert server_args.nsa_indexer_mode == "indexcache"
    assert server_args.nsa_indexcache_freq == 2
    assert server_args.nsa_indexcache_pattern == "FSFS"
    assert server_args.enable_nsa_nvfp4_hisa is False


def test_indexer_quantization_indexcache_can_be_mode_only():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "indexer_mode": "indexcache",
                "indexcache": {
                    "index_topk_freq": 3,
                    "index_topk_pattern": "FSS",
                },
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared is None
    assert server_args.nsa_indexer_mode == "indexcache"
    assert server_args.nsa_indexcache_freq == 3
    assert server_args.nsa_indexcache_pattern == "FSS"


def test_indexer_quantization_indexcache_preserves_cli_mode_and_pattern():
    server_args = FakeServerArgs(
        nsa_indexer_mode="indexcache",
        nsa_indexcache_freq=8,
        nsa_indexcache_pattern="FFFF",
    )
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "indexer_mode": "indexcache",
                "indexcache": {
                    "enabled": True,
                    "freq": 2,
                    "pattern": "FSFS",
                },
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.nsa_indexer_mode == "indexcache"
    assert server_args.nsa_indexcache_freq == 8
    assert server_args.nsa_indexcache_pattern == "FFFF"


def test_indexer_quantization_nvfp4_hisa_enables_config_surface():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "hisa": {
                    "enabled": True,
                    "block_size": 128,
                    "compression_ratio": 4.0,
                    "min_seq_len": 8192,
                    "execution_mode": "optimized",
                },
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_nsa_nvfp4_hisa is True
    assert server_args.nsa_indexer_mode == "indexcache-hisa"
    assert server_args.hisa_block_size == 128
    assert server_args.hisa_block_topk == 64
    assert server_args.hisa_compression_ratio == 4.0
    assert server_args.hisa_min_seq_len == 8192
    assert server_args.hisa_execution_mode == "optimized"


def test_indexer_quantization_nvfp4_hisa_keeps_hisa_mode_with_indexcache_cfg():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "value_format": "e2m1",
                "scale_format": "ue8m0",
                "scale_block_size": 32,
                "indexcache": {
                    "enabled": True,
                    "freq": 2,
                    "pattern": "FSFS",
                },
                "hisa": {
                    "enabled": True,
                    "mode": "indexcache-hisa",
                    "block_size": 128,
                    "compression_ratio": 4.0,
                    "min_seq_len": 8192,
                    "execution_mode": "optimized",
                },
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_nsa_nvfp4_hisa is True
    assert server_args.nsa_indexer_mode == "indexcache-hisa"
    assert server_args.nsa_indexcache_freq == 2
    assert server_args.nsa_indexcache_pattern == "FSFS"
    assert server_args.hisa_block_size == 128
    assert server_args.hisa_compression_ratio == 4.0
    assert server_args.hisa_min_seq_len == 8192
    assert server_args.hisa_execution_mode == "optimized"


def test_indexer_quantization_nvfp4_hisa_rejects_non_combo_mode():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "indexcache": {"enabled": True},
                "hisa": {"enabled": True, "mode": "hisa"},
            },
        }
    )
    try:
        dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    except ValueError as exc:
        message = str(exc)
        assert "indexcache.enabled=true" in message
        assert "indexcache-hisa" in message
    else:
        raise AssertionError(
            "A model config that enables IndexCache but selects standalone HISA "
            "must fail loudly instead of silently disabling the production combo."
        )


def test_indexer_quantization_nvfp4_hisa_preserves_cli_hisa_values():
    server_args = FakeServerArgs(
        nsa_indexer_mode="indexcache-hisa",
        hisa_compression_ratio=4.0,
        hisa_min_seq_len=32768,
    )
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {
                "quant_method": "nvfp4_e2m1_ue8m0",
                "hisa": {"enabled": True, "min_seq_len": 8192},
            },
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.enable_nsa_nvfp4_hisa is True
    assert server_args.nsa_indexer_mode == "indexcache-hisa"
    assert server_args.hisa_compression_ratio == 4.0
    assert server_args.hisa_min_seq_len == 32768


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
    assert server_args.enable_higgs_dense_2bit_kv_cache is False
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


def test_warp_decode_moe_runner_backend_enables_from_config():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "moe_runner_backend": "warp_decode",
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.moe_runner_backend == "warp_decode"


def test_warp_decode_moe_runner_backend_preserves_cli_backend():
    server_args = FakeServerArgs(moe_runner_backend="flashinfer_trtllm")
    hf_config = FakeHfConfig(
        quantization_config={
            "moe_runner_backend": "warp_decode",
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.moe_runner_backend == "flashinfer_trtllm"


def test_unknown_moe_runner_backend_is_ignored():
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "moe_runner_backend": "not_a_backend",
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.moe_runner_backend == "auto"


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


def test_indexer_quantization_disabled_records_declaration():
    """A ``disabled`` declaration is recorded (used to force the fallback)."""
    server_args = FakeServerArgs()
    hf_config = FakeHfConfig(
        quantization_config={
            "indexer_quantization": {"quant_method": "disabled"},
        }
    )
    dispatcher.apply_quantization_config_dispatch(server_args, hf_config)
    assert server_args.indexer_quantization_declared == {"quant_method": "disabled"}


# ---------------------------------------------------------------------------
# Tests for ``should_use_nsa_fused_store`` — the runtime gate that the
# fused-store dispatch site in ``nsa_indexer.py`` consults.
# ---------------------------------------------------------------------------


class _FakeDtype:
    """Stand-in for ``torch.dtype`` so the helper tests don't need torch."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"FakeDtype({self.name!r})"


def _always_compatible(*_args, **_kwargs) -> bool:
    return True


def _never_compatible(*_args, **_kwargs) -> bool:
    return False


def test_should_use_fused_store_no_declaration_matches_auto_detect():
    """No declaration → behavior is identical to current auto-detect (regression guard)."""
    server_args = FakeServerArgs()  # indexer_quantization_declared is None

    # Auto-detect says yes AND platform OK → use fused store.
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=True,
        )
        is True
    )

    # Auto-detect says yes BUT platform not OK → fallback.
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=False,
        )
        is False
    )

    # Auto-detect says no → fallback.
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_never_compatible,
            auto_platform_ok=True,
        )
        is False
    )


def test_should_use_fused_store_declared_fp8_compatible_forces_path():
    server_args = FakeServerArgs(
        indexer_quantization_declared={"quant_method": "fp8_e4m3"}
    )
    # Even if platform auto-flag is False, declaration forces FP8 path
    # provided the kernel compat check accepts the runtime shapes.
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=False,
        )
        is True
    )


def test_should_use_fused_store_declared_fp8_incompatible_raises():
    server_args = FakeServerArgs(
        indexer_quantization_declared={"quant_method": "fp8_e4m3"}
    )
    try:
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("float32"),
            _FakeDtype("int64"),
            page_size=128,
            auto_compat_check=_never_compatible,
            auto_platform_ok=True,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "fp8_e4m3" in message
        assert "FakeDtype('float32')" in message
        assert "page_size=128" in message
    else:
        raise AssertionError(
            "should_use_nsa_fused_store should raise when declared fp8_e4m3 is "
            "incompatible with the runtime tensor shapes."
        )


def test_should_use_fused_store_declared_disabled_forces_fallback():
    server_args = FakeServerArgs(
        indexer_quantization_declared={"quant_method": "disabled"}
    )
    # Even if auto-detect would happily pick the fused path, the
    # declaration forces fallback.
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=True,
        )
        is False
    )


def test_should_use_fused_store_declared_nvfp4_disables_fp8_path():
    server_args = FakeServerArgs(
        indexer_quantization_declared={"quant_method": "nvfp4_e2m1_ue8m0"}
    )
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=True,
        )
        is False
    )


def test_should_use_fused_store_cli_nvfp4_disables_fp8_path():
    server_args = FakeServerArgs(nsa_indexer_quantization="nvfp4_e2m1_ue8m0")
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=True,
        )
        is False
    )


def test_should_use_fused_store_unknown_declared_method_falls_through():
    """An unrecognized declared method falls back to auto-detection."""
    server_args = FakeServerArgs(
        indexer_quantization_declared={"quant_method": "future_kernel_v2"}
    )
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_always_compatible,
            auto_platform_ok=True,
        )
        is True
    )
    assert (
        dispatcher.should_use_nsa_fused_store(
            server_args,
            _FakeDtype("bfloat16"),
            _FakeDtype("int64"),
            page_size=64,
            auto_compat_check=_never_compatible,
            auto_platform_ok=True,
        )
        is False
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
