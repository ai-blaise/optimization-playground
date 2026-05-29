"""Config-side dispatch for custom KV-cache and indexer kernels.

Reads optional declarative fields from a model's ``quantization_config``
(typically ``hf_config.quantization_config``) and promotes them onto
``server_args``, so that a checkpoint can opt into TurboQuant 2.5-bit
dense KV, HIGGS 2-bit dense KV, DSA IndexCache, DSA indexer cache formats,
or a supported MoE runner without requiring the operator to pass CLI flags.

Three fields are recognized:

1. ``kv_cache_scheme`` (extension of the existing dict already parsed by
   ``modelopt_quant``): when ``quant_method == "turboquant_dense"``, sets
   ``server_args.enable_turboquant_dense_kv_cache = True`` and copies the
   ``preset`` field into ``server_args.turboquant_dense_kv_preset``. When
   ``quant_method == "higgs_dense_2bit"``, sets
   ``server_args.enable_higgs_dense_2bit_kv_cache = True`` and may select a
   validated HIGGS B200 candidate through ``kv_cache_scheme.b200_candidate``.

2. ``indexer_quantization``: a new top-level dict; records supported
   cache formats on ``server_args.indexer_quantization_declared`` and can
   select ordinary IndexCache through either ``indexer_mode`` or a nested
   ``indexcache`` declaration.

3. ``moe_runner_backend``: when set to ``"warp_decode"``, selects the
   small-batch Warp Decode MoE runner if the operator left
   ``--moe-runner-backend`` at ``auto``.

CLI flags take precedence: if the operator already passed
``--enable-turboquant-dense-kv-cache`` (i.e. the flag is ``True``) the
config-side dispatch is a no-op for that knob. The preset string is
copied from config only when ``server_args.turboquant_dense_kv_preset``
still equals the dataclass default.

Unknown ``quant_method`` values are ignored (logged once at INFO) so
that an unfamiliar checkpoint loads cleanly on a stock SGLang build.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from sglang.srt.layers.attention.dsa.indexer_quantization import (
    INDEXER_DISABLED_QUANT_METHOD,
    INDEXER_FP8_QUANT_METHOD,
    INDEXER_NVFP4_QUANT_METHOD,
    SUPPORTED_INDEXER_QUANT_METHODS,
)
from sglang.srt.layers.quantization.higgs_dense_2bit_kv import (
    HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
    get_higgs_dense_2bit_b200_candidate,
)
from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TURBOQUANT_DENSE_KV_PRESETS,
)

logger = logging.getLogger(__name__)

TURBOQUANT_DENSE_QUANT_METHOD = "turboquant_dense"
DEFAULT_TURBOQUANT_DENSE_KV_PRESET = "latent_2p5bit_nc"
HIGGS_DENSE_2BIT_QUANT_METHOD = "higgs_dense_2bit"
HIGGS_MHA_2BIT_QUANT_METHOD = "higgs_mha_2bit"
DEFAULT_DSA_INDEXCACHE_FREQ = 4
DEFAULT_NSA_INDEXCACHE_FREQ = DEFAULT_DSA_INDEXCACHE_FREQ
SUPPORTED_CONFIG_INDEXER_MODES = ("vanilla", "indexcache")
SUPPORTED_CONFIG_MOE_RUNNER_BACKENDS = ("warp_decode",)


def _coerce_dict(value: Any) -> Optional[Dict[str, Any]]:
    """Return ``value`` as a dict if possible, else ``None``."""
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        return value.to_dict()
    return None


def _get_quantization_config(hf_config: Any) -> Optional[Dict[str, Any]]:
    quant_cfg = getattr(hf_config, "quantization_config", None)
    return _coerce_dict(quant_cfg)


def _maybe_apply_turboquant_dense(server_args: Any, quant_cfg: Dict[str, Any]) -> None:
    kv_cache_scheme = _coerce_dict(quant_cfg.get("kv_cache_scheme"))
    if kv_cache_scheme is None:
        return
    if kv_cache_scheme.get("quant_method") != TURBOQUANT_DENSE_QUANT_METHOD:
        return

    preset = kv_cache_scheme.get("preset")
    if preset is not None and preset not in TURBOQUANT_DENSE_KV_PRESETS:
        logger.info(
            "quantization_config.kv_cache_scheme.preset=%r is not a known "
            "TurboQuant preset; ignoring config-side dispatch.",
            preset,
        )
        return

    if getattr(server_args, "enable_higgs_dense_2bit_kv_cache", False):
        raise ValueError(
            "quantization_config.kv_cache_scheme.quant_method="
            f"{TURBOQUANT_DENSE_QUANT_METHOD!r} conflicts with "
            "--enable-higgs-dense-2bit-kv-cache; the two dense KV "
            "compression paths are mutually exclusive."
        )

    if not server_args.enable_turboquant_dense_kv_cache:
        server_args.enable_turboquant_dense_kv_cache = True
        logger.info(
            "Enabling TurboQuant dense KV from quantization_config "
            "(kv_cache_scheme.quant_method=turboquant_dense)."
        )

    if (
        preset is not None
        and server_args.turboquant_dense_kv_preset == DEFAULT_TURBOQUANT_DENSE_KV_PRESET
        and preset != DEFAULT_TURBOQUANT_DENSE_KV_PRESET
    ):
        server_args.turboquant_dense_kv_preset = preset
        logger.info(
            "Setting --turboquant-dense-kv-preset=%s from quantization_config.",
            preset,
        )


def _maybe_apply_higgs_dense_2bit(server_args: Any, quant_cfg: Dict[str, Any]) -> None:
    """Promote a HIGGS 2-bit ``kv_cache_scheme`` declaration onto args.

    Recognised JSON shape::

        {
          "kv_cache_scheme": {
            "quant_method": "higgs_dense_2bit",
            "latent_dim": 512,
            "rope_dim": 64,
            "slot_bytes": 258
          }
        }

    Effect: sets ``server_args.enable_higgs_dense_2bit_kv_cache=True``.
    The two TurboQuant and HIGGS dense KV paths are mutually
    exclusive; a config that declares both, or a CLI override that
    enables TurboQuant alongside this declaration, raises
    ``ValueError``.
    """
    kv_cache_scheme = _coerce_dict(quant_cfg.get("kv_cache_scheme"))
    if kv_cache_scheme is None:
        return
    if kv_cache_scheme.get("quant_method") != HIGGS_DENSE_2BIT_QUANT_METHOD:
        return

    if getattr(server_args, "enable_turboquant_dense_kv_cache", False):
        raise ValueError(
            "quantization_config.kv_cache_scheme.quant_method="
            f"{HIGGS_DENSE_2BIT_QUANT_METHOD!r} conflicts with "
            "--enable-turboquant-dense-kv-cache; the two dense KV "
            "compression paths are mutually exclusive."
        )

    if not getattr(server_args, "enable_higgs_dense_2bit_kv_cache", False):
        server_args.enable_higgs_dense_2bit_kv_cache = True
        logger.info(
            "Enabling HIGGS 2-bit dense KV from quantization_config "
            "(kv_cache_scheme.quant_method=higgs_dense_2bit)."
        )

    candidate_name = kv_cache_scheme.get("b200_candidate") or kv_cache_scheme.get(
        "candidate"
    )
    if candidate_name is None:
        return

    candidate = get_higgs_dense_2bit_b200_candidate(str(candidate_name))
    current = os.environ.get(HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV)
    if current and current != candidate.name:
        logger.info(
            "Keeping %s=%s over quantization_config.kv_cache_scheme "
            "candidate=%s.",
            HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV,
            current,
            candidate.name,
        )
        return

    os.environ[HIGGS_DENSE_2BIT_B200_CANDIDATE_ENV] = candidate.name
    logger.info(
        "Selecting HIGGS 2-bit dense KV candidate %s from "
        "quantization_config.kv_cache_scheme.",
        candidate.name,
    )


def get_smc_draft_kv_cache_dtype_from_config(hf_config: Any) -> Optional[str]:
    """Return the SMC draft KV dtype requested by a draft model config."""
    quant_cfg = _get_quantization_config(hf_config)
    if quant_cfg is None:
        return None
    kv_cache_scheme = _coerce_dict(
        quant_cfg.get("smc_draft_kv_cache_scheme")
    ) or _coerce_dict(quant_cfg.get("kv_cache_scheme"))
    if kv_cache_scheme is None:
        return None
    if kv_cache_scheme.get("quant_method") == HIGGS_MHA_2BIT_QUANT_METHOD:
        return "higgs_2bit"
    return None


def _maybe_apply_indexer_quantization(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
    indexer_quant = _coerce_dict(quant_cfg.get("indexer_quantization"))
    if indexer_quant is None:
        return
    method = indexer_quant.get("quant_method")
    if method is not None and method not in SUPPORTED_INDEXER_QUANT_METHODS:
        logger.info(
            "quantization_config.indexer_quantization.quant_method=%r is "
            "not supported by the DSA Indexer quantization path; "
            "ignoring config-side dispatch.",
            method,
        )
        return

    if method in SUPPORTED_INDEXER_QUANT_METHODS:
        server_args.indexer_quantization_declared = dict(indexer_quant)
        logger.info(
            "Recording indexer_quantization declaration from quantization_config "
            "(indexer_quantization.quant_method=%s).",
            method,
        )

    indexcache_cfg = _coerce_dict(indexer_quant.get("indexcache"))
    indexcache_enabled = indexcache_cfg is not None and bool(
        indexcache_cfg.get("enabled", False)
    )
    hisa_cfg = _coerce_dict(indexer_quant.get("hisa"))
    hisa_enabled = (
        method == INDEXER_NVFP4_QUANT_METHOD
        and hisa_cfg is not None
        and bool(hisa_cfg.get("enabled", False))
    )

    _maybe_apply_indexcache(server_args, indexer_quant, apply_mode=not hisa_enabled)

    if not hisa_enabled:
        return

    server_args.enable_dsa_nvfp4_hisa = True
    server_args.enable_nsa_nvfp4_hisa = True
    mode = str(hisa_cfg.get("mode", "indexcache-hisa"))
    if indexcache_enabled and mode == "hisa":
        raise ValueError(
            "quantization_config.indexer_quantization declares "
            "indexcache.enabled=true but hisa.mode='hisa'. Use "
            "hisa.mode='indexcache-hisa' for the NVFP4 IndexCache+HISA "
            "deployment path."
        )
    if mode not in ("hisa", "indexcache-hisa"):
        logger.info(
            "quantization_config.indexer_quantization.hisa.mode=%r is not "
            "supported; keeping the existing DSA indexer mode.",
            mode,
        )
    elif (
        getattr(
            server_args,
            "dsa_indexer_mode",
            getattr(server_args, "nsa_indexer_mode", "vanilla"),
        )
        == "vanilla"
    ):
        server_args.dsa_indexer_mode = mode
        server_args.nsa_indexer_mode = mode
    if "block_size" in hisa_cfg and getattr(server_args, "hisa_block_size", 128) == 128:
        server_args.hisa_block_size = int(hisa_cfg["block_size"])
    if "block_topk" in hisa_cfg and getattr(server_args, "hisa_block_topk", 64) == 64:
        server_args.hisa_block_topk = int(hisa_cfg["block_topk"])
    if (
        "compression_ratio" in hisa_cfg
        and getattr(server_args, "hisa_compression_ratio", 4.0) == 4.0
    ):
        server_args.hisa_compression_ratio = float(hisa_cfg["compression_ratio"])
    if (
        "min_seq_len" in hisa_cfg
        and getattr(server_args, "hisa_min_seq_len", 65536) == 65536
    ):
        server_args.hisa_min_seq_len = int(hisa_cfg["min_seq_len"])
    if (
        "execution_mode" in hisa_cfg
        and getattr(server_args, "hisa_execution_mode", "optimized") == "optimized"
    ):
        server_args.hisa_execution_mode = str(hisa_cfg["execution_mode"])
    logger.info(
        "Enabling NVFP4 HISA IndexCache indexer from "
        "quantization_config.indexer_quantization.hisa."
    )


def _maybe_apply_indexcache(
    server_args: Any, indexer_quant: Dict[str, Any], *, apply_mode: bool
) -> None:
    indexcache_cfg = _coerce_dict(indexer_quant.get("indexcache"))
    mode = None
    if apply_mode:
        mode = indexer_quant.get("indexer_mode")
        if (
            mode is None
            and indexcache_cfg is not None
            and indexcache_cfg.get("enabled")
        ):
            mode = "indexcache"

    if mode is not None:
        mode = str(mode)
        if mode not in SUPPORTED_CONFIG_INDEXER_MODES:
            logger.info(
                "quantization_config.indexer_quantization.indexer_mode=%r is "
                "not supported for config-side dispatch; keeping the existing "
                "DSA indexer mode.",
                mode,
            )
        elif getattr(
            server_args,
            "dsa_indexer_mode",
            getattr(server_args, "nsa_indexer_mode", "vanilla"),
        ) == "vanilla":
            server_args.dsa_indexer_mode = mode
            server_args.nsa_indexer_mode = mode

    if indexcache_cfg is None:
        return

    freq = indexcache_cfg.get("freq", indexcache_cfg.get("index_topk_freq"))
    if (
        freq is not None
        and getattr(
            server_args,
            "dsa_indexcache_freq",
            getattr(
                server_args,
                "nsa_indexcache_freq",
                DEFAULT_DSA_INDEXCACHE_FREQ,
            ),
        )
        == DEFAULT_DSA_INDEXCACHE_FREQ
    ):
        server_args.dsa_indexcache_freq = int(freq)
        server_args.nsa_indexcache_freq = int(freq)

    pattern = indexcache_cfg.get("pattern", indexcache_cfg.get("index_topk_pattern"))
    if (
        pattern is not None
        and getattr(
            server_args,
            "dsa_indexcache_pattern",
            getattr(server_args, "nsa_indexcache_pattern", None),
        )
        is None
    ):
        server_args.dsa_indexcache_pattern = str(pattern)
        server_args.nsa_indexcache_pattern = str(pattern)

def _maybe_apply_moe_runner_backend(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
    backend = quant_cfg.get("moe_runner_backend")
    if backend is None:
        return

    backend = str(backend)
    if backend not in SUPPORTED_CONFIG_MOE_RUNNER_BACKENDS:
        logger.info(
            "quantization_config.moe_runner_backend=%r is not supported for "
            "config-side dispatch; keeping the existing MoE runner backend.",
            backend,
        )
        return

    if getattr(server_args, "moe_runner_backend", "auto") == "auto":
        server_args.moe_runner_backend = backend
        logger.info(
            "Setting --moe-runner-backend=%s from quantization_config.",
            backend,
        )


def apply_quantization_config_dispatch(
    server_args: Any, hf_config: Any
) -> None:
    """Promote declarative ``quantization_config`` fields onto ``server_args``.

    Behavior is a no-op when ``hf_config.quantization_config`` is absent
    or contains no recognized fields. Existing CLI flags always win.
    """
    quant_cfg = _get_quantization_config(hf_config)
    if quant_cfg is None:
        return
    _maybe_apply_turboquant_dense(server_args, quant_cfg)
    _maybe_apply_higgs_dense_2bit(server_args, quant_cfg)
    _maybe_apply_indexer_quantization(server_args, quant_cfg)
    _maybe_apply_moe_runner_backend(server_args, quant_cfg)


def should_use_dsa_fused_store(
    server_args: Any,
    key_dtype: Any,
    indices_dtype: Any,
    page_size: int,
    *,
    auto_compat_check: Callable[..., bool],
    auto_platform_ok: bool = True,
) -> bool:
    """Decide whether to dispatch to the DSA IndexCache FP8 fused-store kernel.

    Precedence, high to low:

    1. ``server_args.indexer_quantization_declared`` (populated by
       ``apply_quantization_config_dispatch``):

       * ``quant_method == "fp8_e4m3"`` forces the fused-store path; the
         caller's ``auto_compat_check`` must accept the runtime tensor
         shapes or this raises ``RuntimeError`` instead of silently
         falling back.
       * ``quant_method == "disabled"`` forces the fallback path.
       * Other recorded values fall through to auto-detection.

    2. Auto-detection: ``auto_platform_ok and auto_compat_check(...)``.
       Preserves the historical behavior when no declaration is present.

    Raises:
      RuntimeError: when the config declares ``fp8_e4m3`` but
        ``auto_compat_check`` rejects the runtime ``(key_dtype,
        indices_dtype, page_size)`` triple.
    """
    cli_method = getattr(
        server_args,
        "dsa_indexer_quantization",
        getattr(server_args, "nsa_indexer_quantization", None),
    )
    if cli_method == INDEXER_FP8_QUANT_METHOD:
        if not auto_compat_check(key_dtype, indices_dtype, page_size):
            raise RuntimeError(
                "--dsa-indexer-quantization=fp8_e4m3 was set but the DSA "
                "fused-store kernel rejected the runtime tensor shapes "
                f"(key_dtype={key_dtype}, indices_dtype={indices_dtype}, "
                f"page_size={page_size})."
            )
        return True
    if cli_method in (INDEXER_DISABLED_QUANT_METHOD, INDEXER_NVFP4_QUANT_METHOD):
        return False

    declared = getattr(server_args, "indexer_quantization_declared", None)
    if isinstance(declared, dict):
        method = declared.get("quant_method")
        if method == INDEXER_FP8_QUANT_METHOD:
            if not auto_compat_check(key_dtype, indices_dtype, page_size):
                raise RuntimeError(
                    "quantization_config.indexer_quantization declared "
                    f"quant_method={INDEXER_FP8_QUANT_METHOD!r} but the DSA "
                    "fused-store kernel rejected the runtime tensor shapes "
                    f"(key_dtype={key_dtype}, indices_dtype={indices_dtype}, "
                    f"page_size={page_size}). Either remove the declaration "
                    "or fix the tensor layout."
                )
            return True
        if method == INDEXER_DISABLED_QUANT_METHOD:
            return False
        if method == INDEXER_NVFP4_QUANT_METHOD:
            return False
    return auto_platform_ok and auto_compat_check(key_dtype, indices_dtype, page_size)


should_use_nsa_fused_store = should_use_dsa_fused_store
