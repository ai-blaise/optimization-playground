"""Config-side dispatch for custom KV-cache and indexer kernels.

Reads optional declarative fields from a model's ``quantization_config``
(typically ``hf_config.quantization_config``) and promotes them onto
``server_args``, so that a checkpoint can opt into TurboQuant 2.5-bit
dense KV or the IndexerK8 FP8 fused-store path without requiring the
operator to pass CLI flags.

Two fields are recognized:

1. ``kv_cache_scheme`` (extension of the existing dict already parsed by
   ``modelopt_quant``): when ``quant_method == "turboquant_dense"``, sets
   ``server_args.enable_turboquant_dense_kv_cache = True`` and copies the
   ``preset`` field into ``server_args.turboquant_dense_kv_preset``.

2. ``indexer_quantization``: a new top-level dict; when
   ``quant_method == "fp8_e4m3"``, the declaration is recorded on
   ``server_args.indexer_quantization_declared``. The runtime fused-store
   dispatch in ``nsa_indexer.py`` consults this attribute via
   ``should_use_nsa_fused_store`` so the field actively gates the FP8
   path rather than only documenting intent.

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
from typing import Any, Callable, Dict, Optional

from sglang.srt.layers.attention.nsa.indexer_quantization import (
    INDEXER_DISABLED_QUANT_METHOD,
    INDEXER_FP8_QUANT_METHOD,
    INDEXER_NVFP4_QUANT_METHOD,
    SUPPORTED_INDEXER_QUANT_METHODS,
)
from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TURBOQUANT_DENSE_KV_PRESETS,
)

logger = logging.getLogger(__name__)

TURBOQUANT_DENSE_QUANT_METHOD = "turboquant_dense"
DEFAULT_TURBOQUANT_DENSE_KV_PRESET = "latent_2p5bit_nc"
HIGGS_DENSE_2BIT_QUANT_METHOD = "higgs_dense_2bit"


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


def _maybe_apply_turboquant_dense(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
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
        and server_args.turboquant_dense_kv_preset
        == DEFAULT_TURBOQUANT_DENSE_KV_PRESET
        and preset != DEFAULT_TURBOQUANT_DENSE_KV_PRESET
    ):
        server_args.turboquant_dense_kv_preset = preset
        logger.info(
            "Setting --turboquant-dense-kv-preset=%s from quantization_config.",
            preset,
        )


def _maybe_apply_higgs_dense_2bit(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
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


def _maybe_apply_indexer_quantization(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
    indexer_quant = _coerce_dict(quant_cfg.get("indexer_quantization"))
    if indexer_quant is None:
        return
    method = indexer_quant.get("quant_method")
    if method not in SUPPORTED_INDEXER_QUANT_METHODS:
        logger.info(
            "quantization_config.indexer_quantization.quant_method=%r is "
            "not supported by the NSA Indexer quantization path; "
            "ignoring config-side dispatch.",
            method,
        )
        return

    server_args.indexer_quantization_declared = dict(indexer_quant)
    logger.info(
        "Recording indexer_quantization declaration from quantization_config "
        "(indexer_quantization.quant_method=%s).",
        method,
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


def should_use_nsa_fused_store(
    server_args: Any,
    key_dtype: Any,
    indices_dtype: Any,
    page_size: int,
    *,
    auto_compat_check: Callable[..., bool],
    auto_platform_ok: bool = True,
) -> bool:
    """Decide whether to dispatch to the NSA IndexCache FP8 fused-store kernel.

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
    cli_method = getattr(server_args, "nsa_indexer_quantization", None)
    if cli_method == INDEXER_FP8_QUANT_METHOD:
        if not auto_compat_check(key_dtype, indices_dtype, page_size):
            raise RuntimeError(
                "--nsa-indexer-quantization=fp8_e4m3 was set but the NSA "
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
                    f"quant_method={INDEXER_FP8_QUANT_METHOD!r} but the NSA "
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
    return auto_platform_ok and auto_compat_check(
        key_dtype, indices_dtype, page_size
    )
