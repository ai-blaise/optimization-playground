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
   ``server_args.indexer_quantization_declared`` so downstream code can
   assert that the FP8 fused-store dispatch in ``nsa_indexer.py`` is
   actually selected at runtime.

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
from typing import Any, Dict, Optional

from sglang.srt.layers.quantization.turboquant_dense_kv import (
    TURBOQUANT_DENSE_KV_PRESETS,
)

logger = logging.getLogger(__name__)

INDEXER_FP8_QUANT_METHOD = "fp8_e4m3"
TURBOQUANT_DENSE_QUANT_METHOD = "turboquant_dense"
DEFAULT_TURBOQUANT_DENSE_KV_PRESET = "latent_2p5bit_nc"


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


def _maybe_apply_indexer_quantization(
    server_args: Any, quant_cfg: Dict[str, Any]
) -> None:
    indexer_quant = _coerce_dict(quant_cfg.get("indexer_quantization"))
    if indexer_quant is None:
        return
    if indexer_quant.get("quant_method") != INDEXER_FP8_QUANT_METHOD:
        logger.info(
            "quantization_config.indexer_quantization.quant_method=%r is "
            "not supported by the IndexerK8 FP8 fused-store path; "
            "ignoring config-side dispatch.",
            indexer_quant.get("quant_method"),
        )
        return

    server_args.indexer_quantization_declared = dict(indexer_quant)
    logger.info(
        "Recording IndexerK8 FP8 declaration from quantization_config "
        "(indexer_quantization.quant_method=fp8_e4m3)."
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
    _maybe_apply_indexer_quantization(server_args, quant_cfg)
