"""NVFP4/UE8M0 quantization kernels for the NSA Indexer."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.nvfp4 import _nvfp4_arch_env, _nvfp4_cuda_flags
from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_cached_module = None
_cached_key = None
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, raw, default)
        return default


# Use the fused exact radix top-k tail by default. Disable via env to fall back
# to the mask + torch.topk + map reference path for A/B testing.
_hisa_fused_topk = _env_bool("SGLANG_NSA_HISA_FUSED_TOPK", True)
_hisa_row_split_candidate_keys = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_ROW_SPLIT_CANDIDATE_KEYS", False
)
_hisa_row_split_candidate_key_splits = max(
    1, _env_int("SGLANG_NSA_NVFP4_HISA_ROW_SPLIT_CANDIDATE_KEY_SPLITS", 1)
)
# iter2 vector 2: route nvfp4_hisa_indexer_paged_torch's candidate scorer
# through the tile-N kernel (kTileN=8 candidates per CTA) by default. Setting
# this env var to 0/false reverts to the iter1 per-candidate kernel.
_hisa_nvfp4_candidate_score_tilen = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_TILEN", True
)
# iter2 vector 2 experimental: kTileN=16 variant; opt-in if the production
# shape grid shows it strictly faster than kTileN=8.
_hisa_nvfp4_candidate_score_tilen_size = _env_int(
    "SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_TILEN_SIZE", 8
)
if _hisa_nvfp4_candidate_score_tilen_size not in (8, 16, 32):
    logger.warning(
        "SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_TILEN_SIZE must be 8, 16, "
        "or 32; got %d, falling back to 8.",
        _hisa_nvfp4_candidate_score_tilen_size,
    )
    _hisa_nvfp4_candidate_score_tilen_size = 8

# iter3 vector 4: per-shape kTileN auto-tune toggle. When True (default),
# the dispatcher routes each (q_rows, candidate_len) cell to the kTileN
# variant that minimized cand_score on the bench harness for that cell.
# When False, the SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_TILEN_SIZE env
# value is used unconditionally. The auto-tune table is fixed at module
# import time (no per-call CUDA event measurements).
_hisa_nvfp4_candidate_score_autotune = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_AUTOTUNE", True
)

# iter3 vector 1: opt-in persistent-block candidate_score kernel.
# Amortizes the per-row NVFP4 Q dequant across many tile-blocks of the
# same row (typ. 32 tiles/CTA at long prefix, vs 1 tile/CTA in iter2
# tile-N). DEFAULT OFF: at the production shape grid (n_heads=64,
# index_topk=1024, hisa_block_size=128, page_size=64, compression=4.0)
# the persistent kernel is uniformly 3-60% slower than the iter2 tile-N
# kernel — the iter2 baseline already amortized Q enough that Q is no
# longer the bottleneck, and the persistent kernel's inner-loop sync
# accounting (3 __syncthreads per tile across 32 tile iters) plus
# heavier per-CTA register footprint dominate the savings. The kernel
# is kept as a checkpoint for future investigation (e.g. SMC-SD with
# different shape mixes, or coupling with a TMA-backed K prefetch).
_hisa_nvfp4_candidate_score_persistent = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_PERSISTENT", False
)

# iter4 PRIMARY: persistent-block candidate_score + cp.async K-row prefetch.
# Pairs the iter3 v1 persistent kernel (Q amortized across all tiles per
# CTA) with a 2-stage cp.async ping-pong prefetch of raw NVFP4 K bytes
# into ping-pong SMEM buffers, hiding the per-tile HBM K-decode latency
# under Stage B/C compute. Default OFF pending the iter4 microbench +
# correctness sweep on the production HISA grid.
_hisa_nvfp4_candidate_score_kprefetch = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_CAND_SCORE_KPREFETCH", False
)

# iter5 PRIMARY: WMMA candidate_score (mma.m16n8k8 fp32 Stage B). Replaces
# the scalar fmuladd + warp_sum dot in the iter3 tilen32 Stage B with a
# B200 SM_100 tensor-core mma over fp16 inputs into an fp32 accumulator.
# Only supports kTileN=32 and n_heads <= 64 (production cap). When True
# the autotune dispatcher routes large-batch cells (total_work >= 1M)
# to the WMMA variant; smaller cells stay on the scalar kTileN=16/8 paths
# because the WMMA SMEM overhead (16 KB Q-table) hurts more than it helps
# at small total_work. Opt-in only on B200 SM_100 builds.
_hisa_nvfp4_candidate_score_wmma = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA", False
)

# iter4 tertiary: predecode-scale mean_pool. The kernel pre-decodes the
# per-token scale word into an fp32 SMEM table during the staging pass,
# so the inner per-dim sum loop is branchless and reads one byte + one
# fp32 per iter (vs iter2's byte + uint32 + shift/mask/__uint_as_float).
# Bit-identical to iter2; ~1.6x faster on B200 production shapes.
# Default ON. Set to 0 to fall back to iter2 cooperative-uint4 mean_pool.
_hisa_nvfp4_mean_pool_predecode = _env_bool(
    "SGLANG_NSA_NVFP4_HISA_MEAN_POOL_PREDECODE", True
)


def _hisa_mean_pool_call(
    index_k_with_scale_buffer: "torch.Tensor",
    page_table: "torch.Tensor",
    seq_lens_flat: "torch.Tensor",
    max_blocks: int,
    page_size: int = 64,
) -> "torch.Tensor":
    """Dispatch the mean_pool kernel selected by the iter4 env var.

    Defaults to the iter4 predecode-scale kernel; falls back to the iter2
    cooperative-uint4 kernel when SGLANG_NSA_NVFP4_HISA_MEAN_POOL_PREDECODE
    is 0. Both kernels have identical FFI and produce bit-identical
    outputs modulo fp32 accumulation order.
    """
    if _hisa_nvfp4_mean_pool_predecode:
        return hisa_mean_pool_predecode_indexer_cache_nvfp4(
            index_k_with_scale_buffer,
            page_table,
            seq_lens_flat,
            max_blocks,
            page_size=page_size,
        )
    return hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
        page_size=page_size,
    )


_hisa_profile_path = os.environ.get(
    "SGLANG_NSA_NVFP4_HISA_PROFILE_PATH"
) or os.environ.get("SGLANG_NSA_HISA_PROFILE_PATH")
_hisa_profile_sync = os.environ.get(
    "SGLANG_NSA_NVFP4_HISA_PROFILE_SYNC", ""
).lower() in ("1", "true", "yes", "on")
_hisa_profile_lock = threading.Lock()
_NVFP4_E2M1_CODEBOOK = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# DeepGEMM splits the head count into two TMEM loads of kNumHeads / 2.
# The current Blackwell FP4 MQA kernels only support TMEM load widths 32/64,
# so 32 heads would instantiate an unsupported 16-wide load and JIT-fail.
_DEEPGEMM_FP4_MQA_HEAD_COUNTS = (64,)
_hisa_candidate_schedule_lock = threading.Lock()
_hisa_candidate_schedule_cache: dict[
    tuple[int, int, int, int], tuple[torch.Tensor, object]
] = {}


def _mathdx_include_paths() -> list[str]:
    spec = importlib.util.find_spec("nvidia.mathdx")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "NVFP4 HISA megakernel requires nvidia-mathdx for cuBLASDx headers."
        )
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    paths = [
        root / "include",
        root / "external" / "cutlass" / "include",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing nvidia-mathdx include paths: {missing}")
    return [str(path) for path in paths]


def _deepgemm_include_paths() -> list[str]:
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "NVFP4 HISA DeepGEMM-source probes require the deep_gemm package."
        )
    include_dir = Path(spec.origin).resolve().parent / "include"
    if not include_dir.exists():
        raise RuntimeError(f"Missing deep_gemm include path: {include_dir}")
    return [str(include_dir)]


def _deepgemm_has_fp4_mqa_logits() -> bool:
    try:
        import deep_gemm
    except (ImportError, ModuleNotFoundError):
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


def _deepgemm_fp4_mqa_supports(q_values: torch.Tensor) -> bool:
    return (
        _deepgemm_has_fp4_mqa_logits()
        and q_values.dim() == 3
        and q_values.shape[1] in _DEEPGEMM_FP4_MQA_HEAD_COUNTS
        and q_values.shape[-1] == 64
    )


def _hisa_uniform_candidate_schedule(
    q_values: torch.Tensor, candidate_len: int, deep_gemm_module
) -> tuple[torch.Tensor, object]:
    """Reuse the uniform candidate schedule used by collective HISA scoring."""

    rows = int(q_values.shape[0])
    candidate_len = int(candidate_len)
    num_sms = int(deep_gemm_module.get_num_sms())
    device = q_values.device
    if device.type != "cuda":
        context_lens = torch.full(
            (rows, 1),
            candidate_len,
            device=device,
            dtype=torch.int32,
        )
        return context_lens, deep_gemm_module.get_paged_mqa_logits_metadata(
            context_lens, 64, num_sms
        )

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (int(device_index), rows, candidate_len, num_sms)
    with _hisa_candidate_schedule_lock:
        cached = _hisa_candidate_schedule_cache.get(key)
        if cached is not None and cached[0].device == device:
            return cached

    context_lens = torch.full(
        (rows, 1),
        candidate_len,
        device=device,
        dtype=torch.int32,
    )
    schedule_metadata = deep_gemm_module.get_paged_mqa_logits_metadata(
        context_lens, 64, num_sms
    )
    with _hisa_candidate_schedule_lock:
        _hisa_candidate_schedule_cache[key] = (context_lens, schedule_metadata)
    return context_lens, schedule_metadata


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


def _should_fallback_to_dense_short(
    fallback_to_dense_if_short: bool, prefix_lens: torch.Tensor, topk_tokens: int
) -> bool:
    if not fallback_to_dense_if_short or _is_cuda_graph_capturing(prefix_lens):
        return False
    return bool(torch.all(prefix_lens <= topk_tokens).item())


def _hisa_max_blocks(
    seq_lens_flat: torch.Tensor, page_table: torch.Tensor, block_size: int
) -> int:
    if _is_cuda_graph_capturing(seq_lens_flat):
        return int(page_table.shape[1])
    return int(((seq_lens_flat.max().item() + block_size - 1) // block_size))


def _hisa_max_blocks_from_seq_len(max_seq_len: int, block_size: int) -> int:
    return max(1, (int(max_seq_len) + int(block_size) - 1) // int(block_size))


def _hisa_max_blocks_from_page_table(
    page_table: torch.Tensor, block_size: int, page_size: int = 64
) -> int:
    max_tokens = int(page_table.shape[1]) * int(page_size)
    return max(1, (max_tokens + int(block_size) - 1) // int(block_size))


@cache_once
def _jit_nvfp4_indexer_module(
    key_dtype: torch.dtype,
    indices_dtype: torch.dtype,
    page_size: int,
    enable_hisa_selector_megakernel: bool = False,
) -> Module:
    with _nvfp4_arch_env():
        args = make_cpp_args(
            key_dtype, indices_dtype, page_size, is_arch_support_pdl()
        )
        cuda_wrappers = [
            (
                "fused_store_index_k_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::store_index_k",
            ),
            (
                "quantize_indexer_q_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::quantize_q",
            ),
            (
                "dequantize_indexer_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::dequantize",
            ),
            (
                "hisa_mean_pool_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mean_pool",
            ),
            (
                # iter3 vector 2: TMA-based mean_pool (cp.async.bulk).
                "hisa_mean_pool_tma_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mean_pool_tma",
            ),
            (
                # iter4 tertiary: predecoded-scale mean_pool. Builds a
                # per-token fp32 scales table during the staging pass so
                # the hot sum loop is a 1-byte + 1-fp32 SMEM-only kernel.
                "hisa_mean_pool_predecode_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mean_pool_predecode",
            ),
            (
                # iter5 SECONDARY: predecoded-scale mean_pool with
                # transposed scales SMEM + 2-iter FMA pair. Hot loop
                # reads 2 scales per LDS.b64 and runs two parallel fp32
                # accumulators for ILP. Bit-identical to iter4 modulo
                # accumulation order.
                "hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mean_pool_predecode_fma2",
            ),
            (
                "hisa_candidate_score_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score",
            ),
            (
                # iter2 vector 2: tile-N candidate_score (kTileN=8 candidates
                # per CTA, amortizing the per-row NVFP4 Q dequant 8x).
                "hisa_candidate_score_tilen_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_tilen",
            ),
            (
                # iter2 vector 2 experimental: kTileN=16 variant; 16x Q dequant
                # amortization vs the iter1 per-candidate kernel.
                "hisa_candidate_score_tilen16_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_tilen16",
            ),
            (
                # iter3 vector 4: kTileN=32 variant; 32x Q dequant
                # amortization vs iter1 per-candidate. 4x fewer CTAs
                # vs iter2 kTileN=8. Larger per-CTA SMEM (~32 KB).
                "hisa_candidate_score_tilen32_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_tilen32",
            ),
            (
                # iter5 PRIMARY: WMMA candidate_score (mma.m16n8k8 fp32
                # Stage B). Replaces the scalar fmuladd + warp_sum chain
                # with a B200 SM_100 tensor-core mma over fp16 inputs into
                # an fp32 accumulator. Routes to one of three kMaxHeads
                # instantiations (8 / 16 / 64) based on params.n_heads
                # so smem_q_h stays sized to the actual production grid.
                # Opt-in via SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA.
                "hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_tilen32_wmma",
            ),
            (
                # iter3 vector 1: persistent-block candidate_score
                # (kTileN=8). Each CTA is bound to one (row, split) pair
                # and sweeps tiles_per_split tile-blocks of that row,
                # amortizing the per-row Q dequant across the entire
                # split's tile range. Default production iter3 path.
                "hisa_candidate_score_persistent_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_persistent",
            ),
            (
                # iter3 vector 1 experimental: persistent-block kTileN=16.
                "hisa_candidate_score_persistent_tilen16_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_persistent_tilen16",
            ),
            (
                # iter4 PRIMARY: persistent-block + cp.async K-row prefetch.
                # Pairs iter3 v1 persistent (Q amortized across all tiles
                # per CTA) with a 2-stage ping-pong cp.async prefetch of
                # raw NVFP4 K bytes into ping-pong SMEM buffers, hiding
                # the per-tile HBM K-decode latency under Stage B/C
                # compute. Opt-in via SGLANG_NSA_NVFP4_HISA_CAND_SCORE_KPREFETCH.
                "hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_score_persistent_kprefetch",
            ),
            (
                "hisa_block_score_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_block_score",
            ),
            (
                "hisa_block_topk_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_block_topk",
            ),
            (
                "hisa_block_topk_map_all_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_block_topk_map_all",
            ),
            (
                "hisa_candidate_pages_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_pages",
            ),
            (
                "hisa_mask_candidate_logits_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mask_candidate_logits",
            ),
            (
                "hisa_mask_logits_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_mask_logits",
            ),
            (
                "hisa_map_topk_indices_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_map_topk_indices",
            ),
            (
                "hisa_map_candidate_indices_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_map_candidate_indices",
            ),
            (
                "hisa_fused_mask_topk_map_indexer_cache_nvfp4",
                f"NVFP4IndexerQuantKernel<{args}>::hisa_fused_mask_topk_map",
            ),
        ]
        extra_cuda_cflags = _nvfp4_cuda_flags()
        extra_include_paths = []
        if enable_hisa_selector_megakernel:
            cuda_wrappers.append(
                (
                    "hisa_selector_megakernel_indexer_cache_nvfp4",
                    f"NVFP4IndexerQuantKernel<{args}>::hisa_selector_megakernel",
                )
            )
            cuda_wrappers.extend(
                [
                    (
                        "hisa_selector_parallel_select_blocks_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_selector_parallel_select_blocks",
                    ),
                    (
                        "hisa_selector_parallel_score_candidates_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_selector_parallel_score_candidates",
                    ),
                    (
                        "hisa_selector_cluster_fused_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_selector_cluster_fused",
                    ),
                    (
                        "hisa_deepgemm_candidate_logits_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_deepgemm_candidate_logits",
                    ),
                    (
                        "hisa_deepgemm_candidate_keys_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_deepgemm_candidate_keys",
                    ),
                    (
                        "hisa_deepgemm_candidate_keys_row_split_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_deepgemm_candidate_keys_row_split",
                    ),
                    (
                        "hisa_deepgemm_candidate_topk_cooperative_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_deepgemm_candidate_topk_cooperative",
                    ),
                    (
                        "hisa_deepgemm_candidate_topk_cluster_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_deepgemm_candidate_topk_cluster",
                    ),
                    (
                        "hisa_candidate_keys_topk_map_indexer_cache_nvfp4",
                        f"NVFP4IndexerQuantKernel<{args}>::hisa_candidate_keys_topk_map",
                    ),
                ]
            )
            extra_cuda_cflags = [
                *extra_cuda_cflags,
                "-DSGLANG_ENABLE_HISA_SELECTOR_MEGAKERNEL=1",
            ]
            extra_include_paths = [
                *_mathdx_include_paths(),
                *_deepgemm_include_paths(),
            ]
        return load_jit(
            (
                "nvfp4_indexer_quant_megakernel"
                if enable_hisa_selector_megakernel
                else "nvfp4_indexer_quant"
            ),
            *args,
            cuda_files=["dsa/nvfp4_indexer_quant.cuh"],
            cuda_wrappers=cuda_wrappers,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            extra_dependencies=["cutlass"],
        )


@cache_once
def can_use_nsa_nvfp4_indexer(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    try:
        _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning("Failed to load NSA NVFP4 indexer JIT kernel: %s", e)
        return False


def _get_module_fast(key_dtype, indices_dtype, page_size):
    global _cached_key, _cached_module
    cache_key = (key_dtype, indices_dtype, page_size)
    if _cached_key == cache_key:
        return _cached_module
    module = _jit_nvfp4_indexer_module(key_dtype, indices_dtype, page_size)
    _cached_key = cache_key
    _cached_module = module
    return module


def _get_module_hisa_selector_megakernel(key_dtype, indices_dtype, page_size):
    return _jit_nvfp4_indexer_module(
        key_dtype, indices_dtype, page_size, enable_hisa_selector_megakernel=True
    )


@debug_kernel_api
def fused_store_index_k_cache_nvfp4(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    if key.dim() != 2:
        key = key.reshape(-1, key.shape[-1])
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    _get_module_fast(
        key.dtype, out_cache_loc.dtype, page_size
    ).fused_store_index_k_cache_nvfp4(key, index_k_with_scale, out_cache_loc)


from sglang.srt.utils.custom_op import register_custom_op as _register_custom_op


@_register_custom_op(mutates_args=["values", "scales"])
def _quantize_indexer_q_nvfp4_op(
    query: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    indices_dtype_idx: int,
    page_size: int,
) -> None:
    # Dispatch on the indices dtype index because torch.library custom op
    # schemas don't accept torch.dtype directly. Caller encodes int64=0,
    # int32=1. Dynamo sees this op as one opaque node and never enters
    # _get_module_fast (which dispatches to a JIT-compiled module).
    indices_dtype = torch.int64 if indices_dtype_idx == 0 else torch.int32
    _get_module_fast(query.dtype, indices_dtype, page_size).quantize_indexer_q_nvfp4(
        query, values, scales
    )


@debug_kernel_api
def quantize_indexer_q_nvfp4(
    query: torch.Tensor, indices_dtype: torch.dtype = torch.int64, page_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = query.shape
    if query.shape[-1] != 128:
        raise ValueError(f"NSA NVFP4 indexer Q expects head_dim=128, got {query.shape}.")
    query = query.reshape(-1, original_shape[-1])
    if not query.is_contiguous():
        query = query.contiguous()

    values = torch.empty(
        (query.shape[0], original_shape[-1] // 2),
        dtype=torch.uint8,
        device=query.device,
    )
    scales = torch.empty((query.shape[0],), dtype=torch.int32, device=query.device)
    indices_dtype_idx = 0 if indices_dtype == torch.int64 else 1
    _quantize_indexer_q_nvfp4_op(
        query, values, scales, indices_dtype_idx, page_size
    )
    return (
        values.view(*original_shape[:-1], original_shape[-1] // 2),
        scales.view(*original_shape[:-1]),
    )


@debug_kernel_api
def hisa_mean_pool_indexer_cache_nvfp4(
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks: int,
    page_size: int = 64,
) -> torch.Tensor:
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    reps = torch.empty(
        (page_table.shape[0], max_blocks, 128),
        dtype=torch.float32,
        device=index_k_with_scale.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_mean_pool_indexer_cache_nvfp4(
        index_k_with_scale, page_table, seq_lens.to(torch.int32), reps
    )
    return reps


@debug_kernel_api
def hisa_mean_pool_tma_indexer_cache_nvfp4(
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks: int,
    page_size: int = 64,
) -> torch.Tensor:
    """iter3 vector 2: TMA-based mean_pool (cp.async.bulk per page).

    Drop-in replacement for hisa_mean_pool_indexer_cache_nvfp4. The per-
    page staging loop is replaced with a single cp.async.bulk per page;
    the rest of the kernel (page-id resolve, zero-fill of invalid pages,
    per-dim sum) is unchanged.
    """
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    reps = torch.empty(
        (page_table.shape[0], max_blocks, 128),
        dtype=torch.float32,
        device=index_k_with_scale.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_mean_pool_tma_indexer_cache_nvfp4(
        index_k_with_scale, page_table, seq_lens.to(torch.int32), reps
    )
    return reps


@debug_kernel_api
def hisa_mean_pool_predecode_indexer_cache_nvfp4(
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks: int,
    page_size: int = 64,
) -> torch.Tensor:
    """iter4 tertiary: predecoded-scale mean_pool.

    Drop-in replacement for hisa_mean_pool_indexer_cache_nvfp4. The per-
    page staging is identical to iter2 (cooperative uint4 loads). After
    staging, a one-pass predecode builds a per-(token, scale-group) fp32
    scales table and a per-token value-byte base offset, so the inner
    per-dim sum loop becomes a branchless 1-byte + 1-fp32 SMEM chain
    feeding decode_e2m1_nibble. Invalid pages contribute zero via the
    zeroed scales table (no page_id branch in the hot loop).
    """
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    reps = torch.empty(
        (page_table.shape[0], max_blocks, 128),
        dtype=torch.float32,
        device=index_k_with_scale.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_mean_pool_predecode_indexer_cache_nvfp4(
        index_k_with_scale, page_table, seq_lens.to(torch.int32), reps
    )
    return reps


@debug_kernel_api
def hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4(
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks: int,
    page_size: int = 64,
) -> torch.Tensor:
    """iter5 SECONDARY: predecoded-scale mean_pool with 2-iter FMA pair.

    Strict superset of the iter4 predecode kernel: same page staging, same
    scale predecode pass, but the scales SMEM is transposed to
    [scale_group][token-local] so the hot per-dim inner loop walks a
    contiguous 128-fp32 row and can LDS.b64 two scales per issue.  The
    inner loop processes two tokens per iter into two separate fp32
    accumulators, exposing more ILP to the SM_100 schedulers (4 FFMA/
    cycle).  Bit-identical to iter4 modulo accumulation order.
    """
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    reps = torch.empty(
        (page_table.shape[0], max_blocks, 128),
        dtype=torch.float32,
        device=index_k_with_scale.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_mean_pool_predecode_fma2_indexer_cache_nvfp4(
        index_k_with_scale, page_table, seq_lens.to(torch.int32), reps
    )
    return reps


@debug_kernel_api
def hisa_candidate_score_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    # iter1 candidate_score (n_heads > 8 branch) no longer atomicAdd-accumulates
    # partial head_group dots into logits — every (cand, row) slot is now
    # written exactly once (warp 0 writes the valid score; thread 0 writes
    # -INFINITY in the invalid-token early-return). The legacy
    # `if q_values.shape[1] > 8: logits.zero_()` zero-init kernel is therefore
    # dead and has been dropped to save a per-call memset.
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_tilen_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter2 vector 2: tile-N candidate_score (kTileN=8 candidates per CTA).

    Drop-in replacement for hisa_candidate_score_indexer_cache_nvfp4 with
    8x amortization of the per-row NVFP4 Q dequant. Output semantics are
    identical: logits[Q, candidate_len] (-INFINITY at invalid token slots),
    candidate_indices[Q, candidate_len] (-1 at invalid).
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_tilen_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_tilen16_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter2 vector 2 experimental: kTileN=16 candidate_score variant.

    Same FFI as hisa_candidate_score_tilen_indexer_cache_nvfp4 but the
    underlying kernel binds kTileN=16, doubling Q-dequant amortization at
    the cost of looping stages A/C twice per CTA. Opt-in via env var.
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_tilen16_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_tilen32_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter3 vector 4: kTileN=32 candidate_score variant.

    32x Q dequant amortization vs iter1 per-candidate kernel
    (4x more than iter2 kTileN=8). Trades grid CTA count (drops 4x
    vs kTileN=8) for larger per-CTA SMEM (~32 KB) and longer
    stage A/B loops (kStageARounds=4 rounds at kMaxWarps=8). Suitable
    when the cand_score kernel is launch-bound or wave-tail-bound at
    very-large batch.
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_tilen32_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter5 PRIMARY: WMMA candidate_score variant (kTileN=32).

    Replaces the iter3 tilen32 scalar fmuladd + warp_sum Stage B with a
    B200 SM_100 tensor-core ``mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32``
    over fp16 inputs into an fp32 accumulator. Same FFI as the iter3
    tilen32 wrapper; routes internally to one of three kMaxHeads
    instantiations (n_heads<=8, <=16, <=64). For n_heads > 64 callers
    must use the scalar tilen32 wrapper.
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_persistent_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter3 vector 1: persistent-block candidate_score (kTileN=8).

    Each CTA is bound to one (row, split) pair. Within the CTA, Q is
    decoded ONCE for the row, then the CTA sweeps a contiguous range of
    tile-blocks of that row, amortizing Q HBM traffic across many tiles
    (32-64 tiles per CTA in production, vs 1 tile per CTA in iter2 tile-N).

    Drop-in replacement for hisa_candidate_score_tilen_indexer_cache_nvfp4
    with output semantics identical to the iter1 per-candidate kernel:
    logits[Q, candidate_len] (-INFINITY at invalid token slots),
    candidate_indices[Q, candidate_len] (-1 at invalid).
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_persistent_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_persistent_tilen16_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter3 vector 1 experimental: persistent-block kTileN=16 variant."""
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_persistent_tilen16_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    top_blocks: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter4 PRIMARY: persistent-block + cp.async K-row prefetch.

    Drop-in replacement for hisa_candidate_score_persistent_indexer_cache_nvfp4
    (kTileN=8 variant). Pairs the iter3 v1 persistent kernel with a 2-stage
    ping-pong cp.async prefetch of raw NVFP4 K bytes so the per-tile HBM
    K-decode latency hides under Stage B/C compute. Outputs are bit-
    identical to iter3 v1 modulo fp32 reduction order; the only kernel-
    level change is WHERE the raw NVFP4 bytes are read from (SMEM after
    cp.async lands them, vs HBM via load_nvfp4_value in iter3 v1).
    """
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    weights = weights.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not top_blocks.is_contiguous():
        top_blocks = top_blocks.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()

    candidate_len = top_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.float32, device=q_values.device
    )
    candidate_indices = torch.empty(
        (q_values.shape[0], candidate_len), dtype=torch.int32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale,
        page_table,
        seq_lens.to(torch.int32),
        weights,
        top_blocks.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        logits,
        candidate_indices,
    )
    return logits, candidate_indices


@debug_kernel_api
def hisa_block_score_indexer_cache_nvfp4(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    reps: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if weights.dtype != torch.float32:
        weights = weights.float()
    if q_values.dtype != torch.uint8:
        raise ValueError(f"NVFP4 HISA q values must be uint8, got {q_values.dtype}.")
    if q_scales.dtype != torch.int32:
        q_scales = q_scales.to(torch.int32)
    q_values = q_values.contiguous()
    q_scales = q_scales.contiguous()
    reps = reps.contiguous()
    weights = weights.contiguous()
    if not seq_lens.is_contiguous():
        seq_lens = seq_lens.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    block_scores = torch.empty(
        (q_values.shape[0], reps.shape[1]), dtype=torch.float32, device=q_values.device
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        reps,
        weights,
        seq_lens.to(torch.int32),
        token_to_batch_idx.to(torch.int32),
        block_scores,
    )
    return block_scores


@debug_kernel_api
def hisa_block_topk_indexer_cache_nvfp4(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    block_scores = block_scores.contiguous()
    if block_counts.dtype != torch.int32:
        block_counts = block_counts.to(torch.int32)
    block_counts = block_counts.contiguous()
    if block_topk_counts is None:
        block_topk_counts = torch.full_like(block_counts, block_topk)
    elif block_topk_counts.dtype != torch.int32:
        block_topk_counts = block_topk_counts.to(torch.int32)
    block_topk_counts = block_topk_counts.contiguous()
    top_blocks = torch.empty(
        (block_scores.shape[0], block_topk),
        dtype=torch.int32,
        device=block_scores.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_topk_indexer_cache_nvfp4(
        block_scores, block_counts, block_topk_counts, top_blocks
    )
    return top_blocks


@debug_kernel_api
def hisa_block_topk_map_all_indexer_cache_nvfp4(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    block_topk: int,
    prefix_lens: torch.Tensor,
    block_topk_counts: Optional[torch.Tensor] = None,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    block_scores = block_scores.contiguous()
    if block_counts.dtype != torch.int32:
        block_counts = block_counts.to(torch.int32)
    block_counts = block_counts.contiguous()
    if block_topk_counts is None:
        block_topk_counts = torch.full_like(block_counts, block_topk)
    elif block_topk_counts.dtype != torch.int32:
        block_topk_counts = block_topk_counts.to(torch.int32)
    block_topk_counts = block_topk_counts.contiguous()
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (block_scores.shape[0], block_topk * 128),
        dtype=torch.int32,
        device=block_scores.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_block_topk_map_all_indexer_cache_nvfp4(
        block_scores, block_counts, block_topk_counts, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_candidate_pages_indexer_cache_nvfp4(
    top_blocks: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if token_to_batch_idx.dtype != torch.int32:
        token_to_batch_idx = token_to_batch_idx.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    page_table = page_table.contiguous()
    token_to_batch_idx = token_to_batch_idx.contiguous()
    candidate_page_table = torch.empty(
        (top_blocks.shape[0], top_blocks.shape[1] * 2),
        dtype=page_table.dtype,
        device=page_table.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx, candidate_page_table
    )
    return candidate_page_table


@debug_kernel_api
def hisa_mask_candidate_logits_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    candidate_indices = torch.empty_like(logits, dtype=torch.int32)
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_mask_candidate_logits_indexer_cache_nvfp4(
        logits, top_blocks, prefix_lens, candidate_indices
    )
    return candidate_indices


@debug_kernel_api
def hisa_mask_logits_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> None:
    if not logits.is_contiguous():
        raise ValueError("NVFP4 HISA logits mask requires a contiguous logits tensor.")
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_mask_logits_indexer_cache_nvfp4(logits, top_blocks, prefix_lens)


@debug_kernel_api
def hisa_map_topk_indices_indexer_cache_nvfp4(
    relevant_topk_indices: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if relevant_topk_indices.dtype != torch.int64:
        relevant_topk_indices = relevant_topk_indices.to(torch.int64)
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    relevant_topk_indices = relevant_topk_indices.contiguous()
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        relevant_topk_indices.shape,
        dtype=torch.int32,
        device=relevant_topk_indices.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_map_topk_indices_indexer_cache_nvfp4(
        relevant_topk_indices, top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_fused_mask_topk_map_indexer_cache_nvfp4(
    logits: torch.Tensor,
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_tokens: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    """Fused exact mask + radix top-k + position-to-token map."""
    if not logits.is_contiguous():
        raise ValueError("NVFP4 HISA fused mask+topk+map requires contiguous logits.")
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (logits.shape[0], topk_tokens),
        dtype=torch.int32,
        device=logits.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_fused_mask_topk_map_indexer_cache_nvfp4(
        logits, top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


@debug_kernel_api
def hisa_selector_megakernel_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    block_counts: torch.Tensor,
    block_topk_counts: torch.Tensor,
    effective_block_topk: int,
    topk_tokens: int,
    page_size: int = 64,
) -> torch.Tensor:
    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_counts = block_counts.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_topk_counts = block_topk_counts.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not rep_values.is_contiguous():
        rep_values = rep_values.contiguous()
    if not rep_scales.is_contiguous():
        rep_scales = rep_scales.contiguous()

    topk_indices = torch.empty(
        (q_values.shape[0], topk_tokens),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_selector_megakernel_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        rep_values,
        rep_scales,
        block_counts,
        block_topk_counts,
        topk_indices,
        max_blocks,
        effective_block_topk,
    )
    return topk_indices


@debug_kernel_api
def hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    block_counts: torch.Tensor,
    block_topk_counts: torch.Tensor,
    effective_block_topk: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_counts = block_counts.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_topk_counts = block_topk_counts.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not rep_values.is_contiguous():
        rep_values = rep_values.contiguous()
    if not rep_scales.is_contiguous():
        rep_scales = rep_scales.contiguous()
    selected_blocks = torch.empty(
        (q_values.shape[0], effective_block_topk),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
        q_values,
        q_scales,
        seq_lens,
        weights,
        token_to_batch_idx,
        rep_values,
        rep_scales,
        block_counts,
        block_topk_counts,
        selected_blocks,
        max_blocks,
        effective_block_topk,
    )
    return selected_blocks


@debug_kernel_api
def hisa_selector_parallel_score_candidates_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    selected_blocks: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    q_values, q_scales = q_fp4
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    candidate_len = selected_blocks.shape[1] * 128
    logits = torch.empty(
        (q_values.shape[0], candidate_len),
        dtype=torch.float32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_selector_parallel_score_candidates_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        selected_blocks,
        logits,
    )
    return logits


@debug_kernel_api
def hisa_deepgemm_candidate_logits_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    candidate_context_lens: torch.Tensor,
    candidate_page_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    candidate_len: int,
    page_size: int = 64,
) -> torch.Tensor:
    """Diagnostic source-level FP4 scorer; not a complete fused HISA selector."""
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA DeepGEMM-source candidate probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA DeepGEMM-source candidate probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if candidate_context_lens.dim() == 1:
        candidate_context_lens = candidate_context_lens.unsqueeze(1)
    candidate_context_lens = candidate_context_lens.to(
        device=q_values.device, dtype=torch.int32
    )
    if candidate_page_table.dtype != torch.int32:
        candidate_page_table = candidate_page_table.to(torch.int32)
    if schedule_metadata.dtype != torch.int32:
        schedule_metadata = schedule_metadata.to(torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not candidate_context_lens.is_contiguous():
        candidate_context_lens = candidate_context_lens.contiguous()
    if not candidate_page_table.is_contiguous():
        candidate_page_table = candidate_page_table.contiguous()
    if not schedule_metadata.is_contiguous():
        schedule_metadata = schedule_metadata.contiguous()

    logits = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.float32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, candidate_page_table.dtype, page_size
    ).hisa_deepgemm_candidate_logits_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        candidate_context_lens,
        candidate_page_table,
        schedule_metadata,
        logits,
    )
    return logits


@debug_kernel_api
def hisa_deepgemm_candidate_keys_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    candidate_context_lens: torch.Tensor,
    candidate_page_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    selected_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    candidate_len: int,
    page_size: int = 64,
) -> torch.Tensor:
    """Diagnostic collective FP4 scorer that emits masked sortable keys."""
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA DeepGEMM-source candidate key probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA DeepGEMM-source candidate key probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if candidate_context_lens.dim() == 1:
        candidate_context_lens = candidate_context_lens.unsqueeze(1)
    candidate_context_lens = candidate_context_lens.to(
        device=q_values.device, dtype=torch.int32
    )
    if candidate_page_table.dtype != torch.int32:
        candidate_page_table = candidate_page_table.to(torch.int32)
    if schedule_metadata.dtype != torch.int32:
        schedule_metadata = schedule_metadata.to(torch.int32)
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    prefix_lens = prefix_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not candidate_context_lens.is_contiguous():
        candidate_context_lens = candidate_context_lens.contiguous()
    if not candidate_page_table.is_contiguous():
        candidate_page_table = candidate_page_table.contiguous()
    if not schedule_metadata.is_contiguous():
        schedule_metadata = schedule_metadata.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not prefix_lens.is_contiguous():
        prefix_lens = prefix_lens.contiguous()

    candidate_keys = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.int32,
        device=q_values.device,
    )
    empty_source_page_table = torch.empty(
        (0, 1), dtype=torch.int32, device=q_values.device
    )
    empty_token_to_batch_idx = torch.empty(
        (0,), dtype=torch.int32, device=q_values.device
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, candidate_page_table.dtype, page_size
    ).hisa_deepgemm_candidate_keys_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        candidate_context_lens,
        candidate_page_table,
        empty_source_page_table,
        schedule_metadata,
        selected_blocks,
        prefix_lens,
        empty_token_to_batch_idx,
        candidate_keys,
    )
    return candidate_keys


@debug_kernel_api
def hisa_deepgemm_candidate_keys_from_blocks_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    candidate_context_lens: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    schedule_metadata: torch.Tensor,
    selected_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    candidate_len: int,
    page_size: int = 64,
) -> torch.Tensor:
    """Collective FP4 scorer with in-kernel candidate page derivation."""
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA DeepGEMM-source candidate key probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA DeepGEMM-source candidate key probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if page_table.dtype != torch.int32:
        raise ValueError(
            "HISA page-fused candidate key scorer currently requires int32 page_table."
        )
    if candidate_context_lens.dim() == 1:
        candidate_context_lens = candidate_context_lens.unsqueeze(1)
    candidate_context_lens = candidate_context_lens.to(
        device=q_values.device, dtype=torch.int32
    )
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if schedule_metadata.dtype != torch.int32:
        schedule_metadata = schedule_metadata.to(torch.int32)
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    prefix_lens = prefix_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not candidate_context_lens.is_contiguous():
        candidate_context_lens = candidate_context_lens.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    if not schedule_metadata.is_contiguous():
        schedule_metadata = schedule_metadata.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not prefix_lens.is_contiguous():
        prefix_lens = prefix_lens.contiguous()

    candidate_page_table = torch.empty(
        (q_values.shape[0], 0),
        dtype=torch.int32,
        device=q_values.device,
    )
    candidate_keys = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, torch.int32, page_size
    ).hisa_deepgemm_candidate_keys_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        candidate_context_lens,
        candidate_page_table,
        page_table,
        schedule_metadata,
        selected_blocks,
        prefix_lens,
        token_to_batch_idx,
        candidate_keys,
    )
    return candidate_keys


@debug_kernel_api
def hisa_deepgemm_candidate_topk_cooperative_from_blocks_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    candidate_context_lens: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    schedule_metadata: torch.Tensor,
    selected_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    candidate_len: int,
    topk_tokens: int,
    page_size: int = 64,
    *,
    return_candidate_keys: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cooperative-grid FP4 scorer with in-kernel candidate top-k/map.

    The scoring phase keeps the DeepGEMM persistent SM100 FP4 schedule. The
    kernel then uses a cooperative grid barrier and reuses resident CTAs to run
    the score-key top-k/map tail without a second launch.
    """
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA cooperative candidate topk probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA cooperative candidate topk probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if page_table.dtype != torch.int32:
        raise ValueError(
            "HISA cooperative candidate topk probe currently requires int32 page_table."
        )
    topk_tokens = int(topk_tokens)
    if topk_tokens <= 0:
        raise ValueError("HISA cooperative candidate topk probe requires positive topk.")
    if candidate_context_lens.dim() == 1:
        candidate_context_lens = candidate_context_lens.unsqueeze(1)
    candidate_context_lens = candidate_context_lens.to(
        device=q_values.device, dtype=torch.int32
    )
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if token_to_batch_idx.numel() < q_values.shape[0]:
        raise ValueError(
            "HISA cooperative candidate topk probe needs token_to_batch_idx per row."
        )
    if schedule_metadata.dtype != torch.int32:
        schedule_metadata = schedule_metadata.to(torch.int32)
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    prefix_lens = prefix_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not candidate_context_lens.is_contiguous():
        candidate_context_lens = candidate_context_lens.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    if not schedule_metadata.is_contiguous():
        schedule_metadata = schedule_metadata.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not prefix_lens.is_contiguous():
        prefix_lens = prefix_lens.contiguous()

    candidate_keys = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.int32,
        device=q_values.device,
    )
    topk_indices = torch.empty(
        (q_values.shape[0], topk_tokens),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, torch.int32, page_size
    ).hisa_deepgemm_candidate_topk_cooperative_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        candidate_context_lens,
        page_table,
        schedule_metadata,
        selected_blocks,
        prefix_lens,
        token_to_batch_idx,
        candidate_keys,
        topk_indices,
    )
    if return_candidate_keys:
        return topk_indices, candidate_keys
    return topk_indices


@debug_kernel_api
def hisa_deepgemm_candidate_keys_row_split_from_blocks_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    selected_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    candidate_len: int,
    *,
    row_splits: int = 1,
    page_size: int = 64,
) -> torch.Tensor:
    """Row-owned SM100 FP4 scorer probe for future fused candidate top-k."""
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA row-split candidate key probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA row-split candidate key probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if page_size != 64:
        raise ValueError("HISA row-split candidate key probe currently requires page_size=64.")
    if page_table.dtype != torch.int32:
        raise ValueError(
            "HISA row-split candidate key scorer currently requires int32 page_table."
        )
    row_splits = max(1, int(row_splits))
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if token_to_batch_idx.numel() < q_values.shape[0]:
        raise ValueError("HISA row-split candidate key scorer needs token_to_batch_idx per row.")
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    prefix_lens = prefix_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not prefix_lens.is_contiguous():
        prefix_lens = prefix_lens.contiguous()

    candidate_keys = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, torch.int32, page_size
    ).hisa_deepgemm_candidate_keys_row_split_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        page_table,
        selected_blocks,
        prefix_lens,
        token_to_batch_idx,
        candidate_keys,
        row_splits,
    )
    return candidate_keys


@debug_kernel_api
def hisa_deepgemm_candidate_topk_cluster_from_blocks_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    selected_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    candidate_len: int,
    topk_tokens: int,
    *,
    page_size: int = 64,
    row_splits: int = 4,
    use_cluster: bool = True,
    return_candidate_keys: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cluster-launched FP4 candidate scorer with fused candidate top-k/map."""
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[2] != 64:
        raise ValueError(
            "HISA candidate topk cluster probe expects q_values [rows, 64, 64]."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape != q_scales.shape:
        raise ValueError(
            f"HISA candidate topk cluster probe expects weights {q_scales.shape}, got {weights.shape}."
        )
    if page_size != 64:
        raise ValueError("HISA candidate topk cluster probe currently requires page_size=64.")
    if page_table.dtype != torch.int32:
        raise ValueError(
            "HISA candidate topk cluster probe currently requires int32 page_table."
        )
    topk_tokens = int(topk_tokens)
    if topk_tokens <= 0:
        raise ValueError("HISA candidate topk cluster probe requires positive topk_tokens.")
    row_splits = int(row_splits)
    if row_splits <= 0:
        raise ValueError("HISA candidate topk cluster probe requires positive row_splits.")
    if use_cluster and row_splits != 4:
        raise ValueError("HISA candidate topk cluster probe requires row_splits=4.")
    if not use_cluster and row_splits != 1:
        raise ValueError("HISA non-cluster candidate topk probe requires row_splits=1.")
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if token_to_batch_idx.numel() < q_values.shape[0]:
        raise ValueError("HISA candidate topk cluster probe needs token_to_batch_idx per row.")
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    prefix_lens = prefix_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if not token_to_batch_idx.is_contiguous():
        token_to_batch_idx = token_to_batch_idx.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    if not prefix_lens.is_contiguous():
        prefix_lens = prefix_lens.contiguous()

    candidate_keys = torch.empty(
        (q_values.shape[0], int(candidate_len)),
        dtype=torch.int32,
        device=q_values.device,
    )
    topk_indices = torch.empty(
        (q_values.shape[0], topk_tokens),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, torch.int32, page_size
    ).hisa_deepgemm_candidate_topk_cluster_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        weights,
        page_table,
        selected_blocks,
        prefix_lens,
        token_to_batch_idx,
        candidate_keys,
        topk_indices,
        row_splits,
        int(bool(use_cluster)),
    )
    if return_candidate_keys:
        return topk_indices, candidate_keys
    return topk_indices


@debug_kernel_api
def hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
    candidate_keys: torch.Tensor,
    selected_blocks: torch.Tensor,
    topk_tokens: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if candidate_keys.dtype != torch.int32:
        raise ValueError("HISA candidate key topk requires int32 candidate keys.")
    if selected_blocks.dtype != torch.int32:
        selected_blocks = selected_blocks.to(torch.int32)
    if not candidate_keys.is_contiguous():
        candidate_keys = candidate_keys.contiguous()
    if not selected_blocks.is_contiguous():
        selected_blocks = selected_blocks.contiguous()
    topk_indices = torch.empty(
        (candidate_keys.shape[0], int(topk_tokens)),
        dtype=torch.int32,
        device=candidate_keys.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
        candidate_keys,
        selected_blocks,
        topk_indices,
    )
    return topk_indices


@debug_kernel_api
def hisa_selector_cluster_fused_indexer_cache_nvfp4(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    block_counts: torch.Tensor,
    block_topk_counts: torch.Tensor,
    effective_block_topk: int,
    topk_tokens: int,
    page_size: int = 64,
) -> torch.Tensor:
    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if not index_k_with_scale_buffer.is_contiguous():
        index_k_with_scale_buffer = index_k_with_scale_buffer.contiguous()
    if not page_table.is_contiguous():
        page_table = page_table.contiguous()
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if not weights.is_contiguous():
        weights = weights.contiguous()
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_counts = block_counts.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    block_topk_counts = block_topk_counts.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    if not q_values.is_contiguous():
        q_values = q_values.contiguous()
    if not q_scales.is_contiguous():
        q_scales = q_scales.contiguous()
    if not rep_values.is_contiguous():
        rep_values = rep_values.contiguous()
    if not rep_scales.is_contiguous():
        rep_scales = rep_scales.contiguous()

    candidate_len = int(effective_block_topk) * 128
    candidate_keys = torch.empty(
        (q_values.shape[0], candidate_len),
        dtype=torch.int32,
        device=q_values.device,
    )
    topk_indices = torch.empty(
        (q_values.shape[0], topk_tokens),
        dtype=torch.int32,
        device=q_values.device,
    )
    _get_module_hisa_selector_megakernel(
        torch.bfloat16, page_table.dtype, page_size
    ).hisa_selector_cluster_fused_indexer_cache_nvfp4(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        page_table,
        seq_lens,
        weights,
        token_to_batch_idx,
        rep_values,
        rep_scales,
        block_counts,
        block_topk_counts,
        candidate_keys,
        topk_indices,
        max_blocks,
        effective_block_topk,
    )
    return topk_indices


@debug_kernel_api
def hisa_map_candidate_indices_indexer_cache_nvfp4(
    top_blocks: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk_tokens: int,
    page_table_dtype: torch.dtype = torch.int32,
    page_size: int = 64,
) -> torch.Tensor:
    if top_blocks.dtype != torch.int32:
        top_blocks = top_blocks.to(torch.int32)
    if prefix_lens.dtype != torch.int32:
        prefix_lens = prefix_lens.to(torch.int32)
    top_blocks = top_blocks.contiguous()
    prefix_lens = prefix_lens.contiguous()
    topk_indices = torch.empty(
        (top_blocks.shape[0], topk_tokens),
        dtype=torch.int32,
        device=top_blocks.device,
    )
    _get_module_fast(
        torch.bfloat16, page_table_dtype, page_size
    ).hisa_map_candidate_indices_indexer_cache_nvfp4(
        top_blocks, prefix_lens, topk_indices
    )
    return topk_indices


def hisa_precompute_block_reps_indexer_cache_nvfp4(
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    block_size: int = 128,
    page_size: int = 64,
    max_blocks: Optional[int] = None,
    return_float_reps: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int] | tuple[tuple[torch.Tensor, torch.Tensor], int, torch.Tensor]:
    if block_size != 128:
        raise ValueError("NVFP4 HISA precomputed reps require block_size=128.")
    seq_lens_flat = seq_lens.reshape(-1).to(device=page_table.device, dtype=torch.int32)
    if max_blocks is None:
        max_blocks = _hisa_max_blocks(seq_lens_flat, page_table, block_size)
    else:
        max_blocks = max(1, int(max_blocks))
    reps = _hisa_mean_pool_call(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
        page_size=page_size,
    )
    rep_values, rep_scales = quantize_indexer_q_nvfp4(
        reps.reshape(-1, reps.shape[-1]).to(torch.bfloat16),
        indices_dtype=page_table.dtype,
        page_size=page_size,
    )
    rep_fp4 = (rep_values, rep_scales)
    if return_float_reps:
        return rep_fp4, max_blocks, reps
    return rep_fp4, max_blocks


def _profile_start(device: torch.device) -> Optional[float]:
    if not _hisa_profile_path:
        return None
    if _hisa_profile_sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_end(
    stage: str, start: Optional[float], device: torch.device, **record
) -> None:
    if start is None or not _hisa_profile_path:
        return
    if _hisa_profile_sync and device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "time": time.time(),
        "path": f"hisa_nvfp4:{stage}",
        "duration_ms": (time.perf_counter() - start) * 1000.0,
        **record,
    }
    directory = os.path.dirname(_hisa_profile_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _hisa_profile_lock:
        with open(_hisa_profile_path, "a") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def dequantize_indexer_nvfp4(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize IndexCache NVFP4/UE8M0 rows to fp32."""

    if values.shape[-1] != 64:
        raise ValueError(
            f"NVFP4 IndexCache values expect 64 packed bytes, got {values.shape}."
        )

    original_shape = values.shape
    if values.is_cuda:
        values_2d = values.reshape(-1, original_shape[-1])
        if values_2d.dtype != torch.uint8:
            values_2d = values_2d.to(torch.uint8)
        if not values_2d.is_contiguous():
            values_2d = values_2d.contiguous()
        scales_1d = scales.reshape(-1)
        if scales_1d.dtype != torch.int32:
            scales_1d = scales_1d.to(torch.int32)
        if not scales_1d.is_contiguous():
            scales_1d = scales_1d.contiguous()
        if scales_1d.numel() != values_2d.shape[0]:
            raise ValueError(
                "NVFP4 IndexCache scales must have one int32 word per values row."
            )
        output = torch.empty(
            (values_2d.shape[0], 128), dtype=torch.float32, device=values.device
        )
        _get_module_fast(
            torch.bfloat16, torch.int64, 64
        ).dequantize_indexer_nvfp4(values_2d, scales_1d, output)
        return output.view(*original_shape[:-1], 128)

    values_u8 = values.to(torch.uint8)
    low = values_u8 & 0x0F
    high = (values_u8 >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*values.shape[:-1], 128)

    codebook = torch.tensor(
        _NVFP4_E2M1_CODEBOOK, device=values.device, dtype=torch.float32
    )
    magnitudes = codebook[(codes & 0x07).long()]
    signed = torch.where((codes & 0x08).bool(), -magnitudes, magnitudes)

    scale_words = scales.to(torch.int64) & 0xFFFFFFFF
    exponents = torch.stack(
        tuple(((scale_words >> shift) & 0xFF) for shift in (0, 8, 16, 24)),
        dim=-1,
    ).to(torch.float32)
    group_scales = torch.pow(torch.full_like(exponents, 2.0), exponents - 127.0)
    return signed * torch.repeat_interleave(group_scales, 32, dim=-1)


def _unpack_paged_nvfp4_cache(
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = 64,
) -> list[torch.Tensor]:
    rows_by_batch = []
    for batch_idx in range(page_table.shape[0]):
        seq_len = int(seq_lens.reshape(-1)[batch_idx].item())
        num_pages = (seq_len + page_size - 1) // page_size
        pages = page_table[batch_idx, :num_pages].long().clamp_min(0)
        packed_pages = index_k_with_scale_buffer.index_select(0, pages)
        values = packed_pages[:, : page_size * 64].reshape(-1, 64)[:seq_len]
        scale_bytes = packed_pages[:, page_size * 64 : page_size * 68]
        scales = scale_bytes.reshape(-1, 4)[:seq_len].contiguous()
        scales = scales.view(torch.int32).reshape(-1)
        rows_by_batch.append(dequantize_indexer_nvfp4(values, scales))
    return rows_by_batch


def _mean_pool_blocks(k_rows: torch.Tensor, block_size: int) -> torch.Tensor:
    block_count = (k_rows.shape[0] + block_size - 1) // block_size
    reps = []
    for block_id in range(block_count):
        start = block_id * block_size
        end = min(start + block_size, k_rows.shape[0])
        reps.append(k_rows[start:end].mean(dim=0))
    if not reps:
        return k_rows.new_zeros((0, k_rows.shape[-1]))
    return torch.stack(reps, dim=0)


def _weighted_relu_dsa_score(
    q_rows: torch.Tensor, k_rows: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    logits = torch.einsum("qhd,kd->qkh", q_rows.float(), k_rows.float())
    logits = torch.relu(logits) * weights.float().unsqueeze(1)
    return logits.sum(dim=-1)


def _forced_hisa_blocks(block_count: int, device: torch.device) -> torch.Tensor:
    if block_count <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    forced = [0]
    forced.append(block_count - 1)
    return torch.tensor(sorted(set(forced)), device=device, dtype=torch.long)


def _hisa_block_topk_counts(
    block_counts: torch.Tensor,
    *,
    block_size: int,
    topk_tokens: int,
    compression_ratio: Optional[float],
    max_block_topk: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    if compression_ratio is None or compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive for dynamic HISA budgets.")
    if abs(compression_ratio - round(compression_ratio)) < 1e-6:
        ratio = int(round(compression_ratio))
        selected = torch.div(block_counts + ratio - 1, ratio, rounding_mode="floor")
    else:
        selected = torch.ceil(block_counts.float() / compression_ratio).to(torch.int32)
    selected = torch.minimum(selected, block_counts)
    selected = torch.where(block_counts > 0, selected, torch.zeros_like(selected))
    if max_block_topk is not None:
        selected = torch.minimum(selected, torch.full_like(selected, max_block_topk))
    if _is_cuda_graph_capturing(block_counts):
        if max_block_topk is None:
            raise RuntimeError("CUDA graph HISA block top-k requires a static upper bound.")
        return selected.to(torch.int32), max(1, max_block_topk)
    max_selected = int(selected.max().item()) if selected.numel() else 0
    return selected.to(torch.int32), max(1, max_selected)


def _hisa_block_topk_counts_static_width(
    block_counts: torch.Tensor,
    *,
    block_size: int,
    compression_ratio: Optional[float],
    max_blocks: int,
) -> tuple[torch.Tensor, int]:
    if compression_ratio is None or compression_ratio <= 0:
        raise ValueError("compression_ratio must be positive for dynamic HISA budgets.")
    if abs(compression_ratio - round(compression_ratio)) < 1e-6:
        ratio = int(round(compression_ratio))
        selected = torch.div(block_counts + ratio - 1, ratio, rounding_mode="floor")
        effective = (int(max_blocks) + ratio - 1) // ratio
    else:
        selected = torch.ceil(block_counts.float() / compression_ratio).to(torch.int32)
        effective = math.ceil(float(max_blocks) / float(compression_ratio))
    selected = torch.minimum(selected, block_counts)
    selected = torch.where(block_counts > 0, selected, torch.zeros_like(selected))
    effective = max(1, int(effective))
    selected = torch.minimum(selected, torch.full_like(selected, effective))
    return selected.to(torch.int32), effective


def _select_hisa_blocks(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    selected = torch.full(
        (block_scores.shape[0], block_topk),
        -1,
        device=block_scores.device,
        dtype=torch.int32,
    )
    for row in range(block_scores.shape[0]):
        block_count = int(block_counts[row].item())
        if block_count == 0:
            continue
        scores = block_scores[row, :block_count].clone()
        scores[_forced_hisa_blocks(block_count, block_scores.device)] = float("inf")
        row_topk = (
            int(block_topk_counts[row].item())
            if block_topk_counts is not None
            else block_topk
        )
        keep = min(row_topk, block_topk, block_count)
        selected[row, :keep] = torch.topk(scores, k=keep, sorted=False).indices.to(
            torch.int32
        )
    return selected


def _select_hisa_blocks_vectorized(
    block_scores: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    block_topk: int,
    block_topk_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if block_scores.shape[1] == 0:
        return torch.full(
            (block_scores.shape[0], block_topk),
            -1,
            device=block_scores.device,
            dtype=torch.int32,
        )
    scores = block_scores.clone()
    rows = torch.arange(scores.shape[0], device=scores.device)
    valid = block_counts > 0
    if bool(valid.any().item()):
        valid_rows = rows[valid]
        counts = block_counts[valid].long()
        scores[valid_rows, 0] = float("inf")
        scores[valid_rows, counts - 1] = float("inf")
    if block_topk_counts is not None and bool((block_topk_counts != block_topk).any().item()):
        return _select_hisa_blocks(
            block_scores,
            block_counts,
            block_topk=block_topk,
            block_topk_counts=block_topk_counts,
        )
    keep = min(block_topk, scores.shape[1])
    top_blocks = torch.topk(scores, k=keep, dim=-1, sorted=False).indices.to(torch.int32)
    if keep == block_topk:
        return top_blocks
    selected = torch.full(
        (scores.shape[0], block_topk),
        -1,
        device=scores.device,
        dtype=torch.int32,
    )
    selected[:, :keep] = top_blocks
    return selected


def nvfp4_hisa_indexer_from_dequant(
    q: torch.Tensor,
    k_by_batch: list[torch.Tensor],
    weights: torch.Tensor,
    prefix_lens: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    """Reference HISA selection over already-dequantized NVFP4 rows."""

    prefix_lens = prefix_lens.reshape(-1).to(device=q.device, dtype=torch.int64)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q.device, dtype=torch.long
    )
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None

    q = q.float()
    weights = weights.float()
    block_counts_by_batch = [
        (rows.shape[0] + block_size - 1) // block_size for rows in k_by_batch
    ]
    max_blocks = max(block_counts_by_batch, default=0)

    stage = _profile_start(q.device)
    reps_by_batch = [_mean_pool_blocks(rows.to(q.device), block_size) for rows in k_by_batch]
    _profile_end("mean_pool", stage, q.device, batches=len(k_by_batch), max_blocks=max_blocks)

    block_scores = q.new_full((q.shape[0], max_blocks), float("-inf"))
    block_counts = torch.empty((q.shape[0],), device=q.device, dtype=torch.int32)
    stage = _profile_start(q.device)
    for batch_idx, reps in enumerate(reps_by_batch):
        row_mask = token_to_batch_idx == batch_idx
        if not bool(row_mask.any().item()) or reps.numel() == 0:
            continue
        rows = row_mask.nonzero(as_tuple=False).flatten()
        scores = _weighted_relu_dsa_score(
            q.index_select(0, rows), reps, weights.index_select(0, rows)
        )
        block_scores[rows, : scores.shape[1]] = scores
        block_counts[rows] = scores.shape[1]
    _profile_end("block_score", stage, q.device, rows=int(q.shape[0]), max_blocks=max_blocks)

    stage = _profile_start(q.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    top_blocks = _select_hisa_blocks(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
    )
    _profile_end(
        "block_topk",
        stage,
        q.device,
        rows=int(q.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
    )

    out = torch.full((q.shape[0], topk_tokens), -1, device=q.device, dtype=torch.int32)
    for row in range(q.shape[0]):
        prefix_len = int(prefix_lens[row].item())
        if fallback_to_dense_if_short and prefix_len <= topk_tokens:
            continue
        batch_idx = int(token_to_batch_idx[row].item())
        k_rows = k_by_batch[batch_idx].to(q.device)
        valid_blocks = top_blocks[row][top_blocks[row] >= 0].long()
        if valid_blocks.numel() == 0:
            continue

        stage = _profile_start(q.device)
        candidate_ranges = []
        for block_id in valid_blocks.tolist():
            start = block_id * block_size
            end = min(start + block_size, prefix_len)
            if start < end:
                candidate_ranges.append(torch.arange(start, end, device=q.device))
        if not candidate_ranges:
            _profile_end("candidate_dequant", stage, q.device, row=row, candidates=0)
            continue
        candidate_indices = torch.cat(candidate_ranges).unique(sorted=False)
        candidate_k = k_rows.index_select(0, candidate_indices)
        _profile_end(
            "candidate_dequant",
            stage,
            q.device,
            row=row,
            candidates=int(candidate_indices.numel()),
        )

        stage = _profile_start(q.device)
        candidate_scores = _weighted_relu_dsa_score(
            q[row : row + 1], candidate_k, weights[row : row + 1]
        ).squeeze(0)
        _profile_end(
            "candidate_score",
            stage,
            q.device,
            row=row,
            candidates=int(candidate_indices.numel()),
        )

        stage = _profile_start(q.device)
        keep = min(topk_tokens, candidate_scores.numel())
        rel = torch.topk(candidate_scores, k=keep, sorted=False).indices
        out[row, :keep] = candidate_indices.index_select(0, rel).to(torch.int32)
        _profile_end("candidate_topk", stage, q.device, row=row, topk=int(keep))

    stage = _profile_start(q.device)
    _profile_end("map/store", stage, q.device, rows=int(q.shape[0]), topk_tokens=topk_tokens)
    return out


def nvfp4_hisa_indexer_paged_torch(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[-1] != 64:
        raise ValueError(
            f"NVFP4 HISA q values expect [tokens, heads, 64], got {q_values.shape}."
        )
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.long
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None

    max_blocks = _hisa_max_blocks(seq_lens_flat, page_table, block_size)

    stage = _profile_start(q_values.device)
    reps = _hisa_mean_pool_call(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
    )
    _profile_end(
        "mean_pool",
        stage,
        q_values.device,
        batches=int(page_table.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
    )

    block_counts = torch.div(
        prefix_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    stage = _profile_start(q_values.device)
    block_scores = hisa_block_score_indexer_cache_nvfp4(
        q_values,
        q_scales,
        reps,
        weights,
        seq_lens_flat,
        token_to_batch_idx.to(torch.int32),
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
            max_block_topk=(
                block_topk if _is_cuda_graph_capturing(block_counts) else None
            ),
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    if not _hisa_nvfp4_candidate_score_tilen:
        _candidate_score_fn = hisa_candidate_score_indexer_cache_nvfp4
    elif _hisa_nvfp4_candidate_score_kprefetch:
        # iter4 PRIMARY OPT-IN: persistent-block + cp.async K-row prefetch.
        # Pairs the iter3 v1 persistent kernel with a 2-stage ping-pong
        # cp.async prefetch of raw NVFP4 K bytes into ping-pong SMEM
        # buffers, hiding the per-tile HBM K-decode latency under Stage
        # B/C compute. kTileN=8 only at the iter4 first-cut variant.
        _candidate_score_fn = (
            hisa_candidate_score_persistent_kprefetch_indexer_cache_nvfp4
        )
    elif _hisa_nvfp4_candidate_score_persistent:
        # iter3 vector 1 OPT-IN: persistent-block kernel. Default off
        # because it's strictly slower than tile-N at the production
        # shape grid (see iter3 vector 1 commit body for the table).
        # When forced on by env var, kTileN=8 is the primary; kTileN=16
        # is opt-in via SGLANG_NSA_NVFP4_HISA_CANDIDATE_SCORE_TILEN_SIZE.
        if _hisa_nvfp4_candidate_score_tilen_size == 16:
            _candidate_score_fn = (
                hisa_candidate_score_persistent_tilen16_indexer_cache_nvfp4
            )
        else:
            _candidate_score_fn = (
                hisa_candidate_score_persistent_indexer_cache_nvfp4
            )
    elif _hisa_nvfp4_candidate_score_autotune:
        # iter3 vector 4: per-shape kTileN auto-tune. Bench harness
        # measurements at (n_heads=64, head_dim=128, index_topk=1024,
        # hisa block_size=128, page_size=64, compression_ratio=4.0)
        # show:
        #   * q_rows*candidate_len < ~262144 (32/8192 cell):
        #     kTileN=16 (~1.5% better than kTileN=8, ~0% kTileN=32).
        #   * q_rows*candidate_len < ~524288 (32/16384, 64/32768):
        #     kTileN=16 wins (~3% better than kTileN=8 at 64/32768).
        #   * q_rows*candidate_len >= ~1048576 (128/32768):
        #     kTileN=32 wins (~20% better than kTileN=8, ~5% better
        #     than kTileN=16 — the launch + per-CTA fixed cost
        #     amortization scales linearly with kTileN at this size).
        # The total-work threshold is the natural cut: tile_blocks *
        # q_rows = candidate_len/kTileN * q_rows ∝ wave count, and
        # waves dominate the kernel's wall time at large batch.
        #
        # iter5 PRIMARY: when SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA is
        # set and the cell already routes to kTileN=32 (large-batch
        # path) with n_heads <= 64, swap in the WMMA mma.m16n8k8 fp32
        # Stage B variant. The WMMA variant has identical FFI; the
        # ~16 KB Q SMEM table only pays off at the kTileN=32 cells
        # where Stage B compute dominates the wall time.
        n_rows = int(q_values.shape[0])
        cand_len_proj = int(top_blocks.shape[1]) * 128
        total_work = n_rows * cand_len_proj
        n_heads_proj = int(q_values.shape[1])
        if total_work >= 1048576:
            if (
                _hisa_nvfp4_candidate_score_wmma
                and n_heads_proj <= 64
            ):
                _candidate_score_fn = (
                    hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4
                )
            else:
                _candidate_score_fn = (
                    hisa_candidate_score_tilen32_indexer_cache_nvfp4
                )
        elif total_work >= 262144:
            _candidate_score_fn = (
                hisa_candidate_score_tilen16_indexer_cache_nvfp4
            )
        else:
            _candidate_score_fn = (
                hisa_candidate_score_tilen_indexer_cache_nvfp4
            )
    elif _hisa_nvfp4_candidate_score_tilen_size == 32:
        if (
            _hisa_nvfp4_candidate_score_wmma
            and int(q_values.shape[1]) <= 64
        ):
            _candidate_score_fn = (
                hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4
            )
        else:
            _candidate_score_fn = (
                hisa_candidate_score_tilen32_indexer_cache_nvfp4
            )
    elif _hisa_nvfp4_candidate_score_tilen_size == 16:
        _candidate_score_fn = hisa_candidate_score_tilen16_indexer_cache_nvfp4
    else:
        _candidate_score_fn = hisa_candidate_score_tilen_indexer_cache_nvfp4
    candidate_logits, candidate_indices = _candidate_score_fn(
        q_values,
        q_scales,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        top_blocks,
        token_to_batch_idx.to(torch.int32),
    )
    _profile_end(
        "candidate_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        fused_dequant=True,
        packed_q=True,
    )
    stage = _profile_start(q_values.device)
    _profile_end(
        "candidate_dequant",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        fused_into="candidate_score",
    )

    stage = _profile_start(q_values.device)
    keep = min(topk_tokens, candidate_logits.shape[-1])
    relevant_topk_indices = torch.topk(
        candidate_logits, k=keep, dim=-1, sorted=False
    ).indices
    topk_indices = torch.gather(candidate_indices, dim=1, index=relevant_topk_indices)
    _profile_end(
        "candidate_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=int(candidate_logits.shape[-1]),
        topk_tokens=int(keep),
    )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end("map/store", stage, q_values.device, rows=int(q_values.shape[0]), topk_tokens=topk_tokens)
    return topk_indices.to(torch.int32)


def nvfp4_hisa_indexer_paged_deepgemm_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_starts: Optional[torch.Tensor] = None,
    prepared_block_ends: Optional[torch.Tensor] = None,
    prepared_candidate_context_lens: Optional[torch.Tensor] = None,
    prepared_candidate_schedule_metadata: Optional[object] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if block_size != 128:
        raise ValueError("DeepGEMM-backed NVFP4 HISA path requires block_size=128.")

    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    import deep_gemm

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None

    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_starts is None:
        block_starts = token_to_batch_idx * max_blocks
    else:
        block_starts = prepared_block_starts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_ends is None:
        block_ends = block_starts + block_counts
    else:
        block_ends = prepared_block_ends.to(device=q_values.device, dtype=torch.int32)
    stage = _profile_start(q_values.device)
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        block_starts.to(torch.int32),
        block_ends.to(torch.int32),
        clean_logits=False,
        max_seqlen_k=max_blocks,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "blockscore_precomputed",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        backend="deepgemm_fp4_mqa",
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        # Hot-path fast path: caller already computed per-row counts AND the
        # max on host side, so we avoid the .item() sync inside this function.
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts_static_width(
            block_counts,
            block_size=block_size,
            compression_ratio=compression_ratio,
            max_blocks=max_blocks,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )
    _profile_end(
        "candidate_pages",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        candidate_pages=int(candidate_page_table.shape[1]),
        fused_cuda=True,
    )

    schedule_metadata = prepared_candidate_schedule_metadata
    if prepared_candidate_context_lens is None and schedule_metadata is None:
        context_lens, schedule_metadata = _hisa_uniform_candidate_schedule(
            q_values, candidate_len, deep_gemm
        )
    elif prepared_candidate_context_lens is None:
        context_lens = torch.full(
            (q_values.shape[0], 1),
            candidate_len,
            device=q_values.device,
            dtype=torch.int32,
        )
    else:
        context_lens = prepared_candidate_context_lens.to(
            device=q_values.device, dtype=torch.int32
        )
    if schedule_metadata is None:
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, 64, deep_gemm.get_num_sms()
        )
    stage = _profile_start(q_values.device)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        index_k_with_scale_buffer.view(index_k_with_scale_buffer.shape[0], 64, 1, 68),
        weights,
        context_lens,
        candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "candidate_logits",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
    )

    keep = min(topk_tokens, candidate_len)
    if _hisa_fused_topk and keep <= candidate_len and top_blocks.shape[1] <= 256:
        # Fused mask + radix-topk + token-id map in a single launch.
        stage = _profile_start(q_values.device)
        topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            keep,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "fused_mask_topk_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
            fused_cuda=True,
        )
    else:
        stage = _profile_start(q_values.device)
        hisa_mask_logits_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "mask_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            fused_cuda=True,
        )

        stage = _profile_start(q_values.device)
        relevant_topk_indices = torch.topk(logits, k=keep, dim=-1, sorted=False).indices
        topk_indices = hisa_map_topk_indices_indexer_cache_nvfp4(
            relevant_topk_indices,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "final_topk_gather",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
        )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end(
        "map/store",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        topk_tokens=topk_tokens,
    )
    return topk_indices.to(torch.int32)


def nvfp4_hisa_indexer_paged_collective_key_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_starts: Optional[torch.Tensor] = None,
    prepared_block_ends: Optional[torch.Tensor] = None,
    prepared_candidate_context_lens: Optional[torch.Tensor] = None,
    prepared_candidate_schedule_metadata: Optional[object] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Collective candidate score-key selector path.

    This keeps the DeepGEMM batched FP4 candidate-scoring shape but changes the
    candidate epilogue to emit masked score-radix keys directly, then runs a
    3-pass key top-k/map tail. The candidate page lookup is derived inside the
    FP4 scorer from selected blocks and the original page table.
    """
    if block_size != 128:
        raise ValueError("Collective-key NVFP4 HISA path requires block_size=128.")

    q_values, q_scales = q_fp4
    rep_values, rep_scales = block_rep_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    import deep_gemm

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None

    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_starts is None:
        block_starts = token_to_batch_idx * max_blocks
    else:
        block_starts = prepared_block_starts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_ends is None:
        block_ends = block_starts + block_counts
    else:
        block_ends = prepared_block_ends.to(device=q_values.device, dtype=torch.int32)

    stage = _profile_start(q_values.device)
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        block_starts.to(torch.int32),
        block_ends.to(torch.int32),
        clean_logits=False,
        max_seqlen_k=max_blocks,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "key_blockscore_precomputed",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        backend="deepgemm_fp4_mqa",
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts_static_width(
            block_counts,
            block_size=block_size,
            compression_ratio=compression_ratio,
            max_blocks=max_blocks,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "key_block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "key_block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    context_lens = None
    schedule_metadata = None
    if not _hisa_row_split_candidate_keys:
        schedule_metadata = prepared_candidate_schedule_metadata
        if prepared_candidate_context_lens is None and schedule_metadata is None:
            context_lens, schedule_metadata = _hisa_uniform_candidate_schedule(
                q_values, candidate_len, deep_gemm
            )
        elif prepared_candidate_context_lens is None:
            context_lens = torch.full(
                (q_values.shape[0], 1),
                candidate_len,
                device=q_values.device,
                dtype=torch.int32,
            )
        else:
            context_lens = prepared_candidate_context_lens.to(
                device=q_values.device, dtype=torch.int32
            )
        if schedule_metadata is None:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                context_lens, 64, deep_gemm.get_num_sms()
            )

    stage = _profile_start(q_values.device)
    if _hisa_row_split_candidate_keys:
        candidate_keys = (
            hisa_deepgemm_candidate_keys_row_split_from_blocks_indexer_cache_nvfp4(
                q_fp4,
                index_k_with_scale_buffer,
                weights,
                page_table,
                token_to_batch_idx,
                top_blocks,
                prefix_lens.to(torch.int32),
                candidate_len,
                row_splits=_hisa_row_split_candidate_key_splits,
                page_size=64,
            )
        )
    else:
        candidate_keys = hisa_deepgemm_candidate_keys_from_blocks_indexer_cache_nvfp4(
            q_fp4,
            index_k_with_scale_buffer,
            weights,
            context_lens,
            page_table,
            token_to_batch_idx,
            schedule_metadata,
            top_blocks,
            prefix_lens.to(torch.int32),
            candidate_len,
            page_size=64,
        )
    _profile_end(
        (
            "candidate_score_keys_row_split"
            if _hisa_row_split_candidate_keys
            else "candidate_score_keys"
        ),
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        fused_score_mask_key=True,
        fused_candidate_pages=True,
        row_splits=(
            _hisa_row_split_candidate_key_splits
            if _hisa_row_split_candidate_keys
            else None
        ),
    )

    keep = min(topk_tokens, candidate_len)
    stage = _profile_start(q_values.device)
    topk_indices = hisa_candidate_keys_topk_map_indexer_cache_nvfp4(
        candidate_keys,
        top_blocks,
        keep,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "candidate_key_topk_map",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        topk_tokens=int(keep),
        fused_cuda=True,
    )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    return topk_indices.to(torch.int32)


def nvfp4_hisa_indexer_paged_collective_key(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    max_seq_len_hint: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Runtime wrapper for the collective FP4 score-key HISA path."""
    if block_size != 128:
        raise ValueError("Collective-key NVFP4 HISA path requires block_size=128.")

    q_values, _ = q_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    if page_table.dtype != torch.int32:
        return None

    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    if fallback_to_dense_if_short:
        if max_seq_len_hint is not None:
            if int(max_seq_len_hint) <= int(topk_tokens):
                return None
        elif _should_fallback_to_dense_short(True, prefix_lens, topk_tokens):
            return None

    max_blocks_hint = (
        _hisa_max_blocks_from_seq_len(max_seq_len_hint, block_size)
        if max_seq_len_hint is not None
        else _hisa_max_blocks_from_page_table(page_table, block_size)
    )
    stage = _profile_start(q_values.device)
    block_rep_fp4, max_blocks = hisa_precompute_block_reps_indexer_cache_nvfp4(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks=max_blocks_hint,
    )
    _profile_end(
        "key_mean_pool",
        stage,
        q_values.device,
        batches=int(page_table.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
    )

    block_counts = torch.div(
        prefix_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts_static_width(
            block_counts,
            block_size=block_size,
            compression_ratio=compression_ratio,
            max_blocks=max_blocks,
        )

    return nvfp4_hisa_indexer_paged_collective_key_precomputed(
        q_fp4,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        block_rep_fp4,
        max_blocks,
        block_size=block_size,
        block_topk=block_topk,
        compression_ratio=compression_ratio,
        topk_tokens=topk_tokens,
        fallback_to_dense_if_short=False,
        prepared_prefix_lens=prefix_lens,
        prepared_block_counts=block_counts,
        prepared_block_starts=token_to_batch_idx * max_blocks,
        prepared_block_ends=token_to_batch_idx * max_blocks + block_counts,
        prepared_block_topk_counts=block_topk_counts,
        prepared_effective_block_topk=effective_block_topk,
    )


def nvfp4_hisa_indexer_paged_megakernel_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Serial cuBLASDx HISA selector probe.

    This mirrors the training memory-conservation megakernel shape from the
    Megatron reference and is kept as an opt-in correctness probe. The target
    inference implementation should use the parallel/streaming selector scope.
    """
    if block_size != 128:
        raise ValueError("NVFP4 HISA megakernel requires block_size=128.")
    q_values, _ = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[-1] != 64:
        return None
    if page_table.dtype not in (torch.int32, torch.int64):
        return None
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None
    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = torch.full_like(block_counts, block_topk)
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size
    candidate_capacity = 1 << (max(topk_tokens, candidate_len, 1) - 1).bit_length()
    if candidate_capacity > 8192:
        return None

    stage = _profile_start(q_values.device)
    topk_indices = hisa_selector_megakernel_indexer_cache_nvfp4(
        q_fp4,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        block_rep_fp4,
        max_blocks,
        block_counts,
        block_topk_counts,
        effective_block_topk,
        topk_tokens,
    )
    _profile_end(
        "selector_megakernel",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        candidate_len=candidate_len,
        topk_tokens=topk_tokens,
        fused_cuda=True,
    )
    return topk_indices


def nvfp4_hisa_indexer_paged_parallel_megakernel_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Parallel cuBLASDx HISA selector probe.

    This follows the inference-oriented Megatron shape more closely than the
    serial probe: block selection is per row, candidate scoring is launched over
    row x selected-block tiles, and the existing fused CUDA tail does mask/top-k
    and candidate-position mapping.
    """
    if block_size != 128:
        raise ValueError("NVFP4 HISA parallel megakernel requires block_size=128.")
    q_values, _ = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[-1] != 64:
        return None
    if page_table.dtype not in (torch.int32, torch.int64):
        return None
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None
    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = torch.full_like(block_counts, block_topk)
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size

    stage = _profile_start(q_values.device)
    top_blocks = hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
        q_fp4,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        block_rep_fp4,
        max_blocks,
        block_counts,
        block_topk_counts,
        effective_block_topk,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "selector_parallel_select_blocks",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        block_topk=effective_block_topk,
        fused_cuda=True,
    )

    keep = min(topk_tokens, candidate_len)
    if candidate_len == topk_tokens:
        stage = _profile_start(q_values.device)
        topk_indices = hisa_map_candidate_indices_indexer_cache_nvfp4(
            top_blocks,
            prefix_lens.to(torch.int32),
            topk_tokens,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "selector_parallel_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=topk_tokens,
            fused_cuda=True,
        )
        return topk_indices

    stage = _profile_start(q_values.device)
    logits = hisa_selector_parallel_score_candidates_indexer_cache_nvfp4(
        q_fp4,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        top_blocks,
    )
    _profile_end(
        "selector_parallel_score_candidates",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
        logits,
        top_blocks,
        prefix_lens.to(torch.int32),
        keep,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "selector_parallel_fused_mask_topk_map",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        topk_tokens=int(keep),
        fused_cuda=True,
    )
    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    return topk_indices


def nvfp4_hisa_indexer_paged_cluster_fused_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Cluster-fused cuBLASDx HISA selector probe.

    This keeps block scoring, block selection, candidate scoring, top-k, and
    candidate-position mapping inside one CUDA launch. Cluster ranks split the
    candidate GEMM tiles for each row, then rank 0 merges the row-local top-k.
    """
    if block_size != 128:
        raise ValueError("NVFP4 HISA cluster-fused selector requires block_size=128.")
    q_values, _ = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[-1] != 64:
        return None
    if page_table.dtype not in (torch.int32, torch.int64):
        return None
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None
    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = torch.full_like(block_counts, block_topk)
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size

    stage = _profile_start(q_values.device)
    topk_indices = hisa_selector_cluster_fused_indexer_cache_nvfp4(
        q_fp4,
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        block_rep_fp4,
        max_blocks,
        block_counts,
        block_topk_counts,
        effective_block_topk,
        topk_tokens,
    )
    _profile_end(
        "selector_cluster_fused",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        candidate_len=candidate_len,
        topk_tokens=topk_tokens,
        cluster_size=4,
        fused_cuda=True,
    )
    return topk_indices


def nvfp4_hisa_indexer_paged_parallel_collective_precomputed(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_rep_fp4: tuple[torch.Tensor, torch.Tensor],
    max_blocks: int,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
    prepared_prefix_lens: Optional[torch.Tensor] = None,
    prepared_block_counts: Optional[torch.Tensor] = None,
    prepared_block_topk_counts: Optional[torch.Tensor] = None,
    prepared_effective_block_topk: Optional[int] = None,
    prepared_candidate_context_lens: Optional[torch.Tensor] = None,
    prepared_candidate_schedule_metadata: Optional[tuple] = None,
) -> Optional[torch.Tensor]:
    """Parallel selector probe with collective Blackwell candidate scoring.

    This keeps the experimental cuBLASDx block-selection probe but replaces the
    bad row x selected-block candidate-scoring fanout with DeepGEMM's batched
    paged FP4 MQA logits kernel. It is a probe for the right candidate-scoring
    shape, not a production selector path.
    """
    if block_size != 128:
        raise ValueError("NVFP4 HISA parallel collective requires block_size=128.")
    q_values, q_scales = q_fp4
    if q_values.dim() != 3 or q_values.shape[1] != 64 or q_values.shape[-1] != 64:
        return None
    if page_table.dtype not in (torch.int32, torch.int64):
        return None
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    if prepared_prefix_lens is None:
        prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    else:
        prefix_lens = prepared_prefix_lens.to(device=q_values.device, dtype=torch.int32)
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None
    if prepared_block_counts is None:
        block_counts = torch.div(
            prefix_lens.to(torch.int32) + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        block_counts = prepared_block_counts.to(device=q_values.device, dtype=torch.int32)
    if prepared_block_topk_counts is not None and prepared_effective_block_topk is not None:
        block_topk_counts = prepared_block_topk_counts.to(
            device=q_values.device, dtype=torch.int32
        )
        effective_block_topk = int(prepared_effective_block_topk)
    elif compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = torch.full_like(block_counts, block_topk)
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size

    stage = _profile_start(q_values.device)
    top_blocks = hisa_selector_parallel_select_blocks_indexer_cache_nvfp4(
        q_fp4,
        seq_lens_flat,
        weights,
        token_to_batch_idx,
        block_rep_fp4,
        max_blocks,
        block_counts,
        block_topk_counts,
        effective_block_topk,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "selector_collective_select_blocks",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        block_topk=effective_block_topk,
        fused_cuda=True,
    )

    keep = min(topk_tokens, candidate_len)
    if candidate_len == topk_tokens:
        stage = _profile_start(q_values.device)
        topk_indices = hisa_map_candidate_indices_indexer_cache_nvfp4(
            top_blocks,
            prefix_lens.to(torch.int32),
            topk_tokens,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "selector_collective_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=topk_tokens,
            fused_cuda=True,
        )
        return topk_indices

    stage = _profile_start(q_values.device)
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )
    _profile_end(
        "selector_collective_candidate_pages",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        candidate_pages=int(candidate_page_table.shape[1]),
        fused_cuda=True,
    )

    import deep_gemm

    schedule_metadata = prepared_candidate_schedule_metadata
    if prepared_candidate_context_lens is None and schedule_metadata is None:
        context_lens, schedule_metadata = _hisa_uniform_candidate_schedule(
            q_values, candidate_len, deep_gemm
        )
    elif prepared_candidate_context_lens is None:
        context_lens = torch.full(
            (q_values.shape[0], 1),
            candidate_len,
            device=q_values.device,
            dtype=torch.int32,
        )
    else:
        context_lens = prepared_candidate_context_lens.to(
            device=q_values.device, dtype=torch.int32
        )
    if schedule_metadata is None:
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, 64, deep_gemm.get_num_sms()
        )
    stage = _profile_start(q_values.device)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        index_k_with_scale_buffer.view(index_k_with_scale_buffer.shape[0], 64, 1, 68),
        weights,
        context_lens,
        candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "selector_collective_candidate_logits",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        backend="deepgemm_fp4_paged_mqa",
    )

    stage = _profile_start(q_values.device)
    topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
        logits,
        top_blocks,
        prefix_lens.to(torch.int32),
        keep,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "selector_collective_fused_mask_topk_map",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
        topk_tokens=int(keep),
        fused_cuda=True,
    )
    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    return topk_indices


def nvfp4_hisa_indexer_paged_deepgemm(
    q_fp4: tuple[torch.Tensor, torch.Tensor],
    index_k_with_scale_buffer: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    weights: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    *,
    block_size: int = 128,
    block_topk: int = 64,
    compression_ratio: Optional[float] = 4.0,
    topk_tokens: int = 2048,
    fallback_to_dense_if_short: bool = True,
) -> Optional[torch.Tensor]:
    if block_size != 128:
        raise ValueError("DeepGEMM-backed NVFP4 HISA path requires block_size=128.")

    q_values, q_scales = q_fp4
    if not _deepgemm_fp4_mqa_supports(q_values):
        return None
    import deep_gemm

    if weights.dim() == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    token_to_batch_idx = token_to_batch_idx.reshape(-1).to(
        device=q_values.device, dtype=torch.int32
    )
    seq_lens_flat = seq_lens.reshape(-1).to(device=q_values.device, dtype=torch.int32)
    prefix_lens = seq_lens_flat.index_select(0, token_to_batch_idx.long())
    if _should_fallback_to_dense_short(
        fallback_to_dense_if_short, prefix_lens, topk_tokens
    ):
        return None

    max_blocks = _hisa_max_blocks(seq_lens_flat, page_table, block_size)
    stage = _profile_start(q_values.device)
    reps = _hisa_mean_pool_call(
        index_k_with_scale_buffer,
        page_table,
        seq_lens_flat,
        max_blocks,
    )
    _profile_end(
        "mean_pool",
        stage,
        q_values.device,
        batches=int(page_table.shape[0]),
        max_blocks=max_blocks,
        fused_cuda=True,
    )

    block_counts = torch.div(
        prefix_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).to(torch.int32)
    stage = _profile_start(q_values.device)
    rep_values, rep_scales = quantize_indexer_q_nvfp4(
        reps.reshape(-1, reps.shape[-1]).to(torch.bfloat16),
        indices_dtype=page_table.dtype,
    )
    _profile_end(
        "block_rep_quant",
        stage,
        q_values.device,
        rows=int(rep_values.shape[0]),
    )

    block_starts = token_to_batch_idx * max_blocks
    block_ends = block_starts + block_counts
    stage = _profile_start(q_values.device)
    block_scores = deep_gemm.fp8_fp4_mqa_logits(
        (q_values.view(torch.int8), q_scales),
        (rep_values.view(torch.int8), rep_scales),
        weights,
        block_starts.to(torch.int32),
        block_ends.to(torch.int32),
        clean_logits=False,
        max_seqlen_k=max_blocks,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "block_score",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        max_blocks=max_blocks,
        backend="deepgemm_fp4_mqa",
        packed_q=True,
    )

    stage = _profile_start(q_values.device)
    if compression_ratio is None or compression_ratio <= 0:
        block_topk_counts = None
        effective_block_topk = block_topk
    else:
        block_topk_counts, effective_block_topk = _hisa_block_topk_counts(
            block_counts,
            block_size=block_size,
            topk_tokens=topk_tokens,
            compression_ratio=compression_ratio,
        )
    candidate_len = effective_block_topk * block_size
    if candidate_len == topk_tokens:
        topk_indices = hisa_block_topk_map_all_indexer_cache_nvfp4(
            block_scores,
            block_counts,
            block_topk=effective_block_topk,
            block_topk_counts=block_topk_counts,
            prefix_lens=prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "block_topk_map_all",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            block_topk=effective_block_topk,
            compression_ratio=compression_ratio,
            fused_cuda=True,
        )
        return topk_indices

    top_blocks = hisa_block_topk_indexer_cache_nvfp4(
        block_scores,
        block_counts,
        block_topk=effective_block_topk,
        block_topk_counts=block_topk_counts,
        page_table_dtype=page_table.dtype,
    )
    _profile_end(
        "block_topk",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        compression_ratio=compression_ratio,
        fused_cuda=True,
    )

    stage = _profile_start(q_values.device)
    candidate_page_table = hisa_candidate_pages_indexer_cache_nvfp4(
        top_blocks, page_table, token_to_batch_idx
    )
    _profile_end(
        "candidate_table",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        block_topk=effective_block_topk,
        candidate_pages=int(candidate_page_table.shape[1]),
        fused_cuda=True,
    )

    context_lens = torch.full(
        (q_values.shape[0], 1),
        candidate_len,
        device=q_values.device,
        dtype=torch.int32,
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, 64, deep_gemm.get_num_sms()
    )
    stage = _profile_start(q_values.device)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
        index_k_with_scale_buffer.view(index_k_with_scale_buffer.shape[0], 64, 1, 68),
        weights,
        context_lens,
        candidate_page_table,
        schedule_metadata,
        candidate_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    _profile_end(
        "candidate_deepgemm",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        candidate_len=candidate_len,
    )

    keep = min(topk_tokens, candidate_len)
    if _hisa_fused_topk and keep <= candidate_len and top_blocks.shape[1] <= 256:
        stage = _profile_start(q_values.device)
        topk_indices = hisa_fused_mask_topk_map_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            keep,
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "fused_mask_topk_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
            fused_cuda=True,
        )
    else:
        stage = _profile_start(q_values.device)
        hisa_mask_logits_indexer_cache_nvfp4(
            logits,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "candidate_mask_map",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            fused_cuda=True,
        )

        stage = _profile_start(q_values.device)
        relevant_topk_indices = torch.topk(logits, k=keep, dim=-1, sorted=False).indices
        topk_indices = hisa_map_topk_indices_indexer_cache_nvfp4(
            relevant_topk_indices,
            top_blocks,
            prefix_lens.to(torch.int32),
            page_table_dtype=page_table.dtype,
        )
        _profile_end(
            "candidate_topk",
            stage,
            q_values.device,
            rows=int(q_values.shape[0]),
            candidate_len=candidate_len,
            topk_tokens=int(keep),
        )

    if keep < topk_tokens:
        padding = torch.full(
            (topk_indices.shape[0], topk_tokens - keep),
            -1,
            device=topk_indices.device,
            dtype=torch.int32,
        )
        topk_indices = torch.cat((topk_indices, padding), dim=1)
    stage = _profile_start(q_values.device)
    _profile_end(
        "map/store",
        stage,
        q_values.device,
        rows=int(q_values.shape[0]),
        topk_tokens=topk_tokens,
    )
    return topk_indices.to(torch.int32)

# DSA is the upstream canonical name; keep NSA aliases for deployment/config compatibility.
can_use_dsa_nvfp4_indexer = can_use_nsa_nvfp4_indexer
