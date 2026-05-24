"""Autonomous iteration loop for HIGGS DSL kernel.

Tries a list of variant patches, runs the bench harness, captures
correctness + timing metrics, picks the winner. Runs in-place on B200.

Usage:
  python3.11 autoiter_higgs_dsl.py

Reports written to /tmp/autoiter_higgs_dsl_results.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


KERNEL_PATH = Path("/home/spencer/optimization-playground/python/sglang/jit_kernel/higgs_dense_2bit_mla_decode_dsl.py")
BENCH_PATH = Path("/home/spencer/optimization-playground/python/sglang/jit_kernel/bench_higgs_dsl.py")
RESULTS_PATH = Path("/tmp/autoiter_higgs_dsl_results.jsonl")
BACKUP_PATH = Path("/tmp/higgs_dsl_iter_backup.py")


@dataclass
class Variant:
    name: str
    description: str
    patch_fn_name: str          # name of patch function in this module
    revert_fn_name: str = "revert_to_baseline"


@dataclass
class Result:
    variant: str
    description: str
    compile_ok: bool = False
    run_ok: bool = False
    elapsed_s: float = 0.0
    max_abs: float = float("inf")
    mean_abs: float = float("inf")
    out_min: float = float("nan")
    out_max: float = float("nan")
    out_std: float = float("nan")
    racecheck_errors: int = -1   # -1 = not run
    czs_proved: int = -1          # -1 = not run; otherwise count of Proved verdicts
    czs_disproved: int = -1
    czs_unknown: int = -1
    error: str = ""


# === Variant patch functions ===

def revert_to_baseline(src: str) -> str:
    """Revert to baseline iter-6 state (no swizzle change)."""
    # Baseline uses MMA's own swizzle inner via q/k/v/p_smem_layout.inner
    return _set_swizzle_bits(src, "Q", "MMA"), "baseline (use MMA's own swizzle)"


def _set_swizzle_bits_block(src: str, bits_Q: str, bits_K: str, bits_V: str, bits_P: str) -> str:
    """Replace the swizzle-bits computation block."""
    old = re.compile(
        r"(        sw_Q = .+?\n)"
        r"(        sw_K = .+?\n)"
        r"(        sw_V = .+?\n)"
        r"(        sw_P = .+?\n)",
        re.DOTALL,
    )
    new_block = (
        f"        sw_Q = {bits_Q}\n"
        f"        sw_K = {bits_K}\n"
        f"        sw_V = {bits_V}\n"
        f"        sw_P = {bits_P}\n"
    )
    return old.sub(new_block, src, count=1)


def _set_swizzle_bits(src: str, mode: str = "MMA") -> str:
    """Set the swizzle for all four buffers."""
    if mode == "MMA":
        # Use MMA's own swizzle inner (iter-6 baseline)
        return _set_swizzle_bits_block(
            src, "q_smem_layout.inner", "k_smem_layout.inner",
            "v_smem_layout.inner", "p_smem_layout.inner",
        )
    elif mode == "Swizzle_3_3_3":
        return _set_swizzle_bits_block(
            src, "cute.make_swizzle(3, 3, 3)", "cute.make_swizzle(3, 3, 3)",
            "cute.make_swizzle(3, 3, 3)", "cute.make_swizzle(3, 3, 3)",
        )
    elif mode == "Swizzle_2_3_3":
        return _set_swizzle_bits_block(
            src, "cute.make_swizzle(2, 3, 3)", "cute.make_swizzle(2, 3, 3)",
            "cute.make_swizzle(2, 3, 3)", "cute.make_swizzle(2, 3, 3)",
        )
    elif mode == "Swizzle_0_0_0":
        return _set_swizzle_bits_block(
            src, "cute.make_swizzle(0, 0, 0)", "cute.make_swizzle(0, 0, 0)",
            "cute.make_swizzle(0, 0, 0)", "cute.make_swizzle(0, 0, 0)",
        )
    elif mode == "Swizzle_1_3_3":
        return _set_swizzle_bits_block(
            src, "cute.make_swizzle(1, 3, 3)", "cute.make_swizzle(1, 3, 3)",
            "cute.make_swizzle(1, 3, 3)", "cute.make_swizzle(1, 3, 3)",
        )
    raise ValueError(f"unknown mode: {mode}")


def patch_swizzle_mma(src: str) -> tuple[str, str]:
    return _set_swizzle_bits(src, "MMA"), "baseline (use MMA's own swizzle inner)"


def patch_swizzle_3_3_3(src: str) -> tuple[str, str]:
    return _set_swizzle_bits(src, "Swizzle_3_3_3"), "Swizzle<3,3,3> on all buffers"


def patch_swizzle_2_3_3(src: str) -> tuple[str, str]:
    return _set_swizzle_bits(src, "Swizzle_2_3_3"), "Swizzle<2,3,3> on all buffers"


def patch_swizzle_1_3_3(src: str) -> tuple[str, str]:
    return _set_swizzle_bits(src, "Swizzle_1_3_3"), "Swizzle<1,3,3> on all buffers"


def patch_swizzle_0_0_0(src: str) -> tuple[str, str]:
    return _set_swizzle_bits(src, "Swizzle_0_0_0"), "no swizzle (bank conflict but should be MMA-correct)"


def patch_constant_kv(src: str) -> tuple[str, str]:
    """Skip HIGGS dequant entirely; write constant K=V=1.0 via the existing
    per-element scatter path. If DSL output matches uniform attention against
    K=V=1.0, the data-flow pipeline works (and the bug is in dequant decoding).
    If output is still ~0, the bug is structural (writes never reach MMA reads
    OR the softmax normalize is wrong).
    """
    # Replace the dequant body with constant writes
    old = re.compile(
        r"            sK\[n, d0\] = v0\n"
        r"            sK\[n, d1\] = v1\n"
        r"            sK\[n, d2\] = v2\n"
        r"            sK\[n, d3\] = v3\n"
        r"            sV\[d0, n\] = v0\n"
        r"            sV\[d1, n\] = v1\n"
        r"            sV\[d2, n\] = v2\n"
        r"            sV\[d3, n\] = v3\n"
    )
    one = "            one_bf = cutlass.BFloat16(1.0)\n"
    new_block = (
        one +
        "            sK[n, d0] = one_bf\n"
        "            sK[n, d1] = one_bf\n"
        "            sK[n, d2] = one_bf\n"
        "            sK[n, d3] = one_bf\n"
        "            sV[d0, n] = one_bf\n"
        "            sV[d1, n] = one_bf\n"
        "            sV[d2, n] = one_bf\n"
        "            sV[d3, n] = one_bf\n"
    )
    new_src = old.sub(new_block, src, count=1)
    return new_src, "Skip HIGGS dequant; write K=V=1.0 via scatter (data-flow test)"


def patch_log_softmax_l(src: str) -> tuple[str, str]:
    """Make the epilogue NOT divide by softmax_l (sanity check on PV MMA path).
    If output magnitude jumps to reasonable values, softmax_l is wrong.
    """
    old = "                acc_val = acc_val / softmax_l_t[m_local]"
    new = "                # acc_val = acc_val / softmax_l_t[m_local]  # ITER DEBUG: skip divide"
    new_src = src.replace(old, new, 1)
    return new_src, "Skip softmax_l divide in epilogue (raw acc to GMEM)"


def patch_softmax_l_constant(src: str) -> tuple[str, str]:
    """Force softmax_l = TOPK in epilogue (correct for uniform attention).
    If output then matches expected, the bug is JUST in softmax_l aggregation.
    """
    old = "                acc_val = acc_val / softmax_l_t[m_local]"
    new = "                acc_val = acc_val / Float32(32.0)  # ITER DEBUG: softmax_l = TOPK"
    new_src = src.replace(old, new, 1)
    return new_src, "Force softmax_l=32 in epilogue divide"


def patch_constant_kv_and_l(src: str) -> tuple[str, str]:
    """Combine: K=V=1.0 dequant skip + softmax_l=32 divide. If output is ~1.0,
    the data flow + epilogue work and the only bug is the softmax_l aggregation race.
    """
    src1, _ = patch_constant_kv(src)
    src2, _ = patch_softmax_l_constant(src1)
    return src2, "K=V=1.0 + softmax_l=32 (isolates softmax_l race as the bug)"


def patch_dequant_to_autovec(src: str) -> tuple[str, str]:
    """Replace per-slot dequant loop with cute.autovec_copy from a
    pre-built GMEM K/V dense tile (passed in via the kernel call). This is
    the simplest 'pre-dequant' variant: the Python wrapper computes
    BF16 K_full and V_full per row, the kernel does autovec_copy into
    swizzled SMEM (canonical cp.async pattern, address-correct by construction).
    """
    # Replace the body of _dequant_tile with autovec_copy calls.
    # NOTE: this requires adding k_full and v_full args to the kernel,
    # which is a substantial rewrite. For an autoloop test, we'll
    # patch the wrapper to PRE-FILL sK and sV directly via a Python-side
    # tensor copy, bypassing the in-kernel dequant entirely.
    #
    # Simpler: just zero-out sK and sV in the dequant (skip per-slot writes),
    # then have the Python wrapper write directly to the SMEM buffer.
    # But that's not feasible without re-architecting kernel storage.
    #
    # ALTERNATIVE: replace per-element writes with cute.fill which
    # writes a tensor-wide value efficiently. This won't give correct
    # output but will tell us if the SMEM buffer is reachable for fill.
    old = re.compile(
        r"            sK\[n, d0\] = v0\n.*?"
        r"            sK\[n, rope_smem_col\] = rope_val\n",
        re.DOTALL,
    )
    new_body = (
        "            # ITER 8: autovec replacement (skip per-element; test fill)\n"
        "            pass\n"
    )
    # Find the broader dequant inner-loop and replace
    pattern = re.compile(
        r"(            v0 = \(scale \* c0\)\.to\(cutlass\.BFloat16\)\n"
        r"            v1 = \(scale \* c1\)\.to\(cutlass\.BFloat16\)\n"
        r"            v2 = \(scale \* c2\)\.to\(cutlass\.BFloat16\)\n"
        r"            v3 = \(scale \* c3\)\.to\(cutlass\.BFloat16\)\n)"
        r"(            sK\[n, d0\] = v0\n"
        r"            sK\[n, d1\] = v1\n"
        r"            sK\[n, d2\] = v2\n"
        r"            sK\[n, d3\] = v3\n"
        r"            sV\[d0, n\] = v0\n"
        r"            sV\[d1, n\] = v1\n"
        r"            sV\[d2, n\] = v2\n"
        r"            sV\[d3, n\] = v3\n)",
    )
    # No-op replacement (just skip the writes)
    new_src = pattern.sub(r"\1            # skipped writes\n", src, count=1)
    return new_src, "Skip K/V SMEM writes (sK, sV uninitialized; tests if PV MMA reads constant init values)"


def patch_iter_atom_k_blocks(src: str) -> tuple[str, str]:
    """CZS FINDING fix: SM100 F16/BF16 atom has K=16 (not MMA_K=64).
    Each MMA-K stage of 64 K-elements requires 4 atom-K iterations.
    Currently the inner loop only does 1 cute.gemm per stage = covers
    K=16 of 64. Output is ~25% of true.

    Fix: iterate k_block in 0..K_blocks_per_stage and call cute.gemm
    for each. Same for PV MMA (K=32, atom K=16 → 2 blocks per stage).
    """
    old = (
        "            for k_stage in cutlass.range_constexpr(ITERATIONS_QK):\n"
        "                cute.gemm(\n"
        "                    qk_tiled_mma,\n"
        "                    tScore,\n"
        "                    tCrQ[None, None, 0, k_stage],\n"
        "                    tCrK[None, None, 0, k_stage],\n"
        "                    tScore,\n"
        "                )\n"
        "                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)"
    )
    new = (
        "            # CZS-driven fix: atom K=16, MMA_K=64 → 4 atom blocks per stage.\n"
        "            ATOM_K_BLOCKS_QK = MMA_K // 16  # 4 atom-K iters per K-tile\n"
        "            for k_stage in cutlass.range_constexpr(ITERATIONS_QK):\n"
        "                for k_block in cutlass.range_constexpr(ATOM_K_BLOCKS_QK):\n"
        "                    cute.gemm(\n"
        "                        qk_tiled_mma,\n"
        "                        tScore,\n"
        "                        tCrQ[None, None, k_block, k_stage],\n"
        "                        tCrK[None, None, k_block, k_stage],\n"
        "                        tScore,\n"
        "                    )\n"
        "                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)"
    )
    new_src = src.replace(old, new, 1)
    # Also fix PV MMA: K=32 tile, atom K=16 → 2 atom blocks per stage
    old_pv = (
        "            for n_chunk in cutlass.range_constexpr(PV_N_CHUNKS):\n"
        "                acc_target = tAcc_lo if n_chunk == 0 else tAcc_hi\n"
        "                for k_stage in cutlass.range_constexpr(self.iterations_pv_k):\n"
        "                    v_stage = n_chunk * self.iterations_pv_k + k_stage\n"
        "                    cute.gemm(\n"
        "                        pv_tiled_mma,\n"
        "                        acc_target,\n"
        "                        tCrP[None, None, 0, k_stage],\n"
        "                        tCrV[None, None, 0, v_stage],\n"
        "                        acc_target,\n"
        "                    )\n"
        "                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)"
    )
    new_pv = (
        "            # CZS-driven fix: PV atom K=16, P stage K=16 (block_n/iter_pv_k = 32/2=16)\n"
        "            # → 1 atom block per stage, no extra inner loop needed for PV.\n"
        "            for n_chunk in cutlass.range_constexpr(PV_N_CHUNKS):\n"
        "                acc_target = tAcc_lo if n_chunk == 0 else tAcc_hi\n"
        "                for k_stage in cutlass.range_constexpr(self.iterations_pv_k):\n"
        "                    v_stage = n_chunk * self.iterations_pv_k + k_stage\n"
        "                    cute.gemm(\n"
        "                        pv_tiled_mma,\n"
        "                        acc_target,\n"
        "                        tCrP[None, None, 0, k_stage],\n"
        "                        tCrV[None, None, 0, v_stage],\n"
        "                        acc_target,\n"
        "                    )\n"
        "                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)"
    )
    new_src = new_src.replace(old_pv, new_pv, 1)
    return new_src, "CZS fix: iter 4 atom-K blocks per QK stage (atom K=16 != MMA_K=64)"


def patch_predequant_via_python(src: str) -> tuple[str, str]:
    """Option A fix: have the Python wrapper pre-decode HIGGS → BF16 K/V,
    pass those to the kernel via the EXISTING `mCR` arg slot (which is
    already BF16 GMEM with shape (num_slots, ROPE_DIM=64)). We repurpose
    it: pre-pack K (FULL_DIM=576) + V (LATENT_DIM=512) into one BF16 tensor
    of shape (num_slots, FULL_DIM + LATENT_DIM = 1088).

    Inside the kernel, replace _dequant_tile call with a per-thread
    cooperative gather of the relevant slot's K and V from this packed
    GMEM tensor, written directly to sK_mma / sV_mma. Use cute.autovec_copy
    for race-free vectorized SMEM stores.

    This is a single-patch variant test; full refactor lands as iter 10.
    Patch strategy: replace the dequant call with cute.fill (race-free
    placeholder) + add a comment marker that the Python wrapper should
    populate mCR with packed K+V. Just confirms autoloop can handle
    Python-side data prep changes.
    """
    # For variant test purposes, use cute.fill of sK_mma/sV_mma to
    # eliminate races. Real Option A needs Python wrapper changes too.
    old = (
        "            # (A) Dequant tile via per-element scatter (iter-4 path).\n"
        "            # ITER 6: replace with R2S TiledCopy once we can compile\n"
        "            # _dequant_tile_r2s with the full (block_n, FULL_DIM) loop\n"
        "            # without 30+ minute compile times.\n"
        "            self._dequant_tile(\n"
        "                storage, sK, sV, mCK, mCS, mCR, mPT,\n"
        "                row, tile_begin, tile_count, tid,\n"
        "            )\n"
        "            cute.arch.barrier()"
    )
    new = (
        "            # ITER 10 Option A: replace per-element dequant with\n"
        "            # cute.fill (race-free) — KV values come from Python-side\n"
        "            # dequant pre-pass. Output magnitude correct; values placeholder.\n"
        "            sK_mma.fill(cutlass.BFloat16(0.1))\n"
        "            sV_mma.fill(cutlass.BFloat16(0.1))\n"
        "            cute.arch.barrier()"
    )
    new_src = src.replace(old, new, 1)
    return new_src, "Option A test: cute.fill with small constant 0.1 (race-free, magnitude check)"


def patch_cute_fill_kv(src: str) -> tuple[str, str]:
    """Replace the per-slot/per-element sK,sV scatter loop with cute.fill
    (single op on the whole MMA-shaped SMEM tensor). If this eliminates the
    32 races and produces uniform output, it CONFIRMS the per-element writes
    are the race source AND that cute.fill / cute.copy patterns are race-free.

    Output won't match HIGGS dequant (constant K=V=1.0), but races should
    drop to 0.
    """
    # ITER 11+ structure: replace self._load_kv_from_dense(...) call.
    # Match both old (_dequant_tile) and new (_load_kv_from_dense) patterns.
    patterns = [
        # ITER 11+ path
        (
            "            self._load_kv_from_dense(\n"
            "                mK, mV, mPT, sK, sV,\n"
            "                row, tile_begin, tile_count, tid,\n"
            "            )\n"
            "            cute.arch.barrier()"
        ),
        # ITER <11 path
        (
            "            self._dequant_tile(\n"
            "                storage, sK, sV, mCK, mCS, mCR, mPT,\n"
            "                row, tile_begin, tile_count, tid,\n"
            "            )\n"
            "            cute.arch.barrier()"
        ),
    ]
    new = (
        "            # IKP race fix probe: cute.fill on sK_mma + sV_mma (race-free per autoloop iter 9).\n"
        "            sK_mma.fill(cutlass.BFloat16(1.0))\n"
        "            sV_mma.fill(cutlass.BFloat16(1.0))\n"
        "            cute.arch.barrier()"
    )
    new_src = src
    for old in patterns:
        if old in new_src:
            new_src = new_src.replace(old, new, 1)
            break
    return new_src, "Replace dequant with cute.fill on sK_mma + sV_mma (race-free; constant K=V=1)"


def patch_softmax_state_skip_writes(src: str) -> tuple[str, str]:
    """Skip softmax state writes entirely. Don't write softmax_m/l/alpha
    in _score_softmax_to_p. With softmax_l init=1.0 (not 0.0!), the
    epilogue divide by softmax_l is identity → output = raw acc.

    Mathematically wrong (no normalization), but eliminates the 16-hazard
    softmax write races. Result tells us if those races are the dominant
    correctness issue.
    """
    old = (
        "        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):\n"
        "            m_idx = tTR_tS[i][0]\n"
        "            softmax_m_t[m_idx] = row_max_new\n"
        "            softmax_l_t[m_idx] = row_sum_new\n"
        "            softmax_alpha_t[m_idx] = correction_factor"
    )
    new = (
        "        # IKP race fix: skip softmax state writes entirely.\n"
        "        # Eliminates 16x16 = 256 thread-races on softmax_l/m/alpha.\n"
        "        pass"
    )
    new_src = src.replace(old, new, 1)
    # Also fix softmax_l init: set to 1.0 instead of 0.0 so epilogue divide is identity
    old2 = "        softmax_l_t[init_idx] = Float32(0.0)"
    new2 = "        softmax_l_t[init_idx] = Float32(32.0)  # uniform-attention divisor"
    new_src = new_src.replace(old2, new2, 1)
    return new_src, "Skip all softmax state writes; init softmax_l=32 for divide identity"


def patch_softmax_state_warp0_only(src: str) -> tuple[str, str]:
    """FIX FOR IKP-IDENTIFIED RACE: only warp 0 lanes write softmax state.

    All other warps compute but don't write — their rows will have stale
    init values from before the slot loop. This is a stopgap that makes
    warp-0's rows numerically correct, identifying whether the race fix
    direction is right. Production fix needs cross-warp reduction via
    softmax_smem_exchange (tokenspeed pattern).
    """
    old = (
        "        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):\n"
        "            m_idx = tTR_tS[i][0]\n"
        "            softmax_m_t[m_idx] = row_max_new\n"
        "            softmax_l_t[m_idx] = row_sum_new\n"
        "            softmax_alpha_t[m_idx] = correction_factor"
    )
    new = (
        "        # IKP race fix: gate write via cutlass.select_ on warp_id == 0.\n"
        "        # Reads existing, writes new only for warp 0 lanes; others\n"
        "        # write the existing value (identical-value race, benign).\n"
        "        warp_id_local = tid // Int32(self.threads_per_warp)\n"
        "        warp_is_zero = warp_id_local == Int32(0)\n"
        "        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):\n"
        "            m_idx = tTR_tS[i][0]\n"
        "            cur_m = softmax_m_t[m_idx]\n"
        "            cur_l = softmax_l_t[m_idx]\n"
        "            cur_a = softmax_alpha_t[m_idx]\n"
        "            softmax_m_t[m_idx] = cutlass.select_(warp_is_zero, row_max_new, cur_m)\n"
        "            softmax_l_t[m_idx] = cutlass.select_(warp_is_zero, row_sum_new, cur_l)\n"
        "            softmax_alpha_t[m_idx] = cutlass.select_(warp_is_zero, correction_factor, cur_a)"
    )
    new_src = src.replace(old, new, 1)
    return new_src, "Race fix: only warp 0 writes softmax state; others write existing (benign-race)"


def patch_dual_writes_to_mma_view(src: str) -> tuple[str, str]:
    """Per-element write to sK and sV uses sK_write/sV_write (composed swizzle).
    Try writing through sK_mma and sV_mma directly (the MMA-staged tensor views).
    sK_mma has shape ((atom_n, atom_k), n_blocks, k_blocks, stages) which is
    NOT (n, d) indexable directly. This patch will fail at AST stage; but if it
    does work, output should be correct.
    """
    old = (
        "        sQ, sK, sV, sP = sQ_write, sK_write, sV_write, sP_write"
    )
    new = (
        "        # ITER 8: try writing directly through MMA-view tensors\n"
        "        # (likely fails — sK_mma has multi-mode shape; here for diagnostic)\n"
        "        sQ, sK, sV, sP = sQ_mma, sK_mma, sV_mma, sP_mma"
    )
    new_src = src.replace(old, new, 1)
    return new_src, "Write directly through sK_mma/sV_mma (test multi-mode indexing)"


VARIANTS = [
    Variant("swizzle_mma", "MMA's own swizzle inner (iter-6 baseline)", "patch_swizzle_mma"),
    Variant("swizzle_3_3_3", "Swizzle<3,3,3>", "patch_swizzle_3_3_3"),
    Variant("swizzle_2_3_3", "Swizzle<2,3,3>", "patch_swizzle_2_3_3"),
    Variant("swizzle_1_3_3", "Swizzle<1,3,3>", "patch_swizzle_1_3_3"),
    Variant("swizzle_0_0_0", "No swizzle (Swizzle<0,0,0>)", "patch_swizzle_0_0_0"),
    Variant("constant_kv", "K=V=1.0 via scatter (skip HIGGS dequant)", "patch_constant_kv"),
    Variant("skip_softmax_l_divide", "Skip softmax_l divide (raw acc to GMEM)", "patch_log_softmax_l"),
    Variant("softmax_l_constant", "Force softmax_l=32 in epilogue divide", "patch_softmax_l_constant"),
    Variant("constant_kv_and_l", "K=V=1.0 + softmax_l=32 (isolates softmax_l race)", "patch_constant_kv_and_l"),
    Variant("skip_kv_writes", "Skip K/V SMEM writes entirely (test for static-init values)", "patch_dequant_to_autovec"),
    Variant("write_via_mma_view", "Try per-element write through sX_mma multi-mode tensor", "patch_dual_writes_to_mma_view"),
    Variant("softmax_warp0_only", "FIX iter-8 race: only warp 0 writes softmax state", "patch_softmax_state_warp0_only"),
    Variant("softmax_skip_writes", "FIX iter-8 race: skip softmax writes; init l=32 for identity divide", "patch_softmax_state_skip_writes"),
    Variant("cute_fill_kv", "Replace dequant with cute.fill(sK_mma)+cute.fill(sV_mma) (race-free probe)", "patch_cute_fill_kv"),
    Variant("iter_atom_k_blocks", "CZS fix: iterate 4 atom-K blocks per QK stage (atom K=16)", "patch_iter_atom_k_blocks"),
    Variant("fill_plus_atom_k", "Combine cute.fill + atom-K-block iter (CZS+IKP combined fix)", "patch_combined_fill_and_atom_k"),
    Variant("autovec_load", "ITER 12: cute.autovec_copy from per-thread RMEM (race-free R2S)", "patch_use_autovec_load"),
]


def patch_combined_fill_and_atom_k(src: str) -> tuple[str, str]:
    """Combine cute.fill (race-free dequant) + atom-K-block iter (CZS fix)."""
    src1, _ = patch_cute_fill_kv(src)
    src2, _ = patch_iter_atom_k_blocks(src1)
    return src2, "Combined: cute.fill K/V + 4-atom-K-block QK iter"


def patch_use_autovec_load(src: str) -> tuple[str, str]:
    """ITER 12: Use _load_kv_from_dense_autovec method (R2S via make_fragment_like + autovec_copy).

    The autovec_copy is race-free per CuTe's atom selection. Per-thread
    RMEM fragment matches sK_mma per-thread distribution. Values are
    placeholder (zeros) in iter 12.1 — purely tests if autovec_copy
    compiles + runs race-free.
    """
    old = (
        "            self._load_kv_from_dense(\n"
        "                mK, mV, mPT, sK, sV,\n"
        "                row, tile_begin, tile_count, tid,\n"
        "            )\n"
        "            cute.arch.barrier()"
    )
    new = (
        "            # ITER 12 race-free path: autovec_copy from per-thread RMEM\n"
        "            self._load_kv_from_dense_autovec(\n"
        "                mK, mV, mPT, sK_mma, sV_mma,\n"
        "                row, tile_begin, tile_count, tid,\n"
        "            )\n"
        "            cute.arch.barrier()"
    )
    new_src = src.replace(old, new, 1)
    return new_src, "ITER 12: Use _load_kv_from_dense_autovec (cute.autovec_copy R2S, race-free)"


# === Bench harness invoked per variant ===

def run_variant_bench(variant: Variant) -> Result:
    """Apply variant patch, run bench, capture results.

    Always starts from BACKUP_PATH (the pristine baseline) so patches
    don't accumulate across variants.
    """
    result = Result(variant=variant.name, description=variant.description)

    # CRITICAL: restore baseline before applying THIS variant's patch
    # to avoid compound effects from prior variants.
    if BACKUP_PATH.exists():
        shutil.copy(BACKUP_PATH, KERNEL_PATH)

    src = KERNEL_PATH.read_text()
    patch_fn = globals()[variant.patch_fn_name]
    new_src, desc = patch_fn(src)
    result.description = desc

    if new_src == src:
        result.error = "patch is no-op (no source change)"
        return result

    # Write patched source
    KERNEL_PATH.write_text(new_src)

    # Run bench script via subprocess (clean python state)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"""
import sys, torch, importlib.util, math, time
sys.path.insert(0, "/home/spencer/optimization-playground/python/sglang/jit_kernel")
from bench_higgs_dsl import _make_inputs, _torch_reference
spec = importlib.util.spec_from_file_location("hdmd_dsl", "{KERNEL_PATH}")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
inputs = _make_inputs(1, 64, 32)
out = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device="cuda")
m.higgs_dense_2bit_mla_decode_dsl(
    inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
    inputs["page_table"], out, inputs["codebook"], inputs["sm_scale"],
)
torch.cuda.synchronize()
ref = _torch_reference(
    inputs["q_nope"], inputs["q_rope"], inputs["compressed"],
    inputs["page_table"], inputs["codebook"], inputs["sm_scale"],
)
diff = (out.float() - ref.float()).abs()
print(f"OUT min={{out.min().item():.6e}} max={{out.max().item():.6e}} std={{out.std().item():.6e}}")
print(f"DIFF max={{diff.max().item():.6e}} mean={{diff.mean().item():.6e}}")
""".strip()],
            timeout=300, capture_output=True, text=True,
        )
        result.elapsed_s = time.perf_counter() - t0
        result.compile_ok = proc.returncode == 0
        result.run_ok = result.compile_ok
        out = proc.stdout + proc.stderr
        # Parse OUT/DIFF lines
        for line in out.splitlines():
            if line.startswith("OUT "):
                for kv in line[4:].split():
                    k, _, v = kv.partition("=")
                    if k == "min": result.out_min = float(v)
                    elif k == "max": result.out_max = float(v)
                    elif k == "std": result.out_std = float(v)
            elif line.startswith("DIFF "):
                for kv in line[5:].split():
                    k, _, v = kv.partition("=")
                    if k == "max": result.max_abs = float(v)
                    elif k == "mean": result.mean_abs = float(v)
        if not result.compile_ok:
            result.error = (out[-500:] if out else "no output")
    except subprocess.TimeoutExpired:
        result.elapsed_s = time.perf_counter() - t0
        result.error = "TIMEOUT 120s"

    return result


CZS_BINARY = "/home/spencer/CZS/build/src/czs"
CZS_MODULE_JSON = "/tmp/higgs_mla_dsl_module_v2.json"


def run_czs_prove() -> tuple[int, int, int]:
    """Run czs prove on the static HIGGS module JSON; return (proved, disproved, unknown).

    Returns (-1, -1, -1) if CZS binary or module JSON not found.
    """
    if not Path(CZS_BINARY).exists() or not Path(CZS_MODULE_JSON).exists():
        return (-1, -1, -1)
    try:
        proc = subprocess.run(
            [CZS_BINARY, "prove", "--json", CZS_MODULE_JSON],
            timeout=30, capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        # Parse '[czs] N Proved | M Disproved | K Unknown' line
        m = re.search(r"\[czs\]\s+(\d+)\s+Proved\s+\|\s+(\d+)\s+Disproved\s+\|\s+(\d+)\s+Unknown", out)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except subprocess.TimeoutExpired:
        pass
    return (-1, -1, -1)


def run_racecheck(variant: Variant) -> int:
    """Run compute-sanitizer racecheck on the patched kernel; return error count.

    Returns -1 if compute-sanitizer not available or hits timeout.
    """
    sanitizer = "/usr/local/cuda-13.0/bin/compute-sanitizer"
    if not Path(sanitizer).exists():
        return -1
    try:
        proc = subprocess.run(
            [sanitizer, "--tool=racecheck", "--print-limit=1",
             sys.executable, "-c", f"""
import sys, torch, importlib.util
sys.path.insert(0, "/home/spencer/optimization-playground/python/sglang/jit_kernel")
from bench_higgs_dsl import _make_inputs
spec = importlib.util.spec_from_file_location("hdmd_dsl", "{KERNEL_PATH}")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
inputs = _make_inputs(1, 64, 32)
out = torch.zeros(1, 64, 512, dtype=torch.bfloat16, device="cuda")
m.higgs_dense_2bit_mla_decode_dsl(inputs["q_nope"], inputs["q_rope"], inputs["compressed"], inputs["page_table"], out, inputs["codebook"], inputs["sm_scale"])
torch.cuda.synchronize()
""".strip()],
            timeout=300, capture_output=True, text=True,
        )
        out = proc.stdout + proc.stderr
        m = re.search(r"RACECHECK SUMMARY:.*?\((\d+) errors", out)
        if m:
            return int(m.group(1))
    except subprocess.TimeoutExpired:
        pass
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of variants to run; default = all")
    parser.add_argument("--results-path", type=str, default=str(RESULTS_PATH))
    parser.add_argument("--racecheck", action="store_true", help="run compute-sanitizer racecheck on each variant (slow)")
    parser.add_argument("--czs", action="store_true", help="run CZS prove on the static layout module (fast)")
    args = parser.parse_args()

    # Backup baseline
    if not BACKUP_PATH.exists():
        shutil.copy(KERNEL_PATH, BACKUP_PATH)
        print(f"Backed up baseline to {BACKUP_PATH}")

    selected = VARIANTS
    if args.variants:
        selected = [v for v in VARIANTS if v.name in args.variants]

    print(f"Running {len(selected)} variants...")
    results = []
    for v in selected:
        print(f"\n=== {v.name}: {v.description} ===")
        # 1. CZS prove (static, fast) — runs on a fixed layout module that
        # describes the kernel's invariant SMEM layouts + MMA atoms
        if args.czs:
            print(f"  running CZS prove (static layout/MMA verify)...")
            (p, d, u) = run_czs_prove()
            r_czs = Result(variant="CZS_pre", description="CZS layout/MMA verify")
            r_czs.czs_proved, r_czs.czs_disproved, r_czs.czs_unknown = p, d, u
        r = run_variant_bench(v)
        if args.czs:
            r.czs_proved, r.czs_disproved, r.czs_unknown = p, d, u
        if r.compile_ok and args.racecheck:
            print(f"  running compute-sanitizer racecheck...")
            r.racecheck_errors = run_racecheck(v)
        results.append(r)
        # Persist immediately
        with open(args.results_path, "a") as f:
            f.write(json.dumps(asdict(r)) + "\n")
        # Summary line
        if r.compile_ok:
            race_str = f" races={r.racecheck_errors}" if r.racecheck_errors >= 0 else ""
            czs_str = f" CZS={r.czs_proved}P/{r.czs_disproved}D/{r.czs_unknown}U" if r.czs_proved >= 0 else ""
            status = f"compile_ok run_t={r.elapsed_s:.1f}s out_std={r.out_std:.3e} diff_mean={r.mean_abs:.3e}{race_str}{czs_str}"
        else:
            status = f"FAIL: {r.error[:100]}"
        print(f"  → {status}")

    # Restore baseline
    shutil.copy(BACKUP_PATH, KERNEL_PATH)
    print(f"\nRestored baseline from {BACKUP_PATH}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'variant':<20} {'compile':<10} {'time_s':<10} {'std':<12} {'diff_mean':<12}")
    print("-" * 80)
    for r in results:
        ok = "ok" if r.compile_ok else "FAIL"
        print(f"{r.variant:<20} {ok:<10} {r.elapsed_s:<10.1f} {r.out_std:<12.3e} {r.mean_abs:<12.3e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
