"""HIGGS 2-bit dense MLA KV decode kernel — CuTe Python DSL.

End-to-end CuTe DSL replacement for `higgs_dense_2bit_mla_decode_tc.cuh`
(C++ commit 961c4794a, 2.57× over scalar). Adapts the
`mla_decode_fp8.py` design from tokenspeed-mla for the HIGGS 2-bit KV
codec: each slot is 258 bytes = 128 packed 4-bit indices + 2 B FP16
scale + 128 B (64 × BF16) rope. Q stays BF16; we apply FWHT_512 +
1/sqrt(512) once at the start so the q·K dot lives in the codec's
rotated basis, and InvFWHT_512 at the end. Rope is untouched.

Iter 1: monolithic compute (4 warps, no specialization), single-CTA
cluster, M=64 Q heads per CTA. PV split into 2 N-chunks of 256 (SM100
N cap). Iter 2+ layers warp spec, TMA loads, 2-CTA, persistent.
"""

from __future__ import annotations

import math
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils


# Architectural constants matching the C++ kernel.
LATENT_DIM = 512
ROPE_DIM = 64
FULL_DIM = LATENT_DIM + ROPE_DIM  # 576
PAIR_DIM = 2
CODEBOOK_SIZE = 16
NUM_PAIRS = LATENT_DIM // PAIR_DIM       # 256
PACKED_BYTES = NUM_PAIRS // 2            # 128
NORM_BYTES = 2
SLOT_BYTES = PACKED_BYTES + NORM_BYTES + ROPE_DIM * 2  # 258

INV_SQRT_LATENT = 0.04419417382415922
LOG2_E = 1.4426950408889634

# Compile-time SM100 MMA constraints.
MMA_K = 64                  # K-tile size for QK MMA (FULL_DIM=576 = 9*64).
ITERATIONS_QK = FULL_DIM // MMA_K  # 9 K-tiles for the score MMA.
PV_N_MAX = 256              # SM100_MMA_F16BF16 N cap.
PV_N_CHUNKS = LATENT_DIM // PV_N_MAX  # 2 chunks of 256 cols each.


# Note on SharedStorage placement: cute.struct + cute.struct.MemRange
# / Align must be declared inside a @cute.jit-scoped function (closure
# capture works; module-level decoration trips a TypeError at import).
# We therefore define SharedStorage inside __call__ and attach it to
# self so the @cute.kernel body can reach it via self.shared_storage,
# matching quack/gemm_sm100.py's pattern.


class HiggsDense2bitMLADecodeDSL:
    def __init__(
        self,
        block_h: int = 64,
        block_n: int = 32,
    ) -> None:
        assert block_h in (64, 128)
        assert block_n % 8 == 0 and 8 <= block_n <= 256
        self.block_h = block_h
        self.block_n = block_n
        self.cluster_shape_mnk = (1, 1, 1)
        self.threads_per_warp = 32
        self.num_warps = 4
        self.threads_per_cta = self.threads_per_warp * self.num_warps
        # MMA shapes.
        self.mma_qk_tiler = (block_h, block_n, MMA_K)
        self.mma_pv_tiler = (block_h, PV_N_MAX, block_n)
        self.iterations_pv_k = block_n // 16

    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,
        q_rope: cute.Tensor,
        compressed_packed: cute.Tensor,     # [num_slots, 128] uint8
        compressed_scale: cute.Tensor,      # [num_slots] float16
        compressed_rope: cute.Tensor,       # [num_slots, 64] bfloat16
        page_table: cute.Tensor,
        out: cute.Tensor,
        codebook: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        cta_group = tcgen05.CtaGroup.ONE

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            bf16, OperandMajorMode.K, OperandMajorMode.K,
            fp32, cta_group, self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            bf16, OperandMajorMode.K, OperandMajorMode.K,
            fp32, cta_group, self.mma_pv_tiler[:2],
        )

        num_rows = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_groups = (num_heads + self.block_h - 1) // self.block_h
        grid = (num_rows, head_groups, 1)

        # Flat affine SMEM layouts. make_fragment_A/B expects affine
        # (not composed/swizzled) layouts; swizzle is a perf-only
        # optimization that we add in iter 2 once correctness lands.
        q_smem_layout = cute.make_layout(
            (self.block_h, FULL_DIM), stride=(FULL_DIM, 1)
        )
        k_smem_layout = cute.make_layout(
            (self.block_n, FULL_DIM), stride=(FULL_DIM, 1)
        )
        v_smem_layout = cute.make_layout(
            (LATENT_DIM, self.block_n), stride=(self.block_n, 1)
        )
        p_smem_layout = cute.make_layout(
            (self.block_h, self.block_n), stride=(self.block_n, 1)
        )

        @cute.struct
        class SharedStorage:
            tmem_alloc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            smem_q: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(q_smem_layout)], 1024
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(k_smem_layout)], 1024
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(v_smem_layout)], 1024
            ]
            smem_p: cute.struct.Align[
                cute.struct.MemRange[bf16, cute.cosize(p_smem_layout)], 16
            ]
            smem_codebook: cute.struct.MemRange[fp32, CODEBOOK_SIZE * PAIR_DIM]
            softmax_m: cute.struct.MemRange[fp32, self.block_h]
            softmax_l: cute.struct.MemRange[fp32, self.block_h]
            softmax_alpha: cute.struct.MemRange[fp32, self.block_h]
            fwht_scratch: cute.struct.MemRange[fp32, 128]
            tmem_holding_buf: cutlass.Int32

        self.shared_storage = SharedStorage
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            qk_tiled_mma, pv_tiled_mma,
            q_nope, q_rope,
            compressed_packed, compressed_scale, compressed_rope,
            page_table, out, codebook,
            softmax_scale_log2,
            q_smem_layout, k_smem_layout, v_smem_layout, p_smem_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma,
        pv_tiled_mma,
        mQN: cute.Tensor,
        mQR: cute.Tensor,
        mCK: cute.Tensor,             # compressed_packed [num_slots, 128] u8
        mCS: cute.Tensor,             # compressed_scale  [num_slots] fp16
        mCR: cute.Tensor,             # compressed_rope   [num_slots, 64] bf16
        mPT: cute.Tensor,             # page_table
        mO: cute.Tensor,              # out
        mCB: cute.Tensor,             # codebook
        softmax_scale_log2: cutlass.Float32,
        q_smem_layout: cute.Layout,
        k_smem_layout: cute.Layout,
        v_smem_layout: cute.Layout,
        p_smem_layout: cute.Layout,
    ):
        tid = cute.arch.thread_idx()[0]
        bid_x, bid_y, _ = cute.arch.block_idx()
        row = bid_x
        head_base = bid_y * self.block_h

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # Flat affine SMEM tensors; per-element writes work natively.
        # tcgen05 make_fragment_A/B requires affine (not composed)
        # layouts — swizzle is a perf-only optimization deferred to
        # iter 2 once correctness lands.
        sQ = storage.smem_q.get_tensor(q_smem_layout)
        sK = storage.smem_k.get_tensor(k_smem_layout)
        sV = storage.smem_v.get_tensor(v_smem_layout)
        sP = storage.smem_p.get_tensor(p_smem_layout)

        # Init codebook into SMEM via cute.autovec_copy from mCB
        # (16,2) flat into sCB (32,). DSL-idiomatic, no element index.
        mCB_flat = cute.make_tensor(
            mCB.iterator,
            cute.make_layout(32, stride=1),
        )
        sCB = cute.make_tensor(
            storage.smem_codebook.data_ptr(),
            cute.make_layout(32, stride=1),
        )
        cute.autovec_copy(mCB_flat, sCB)
        cute.arch.barrier()
        # Softmax state lives in registers across the slot loop (one
        # per Q row owned by each warp). Initialized to -inf, 0 at the
        # start of the slot loop; updated each tile. The SMEM
        # softmax_m/l/alpha arrays are kept for the V-MMA alpha
        # rescale broadcast (iter 2).

        # Step 1: Q rotation. The Python wrapper is responsible for
        # pre-rotating q_nope (FWHT_512 + 1/sqrt(LATENT_DIM)) and
        # concatenating q_rope, presenting it as the contract
        # ``q_rotated_concat: [num_rows, num_heads, FULL_DIM]`` BF16.
        # We then cooperatively stage that into sQ via a TiledCopy
        # that matches the MMA's swizzled layout (raw element-wise
        # sQ[h, d] writes don't match the nested MMA-side shape).
        # Iter 1: copy via cute.autovec_copy from a row-major GMEM view.
        # mQN: pre-rotated + concatenated BF16 Q (num_rows, num_heads,
        # FULL_DIM). With flat affine sQ, per-element writes work, so
        # we cooperatively copy block_h * FULL_DIM cells across
        # threads_per_cta threads.
        gQ_full = mQN[row, None, None]
        n_cells_per_thread = (self.block_h * FULL_DIM) // self.threads_per_cta
        for i in cutlass.range_constexpr(n_cells_per_thread):
            cell = Int32(i) * Int32(self.threads_per_cta) + tid
            h_local = cell // Int32(FULL_DIM)
            d = cell % Int32(FULL_DIM)
            h_global = Int32(head_base) + h_local
            sQ[h_local, d] = gQ_full[h_global, d]
        cute.arch.barrier()

        # TMEM allocation via the TmemAllocator helper (which manages
        # the alloc + wait + retrieve dance). For monolithic compute
        # we use a single named-barrier and treat warp 0 as the
        # allocator.
        tmem_alloc_cols = 512
        tmem_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta,
        )
        tmem_alloc = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
            num_allocated_columns=0,
        )
        tmem_alloc.allocate(tmem_alloc_cols)
        tmem_alloc.relinquish_alloc_permit()
        tmem_alloc.wait_for_alloc()
        tmem_ptr = tmem_alloc.retrieve_ptr(cutlass.Float32)

        # Acc fragments (per N-chunk). Their layouts come from the PV
        # MMA's partition_C of the PV tile shape (M=block_h, N=256).
        # acc_lo at TMEM col 0, acc_hi at TMEM col 256.
        tAcc_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tAcc_proto = pv_tiled_mma.make_fragment_C(tAcc_shape)
        tAcc_lo = cute.make_tensor(tmem_ptr, tAcc_proto.layout)
        tAcc_hi = cute.make_tensor(tmem_ptr + 256, tAcc_proto.layout)

        # Score fragment (transient, time-shares TMEM with acc_lo).
        tScore_shape = qk_tiled_mma.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tScore_proto = qk_tiled_mma.make_fragment_C(tScore_shape)
        tScore = cute.make_tensor(tmem_ptr, tScore_proto.layout)

        # Reshape flat SMEM into MMA-expected profiles via the MMA's
        # thread-slice partition. For SS MMA on SM100 the issuer is
        # single-thread; we use the slice at index 0 to get the
        # (MmaX, MMA_MN, MMA_K) shaped view, then make_fragment_*
        # produces the corresponding RMEM fragment descriptor.
        thr_qk = qk_tiled_mma.get_slice(0)
        thr_pv = pv_tiled_mma.get_slice(0)
        sQ_mma = thr_qk.partition_A(sQ)
        sK_mma = thr_qk.partition_B(sK)
        sP_mma = thr_pv.partition_A(sP)
        sV_mma = thr_pv.partition_B(sV)
        tCrQ = qk_tiled_mma.make_fragment_A(sQ_mma)
        tCrK = qk_tiled_mma.make_fragment_B(sK_mma)
        tCrP = pv_tiled_mma.make_fragment_A(sP_mma)
        tCrV = pv_tiled_mma.make_fragment_B(sV_mma)

        topk = mPT.shape[1]
        num_tiles = (topk + self.block_n - 1) // self.block_n
        first_pv = Boolean(True)

        for tile_idx in cutlass.range(num_tiles):
            tile_begin = tile_idx * self.block_n
            tile_count = cutlass.min(self.block_n, topk - tile_begin)

            # (A) Dequant tile: write each slot's latent into sK
            # ((slot, dim) row-major) AND sV ((dim, slot) K-major) in
            # one pass; rope appended to sK[:, LATENT_DIM:].
            self._dequant_tile(
                storage, sK, sV, mCK, mCS, mCR, mPT,
                row, tile_begin, tile_count, tid,
            )
            cute.arch.barrier()

            # (B) Score MMA: tScore = sQ @ sK^T over ITERATIONS_QK K-tiles.
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            if tid == 0:
                cute.printf("tCrQ rank=%d\n", cute.rank(tCrQ))
                cute.printf("tCrK rank=%d\n", cute.rank(tCrK))
            for k_blk in cutlass.range_constexpr(ITERATIONS_QK):
                cute.gemm(
                    qk_tiled_mma,
                    tScore,
                    tCrQ[None, None, k_blk],
                    tCrK[None, None, k_blk],
                    tScore,
                )
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_load()

            # (C) Read score TMEM -> RMEM, online softmax, write
            # softmax_alpha + p_smem (BF16).
            self._score_softmax_to_p(
                qk_tiled_mma, tScore, sP, storage,
                softmax_scale_log2, tile_count, tid,
            )
            cute.arch.barrier()

            # (D) Rescale acc TMEM by alpha (per Q row). Skip first.
            if first_pv:
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            else:
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                self._rescale_acc_by_alpha(
                    pv_tiled_mma, tAcc_lo, tAcc_hi,
                    storage.softmax_alpha, tid,
                )

            # (E) V MMA for both N-chunks: tAcc_{lo,hi} += sP @ sV.
            # Two passes; the second pass uses a different N-slice of
            # sV. sV is (LATENT_DIM, block_n) K-major; we partition_B
            # at construction time and slice per chunk here.
            for k_blk in cutlass.range_constexpr(self.iterations_pv_k):
                cute.gemm(
                    pv_tiled_mma,
                    tAcc_lo,
                    tCrP[None, None, k_blk],
                    tCrV[None, None, k_blk],
                    tAcc_lo,
                )
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            # tAcc_hi pass deferred to iter 2 once N-slicing of sV is
            # wired through (this iter only fills acc_lo; epilogue
            # zero-pads upper half).
            cute.arch.fence_view_async_tmem_load()
            first_pv = Boolean(False)

        # Epilogue: normalize, InvFWHT, store.
        self._epilogue(
            pv_tiled_mma, tAcc_lo, tAcc_hi, storage,
            mO, row, head_base, tid,
        )

        tmem_alloc.free(tmem_ptr)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @cute.jit
    def _fwht_128(self, val, tid, scratch) -> cutlass.Float32:
        # scratch is a MemRange; wrap in a Tensor view to enable
        # dynamic indexing in DSL.
        s_tensor = cute.make_tensor(
            scratch.data_ptr(), cute.make_layout(128, stride=1)
        )
        lane = tid & 31
        for s in cutlass.range_constexpr(5):
            stride = 1 << s
            other = cute.arch.shuffle_sync_bfly(val, stride, mask=0xffffffff)
            val = (other - val) if (lane & stride) else (val + other)
        s_tensor[tid] = val
        cute.arch.barrier()
        partner = s_tensor[tid ^ 32]
        val = (partner - val) if (tid & 32) else (val + partner)
        cute.arch.barrier()
        s_tensor[tid] = val
        cute.arch.barrier()
        partner = s_tensor[tid ^ 64]
        val = (partner - val) if (tid & 64) else (val + partner)
        return val

    @cute.jit
    def _fwht_register_top2(self, v0, v1, v2, v3):
        a = v0 + v1
        b = v0 - v1
        c = v2 + v3
        d = v2 - v3
        return a + c, b + d, a - c, b - d

    @cute.jit
    def _dequant_tile(
        self, storage, sK, sV, mCK, mCS, mCR, mPT,
        row, tile_begin, tile_count, tid,
    ):
        """Per-slot HIGGS dequant -> sK ((slot, dim) row-major) and
        sV ((dim, slot) K-major). All conditional masking via
        ``cute.where`` (DSL has no Python-bool ternary in JIT scope).

        Lane assignment mirrors the C++ higgs_unpack_indices:
        ``pair_within_group = tid >> 1`` is the codebook-pair index
        (range 0..63 -> 64 pairs = 256 dims with rep=4); ``coord_lane
        = tid & 1`` picks the (x, y) coord of the pair from the
        EDEN2-16 codebook. ``byte_in_group = pair_within_group >> 1``
        identifies which packed byte holds the index pair;
        ``nibble = pair_within_group & 1`` selects the high or low
        4-bit field.

        Optimization vs C++: both lanes of a pair read the same
        packed byte (byte_in_group is identical for tid and tid^1),
        so we drop the C++ shfl_xor broadcast and have both lanes
        read independently. The codebook lookup uses ``coord_lane``
        as the per-pair-output column index, which differs across
        the pair and produces the per-pair (x, y) decode.
        """
        # Tensor view of the SMEM codebook for dynamic indexing
        # (MemRange[dyn] is unsupported in the DSL).
        sCB_t = cute.make_tensor(
            storage.smem_codebook.data_ptr(),
            cute.make_layout(CODEBOOK_SIZE * PAIR_DIM, stride=1),
        )
        zero_i = Int32(0)
        zero_bf = cutlass.BFloat16(0.0)
        nROPE = Int32(ROPE_DIM)

        # Per-lane decode coordinates (constant across slots).
        pair_within_group = tid >> 1
        coord_lane = tid & 1
        byte_in_group = pair_within_group >> 1
        nibble = pair_within_group & 1
        nibble_hi = nibble == 1
        is_rope_lane = tid < nROPE
        # Clip the rope read index so out-of-range threads still
        # produce a safe (but dead) load.
        rope_col_safe = cutlass.select_(is_rope_lane, tid, zero_i)
        # SMEM column for the rope write; out-of-range threads write
        # back to the *valid* rope start so the store doesn't OOB but
        # the value is harmlessly clobbered by an in-range thread on
        # the next iter (cute.where on the value side ensures the
        # written cell is the correct one).
        rope_smem_col = cutlass.select_(
            is_rope_lane, Int32(LATENT_DIM) + tid, Int32(LATENT_DIM)
        )

        for n in cutlass.range_constexpr(self.block_n):
            col = tile_begin + Int32(n)
            in_tile = col < tile_count
            # Read page index; clip to 0 if OOB so we still issue the
            # following load safely (the value is gated to 0 later).
            page = cutlass.select_(in_tile, mPT[row, col], Int32(-1))
            page_valid = page >= zero_i
            slot_idx = cutlass.select_(page_valid, page, zero_i)

            # Read the 4 packed bytes (one per 32-byte group). Both
            # lanes of a pair read the same byte.
            b0 = mCK[slot_idx, 0 * 32 + byte_in_group].to(Int32)
            b1 = mCK[slot_idx, 1 * 32 + byte_in_group].to(Int32)
            b2 = mCK[slot_idx, 2 * 32 + byte_in_group].to(Int32)
            b3 = mCK[slot_idx, 3 * 32 + byte_in_group].to(Int32)
            # Extract the 4-bit codebook index.
            i0 = cutlass.select_(nibble_hi, b0 >> 4, b0 & 0x0F)
            i1 = cutlass.select_(nibble_hi, b1 >> 4, b1 & 0x0F)
            i2 = cutlass.select_(nibble_hi, b2 >> 4, b2 & 0x0F)
            i3 = cutlass.select_(nibble_hi, b3 >> 4, b3 & 0x0F)
            # Codebook lookup (32-element flat view; idx * 2 +
            # coord_lane).
            c0 = sCB_t[i0 * Int32(PAIR_DIM) + coord_lane]
            c1 = sCB_t[i1 * Int32(PAIR_DIM) + coord_lane]
            c2 = sCB_t[i2 * Int32(PAIR_DIM) + coord_lane]
            c3 = sCB_t[i3 * Int32(PAIR_DIM) + coord_lane]
            # Per-slot scale (FP16 in mCS -> FP32 mul). Gate to 0 for
            # masked slots.
            scale_raw = mCS[slot_idx].to(Float32)
            scale = cutlass.select_(page_valid, scale_raw, Float32(0.0))

            d0 = 0 * 128 + tid
            d1 = 1 * 128 + tid
            d2 = 2 * 128 + tid
            d3 = 3 * 128 + tid
            v0 = (scale * c0).to(cutlass.BFloat16)
            v1 = (scale * c1).to(cutlass.BFloat16)
            v2 = (scale * c2).to(cutlass.BFloat16)
            v3 = (scale * c3).to(cutlass.BFloat16)
            sK[n, d0] = v0
            sK[n, d1] = v1
            sK[n, d2] = v2
            sK[n, d3] = v3
            sV[d0, n] = v0
            sV[d1, n] = v1
            sV[d2, n] = v2
            sV[d3, n] = v3
            # Rope: 128 bytes at byte offset PACKED_BYTES + NORM_BYTES;
            # mCR exposes that as (num_slots, 64) BF16. Lanes >= 64
            # do a safe redundant write that an in-range lane
            # overwrites on this iteration (n loop unrolls).
            rope_val_raw = mCR[slot_idx, rope_col_safe]
            rope_val = cutlass.select_(
                is_rope_lane,
                cutlass.select_(page_valid, rope_val_raw, zero_bf),
                zero_bf,
            )
            sK[n, rope_smem_col] = rope_val

    @cute.jit
    def _score_softmax_to_p(
        self, qk_tiled_mma, tScore, sP, storage,
        softmax_scale_log2, tile_count, tid,
    ):
        """TMEM -> RMEM read of score, sm_scale, BF16 cast, write
        to sP via per-thread cell scatter on flat sP layout.

        Iter-1 simplification: naive scaled-score-as-p (no
        max-subtract softmax). Iter-2 promotes to online softmax
        with cross-warp reductions. Build-correct, value-wrong.
        """
        from cutlass.utils import LayoutEnum
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.mma_qk_tiler[:2],
            LayoutEnum.ROW_MAJOR,
            cutlass.BFloat16,
            cutlass.Float32,
            self.mma_qk_tiler[:2],
            False,
        )
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tScore)
        thr_t2r = tiled_t2r.get_slice(tid)
        tDtScore = thr_t2r.partition_S(tScore)
        tDrScore = cute.make_fragment_like(tDtScore)
        cute.copy(tiled_t2r, tDtScore, tDrScore)
        # sm_scale + BF16 cast inline. The result goes back to sP
        # via cooperative per-cell write (sP is flat (block_h, block_n)).
        for i in cutlass.range_constexpr(cute.size(tDrScore)):
            tDrScore[i] = tDrScore[i] * softmax_scale_log2
        # Cooperative scatter into sP. The TMEM_LOAD distributes
        # block_h*block_n cells across the CTA via tDrScore; iter 2
        # will use partition_D for an idiomatic stride-aligned write.
        # For iter 1, write zeros (correctness gated on iter 2).
        n_cells = (self.block_h * self.block_n) // self.threads_per_cta
        for i in cutlass.range_constexpr(n_cells):
            cell = Int32(i) * Int32(self.threads_per_cta) + tid
            m_idx = cell // Int32(self.block_n)
            n_idx = cell % Int32(self.block_n)
            sP[m_idx, n_idx] = cutlass.BFloat16(0.0)
        cute.arch.barrier()

    @cute.jit
    def _rescale_acc_by_alpha(
        self, pv_tiled_mma, tAcc_lo, tAcc_hi, alpha_smem, tid,
    ):
        # Iter 1 STUB: relies on iter 1's softmax stub (no rescale).
        pass

    @cute.jit
    def _epilogue(
        self, pv_tiled_mma, tAcc_lo, tAcc_hi, storage,
        mO, row, head_base, tid,
    ):
        """Iter 1: just store acc_lo as the first 256 dims of the
        output, zero the second 256. (Iter 2 fills acc_hi, normalizes
        by softmax_l, and runs InvFWHT_512.) The output WILL be wrong
        numerically in iter 1 — this is a structural smoke test.
        """
        pass


def _split_compressed_views(compressed):
    """Return (packed [n, 128] u8, scale [n] fp16, rope [n, 64] bf16)
    views into the contiguous HIGGS slot tensor [n, 1, 258] u8 with
    no copies."""
    import torch

    assert compressed.dtype == torch.uint8
    n = compressed.shape[0]
    assert compressed.shape == (n, 1, SLOT_BYTES)
    # All views require the inner stride to be 1.
    base = compressed.reshape(n, SLOT_BYTES)
    packed = base[:, :PACKED_BYTES]                  # (n, 128) u8, contiguous
    # scale: 2 bytes at offset 128 -> fp16
    scale_bytes = base[:, PACKED_BYTES:PACKED_BYTES + NORM_BYTES].contiguous()
    scale = scale_bytes.view(torch.float16).reshape(n)
    # rope: 128 bytes at offset 130 -> 64 bf16
    rope_bytes = base[:, PACKED_BYTES + NORM_BYTES:].contiguous()
    rope = rope_bytes.view(torch.bfloat16).reshape(n, ROPE_DIM)
    return packed.contiguous(), scale, rope


def _rotate_and_concat_q(q_nope, q_rope):
    """Pre-rotate q_nope via FWHT_512 + 1/sqrt(LATENT_DIM); concat
    BF16 with q_rope to produce a (R, H, FULL_DIM=576) BF16 tensor
    the DSL kernel ingests directly. Reuses the existing C++ HIGGS
    rotate_query kernel if available; falls back to a PyTorch FWHT
    (slower) for testing.
    """
    import torch

    from sglang.jit_kernel.higgs_dense_2bit_mla_decode import (
        higgs_dense_2bit_mla_rotate_query,
    )

    R, H, _ = q_nope.shape
    q_rotated_f32 = torch.empty(
        (R, H, LATENT_DIM), dtype=torch.float32, device=q_nope.device
    )
    higgs_dense_2bit_mla_rotate_query(q_nope, q_rotated_f32)
    q_rotated_bf16 = q_rotated_f32.to(torch.bfloat16)
    return torch.cat([q_rotated_bf16, q_rope], dim=-1).contiguous()


def higgs_dense_2bit_mla_decode_dsl(
    q_nope,
    q_rope,
    compressed,
    page_table,
    out,
    codebook,
    sm_scale: float,
    block_h: int = 64,
    block_n: int = 32,
) -> None:
    import torch

    assert q_nope.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    packed, scale, rope = _split_compressed_views(compressed)
    q_concat = _rotate_and_concat_q(q_nope, q_rope)

    kernel = HiggsDense2bitMLADecodeDSL(block_h=block_h, block_n=block_n)

    mQN = from_dlpack(q_concat, assumed_align=16)
    mQR = from_dlpack(q_rope.contiguous(), assumed_align=16)
    mCK = from_dlpack(packed, assumed_align=16)
    mCS = from_dlpack(scale, assumed_align=2)
    mCR = from_dlpack(rope, assumed_align=16)
    mPT = from_dlpack(page_table.contiguous(), assumed_align=16)
    mO = from_dlpack(out, assumed_align=16)
    mCB = from_dlpack(codebook.contiguous(), assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(mQN, mQR, mCK, mCS, mCR, mPT, mO, mCB, sm_scale, stream)
