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

        # Use sm100_utils SMEM-layout helpers so the layout matches
        # the MMA atom's swizzle / major-mode requirements (raw
        # row-major plain layouts produce "Operation creation failed"
        # from the MMA fragment builders).
        q_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.mma_qk_tiler, bf16, 1
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.mma_qk_tiler, bf16, 1
        )
        v_smem_layout = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.mma_pv_tiler, bf16, 1
        )
        p_smem_layout = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.mma_pv_tiler, bf16, 1
        )

        @cute.struct
        class SharedStorage:
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
        # Build SMEM tensors via MemRange.get_tensor(outer, swizzle=inner)
        # so the swizzle attaches to the pointer (required by tcgen05
        # make_fragment_A/B; passing a composed layout to make_tensor
        # produces an "Expected affine layout" error).
        sQ = storage.smem_q.get_tensor(
            q_smem_layout.outer, swizzle=q_smem_layout.inner
        )
        sK = storage.smem_k.get_tensor(
            k_smem_layout.outer, swizzle=k_smem_layout.inner
        )
        sV = storage.smem_v.get_tensor(
            v_smem_layout.outer, swizzle=v_smem_layout.inner
        )
        sP = storage.smem_p.get_tensor(
            p_smem_layout.outer, swizzle=p_smem_layout.inner
        )

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

        # Step 1: Q rotation. FWHT_512 + 1/sqrt(LATENT_DIM) on each Q
        # row of q_nope into sQ[h, :LATENT_DIM]; append q_rope (BF16,
        # no transform) into sQ[h, LATENT_DIM:].
        num_heads_total = mQN.shape[1]
        inv_root_d = Float32(INV_SQRT_LATENT)
        for h_local in cutlass.range_constexpr(self.block_h):
            h_global = head_base + h_local
            valid = h_global < num_heads_total
            v0 = Float32(0.0)
            v1 = Float32(0.0)
            v2 = Float32(0.0)
            v3 = Float32(0.0)
            if valid:
                v0 = mQN[row, h_global, 0 * 128 + tid].to(Float32)
                v1 = mQN[row, h_global, 1 * 128 + tid].to(Float32)
                v2 = mQN[row, h_global, 2 * 128 + tid].to(Float32)
                v3 = mQN[row, h_global, 3 * 128 + tid].to(Float32)
            v0 = self._fwht_128(v0, tid, storage.fwht_scratch)
            cute.arch.barrier()
            v1 = self._fwht_128(v1, tid, storage.fwht_scratch)
            cute.arch.barrier()
            v2 = self._fwht_128(v2, tid, storage.fwht_scratch)
            cute.arch.barrier()
            v3 = self._fwht_128(v3, tid, storage.fwht_scratch)
            cute.arch.barrier()
            v0, v1, v2, v3 = self._fwht_register_top2(v0, v1, v2, v3)
            v0 = v0 * inv_root_d
            v1 = v1 * inv_root_d
            v2 = v2 * inv_root_d
            v3 = v3 * inv_root_d
            sQ[h_local, 0 * 128 + tid] = v0.to(cutlass.BFloat16)
            sQ[h_local, 1 * 128 + tid] = v1.to(cutlass.BFloat16)
            sQ[h_local, 2 * 128 + tid] = v2.to(cutlass.BFloat16)
            sQ[h_local, 3 * 128 + tid] = v3.to(cutlass.BFloat16)
            if tid < ROPE_DIM:
                rope_val = (
                    mQR[row, h_global, tid]
                    if valid
                    else cutlass.BFloat16(0.0)
                )
                sQ[h_local, LATENT_DIM + tid] = rope_val
            cute.arch.barrier()

        # TMEM allocation.
        # acc TMEM partitioned across 2 chunks for the latent dim:
        #   acc_lo at cols 0..255 holds acc[block_h, 0:256]
        #   acc_hi at cols 256..511 holds acc[block_h, 256:512]
        # score TMEM time-shares cols 0..(block_n-1) — overwrites
        # acc_lo during MMA, read out immediately, then acc MMA writes
        # back into the same TMEM region with accumulate=False on the
        # first iter / True after.
        tmem_alloc_cols = 512
        cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = storage.tmem_holding_buf

        # Acc fragments (per N-chunk). Their layouts come from the PV
        # MMA's partition_C of the PV tile shape (M=block_h, N=256).
        tAcc_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tAcc_proto = pv_tiled_mma.make_fragment_C(tAcc_shape)
        # acc_lo lives at TMEM col 0 (overlaps score during MMA).
        tAcc_lo = cute.make_tensor(tmem_ptr, tAcc_proto.layout)
        # acc_hi lives at TMEM col 256.
        tAcc_hi = cute.make_tensor(tmem_ptr + 256, tAcc_proto.layout)

        # Score fragment (QK tile output). Same TMEM region as acc_lo;
        # transient, consumed by softmax right after each MMA.
        tScore_shape = qk_tiled_mma.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tScore_proto = qk_tiled_mma.make_fragment_C(tScore_shape)
        tScore = cute.make_tensor(tmem_ptr, tScore_proto.layout)

        # MMA operand fragments.
        tCrQ = qk_tiled_mma.make_fragment_A(sQ)
        tCrK = qk_tiled_mma.make_fragment_B(sK)
        tCrP = pv_tiled_mma.make_fragment_A(sP)
        tCrV = pv_tiled_mma.make_fragment_B(sV)

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

            # (B) Score MMA: tScore = sQ @ sK^T over 9 K-tiles.
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
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

        cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

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
        """Per-slot dequant: latent → sK + sV simultaneously; rope →
        sK only. Mirrors C++ higgs_unpack_indices lane assignment.
        """
        zero_bf = cutlass.BFloat16(0.0)
        for n in cutlass.range_constexpr(self.block_n):
            col = tile_begin + n
            valid = col < tile_count
            page = Int32(-1)
            if valid:
                page = mPT[row, col]
            page_valid = page >= Int32(0)
            slot_idx = page if page_valid else Int32(0)

            pair_within_group = tid >> 1
            coord_lane = tid & 1
            byte_in_group = pair_within_group >> 1
            nibble = pair_within_group & 1
            b0 = Int32(0); b1 = Int32(0); b2 = Int32(0); b3 = Int32(0)
            if coord_lane == 0:
                b0 = mCK[slot_idx, 0 * 32 + byte_in_group].to(Int32)
                b1 = mCK[slot_idx, 1 * 32 + byte_in_group].to(Int32)
                b2 = mCK[slot_idx, 2 * 32 + byte_in_group].to(Int32)
                b3 = mCK[slot_idx, 3 * 32 + byte_in_group].to(Int32)
            pb0 = cute.arch.shuffle_sync_bfly(b0, 1, mask=0xffffffff)
            pb1 = cute.arch.shuffle_sync_bfly(b1, 1, mask=0xffffffff)
            pb2 = cute.arch.shuffle_sync_bfly(b2, 1, mask=0xffffffff)
            pb3 = cute.arch.shuffle_sync_bfly(b3, 1, mask=0xffffffff)
            b0 = pb0 if coord_lane else b0
            b1 = pb1 if coord_lane else b1
            b2 = pb2 if coord_lane else b2
            b3 = pb3 if coord_lane else b3
            i0 = (b0 >> 4) if nibble else (b0 & 0x0F)
            i1 = (b1 >> 4) if nibble else (b1 & 0x0F)
            i2 = (b2 >> 4) if nibble else (b2 & 0x0F)
            i3 = (b3 >> 4) if nibble else (b3 & 0x0F)
            c0 = storage.smem_codebook[i0 * PAIR_DIM + coord_lane]
            c1 = storage.smem_codebook[i1 * PAIR_DIM + coord_lane]
            c2 = storage.smem_codebook[i2 * PAIR_DIM + coord_lane]
            c3 = storage.smem_codebook[i3 * PAIR_DIM + coord_lane]
            scale_h = mCS[slot_idx]
            scale = scale_h.to(Float32) if page_valid else Float32(0.0)
            d0 = const_expr(0 * 128) + tid
            d1 = const_expr(1 * 128) + tid
            d2 = const_expr(2 * 128) + tid
            d3 = const_expr(3 * 128) + tid
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
            if tid < ROPE_DIM:
                rope_val = mCR[slot_idx, tid] if page_valid else zero_bf
                sK[n, const_expr(LATENT_DIM) + tid] = rope_val

    @cute.jit
    def _score_softmax_to_p(
        self, qk_tiled_mma, tScore, sP, storage,
        softmax_scale_log2, tile_count, tid,
    ):
        """TMEM->RMEM read of score; per-row online softmax; alpha to
        SMEM; p (BF16) to sP. Iter 1 uses a coordinate scatter into
        ss.softmax_alpha; iter 2 will switch to a per-warp reduction.
        """
        # Build the TMEM->RMEM tiled copy.
        tiled_t2r = tcgen05.make_tmem_copy(
            tcgen05.copy.Ld16x32bx2Op(num_dp=16),
            tScore,
        )
        thr_t2r = tiled_t2r.get_slice(tid)
        tDtScore = thr_t2r.partition_S(tScore)
        tDrScore = cute.make_fragment_like(tDtScore)
        cute.copy(tiled_t2r, tDtScore, tDrScore)
        # Per-thread fragment scatter into sP (as fp32 first, then
        # converted at p_smem write). Coordinate iter via identity.
        cScore = cute.make_identity_tensor(
            (const_expr(self.block_h), const_expr(self.block_n))
        )
        # Iter-1 simplification: rely on the C++ kernel's correctness
        # pattern — write per-element through coord iterator. Cast to
        # BF16 in the write.
        for i in cutlass.range_constexpr(cute.size(tDrScore)):
            coord = tScore.crd(thr_t2r.idx, i)  # may need adjustment
            m_idx = cute.get(coord, 0)
            n_idx = cute.get(coord, 1)
            s = tDrScore[i] * softmax_scale_log2 if (n_idx < tile_count) else Float32(-3e38)
            sP[m_idx, n_idx] = s.to(cutlass.BFloat16)
        # Per-row max + sum reduction across the block. ITER 1 STUB:
        # for now we use sm_scale-only softmax (no max subtraction),
        # which works numerically only for small score magnitudes;
        # iter 2 adds proper online softmax.
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

    kernel = HiggsDense2bitMLADecodeDSL(block_h=block_h, block_n=block_n)

    mQN = from_dlpack(q_nope.contiguous(), assumed_align=16)
    mQR = from_dlpack(q_rope.contiguous(), assumed_align=16)
    mCK = from_dlpack(packed, assumed_align=16)
    mCS = from_dlpack(scale, assumed_align=2)
    mCR = from_dlpack(rope, assumed_align=16)
    mPT = from_dlpack(page_table.contiguous(), assumed_align=16)
    mO = from_dlpack(out, assumed_align=16)
    mCB = from_dlpack(codebook.contiguous(), assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(mQN, mQR, mCK, mCS, mCR, mPT, mO, mCB, sm_scale, stream)
