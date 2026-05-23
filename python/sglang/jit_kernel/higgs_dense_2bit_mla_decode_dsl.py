"""HIGGS 2-bit dense MLA KV decode kernel — CuTe Python DSL.

End-to-end CuTe DSL replacement for `higgs_dense_2bit_mla_decode_tc.cuh`
(C++ commit 961c4794a, 2.57x over scalar baseline). Adapts the
`mla_decode_fp8.py` design from tokenspeed-mla but with a HIGGS 2-bit
KV codec: each slot is one packed uint8[258] tensor (128 packed 4-bit
indices + 2 B FP16 scale + 64 BF16 rope). Q stays BF16; we apply
FWHT_512 + 1/sqrt(512) once at the start so dot products live in the
rotated basis the codec uses on K, and a single InvFWHT_512 at the
end. The rope dim is untouched.

Iter 1 design (this file):
- Monolithic compute model: 4 warps / 128 threads per CTA, no
  warp-role specialization yet.
- Single-CTA cluster (no 2-CTA tcgen05 yet).
- M=64 Q heads per CTA (SM100 tcgen05.mma F16BF16 atom natural M),
  BLOCK_N=32 slots per tile, MMA K=64 (the K dim of the score MMA is
  the full_dim=576; we run 9 K-tiles of 64).
- Acc lives in TMEM via tcgen05; score uses TMEM time-shared with
  acc by reading score out to RMEM immediately after each MMA.

Iter 2+ will layer in: 12-warp specialization (tokenspeed pattern),
TMA async loads, 2-CTA cluster + use_2cta_instrs, persistent +
var-split-KV scheduler, skip-correction threshold, fold-Sq.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils


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


class HiggsDense2bitMLADecodeDSL:
    def __init__(
        self,
        block_h: int = 64,
        block_n: int = 32,
        cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        assert block_h in (64, 128), "SM100_MMA_F16BF16 M must be 64 or 128"
        assert block_n % 8 == 0 and 8 <= block_n <= 256
        self.block_h = block_h
        self.block_n = block_n
        self.cluster_shape_mnk = cluster_shape_mnk
        self.use_2cta_instrs = cluster_shape_mnk[0] == 2
        self.threads_per_warp = 32
        self.num_warps = 4
        self.threads_per_cta = self.threads_per_warp * self.num_warps

        # MMA K dim. 576 = 9 * 64. Use K=64 for clean tiling.
        self.mma_k = 64
        self.iterations_qk = FULL_DIM // self.mma_k  # 9
        self.iterations_pv_k = block_n // 16

        self.mma_qk_tiler = (block_h, block_n, self.mma_k)
        # PV N is capped at 256 by SM100_MMA_F16BF16; we split
        # LATENT_DIM=512 into PV_N_CHUNKS chunks of pv_n each.
        self.pv_n = 256
        self.pv_n_chunks = LATENT_DIM // self.pv_n  # 2
        assert LATENT_DIM == self.pv_n * self.pv_n_chunks
        self.mma_pv_tiler = (block_h, self.pv_n, block_n)

    @cute.jit
    def __call__(
        self,
        q_nope: cute.Tensor,
        q_rope: cute.Tensor,
        compressed: cute.Tensor,
        page_table: cute.Tensor,
        out: cute.Tensor,
        codebook: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode

        bf16 = cutlass.BFloat16
        fp32 = cutlass.Float32
        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

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
        topk = page_table.shape[1]
        head_groups = (num_heads + self.block_h - 1) // self.block_h
        grid = (num_rows, head_groups, 1)

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
            # FWHT scratchpad: 128 fp32 = 512 B per CTA, shared across heads.
            fwht_scratch: cute.struct.MemRange[fp32, 128]
            # Acc staging: cast TMEM acc->RMEM->SMEM here for the InvFWHT.
            acc_smem: cute.struct.MemRange[fp32, self.block_h * LATENT_DIM]
            # MMA + TMEM barriers.
            mma_barrier: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        softmax_scale_log2 = softmax_scale * LOG2_E
        self._shared_storage = SharedStorage

        self.kernel(
            qk_tiled_mma, pv_tiled_mma,
            q_nope, q_rope, compressed, page_table, out, codebook,
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
        mCP: cute.Tensor,
        mPT: cute.Tensor,
        mO: cute.Tensor,
        mCB: cute.Tensor,
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
        storage = smem.allocate(self._shared_storage)
        sQ = cute.make_tensor(storage.smem_q.data_ptr(), q_smem_layout)
        sK = cute.make_tensor(storage.smem_k.data_ptr(), k_smem_layout)
        sV = cute.make_tensor(storage.smem_v.data_ptr(), v_smem_layout)
        sP = cute.make_tensor(storage.smem_p.data_ptr(), p_smem_layout)

        # Init codebook + softmax state.
        cb_total = Int32(CODEBOOK_SIZE * PAIR_DIM)
        block_h_dyn = Int32(self.block_h)
        if tid < cb_total:
            storage.smem_codebook[tid] = mCB[tid // PAIR_DIM, tid % PAIR_DIM]
        if tid < block_h_dyn:
            storage.softmax_m[tid] = Float32(-3.4028234663852886e38)
            storage.softmax_l[tid] = Float32(0.0)
        cute.arch.barrier()

        # Step 1: rotate q_nope through FWHT_512 + 1/sqrt(LATENT_DIM)
        # into sQ[h, 0..512]; copy q_rope into sQ[h, 512..576]. We
        # serialize heads (block_h iterations). Each FWHT_512 is 4
        # FWHT_128s + an inter-register FWHT_4.
        num_heads_total = mQN.shape[1]
        for h_local in cutlass.range_constexpr(self.block_h):
            h_global = head_base + h_local
            valid = h_global < num_heads_total
            # Load 4 values per thread from q_nope row.
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
            inv = Float32(INV_SQRT_LATENT)
            v0 = v0 * inv
            v1 = v1 * inv
            v2 = v2 * inv
            v3 = v3 * inv
            sQ[h_local, 0 * 128 + tid] = cutlass.BFloat16(v0)
            sQ[h_local, 1 * 128 + tid] = cutlass.BFloat16(v1)
            sQ[h_local, 2 * 128 + tid] = cutlass.BFloat16(v2)
            sQ[h_local, 3 * 128 + tid] = cutlass.BFloat16(v3)
            if tid < ROPE_DIM:
                rope_val = (
                    mQR[row, h_global, tid] if valid else cutlass.BFloat16(0.0)
                )
                sQ[h_local, LATENT_DIM + tid] = rope_val
            cute.arch.barrier()

        # TMEM allocation. We use TMEM for both score (transient) and
        # acc (persistent). With M=64 atom, lanes 0..63 are populated;
        # 512 cols available per CTA. Layout:
        #   acc TMEM: cols 0..511 holds acc[64, 512] FP32.
        #   score TMEM: cols 0..63 are TIME-SHARED with acc — we
        #     overwrite during score MMA, read out immediately to
        #     RMEM, then the next acc MMA writes those cols back.
        # The acc MMA's accumulate flag is True (we accumulate across
        # slot tiles); the score MMA's flag is False (fresh per tile).
        tmem_alloc_cols = 512
        tcgen05.alloc(storage.tmem_holding_buf, tmem_alloc_cols)
        tcgen05.relinquish_alloc_permit()
        tmem_ptr = storage.tmem_holding_buf

        # Acc fragment in TMEM.
        tAcc_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tAcc = pv_tiled_mma.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(tmem_ptr, tAcc.layout)
        # Initialize acc to 0 via accumulate=False on the first PV MMA;
        # tracked by `first_pv` flag.

        # Score fragment in TMEM (overlaps acc; we read it out before
        # the next acc MMA writes the same region).
        tScore_shape = qk_tiled_mma.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tScore = qk_tiled_mma.make_fragment_C(tScore_shape)
        tScore = cute.make_tensor(tmem_ptr, tScore.layout)

        # MMA A/B fragments.
        tCrQ = qk_tiled_mma.make_fragment_A(sQ)
        tCrK = qk_tiled_mma.make_fragment_B(sK)
        tCrP = pv_tiled_mma.make_fragment_A(sP)
        tCrV = pv_tiled_mma.make_fragment_B(sV)

        topk = mPT.shape[1]
        page_row_offset = row

        # Slot tile loop.
        num_tiles = (topk + self.block_n - 1) // self.block_n
        first_pv = Boolean(True)
        for tile_idx in cutlass.range(num_tiles):
            tile_begin = tile_idx * self.block_n
            tile_count = cutlass.min(self.block_n, topk - tile_begin)

            # Dequant: each warp dequants block_n/4 = 8 slots in
            # parallel. Within a warp, the 32 lanes split the per-slot
            # work: lane t handles 4 latent dims (mirrors the C++
            # higgs_unpack_indices). Iter 2 will switch to TMA-staged
            # packed bytes; for iter 1 we read directly from GMEM per
            # thread.
            self._dequant_tile(
                storage, sK, sV, mCP, mPT,
                page_row_offset, tile_begin, tile_count, tid,
            )
            cute.arch.barrier()

            # Score MMA: tScore = sQ @ sK^T over 9 K-tiles.
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_blk in cutlass.range_constexpr(self.iterations_qk):
                cute.gemm(
                    qk_tiled_mma,
                    tScore,
                    tCrQ[None, None, k_blk],
                    tCrK[None, None, k_blk],
                    tScore,
                )
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_load()

            # Read score TMEM -> RMEM per thread; per-row online softmax;
            # write softmax state + p_smem.
            self._score_softmax_to_p(
                qk_tiled_mma, tScore, sP, storage,
                softmax_scale_log2, tile_count, tid,
            )
            cute.arch.barrier()

            # Acc rescale: read acc TMEM -> RMEM -> scale by alpha -> write back.
            # Skip on first iter (acc is undefined / will be overwritten by
            # accumulate=False on first PV MMA).
            if first_pv:
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            else:
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                self._rescale_acc_by_alpha(
                    pv_tiled_mma, tAcc, storage.softmax_alpha, tid,
                )

            # V MMA: tAcc += sP @ sV over (block_n // 16) K-tiles.
            for k_blk in cutlass.range_constexpr(self.iterations_pv_k):
                cute.gemm(
                    pv_tiled_mma,
                    tAcc,
                    tCrP[None, None, k_blk],
                    tCrV[None, None, k_blk],
                    tAcc,
                )
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_load()
            first_pv = Boolean(False)

        # Epilogue: normalize acc by softmax_l per row, InvFWHT_512,
        # store to out.
        self._epilogue(
            pv_tiled_mma, tAcc, storage, sQ.iterator,
            mO, row, head_base, tid,
        )

        # Release TMEM.
        tcgen05.dealloc(tmem_ptr, tmem_alloc_cols)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @cute.jit
    def _fwht_128(self, val, tid, scratch) -> cutlass.Float32:
        """FWHT_128 over 128 threads, one fp32 per thread. Levels 0..6.
        Uses warp-shuffle for levels 0..4, SMEM for levels 5..6.
        """
        # Levels 0..4: __shfl_xor within warp.
        lane = tid & 31
        for stride in cutlass.range_constexpr(5):
            s = 1 << stride
            other = cute.arch.shuffle_sync_bfly(val, s, mask=0xffffffff)
            val = (other - val) if (lane & s) else (val + other)
        # Level 5: stride 32 cross-warp.
        scratch[tid] = val
        cute.arch.barrier()
        partner = scratch[tid ^ 32]
        val = (partner - val) if (tid & 32) else (val + partner)
        cute.arch.barrier()
        # Level 6: stride 64 cross-warp.
        scratch[tid] = val
        cute.arch.barrier()
        partner = scratch[tid ^ 64]
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
        self, storage, sK, sV, mCP, mPT,
        page_row_offset, tile_begin, tile_count, tid,
    ):
        """Dequant block_n slots: write each slot's latent into sK
        ((slot, dim) row-major) AND sV ((dim, slot) K-major).
        """
        # Each thread handles 4 latent dims for ONE slot at a time.
        # Lane-to-(pair_within_group, coord_lane) mirrors C++
        # higgs_unpack_indices.
        for n in cutlass.range_constexpr(self.block_n):
            col = tile_begin + n
            valid = col < tile_count
            page = Int32(-1)
            if valid:
                page = mPT[page_row_offset, col]
            page_valid = page >= 0
            slot_idx = page if page_valid else Int32(0)
            # higgs_unpack_indices analog:
            pair_within_group = tid >> 1
            coord_lane = tid & 1
            byte_in_group = pair_within_group >> 1
            nibble = pair_within_group & 1
            # Read 4 packed bytes (one per 32-byte group within the
            # 128 packed bytes per slot).
            b0 = Int32(0); b1 = Int32(0); b2 = Int32(0); b3 = Int32(0)
            if coord_lane == 0:
                b0 = mCP[slot_idx, 0, 0 * 32 + byte_in_group].to(Int32)
                b1 = mCP[slot_idx, 0, 1 * 32 + byte_in_group].to(Int32)
                b2 = mCP[slot_idx, 0, 2 * 32 + byte_in_group].to(Int32)
                b3 = mCP[slot_idx, 0, 3 * 32 + byte_in_group].to(Int32)
            # Share with peer lane (XOR 1).
            pb0 = cute.arch.shuffle_sync_bfly(b0, 1)
            pb1 = cute.arch.shuffle_sync_bfly(b1, 1)
            pb2 = cute.arch.shuffle_sync_bfly(b2, 1)
            pb3 = cute.arch.shuffle_sync_bfly(b3, 1)
            b0 = pb0 if coord_lane else b0
            b1 = pb1 if coord_lane else b1
            b2 = pb2 if coord_lane else b2
            b3 = pb3 if coord_lane else b3
            i0 = (b0 >> 4) if nibble else (b0 & 0x0F)
            i1 = (b1 >> 4) if nibble else (b1 & 0x0F)
            i2 = (b2 >> 4) if nibble else (b2 & 0x0F)
            i3 = (b3 >> 4) if nibble else (b3 & 0x0F)
            # Codebook lookup.
            c0 = storage.smem_codebook[i0 * PAIR_DIM + coord_lane]
            c1 = storage.smem_codebook[i1 * PAIR_DIM + coord_lane]
            c2 = storage.smem_codebook[i2 * PAIR_DIM + coord_lane]
            c3 = storage.smem_codebook[i3 * PAIR_DIM + coord_lane]
            # Norm (per-slot FP16 scale at byte offset PACKED_BYTES).
            # In cute DSL we read it as 2 uint8 bytes and reconstruct
            # via bit-cast; for simplicity we read as a half via
            # tensor index and cast. (Iter 2 will batch this load.)
            scale_lo = mCP[slot_idx, 0, PACKED_BYTES].to(Int32)
            scale_hi = mCP[slot_idx, 0, PACKED_BYTES + 1].to(Int32)
            scale_bits = (scale_hi << 8) | scale_lo
            # Half bits -> float (assumes IEEE 754 half).
            scale = cute.arch.half_to_float(cutlass.Uint16(scale_bits))
            if not page_valid:
                scale = Float32(0.0)
            d0 = 0 * 128 + tid
            d1 = 1 * 128 + tid
            d2 = 2 * 128 + tid
            d3 = 3 * 128 + tid
            v0 = cutlass.BFloat16(scale * c0)
            v1 = cutlass.BFloat16(scale * c1)
            v2 = cutlass.BFloat16(scale * c2)
            v3 = cutlass.BFloat16(scale * c3)
            sK[n, d0] = v0
            sK[n, d1] = v1
            sK[n, d2] = v2
            sK[n, d3] = v3
            sV[d0, n] = v0
            sV[d1, n] = v1
            sV[d2, n] = v2
            sV[d3, n] = v3
            if tid < ROPE_DIM:
                # Rope is BF16 (2 bytes per element) at byte offset
                # PACKED_BYTES + NORM_BYTES.
                rope_lo = mCP[slot_idx, 0, PACKED_BYTES + NORM_BYTES + tid * 2].to(Int32)
                rope_hi = mCP[slot_idx, 0, PACKED_BYTES + NORM_BYTES + tid * 2 + 1].to(Int32)
                rope_bits = (rope_hi << 8) | rope_lo
                rope_val = cute.arch.bits_to_bf16(cutlass.Uint16(rope_bits))
                if not page_valid:
                    rope_val = cutlass.BFloat16(0.0)
                sK[n, LATENT_DIM + tid] = rope_val

    @cute.jit
    def _score_softmax_to_p(
        self, qk_tiled_mma, tScore, sP, storage,
        softmax_scale_log2, tile_count, tid,
    ):
        """Read score TMEM -> RMEM per thread; per-row online softmax;
        write softmax_m/l/alpha to SMEM; cast p to BF16 in sP.
        """
        # tcgen05.ld with a suitable atom for M=64. We let CuTe pick.
        tiled_t2r = tcgen05.make_tmem_copy(
            tcgen05.copy.Ld32x32bAtom(),  # placeholder atom — may need fix
            tScore,
        )
        thr_t2r = tiled_t2r.get_slice(tid)
        tDtScore = thr_t2r.partition_S(tScore)
        tDrScore = cute.make_fragment_like(tDtScore)
        cute.copy(tiled_t2r, tDtScore, tDrScore)
        # Per-row max+sum then BF16 store to sP — STUB:
        # the per-warp partition + cross-thread reductions are written
        # in iter 2. For now we scatter via coord iter to sP and let
        # iter 2 fold in the proper warp-level softmax.
        cute.arch.barrier()

    @cute.jit
    def _rescale_acc_by_alpha(
        self, pv_tiled_mma, tAcc, alpha_smem, tid,
    ):
        """Read acc TMEM -> RMEM per thread, multiply by alpha (looked
        up per Q row), write back to TMEM. STUB: implemented in iter 2.
        """
        pass

    @cute.jit
    def _epilogue(
        self, pv_tiled_mma, tAcc, storage, dummy_ptr,
        mO, row, head_base, tid,
    ):
        """Normalize acc by softmax_l per Q row, apply InvFWHT_512 per
        head, store BF16 to mO. STUB: implemented in iter 2.
        """
        pass


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
    use_2cta: bool = False,
) -> None:
    import torch

    assert q_nope.is_cuda
    assert q_nope.dtype == torch.bfloat16
    assert codebook.dtype == torch.float32

    cluster_mnk = (2 if use_2cta else 1, 1, 1)
    kernel = HiggsDense2bitMLADecodeDSL(
        block_h=block_h, block_n=block_n, cluster_shape_mnk=cluster_mnk
    )

    mQN = from_dlpack(q_nope.contiguous(), assumed_align=16)
    mQR = from_dlpack(q_rope.contiguous(), assumed_align=16)
    mCP = from_dlpack(compressed.contiguous(), assumed_align=16)
    mPT = from_dlpack(page_table.contiguous(), assumed_align=16)
    mO = from_dlpack(out, assumed_align=16)
    mCB = from_dlpack(codebook.contiguous(), assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(mQN, mQR, mCP, mPT, mO, mCB, sm_scale, stream)
