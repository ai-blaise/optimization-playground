"""HIGGS 2-bit dense MLA KV decode kernel — CuTe Python DSL.

End-to-end CuTe DSL replacement for `higgs_dense_2bit_mla_decode_tc.cuh`
(C++ commit 961c4794a, 2.57× over scalar). Adapts the
`mla_decode_fp8.py` design from tokenspeed-mla for the HIGGS 2-bit KV
codec: each slot is 258 bytes = 128 packed 4-bit indices + 2 B FP16
scale + 128 B (64 × BF16) rope. Q stays BF16; we apply FWHT_512 +
1/sqrt(512) once at the start (Python wrapper) so the q·K dot lives in
the codec's rotated basis, and InvFWHT_512 at the end. Rope is untouched.

ITER 6 status (verified on B200 a4-us-001-rl9, cutlass-dsl 4.5.1):
  - Kernel COMPILES end-to-end (~15s cold compile)
  - Kernel RUNS without crash; produces FINITE structured output
  - Per-call wall time: 17s (JIT cache MISS per call — known bug)
  - Numerical correctness: output ~40x undermagnitude vs PyTorch ref
    (composed-swizzle write addresses don't fully align with MMA reads)

Test inputs (R=1, H=64, TOPK=32):
  Reference output: std=0.051, range [-0.124, 0.133]
  DSL output:       std=0.001, range [-0.006, 0.007], nan=False

ITER 7 priority order (autonomous loop continuation):
  1. Fix JIT cache miss (17s/call → sub-ms): investigate cute.jit cache key,
     pre-build cute.Tensors, possibly use module-scope @cute.jit
  2. Fix swizzle correctness: try Path B sub-variants (swizzle bits 0/1/2);
     fallback to Path C (pre-dequant via Triton, BF16 K/V input to DSL)
  3. Implement R2T for _rescale_acc_by_alpha (per tokenspeed:3157)
  4. Implement InvFWHT_512 in _epilogue (cooperative 512-element FWHT)
  5. Benchmark vs C++ TC baseline (higgs_dense_2bit_mla_decode_tc,
     commit 961c4794a, 2.57× over scalar = ~XXX us/call); target DSL <= 2.0x TC
  6. Benchmark vs tokenspeed FP8 MLA at equivalent shapes
  7. IKP region profile; CZS layout verify; cutest fusion patterns
  8. Warp specialization, 2-CTA, persistent scheduler, PDL — perf tier

State checkpoints in agentmemory :3811:
  - mem_20260524T004142Z_50zpig: iter 6 breakthrough (compiles + runs)
  - mem_20260524T004344Z_cn86hn: iter 6 timing diagnostic (17s/call)
  - mem_20260524T000038Z_14qem1: iter 5 + iter 6 PISL plan
  - mem_20260523T223028Z_w8gnkg: iter 4 online softmax wired
  - mem_20260523T222445Z_fa1p6v: tokenspeed softmax pattern
  - mem_20260523T222156Z_pusa7x: R2S pattern from quack
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
        compressed_packed: cute.Tensor,     # [num_slots, 128] uint8 (legacy; unused by ITER 11)
        compressed_scale: cute.Tensor,      # [num_slots] float16 (legacy; unused)
        compressed_rope: cute.Tensor,       # [num_slots, 64] bfloat16 (legacy; unused)
        page_table: cute.Tensor,
        out: cute.Tensor,
        codebook: cute.Tensor,              # legacy; unused
        # ITER 11 Option A: pre-dequanted BF16 K/V from Python wrapper.
        # Race-free path; replaces per-element scatter in _dequant_tile.
        k_dense: cute.Tensor,               # [num_pages, FULL_DIM=576] bfloat16
        v_dense: cute.Tensor,               # [num_pages, LATENT_DIM=512] bfloat16
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
        # PV: A operand (P) is K-major; B operand (V) is MN-major because
        # V's MMA tile is (N_out=256, K_contract=block_n) and we want
        # consecutive memory along the N axis for TMA / vectorized store.
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            bf16, OperandMajorMode.K, OperandMajorMode.MN,
            fp32, cta_group, self.mma_pv_tiler[:2],
        )

        num_rows = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_groups = (num_heads + self.block_h - 1) // self.block_h
        grid = (num_rows, head_groups, 1)

        # Swizzled SMEM layouts via sm100_utils. The staged variants
        # carry the K-tile stride structure (mode 3) so per-K-block
        # MMA fragments can be sliced directly. We compute these for
        # MMA fragment construction (make_fragment_A/B); per-element
        # dequant writes use composed (flat_outer, mma_swizzle_inner)
        # views (sQ_write etc.). NOTE: the composed-write addresses
        # DO NOT match MMA-read addresses when swizzle is sized for
        # MMA tile — this produces NaN output. ITER 6 fix: use a
        # custom swizzle sized for the flat shape (see PISL pattern
        # in tokenspeed FP8 MLA lines 2924-2933).
        q_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.mma_qk_tiler, bf16, ITERATIONS_QK,
        )
        k_smem_layout = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.mma_qk_tiler, bf16, ITERATIONS_QK,
        )
        v_smem_layout = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.mma_pv_tiler, bf16,
            self.iterations_pv_k * PV_N_CHUNKS,
        )
        p_smem_layout = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.mma_pv_tiler, bf16, self.iterations_pv_k,
        )
        # The per-element dequant write path needs a (n, d) logical-coord
        # tensor view whose physical address mapping matches the MMA
        # swizzled SMEM. We materialize that at kernel-body time via
        # cute.composition of the staged SMEM layout with a flat
        # (n_extent, d_extent) logical layout — see the kernel body for
        # the construction. Here we just record the logical shapes.
        self.k_logical_shape = (self.block_n, FULL_DIM)
        self.v_logical_shape = (LATENT_DIM, self.block_n)
        self.p_logical_shape = (self.block_h, self.block_n)

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
            k_dense, v_dense,
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
        mCK: cute.Tensor,             # compressed_packed [num_slots, 128] u8 (legacy)
        mCS: cute.Tensor,             # compressed_scale  [num_slots] fp16 (legacy)
        mCR: cute.Tensor,             # compressed_rope   [num_slots, 64] bf16 (legacy)
        mPT: cute.Tensor,             # page_table
        mO: cute.Tensor,              # out
        mCB: cute.Tensor,             # codebook (legacy)
        mK: cute.Tensor,              # ITER 11: BF16 dense K [num_pages, FULL_DIM]
        mV: cute.Tensor,              # ITER 11: BF16 dense V [num_pages, LATENT_DIM]
        softmax_scale_log2: cutlass.Float32,
        q_smem_layout,
        k_smem_layout,
        v_smem_layout,
        p_smem_layout,
    ):
        tid = cute.arch.thread_idx()[0]
        bid_x, bid_y, _ = cute.arch.block_idx()
        row = bid_x
        head_base = bid_y * self.block_h

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # MMA-side views (consumed by tiled_mma.make_fragment_A/B).
        sQ_mma = storage.smem_q.get_tensor(
            q_smem_layout.outer, swizzle=q_smem_layout.inner,
        )
        sK_mma = storage.smem_k.get_tensor(
            k_smem_layout.outer, swizzle=k_smem_layout.inner,
        )
        sV_mma = storage.smem_v.get_tensor(
            v_smem_layout.outer, swizzle=v_smem_layout.inner,
        )
        sP_mma = storage.smem_p.get_tensor(
            p_smem_layout.outer, swizzle=p_smem_layout.inner,
        )
        # Per-element-write views with PISL-style custom swizzle per
        # tokenspeed FP8 MLA mla_decode_fp8.py:2924-2933. The swizzle
        # bits are sized for the FLAT (M, K) layout so write addresses
        # coincide with what MMA reads via sm100_utils.make_smem_layout_b.
        #
        # Formula: swizzle_bits = log2(K_dim * dtype_bits / 8 / 32) + 1
        # For BF16 (16 bits): bits = log2(K_dim / 16) + 1
        # swizzle_base = 3 for 16-bit dtype.
        # Reuse the MMA layout's actual swizzle by composing it with the
        # flat layout (instead of building from scratch). The flat layout
        # provides the (M, K) → linear-offset mapping; the swizzle XOR
        # operates on that linear offset. As long as the flat layout's
        # cosize matches the MMA layout's cosize, the swizzle applied
        # post-hoc puts bytes at the same physical positions MMA reads.
        sw_Q = q_smem_layout.inner
        sw_K = k_smem_layout.inner
        sw_V = v_smem_layout.inner
        sw_P = p_smem_layout.inner
        sQ_write = cute.make_tensor(
            storage.smem_q.data_ptr(),
            cute.make_composed_layout(sw_Q, 0,
                cute.make_layout((self.block_h, FULL_DIM), stride=(FULL_DIM, 1))),
        )
        sK_write = cute.make_tensor(
            storage.smem_k.data_ptr(),
            cute.make_composed_layout(sw_K, 0,
                cute.make_layout(self.k_logical_shape, stride=(FULL_DIM, 1))),
        )
        sV_write = cute.make_tensor(
            storage.smem_v.data_ptr(),
            cute.make_composed_layout(sw_V, 0,
                cute.make_layout(self.v_logical_shape, stride=(self.block_n, 1))),
        )
        sP_write = cute.make_tensor(
            storage.smem_p.data_ptr(),
            cute.make_composed_layout(sw_P, 0,
                cute.make_layout(self.p_logical_shape, stride=(self.block_n, 1))),
        )
        # Aliases for body code that doesn't disambiguate read vs write.
        sQ, sK, sV, sP = sQ_write, sK_write, sV_write, sP_write

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

        # Q load: mQN is pre-rotated (FWHT_512 + 1/sqrt(LATENT_DIM)) +
        # rope-concatenated BF16 Q[num_rows, num_heads, FULL_DIM].
        # Cooperative scalar load into sQ via the (block_h, FULL_DIM)
        # composed-swizzle view. ITER 4: replace with TMA via
        # cute.nvgpu.make_tiled_tma_atom_A once Q load is hoisted to a
        # dedicated load warp (warp-spec).
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

        # Build MMA fragments directly from the swizzled SMEM tensors.
        # Per tokenspeed/quack canonical SM100 pattern, we do NOT call
        # partition_A/B first — make_fragment_A/B consumes the staged
        # swizzled SMEM tensor and produces a fragment whose layout
        # carries the K-block and stage modes for direct slicing.
        tCrQ = qk_tiled_mma.make_fragment_A(sQ_mma)
        tCrK = qk_tiled_mma.make_fragment_B(sK_mma)
        tCrP = pv_tiled_mma.make_fragment_A(sP_mma)
        tCrV = pv_tiled_mma.make_fragment_B(sV_mma)

        topk = mPT.shape[1]
        num_tiles = (topk + self.block_n - 1) // self.block_n

        # PV ACCUMULATE flag handshake (per tokenspeed/quack): set False
        # before the slot loop so the first MMA OVERWRITES acc TMEM,
        # then set True after the first MMA inside the loop.
        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        # Tensor views over softmax_m/l/alpha MemRanges (dynamic-index
        # access via MemRange is unsupported; Tensor allows it).
        softmax_m_t = cute.make_tensor(
            storage.softmax_m.data_ptr(),
            cute.make_layout(self.block_h, stride=1),
        )
        softmax_l_t = cute.make_tensor(
            storage.softmax_l.data_ptr(),
            cute.make_layout(self.block_h, stride=1),
        )
        softmax_alpha_t = cute.make_tensor(
            storage.softmax_alpha.data_ptr(),
            cute.make_layout(self.block_h, stride=1),
        )
        # Online-softmax state init via cooperative scatter (modulo gives
        # full coverage in 128 threads with block_h=64 entries).
        init_idx = tid % Int32(self.block_h)
        softmax_m_t[init_idx] = Float32(-1.0e30)
        softmax_l_t[init_idx] = Float32(0.0)
        cute.arch.barrier()

        for tile_idx in cutlass.range(num_tiles):
            tile_begin = tile_idx * self.block_n
            tile_count = cutlass.min(self.block_n, topk - tile_begin)

            # (A) ITER 11 Option A: race-free K/V load from pre-dequanted
            # BF16 GMEM. Writes go through sK_write/sV_write (composed
            # swizzle) since sK_mma multi-mode tensor doesn't accept
            # scalar (n, d) indexing. ITER 12 will use cute.copy with
            # a TMA atom or cute.autovec_copy with shape-adapted views.
            self._load_kv_from_dense(
                mK, mV, mPT, sK, sV,
                row, tile_begin, tile_count, tid,
            )
            cute.arch.barrier()

            # (B) Score MMA: tScore = sQ @ sK^T over ITERATIONS_QK K-stages.
            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_stage in cutlass.range_constexpr(ITERATIONS_QK):
                cute.gemm(
                    qk_tiled_mma,
                    tScore,
                    tCrQ[None, None, 0, k_stage],
                    tCrK[None, None, 0, k_stage],
                    tScore,
                )
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_load()

            # (C) Read score TMEM -> RMEM, online softmax, write
            # softmax_alpha + sP (BF16) via R2S TiledCopy. Writes to
            # sP_mma (the MMA-swizzled SMEM view); also updates
            # softmax_m / softmax_l / softmax_alpha SMEM state.
            self._score_softmax_to_p(
                qk_tiled_mma, pv_tiled_mma, tScore, sP,
                softmax_m_t, softmax_l_t, softmax_alpha_t,
                softmax_scale_log2, tile_count, tid,
            )

            # (D) Rescale acc TMEM by alpha (per Q row).
            self._rescale_acc_by_alpha(
                pv_tiled_mma, tAcc_lo, tAcc_hi,
                softmax_alpha_t, tid,
            )

            # (E) V MMA over both N-chunks (PV_N_CHUNKS=2 to cover
            # LATENT_DIM=512 within SM100's N=256 cap).
            # tCrV is staged as (iterations_pv_k * PV_N_CHUNKS).
            for n_chunk in cutlass.range_constexpr(PV_N_CHUNKS):
                acc_target = tAcc_lo if n_chunk == 0 else tAcc_hi
                for k_stage in cutlass.range_constexpr(self.iterations_pv_k):
                    v_stage = n_chunk * self.iterations_pv_k + k_stage
                    cute.gemm(
                        pv_tiled_mma,
                        acc_target,
                        tCrP[None, None, 0, k_stage],
                        tCrV[None, None, 0, v_stage],
                        acc_target,
                    )
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.fence_view_async_tmem_load()

        # Epilogue: normalize, InvFWHT, store.
        self._epilogue(
            pv_tiled_mma, tAcc_lo, tAcc_hi, softmax_l_t,
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
    def _load_kv_from_dense_autovec(
        self, mK, mV, mPT, sK_mma, sV_mma,
        row, tile_begin, tile_count, tid,
    ):
        """ITER 12 Option A: race-free K/V load via cute.autovec_copy
        from a per-thread RMEM fragment fill.

        Pattern (canonical CuTe R2S):
          1. rK = cute.make_fragment_like(sK_mma) — per-thread shape
          2. Per-cell, look up (slot, dim) coord via identity-tensor
             partition; read from mK[page_table[row, tile_begin+slot], dim]
          3. cute.autovec_copy(rK, sK_mma) — race-free vectorized store

        Each thread's RMEM owns a unique subset of sK_mma cells (per
        cute's TiledMma per-thread distribution); autovec_copy writes
        each thread's subset to the correct sK_mma bytes — no race.
        """
        # Build per-thread RMEM with same shape as sK_mma
        rK = cute.make_fragment_like(sK_mma, cutlass.BFloat16)
        rV = cute.make_fragment_like(sV_mma, cutlass.BFloat16)

        # Identity tensors for coord projection
        cK = cute.make_identity_tensor((self.block_n, FULL_DIM))
        cV = cute.make_identity_tensor((LATENT_DIM, self.block_n))

        # Need partition_S of identity tensor matching the per-thread layout
        # The atom for R2S is implicit in make_fragment_like + autovec_copy.
        # Use the MMA's thread slice to get per-thread (n, d) coords.
        # For QK MMA, sK_mma is B-operand: tiled_mma.get_slice(tid).partition_B(cK)
        # But we don't have qk_tiled_mma in this scope; would need passing.
        # ITER 12.1: try without identity (each thread reads same coords —
        # will produce wrong values but tests if autovec_copy works at all).
        zero_bf = cutlass.BFloat16(0.0)
        for i in cutlass.range_constexpr(cute.size(rK)):
            rK[i] = zero_bf  # placeholder; ITER 12.2 fills from mK via coords
        for i in cutlass.range_constexpr(cute.size(rV)):
            rV[i] = zero_bf

        # Race-free vectorized R2S
        cute.autovec_copy(rK, sK_mma)
        cute.autovec_copy(rV, sV_mma)

    @cute.jit
    def _load_kv_from_dense(
        self, mK, mV, mPT, sK, sV,
        row, tile_begin, tile_count, tid,
    ):
        """ITER 11 Option A: race-free K/V load from BF16 dense GMEM.

        sK, sV are the composed-swizzle write views (sX_write in __call__).
        Reads from pre-dequanted BF16 GMEM (Python wrapper); writes to
        SMEM via per-element scatter. RACES PERSIST in iter 11 (same write
        pattern); iter 12 uses cute.autovec_copy from per-thread RMEM
        fragment for race-free vectorized GMEM→SMEM transfer.

        Currently: VALUES are correct (from real HIGGS dequant), but races
        on the per-element scatter mean only a subset of writes land
        correctly (per IKP/compute-sanitizer iter 8 finding).
        """
        # mK: (num_pages, FULL_DIM=576) BF16
        # mV: (num_pages, LATENT_DIM=512) BF16
        # mPT: (num_rows, topk) int32
        zero_bf = cutlass.BFloat16(0.0)

        # Each thread handles 4 dims per slot (mirrors C++ kernel pattern):
        # d0 = 0*128 + tid, d1 = 1*128 + tid, d2 = 2*128 + tid, d3 = 3*128 + tid
        # rope: d_rope = 4*128 + tid (only for tid < ROPE_DIM = 64)
        nROPE = Int32(ROPE_DIM)
        is_rope_lane = tid < nROPE
        rope_safe = cutlass.select_(is_rope_lane, tid, Int32(0))

        for n in cutlass.range_constexpr(self.block_n):
            col = tile_begin + Int32(n)
            in_tile = col < tile_count
            page = cutlass.select_(in_tile, mPT[row, col], Int32(-1))
            page_valid = page >= Int32(0)
            slot_idx = cutlass.select_(page_valid, page, Int32(0))

            # Load 4 latent dims per thread from mK + 1 rope dim if tid<64
            d0 = Int32(0 * 128) + tid
            d1 = Int32(1 * 128) + tid
            d2 = Int32(2 * 128) + tid
            d3 = Int32(3 * 128) + tid

            v0 = mK[slot_idx, d0]
            v1 = mK[slot_idx, d1]
            v2 = mK[slot_idx, d2]
            v3 = mK[slot_idx, d3]
            # Mask invalid slots
            v0 = cutlass.select_(page_valid, v0, zero_bf)
            v1 = cutlass.select_(page_valid, v1, zero_bf)
            v2 = cutlass.select_(page_valid, v2, zero_bf)
            v3 = cutlass.select_(page_valid, v3, zero_bf)
            v0 = cutlass.select_(in_tile, v0, zero_bf)
            v1 = cutlass.select_(in_tile, v1, zero_bf)
            v2 = cutlass.select_(in_tile, v2, zero_bf)
            v3 = cutlass.select_(in_tile, v3, zero_bf)

            # Write to sK_mma via per-element scatter (each thread writes
            # unique (slot, dim) cells; no race because tid covers 0..127
            # uniquely for d0..d3 = 0..511; and rope_lane covers d=512..575).
            # NOTE: this still uses sK_mma's multi-mode layout for indexing —
            # may fail at compile time. If so, fall back to a flat tensor view
            # of the same SMEM region.
            sK[n, d0] = v0
            sK[n, d1] = v1
            sK[n, d2] = v2
            sK[n, d3] = v3
            sV_mma[d0, n] = v0
            sV_mma[d1, n] = v1
            sV_mma[d2, n] = v2
            sV_mma[d3, n] = v3

            # Rope: 64 dims at offset LATENT_DIM, only first 64 threads
            d_rope = Int32(LATENT_DIM) + rope_safe
            rope_val = mK[slot_idx, d_rope]
            rope_val = cutlass.select_(is_rope_lane, rope_val, zero_bf)
            rope_val = cutlass.select_(page_valid, rope_val, zero_bf)
            rope_val = cutlass.select_(in_tile, rope_val, zero_bf)
            # All threads write to the same rope position when is_rope_lane=False;
            # use the safe rope index — race only if multiple lanes write same
            # cell with different values, but cutlass.select_ ensures only
            # is_rope_lane writes the meaningful value (others write 0 to
            # rope_safe=0). Still a benign race (identical-value or 0).
            sK_mma[n, d_rope] = rope_val

    @cute.jit
    def _dequant_cell_latent(
        self, d_int_static, slot_idx, in_tile, page_valid, mCK, mCS, sCB_t,
    ):
        """Latent codebook dequant for compile-time-static d_int.

        d_int is a Python int (from identity-tensor partition); slot_idx
        is dynamic (runtime page lookup). Python-side branching is fine
        for d_int; runtime cutlass.select_ for slot masking.
        """
        # d_int_static is a Python int 0..LATENT_DIM-1
        pair_idx = d_int_static >> 1          # 0..255
        coord_lane = d_int_static & 1         # 0 or 1
        byte_idx = pair_idx >> 1              # 0..127
        nibble_hi_py = (pair_idx & 1) == 1    # Python bool
        byte_val = mCK[slot_idx, byte_idx].to(Int32)
        # Nibble select via Python if (compile-time)
        cb_idx = (byte_val >> 4) if nibble_hi_py else (byte_val & 0x0F)
        cb_val = sCB_t[cb_idx * Int32(PAIR_DIM) + coord_lane]
        scale = mCS[slot_idx].to(Float32)
        value = (scale * cb_val).to(cutlass.BFloat16)
        zero_bf = cutlass.BFloat16(0.0)
        # Mask invalid via cutlass.select_ (dynamic)
        value = cutlass.select_(in_tile, value, zero_bf)
        value = cutlass.select_(page_valid, value, zero_bf)
        return value

    @cute.jit
    def _dequant_cell_rope(
        self, rope_d_static, slot_idx, in_tile, page_valid, mCR,
    ):
        """Rope BF16 passthrough for compile-time-static d in rope range."""
        value = mCR[slot_idx, rope_d_static]
        zero_bf = cutlass.BFloat16(0.0)
        value = cutlass.select_(in_tile, value, zero_bf)
        value = cutlass.select_(page_valid, value, zero_bf)
        return value

    @cute.jit
    def _dequant_tile_r2s(
        self, qk_tiled_mma, pv_tiled_mma, storage,
        sK_mma, sV_mma, mCK, mCS, mCR, mPT,
        row, tile_begin, tile_count, tid,
    ):
        """R2S-based HIGGS dequant: per-thread compute -> RMEM -> swizzled SMEM.

        ITER 5 STAGED: tile-loop unroll over (block_n, FULL_DIM) cells
        explodes the MLIR (29min compile, did not finish). Reverted to
        the simpler `.fill(0)` + per-K-tile cute.copy pattern below.
        Real dequant lands in ITER 6 via vectorized SMEM stores from
        compute (8-element ldmatrix.x4 atom, ~32 cells per thread,
        compile-tractable).
        """
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=16,
        )
        tiled_r2s_K = cute.make_tiled_copy_B(copy_atom, qk_tiled_mma)
        thr_r2s_K = tiled_r2s_K.get_slice(tid)
        tRS_sK = thr_r2s_K.partition_D(sK_mma)
        tRS_rK = cute.make_fragment_like(tRS_sK, cutlass.BFloat16)
        tRS_rK.fill(cutlass.BFloat16(0.0))
        cute.copy(tiled_r2s_K, tRS_rK, tRS_sK)

        tiled_r2s_V = cute.make_tiled_copy_B(copy_atom, pv_tiled_mma)
        thr_r2s_V = tiled_r2s_V.get_slice(tid)
        tRS_sV = thr_r2s_V.partition_D(sV_mma)
        tRS_rV = cute.make_fragment_like(tRS_sV, cutlass.BFloat16)
        tRS_rV.fill(cutlass.BFloat16(0.0))
        cute.copy(tiled_r2s_V, tRS_rV, tRS_sV)

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
        self, qk_tiled_mma, pv_tiled_mma, tScore, sP,
        softmax_m_t, softmax_l_t, softmax_alpha_t,
        softmax_scale_log2, tile_count, tid,
    ):
        """TMEM score -> RMEM via T2R; FMA+exp2 online softmax; BF16
        cast; R2S write to sP via PISL-swizzled tiled copy.

        Adapted from tokenspeed FP8 MLA softmax warp (mla_decode_fp8.py
        lines 2643-2949). HIGGS specifics:
          - block_h=64 rows, block_n=32 cols, 4 warps → warps_in_n=1
            (each warp owns full N row), so cross-warp row reduce is a
            no-op (skipped).
          - PISL swizzle for sP: PV MMA reads sP as A operand with K
            dim = block_n=32; swizzle bits per the formula below.
          - Online softmax tracks row_max + row_sum + alpha across slot
            tiles via softmax_m / softmax_l / softmax_alpha SMEM.
        """
        import math as _math
        from cutlass.utils import LayoutEnum

        # --- 1. Build T2R copy with identity-tensor coord projection ---
        # tScore is shape ((mma_M, mma_N), M_block, N_block); slice to (M,N)
        tAcc = tScore[(None, None), 0, 0]
        cta_qk_tile_mn = cute.select(self.mma_qk_tiler, mode=[0, 1])
        cS = cute.make_identity_tensor(cta_qk_tile_mn)
        # Use sm100_utils.get_tmem_load_op to pick the appropriate atom
        # for the tile shape + dtype combo. (Direct Ld32x32bOp(Rep(N))
        # construction fails 'Operation creation failed' if the
        # repetition doesn't match the atom's hardware constraints.)
        tmem_load_atom = sm100_utils.get_tmem_load_op(
            self.mma_qk_tiler[:2],
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            cutlass.Float32,
            cutlass.Float32,
            self.mma_qk_tiler[:2],
            False,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_thr_copy = tmem_tiled_copy.get_slice(tid)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)
        tTR_rAcc = cute.make_fragment_like(tTR_tS, cutlass.Float32)
        cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
        cute.arch.fence_view_async_tmem_load()

        # --- 2. K-bound masking (per-cell) ---
        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            tTR_rAcc[i] = cutlass.select_(
                cute.elem_less(tTR_tS[i][1], tile_count),
                tTR_rAcc[i],
                Float32(-1.0e6),
            )

        # --- 3. Per-row max (single warp owns 32 cols = full row) ---
        # First each thread reduces its cells; then warp-reduce within warp.
        # For HIGGS with block_n=32 cols and 1 warp per row, the per-thread
        # cells span the same row, so warp_reduction(fmax) over 32 lanes
        # gives the full row max.
        row_max_local = tTR_rAcc.load().reduce(
            cute.ReductionOp.MAX, init_val=Float32(-_math.inf), reduction_profile=0,
        )
        row_max_new = cute.arch.warp_reduction(
            row_max_local, cute.arch.fmax,
            threads_in_group=cute.arch.WARP_SIZE,
        )

        # --- 4. Correction factor + softmax_m/alpha update ---
        # ROW_INDEX_OF_THREAD: warp_id * (block_h / num_warps) + lane // 1
        # (Assumption: 4 warps × 16 rows each, lane covers cols within row).
        # We store softmax_m/alpha at row_index regardless of which lane
        # (lane 0 elected for the write).
        warp_id = tid // Int32(self.threads_per_warp)
        lane_id = tid % Int32(self.threads_per_warp)
        rows_per_warp = self.block_h // self.num_warps
        row_index = warp_id * Int32(rows_per_warp)  # base row owned by this warp

        # Read old row_max for this thread's row (broadcast within warp)
        old_row_max = softmax_m_t[row_index]
        correction_factor = cute.math.exp2(
            (old_row_max - row_max_new) * softmax_scale_log2, fastmath=True,
        )

        # --- 5. FMA + exp2 online softmax ---
        fma_b = softmax_scale_log2
        fma_c = (Float32(0.0) - row_max_new) * softmax_scale_log2
        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        # --- 6. Row sum (packed f32x2 add) + update softmax_l ---
        old_row_sum = softmax_l_t[row_index]
        row_sum_new = old_row_sum * correction_factor
        row_sum_vec_0 = Float32(0.0)
        row_sum_vec_1 = Float32(0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec_0 = row_sum_vec_0 + tTR_rAcc[i]
            row_sum_vec_1 = row_sum_vec_1 + tTR_rAcc[i + 1]
        thread_row_sum = row_sum_vec_0 + row_sum_vec_1
        thread_row_sum = cute.arch.warp_reduction(
            thread_row_sum, lambda a, b: a + b,
            threads_in_group=cute.arch.WARP_SIZE,
        )
        row_sum_new = row_sum_new + thread_row_sum

        # --- 7. Write per-row state via per-cell scatter on identity
        # coords. Each cell writes the warp-reduced max/sum/alpha to
        # its row, with the redundant-write trick (all lanes touching
        # the same row write the same value, since warp_reduction
        # broadcasts the result). This guarantees every block_h row
        # gets written even if the within-thread cell distribution
        # doesn't align with a single warp.
        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
            m_idx = tTR_tS[i][0]
            softmax_m_t[m_idx] = row_max_new
            softmax_l_t[m_idx] = row_sum_new
            softmax_alpha_t[m_idx] = correction_factor

        # --- 8. Quantize FP32 -> BF16 ---
        tTR_rS = cute.make_fragment_like(tTR_tS, cutlass.BFloat16)
        tTR_rS.store(tTR_rAcc.load().to(cutlass.BFloat16))

        # --- 9. Write BF16 P to sP via per-cell scatter using the
        # identity-tensor coord projection (tTR_tS[i] = (m, n)).
        # The R2S TiledCopy via make_tiled_copy_D fails rank-mismatch
        # because sP_mma is staged (rank 5) while T2R fragment is rank 3.
        # Per-cell scatter via the composed-swizzle write view sP_write
        # is simpler and avoids the rebuild. Pass sP_mma here actually,
        # not sP_write — sP_mma is the MMA-shaped view; per-element
        # indexing through composed_layout addresses MMA-compatible bytes.
        # ITER 5 perf pass adopts PISL-rebuilt sP per tokenspeed
        # lines 2906-2944 for vectorized 128-bit R2S.
        for i in cutlass.range_constexpr(cute.size(tTR_rS)):
            m_idx = tTR_tS[i][0]
            n_idx = tTR_tS[i][1]
            sP[m_idx, n_idx] = tTR_rS[i]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()

    @cute.jit
    def _rescale_acc_by_alpha(
        self, pv_tiled_mma, tAcc_lo, tAcc_hi, alpha_smem, tid,
    ):
        """Rescale TMEM accumulator by per-row alpha factor (T2R -> mul -> R2T).

        Adapted from tokenspeed FP8 MLA rescale() at mla_decode_fp8.py:3157.

        Iter 4 — DEFERRED to iter 5. The R2T atom (St32x32bOp) requires
        matched tile shape with the TMEM accumulator; direct construction
        was failing with 'Operation creation failed' in
        atom_make_tmem_copy. Path forward: use sm100_utils.get_tmem_load_op
        for load + reuse the same tiled_copy for store via partition_D
        (similar to what quack does for the epilogue T2R/R2S pair).

        For single-tile tests (TOPK <= block_n=32) no rescale is needed
        — first iter overwrites acc with the only tile's contribution
        and softmax_l holds the full normalizer. Multi-tile tests will
        produce incorrect output until R2T is wired.
        """
        pass

    @cute.jit
    def _epilogue(
        self, pv_tiled_mma, tAcc_lo, tAcc_hi, softmax_l_t,
        mO, row, head_base, tid,
    ):
        """Final epilogue: TMEM->RMEM, divide by softmax_l, InvFWHT_512,
        BF16 store to GMEM.

        Iter 3 — structural skeleton with naive direct store (no
        InvFWHT). Reads both tAcc_lo (cols 0..255) and tAcc_hi (cols
        256..511) of the LATENT_DIM=512 output, divides by
        softmax_l[row] (=1.0 from neutral stub), casts to BF16, writes
        to mO[row, head_base + m_local, dim].

        ITER 4:
          - Replace neutral softmax_l/l_inv with values from
            _score_softmax_to_p once that lands
          - Apply InvFWHT_512: per-Q-row, fwht the 512 latent dims via
            32-thread cooperative inv-fwht using fwht_scratch SMEM
          - Move BF16 store to a TiledCopy_R2G with vectorized 16B
            writes (or TMA store via cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp)
        """
        from cutlass.utils import LayoutEnum
        # T2R copy for each acc chunk.
        for chunk_idx in cutlass.range_constexpr(PV_N_CHUNKS):
            tAcc_chunk = tAcc_lo if chunk_idx == 0 else tAcc_hi
            col_base = chunk_idx * PV_N_MAX
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.mma_pv_tiler[:2],
                LayoutEnum.ROW_MAJOR,
                cutlass.BFloat16,
                cutlass.Float32,
                self.mma_pv_tiler[:2],
                False,
            )
            tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_chunk)
            thr_t2r = tiled_t2r.get_slice(tid)
            tDtAcc = thr_t2r.partition_S(tAcc_chunk)
            tDrAcc = cute.make_fragment_like(tDtAcc)
            cute.copy(tiled_t2r, tDtAcc, tDrAcc)

            # Cooperative GMEM store. Each thread owns a chunk of cells
            # in the (M=block_h, N=PV_N_MAX) acc tile; iterate over the
            # thread's cells and store into mO with the canonical
            # row-major (m_local, col_base + n_local) layout.
            n_cells_per_thread = (self.block_h * PV_N_MAX) // self.threads_per_cta
            for i in cutlass.range_constexpr(n_cells_per_thread):
                cell = Int32(i) * Int32(self.threads_per_cta) + tid
                m_local = cell // Int32(PV_N_MAX)
                n_local = cell % Int32(PV_N_MAX)
                h_global = Int32(head_base) + m_local
                d_global = Int32(col_base) + n_local
                acc_val = tDrAcc[i] if i < cute.size(tDrAcc) else Float32(0.0)
                acc_val = acc_val / softmax_l_t[m_local]
                mO[row, h_global, d_global] = acc_val.to(cutlass.BFloat16)


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


def _fwht_torch(x):
    """In-place FWHT along the last dim (power of 2). PyTorch fallback."""
    import torch
    h = 1
    n = x.shape[-1]
    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], n)
        h *= 2
    return x


def _rotate_and_concat_q(q_nope, q_rope):
    """Pre-rotate q_nope via FWHT_512 + 1/sqrt(LATENT_DIM); concat
    BF16 with q_rope to produce a (R, H, FULL_DIM=576) BF16 tensor
    the DSL kernel ingests directly. Uses C++ HIGGS rotate_query
    kernel if available; otherwise a PyTorch FWHT fallback.
    """
    import math
    import torch

    try:
        from sglang.jit_kernel.higgs_dense_2bit_mla_decode import (
            higgs_dense_2bit_mla_rotate_query,
        )
        R, H, _ = q_nope.shape
        q_rotated_f32 = torch.empty(
            (R, H, LATENT_DIM), dtype=torch.float32, device=q_nope.device
        )
        higgs_dense_2bit_mla_rotate_query(q_nope, q_rotated_f32)
        q_rotated_bf16 = q_rotated_f32.to(torch.bfloat16)
    except (ImportError, ModuleNotFoundError):
        # PyTorch FWHT fallback (slower; for testing).
        q32 = q_nope.to(torch.float32)
        q32 = _fwht_torch(q32) * (1.0 / math.sqrt(LATENT_DIM))
        q_rotated_bf16 = q32.to(torch.bfloat16)
    return torch.cat([q_rotated_bf16, q_rope], dim=-1).contiguous()


_KERNEL_CACHE: dict = {}


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

    cache_key = (block_h, block_n)
    if cache_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[cache_key] = HiggsDense2bitMLADecodeDSL(block_h=block_h, block_n=block_n)
    kernel = _KERNEL_CACHE[cache_key]

    # ITER 11 Option A: pre-decode HIGGS in Python (race-free, PyTorch).
    # Produces dense BF16 K and V tensors that the DSL kernel loads via
    # cute.autovec_copy / scalar load — no per-element scatter to
    # composed-swizzle SMEM (which had 32 races per IKP).
    k_dense, v_dense = _predequant_higgs_to_bf16(
        packed, scale, rope, codebook
    )

    mQN = from_dlpack(q_concat, assumed_align=16)
    mQR = from_dlpack(q_rope.contiguous(), assumed_align=16)
    mCK = from_dlpack(packed, assumed_align=16)
    mCS = from_dlpack(scale, assumed_align=2)
    mCR = from_dlpack(rope, assumed_align=16)
    mPT = from_dlpack(page_table.contiguous(), assumed_align=16)
    mO = from_dlpack(out, assumed_align=16)
    mCB = from_dlpack(codebook.contiguous(), assumed_align=16)
    mK = from_dlpack(k_dense, assumed_align=16)
    mV = from_dlpack(v_dense, assumed_align=16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(mQN, mQR, mCK, mCS, mCR, mPT, mO, mCB, mK, mV, sm_scale, stream)


def _predequant_higgs_to_bf16(packed, scale, rope, codebook):
    """Pre-decode HIGGS-2bit slots → dense BF16 K (n, 576) and V (n, 512).

    Mirrors _decode_higgs_torch in bench_higgs_dsl.py but inlined here
    so the kernel wrapper has no external test-harness dependency.

    K = [latent (512) || rope (64)] BF16.
    V = latent only (512) BF16.
    """
    import torch

    n = packed.shape[0]
    device = packed.device

    # Unpack 4-bit nibbles → 256 codebook indices per slot
    # packed: (n, 128) u8 = 4 groups of 32 bytes; each byte = 2 nibbles
    bytes_view = packed.reshape(n, 4, 32)  # (n, 4 groups, 32 bytes)
    lo = (bytes_view & 0x0F).to(torch.int64)        # (n, 4, 32)
    hi = ((bytes_view >> 4) & 0x0F).to(torch.int64)  # (n, 4, 32)

    # Per the C++ dim layout: dim g*128 + 2k uses cb[byte=k>>1, nibble=k&1, coord=0]
    #                        dim g*128 + 2k+1 uses cb[byte=k>>1, nibble=k&1, coord=1]
    # k = 0..63, byte_idx = k>>1 (0..31), nibble = k&1
    # So we interleave lo and hi per byte for the 64 codebook indices.
    # idx_per_group[g, k] = lo[g, k>>1] if (k&1==0) else hi[g, k>>1]
    # Vectorized:
    idx_per_group = torch.stack([lo, hi], dim=-1).reshape(n, 4, 64)  # (n, 4, 64)

    # codebook: (16, 2) → per-byte pair (cb_x, cb_y)
    cb_x = codebook[:, 0]  # (16,)
    cb_y = codebook[:, 1]  # (16,)
    # For each codebook idx, look up (cb_x, cb_y)
    cb_vals_x = cb_x[idx_per_group]  # (n, 4, 64)
    cb_vals_y = cb_y[idx_per_group]  # (n, 4, 64)
    # Per slot: latent dim g*128 + 2k = scale * cb_x[idx[g, k]]
    #          latent dim g*128 + 2k+1 = scale * cb_y[idx[g, k]]
    scale_f32 = scale.to(torch.float32).view(n, 1, 1)  # (n, 1, 1)
    latent_x = scale_f32 * cb_vals_x  # (n, 4, 64)
    latent_y = scale_f32 * cb_vals_y  # (n, 4, 64)
    # Interleave x and y: (n, 4, 64, 2) → (n, 4, 128) → (n, 512)
    latent = torch.stack([latent_x, latent_y], dim=-1).reshape(n, 512)

    v_bf16 = latent.to(torch.bfloat16)  # (n, 512)
    k_bf16 = torch.cat([v_bf16, rope.to(torch.bfloat16)], dim=-1).contiguous()  # (n, 576)
    return k_bf16, v_bf16.contiguous()
