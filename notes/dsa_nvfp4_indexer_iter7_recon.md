DSA NVFP4 indexer iter7 — recon plan
====================================

Where iter6 PRIMARY landed (commits f7c42ea9a + 814361203):

  - iter6 PRIMARY adds smem_values_t[64][132] (8448 B, 132-byte stride
    breaks the 16-way bank conflict that would otherwise hit when
    16 distinct dim_byte rows share the same SMEM bank set).
  - Inner loop: 1 LDS.b64 scales + 1 LDS.b16 values per 2-token pair
    (vs 1 LDS.b64 + 2 LDS.b8 in iter5 SECONDARY).
  - PTXAS: 31 reg (1 fewer than iter5 SECONDARY's 32), 19216 B SMEM,
    0 spill, 0 stack frame.
  - Measured win at 64/32768: mean_pool 240us -> 235us (~2-3% over
    iter5 SECONDARY, ~7-8% over iter4 predecode baseline).
  - Pipeline win at 64/32768: 1075us -> 1071us (<1%) because
    mean_pool is no longer the pipeline bottleneck once iter5 PRIMARY
    WMMA cand_score is on.

Where the brief's 10-15% mean_pool projection failed:

  - The brief assumed the iter5 SECONDARY LDS.b8 issue throughput
    was the bottleneck, with the value-byte LDS.b16 collapse winning
    a clean ~33% LDS-issue reduction in the hot loop.
  - Reality: the staging path adds 64 STS.b8 per thread (one per
    dim_byte) which is warp-coalesced into ~64 cycles via byte-enable.
    The inner-loop savings are ~64 cycles per warp (16 fewer LDS
    issues per warp per pair * pair_end pairs).
  - Net is positive but small. The mean_pool kernel is now bound by
    the SMEM-to-SMEM byte movement throughput, not LDS issue
    throughput.

Where the win actually is at production cells:

  - block_score: measured 162us at 64/32768 isolated. That is ~15%
    of pipeline (vs 18us mean_pool savings already squeezed). Even
    20-30% block_score win = 30-50us pipeline savings >> 5us
    mean_pool win.
  - cand_score WMMA Stage A: still scalar dequant + Q-prep before
    the iter5 PRIMARY WMMA Stage B.

iter7 vectors, ordered by expected ROI:

1. PRIMARY: persistent block_score with Q-value pre-staging.

   Current kernel grid: (max_blocks, q_rows) = 16384 CTAs at 64/32768.
   Each CTA does 32 NVFP4 dequants of the SAME Q tile (Q[row, *]).
   That's 16384 * 32 = 524288 redundant dequants per launch.

   Proposed: grid (q_rows,), each CTA loops over all max_blocks,
   stages Q[64 heads][128 dims] predecoded to fp32 = 32 KB SMEM
   ONCE, then runs the per-block dot.

   SMEM budget:
     existing:                   768 B
     +Q_predecoded[64][128]fp32: 32768 B  -> total ~33.5 KB
   Under sm_100a 48 KB default; no FuncSetAttribute needed.

   Trade-off: 256 inner block iters per CTA instead of 1. Per-CTA
   wall time grows ~256x, but launch overhead drops 256x. At 64/32768:
   16384 CTAs * ~10us each = 162us serialized; under persistent
   1 CTA per row * 64 rows = 64 CTAs * ~50us each = ~50us (148 SMs
   give >2 row-CTAs per SM). Projected ~3x block_score speedup,
   ~110us pipeline savings.

   Risk: per-row max_blocks varies (prefix_len/128); CTAs with short
   sequences finish early and idle. Mitigation: persistent-loop
   pattern with atomic block-counter (existing iter3 persistent
   cand_score kernel uses this; pattern can be ported).

   Files to touch:
     - python/sglang/jit_kernel/csrc/dsa/nvfp4_indexer_quant.cuh:
       add hisa_block_score_persistent_indexer_cache_nvfp4 after
       line 5685 (end of current block_score kernel).
     - python/sglang/jit_kernel/nvfp4_indexer.py: register kernel
       + add env var SGLANG_NSA_NVFP4_HISA_BLOCK_SCORE_PERSISTENT.
     - test/srt/test_dsa_indexer_iter7_block_score_persistent.py:
       bit-identical correctness vs current block_score (warp_sum
       reduction order is identical when iter loop is sequential).

2. SECONDARY: cand_score Stage A WMMA.

   iter5 PRIMARY did Stage B (Q*K dot). Stage A is per-tile
   K-dequant + Q-prep -- another mma.m16n8k8 fp32 target. Per-row
   Q is shared across the tile; the dequant fan-out matches the
   m=16 dimension well.

   Projected ~5-10% pipeline win compounded with iter5 PRIMARY.

3. TERTIARY: iter6 PRIMARY 4-token LDS.b32 extension.

   The smem_values_t[64][132] layout already supports LDS.b32 of
   4 consecutive bytes per row (any 4-byte aligned i is naturally
   aligned within the 132-byte row). Halves LDS issue count again.

   To keep bit-identical to iter5 SECONDARY: extract bytes from
   the uint32 in even/odd order so the sum0/sum1 partition holds.

   Expected: another 1-3% mean_pool win on top of iter6 PRIMARY.
   Worth doing only if iter7 PRIMARY (block_score) doesn't deliver.

4. STRETCH: TMA cp.async.bulk into smem_values_t for iter6
   PRIMARY's staging pass. The current byte scatter is 64 STS.b8
   per thread; a cp.async.bulk could replace this with a single
   8 KB DMA, freeing the scatter cycles for the hot loop. SM_100
   cp.async.bulk supports tensor-mapped reshapes, so the transpose
   could happen as a TMA shuffle rather than per-thread scatter.

Measurement discipline:
  - 64/32768 is THE production cell. Optimize for it; check
    32/16384 and 128/32768 as sanity bounds.
  - Bit-identical correctness against the current iter4 predecode
    baseline at every cell. Use max_diff == 0.0 (strict, not 1e-5).
  - Pipeline-level wall time at index_topk_freq=4 amortization.

Time budget for iter7: 3 hours per iter6 brief. Allocate:
  - 30 min: persistent block_score recon + Q-staging plan
  - 60 min: kernel implementation
  - 30 min: test + bench
  - 30 min: commit
  - 30 min: SECONDARY (cand_score Stage A WMMA) if PRIMARY converges
