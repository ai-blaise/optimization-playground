# #16 HIGGS dense MLA decode iter7 recon checkpoint

Target: another -5 to -10 % on iter5 stage A baseline (8d802dfca,
production grid B in {1, 8, 16, 32, 64, 128}, top_k=2048,
num_heads=16, num_splits=16, on a4-us-002-rl9 cuda-snap-v9 sm_100).

## State at start (HEAD 77428599f)

iter5 stage A is the live production kernel; iter6 depth-2 closed
out as measured-negative (37f1d13de). Microbench baseline (per the
iter6 close-out re-measurement on a4-us-002-rl9 cuda-snap-v9):

```
B       iter5 us
1         54.46
8        167.07
16       304.18
32       574.56
64      1109.95
128     2195.55
```

Cumulative #16 iter1-6 TPOT: ~33.4 ms (from 38.6 ms baseline,
-13.5 % net).

## iter7 PRIMARY vector design — sorted-page-table prepass +
## cp.async.bulk.tensor decode

### Architectural reality check

The iter6 close-out queued ``cp.async.bulk.tensor + sorted-page-table
TMA prepass`` as iter7 PRIMARY. Before writing kernel code, one
**critical** correctness gate: TMA tile load
(``cp.async.bulk.tensor.shared::cluster.global``) requires **contiguous
or strided** gmem access; it does not implement arbitrary gather. The
sm_100 TMA descriptors (``CUtensorMap`` with ``CU_TENSOR_MAP_INTERLEAVE_
NONE``) encode a contiguous (or strided) tile in the source tensor.

The HIGGS dense KV ``compressed`` tensor is ``[num_slots, 1,
kSlotBytes=272]``; the topk page indices are a **random permutation**
of slot IDs (the DSA indexer produces them per-row, scored by
relevance, so consecutive topk entries point to scattered slots in
gmem). Even after **sorting** the topk page indices ascending, the
**slots they point to remain at scattered gmem positions** — sorting
guarantees monotonicity but not adjacency. ``s = [5, 17, 99, 234,
800, ...]`` has sorted slot indices but consecutive slots are at
gmem offsets ``5*272``, ``17*272``, ``99*272``, etc. — not a
contiguous tile.

Therefore TMA bulk.tensor cannot directly amortize the per-slot
cp.async.16 issues without a **scatter→gather prepass** that packs
the topk-selected slots into a per-row contiguous scratch buffer.

### Two-pass architecture

#### Pass 1 — sort + gather prepass (new kernel)

For each (row, split) tile, gather the selected slots into a packed
``[num_rows, num_splits, chunk, kSlotBytes]`` scratch buffer.

Optional: pre-sort the topk page indices per row (no benefit beyond
prefetch locality at L2; consecutive sorted slots still scatter
across HBM). Skip the sort initially and revisit if L2 hit rate
matters.

The gather is a ``[num_rows * topk]`` thread-grid where each thread
loads 16 B from ``compressed[page_table[row, k] * 272 + offset]``
into ``packed_scratch[row, k * 272 + offset]``. ``17 lanes × 16 B =
272 B`` per slot (mirrors the iter5 stage1 cp.async tile width). For
``B=128, topk=2048``, total bytes moved: ``128 × 2048 × 272 = 71.3
MB``. At HBM 8 TB/s = **~9 us** transfer time (assuming
bandwidth-optimal).

Kernel launch overhead at B200: ~3-5 us per kernel.

**Pass 1 total: ~12-14 us added at B=128.** Iter5 at B=128 measures
2195 us total, so pass 1 alone is +0.6 %. **At B=1 (52 us total),
pass 1 is +20-25 %** — almost certainly a net regression at small B.

#### Pass 2 — decode against packed scratch (modified iter5 kernel)

Stage 1 main loop reads from ``packed_scratch[row, k_chunk_offset +
offset]`` (contiguous in gmem) instead of from ``compressed[pages[k]
* 272 + offset]`` (scattered). The cp.async issues collapse from
17 per slot to 1 per slot via TMA bulk.tensor.

**Per-slot saving:** ~16 ns instruction-issue overhead (16 fewer cp.
async.16 issues, each ~1 ns at sm_100 issue rate per CTA). Per CTA
at chunk=128 slots: ~2 us saved. Across all ``num_rows × num_heads
× num_splits = 128 × 16 × 16 = 32768`` CTAs amortized across 132 SMs
and ~8 CTAs/SM resident = ~31 wave-rounds, so **~2 us × 31 = 62 us
per pipeline step at B=128**. That's -3 % at B=128.

But the gather pass moves the same 71 MB once **extra** through HBM
that we'd be moving in cp.async anyway in iter5. So total HBM traffic
**doubles**: iter5 reads 71 MB, iter7 reads 71 MB (pass 1) +
71 MB (pass 2) = **142 MB**.

#### Net at B=128

```
iter5  total = 2195 us   (HBM 71 MB, compute-bound)
iter7  pass1 = ~13 us    (HBM 71 MB)
iter7  pass2 = ~2133 us  (HBM 71 MB, -62 us from fewer cp.async issues)
iter7  total = ~2146 us  (HBM 142 MB)
delta  = -49 us = -2.2 %
```

#### Net at B=1

```
iter5  total = 54 us
iter7  pass1 = ~6 us    (B=1 → 558 KB, ~0.07 us HBM but ~5 us launch)
iter7  pass2 = ~52 us   (per-slot saving smaller at fewer CTAs)
iter7  total = ~58 us
delta  = +4 us = +7 %
```

### Honest projection

iter7 as designed is **at risk of net regression** at small B
(launch overhead amortizes poorly) and **marginal gain** (-2 to -3 %)
at large B. The compute-bound diagnosis from iter6 close-out applies
here too: this kernel is not memory-bound, so reducing HBM-issue
instruction count by 16× per slot only buys back microseconds, not
milliseconds.

**Risk: high probability of measured-negative or break-even.**

### Decision

Implement iter7 as a **2-pass packed-scratch** prototype:

1. New kernel ``higgs_dense_2bit_mla_decode_gather_packed_kernel``
   that runs grid ``(num_rows, num_splits)`` × 128 threads, gathers
   the per-split chunk of slots into a packed scratch buffer.
2. Modified ``higgs_dense_2bit_mla_decode_stage1_split_kernel``
   reads from the packed scratch using TMA bulk.tensor (descriptor
   built in the host-side launcher per (num_rows, num_splits)).
3. Microbench, expect ~break-even at B=8-32, ~-2 to -3 % at B=128,
   ~+5 to +10 % at B=1. If the measured profile matches, the
   ``num_splits >= 16`` regime where chunk=128 is large enough for
   TMA to win is the only production cell where iter7 helps.
4. **If measured-negative**, close out honest like iter6 and queue
   the iter7 SECONDARY (codebook bank-conflict elimination via
   cb_smem[coord][i] transpose) as iter8 PRIMARY — that's the
   1-2 % bounded-risk vector.

### TMA descriptor build

Build one ``CUtensorMap`` per call, encoding the packed scratch
buffer as a 1D contiguous tile: ``cuTensorMapEncodeTiled`` with
``rank=1``, ``gmem_dim0 = num_rows * num_splits * chunk * kSlotBytes
/ 16`` (units of 16 B), ``smem_dim0 = chunk * kSlotBytes / 16``,
``swizzle = CU_TENSOR_MAP_SWIZZLE_NONE``. Each cp.async.bulk.tensor
issue inside the kernel moves one chunk of ``kSlotBytes / 16 = 17``
16-byte units. Reuse the descriptor across all (row, head, split)
CTAs.

But wait — at the kernel level, each CTA processes one (row, head,
split). The chunk is per (row, split). So the TMA descriptor encodes
the full ``[num_rows * num_splits, chunk * kSlotBytes / 16]`` tile,
and each CTA issues a tile load with offset
``(row * num_splits + split) * chunk * kSlotBytes / 16``.

### PTXAS projection

- Register count: TMA replaces cp.async.16 helpers; expect 0-2 reg
  delta vs iter5 (48 reg/thread); should stay well under the 64
  reg/thread budget.
- SMEM: packed scratch isn't SMEM (it's gmem). Slot SMEM unchanged
  at 4×kSlotBytes = 1088 B per CTA.

### Risk gates

1. ``cuTensorMapEncodeTiled`` requires CUDA 12.0+; nvfp4_indexer
   already uses it, so the runtime supports it. Driver entry point
   resolution via ``cudaGetDriverEntryPointByVersion`` already
   present in the repo (``hisa_get_cu_tensor_map_encode_tiled``).
2. Packed scratch buffer size at B=128, topk=2048, kSlotBytes=272 =
   71 MB. Fits in the existing layersplit scratch pool with room.
3. cp.async.bulk.tensor on sm_100 requires the source tile to be at
   16-B alignment; packed scratch base will be naturally aligned.
4. Bit-identical correctness vs iter5: same math, just different
   data path. Easy to verify with the bench harness's existing
   max_abs_diff probe.

## iter7 SECONDARY (queued if PRIMARY measures negative)

Codebook bank-conflict elimination via ``cb_smem[coord][i]``
transposed layout. Iter6 close-out noted this as 1-2 % gain at
production shape but flagged IKP measurement is needed to confirm
conflict count. Lower-risk, smaller-ROI fallback.

## iter7 TERTIARY (queued)

rope-tail uint64 4-bf16 batched load (<1 % polish).

## Plan of work (within 4-hour budget)

1. **NOW** (within hour 1): commit this recon checkpoint, sync to
   a4-us-002-rl9 cuda-snap-v9 bench pod.
2. **Hour 1-2**: write the gather kernel + TMA descriptor build +
   modified split kernel. PTXAS check.
3. **Hour 2-3**: run microbench at production grid, compare to
   iter5 baseline.
4. **Hour 3-4**:
   - If positive: commit, write close-out commit body with measured
     B-table, queue iter8.
   - If negative: close-out honest, fall back to iter7 SECONDARY
     (codebook bank conflict) — implement, PTXAS, microbench,
     commit.

## Cumulative #16 TPOT projection at recon time (unchanged)

```
baseline 38.6 ms TPOT -> iter2  ~37.0 ms TPOT (-3.9%)
                     -> iter4  ~34.0 ms TPOT (-8.1%)
                     -> iter5  ~33.4 ms TPOT (-2.4% kernel)
                     -> iter6  ~33.4 ms TPOT (no change; depth-2 closed
                                             out as measured-negative)
                     -> iter7  (TBD)
```

iter7 honest target: ~32-33 ms TPOT (-1 to -4 % on iter6).
