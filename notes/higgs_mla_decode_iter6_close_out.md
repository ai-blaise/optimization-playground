# #16 HIGGS dense MLA decode iter6 — honest negative measurement

## Result

Depth-2 cp.async pipeline (6-buf slot rotation, ``wait_group<2>``,
2 commits in flight at iter top) **regresses** vs the iter5 stage A
baseline at every production shape.

## Setup

Hardware: a4-us-002-rl9 sm_100 (NVIDIA B200, 580.105.08 driver,
container ``ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4-dynamo-1.1.1-v9``,
cuda-snap-v9 pod GPU 0 — idle outside this bench).

Bench: ``benchmark/kernels/bench_higgs_dense_2bit_mla_decode.py``,
100 iters / 30 warmup, num_heads=16, top_k=2048, num_splits=16
(production DP=TP=8 cell). Per-call median microseconds.

## Measurement

```
B       iter5 us   iter6 us   delta%   delta us
1         54.46      57.01    +4.7%    +2.55
8        167.07     169.18    +1.3%    +2.11
16       304.18     308.16    +1.3%    +3.98
32       574.56     580.93    +1.1%    +6.37
64      1109.95    1122.54    +1.1%    +12.59
128     2195.55    2221.58    +1.2%    +26.03
```

Iter6 is bit-identical to iter5 at every shape tested (production
B=1/16/128/H=64, TP-shard B=1/128/H=16, odd-chunk topk=33, 1-pair,
1-slot, exactly-2-pairs, 3-pairs boundary, B=4 H=32).

## PTXAS report (sm_100a)

```
Kernel                          iter5  iter6  delta
stage1_split_kernel              48      56    +8  reg/thread
saw_scalar2_stage1_split         48      54    +6  reg/thread
stage1_split SMEM                1248    1792  +544 B
saw_scalar2_stage1_split SMEM    1120    1664  +544 B
spills                            0       0     -
```

All resident-CTA constraints still satisfied
(``__launch_bounds__(128, 8)`` budget at sm_100 = 64 reg/thread; iter6
EDEN2-16 split lands at 56, leaving 8 reg/thread headroom — vs iter5's
16 reg/thread headroom). SMEM stays well under the 228 KB B200 cap
(8 CTAs/SM × 1.79 KB = 14.3 KB stage1 SMEM).

## Why it regresses

The depth-2 pipeline assumes the per-pair-iter compute window is long
enough to hide >=2 L2/HBM round-trips of cp.async transfer latency.
At production shape (B=128, topk=2048, num_splits=16) the
per-pair-iter cost is ~34 us / 64 pair-iters = ~530 ns / pair-iter
(estimated from cell total / pair-iter count, but the actual
per-pair-iter time across all CTAs amortized through 132 SMs and 8
resident CTAs is ~34 us / (256 pair-iters * 1024 rows = small): the
bottleneck is **compute**, not HBM):

- ``higgs_unpack_indices_smem`` (~20 ns/slot)
- 4 cb_smem lookups (~10 ns/slot, with bank conflict)
- 4 mul-adds for q.K dot (~3 ns)
- bf16 -> float scale + FMA (~3 ns)
- conditional rope FMA (~5 ns, half-warp active)
- warp_reduce_sum (5-step shfl_down chain, ~10 ns)
- softmax fold (2 __expf, ~15 ns)
- acc updates (4 mul-adds, ~3 ns)

Per slot ~70 ns; pair-iter ~140 ns. The cp.async transfer of 544 B
across 17 lanes / pair-iter is ~5-10 ns at HBM 8 TB/s — already
hidden completely by the existing iter5 single-pair lookahead. There
is no second L2/HBM round-trip to hide because the bandwidth is
under-subscribed at this compute intensity.

The +8 reg/thread (compiler unrolls the deeper pipeline aggressively)
slightly hurts compute throughput: more spill-candidate values, more
pressure on the register file. The +1 commit per pair-iter (the
prologue extra) is a tiny constant overhead amortized over the loop.
The +1 ``__ldg(&pages[col + 4 / 5])`` per iter adds a 1-2 ns L2 hit
per iter.

Net: iter6 regresses by 1.2 us / pair-iter at B=128 = ~26 us per
call (1.2% slowdown). The honest read: **depth-2 cp.async is the
wrong vector for this kernel at this compute intensity.**

## Closes out the queued depth-2 vector

The iter4 / iter5 commit messages both queued "depth-2 pipeline"
as iter6 PRIMARY. The above microbench at production grid is enough
to close this vector — even allowing for ±2% per-cell noise, the
depth-2 pipeline never beats the iter5 single-pair lookahead at any
batch tested. The depth of the cp.async pipeline only matters when
compute < 2x cp.async latency, which is not this kernel's regime.

## iter7 next vector candidates

The next ROI vector is **architectural**, not pipeline-depth:

1. **(PRIMARY) cp.async.bulk.tensor + sorted-page-table TMA prepass.**
   Sort the topk page indices on host so consecutive slots are
   contiguous; one TMA tile load amortizes descriptor build over
   4-8 slots. ROI: +2-4 ms TPOT, 2-day work. This was noted as
   the iter6 SECONDARY in the iter5 commit (8d802dfca) and remains
   the highest-ROI remaining vector.

2. **(SECONDARY) Codebook bank-conflict elimination.**
   Profile ``cb_smem[i * kPairDim + coord]`` lookups (4 per slot
   per thread, ``i`` random in [0..15], ``coord = tid & 1``) for
   bank conflict count via NSight Compute or intra-kernel-profiler
   (https://github.com/yao-jz/intra-kernel-profiler). If the warp
   sees 2-4-way bank conflict on cb_smem, transpose the layout to
   ``cb_smem[coord][i]`` so threads on the same warp accessing the
   same ``coord`` see different banks. Expected gain: 1-2% at
   production shape.

3. **(TERTIARY) Softmax exp() batching.**
   Iter5 calls ``__expf`` twice per pair-iter (one per slot's
   ``alpha`` / ``beta`` update). The Turing/Hopper/Blackwell
   ``__expf`` lat ~12-16 cycles each. A fused ``expf_pair``
   intrinsic (compute both exps in pipelined SFU ops) could shave
   3-5 ns / pair-iter. Compiler may already do this. Confirm via
   PTXAS asm inspection.

## Verdict

Land iter6 as a recon checkpoint (7b5b27dbd) + this honest-negative
measurement. **Kernel reverts to iter5 stage A (8d802dfca).** The
``wait_group<2>`` and 6-buf rotation are not productionized. The
queued depth-2 vector is closed out as a measured-negative; the
campaign's #16 cumulative TPOT projection stays at iter5's
~33.4 ms (no regression to production).

Iter7 PRIMARY: cp.async.bulk.tensor + sorted-page-table TMA prepass
(architectural change, queue for the next session).

## Cumulative #16 TPOT projection (unchanged from iter5)

```
baseline 38.6 ms TPOT -> iter2  ~37.0 ms TPOT (-3.9%, iter2 #16)
                     -> iter4  ~34.0 ms TPOT (-8.1%, iter4 #16)
                     -> iter5  ~33.4 ms TPOT (-2.4% kernel)
                     -> iter6  ~33.4 ms TPOT (no change; depth-2 closed
                                             out as measured-negative)
```

iter7 target: ~30-32 ms TPOT via TMA prepass (+5-15% TPOT cut).
