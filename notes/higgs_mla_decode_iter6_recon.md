# #16 HIGGS dense MLA decode iter6 recon checkpoint

Target: another 3-5% TPOT cut on top of iter5 (8d802dfca; cumulative
~33.4 ms TPOT on production B=128 cell).

## State at start (commit 8987cb270 / fe183d144 base)

iter5 stage A is the immediate baseline. Microbench at production grid:

```
B     iter5 us   (vs iter4)
1       52.45     -16.3 %
8      163.04     -4.8 %
16     296.16     -3.4 %
32     562.34     -2.7 %
64    1086.82     -2.4 %
128   2152.77     -2.3 %
```

Iter5 invariants:

- `__shared__ __align__(16) uint8_t slot_smem[4][kSlotBytes]` (1088 B).
- `__shared__ float warp_partials[2][4]` (32 B; iter4: 16 B).
- 1 commit + 1 `wait_group<1>` + 1 `__syncthreads` (cp.async readiness)
  + 1 `__syncthreads` (warp_partials reduce) per *pair-iter*
  (was: per slot at iter4).
- 48 reg/thread, 1248 B SMEM per CTA (EDEN2-16) / 1120 B (saw_scalar2).
- `__launch_bounds__(128, 8)`: 8 CTAs/SM at sm_100, 64 reg/thread budget
  → headroom 16 reg.

Iter5 invariant on cp.async pipeline depth: at the top of each pair-iter
exactly 1 commit group is in flight (the next-pair prefetch from the
prev iter). After the iter's own `commit_group;` that becomes 2 groups;
`wait_group<1>` immediately drains it back to 1.

## iter6 PRIMARY vector — depth-2 cp.async pipeline

Extend to **2 commits in flight at iter top**: depth-2 = 3 pair-iters
worth of slot data resident in smem at the peak (current consuming +
next-pair-in-flight + after-next-pair-in-flight). Hides one more L2/HBM
round-trip per pair-iter under compute.

### Buffer indexing

iter5 indexes `slot_smem[col & 3]` with stride 2 per loop (so each pair
uses bufs `[c%4][c%4+1 mod 4]`). With 2 pairs in flight you only need
2 pair-pairs = 4 buffers; with 3 pairs in flight (current + 2
prefetched) you need 6 buffers.

Use a *pair-iter* counter `it = (col - begin) / 2` and index by
`(it mod 3)`:

```
lo buf at iter ``it`` = slot_smem[2 * (it % 3) + 0]
hi buf at iter ``it`` = slot_smem[2 * (it % 3) + 1]
```

After `wait_group<2>` drains pair-iter ``it - 1`` (and earlier), the
buf at `(it - 1) % 3` is free; the new prefetch for ``it + 2``
(`(it + 2) % 3 == (it - 1) % 3`) lands on those same bufs. No conflict
under the depth-2 invariant.

### Prologue

Iter5 prefetches 1 pair (pair-iter 0) and submits 1 commit. Iter6
prefetches **2 pairs** (pair-iter 0 and pair-iter 1) and submits
**2 commits**. That leaves 2 groups in flight at the top of the main
loop, satisfying the iter6 invariant.

### Main loop body

```
for (it = 0; col < end; it += 1, col += 2):
    next_it_to_prefetch = it + 2
    cols_to_prefetch = col + 4, col + 5
    bufs = slot_smem[(it+2)%3 * 2 + 0/1]
    prefetch lo + hi (safe page 0 fallback)
    commit_group;            # now 3 in flight
    wait_group<2>;           # drain to 2 (pair-iter ``it+1`` and ``it+2`` left)
    __syncthreads;           # readers for it == col, col+1 see fresh smem
    consume pair-iter it
    update prev_valid_*
```

### SMEM after iter6

```
slot_smem[6][272] = 1632 B   (iter5: 1088 B; +544 B)
warp_partials[2][4] = 32 B   (unchanged)
cb_smem[16 * 2] = 128 B      (EDEN2-16 only; unchanged)
```

Per CTA: 1792 B EDEN2-16 / 1664 B saw_scalar2. At 8 CTAs/SM: ~14.3 KB
total stage1 SMEM, well within the B200 228 KB SMEM budget.

### Register cost prediction

iter6 adds *no* new live registers per slot — the per-slot
`i0..i3 / c0..c3 / scale_a/b / val_a/b / warp_sum_a/b` accumulators
are unchanged. The only delta is 2 extra `next_buf_*` index arithmetic
ints inside the prefetch block; PTXAS will likely fold these into
the address computation. Expected: 48 reg/thread unchanged (and a
tolerance of +0 to +4 reg if the loop unrolls more aggressively).

### Bandwidth math

Per pair-iter: 2 slots × 272 B = 544 B of cp.async traffic. At
B200 HBM 8 TB/s and 132 SMs / 8 resident CTA = 1056 in-flight CTAs,
amortized: ~5.1 ns per pair-iter for the actual transfer. But each
cp.async issue has ~150 ns L2 hit latency before bytes show up. iter5
covers 1 pair of that latency under the current pair-iter's compute
(~50-80 ns total: warp reduce + softmax + acc update); iter6 covers
1.5-2 pairs (the second commit issued during iter ``it``'s compute
window starts L2 lookup before iter ``it+1`` needs it). Expected
TPOT gain: -1.5 to -3 % on memory-bound regimes (B >= 8).

## Risk

1. `cp.async.wait_group<2>` exposes a longer dependency chain — if the
   compute window per pair-iter is shorter than 2× HBM latency, no win.
   iter5's measured per-pair time scales as (B=1 → 52 us, B=128 →
   2153 us). Per pair-iter at B=128 the kernel processes ~16 pairs ×
   16 splits = ~2153/256 = 8.4 us. At B=1 it's 52/256 = 0.2 us
   (compute-bound), so iter6 should help most at B>=16.
2. `(it % 3)` is a non-power-of-2 modulus. PTXAS should compile it as
   `iadd3 + isetp + sub` (3-4 ops); micro-cost compared to a full
   pair-iter is negligible. Could fall back to a `slot_smem[8]` 4-pair
   ring (`(it & 3)` mod) if PTXAS emits a div.s32, but `(it % 3)` is
   small-constant — PTXAS recognises this.
3. Tail (end - begin odd, or end - begin < 4): iter5's valid_hi mask
   already handles odd end. For end - begin < 4 the prologue's
   pair-iter-1 prefetch lands fully outside the range — every slot
   uses the safe fallback page 0; iter6's valid masks gate the math
   identically. Add an `(begin + 2 < end)` check before prefetching
   pair-iter 1 in the prologue to avoid issuing 0 commits when
   `begin + 2 >= end` (a 2-slot-or-fewer split).

## iter6 SECONDARY (queued, time-permitting)

cp.async.bulk.tensor + sorted-page-table TMA prepass — architectural
change, 2-day work, +2-4 ms TPOT. Defer to iter7+.

## iter6 TERTIARY

rope-tail uint64 4-bf16 batch load (<1% polish). Defer.

## Implementation plan

1. Edit `python/sglang/jit_kernel/csrc/quantization/higgs_dense_2bit_mla_decode.cuh`:
   - Bump `slot_smem` from `[4]` to `[6]` (both EDEN2-16 and saw_scalar2 split kernels).
   - Rework prologue: prefetch pair-iter 0 + pair-iter 1, submit 2 commits.
   - Inside main loop: change index from `col & 3` to `2 * (((col - begin) / 2) % 3) + (col & 1 ? 1 : 0)`.
     Simpler: track `it` as a loop variable, compute `2 * (it % 3) + 0` and `+ 1`.
   - Change `wait_group<1>` → `wait_group<2>` inside the main loop.
   - Update comments in the file header.

2. PTXAS check: `nvcc -arch=sm_100a -ptx -Xptxas -v -DBUILD_PTX_ONLY` confirm
   register count <= 64 and SMEM per CTA fits __launch_bounds__(128, 8).

3. Microbench: production grid `B in {1, 8, 16, 32, 64, 128}`, top_k=2048,
   num_heads=16, num_splits=16, on a4-us-001-rl9 sm_100 via the existing
   `benchmark/kernels/bench_higgs_dense_2bit_mla_decode.py` harness.

4. Bit-identical correctness check vs iter5 via running the existing test:
   `kubectl exec -n dynamo-system <pod> -- python -m pytest
   /opt/optimization-playground/python/sglang/test/test_higgs_dense_2bit_kv.py`.

5. Commit.

## iter7 next vector preview

cp.async.bulk.tensor + sorted-page-table TMA prepass remains the queued
SECONDARY. If iter6 hits its target the cumulative becomes ~32 ms TPOT
(iter5 ~33.4 ms × 0.97 = ~32.4 ms), making the iter7 TMA prepass the
direct next vector.
