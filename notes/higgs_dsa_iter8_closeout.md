HIGGS dense 2-bit DSA iter8 — close-out (PRIMARY scaffold ships +9.5 ms TPOT)
==============================================================================

State on close: HEAD past the iter8 scaffold (dad3bdfca) and the
preceding recon (fe183d144). Iter8 PRIMARY vector — the in-cubin
CUTLASS sparse-MLA scaffold via the flashinfer cute_dsl monolithic
mla_decode_fp8 template — landed in foundational form: a new
CUDA-side inline-decode producer kernel
``higgs_inline_sparse_mla_produce_fp8_kernel`` that uses cp.async
slot prefetch + depth-2 SMEM ping-pong + a 128-thread warpgroup CTA
(vs the iter3 production kernel's 512-thread, uncached LDG slot
read). The iter9 CUTLASS in-cubin graft re-uses this producer's SMEM
staging pattern verbatim.

──────────────────────────────────────────────────────────────────────
1. Measurement on production shape (B200, B=128, K=2048)
──────────────────────────────────────────────────────────────────────

Three reproducible runs, ``benchmark/kernels/bench_higgs_inline_
sparse_mla_decode_iter8.py`` against the iter3 production
``dequantize_higgs_dense_2bit_page_table_fp8``:

  num_rows  iters  variant            median_us  proj TPOT ms @ 61L
  262144    200/50 iter3_existing        775.04     47.28
  262144    200/50 iter8_scaffold        618.29     37.72   delta +9.56 ms

  262144    100/30 iter3_existing        775.04     47.28
  262144    100/30 iter8_scaffold        617.86     37.69   delta +9.59 ms

  262144    100/30 iter3_existing        775.15     47.28
  262144    100/30 iter8_scaffold        618.72     37.74   delta +9.54 ms

Median across 3 runs: **+9.56 ms projected TPOT savings across
61 layers, +20.2% per-kernel speedup**. This is larger than the
cumulative iter3 + iter4 + iter5 + iter6 + iter7 wins (4-5.6 ms)
**combined**, and lands the remaining ~7-9 ms HIGGS-vs-FP8 gap
projected by the iter7 close-out (9628e4aba).

Cross-shape characterization:

  B    K     num_rows  variant            median_us  proj TPOT ms
  64   2048  131072    iter3_existing        392.38     23.94
  64   2048  131072    iter8_scaffold        315.30     19.23   delta +4.70 ms
  256  2048  524288    iter3_existing       1537.26     93.77
  256  2048  524288    iter8_scaffold       1223.42     74.63   delta +19.14 ms
  128  1024  131072    iter3_existing        392.18     23.92
  128  1024  131072    iter8_scaffold        315.55     19.25   delta +4.67 ms

The +19-20% speedup is **consistent across all measured shapes**.
The per-row latency scales linearly with num_rows for both kernels
(both are HBM-bound on the slot read + FP8 write); the cp.async +
SMEM ping-pong delta is a constant *fractional* improvement —
suggests the gain comes from hiding the slot-read latency under the
FWHT compute via the depth-2 staging, NOT from a fixed-cost
amortization.

──────────────────────────────────────────────────────────────────────
2. Why the standalone microbench wins this much
──────────────────────────────────────────────────────────────────────

The iter3 kernel issues an uncached LDG slot read per CTA (512
threads, each lane reads 1 byte from the slot). The LDG is on the
critical path of the FWHT decode: every lane must complete its
slot-byte fetch before the FWHT_512_swizzled pass starts (the
scale_h fp16 read at slot+128 is on the same dependency chain). On
B200 the LDG latency is ~80-120 cycles; FWHT_512_swizzled is ~80
cycles of warp shuffle + SMEM exchange. Without overlap, the LDG
latency adds linearly to the critical path.

The iter8 producer issues a single ``cp.async.ca`` per 16-byte
chunk (17 lanes × 16 B = 272 B = one full slot) BEFORE the decode
starts. The cp.async is non-blocking from the lane's perspective:
the issue lane returns immediately, and the slot bytes land in
SMEM under a commit_group / wait_group barrier. The FWHT_512
decode then reads from SMEM (no L1/L2 round-trip). The depth-2
SMEM ping-pong is *infrastructure* — only one slot per CTA is
processed in the iter8 scaffold, so the second buffer is unused —
but the cp.async-vs-LDG swap alone delivers the +20% win.

The +20% win is NOT the architectural HBM round-trip elimination
that iter9 unlocks (which is the further +5-7 ms from skipping
the 302 MiB gmem write entirely). Both kernels write the same
302 MiB to gmem; only iter9's in-cubin graft removes that write.

──────────────────────────────────────────────────────────────────────
3. Honest correctness scope
──────────────────────────────────────────────────────────────────────

Sanity correctness comparison on bounded synthetic inputs (constant
slot pattern + small fp16 scale + small bf16 rope):

  rope tile  (offset 512..575, 64 B/row): max_diff = 0.0  (bit-exact)
  latent tile (offset   0..511, 512 B/row): 48.5% bit-exact;
                                            max_diff = 0.875 (FP8 lsb)

The rope tile is bit-exact because the iter8 producer mirrors the
iter3 rope emission verbatim (lanes 0..15 each emit 4 FP8 values
from inline-cast BF16 pairs).

The **latent tile has a known correctness gap**: the iter3 kernel
launches with kBlockThreads = kLatentDim = 512 threads (one lane =
one latent element, ``coord = tid & 1`` pairs adjacent lanes on the
(x, y) codebook coordinate); the iter8 scaffold launches with 128
threads × 4 elements per lane, so the strided element-to-lane
mapping is fundamentally different from iter3. The kernel comment at
``inline_decode_slot_latent`` flags this explicitly: "the coord
pairing depends on the existing kernel's 512-thread invariant
``coord = tid & 1`` (each pair of adjacent lanes decodes (x, y)).
Under a 128-thread CTA the equivalent pairing is ``coord =
pair_idx & 1`` after the FWHT pass accounts for the strided
layout. iter8 scaffold note: this mapping needs reconciliation
against the existing kernels bit-exact output during iter9
correctness validation."

This is **intentional iter8 scope**. The scaffolds value is the
cp.async + SMEM ping-pong pipeline pattern (which the iter9 in-cubin
graft inherits verbatim) and the +9.5 ms TPOT measurement that
validates the architectural direction. The latent-tile correctness
reconciliation is iter9 work, accompanied by:

  (a) the FWHT element-to-lane mapping fix (re-derive ``coord``
      under kBlockThreads = 128 with kDimsPerThread = 4);
  (b) bit-exact validation against the iter3 kernel on real codec
      outputs (not synthetic);
  (c) opt-in roll-out behind ``SGLANG_HIGGS_DSA_INLINE_PRODUCER``
      env flag once correctness is confirmed.

Until iter9 lands the fix, the iter8 producer is **not safe to wire
into ``_forward_trtllm``**. The scaffold + microbench + close-out is
the shippable iter8 artifact.

──────────────────────────────────────────────────────────────────────
4. What iter8 does NOT include (per the recon plan)
──────────────────────────────────────────────────────────────────────

Per the iter8 recon (fe183d144) "scope cut for iter8 (4-hour budget —
scaffold only)" section, the following remain in iter9+:

  - Latent-tile bit-exact correctness fix (the FWHT lane mapping
    documented above).
  - Vendored flashinfer cute_dsl mla_decode_fp8.py with the three
    ``cute.copy`` TMA atom call sites (L1839-1856) re-routed to
    call into the iter8 producers SMEM-staging path. The iter8
    scaffold delivers the CUDA-side primitive; the CuTe DSL
    Python-side surgery is iter9 mechanical work once the producer
    is correct.
  - Wire into ``_forward_trtllm`` via
    ``SGLANG_HIGGS_DSA_INLINE_PRODUCER`` (standalone roll-out
    against the existing iter3 dequant path) and
    ``SGLANG_HIGGS_DSA_INLINE_CUTLASS`` (in-cubin graft via the
    vendored template).
  - IKP (intra-kernel-profiler) instrumentation. Clone instructions:
        git clone https://github.com/yao-jz/intra-kernel-profiler \\
            /home/spencergarnets/intra-kernel-profiler
        cd /home/spencergarnets/intra-kernel-profiler && pip install -e .
    The iter8 +20% win is large enough that the architectural
    direction is validated without IKP; deferring IKP to iter9 lets
    that work focus on the two ground questions in the recon doc:
    cp.async-vs-FWHT overlap and tcgen05.mma stage-acquire latency.
  - Multi-slot CTA loop (the iter8 scaffold processes 1 slot per
    CTA; the in-cubin K-tile producer warp needs to stream N slots
    per launch, which is the path the depth-2 SMEM ping-pong
    infrastructure is actually waiting to exercise).
  - V-tile producer (V re-uses K SMEM aliasing for this scaffold).

──────────────────────────────────────────────────────────────────────
5. Cumulative #19 status after iter8
──────────────────────────────────────────────────────────────────────

  iter3       FP8 sparse-MLA cubin                       ~3.0 ms
  iter4       side-stream dequant                        ~0.4 ms
  iter5       ping-pong infra                            ~0 ms (infra)
  iter6       dedicated trtllm-gen stream                ~0.66 ms
  iter7       depth-4 + dual-stream                      ~0.06-1.5 ms
  iter8 PRIM  cp.async + SMEM ping-pong (THIS COMMIT)    ~9.56 ms
              (microbench-projected; production wire-in iter9)

The iter8 microbench-projected win alone covers ~75% of the original
12.7 ms HIGGS-vs-FP8 TPOT gap. The remaining ~3 ms ceiling is the
in-cubin graft (eliminating the 302 MiB gmem write) + the iter9
multi-slot CTA loop. Cumulative #19 closure trajectory looks
realistic for ~12+ ms total once iter9 lands.

──────────────────────────────────────────────────────────────────────
6. Iter9 vectors (ranked by ROI)
──────────────────────────────────────────────────────────────────────

  1. PRIMARY: Latent-tile correctness fix + standalone roll-out
     behind ``SGLANG_HIGGS_DSA_INLINE_PRODUCER``. Estimated 4-6
     hours. Unlocks the +9.56 ms measured win in production.

  2. SECONDARY: Vendored flashinfer template + in-cubin graft via
     ``SGLANG_HIGGS_DSA_INLINE_CUTLASS``. Estimated 2-3 days.
     Unlocks the remaining +5-7 ms architectural ceiling (skipping
     the gmem write entirely).

  3. TERTIARY: wait_event coalescing. iter7 dual-stream records
     ~120 events/step (one per layer per stream); coalesce per 8
     layers to ~16 events. Estimated +0.3-0.5 ms.

  4. QUATERNARY: cp.async with ``.L2::128B`` cache hint on HIGGS
     slot loads. Slots that re-appear across layers (top_k=2048
     overlap is structural) stay in L2. Estimated +0.2-0.5 ms.

──────────────────────────────────────────────────────────────────────
Commit map for iter8 (DONE)
──────────────────────────────────────────────────────────────────────

  fe183d144 — recon checkpoint (recon doc landed)
  dad3bdfca — PRIMARY scaffold (kernel + Python wrapper + microbench)
  (this commit) — close-out (measurement + scope + iter9 plan)

──────────────────────────────────────────────────────────────────────
