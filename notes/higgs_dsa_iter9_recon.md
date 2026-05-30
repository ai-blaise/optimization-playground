HIGGS dense 2-bit DSA iter9 — recon (latent-tile correctness fix plan)
======================================================================

State on open: HEAD past the iter8 close-out (b95bb0af8). Iter9
PRIMARY vector: fix the latent-tile element-to-lane mapping in the
iter8 inline producer (``higgs_inline_sparse_mla_decode.cuh``) so the
latent tile is bit-exact (or within 1 fp8 lsb) vs the iter3 baseline,
then wire the producer into ``get_higgs_selected_kv_buffer`` (the
caller of ``dequantize_higgs_dense_2bit_page_table_fp8`` in
``memory_pool.py``) behind a default-OFF env flag
``SGLANG_HIGGS_DSA_INLINE_PRODUCER``.

──────────────────────────────────────────────────────────────────────
1. Diagnosis — why iter8 latent tile is only 48.5% bit-exact
──────────────────────────────────────────────────────────────────────

The iter3 production kernel
``higgs_dense_2bit_dequant_page_table_fp8_kernel`` (in
``higgs_dense_2bit_kv.cuh``) launches with
``kBlockThreads = kLatentDim = 512`` and one element per thread:

  - ``pair_idx = tid >> 1``  in [0, 256)  — adjacent-tid pairing
  - ``byte_idx = pair_idx >> 1`` in [0, 128) — packed-byte index
  - ``nibble   = pair_idx & 1`` in {0, 1}   — nibble select
  - ``cb_idx   = nibble ? (packed >> 4) : (packed & 0x0F)``
  - ``coord    = tid & 1``      in {0, 1}   — codebook (x, y) lane pair
  - ``g        = codebook[cb_idx * 2 + coord]``
  - ``rot_recon = scale * g``
  - ``result   = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim``
  - ``row_out[tid] = __nv_fp8_e4m3(result * inv_kv_scale)``

The critical invariant: each lane tid logically owns element index
tid (0..511). ``fwht_512_swizzled`` uses ``threadIdx.x`` internally —
it writes ``buf[swizzle(tid)]`` after a warp-shuffle butterfly, runs
SMEM-exchange passes whose addressing is in terms of tid
(``wht_group = tid / (len << 1)``, etc.), and returns
``buf[smem_swizzle_idx(tid)]``. The function inherently requires a
512-lane launch (one lane per FWHT element). It cannot be reduced to
128 lanes by giving each lane more elements: the warp-shuffle
butterfly only spans 32 lanes per level, the cross-warp passes assume
the address mapping ``pos = tid mod (2*len)`` with tid in [0, 512),
and the swizzled-write pattern ``buf[swizzle(tid)]`` covers exactly
512 distinct slots.

The iter8 scaffold launches with ``kBlockThreads = 128`` and
``kDimsPerThread = 4``, looping 4 sequential calls to
``fwht_512_swizzled`` per slot:

    for e in 0..3:
      pair_idx = e * 128 + tid       // [0, 512) in stride-128 chunks
      byte_idx, nibble, cb_idx       // as above
      coord = (pair_idx & 1) ? (~tid & 1) : (tid & 1)  // iter8 patch
      g = codebook[cb_idx*2 + coord]
      rot_recon = scale * g
      latent_out[e] = fwht_512_swizzled(rot_recon, buf) * kInvSqrtLatentDim

Three independent bugs combine:

  B1. The per-call FWHT_512 is run with only 128 active lanes. The
      warp-shuffle butterfly works fine within each of the 4 warps,
      but the SMEM-exchange passes for ``len >= 32`` only write
      ``buf[swizzle(tid)]`` for tid in [0, 128) — the upper 384 slots
      hold stale data from the prior FWHT call (or uninitialized junk
      on the first call). Each SMEM pass then reads back
      ``buf[swizzle(b)]`` where b may exceed 128, returning garbage.
      Each FWHT_512 call thus computes WHT on a 128-element subspace
      polluted by 384 stale values.

  B2. The element-to-lane mapping per pass is ``pair_idx = e*128 + tid``
      (lane tid owns positions {tid, 128+tid, 256+tid, 384+tid}) but
      ``fwht_512_swizzled`` reads/writes ``buf[swizzle(tid)]`` — it
      assumes element index == tid. Even if B1 were magically fixed
      (e.g., by zero-padding the upper 384), each lane would receive
      the WHT result for FWHT-index tid, not for the element it owns
      (e*128 + tid).

  B3. The ``coord = (pair_idx & 1) ? (~tid & 1) : (tid & 1)`` patch in
      iter8 tries to reconcile the lane-pair mapping but is
      inconsistent with the iter3 ``coord = tid & 1``. For e=0
      (pair_idx == tid), ``pair_idx & 1 == tid & 1``, so the ternary
      reduces to ``tid & 1 ? (~tid & 1) : (tid & 1)`` — for tid even
      this yields 0, for tid odd this yields 0, so coord is always 0
      on the e=0 pass. The iter3 invariant is ``coord ∈ {0, 1}``
      alternating by tid parity.

These bugs cancel coincidentally for ~48.5% of the output (the FWHT
on the random-seeded synthetic input has many small magnitudes that
happen to FP8-round to the same value despite garbage), but the path
is fundamentally broken — ``max_diff = 0.875`` (an FP8 lsb on a
~1.0-magnitude value is 0.0625; 0.875 implies multi-bit errors)
confirms multi-bit decode wrongness, not just rounding noise.

──────────────────────────────────────────────────────────────────────
2. Fix — restore the 512-thread launch, keep the cp.async pipeline
──────────────────────────────────────────────────────────────────────

The cp.async + SMEM ping-pong win comes from issuing one async slot
prefetch (17 × 16 B = 272 B) into SMEM BEFORE the FWHT decode starts,
hiding the slot-read latency under the FWHT compute. This win is
independent of the FWHT lane count — both 128- and 512-thread launches
benefit identically from the LDG-to-cp.async swap, because the cp.async
group is a CTA-wide commit/wait barrier.

iter9 fix: revert ``kBlockThreads`` to 512 (= kLatentDim), one element
per thread. Each lane decodes 1 latent element exactly as iter3, runs
ONE ``fwht_512_swizzled`` pass per slot, and writes one FP8 byte per
slot. The cp.async prefetch (lanes 0..16 issue the 17 × 16 B transfer;
lanes 17..511 idle on the issue but participate in the CTA-wide
commit/wait_group) survives unchanged. The rope tile (lanes 0..15)
survives unchanged.

The iter8 comment "the block size matches the CuTe DSL producer-warp
idiom: 128 threads = 1 warpgroup" is forward-looking for the iter9
SECONDARY (in-cubin CUTLASS graft via the flashinfer cute_dsl
template, 2-3 days). For the PRIMARY (standalone roll-out behind
``SGLANG_HIGGS_DSA_INLINE_PRODUCER``), the 512-thread launch is
correct and necessary. The SECONDARY graft will need a different
latent-decode path: the in-cubin producer warp processes N slots per
CTA via the depth-2 SMEM ping-pong, but the 512-vs-128 question is
moot because the in-cubin path emits into the cubin's SMEM, not gmem,
and the FWHT inside the cubin runs on the cubin's CTA size, not the
producer's.

──────────────────────────────────────────────────────────────────────
3. Validation plan
──────────────────────────────────────────────────────────────────────

  (a) Standalone correctness — extend
      ``benchmark/kernels/bench_higgs_inline_sparse_mla_decode_iter8.py``
      with a ``--correctness`` mode that compares both kernels' FP8
      outputs slot-by-slot. Acceptance: max_diff <= 1 FP8 lsb (ideally
      0) at production shapes B={64, 128, 256} K={1024, 2048}.

  (b) Standalone latency — same microbench, confirm the +20% kernel
      speedup is preserved at all shapes after the fix.

  (c) Wire-in — opt-in at ``get_higgs_selected_kv_buffer`` callsite in
      ``memory_pool.py:3737``. Gate with
      ``envs.SGLANG_HIGGS_DSA_INLINE_PRODUCER`` (default False). When
      the env is True AND ``fp8_layout`` is True, route to
      ``higgs_inline_sparse_mla_produce_fp8`` instead of
      ``dequantize_higgs_dense_2bit_page_table_fp8``.

  (d) PTXAS check —
      ``nvcc -arch=sm_100a -ptx -Xptxas -v`` for register/SMEM/stack
      footprint. Acceptance: no spills, SMEM <= 16 KiB (one slot-buf
      depth-2 = 544 B + fwht_buf 2 KiB + alignment = ~2.6 KiB; well
      under the 100 KiB/CTA budget).

──────────────────────────────────────────────────────────────────────
4. Scope notes
──────────────────────────────────────────────────────────────────────

This recon is the iter9 PRIMARY. It does NOT include:
  - The vendored flashinfer cute_dsl template + in-cubin graft (iter9
    SECONDARY, 2-3 days; +5-7 ms architectural ceiling).
  - IKP instrumentation. The fix mechanism is mechanically deducible
    from the iter3 kernel — no per-instruction profiling needed to
    validate the correctness claim. IKP is queued for the SECONDARY
    when the cp.async-vs-FWHT overlap question matters.
  - Multi-slot CTA loop (the iter8 scaffold processes 1 slot per CTA;
    same in iter9 PRIMARY — the multi-slot streaming is a
    SECONDARY-only concern when the in-cubin producer warp services
    N K-tile slots per launch).

Commit map (planned):
  (this commit) — recon checkpoint (recon doc)
  next         — PRIMARY fix (kBlockThreads=512, latent bit-exact)
  next         — wire-in behind SGLANG_HIGGS_DSA_INLINE_PRODUCER
  next         — close-out (microbench + bit-identity + PTXAS + pod
                 end-to-end if env var flip is safe)

──────────────────────────────────────────────────────────────────────
