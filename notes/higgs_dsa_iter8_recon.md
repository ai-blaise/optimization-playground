HIGGS dense 2-bit DSA iter8 — recon checkpoint
================================================

State on entry: origin/main at 8987cb270, local at 1e93c8810. iter7
close-out (9628e4aba) landed +0.06-1.5 ms TPOT (depth-4 + dual-stream)
on top of iter6's +0.66 ms. Cumulative iter3..iter7 = 4-5.6 ms closed
of the original ~12.7 ms HIGGS-vs-FP8 TPOT gap. Remaining gap ~7-9 ms
sits in the dequant→trtllm-gen HBM round-trip (302 MiB writes by the
sparse-materialize, then 302 MiB reads by trtllm-gen, per layer-step).

iter8 PRIMARY vector: in-cubin HIGGS dequant — scaffold a CUTLASS
sparse-MLA kernel via the flashinfer cute_dsl monolithic mla_decode
template. The TMA bulk-tile copy of the K-latent / K-rope tiles is
replaced by per-CTA cp.async of the 272 B HIGGS slot + inline
FWHT_128 / EDEN2-16 (or saw_scalar2) decode into the SMEM layout the
template's tcgen05.mma expects.

This recon documents the surface area, the existing primitives that
will be re-used wholesale, the surgical excision points in the
flashinfer template, and the scope cut for the iter8 ship.

──────────────────────────────────────────────────────────────────────
1. Surface area on the flashinfer side
──────────────────────────────────────────────────────────────────────

flashinfer monolithic mla decode source (host filesystem, NOT
container — the container path the iter7 close-out mentioned does
not exist on this VM):

  /home/spencer/.local/lib/python3.11/site-packages/flashinfer/
  cute_dsl/attention/monolithic/mla_decode_fp8.py     (3501 LOC)
  cute_dsl/attention/monolithic/mla_decode_fp16.py
  cute_dsl/attention/monolithic/mla_decode.py
  cute_dsl/attention/monolithic/mla_helpers.py        (304 LOC)
  cute_dsl/attention/monolithic/__init__.py

The TMA bulk-tile copy setup the iter7 close-out flagged at "lines
565-593" lives at L566-595 of mla_decode_fp8.py and binds three
``make_paged_tiled_tma_atom`` calls:

  L566-575  tma_atom_c_latent, tma_tensor_c_latent =
              self.make_paged_tiled_tma_atom(
                  tma_load_op, c_latent,
                  kc_smem_layout,
                  (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
                  qk_tiled_mma, is_k_load=True)

  L577-585  tma_atom_c_rope, tma_tensor_c_rope =
              self.make_paged_tiled_tma_atom(
                  tma_load_op, c_rope,
                  kc_rope_smem_layout,
                  (self.mma_qk_rope_tiler[1], self.mma_qk_rope_tiler[2]),
                  qk_tiled_mma, is_k_load=True)

  L588-595  tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose =
              self.make_paged_tiled_tma_atom(
                  tma_load_op, c_latent_transpose,
                  vc_smem_layout,
                  (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                  pv_tiled_mma, is_k_load=False)

The c_latent / c_rope binds drive the Q·K^T phase, the
c_latent_transpose binds drives the P·V phase. Both phases read the
SAME 512-D latent (the FWHT-rotated K) and the SAME 64-D rope. In the
trtllm-gen FP8 path that the iter3 commit (dec12706b) already wires
into, the dense materialize writes K^T = K (same buffer, different
view). In an in-cubin path, V can either (a) re-use the K SMEM via a
PV-stage SMEM transpose, or (b) stage a second SMEM tile.

The iter8 scaffold takes the path-of-least-surgery: rebuild the K
SMEM tiles (kc_smem + kc_rope_smem) inline from HIGGS slots, but
keep the V-load via flashinfer's existing TMA atom by aliasing the
freshly-decoded K SMEM as V (HIGGS encodes the SAME 512-D latent
for both Q·K and P·V; the c_latent_transpose path simply re-reads
the latent under a transposed CTA tiling).

──────────────────────────────────────────────────────────────────────
2. Existing HIGGS device primitives we re-use wholesale
──────────────────────────────────────────────────────────────────────

python/sglang/jit_kernel/csrc/quantization/higgs_dense_2bit_mla_decode.cuh
(1339 LOC). Inline-decode building blocks the iter8 scaffold composes:

  L120  kLatentDim = 512, kRopeDim = 64, kPairDim = 2
  L123  kCodebookSize = 16  (EDEN2-16)
  L125  kPackedBytes  = 128 (256 packed indices)
  L127  kNormBytes    = 2   (fp16 per-row scale)
  L131  kPayloadBytes = 258 (128 + 2 + 128 rope)
  L137  kSlotBytes    = 272 (16-byte aligned for cp.async.16)
  L138  kSlotPadBytes = 14  (rule-7 fingerprint)
  L141  kInvSqrtLatentDim = 1/sqrt(512)

  L182  fwht_register_top2(v0..v3)              — FWHT_4 register
  L195  fwht_lane_levels_under32(val, lane)     — FWHT_32 warp shuffle
  L207  fwht_128elem(val, tid, smem128)         — FWHT_128 warp+smem
  L223  higgs_unpack_indices(slot_gmem, tid, i0..i3)  — LDG path
  L254  saw_scalar2_value(code)                 — scalar2 lookup
  L263  saw_scalar2_unpack_pair_lanes(slot_gmem, tid, c0..c3)
  L280  higgs_unpack_indices_smem(slot_smem, tid, i0..i3) — SMEM path
  L313  saw_scalar2_unpack_pair_lanes_smem(slot_smem, tid, c0..c3)
  L340  higgs_cp_async_prefetch_slot(slot_smem, slot_gmem, tid)
        — 17 lanes × cp.async.ca.shared.global of 16 B = 272 B exactly
  L352  higgs_cp_async_commit()
  L358  higgs_cp_async_wait_group<N>()

Iter4 already proved the cp.async slot-prefetch + SMEM dequant path
in higgs_dense_2bit_mla_decode_stage1_split_kernel (L415). Iter8
re-uses these primitives verbatim — the new work is the SMEM layout
adapter that emits the kc_latent_smem_layout_for_tma shape the
flashinfer template's qk_tiled_mma consumes.

──────────────────────────────────────────────────────────────────────
3. K-cache shape mapping (DSV3.2-REAP-345B production)
──────────────────────────────────────────────────────────────────────

  B           = 128       (decode batch, per-rank)
  top_k       = 2048      (selected indices per (row, layer))
  n_heads     = 64        (model-side; rank-side after TP=8 = 8)
  head_dim    = 128       (MLA absorbed: 512 latent + 64 rope = 576)
  kv_lora_rank = 512      (latent dim)
  61 layers, index_topk_freq = 4

flashinfer template's KV-page abstraction expects a contiguous
``[num_pages, page_size, num_heads, head_dim]`` paged K-cache and
``block_tables[B, ceil(top_k / page_size)]`` selectors. In the
sparse-MLA case, top_k=2048 selected slots are indexed individually
(page_size=1 in the sparse-materialize view), so block_tables
becomes ``[B, top_k] = [128, 2048]`` and each entry is a flat slot id
into the HIGGS slab.

HIGGS K-cache global layout per rank (one layer):

  k_slab : uint8[num_slots, kSlotBytes]  = uint8[N, 272]
  N      = num_total_kv_tokens (≈ context_len × 32 in steady state)

The selected ``B × top_k`` slots are packed into the cubin's
register tiles via per-CTA cp.async + inline decode. No 302 MiB
sparse-materialize buffer.

The flashinfer template's SMEM layout the template binds to:

  kc_latent_smem_layout_for_tma : Layout((kc_tile_m, kc_tile_k, stages))
    where kc_tile_m × kc_tile_k = (e.g.) 64 × 64 fp8 = 4 KB per stage
  kc_rope_smem_layout_for_tma   : Layout((rc_tile_m, rc_tile_k, stages))
    where rc_tile_m × rc_tile_k = (e.g.) 64 × 64 fp8 = 4 KB per stage

The iter8 scaffold needs:

  iter8 inline-K SMEM:
    higgs_slot_smem[depth][kCTAPerSlot][kSlotBytes]
      ≈ 2 × 8 × 272 = 4352 B per stage (depth=2 ping-pong, 8 CTAs/slot)
    kc_latent_smem[stages][kc_tile_m × kc_tile_k]  (unchanged)
    kc_rope_smem  [stages][rc_tile_m × rc_tile_k]  (unchanged)

The decode hop writes the same kc_latent_smem / kc_rope_smem tiles the
flashinfer template's tcgen05.mma reads — the surgery is the producer,
not the consumer.

──────────────────────────────────────────────────────────────────────
4. Surgical excision plan for iter8
──────────────────────────────────────────────────────────────────────

Step 1 (this commit + next 1-2). Vendor a stripped copy of
mla_decode_fp8.py into the optimization-playground tree at:

  python/sglang/jit_kernel/higgs_inline_sparse_mla/
    mla_decode_higgs_inline.py        — vendored + edited
    higgs_inline_sparse_mla_decode.cuh — new producer (CUTLASS-friendly)
    __init__.py

Step 2. Excise the three TMA atom binds (mla_decode_fp8.py L566-595)
and replace with HIGGS slot-load + inline FWHT_128 + EDEN2-16/scalar2
decode producers that write into kc_smem / kc_rope_smem. The TMA atom
for V (c_latent_transpose) stays — V is the transposed view of the
same K SMEM, so its TMA atom now reads from SMEM (or we alias).

Step 3. Microbench. Existing
``benchmark/kernels/bench_higgs_trtllm_dsa_iter7.py`` provides the
harness pattern; iter8's variant is a 7th column (``inline_higgs``)
that runs the new kernel through the same 8-layer simulation. iter7
baseline is depth4_dual at 6471.60 us/8layer (Regime B) and
7601.90 us/8layer (Regime C).

Step 4. Correctness: bit-exact comparison against the existing
higgs_dense_2bit_mla_decode kernel's output on a single layer at
representative shape (B=128, n_heads=8, top_k=2048).

──────────────────────────────────────────────────────────────────────
5. Scope cut for iter8 (4-hour budget — scaffold only)
──────────────────────────────────────────────────────────────────────

IN scope for this iter:
  - Vendored mla_decode_higgs_inline.py with the three TMA atoms
    replaced by HIGGS inline-decode producer call sites (stubs
    acceptable; the kernel does not have to be measurably faster).
  - New higgs_inline_sparse_mla_decode.cuh that exposes the
    producer device function the vendored template calls into.
  - Microbench scaffold (iter8 bench script) running the new path
    in isolation against synthetic shapes.
  - Honest documentation of compilation + correctness state.

OUT of scope for this iter (iter9+):
  - Wiring into ``_forward_trtllm`` in dsa_backend.py.
  - Tuning the producer's cp.async issue width, SMEM ping-pong
    depth, or FWHT register-vs-SMEM split.
  - IKP (intra-kernel-profiler) instrumentation of the producer
    against the existing dequant path; IKP scaffold lands here but
    actual diagnosis defers to iter9.
  - V-tile producer (V re-uses K SMEM aliasing for this scaffold).

Projected TPOT savings ceiling on iter9-10 maturation:
  - Eliminate 302 MiB dequant write + 302 MiB trtllm-gen read =
    604 MiB HBM saving per layer at ~3.2 TB/s effective HBM-BW =>
    ~0.19 ms/layer => ~11.5 ms across 61 layers, capped by the
    cubin's compute-bound floor (~25.9 ms FP8 baseline). Realistic
    closure of the remaining 7-9 ms gap is achievable.

──────────────────────────────────────────────────────────────────────
6. IKP instrumentation plan (deferred to iter9 — clone scaffolded here)
──────────────────────────────────────────────────────────────────────

The iter8 PRIMARY mission is the kernel scaffold. IKP integration is
a 30-min add once the scaffold compiles:

  git clone https://github.com/yao-jz/intra-kernel-profiler \
    /home/spencergarnets/intra-kernel-profiler
  cd /home/spencergarnets/intra-kernel-profiler && pip install -e .

IKP gives per-instruction latency inside the cubin. The two ground
questions for iter9 are:

  Q1. Does the cp.async.ca slot-load latency hide behind the FWHT_128
      compute? (Expectation: yes — cp.async issues 17×16 B = 272 B
      per slot from L2 at ~80-120 cycles, FWHT_128 is ~80 cycles of
      reduce + shuffle, so they should overlap if we run depth-2.)

  Q2. Does the eliminated kc_latent_smem → tcgen05 path bypass let
      tcgen05.mma start ~1 stage sooner per K-tile? (Expectation:
      yes — the producer's commit_group becomes the consumer's
      mbarrier arrival, removing one bulk-tile TMA latency.)

──────────────────────────────────────────────────────────────────────
7. Iter9 next vectors
──────────────────────────────────────────────────────────────────────

  1. Wire the scaffold into _forward_trtllm via a new env flag
     (SGLANG_HIGGS_DSA_INLINE_CUTLASS) and measure end-to-end on a
     production decode pod.
  2. Tune SMEM ping-pong depth (2 vs 3 vs 4) under IKP.
  3. SECONDARY (if PRIMARY stalls): wait_event coalescing. iter7
     dual-stream records ~120 events/step (one per layer per
     stream); coalesce per 8 layers to ~16 events => -0.3-0.5 ms.
  4. TERTIARY (if PRIMARY stalls): cp.async with .L2::128B cache
     hint on the HIGGS slot load. Slots that re-appear across
     layers (top_k=2048 overlap is structural) stay in L2.

──────────────────────────────────────────────────────────────────────
Commit map for iter8 (target: 4-5 commits over 4 hours)
──────────────────────────────────────────────────────────────────────

  THIS COMMIT (recon checkpoint): notes/higgs_dsa_iter8_recon.md
    - Surface area, primitives, excision plan, scope cut.

  +1 (vendor): copy mla_decode_fp8.py to
    python/sglang/jit_kernel/higgs_inline_sparse_mla/
    mla_decode_higgs_inline.py with comment markers at the three
    TMA atom sites. No behaviour change yet.

  +2 (scaffold CUDA): higgs_inline_sparse_mla_decode.cuh exposing
    a ``produce_kc_tile_from_higgs_slots`` device entry the vendored
    template will call. Compiles standalone; correctness via the
    existing higgs_dense_2bit_mla_decode kernel comparison.

  +3 (microbench scaffold): bench script that runs the new path
    against the iter7 baseline at production shape. Reports honest
    numbers (faster, slower, or compile-only).

  +4 (close-out): honest scope summary, IKP install instructions,
    iter9 vectors.

──────────────────────────────────────────────────────────────────────
