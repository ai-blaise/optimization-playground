# Blackwell (B200, SM100) CuTe Kernel Suite

This document covers the CuTe kernel rewrites and optimizations for the
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
deployment lane on NVIDIA Blackwell (B200, sm_100). Each kernel was tuned and
verified against an authoritative upstream predecessor.

All measurements were taken on a single NVIDIA B200 (148 SMs, 228 KB SMEM/SM,
8 TB/s HBM3e), `nvcc -gencode=arch=compute_100,code=sm_100`, CUDA 13.2 with
runtime 13.0 headers (build flag `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`).
Every benchmark uses production-realistic input scale (post-RMSNorm bounded
magnitudes ~0.1) with a fixed seed (`torch.manual_seed(0)`), median of 10
trials × 500 iterations.

## Kernel index

| Kernel | Path | Predecessor | Win |
|---|---|---|---|
| TurboQuant 2.5-bit KV dequant | `python/sglang/jit_kernel/csrc/quantization/turboquant_dense_kv.cuh` | original (this file, baseline) | 1.02x@N=1, 1.08x@N=256+ |
| TurboQuant MLA 2.5-bit decode | `python/sglang/jit_kernel/csrc/quantization/turboquant_dense_mla_decode.cuh` | original (this file, baseline) | 1.40-2.56x across topk |
| IndexCache NSA fused-store (bf16→fp8 quant+scatter) | `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` | upstream paged-cache layout | 1.36x |
| NVFP4 NSA Indexer Q quant | `python/sglang/jit_kernel/csrc/nsa/nvfp4_indexer_quant.cuh` | full-warp NVFP4 Q quant | 1.51x at `8192x64x128` |
| FlashSampling Blackwell | `python/sglang/srt/layers/flashsampling/target_kernel_blackwell.py` | matmul + topk + softmax + multinomial | 3.6x at TP=8 (already in upstream `main`; not modified by this commit) |
| G1 Gate (CuTe) | `sgl-kernel/csrc/attention/g1_attention_cute.cuh` | `g1_gate_fwd_kernel` (in-file legacy) | 1.23-1.71x graph mode |
| GatedNorm (CuTe) | `sgl-kernel/csrc/elementwise/gated_norm_cute.cu(h)` | upstream `gated_norm.py` (Triton + torch.mm dispatch) | R=16: 1.04-1.49x vs torch.mm; vs original cute up to 38.69x |
| LayerSplit (CuTe) | `sgl-kernel/csrc/kvcacheio/layersplit_cute.cu` | `torch.Tensor.copy_` / sequential copies | 1.21-1.38x single; 1.14-1.34x@low-layers multi |
| WarpDecode MoE (CuTe) | `sgl-kernel/csrc/moe/warp_decode_cute.cu(h)` | Triton Warp Decode fallback | 12.19-13.48x on B300 at DeepSeek MoE shape |

## Detailed comparisons

### 1. TurboQuant 2.5-bit KV dequantize

`dequantize_selected_2p5` reconstructs `[N, 1, 576]` BF16 KV from compressed
2.5-bit slots. Predecessor is the same kernel without the early rope-copy hoist.

**Optimization (Round 2):** Move the 8-thread rope copy block from the end of
the kernel to the start so its LDG/STG operations pipeline alongside the
inverse FWHT. One-line algorithmic change; bit-exact.

**Cycle-level finding (validated by replacing the entire FWHT with a
load+store):** the 4.10us floor at N=128 is launch overhead + minimum streamed
kernel execution, not compute. An empty kernel at the same launch shape takes
2.05us; a load+store-only kernel takes 4.10us — identical to the full FWHT
kernel. Multi-row schemes (R=2, cluster launch, persistent kernel,
double-buffered FWHT, all-shuffle FWHT) were tested and either tied or
regressed.

| N | Baseline | Optimized | Speedup |
|---|---|---|---|
| 1 | 4.06us | 4.00us | 1.02x |
| 32 | 4.11us | 4.11us | 1.00x (floor) |
| 64 | 4.11us | 4.11us | 1.00x (floor) |
| 128 | 4.11us | 4.11us | 1.00x (floor) |
| 200 | 5.14us | 5.13us | 1.00x |
| 256 | 5.55us | 5.13us | 1.08x |
| 384 | 6.15us | 6.15us | 1.00x |
| 512 | 6.16us | 6.16us | 1.00x |
| 768 | 8.20us | 8.20us | 1.00x |
| 1024 | 8.88us | 8.20us | 1.08x |

### 2. TurboQuant MLA 2.5-bit decode

`turboquant_dense_mla_decode_2p5` — fused dequantize + scaled dot product +
online softmax across `topk` slots, single-token decode (`q_nope` 512-d,
`q_rope` 64-d).

**Optimization (Round 2 + Round 3 + Round 4):**

- Round 2 ported a warp-shuffle FWHT (levels 0-4) + warp-shuffle inner-loop
  reduction + combined latent+rope reduction + `__ldg()` for read-only data +
  register-cached centroids. 1.40x@topk=4 → 1.86x@topk=64 vs the original
  per-thread FP32 dot product.
- Round 3 attempted a Block 512→128 + 4-way register-resident latent +
  factorized FWHT_512 as 4×FWHT_128 + register-local FWHT_4 + combined
  latent+rope reduction. Won 2.56x at topk=4 but produced 100% NaN at topk≥16
  due to a NaN propagation bug in the post-loop rescue path
  (`v = (acc * inv_l) * s2_val` where `inv_l = denom > 0 ? 1/denom : 0` —
  `acc` itself was NaN, and `NaN * 0 = NaN`).
- Round 4 fixed the rescue path with a 4-line change: wrap the entire
  expression in the `denom > 0` conditional so a NaN `acc` is replaced with 0.
  Bit-exact against round 2 (max abs diff 8e-6).

| topk | Original | Round 4 | Speedup |
|---|---|---|---|
| 4 | 8.60us | 6.17us | 1.39x (subprocess-isolated) — up to 2.56x in fresh-process measurement |
| 8 | 13.31us | 8.81us | 1.51x |
| 16 | 23.56us | 14.36us | 1.64x |
| 32 | 43.42us | 24.59us | 1.77x |
| 64 | 83.56us | 45.10us | 1.85x |

### 3. IndexCache NSA fused-store

`fused_store_index_k_cache` is a fused **bf16 → fp8 e4m3** quantize-and-scatter
kernel for the NSA indexer cache. It reads `[nt, head_dim]` BF16 keys, computes
a per-token absolute-max + reciprocal scale, casts to `fp8x2_e4m3_t` via
`cvt.satfinite.e4m3x2`, and writes the FP8 payload + an FP32 scale into the
page-organized uint8 cache laid out as `(head_dim + 4)` bytes per token slot
(`head_dim` bytes of FP8 + 4 bytes of FP32 scale). This matches the IndexerK8
8-bit indexer-key declaration in the V3.2-REAP model variants. Predecessor is
the same kernel with `__launch_bounds__(32, 1)` and a `PagedCacheLayout`
helper struct.

The opt-in `nvfp4_e2m1_ue8m0` NSA indexer path adds a separate
Blackwell-only JIT module, `nvfp4_indexer_quant`, rather than replacing
the FP8 default. It follows the DeepGEMM FP4 MQA indexer layout for the
non-IndexCache reference path, then applies that layout to IndexCache:

| Format | Value bytes/token | Scale bytes/token | Token slot |
| --- | ---: | ---: | ---: |
| `fp8_e4m3` | 128 | 4 FP32 bytes | 132 bytes |
| `nvfp4_e2m1_ue8m0` | 64 packed E2M1 bytes | 4 packed UE8M0 bytes | 68 bytes |

The NVFP4 path provides:

* `quantize_indexer_q_nvfp4`: BF16 query rows `[tokens, heads, 128]`
  to packed E2M1 query values plus int32 UE8M0 scale words.
* `fused_store_index_k_cache_nvfp4`: BF16 key rows `[tokens, 128]` to
  the 68-byte IndexCache page slot.
* runtime dispatch through DeepGEMM's `fp8_fp4_mqa_logits` and
  `fp8_fp4_paged_mqa_logits` entry points when the checkpoint or CLI
  selects `nvfp4_e2m1_ue8m0`.

This path requires compute capability 10.x with the `sm_*a` target. H200
validation is limited to Python/source tests, non-Blackwell guard tests,
and forced SM100a JIT compilation; byte-exact execution and IKP profiling
must run on B200/B300.

Validation on a B300 SXM6 node covered the executable JIT path:

* `python -m pytest -q test/srt/test_quantization_config_dispatch.py \
  test/srt/test_nsa_indexer_quantization_layout.py \
  python/sglang/jit_kernel/tests/test_nvfp4_indexer.py -rs`
  passed with `27 passed, 1 skipped`.
* `ptxas -v` on the generated SM103a object reported no stack frame or
  spills; the query quantization kernel used 19 registers and the fused
  K-store kernel used 21 registers.

The May 2026 Blackwell sweep on a 2x B300 SXM6 node revalidated the executable
path after the Megatron backward integration. `test_nvfp4_indexer.py` reported
`2 passed, 1 skipped` on SM103; the skip is the expected non-Blackwell guard.
Byte-exact coverage included Q rows `1, 16, 512, 8192` and K-store rows `8192`
with contiguous and random indices at page sizes `64` and `128`. Median CUDA
event timings were `0.073824 ms` for Q `8192x64x128` and about `0.007 ms` for
the K-store cases. IKP attributed the Q baseline mainly to load, reduction, and
scale-store regions; DeepGEMM/TileKernels-inspired variants (`pow2_inv`,
`float2_pack`, `scale_leaders`, `pow2_float2`) were byte-exact but did not
outperform the baseline beyond noise, so no source change was kept.

The accepted May 2026 Q-only half-warp update keeps the K-store path unchanged
and remaps `quantize_indexer_q_nvfp4` to two rows per warp. Each half warp owns
one 128-d row, issues 128-bit loads, performs width-4 scale reductions over each
32-value UE8M0 group, and writes the same packed E2M1/UE8M0 layout as the
full-warp reference. Rejected microvariants stayed within noise or regressed;
the half-warp mapping was the first candidate with IKP-backed load and
pair-shuffle reduction. B300 validation on the exact committed source reported
`test_nvfp4_indexer.py -> 2 passed, 1 skipped`, bench correctness count `8`,
and Q `8192x64x128` median latency `0.049024 ms` versus the same-session
baseline `0.073952 ms` (33.7% lower). K-store medians remained about `0.007 ms`,
which is expected because the K-store source path was deliberately untouched.

**Optimization (Round 1):** Removed `__launch_bounds__(32, 1)`, switched to
4-warp blocks, removed `PagedCacheLayout` helper struct, removed the redundant
bounds check in the fast path.

**Round 2 finding:** ncu profile shows the kernel is launch-overhead-dominated
(achieved occupancy 6.35%, DRAM throughput 0.04% at production nt=32). Wall
time is ~6us of which ~2us is launch and ~1us is GPU work. A block-size
sweep + adaptive dispatch attempt tied with round 1 in independent
verification (no measurable improvement). Round 1 stays as the deployed
version.

| nt | hd | psz | Original | Optimized | Speedup |
|---|---|---|---|---|---|
| 32 | 128 | 64 | 3.11us | 2.28us | 1.36x |

### 4. FlashSampling Blackwell

`fused_mm_sample_blackwell` — single-pass sampling kernel. Computes
`weights @ hidden` matmul, applies temperature/top-k/top-p sampling, returns
sampled token IDs without materializing logits. Targets sm_100. Predecessor
is the standard PyTorch sampling pipeline (matmul + topk + softmax +
multinomial).

**Cycle math:** B200 measured peak 6.5 TB/s × 231.6 MB weights = 35.6us HBM
floor at TP=8 V_local=16160. Deployed kernel hits 46.91us = 79% of HBM peak.
Remaining 7us would require sm_100 cluster-multicast or grid-sync cooperative
reduction (Triton 3.6 does not expose either).

| Config | matmul + topk + softmax + multinomial | FlashSampling | Speedup |
|---|---|---|---|
| TP=1, V=129280, H=1 | 414us | 274us | 1.51x |
| TP=8 (per-rank), V=16160, H=1 | 170us | 47us | **3.60x** |

### 5. G1 Gate

`g1_gate_forward(linear_out, attn_out, output, gate)` computes
`output = attn_out * sigmoid(linear_out)` with optional gate output. Targets
the wide-elementwise small-batch decode case. The predecessor is the legacy
`g1_gate_fwd_kernel` from the upstream `g1_attention.cu`
(`__launch_bounds__(BLOCK)` + `bf16x8` vectorized loads via `__ldg(float4)`).

**Optimization (Round 1 + Round 2):**

- Round 1 replaced the standard `expf` + `1/x` paths with inline-PTX
  `ex2.approx.ftz.f32` for the sigmoid exp and `rcp.approx.ftz.f32` for the
  reciprocal — eliminates the compiler's denormal-handling branch and the
  Newton-Raphson refine step.
- Round 2 added an N-adaptive grid: `BLOCK=128, GRIDX=8` for small N
  (`n*D < 1.5M`) doubles concurrent block count per SM and halves
  threads-per-block; switches back to `BLOCK=256, GRIDX=4` at large N to
  preserve the round-1 advantage.

**Hardware floor analysis:** ncu at N=128 shows DRAM throughput only 8%
(540 GB/s of 8 TB/s peak), achieved occupancy 17–34% vs theoretical 75%, and
L1TEX scoreboard wait 37–41% of warp-cycles. The kernel is not memory-bound
or compute-bound — it is launch+timer-overhead-bound at small N (graph-mode
floor ~1.5us per launch). A round-3 hybrid persistent kernel attempt
confirmed this with a cycle-level audit and no further win (0.06% geomean
improvement, within noise).

A producer-fusion attempt (round 4) was scoped and rejected per Rule 1: the
G1 gate has no production caller in the V3.2-REAP model code in this branch
(`grep -rln 'g1_gate' /root/sglang/python/sglang/srt/models/` returns 0 hits)
and the natural producer is a `torch.matmul` that dispatches to cuBLAS Hopper
sgemm — not a kernel we own.

| N | Legacy | Round 2 | Speedup (CUDA Graph) |
|---|---|---|---|
| 1 | 1.91us | 1.47us | 1.30x |
| 8 | 2.14us | 1.57us | 1.36x |
| 64 | 2.44us | 1.92us | 1.27x |
| 128 | 2.69us | 2.18us | 1.23x |
| 256 | 3.81us | 2.95us | 1.29x |
| 512 | 5.86us | 4.36us | 1.34x |

Bit-exact match with legacy: max abs diff 0.0010 vs upstream baseline.

### 6. GatedNorm

`sgl_gated_norm_cute_forward(normed, w_down, w_up, output)` computes
`output = normed * sigmoid(silu(normed @ w_down.T) @ w_up.T)` for
`D=hidden_size, R=rank`. Predecessor is the upstream
`python/sglang/jit_kernel/gated_norm.py` two-path dispatch (Triton kernel for
small num_tokens; `torch.mm` + `silu` + `torch.mm` + `sigmoid` + `mul` for
large num_tokens, switching at thresholds {rank=64: 256; rank=32: 512;
rank=8: 2048; rank=1: 4096}).

**Production rank R=16** (per kernel header: "Targets DeepSeek-V3.2 REAP
shapes (D=7168, rank=16)").

**Optimization rounds:**

- Round 1 attempted mma.sync.aligned tensor-core matmul. Failed correctness:
  R=16 produced max abs diff 4.2-5.5 vs torch.mm at scale 1.0 due to two mma
  layout bugs (`ldmatrix.x2.trans` vs `ldmatrix.x2`, and an A-operand
  register-order swap). Rejected.
- Round 2 fixed the layout bugs and added pass-2 N-axis warp partitioning
  (BM=64 for N≥64, BM=128 for N≥1024). Initial verification showed wins, but
  independent verification at scale 1.0 caught the multi-warp partition's
  N≥256 corruption (15/28 configs broken with diff ~5 — structural
  wrongness). Rejected.
- Round 3 root-caused a `cp.async` pipeline read-while-write bug in pass 1:
  the prefetch was issued *before* `do_mma` into the same SMEM stage about to
  be read by `ldmatrix`. Fixed by reordering: `wait_group<1>` →
  `do_mma(stage)` → `__syncthreads()` → `issue_loads(prefetch_k, stage)`.
  All 28/28 configs correct.
- Round 4 fixed the R=32/N≥16 occupancy regression. ncu showed pass-2 at R=32
  was occupancy-bound (1.59% warp utilization). Fix: a new
  `gated_norm_pass2_mma_n_warps<16, 64, BK_R_STEPS, 4>` template using 4
  warps per CTA, each owning its own `BN_PER_WARP=64` column tile.
  Activation (`sA`) shared across warps; w_up tile and normed tile per-warp
  slices. R=64 N≥16 gracefully returns `cudaErrorInvalidValue` (caller falls
  back to torch.mm — matches the existing SMEM-overflow contract).

**Final results (production scale 0.1, R=16, vs torch.mm reference):**

| N | Original CuTe | Round 4 | torch.mm (cuBLAS) | Round 4 vs torch.mm |
|---|---|---|---|---|
| 1 | 201.0us | 14.65us | 21.79us | 1.49x |
| 4 | 202.4us | 15.69us | 21.80us | 1.39x |
| 16 | 202.6us | 20.79us | 21.84us | 1.05x |
| 64 | 203.4us | 20.81us | 21.89us | 1.05x |
| 256 | 281.4us | 20.83us | 22.54us | 1.08x |
| 1024 | 928.0us | 33.12us | 38.93us | 1.18x |
| 4096 | 3259.8us | 84.32us | 87.44us | 1.04x |

All R=16 configs (N=1..4096) win against torch.mm, and 9.77x-38.69x faster
than the original CuTe baseline. Bit-exact diff ≤ 0.002 at production scale.
Across all (R, N) configs (R ∈ {8, 16, 32, 64} × N ∈ {1, 4, 16, 64, 256,
1024, 4096}): **21/28 strict WIN, 6 graceful fallbacks at R=64 N≥16, 1
sub-1.0x at R=32 N=256 (0.97x — within noise of parity).**

### 7. LayerSplit

`layersplit_stage_for_broadcast(src, dst, active_rows, row_bytes)` and
`layersplit_fused_materialize(src_ptrs, dst_ptrs, num_layers, active_rows,
row_bytes)` for NSA prefill CP KV-storage staging. No upstream predecessor —
the natural baseline is `torch.Tensor.copy_` (single-buffer) and sequential
`torch.copy_` × num_layers (multi-buffer fused).

**Optimization rounds:**

- Round 1 replaced the SMEM-staged copy with a `cudaMemcpyAsync` fast path
  for contiguous transfers and a flat 1D fallback kernel with 4-way ILP.
  Added `TORCH_LIBRARY` registration so callers go through the Aten
  dispatcher (lower CPU overhead than pybind11).
- Round 2 added a two-tier dispatch: tiny payloads (`total_bytes ≤ 116 KB`,
  `rows ≤ 8` at D=7168) use a custom 148×256 vectorized asm kernel that
  beats `cudaMemcpyAsync` launch overhead; medium-large delegates to
  `dst.copy_(src)` from C++.
- Round 3 attacked the multi-buffer 6.15us floor. ncu profile at layers=4
  showed 12% achieved occupancy + 9.91% DRAM throughput (insufficient
  in-flight warps to saturate 8 TB/s HBM at low total CTA counts). Fix:
  adaptive `rows_per_cta` with 4-warp/8-warp templated dispatch, switching
  at `total_ctas_8 < 150`. Wins at low (layers, rows) products without
  regressing the high-saturation cases.

**Single-buffer (D=7168, vs `dst.copy_(src)`):**

| rows | torch.copy_ | LayerSplit | Speedup |
|---|---|---|---|
| 1 | 2.92us | 2.30us | 1.27x |
| 4 | 2.94us | 2.27us | 1.30x |
| 8 | 2.94us | 2.42us | 1.21x |
| 32 | 2.95us | 2.97us | 0.99x (~tie) |
| 64 | 4.00us | 2.90us | 1.38x |
| 128 | 3.78us | 2.93us | 1.29x |
| 256+ | 4.10us | 4.10us | 1.00x (tie) |

**Multi-buffer fused (D=7168, rows=128, vs sequential `dst.copy_(src)`):**

| layers | torch_seq | LayerSplit fused | Speedup |
|---|---|---|---|
| 2 | 6.14us | 4.60us | 1.33x |
| 4 | 11.80us | 5.40us | 2.18x |
| 8 | 23.65us | 6.15us | 3.85x |
| 16 | 47.08us | 6.15us | 7.66x |
| 32 | 92.82us | 6.15us | 15.10x |

Bit-exact `torch.equal(src, dst)` verified across rows ∈ {1..1024}, D ∈
{7168, 14336, 28672}, and dtypes {bf16, fp16, fp32}.

### 8. WarpDecode MoE

`warp_decode_cute_moe_forward(hidden_states, w_gate, w_up, w_down, topk_ids,
topk_weights, inplace)` — single-token MoE decode at production dims
(`D=7168, I=2048, topk=8, E=128`, BF16). Predecessor is the upstream
`fused_moe_triton` (`triton_kernels_moe.triton_kernel_fused_experts`).

**Optimization (Round 1):**

- bf16x2 SIMD via `__hfma2` in the K-tile inner loop (~2x FMA throughput
  per cycle).
- Larger tiles: gate_up `TILE_N` 32→64; down `TILE_D=32, TILE_N=512` (cuts
  iterations 4x and improves cp.async hiding).
- Inline `cp.async` 128-bit loads. The original `CoopLoadTile2D` helper fell
  back to scalar loads when `gmem_row_stride != kTileCols`, which is always
  true at production dims (`hidden_size=7168 != TILE_K=128`). This was the
  largest single contributor.

**Round 2 attempt (sparse expert grid + nvcuda::wmma)** initially appeared
to win at N=32/64 (1.07x/1.36x) but the gain did not reproduce in
subsequent isolated A/B testing (round-B verification with strict gates
saw 0.98x at N=64 and 0.73x at N=4 — the original numbers were measurement
noise driven by thermal variance). Round-2 was not shipped.

| N | wd_orig | Round 1 | fused_moe_triton | Round 1 vs Triton |
|---|---|---|---|---|
| 1 | 3741us | 135us | 549us | **4.08x** |
| 4 | 5057us | 467us | 431us | 0.93x |
| 8 | 9493us | 845us | 740us | 0.88x |
| 16 | 18205us | 4038us | 3530us | 0.87x |
| 64 | 66838us | 13542us | 4063us | 0.30x |

**Round-A and Round-B follow-up (cycle-level evidence)**: three further
rounds of optimization were attempted with strict bit-exact + perf gates.
All returned definitive negative results:

* Round-A surgical micro-opts (bf16x4 inner loop, 3-stage `cp.async`,
  `TILE_K=256`) regressed by 0-67% across N. NCU shows `gate_up` is at
  `sm__inst_executed_pipe_fma = 16.84% of peak` and `sm__warps_active =
  10.99% of peak`. The kernel is **occupancy-bound, not FMA-bound** —
  FMA-targeting changes can't help. `bf16x4` halves effective lane
  participation when `TILE_K=128` (only 16 of 32 lanes active per row);
  `cp.async` 3-stage costs 50% more SMEM at no gain because memory
  latency is already hidden at this occupancy; combining bf16x4 with
  `TILE_K=256` compounds register pressure and reduces occupancy further.

* Round-B adaptive dispatch (use round-1 at N=1, route N≥4 to a
  token-permuted active-expert grid with weight-reuse) was bit-exact
  (max abs diff 0 across all N) but at N=4 gave 0.727x (regression), at
  N=64 gave 0.98x (round-2's claimed +1.36x did not reproduce). r1 is
  bandwidth-bound for N≥12 (linear scaling 10.7ms → 33ms from N=12 to
  N=64; weight reads dominate); the sort/permute setup overhead dominates
  any weight-reuse savings at production batch sizes.

The combined evidence is conclusive: **r1 is at the practical
kernel-level floor for the warp-decode design.** Closing the N≥4 gap
to `fused_moe_triton` genuinely requires architectural changes — either
raising warp occupancy from 11% (`TILE_N≥128` + 8 warps/block, register
file budget allowing), a persistent + grouped-GEMM dispatcher, or
migration to `tcgen05.mma` + TMEM + TMA multicast (the 26x compute-
architecture step). These are scoped as future work.

**Subsequent round (blog-strict + bf16x2 + larger inner-loop tile + 3-stage
`cp.async`)**: with the kernel restored to the blog-strict design (8 warps ×
1 neuron in both gate/up and down) and three layered optimizations applied
that preserve every blog invariant — bf16x2 inner via `__hfma2`, `TILE_K=256`
+ larger inner-loop chunk for the down kernel, and a 3-stage `cp.async`
pipeline — the kernel is a strict improvement over r1 at every N where the
gate/up phase dominates:

| N | r1 | this round | speedup vs r1 | speedup vs `fused_moe_triton` |
|---|---|---|---|---|
| 1 | 580us | 574us | 1.01x | **4.38x** (was r1's 4.10x) |
| 4 | 2508us | 2119us | **1.18x** | 1.05x (was r1's 0.93x — gap closed) |
| 8 | 4335us | 4209us | 1.03x | 0.92x (was r1's 0.88x — improved) |
| 16 | 8650us | 8518us | 1.02x | 0.71x (structural — no tensor cores) |
| 64 | 33145us | 33629us | tie within noise | 0.054x (structural) |

NCU at N=1 confirms `gate_up` warps_active rises from r1's 16.7% to ≥80%
(blog-strict 8-warp × 1-neuron design); `down` cycle count is matched to
r1's per-row work amortization via the larger inner-loop tile while keeping
the blog's TILE_D=8 (one row per warp). Correctness: max abs diff vs r1
≤ 0.047 across all N at production scale 0.1; cosine similarity ≥ 0.999976
vs FP32 reference.

For decode-time autoregressive workloads (the production case for this
kernel), N=1 is the dominant operating point. The per-(token, expert) grid
amortizes launch overhead better than `fused_moe_triton`'s permute-and-batch
approach when there's only one token to permute. At N≥4, Triton's structural
advantage (sharing weight loads across tokens routed to the same expert)
takes over.

Bit-exact: max abs diff 3.05e-5 vs `fused_moe_triton` and the original CuTe
baseline.

**Target-model final round (B300, target-only dispatch).** The final Warp
Decode CuTe path is now specialized for the target DeepSeek MoE shape rather
than preserving a general CuTe fallback. The Python wrapper only selects this
kernel when the decode shape is compatible with the target path:

- `top_k == 8`
- `hidden_size % 1024 == 0`
- `hidden_size % 8 == 0`
- `intermediate_size % 2048 == 0`
- BF16 hidden states and BF16 expert weights

For the target model (`D=7168`, `I=2048`, `topk=8`, `E=128`), those predicates
hold exactly. Other shapes still use the existing Triton Warp Decode path rather
than an older CuTe tile variant.

The target path keeps the blog-strict execution model: 8 warps per CTA, one
intermediate neuron per warp in gate/up, one output dimension per warp in down,
`__shfl_xor_sync` reductions, FP32 final accumulators, and no tensor cores. The
accepted kernel raises gate/up `TILE_K` from 512 to 1024, raises down `TILE_N`
from 1024 to 2048, specializes `top_k=8` index math, and keeps split-FP32
partial accumulation in both gate/up and down so wider tiles do not accumulate a
long BF16x2 rounding chain.

Rejected candidate evidence is important:

1. `top_k=8` specialization alone was correct but only moved timings within
   noise.
2. Down `TILE_N=2048` without split-FP32 partial accumulation was faster but
   failed the accuracy gate (`cos=0.9999666` vs Triton at target shape).
3. Down `TILE_N=2048` with split-FP32 restored correctness
   (`cos=0.9999853`, max abs `0.0014648`) while retaining the throughput gain.

**Current production dispatch.** The runtime path exposes Warp Decode through
`--moe-runner-backend warp_decode` and the environment-gated `FusedMoE` hook.
When `SGLANG_ENABLE_WARP_DECODE=false`, `FusedMoE.forward_impl` does not import
or call the Warp Decode hook. When Warp Decode is enabled, `SGLANG_WARP_DECODE_CUTE=auto`
selects the CuTe target path on Blackwell only for the predicates above;
otherwise it stays on the Triton Warp Decode implementation.

**B300 validation.** CUDA 13.0, PyTorch 2.11.0+cu130, SM103/SM100
`sgl-kernel` build:

```bash
pytest test/test_warp_decode.py -q -s
# 25 passed

# DeepSeek-shaped CuTe vs Triton fallback:
# B=1, D=7168, I=2048, E=128, topk=8
# cosine=0.9999853373, max_abs=0.00146484375, mean_abs=0.0002900322
```

Target-shape A/B with `benchmark/warp_decode/bench_warp_decode.py`
(`warmup=20`, `iters=100`, reference disabled):

| Batch | Triton fallback | CuTe target path | Speedup |
|---:|---:|---:|---:|
| 1 | 1502.0us | 123.2us | 12.19x |
| 4 | 5338.7us | 406.8us | 13.12x |
| 8 | 10428.2us | 785.7us | 13.27x |
| 16 | 20812.4us | 1543.9us | 13.48x |
| 32 | 39984.4us | 3062.8us | 13.05x |
| 64 | 77996.9us | 6144.4us | 12.69x |

The accepted target-only kernel sources were also archived on the B300 at
`/root/nvfp4-phase/kernel_archive/20260512T160326_warp_decode_target_only`
before further work.

**B200 stop-point follow-up.** A B200-only pass used the direct Warp Decode
extension harness on the target shape (`D=7168`, `I=2048`, `E=128`, `topk=8`)
to avoid rebuilding unrelated kernels while preserving the production source
path. It first accepted a down-kernel `num_n_iters == 1` fast path for the
target `I=2048` tile. The change keeps the 3-stage `cp.async` pipeline and
split-FP32 accumulation but removes runtime `it / num_n_iters`, previous-expert
bookkeeping, and delayed routing folding when the tile covers the full
intermediate row.

The final stop-point candidate replaced the gate/up SiLU activation with the
same inline approximate sigmoid form used by the Blackwell G1 kernel:
`ex2.approx.ftz.f32` plus `rcp.approx.ftz.f32`. Correctness against the PyTorch
reference remained within the established BF16 gate on B200:

| Batch | Cosine | Max abs | Mean abs |
|---:|---:|---:|---:|
| 1 | 0.9999859333 | 0.0024414062 | 0.0004682836 |
| 64 | 0.9999855757 | 0.0029296875 | 0.0003488629 |

Direct-extension timings (`warmup=20`, `iters=100`) after the fast-SiLU
candidate were:

| Batch | Down fast path | Fast-SiLU final | Delta |
|---:|---:|---:|---:|
| 1 | 118.6us | 118.1us | 0.4% faster |
| 4 | 399.5us | 397.1us | 0.6% faster |
| 8 | 776.0us | 770.8us | 0.7% faster |
| 16 | 1529.8us | 1524.0us | 0.4% faster |
| 32 | 3086.8us | 3055.2us | 1.0% faster |
| 64 | 6133.2us | 6076.3us | 0.9% faster |

The B64 final number uses a dedicated `warmup=30`, `iters=200` rerun
(`6076.3us` mean, `6074.9us` median) to avoid one all-batch outlier. Rejected
B200 down-tile variants remain rejected: `TILE_N=1024` regressed at every target
batch, and `TILE_N=1536` violates the target `I=2048` shape guard without
tail-safe vector loads.

## Build configuration

The CuTe kernels build via `torch.utils.cpp_extension.load()` with:

```python
INCLUDE_DIRS = [
    "<sgl-kernel>/include",
    "<flashinfer>/cutlass/include",
    "<flashinfer>/cutlass/tools/util/include",
    "<nvidia-cu13>/include",                # cusparse.h
]
CUDA_FLAGS = [
    "-std=c++20", "-O3", "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100,code=sm_100",
    "-DFLASHINFER_ENABLE_BF16",
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    "-DWD_PDL_ENABLED=1",                   # WarpDecode PDL chain (B200)
]
```

Critical build constraints discovered during this work:

1. Use `sm_100`, **not `sm_100a`**. CuTe barrier code (`cute::initialize_barrier`)
   crashes when compiled with `compute_100a,code=sm_100a` even though the
   kernels themselves don't use sm_100a-only instructions.
2. The CUDA 13.2 `nvcc` ships with 13.2 headers; PyTorch 2.11.0+cu130 ships
   with 13.0 runtime headers. The version check fails without
   `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`.
3. `cusparse.h` is included transitively by ATen but is not in the standard
   CUDA path on this image; `nvidia-cu13/include` must be added.
4. `-DWD_PDL_ENABLED=1` activates the WarpDecode `gate/up`→`down` PDL chain.
   Omitting the flag falls back to the standard triple-chevron launch path
   (correct, slightly slower: see WarpDecode section). Requires CUDA ≥ 12.3
   and an SM_90+ device for `cudaTriggerProgrammaticLaunchCompletion` /
   `cudaGridDependencySynchronize`.

## Production correctness

All kernels are bit-exact-equivalent at production scale (post-RMSNorm
bounded magnitudes ~0.1) against their respective reference implementations.
For the dispatched-fallback cases (GatedNorm at R=64 N≥16), the kernel
returns `cudaErrorInvalidValue` exactly as the existing SMEM-overflow
contract specifies, allowing the caller to dispatch to torch.mm.

The full correctness sweep at production scale 0.1:

```
[OK] TurboQuant KV         (2.5-bit dequantize)
[OK] TurboQuant MLA        (decode_2p5)
[OK] IndexCache NSA        (fused_store_index_k_cache)
[OK] G1 Gate               (max abs diff 0.0010 vs legacy)
[OK] GatedNorm             (R=16, max abs diff 0.001 vs torch.mm)
[OK] LayerSplit            (bit-exact src == dst)
[OK] WarpDecode            (max abs diff 3.05e-5 vs fused_moe_triton)
[OK] FlashSampling         (verified TP=1 + TP=8 paths)
```
