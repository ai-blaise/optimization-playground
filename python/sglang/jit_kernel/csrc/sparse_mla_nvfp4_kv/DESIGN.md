# Sparse-MLA Decode with NVFP4 KV — native-UMMA design (v2)

Source: FlashMLA `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/` (Apache-2.0).
Target: drop-in replacement for the trtllm-gen sparse-MLA decode cubin, **native NVFP4 KV read** via SM_100 block-scaled tensor cores (no dequant materialize, no SMEM dequant pass).

## Architecture (corrected from v1)

Previous v1 scaffolding had a SMEM-dequant pass (FP4 → BF16 in SMEM, then BF16 UMMA). That's the suboptimal middle path. **v2 uses the native block-scaled UMMA**: K stays in packed FP4 form right through to the tensor cores; per-block E4M3 scales are passed as a UMMA scale operand.

## Format alignment with target model

`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` declares:
- `kv_cache_scheme.quant_method = "higgs_dense_2bit"` (calibrated default)
- `indexer_quantization.quant_method = "nvfp4_e2m1_ue8m0"` (NVFP4 with MXFP4-style ue8m0 scales)
- `spinquant_k_bits = 4, spinquant_v_bits = 4` (SpinQuant rotation prepared for 4-bit KV)

Per directive: use **NVFP4 (E4M3) scales** for the KV cache (not MXFP4 ue8m0). The SpinQuant rotation makes any uniform 4-bit format with per-block scales accurate; switching from HIGGS to NVFP4-E4M3 KV is a runtime override that overrides the model's declared `kv_cache_scheme`. Quality may be ~1-2% lower than HIGGS-calibrated until we add the NVFP4 calibration pass (aquakv-style follow-up).

## SMEM layout

| Buffer | FP8 baseline (FlashMLA) | Native NVFP4 UMMA (this design) |
|---|---|---|
| `Q` (BF16) | (H_Q/2) × D_Q | (H_Q/2) × D_Q |
| `K` (BF16 dequanted) | B_TOPK × (D_K/2) × 3 buf = 24 KB | **REMOVED** |
| `K_raw` (FP8 packed) | B_TOPK × (D_K/2) × 2 buf = 32 KB | **REMOVED** |
| `K` (FP4 packed, new) | — | B_TOPK × (D_K/2 nibbles) × 3 buf = **12 KB** (4-bit packed) |
| `K_scales` (E4M3) | — | B_TOPK × 32 × 3 buf = **6 KB** |
| `V_scales` (E4M3) | — | B_TOPK × 32 × 3 buf = **6 KB** |
| scales (E8M0, 7+1/token) | 4 × 4 buf = 1 KB | — |
| TOTAL K-side SMEM | ~57 KB | **~24 KB** (~58% reduction) |

## UMMA atom

Replaces `SM100_MMA_F16BF16_2x1SM_*` with `SM100_MMA_F8F6F4_BS_2x1SM_SS_NOELECT`:

```cpp
// FP8 baseline:
SM100_MMA_F16BF16_2x1SM_TS_NOELECT<bf16, bf16, float, H_Q, B_TOPK*2, K, K>

// NVFP4 native:
SM100_MMA_F8F6F4_BS_2x1SM_SS_NOELECT<
    fp8_e4m3,   // Q operand (FP8 from mla_quantize_and_rope_for_fp8 pipeline)
    fp4_e2m1,   // K operand (NVFP4, no dequant)
    float,      // accumulator
    fp8_e4m3,   // SCALE operand type (E4M3 per-block)
    H_Q, B_TOPK*2, K, K>
```

PTX: `tcgen05.mma.kind::f8f6f4.block_scale_vec::1X` — block-scaled FP8 × FP4 multiply with E4M3 scale tensor lookup fused into accumulation.

Reference: DeepGEMM `sm100_fp4_mqa_logits.cuh` uses the same family with E8M0 scales (`kind::mxf4`). The `::nvf4` / `::f8f6f4` PTX variants with E4M3 scales exist on SM_100 per PTX ISA 8.5+.

## Warpgroup roles (changed from FP8 baseline)

| WG | FP8 baseline | NVFP4 native |
|---|---|---|
| 0 | Q fetching + O writeback | unchanged |
| **1** | KV fetching + FP8→BF16 dequant inner loop (warp 4-7) | **Coord prep + scale TMA loading only (no dequant)** |
| 2 | UMMA executor + math | UMMA executor with block-scaled MMA |
| 3 | Idle / sync | unchanged |

The dequant warpgroup elimination is the structural win. WG1's `launch_dequant_wg` lambda (FlashMLA phase1.cuh lines ~456-528) **goes away entirely**. Replace with a simpler `launch_coord_prep_wg` that issues the K scales TMA loads + coord lookups, then signals `bar_KV_full`.

## Math warpgroup (WG2) changes

Today (FP8 baseline, phase1.cuh:~635-720): math warpgroup issues `tcgen05.mma.kind::f16/bf16` UMMA with BF16 operands fetched from `smem.K[k_buf_idx]`.

NVFP4 native: issues `tcgen05.mma.kind::f8f6f4.block_scale_vec::1X` with:
- A operand: BF16 Q from `smem.Q` (or FP8 Q if we want FP8×FP4 — we have both forms in pipeline)
- B operand: FP4 K from `smem.K[k_buf_idx]` (packed)
- Scale operand: E4M3 from `smem.K_scales[k_buf_idx]`

`make_instr_desc_block_scaled` builds the runtime descriptor that ties the scale-tensor SMEM offset into the UMMA. See `nvfp4_umma_descriptors.cuh`.

## TMA descriptors (host-side, run())

Three K descriptors per slot (vs two for FP8):
1. `tensor_map_kv_nope`: packed FP4 latent values, stride 224 B/token (vs 448 FP8)
2. `tensor_map_kv_scales`: E4M3 per-block scales, stride 32 B/token (new)
3. `tensor_map_kv_rope`: BF16 rope, stride 128 B/token (unchanged)

Total per-token HBM stride: 224 + 32 + 128 = 384 B (vs 576 FP8) → **33% reduction in KV bandwidth**.

## Expected impact

| Component | FP8 baseline | Native NVFP4 UMMA | Δ |
|---|---|---|---|
| HBM KV read per slot | 576 B | 384 B | -33% |
| HBM total KV per step (B=8, K=1024, 61 layers) | 288 MB | 192 MB | -33% |
| SMEM K bytes (3 bufs) | ~57 KB | ~24 KB | -58% |
| Dequant compute (per slot per layer) | full FP8→BF16 dequant pass | **none** | -100% |
| UMMA throughput (FP8×FP4 vs BF16×BF16) | baseline | ~2x | +100% on compute |

Combined: at our cell (FP8 baseline 30.95 ms TPOT), expected NVFP4 native ~**21-24 ms TPOT** (beats FP8 by 7-10 ms).

## Implementation plan

| Step | Status |
|---|---|
| 1. Design + scaffolding | ✓ This commit |
| 2. `nvfp4_umma_descriptors.cuh` (UMMA atoms + instr_desc helpers) | ✓ |
| 3. `config.h` rewrite (FP4 SMEM, no dequant buffers, block-scaled UMMA) | ✓ |
| 4. `phase1.cuh` WG1 surgery — replace dequant with coord-prep + scale TMA | DEFERRED |
| 5. `phase1.cuh` WG2 surgery — swap UMMA atom + add block-scale operand | DEFERRED |
| 6. Host-side `run()` — build 3 K TMA descriptors (values + scales + rope) | DEFERRED |
| 7. Python wrapper + JIT build | DEFERRED |
| 8. `dsa_backend.py:_forward_trtllm` — dispatch when kv_dtype==uint8 + NVFP4 | DEFERRED |
| 9. `server_args.py:1867` — extend allowed dtype list | DEFERRED |
| 10. Correctness harness vs FP8 baseline (BMM1/BMM2 numerics in FP4 noise band) | DEFERRED |

## Implementation note: kv_dtype dispatch in dsa_backend.py

```cpp
if (
    self.kv_cache_dtype == torch.uint8   // NVFP4 packed bytes
    and isinstance(self.token_to_kv_pool, NVFP4KVPool)
):
    return self._forward_flashmla_nvfp4_native(
        q, q_rope, ..., topk_indices,
        kv_nope_packed,           // (num_pages * page_size, 224 B)
        kv_scales,                // (num_pages * page_size, 32 B E4M3)
        kv_rope,                  // (num_pages * page_size, 128 B BF16)
    )
```

The new kernel entry point lives in `sglang.jit_kernel.sparse_mla_nvfp4_kv.forward`. It builds the 3 TMA descriptors and launches the kernel with the existing block_tables + seq_lens machinery.
