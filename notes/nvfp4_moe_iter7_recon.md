# #15 NVFP4 MoE iter7 reconnaissance

## Mission

PRIMARY (architectural unlock, multi-day): patch flashinfer trtllm_fused_moe_kernel_launcher.cu
to relax the dtype gate that rejects (Bfloat16 act + E2m1 weight). The
in-cubin export already ships **58 Bmm_Bfloat16_E2m1E2m1_*.cubin** SM_100
variants that accept BF16 activation directly and quantize fused. Eliminating
the host-side fp4_quantize call ceilings at ~200-300us/step for 58 MoE
layers (per iter4 PRIMARY commit projection).

## Cubin catalog (58 variants)

Located at:
  /flashinfer_cubin/.../batched_gemm-b3c1646-c111d7c/Bmm_Bfloat16_E2m1E2m1_*.cubin

Tile shape decoding (from kernel filenames):
  Bmm_<Act>_<Wt><Sf>_<Acc>_bA<bitsA>_bB<bitsB>_t<M>x<N>x<K>{u2?}_s<stages>_et<epiM>x<epiN>_m<mmaM>x<mmaN>x<mmaK>_cga<X>x<Y>x<Z>_<...>

Tile (M, N, K) coverage in the 58 set (sorted by N then K):
  M=128 fixed across all.
  N variants: 8, 16, 32, 64, 128, 256
  K variants: 256, 512
  Pipeline depths (s): 3, 4, 5, 6, 9
  CGA: 1x1x1 (single-CTA), 2x1x1 (cga2 cluster), 1x1x2/3/4 (splitK).
  schedS vs schPd: serial vs persistent scheduler.
  All ship biasM_bN_rgTma_clmp + dynamic-batch + Bfloat16 epilogue.

Production shape match (DSv3.2-REAP, m=top_k*B=1024, H=7168):
  GEMM1: (m, intermediate=4096) per-expert. Tile (M=128, N=128, K=128) lands
         in 4 N-tiles times 32 M-tiles per expert. The 6 tile-shape rows
         {t128x128x128_s3, t128x128x128_s6, t128x128x128u2_s3, t128x128x128u2_s6,
          t128x128x256_s2..s4} from the Gemm_Bfloat16_E2m1E2m1_* dense cube
         already exist; the Bmm batched export adds tile-N=256 (53rd entry,
         t128x256x256_s5_et128x64_m256x256x64_cga2x1x1) for the gather/
         scatter-batched form needed by trtllm-moe.

  GEMM2: (m, hidden=7168) per-expert. Tile (M=128, N=128, K=64-256) covers.
         The cga2 cluster variants are the high-occupancy path; splitK1-4
         covers low-batch tails.

  Per the iter4 commit, m_global peak = 128 (batch=128, DP=TP=8). The hot
  tiles are t128x128x256, t128x128x128, t128x64x512 -- all 3 are present in
  the 58 set.

## The blocking gate

trtllm_fused_moe_kernel_launcher.cu FP4BlockScaleLauncher::check_moe() at
L1597-1616:

    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::Bfloat16 ||
                   mDtypeAct == btg::Dtype::E4m3 || mDtypeAct == btg::Dtype::MxE4m3)
        << "Only E2m1, Bfloat16, MxE4m3 and E4m3 are supported by Fp4 block scale MoE";

    if (mDtypeAct == btg::Dtype::E2m1) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::E2m1)
          << "Only E2m1 and MxE2m1 are supported by block scale MoE with E2m1 activation";
      ...
    } else if (mDtypeAct == btg::Dtype::Bfloat16 || mDtypeAct == btg::Dtype::E4m3 ||
               mDtypeAct == btg::Dtype::MxE4m3) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::MxE2m1)
          << "Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or "
             "MxE4m3 activation";
    }

The gate REJECTS (Bfloat16 act + E2m1 weight). It accepts only
(Bfloat16 act + MxE2m1 weight). MxE2m1 vs E2m1 is the SF block size:
MxE2m1 uses E8M0 group scales of size 32 (NVFP4 with mxfp4-style sf),
E2m1 uses E4M3 per-token scales of size 16 (DSv3.2-REAP NVFP4 layout).

DSv3.2-REAP ships E2m1 weights with E4M3 group scales of size 16 -- exactly
the mDtypeWeights == E2m1 path. Hence the rejection: BF16 act x E2m1 weight
is the unsupported quadrant, even though the cubin export ships the kernels.

## Python-side mirror gate

flashinfer/fused_moe/core.py:is_trtllm_moe_supported() at L122:

    if dtype_weights == DtypeTrtllmGen.E2m1 and dtype_act != DtypeTrtllmGen.E2m1:
        return False

Same rejection on the Python guard side. Both must be relaxed.

## Why the cubins are valid for this combo (bA16/bB16 decoding)

The cubin name Bmm_Bfloat16_E2m1E2m1_Fp32_bA16_bB16_*:
  bA16 = A operand block size = 16 (NVFP4 swizzle granularity)
  bB16 = B operand block size = 16 (same)
This is the E2m1-block-scale layout (matches mDtypeWeights == E2m1, sf_vec_size=16).
Cf. MxE2m1 cubins which carry bA32_bB32 (sf_vec_size=32).

In the launcher prepare_moe() at L1670:
    auto const sf_vec_size = mDtypeWeights == btg::Dtype::MxE2m1 ? 32 : 16;
the sf_vec_size already branches correctly. The runner accepts both at L590:
    mGemm2(Gemm2::Runner(dtypeAct, dtypeWeights, btg::Dtype::Bfloat16, ...))
no signature-level rejection downstream. The cubin lookup is by
(dtype_act, dtype_wt) tuple; the (Bfloat16, E2m1) tuple resolves to the
58-variant Bmm_Bfloat16_E2m1E2m1_* set. Confirmed by inspection of the
batched_gemm-b3c1646-c111d7c export dir.

## Patch scaffold plan (4 stages)

Stage A: vendored launcher inside SGL (one-time copy from flashinfer site-packages
into python/sglang/srt/external_kernels/flashinfer/csrc/trtllm_fused_moe_kernel_launcher_v2.cu)
with check_moe() and python validation gates relaxed. Keep the upstream
file untouched so a flashinfer upgrade does not silently revert the patch.

Stage B: bind a new python entry trtllm_fp4_block_scale_moe_bf16_act in
the SGL kernel layer. Same signature as flashinfer's
trtllm_fp4_block_scale_moe but accepts BF16 hidden_states (not packed FP4)
and elides hidden_states_scale (the cubin computes per-block SF on the fly
from BF16 input within the bmm epilogue).

Stage C: SGL wire in compressed_tensors_w4a4_nvfp4_moe.py L343 area:
when SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE env=1 AND the runtime cubin export
contains Bmm_Bfloat16_E2m1E2m1_* (probe at startup), skip the entire
scaled_fp4_quant_linear / fp4_quantize block and pass x BF16 directly to
the new entry. Otherwise fall back to the iter1-3 quantize path. Opt-in
checkpoint pattern matches iter4 SECONDARY / iter5 SECONDARY discipline.

Stage D: standalone microbench
  test/srt/test_nvfp4_moe_bmm_bf16_act_bench.py
to validate the in-cubin BF16-input bmm path matches the
fp4_quantize+bmm path bit-exact at end-to-end output (BF16 result tensor),
and to attribute the per-layer us delta.

## Expected wins

Per the iter4 PRIMARY close-out: scaled_fp4_quant_linear costs ~40us/call
at m_global=128 (the sgl-native fastest quantize). 58 NVFP4 MoE layers
per step => 58 * 40us = 2320us upper bound, but the bmm in-cubin path
will pay back some of that for the fused SF generation. Net target per
the iter4 projection: 200-300us/step ceiling.

At smaller decode batches (m_global=64) the quantize relative cost climbs
(launch overhead dominates the small-tensor quantize), pushing potential
wins higher. The opt-in env gate is mandatory until iter7+1 (post-bench
ablation) so the production wire stays on the iter4-iter6 stack.

## Files to add / modify

ADD:
  python/sglang/srt/external_kernels/flashinfer/csrc/trtllm_fused_moe_kernel_launcher_v2.cu
  python/sglang/srt/external_kernels/flashinfer/csrc/trtllm_fused_moe_runner_v2.cu (verbatim copy unless gate-relaxation requires runner-side touch)
  python/sglang/srt/external_kernels/flashinfer/__init__.py  (build glue)
  python/sglang/srt/external_kernels/flashinfer/_register.py (TVM-FFI registration)
  test/srt/test_nvfp4_moe_bmm_bf16_act_bench.py

MODIFY:
  python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py
    -- new opt-in branch around L343, gated on SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE

  python/sglang/srt/managers/expert_location_dispatch.py (only if env-vars catalog touched)
  python/sglang/srt/utils.py (env-var declaration)

## Time budget mapping

  Stage A vendor + gate-relax compile: 2-4 hours (this iter delivers ONLY the file copy + diff)
  Stage B python bindings: 2-3 hours
  Stage C wire + opt-in env: 1 hour
  Stage D microbench: 1 hour
  Stage E end-to-end deploy bench: 1 hour
  Total: 7-10 hours, well past the iter7 3-hour budget.

  This iter ships: Stage A vendor copy, the recon catalog (this file),
  the launcher patch DIFF as a checked-in .patch file in
  external_kernels/flashinfer/patches/, and the iter8 wire plan. No
  runtime change. Honest scaffold.

## Honest negatives / open questions

1. The 58 cubin variants assume the cubin loader's hash-to-path map already
   knows about Bfloat16_E2m1E2m1 entries. If the kernel selector indexes
   by (mDtypeAct, mDtypeWt) and the existing FP4BlockScaleLauncher map
   only ships entries for (E2m1, E2m1) and (Bfloat16, MxE2m1) tuples,
   we need to extend the map. To be checked in Stage A.

2. The bmm epilogue may require an explicit per-token scale tensor for
   the (BF16 act + E2m1 weight) path -- iter4 already pre-computes
   layer.w13_input_scale_quant for the fp4_quantize call. If the
   in-cubin epilogue derives SF from input absmax automatically, this
   tensor is unused; if it requires a per-token scale, we plumb the
   same w13_input_scale_quant through hidden_states_scale (semantic
   meaning differs from the FP4-input case; doc carefully).

3. Output bit-exactness vs the iter4-iter6 production path is the
   acceptance gate. If the in-cubin SF-gen differs from
   scaled_fp4_quant_linear by more than +/-1 ULP at FP4 quantize
   boundary, the iter8 wire stays opt-in even if perf-positive.
