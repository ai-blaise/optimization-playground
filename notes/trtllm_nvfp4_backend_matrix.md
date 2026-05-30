# TensorRT-LLM × SGLang NVFP4 backend matrix for DeepSeek-V3.2-REAP-345B on 8×B200

**Target model**: `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft`
NVFP4 weights · fp8/bf16/HIGGS KV · NVFP4 indexer · DSA + IndexCache + HISA + HiSparse
**Hardware**: 8×B200 single node, SM_100
**Date**: 2026-05-30 · Spencer Garnets · ai-blaise/optimization-playground
**Sources**: NVIDIA/TensorRT-LLM main @ 2026-05-30, ai-blaise/optimization-playground main

---

## 1. Executive summary

Three intersection backends should be tested in the shootout, since each is the closest TRT-LLM analogue of a SGLang path Spencer is already running on B200:

1. **TRTLLM-Gen MoE × trtllm-gen FMHA × FlashMLA**
   = SGLang `flashinfer_trtllm` MoE × `trtllm_mla` × FlashMLA. This is the head-to-head match for the current SGLang baseline (TP=8 + EP=8 + flashinfer_trtllm + vanilla CUDA graphs, ~42 ms TPOT today / ~36 ms TPOT projected with WMMA). Both stacks compile to the *same* trtllm-gen cubins via FlashInfer, so a real perf delta here is configuration / scheduling overhead, not kernel skill.

2. **CUTLASS MoE × trtllm-gen FMHA**
   = SGLang `flashinfer_cutlass` MoE × `trtllm_mla`. NVIDIA's DeepSeek-R1 deployment guide explicitly lists CUTLASS as the *default* MoE backend at B200 EP≤8 NVFP4, and the deepseek-r1-latency.yaml uses `backend: TRTLLM`. So we should bench both CUTLASS and TRTLLM-Gen MoE on each stack at our exact 4096in/1024out/64conc shape, since the crossover point is workload-dependent.

3. **DeepGEMM MoE (fp8\_block\_scales path) + trtllm-gen FMHA × DSA**
   = SGLang `deep_gemm` MoE + `trtllm_mla` × `dsa` backend. This is the test for whether shifting the experts to FP8 block-scales (TRT-LLM's deepseek-r1-deepgemm.yaml shape) recovers anything vs NVFP4 MoE on this 345B-REAP weight footprint. Cross-frame intersection exists; expect TRTLLM-Gen MoE to win on memory, DeepGEMM to potentially win on TPS once DSA + indexer K cache + topK overhead dominates.

**Two non-intersection backends from TRT-LLM are worth porting to SGLang** as research direction (section 5): **DENSEGEMM** (CuTe DSL NVFP4 MoE for the 64–208 token band, blog24, up to 1.12× over TRTLLM-Gen MoE in that range) and **GVR Top-K** for DSA decode (blog21, 1.88× avg / 2.42× peak speedup on indexer Top-K, ~7.5% E2E TPOT win). Both are missing from SGLang as of 2026-05-30. GVR is the higher-priority research item since it touches the exact hot-path Spencer is iterating on (iter6-9 reports).

---

## 2. TRT-LLM NVFP4 backend inventory

### 2.1 NVFP4 weight-quantized GEMM backends

Discovered from `cpp/tensorrt_llm/kernels/`, `tensorrt_llm/_torch/modules/fused_moe/`, and `cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/`. NVFP4 = SM≥100 NVIDIA block-scaled FP4 (E2M1 weights, FP8 E4M3 per-16-element scale, FP32 per-tensor alpha).

| TRT-LLM backend | Class / source | NVFP4 weights | MoE? | MLA? | EP? | SM coverage | Use case |
|---|---|---|---|---|---|---|---|
| **TRTLLM-Gen MoE** | `TRTLLMGenFusedMoE` (`fused_moe_trtllm_gen.py`) + `trtllmGenKernels/blockScaleMoe/` + `trtllmGenKernels/batchedGemm/` | YES (`QuantAlgo.NVFP4`, `W4A8_NVFP4_FP8`, `W4A16_MXFP4`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`, `FP8_BLOCK_SCALES`) | YES (fused dispatch+gemm1+swiglu+gemm2+combine, "min-latency" mode) | n/a | YES (alltoall via mnnvl/DeepEP) | SM100, SM103 only | Default for B200 NVFP4 latency, primary kernel set used by FlashInfer's `trtllm_*` ops |
| **CUTLASS MoE** | `CutlassFusedMoE` (`fused_moe_cutlass.py`) + `cutlass_kernels/moe_gemm/moe_gemm_kernels_{fp4_fp4,fp8_fp4,bf16_fp4,fp16_fp4}.cu` + `cutlass_kernels/fp4_gemm/{nvfp4_nvfp4,mxfp8_mxfp4}_gemm_template_sm{100,120}.h` | YES (NVFP4 on SM 100/103/120/121; W4A8_NVFP4_FP8) | YES (max-throughput, reducescatter path with attn-DP on) | n/a | YES | SM≥80 unquantized; SM≥100 NVFP4 | Default MoE backend B200/GB200 EP≤8 NVFP4 (DeepSeek-R1 deployment guide). The throughput-oriented sibling of TRTLLM-Gen |
| **WIDEEP MoE** | `WideEPMoE` (`fused_moe_wide_ep.py`) | YES (inherits CUTLASS + FP8-block-scales paths) | YES (wide-EP variant with EPLB / load balancer) | n/a | YES (EP>8 on GB200 NVL72) | SM≥100 | Mandatory for GB200 NVL72 EP>8 NVFP4. *Not applicable for our 8×B200 EP=8 target.* |
| **DEEPGEMM MoE** | `DeepGemmFusedMoE` (`fused_moe_deepgemm.py`) — wraps DeepSeek's DeepGEMM library | FP8 block-scales only by default; **does NOT do NVFP4 weights directly** | YES (fp8\_block\_scales) | n/a | YES | SM100 | The TRT-LLM analogue of SGLang's `deep_gemm` runner. NVFP4 model would need conversion. Used by deepseek-r1-deepgemm.yaml |
| **CUTEDSL MoE** | `CuteDslFusedMoE` (`fused_moe_cute_dsl.py`) + `CuteDslB12xFusedMoE` (`fused_moe_cute_dsl_b12x.py`) | YES (fp8\_block\_scales OR nvfp4); B12x = SM 120/121 hybrid CUTLASS-prefill / FlashInfer NVFP4 decode | YES | n/a | YES | SM100/SM103 (vanilla); SM120/SM121 (B12x variant requires flashinfer) | CuTe DSL implementation of MoE; B12x variant is for workstation Blackwell (RTX 6000 Pro / RTX 5090 family) |
| **DENSEGEMM MoE** | `DenseGEMMFusedMoE` (`fused_moe_densegemm.py`) — blog24 backend | YES (NVFP4 only) | YES (dense FC1/FC2 over all experts + alpha mask) | n/a | YES | SM100, SM103 only | **Specialized for the 64–208-tokens-per-MoE-fwd band**; up to 1.12× over TRTLLM-Gen MoE in that range. Not a general replacement. ⚠️ MISSING from SGLang. |
| **VANILLA MoE** | `VanillaMoE` (`fused_moe_vanilla.py`) | NO (BF16 reference impl) | YES | n/a | — | Any | Reference path only |
| **TRITON MoE** | `TritonFusedMoE` (`fused_moe_triton.py`) | NO (fp8 / bf16) | YES | n/a | — | Any | Generic Triton fallback |
| **MEGAMOE\_DEEPGEMM** | `MegaMoEDeepGemm` (`mega_moe/`) — bundled DeepGEMM fp8\_fp4\_mega\_moe | Accepts W4A8\_MXFP4\_MXFP8 (MXFP4 weights, not NVFP4) | YES (fused dispatch+gemm1+act+gemm2+combine) | n/a | YES | SM100 | GPT-OSS-style MXFP4; not relevant for our NVFP4 target |
| **CUTLASS dense GEMM (non-MoE)** | `cutlass_kernels/fp4_gemm/` (.cu files) used by `Linear`, `MLP`, `GatedMLP`, indexer linears | YES (NVFP4 GEMM) | n/a | Used for MLA `q_a/q_b/kv_a_proj_with_mqa/kv_b_proj`, indexer `wk/wq/weight_proj`, MoE shared-expert MLP, lm\_head | n/a | SM100/103/120/121 | This is what NVFP4 weights for *non-MoE* linears go through |

### 2.2 NVFP4-compatible attention backends (MLA / sparse-attention)

| Attention backend | Source | NVFP4 KV cache? | MLA? | DSA? | EP/DP? | Use case |
|---|---|---|---|---|---|---|
| **trtllm-gen FMHA** | `trtllm_gen.py` (FlashInferTrtllmGenAttention) + `trtllmGenKernels/fmha/` | YES (DataType.NVFP4 in `SUPPORTED_KV_CACHE_DTYPES`); also FP8/FP16/BF16 KV | YES (MLA-capable Blackwell FMHA, calls into `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` and `flashinfer.prefill.trtllm_ragged_attention_deepseek`) | NO (DSA selector lives outside) | DP-attn supported | Default for B200 MLA decode |
| **TrtllmAttention** | `trtllm.py` (TRT-LLM C++ thop attention, FlashAttention v2-style) | YES (FP8/NVFP4 KV) | YES | NO | Yes | Older path, used when trtllm-gen FMHA not supported |
| **DSA attention** | `attention_backend/sparse/dsa.py` | NVFP4 weights for indexer linears, FP8 blockwise KV for indexer K cache, per-tensor FP8 sparse MLA KV (blog15 §"Precision Strategy") | YES (sparse MLA via `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` + Top-K mask + `fp8_paged_mqa_logits`) | YES (DeepSeek Sparse Attention, "algorithm: dsa") | DP/EP/TP supported | **The DSV3.2 production path on TRT-LLM**. Builds on TrtllmAttention + indexer + Top-K kernel |
| **Rocket (rocket.py)** | `attention_backend/sparse/rocket.py` | n/a | NO | NO (different sparse algorithm: 2-stage eviction + dynamic selection) | Yes | Not relevant for DSV3.2 |
| **flashinfer** | `attention_backend/flashinfer.py` | FP8 KV | YES (limited) | NO | Yes | General FlashInfer kernels |
| **vanilla** | `attention_backend/vanilla.py` | — | — | NO | — | Reference impl |

### 2.3 NVFP4 quantization touch points (for completeness)

Files demonstrating NVFP4 is end-to-end first-class:
- `cpp/tensorrt_llm/thop/fp4Gemm.cpp`, `fp4Quantize.{cpp,h}`, `cudaNvfp4MM.cpp` — torch op surface
- `cpp/include/tensorrt_llm/common/quantization.h` — `QuantMode::nvfp4()` at bit 12, `fp4KvCache()` at bit 13, `w4a8Mxfp4Fp8()`/`w4a8Mxfp4Mxfp8()`/`w4a16Mxfp4()` at bits 14/15/16
- `tensorrt_llm/quantization/utils/fp4_utils.py`, `tensorrt_llm/quantization/modelopt_config.py` — Python NVFP4 utils + ModelOpt config (`KV_CACHE_NVFP4`)
- `tensorrt_llm/_torch/auto_deploy/transform/library/fuse_relu2_quant_nvfp4.py`, `fuse_rmsnorm_quant_nvfp4.py` — auto-deploy fusion passes
- `cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/fusedGatedRMSNormQuant.cuh`, `arcquantFP4.{cu,h}`, `fusedCatFp4.{cu,h}` — fused norm/cat into NVFP4 quant

---

## 3. SGLang backend inventory (B200 NVFP4 relevant)

Discovered from `/home/spencergarnets/work/optimization-playground/python/sglang/srt/layers/{moe,attention,quantization}/`.

### 3.1 SGLang MoE runner backends (`MoeRunnerBackend` enum in `moe/utils.py`)

| SGLang backend | Source | NVFP4 weights? | MLA-compat | EP? | TRT-LLM analogue |
|---|---|---|---|---|---|
| **FLASHINFER_TRTLLM** | `moe/moe_runner/flashinfer_trtllm.py` + `moe/flashinfer_trtllm_moe.py` (wrappers `trtllm_fp8_block_scale_moe_wrapper`, `trtllm_fp8_per_tensor_scale_moe_wrapper`, etc.) | YES (FP8 block-scale + NVFP4 via FlashInfer's `trtllm_fp4_block_scale_moe`) | yes | YES (via DeepEP/Mori-EP token dispatcher) | **TRTLLMGenFusedMoE** |
| **FLASHINFER_TRTLLM_ROUTED** | same module, "routed" variant | YES | yes | YES | TRTLLMGenFusedMoE routed variant |
| **FLASHINFER_CUTLASS** | `moe/moe_runner/runner.py` calls into `flashinfer.cutlass_fp4_moe` | YES (CUTLASS NVFP4 MoE) | yes | YES | **CutlassFusedMoE** |
| **FLASHINFER_CUTEDSL** | `moe/flashinfer_cutedsl_moe.py` + `moe/moe_runner/flashinfer_cutedsl.py` | YES (NVFP4 CuTe DSL path) | yes | YES | **CuteDslFusedMoE** / **CuteDslB12xFusedMoE** |
| **FLASHINFER_MXFP4** | `moe/moe_runner/runner.py` | MXFP4 (not NVFP4) | yes | YES | MegaMoE_DeepGemm |
| **DEEP_GEMM** | `moe/moe_runner/deep_gemm.py` | FP8 block-scale (no NVFP4 weights directly) | yes | YES | **DeepGemmFusedMoE** |
| **CUTLASS** | `moe/moe_runner/runner.py` (legacy) | FP8/INT8 mainly | yes | YES | (older CutlassFusedMoE FP8 path) |
| **TRITON** | `moe/moe_runner/triton.py` | FP8/INT8/W4 | yes | YES | TritonFusedMoE |
| **TRITON_KERNELS** | `moe/moe_runner/triton_kernels.py` | FP8 | yes | YES | TritonFusedMoE |
| **AITER** | `moe/moe_runner/aiter.py` | ROCm/AITer | no | — | (n/a) |
| **MARLIN** | `moe/moe_runner/marlin.py` | W4A16 | no | — | fpA_intB_gemm |
| **WARP_DECODE** | `moe/moe_runner/runner.py` | n/a | yes | — | n/a |

### 3.2 SGLang attention backends (`@register_attention_backend` decorators)

| SGLang backend | Source | NVFP4-KV? | MLA? | DSA? | TRT-LLM analogue |
|---|---|---|---|---|---|
| **trtllm_mla** | `attention/trtllm_mla_backend.py` (wraps `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` + `flashinfer.prefill.trtllm_ragged_attention_deepseek`) | FP8 KV (NVFP4 KV path exists via flashinfer but not heavily used in our config) | YES | NO | **trtllm-gen FMHA** (FlashInferTrtllmGenAttention) |
| **trtllm_mha** | `attention/trtllm_mha_backend.py` | FP8/NVFP4 KV | NO | NO | trtllm-gen FMHA non-MLA path |
| **dsa** | `attention/dsa_backend.py` + `attention/dsa/` (NVFP4 indexer, HiSparse, HISA, IndexCache, Iter6-9 patches) | FP8 indexer K cache (HIGGS variant locally), per-tensor FP8 sparse MLA KV | YES (sparse MLA path) | **YES** (DSV3.2 production path) | **DSA attention backend** (`attention_backend/sparse/dsa.py`) |
| **dsv4** | `attention/dsv4_backend.py` + `attention/dsv4/` | local research path | YES | YES (next-gen DSv4 variant) | (no TRT-LLM analogue yet — ai-blaise-only) |
| **flashinfer** | `attention/flashinfer_backend.py` | FP8 | yes | NO | flashinfer attention backend |
| **flashinfer_mla** | `attention/flashinfer_mla_backend.py` | FP8 | YES | NO | flashinfer MLA |
| **flashmla** | `attention/flashmla_backend.py` (wraps DeepSeek FlashMLA cubins) | FP8 | YES | NO | `cpp/tensorrt_llm/kernels/flashMLA/` (TRT-LLM bundles FlashMLA too) |
| **fa3 / fa4** | `attention/flashattention_backend.py` | FP8/FP16 | NO | NO | (TrtllmAttention thop path; no direct FA3/4 in TRT-LLM) |
| **cutlass_mla** | `attention/cutlass_mla_backend.py` | FP8 | YES | NO | TrtllmAttention thop |
| **nsa** | `attention/nsa_backend.py` | research | YES | YES (NSA variant) | (no TRT-LLM analogue) |
| **tokenspeed_mla** | `attention/tokenspeed_mla_backend.py` | FP8 | YES | NO | (no direct analogue) |

### 3.3 SGLang NVFP4 GEMM (non-MoE, non-attention)

- `python/sglang/srt/layers/quantization/modelopt_quant.py` — ModelOpt loader (mirrors TRT-LLM's ModelOpt path; same checkpoint format)
- `python/sglang/srt/layers/quantization/fp4_utils.py` — `fp4_quantize`, `get_fp4_gemm_runner_backend`, `cutlass_fp4_gemm` (calls `sglang.jit_kernel.nvfp4.cutlass_scaled_fp4_mm`)
- `python/sglang/jit_kernel/nvfp4.py` + `python/sglang/jit_kernel/csrc/quantization/` — local CUDA kernels for NVFP4 dense GEMM, plus the HIGGS dense-2bit MLA-decode kernel (`higgs_dense_2bit_mla_decode.cuh` — pre-WMMA-always-on iter9, post the b8318c5 commit it's WMMA on for n_heads≤64)
- Indexer-specific NVFP4 path: `python/sglang/jit_kernel/csrc/dsa/nvfp4_indexer_quant.cuh` (local kernel, currently in active iter9 modification)

---

## 4. Intersection backends (the actual shootout matrix)

These three rows are the configurations to bench in both frameworks at the exact same input/output/concurrency shape. The TRTLLM-Gen MoE × trtllm_mla path is the *direct* head-to-head with the current SGLang baseline.

| Row | TRT-LLM config | SGLang config | Equiv? | Notes |
|---|---|---|---|---|
| **A** | `moe_config.backend: TRTLLM` + trtllm-gen FMHA (TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION=1) + sparse_attention `algorithm: dsa` | `moe-runner-backend=flashinfer_trtllm` + `attention-backend=trtllm_mla` (or `dsa` for the sparse path) + flashinfer trtllm-gen | YES — same trtllm-gen cubins under the hood | This is the apples-to-apples bench. Any delta = scheduler / DP-attn / overlap / CUDA-graph machinery, not kernel speed |
| **B** | `moe_config.backend: CUTLASS` + trtllm-gen FMHA + DSA | `moe-runner-backend=flashinfer_cutlass` + `trtllm_mla` + (dsa) | YES | Tests the CUTLASS MoE path at our shape. NVIDIA's DSR1 default is CUTLASS; we should confirm whether at 4096in/1024out/64conc the CUTLASS MoE wins on either stack |
| **C** | `moe_config.backend: DEEPGEMM` + trtllm-gen FMHA + DSA *(requires FP8 block-scale checkpoint, not NVFP4)* | `moe-runner-backend=deep_gemm` + `trtllm_mla` + (dsa) *(also FP8 block-scale)* | YES, but on FP8 weights | Side test: how much does FP8-block-scale MoE recover vs NVFP4 at our 345B-REAP footprint? Only valid against an FP8-weights checkpoint of the same model |

**For each row, run the same matrix** at 4096in/1024out/64conc (matching Spencer's current bench):

```
TP=8, EP=8, DP-attn=on, vanilla CUDA graphs (no PCG), MTP=1 (NextN),
DSA k=2048, HISA=on, HiSparse=on, IndexCache=on
KV: fp8 (sparse MLA), fp8-blockwise (indexer K), HIGGS for dense MLA where available
fp4_quant: NVFP4 (E2M1, 16-elem FP8 E4M3 scale, FP32 alpha)
```

Single primary metric: **TPOT (ms)**. Spencer's V14-real M4 measurement discipline (linear\_decode\_tps + intel-correctness two-stage) does not apply here — this is the B200 track per `/Users/spencer/.claude/projects/-Users-spencer/memory/project_b200_track.md`, where the convention is per-iteration TPOT at fixed shape.

### What I'd expect, pre-bench (priors only, not predictions)

- **Row A (TRTLLM-Gen MoE)**: TRT-LLM should be at most marginally faster on raw kernels because both call into the same cubins. SGLang has a known scheduler / overlap-scheduler gap that TRT-LLM has been hammering on (blog10 ADP Balance) — that's where the delta may show.
- **Row B (CUTLASS MoE)**: less clear. NVIDIA defaults CUTLASS for DSR1 throughput, TRTLLM-Gen for latency. At 64-concurrency this is the *low-latency* regime (per blog24 token-band partitioning, MoE-input tokens-per-fwd ≈ 64 × 1 token/decode = 64, which is in blog24's "low-latency" band of 32–320). So TRTLLM-Gen MoE should win on raw GEMM but CUTLASS may win on the rest of the pipeline.
- **Row C (FP8-block-scale MoE)**: only relevant if the FP8 weights version of the REAP model is on hand. Per the deepseek-r1-deepgemm.yaml shape, this exists as a productionized TRT-LLM path.

---

## 5. TRT-LLM-only backends (research direction for SGLang)

Two distinct items, prioritized:

### 5.1 DENSEGEMM MoE (blog24) — `DenseGEMMFusedMoE`

CuTe DSL NVFP4 MoE backend that recasts routed FC1/FC2 as one dense GEMM over all experts with per-token alpha masking. Targets the 64–208 tokens-per-MoE-fwd "low-latency" band. Up to **1.12× over TRTLLM-Gen MoE** on DeepSeek-V3-style TP8 B200 in the measured sweet spot.

- Source: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_densegemm.py`
- Kernel: CuTe DSL, requires `TRTLLM_MOE_FUSED_FC2_ALPHA=1` env to fully fuse
- Constraints: NVFP4-only, SM100/103-only, shared-expert still standalone (current impl), redundant compute starts to hurt above ~208 tokens
- **Status in SGLang**: not present. `FLASHINFER_CUTEDSL` exists but executes the standard grouped-GEMM CuTe DSL path, not the dense-over-all-experts variant. Porting cost: medium — CuTe DSL kernels would need to be authored or wrapped from TRT-LLM's `cuteDslKernels`. **Risk**: highly workload-dependent (only the 64–208 band wins), but our 64-conc decode shape is right in the sweet spot.

### 5.2 GVR Top-K for DSA decode (blog21) — `enable_heuristic_topk=True`

Guess-Verify-Refine exact Top-K that reuses previous decode step's Top-K as a warm-start. Phases: Guess from prev-Top-K → Verify with secant threshold → ballot-free candidate collect → exact refinement in SMEM. Currently `index_topk=2048` only.

- Source: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` (search for "warmup_heuristic_topk_decode"); kernel cubins live in flashinfer dependency
- Reported: **1.88× avg / 2.42× peak** single-op speedup over the production radix-select Top-K, **up to 7.52% E2E TPOT** reduction in fixed-OSL TEP8 min-latency DSV3.2
- Falls back to radix select automatically when hardware-aware thresholds not met
- **Status in SGLang**: not present. `python/sglang/jit_kernel/csrc/dsa/nvfp4_indexer_quant.cuh` does radix-select-style work. **This is the highest-priority research direction** because (a) it touches the exact hot path Spencer is iterating on in `dsa_nvfp4_indexer_iter{6,7}_recon.md` / `iter9_recon.md`, (b) the 7.5% E2E TPOT win is meaningful, (c) it's algorithm-level (not just kernel-level), so it's a stable target. Porting cost: medium — paper at arXiv:2604.22312 has full algorithm; can be implemented as a SGLang JIT kernel under `python/sglang/jit_kernel/csrc/dsa/` parallel to the existing radix kernel.

### 5.3 Lower-priority TRT-LLM-only features

- **MegaMoE DeepGEMM fused-MoE** (`MegaMoEDeepGemm`) — only useful if we move to MXFP4 weights (W4A8\_MXFP4\_MXFP8). Not applicable to NVFP4 target.
- **DENSEGEMM target workflow** (the full version where shared experts and Router/TopK also fold into one dense GEMM with FC2-alpha fusion) — currently incomplete in TRT-LLM itself. Not actionable yet.
- **Helix Parallelism** (blog22) — for multi-million-token decoding via KV-cache sharding. Not applicable to our 4096in/1024out target.
- **Wide-EP + EPLB** — applies to GB200 NVL72 EP>8 only. Our 8×B200 EP=8 target is below the threshold.
- **trtllm-gen BF16 routing for fused MoE** (`_supports_flashinfer_bf16_routing_method`) — TRT-LLM has a BF16 MoE on the trtllm-gen path that SGLang doesn't. Relevant only if we revisit BF16 MoE for an A/B accuracy check.
- **Skip Softmax / BLASST sparse attention** (blog16, `algorithm: skip_softmax`) — drop-in dynamic skip of softmax/BMM2 work. Orthogonal to DSA; not relevant for DSV3.2 path.

---

## 6. Test matrix (exact config variants to bench)

Single dataset: synthetic, 4096in/1024out/64conc, ≥256 requests (so warmup variance averages out). One node, 8×B200, SM_100. One `trtllm-bench` / `python -m sglang.bench_serving` per row; report TPOT (ms), TPS/user, TPS/GPU, TTFT (ms) for completeness.

### 6.1 TRT-LLM rows

| Row | YAML key bits | trtllm-bench / trtllm-serve cmd |
|---|---|---|
| **T-A** TRTLLM-Gen MoE + DSA | `moe_config.backend: TRTLLM`, `enable_attention_dp: true`, `kv_cache_config.dtype: fp8`, `sparse_attention_config.algorithm: dsa`, `sparse_attention_config.index_topk: 2048`, `speculative_config.decoding_type: MTP`, `num_nextn_predict_layers: 1`, `cuda_graph_config.enable_padding: true` | `trtllm-bench -m BlaiseAI/DeepSeek-V3.2-REAP-345B-... throughput --tp 8 --ep 8 --backend pytorch --max_batch_size 64 --max_num_tokens 5120 --concurrency 64 --num_requests 256 --config ./config.yml --streaming` |
| **T-A'** T-A + GVR Top-K | T-A + `sparse_attention_config.enable_heuristic_topk: true` | same | (this is the GVR-on variant; only valid for `index_topk=2048`) |
| **T-B** CUTLASS MoE + DSA | T-A but `moe_config.backend: CUTLASS` | same | |
| **T-C** DENSEGEMM MoE + DSA | T-A but `moe_config.backend: DENSEGEMM`, env `TRTLLM_MOE_FUSED_FC2_ALPHA=1` | same | NVFP4-only, expected sweet spot in our 64-conc shape per blog24 |

### 6.2 SGLang rows (match the current `optimization-playground/main` HEAD)

| Row | flags | sglang launch |
|---|---|---|
| **S-A** flashinfer_trtllm MoE + trtllm_mla (current baseline) | `--moe-runner-backend flashinfer_trtllm --attention-backend trtllm_mla --enable-dp-attention --tp 8 --ep-size 8 --speculative-algorithm NEXTN --speculative-num-steps 1 --enable-cuda-graph` | `python -m sglang.launch_server --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-... ...` |
| **S-A-DSA** S-A + dsa attention | `... --attention-backend dsa` (with HiSparse + HISA + IndexCache config-side) | same, with iter9 patches applied |
| **S-B** flashinfer_cutlass MoE + trtllm_mla | `--moe-runner-backend flashinfer_cutlass --attention-backend trtllm_mla ...` | same |
| **S-B-DSA** S-B + dsa | `... --attention-backend dsa` | same |
| **S-C** deep_gemm MoE + trtllm_mla *(FP8 weights variant)* | `--moe-runner-backend deep_gemm --attention-backend trtllm_mla ...` | same — only if FP8 weights version of REAP model exists |

### 6.3 Reporting

Per row record: TPOT (ms), TTFT (ms), TPS/user, TPS/GPU, HBM peak (GiB), p50/p95 latency, accuracy delta on GSM8K + MMLU + GPQA-Diamond (for sanity that we're not regressing accuracy across stacks).

The **headline comparison** is `(S-A-DSA, current iter9) vs (T-A) vs (T-A')` — that's the trio that tells us:
- How much SGLang scheduler / DP-attn / overlap is leaving on the table vs TRT-LLM at otherwise-identical kernel stack
- How much GVR Top-K alone would buy us on the TRT-LLM side (and therefore what the upper bound on porting it to SGLang is)

### 6.4 Reproducibility / pinning

- TRT-LLM container: `nvcr.io/nvidia/tensorrt-llm/release:x.y.z` (rc-suffixed release matching 2026-05-30 main; pin SHA in bench log)
- SGLang: `ai-blaise/optimization-playground@main` @ commit 8151a8aea ("DSA NVFP4 indexer: force WMMA cand_score always-on for n_heads<=64") or later
- Model: `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` HF revision pinned in bench log
- ModelOpt: pin Model-Optimizer commit used for the NVFP4 quant of REAP base

---

## References

- TRT-LLM blog15 (DSV3.2 on Blackwell, NVFP4 strategy, indexer kernels): `docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md`
- TRT-LLM blog24 (DENSEGEMM): `docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md`
- TRT-LLM blog21 (GVR Top-K): `docs/source/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.md`
- TRT-LLM DSR1 deployment guide (MoE backend support matrix at B200 EP≤8 NVFP4): `docs/source/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.md`
- TRT-LLM quantization doc (NVFP4 hardware support matrix): `docs/source/features/quantization.md`
- TRT-LLM sparse attention doc (DSA / RocketKV / BLASST): `docs/source/features/sparse-attention.md`
- TRT-LLM feature combination matrix: `docs/source/features/feature-combination-matrix.md`
- TRT-LLM MoE backend factory: `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`
- TRT-LLM TRTLLMGenFusedMoE: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py`
- TRT-LLM CutlassFusedMoE: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`
- TRT-LLM DSA backend: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
- TRT-LLM trtllm-gen FMHA: `tensorrt_llm/_torch/attention_backend/trtllm_gen.py`
- TRT-LLM curated configs: `examples/configs/curated/deepseek-r1-{latency,throughput,deepgemm}.yaml`
- TRT-LLM DSV3 example README: `examples/models/core/deepseek_v3/README.md`
- TRT-LLM NVFP4 kernels: `cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/{nvfp4_nvfp4,mxfp8_mxfp4}_gemm_template_sm{100,120}.h`, `cpp/tensorrt_llm/kernels/trtllmGenKernels/{gemm,batchedGemm,blockScaleMoe,fmha}/`
- TRT-LLM QuantMode: `cpp/include/tensorrt_llm/common/quantization.h`
- SGLang MoeRunnerBackend enum: `python/sglang/srt/layers/moe/utils.py`
- SGLang flashinfer_trtllm runner: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`
- SGLang attention backend registry: `python/sglang/srt/layers/attention/attention_registry.py`
- SGLang trtllm_mla backend: `python/sglang/srt/layers/attention/trtllm_mla_backend.py`
- SGLang DSA backend: `python/sglang/srt/layers/attention/dsa_backend.py`, `python/sglang/srt/layers/attention/dsa/`
- SGLang ModelOpt NVFP4 quantization: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/layers/quantization/fp4_utils.py`
- SGLang local NVFP4 / HIGGS / DSA kernels: `python/sglang/jit_kernel/csrc/dsa/nvfp4_indexer_quant.cuh`, `python/sglang/jit_kernel/csrc/quantization/higgs_*_mla_decode.cuh`
- Prior reconnaissance in this repo: `notes/dsa_nvfp4_indexer_iter{6,7,8,9}_recon.md`, `notes/higgs_dsa_iter{8,9}_recon.md`, `notes/higgs_mla_decode_iter{6,7}_recon.md`, `notes/nvfp4_moe_iter{6,7}_recon.md`, `notes/ep_nvfp4_moe_path_research.md`, `notes/pcg_vs_vanilla_cuda_graph_research.md`
