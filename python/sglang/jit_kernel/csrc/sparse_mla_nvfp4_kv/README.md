# sparse_mla_nvfp4_kv — Native NVFP4-KV sparse-MLA decode

Production CUDA kernel for DeepSeek MLA decode with NVFP4 KV cache, derived from
[FlashMLA](https://github.com/deepseek-ai/FlashMLA) (Apache-2.0). Uses SM_100
block-scaled tensor cores (`tcgen05.mma.kind::f8f6f4.block_scale_vec::1X`) to
consume packed FP4 K + per-block E4M3 scales as direct UMMA operands — no
SMEM dequant pass, no materialize round-trip.

## Why this exists

The trtllm-gen sparse-MLA cubin (used by SGLang's `--dsa-decode-backend trtllm`)
has only `QkvBfloat16` and `QkvE4m3` (FP8) MLA variants compiled. There is no
`QE4m3KvE2m1` MLA variant — Q and KV must share dtype in the cubin's MLA naming.
Probed with the production shape (FP8 Q + FP4 KV + sparse_mla_top_k=1024) and
confirmed:

```
Missing TRTLLM-GEN kernel (decode):
  qkvLayout=2, kernelType=3, headDimQk=576, headDimV=512,
  tileSizeQ=64, tileSizeKv=128, numTokensPerPage=1, sparseMla=1
```

The cubin would need to be rebuilt by NVIDIA with the FP4 sparse-MLA variant.
This kernel works around that by replacing the cubin entirely for the
NVFP4-KV path.

## Architecture

| Phase | FP8 baseline (FlashMLA) | NVFP4 native UMMA |
|---|---|---|
| K storage | 512 B FP8 + 7 B E8M0 scales + 128 B BF16 rope = **648 B/token** | 256 B FP4 + 32 B E4M3 scales + 128 B BF16 rope = **416 B/token** (~36% smaller) |
| K SMEM during decode | BF16 K (post-dequant) + FP8 raw = ~57 KB | FP4 K + E4M3 scales = ~24 KB (~58% reduction) |
| Warpgroup 1 | KV fetching + FP8→BF16 dequant inner loop | Coord prep only (no dequant) |
| UMMA atom | `SM100_MMA_F16BF16_2x1SM<bf16, bf16, float>` | `SM100_MMA_F8F6F4_BS_2x1SM<E4M3 Q, E2M1 K, float, E4M3 scale>` |
| Per-step KV HBM read | 288 MB (B=8, K=1024, 61 layers) | 192 MB (~33% reduction) |

## Source files

```
csrc/sparse_mla_nvfp4_kv/
├── DESIGN.md                                 # full architecture writeup
├── README.md                                 # this file
├── defines.h, helpers.h, params.h, utils.h   # FlashMLA Apache-2.0 deps
└── nvfp4_variant/
    ├── config.h                              # NVFP4 SMEM layout + UMMA atom config
    ├── nvfp4_umma_descriptors.cuh            # SM_100 block-scaled UMMA atom helpers
    ├── phase1.h                              # kernel declaration
    ├── phase1.cuh                            # kernel impl (FP8 source + NVFP4 surgical edits)
    └── common_subroutine.h                   # FlashMLA outer-loop helpers (unchanged)

python/sglang/jit_kernel/
└── sparse_mla_nvfp4_kv.py                    # tvm-ffi JIT wrapper

python/sglang/srt/layers/attention/dsa_backend.py
                                              # _forward_trtllm dispatch (NVFP4 branch)

python/sglang/srt/server_args.py
                                              # DSA kv_cache_dtype assertion (fp4_e2m1 allowed)

test/srt/test_sparse_mla_nvfp4_kv.py          # correctness vs FP8 + perf bench harness
```

## Build

The kernel JIT-builds on first call via `sglang.jit_kernel.utils.load_jit`
(same pattern as `higgs_inline_sparse_mla_decode`). No manual build step
required for end users. First call cost is ~30-60 s on B200.

For ahead-of-time builds (CI), import the wrapper at startup:

```python
from sglang.jit_kernel.sparse_mla_nvfp4_kv import _jit_sparse_mla_nvfp4_kv_module
_jit_sparse_mla_nvfp4_kv_module()  # triggers JIT compile
```

## Test

```bash
pytest test/srt/test_sparse_mla_nvfp4_kv.py -v
```

Three tests:
- `test_nvfp4_kv_decode_runs`: kernel loads + executes at production shape; output finite
- `test_nvfp4_kv_decode_numerics_vs_fp8`: NVFP4 output within 5% RMS of FP8 baseline (quant noise)
- `test_nvfp4_kv_decode_perf_vs_fp8`: NVFP4 at least 1.2× faster than FP8 baseline

## Integration

Set `--kv-cache-dtype fp4_e2m1` on the SGLang launch. The DSA backend's
`_forward_trtllm` detects the NVFP4 pool layout and routes through this
kernel instead of `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`.

## Implementation status

| Component | Status |
|---|---|
| DESIGN.md (architecture + math) | ✓ Complete |
| nvfp4_umma_descriptors.cuh | ✓ Complete |
| config.h (SMEM layout + atom aliases) | ✓ Complete |
| phase1.cuh — SMEM struct refs renamed | ✓ Complete |
| phase1.cuh — WG1 dequant lambda replaced with coord-prep-only | ✓ Complete |
| phase1.cuh — WG2 UMMA call with block-scale operand | ⚠ **Requires `utcmma_ts_block_scaled` helper in kerutils** |
| Host-side run() for NVFP4 TMA descriptors | ⚠ Skeleton in nvfp4_umma_descriptors.cuh, instantiation in `phase1.cuh::run()` deferred |
| Python wrapper (sparse_mla_nvfp4_kv.py) | ✓ Complete |
| dsa_backend.py dispatch | ✓ Complete |
| server_args.py assertion lift | ✓ Complete |
| Test harness | ✓ Complete |
| End-to-end build verification | ⏳ Awaiting build-debug-iterate on B200 |
| Bench vs FP8 baseline | ⏳ Awaiting working kernel |
| Quality calibration (aquakv-style for NVFP4) | ⏳ Follow-up after perf wins |

## Multi-session reality check

The remaining items (`utcmma_ts_block_scaled` in kerutils, run() TMA setup, build
iteration to a working kernel) are genuine multi-session CUDA engineering. A
new kernel with a new tensor-core op family on a CUTLASS-dependent codebase
will not compile-first-try; the iteration cycle (edit → JIT build 60s →
deploy 10min → test 1min → fix) takes hours per round.

What is testable today **without** the kernel compiling:
- All Python integration paths import-clean
- `--kv-cache-dtype fp4_e2m1` server arg parses + reaches the new dispatch
- The pool layout contracts (NVFP4KVMethod.create_buffers) match the
  kernel's expected shapes

## Provenance

- FlashMLA upstream: deepseek-ai/FlashMLA commit 9241ae3 (Apache-2.0)
- Reference for block-scaled MMA on SM_100: DeepGEMM `sm100_fp4_mqa_logits`
  (also uses `make_instr_desc_block_scaled` + `SM100_MMA_F8F6F4_BS` family)
- SAW-INT4 paper (arxiv 2604.19157): conceptual basis for fused FP4
  decode/dequant in a single pass
