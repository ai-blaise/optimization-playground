# NVFP4 HISA + IndexCache B200 Result, 2026-05-17

## Scope

Combined NVFP4 IndexCache + HISA candidate mapping and DeepGEMM dispatch. The
standalone HISA lane is not claimed as an independent throughput win here.

## Environment

- Hardware: NVIDIA B200 (`sm100`)
- Kernel type: direct CUDA JIT exact-pool mapping plus DeepGEMM FP8/FP4 MQA
- Profiling: CUDA-event exact-pool benchmarks and Nsight/IKP attribution
- Privacy: synthetic tensors only

## Current Result

Exact-pool final versus two-step incumbent:

| Shape | Two-Step Incumbent | Final | Speedup |
| --- | ---: | ---: | ---: |
| topk1024 prefix4096 rows1 | 0.013776 ms | 0.011488 ms | 1.20x |
| topk1024 prefix4096 rows32 | 0.013760 ms | 0.011120 ms | 1.24x |
| topk1024 prefix4096 rows1024 | 0.015584 ms | 0.012864 ms | 1.21x |
| topk2048 prefix8192 rows1 | 0.017760 ms | 0.013104 ms | 1.36x |
| topk2048 prefix8192 rows32 | 0.015712 ms | 0.013216 ms | 1.19x |
| topk2048 prefix8192 rows1024 | 0.022768 ms | 0.015136 ms | 1.50x |

Nsight measured `hisa_block_topk_map_all_indexer_cache_nvfp4` at 8.688 us
median over 10 launches for `topk2048`, `prefix8192`, `rows1024`.

Supported 64-head DeepGEMM path:

| Path | Time | Relative |
| --- | ---: | ---: |
| Ordinary NVFP4 IndexCache | 0.154877 ms | 1.00x |
| HISA precomputed 64-head | 0.052403 ms | 2.96x faster |

## Accepted Changes

- Fuse HISA exact-pool block mapping.
- Avoid redundant HISA exact-pool clears.
- Guard unsupported HISA DeepGEMM head widths and head counts.
- Keep unsupported 32-head DeepGEMM shapes on fallback instead of triggering
  DeepGEMM assertions or compile failures.

## Rejected Candidates

| Candidate | Decision |
| --- | --- |
| Cap fused helper at 32 launch threads | Rejected because `topk2048/prefix8192/rows1024` regressed from 0.015136 ms to 0.029600 ms. |
| Claim 8-head DeepGEMM throughput | Rejected because DeepGEMM only supports 32 or 64 heads for the relevant packed Q contract. |
| Promote 32-head DeepGEMM | Rejected because the current DeepGEMM build fails with `Unsupported TMEM load size`; fallback is guarded. |

## Verification

- Focused pytest artifacts passed 16, 16, and 17 tests across the accepted
  rounds.
- Direct guarded fallback proof showed unsupported 32-head HISA DeepGEMM
  returned `None`.
- Final GPU0 integration bundle covering NVFP4, HISA, and HIGGS passed 27 tests
  with 1 skip.

## Limits And Next Work

Current 32-head HISA DeepGEMM remains fallback-only until the DeepGEMM TMEM
loader supports the required width. Keep new claims scoped to supported 64-head
paths or exact-pool mapping.
