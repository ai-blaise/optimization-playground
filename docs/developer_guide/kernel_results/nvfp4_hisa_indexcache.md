# NVFP4 HISA + IndexCache

## Scope

Combined NVFP4 IndexCache + HISA candidate mapping and DeepGEMM dispatch. This
file does not claim a standalone HISA throughput win.

## Current Result

| Shape | Two-Step Incumbent | Current | Speedup |
| --- | ---: | ---: | ---: |
| topk1024 prefix4096 rows1 | 0.013776 ms | 0.011488 ms | 1.20x |
| topk1024 prefix4096 rows32 | 0.013760 ms | 0.011120 ms | 1.24x |
| topk1024 prefix4096 rows1024 | 0.015584 ms | 0.012864 ms | 1.21x |
| topk2048 prefix8192 rows1 | 0.017760 ms | 0.013104 ms | 1.36x |
| topk2048 prefix8192 rows32 | 0.015712 ms | 0.013216 ms | 1.19x |
| topk2048 prefix8192 rows1024 | 0.022768 ms | 0.015136 ms | 1.50x |

Supported 64-head DeepGEMM precomputed HISA measured 0.052403 ms versus
ordinary NVFP4 IndexCache at 0.154877 ms, a 2.96x speedup.

## Optimization History

- Fused HISA exact-pool block mapping and removed redundant exact-pool clears.
- Guarded unsupported DeepGEMM head widths and head counts.
- Rejected 32-thread launch capping because `topk2048/prefix8192/rows1024`
  regressed from 0.015136 ms to 0.029600 ms.
- Rejected 8-head and 32-head DeepGEMM speed claims because current DeepGEMM
  only supports the measured 64-head path for this contract.

## Verification

- Focused pytest artifacts passed 16, 16, and 17 tests across the accepted
  rounds.
- Unsupported 32-head HISA DeepGEMM returned `None` through the guarded fallback.
- Final GPU0 integration bundle covering NVFP4, HISA, and HIGGS passed 27 tests
  with 1 skip.

## Next

Keep 32-head shapes on fallback until DeepGEMM supports the required TMEM loader
width. New claims must stay scoped to supported 64-head paths or exact-pool
mapping.
