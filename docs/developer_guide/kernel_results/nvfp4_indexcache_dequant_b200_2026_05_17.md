# NVFP4 IndexCache Dequant B200 Result, 2026-05-17

## Scope

Standalone NVFP4 IndexCache dequantization from packed E2M1 values and UE8M0
scale words into expanded FP32 rows.

## Environment

- Hardware: NVIDIA B200 (`sm100`)
- Kernel type: direct CUDA JIT
- Profiling: CUDA-event row sweeps and direct IKP instrumentation
- Privacy: synthetic tensors only

## Current Result

Initial CUDA JIT path versus the previous Torch-expression expansion:

| Rows | JIT Dequant | Torch Expansion | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.012256 ms | 0.154848 ms | 12.63x |
| 65,536 | 0.016320 ms | 0.254272 ms | 15.58x |
| 131,072 | 0.024256 ms | 0.426464 ms | 17.58x |

Restart final versus fresh JIT incumbent:

| Rows | Fresh JIT Incumbent | Final | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.011648 ms | 0.011552 ms | 1.0083x |
| 65,536 | 0.015808 ms | 0.015712 ms | 1.0061x |
| 131,072 | 0.023616 ms | 0.022080 ms | 1.0696x |
| 262,144 | 0.041888 ms | 0.036160 ms | 1.1584x |
| 524,288 | 0.072864 ms | 0.061920 ms | 1.1767x |
| 1,048,576 | 0.136544 ms | 0.112160 ms | 1.2174x |

IKP at 1,048,576 rows:

| Region | Fresh Incumbent | Final |
| --- | ---: | ---: |
| load | 1863.5 ns | 78.5 ns |
| decode | 1046.0 ns | 2247.5 ns |
| store | 811.5 ns | 298.5 ns |

## Accepted Changes

- Add CUDA JIT dequant path for NVFP4 IndexCache rows.
- Use 8-lane row mapping for large rows.
- Use branchless sign decode for large rows only.

## Rejected Candidates

| Candidate | Decision |
| --- | --- |
| 512-thread launch block | Rejected because it regressed 64K-1M rows. |
| `__ldg` read-only hints | Rejected because improvements were mixed/noise and did not hold at large rows. |
| Half-warp scale-word broadcast | Rejected because shuffle overhead made all measured sizes slower. |
| Full-warp row mapping | Rejected because larger bandwidth-shaped cases regressed. |
| 4-lane row mapping | Rejected because every measured row count was slower. |
| Contiguous `uint2` word-pair loads | Rejected because store order worsened; 1M rows regressed to 0.150688 ms. |

## Verification

- NVFP4 indexer tests: 3 passed, 1 skipped.
- HISA compatibility smoke: 2 passed.
- Final GPU0 integration bundle covering NVFP4, HISA, and HIGGS passed 27 tests
  with 1 skip.

## Limits And Next Work

This lane is unpack/decode/store work with no matrix multiply, dot product, or
reduction. A tensor-op rewrite is not appropriate for the standalone dequant
path; future tensor-op work belongs in the HISA candidate-scoring path.
