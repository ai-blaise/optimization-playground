# NVFP4 IndexCache Dequant

## Scope

Standalone NVFP4 IndexCache dequantization from packed E2M1 values and UE8M0
scale words into expanded FP32 rows.

## Current Result

Initial CUDA JIT path versus Torch-expression expansion:

| Rows | JIT Dequant | Torch Expansion | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.012256 ms | 0.154848 ms | 12.63x |
| 65,536 | 0.016320 ms | 0.254272 ms | 15.58x |
| 131,072 | 0.024256 ms | 0.426464 ms | 17.58x |

Restart final versus fresh JIT incumbent:

| Rows | Fresh JIT | Current | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.011648 ms | 0.011552 ms | 1.0083x |
| 65,536 | 0.015808 ms | 0.015712 ms | 1.0061x |
| 131,072 | 0.023616 ms | 0.022080 ms | 1.0696x |
| 262,144 | 0.041888 ms | 0.036160 ms | 1.1584x |
| 524,288 | 0.072864 ms | 0.061920 ms | 1.1767x |
| 1,048,576 | 0.136544 ms | 0.112160 ms | 1.2174x |

## Optimization History

- Added the CUDA JIT dequant path.
- Accepted 8-lane row mapping for large rows and branchless sign decode for
  large rows only.
- Rejected 512-thread launch blocks, `__ldg` hints, scale-word broadcast,
  full-warp rows, 4-lane rows, and contiguous `uint2` pair loads.

## Verification

- NVFP4 indexer tests: 3 passed, 1 skipped.
- HISA compatibility smoke: 2 passed.
- Final GPU0 integration bundle covering NVFP4, HISA, and HIGGS passed 27 tests
  with 1 skip.

## Next

This path is unpack/decode/store work, not a tensor-op target. Put future
tensor-op work in the HISA candidate-scoring path.
