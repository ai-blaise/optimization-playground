# NVFP4 IndexCache Dequant B200 Result, 2026-05-17

## Scope

Branch: `codex/indexcache-nvfp4-loop-20260517`

Commit: `40afbe8af Add CUDA dequant path for NVFP4 IndexCache`

Hardware: 1x NVIDIA B200 on `root@31.22.104.123`, CUDA 12.8 driver stack.

This iteration added a CUDA JIT dequant path for NVFP4 IndexCache rows while
preserving the Torch CPU fallback. The kernel uses half-warp rows, packed
`uint32` loads for E2M1 values, packed UE8M0 scale-word loads, and `float4`
stores for expanded FP32 rows.

## Benchmark

The benchmark compared the new JIT CUDA dequant path against the previous
Torch-expression CUDA expansion. Each shape checked exact equality
(`rtol=0`, `atol=0`) before timing.

| Rows | JIT dequant median ms | Torch expression median ms | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.012256 | 0.154848 | 12.63x |
| 65,536 | 0.016320 | 0.254272 | 15.58x |
| 131,072 | 0.024256 | 0.426464 | 17.58x |

## Verification

```text
python -m compileall -q python/sglang/jit_kernel/nvfp4_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_indexer.py: passed
git diff --check: passed
python/sglang/jit_kernel/tests/test_nvfp4_indexer.py: 3 passed, 1 skipped
HISA compatibility smoke: 2 passed
```

## Caveat

This is a direct CUDA JIT dequant kernel, not a CuTe/CZS kernel. The change is
limited to IndexCache NVFP4 dequantization and does not alter the HISA selector
policy.
