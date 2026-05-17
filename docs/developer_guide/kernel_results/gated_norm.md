# GatedNorm

## Scope

Forward-only BF16 GatedNorm inference for DeepSeek-V3.2-REAP. The path computes
`normed * sigmoid(silu(normed @ w_down.T) @ w_up.T)`.

## Current Result

| Bucket | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| rank8, tokens512 | 0.03706 ms | 0.02887 ms | 1.28x |
| rank16, tokens512 | 0.05347 ms | 0.02808 ms | 1.90x |
| rank32, tokens256 | 0.05957 ms | 0.02822 ms | 2.11x |
| rank31, tokens4 | 0.108578 ms | 0.026668 ms | 4.07x |
| rank15, tokens16 | 0.067015 ms | 0.026727 ms | 2.51x |
| rank6, tokens64 | 0.039024 ms | 0.026351 ms | 1.48x |

Final rank16/tokens16 profiling is launch-floor dominated: first GEMM 3.43 us,
split-K reduce 2.94 us, second GEMM 2.42 us, sigmoid 2.29 us, SiLU 2.06 us,
and multiply 1.64 us.

## Optimization History

- Retuned rank-specific GEMM and low-rank decode thresholds from B200 sweeps.
- Added a production fallback when Python exposes the native wrapper but the
  installed shared object lacks `torch.ops.sgl_kernel.gated_norm_cute_forward`.
- Rejected more aggressive tiny-bucket GEMM dispatch where it tied or regressed.

## Verification

- GatedNorm lane tests passed.
- Stale-binary fallback integration run: 22 passed.
- Final GPU1 integration bundle passed 64 focused tests.

## Next

Further gains need cross-layer or fused-MMA design work; threshold-only tuning
is saturated for the measured low-rank decode buckets.
