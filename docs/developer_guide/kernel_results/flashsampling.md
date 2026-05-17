# FlashSampling

## Scope

TP-sharded decode sampling on Blackwell B200. The production-relevant shape is
`V=16160`, `D=7168`, BF16 weights/hidden states, and greedy or exact sampling.

## Current Result

| Case | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| Target provider BS1 | 0.049059 ms generic target | 0.047400 ms Blackwell target | 1.035x |
| Target provider BS32 | 0.048544 ms generic target | 0.047206 ms Blackwell target | 1.028x |
| Greedy H2/H4/H8 | 0.045983 / 0.045770 / 0.045824 ms | 0.045243 / 0.045088 / 0.045118 ms | 1.015x-1.016x |
| Non-greedy H1 | 0.047392 ms | 0.045498 ms | 1.042x |

The final target path also beats the dense matmul plus argmax floor by
1.057x, 1.088x, and 1.067x at batches 1, 32, and 64.

## Optimization History

- Routed the `target` provider to the Blackwell kernel on SM100.
- Retuned `BLOCK_H=8` for greedy `2 <= H <= 8` and non-greedy `H=1`.
- Rejected `BLOCK_D`, `BLOCK_V`, stage-count, warp-count, two-wave target,
  `BLOCK_H=128`, local-reduce fusion, and `maxnreg` variants because they tied
  or regressed the current path.

## Verification

- Direct kernel tests cover greedy equality, logits debug mode, sampled-id
  range, seed sensitivity, shard offsets, and compact local-index workspace.
- Final GPU1 integration bundle passed 64 focused tests across FlashSampling,
  G1, GatedNorm, and WarpDecode.

## Next

Rerun end-to-end B200 TPOT after the production image rebuilds `sgl-kernel`;
the kernel-only result is valid, but the VM session had a stale shared install.
