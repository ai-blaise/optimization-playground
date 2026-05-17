# FlashSampling B200 Result, 2026-05-17

## Scope

FlashSampling target-provider optimization for TP-sharded decode shapes on
Blackwell B200. The production-relevant shape was `V=16160`, `D=7168`, BF16
weights and hidden states, greedy or exact sampling.

## Environment

- Hardware: NVIDIA B200 (`sm100`)
- Driver/toolchain: CUDA 12.8/13.x stack as available on the B200 VM
- Benchmark style: CUDA-event kernel timing under an explicit GPU lock
- Privacy: synthetic tensors only

## Current Result

| Comparison | Baseline | Final | Speedup |
| --- | ---: | ---: | ---: |
| Target provider BS1 | 0.049059 ms generic target | 0.047400 ms Blackwell target | 1.035x |
| Target provider BS32 | 0.048544 ms generic target | 0.047206 ms Blackwell target | 1.028x |
| Greedy H2 | 0.045983 ms old `BLOCK_H=16` | 0.045243 ms `BLOCK_H=8` | 1.016x |
| Greedy H4 | 0.045770 ms old `BLOCK_H=16` | 0.045088 ms `BLOCK_H=8` | 1.015x |
| Greedy H8 | 0.045824 ms old `BLOCK_H=16` | 0.045118 ms `BLOCK_H=8` | 1.016x |
| Non-greedy H1 | 0.047392 ms old path | 0.045498 ms `BLOCK_H=8` | 1.042x |

Post-restart target path versus dense matmul plus argmax floor:

| Batch | Dense floor | Final target path | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.049364 ms | 0.046680 ms | 1.057x |
| 32 | 0.051334 ms | 0.047191 ms | 1.088x |
| 64 | 0.053421 ms | 0.050074 ms | 1.067x |

Final IKP attribution for non-greedy H1 measured
`flashsample_blackwell_kernel` at 39.937 us mean and
`_local_reduce_samples_kernel` at 2.415 us mean.

## Accepted Changes

- Route the `target` provider to the Blackwell target kernel on SM100.
- Use `BLOCK_H=8` for greedy `2 <= H <= 8`.
- Use `BLOCK_H=8` for non-greedy `H=1`.

## Rejected Candidates

| Candidate | Decision |
| --- | --- |
| `BLOCK_D=64`, `BLOCK_V=256`, fixed stage counts, and warp-count changes | Rejected because they regressed or tied the accepted Blackwell target provider. |
| Force non-persistent target dispatch beyond one SM wave | Rejected because H128/H256 reread weights and were much slower. |
| `BLOCK_H=128` for H128/H256 | Rejected because accumulator pressure outweighed fewer H tiles. |
| Local-reduce CTA fusion for small H | Rejected because the reduction launch was already at the launch floor. |
| `maxnreg=255` | Rejected as measurement noise without stable improvement. |

## Verification

- Direct kernel tests cover greedy equality, logits debug mode, sampled-id range,
  seed sensitivity, vocabulary-shard offsets, and compact local index workspace
  dtype.
- Final integration bundle on GPU1 passed 64 focused tests across
  FlashSampling, G1, GatedNorm, and WarpDecode.

## Limits And Next Work

Server-level B200 TPOT benchmarking was blocked by a stale shared `sgl-kernel`
install during the VM session. The standalone kernel result is valid, but a
production image should rebuild `sgl-kernel` and rerun end-to-end serving.
