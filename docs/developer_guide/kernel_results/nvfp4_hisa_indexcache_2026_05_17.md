# NVFP4 HISA + IndexCache B200 Result, 2026-05-17

## Scope

Branch: `codex/hisa-indexcache-tensor-op-loop-20260517`

Commits:

```text
0f1ecdc17 Enable NVFP4 HISA for packed small-head queries
6d69fab1e Guard NVFP4 HISA DeepGEMM head counts
```

Hardware: 1x NVIDIA B200 on `root@31.22.104.123`, CUDA 12.8 driver stack.

This iteration fixed NVFP4 HISA paged dispatch shape validation and then
encoded the current DeepGEMM FP8/FP4 MQA support limit: packed Q tensors must
be `[tokens, heads, 64]` with `heads` equal to 32 or 64 before the DeepGEMM
tensor-op path is called.

## Performance Result

No throughput speedup is claimed for this branch. The direct performance
outcome is path correctness: supported 64-head packed NVFP4 HISA
queries stay eligible for the DeepGEMM tensor-op path, while unsupported head
counts, including 32 heads in the current DeepGEMM build, return `None` and let runtime fallback/skip logic handle the request
without triggering a DeepGEMM assertion.

The previous 8-head test attempted to call DeepGEMM directly and failed with:

```text
num_heads == 32 or num_heads == 64
```

That case is now documented as unsupported by DeepGEMM rather than treated as
a benchmarkable fast path. Runtime skip records use
`hisa_nvfp4_paged_skip` with `reason="unsupported_head_count"`.

## Verification

```text
python -m compileall -q python/sglang/jit_kernel/nvfp4_indexer.py \
  python/sglang/srt/layers/attention/nsa/nsa_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py: passed
git diff --check: passed
python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py: 15 passed
```

DeepGEMM emitted only the expected CUDA 12.8 warning recommending NVCC 12.9 for
best performance.

## Caveat

The branch removes a false optimization claim for unsupported head counts. A
future HISA optimization pass should benchmark supported 64-head
DeepGEMM paths against the Torch reference and only document speedups from
those supported shapes.


## Stricter Restart Follow-up

Additional commits:

```text
fa9a1ff60 Fuse HISA exact-pool block mapping
5392bb5d0 Avoid redundant HISA exact-pool clears
8b47d6c1c Guard unsupported HISA DeepGEMM head width
```

Hardware and benchmark provenance: B200 VM `root@31.22.104.123`; GPU 1 for
CUDA-event exact-pool microbenchmarks and pytest; GPU 0 for final Nsight and
64-head DeepGEMM probes. Benchmarks ran under `/root/agent-runs/gpu_locked.sh`.
The exact-pool microbench artifacts are
`/root/agent-runs/hisa-indexcache-round1-mapall-candidate-gpu1.jsonl` and
`/root/agent-runs/hisa-indexcache-round2-tail-clear-candidate-gpu1.jsonl`.

Accepted exact-pool performance, B200 CUDA-event medians:

| Shape | Two-step incumbent ms | Fused accepted ms | Tail-clear final ms | Final speedup |
| --- | ---: | ---: | ---: | ---: |
| topk1024 prefix4096 rows1 | 0.013776 | 0.011488 | 0.011488 | 1.20x |
| topk1024 prefix4096 rows32 | 0.013760 | 0.011776 | 0.011120 | 1.24x |
| topk1024 prefix4096 rows1024 | 0.015584 | 0.013536 | 0.012864 | 1.21x |
| topk2048 prefix8192 rows1 | 0.017760 | 0.015792 | 0.013104 | 1.36x |
| topk2048 prefix8192 rows32 | 0.015712 | 0.015008 | 0.013216 | 1.19x |
| topk2048 prefix8192 rows1024 | 0.022768 | 0.017696 | 0.015136 | 1.50x |

Nsight Systems artifact `/root/agent-runs/hisa-indexcache-final-fused-mapall-nsys.nsys-rep`
with CSV `/root/agent-runs/hisa-indexcache-final-fused-mapall-kernels.csv`
measured `hisa_block_topk_map_all_indexer_cache_nvfp4` at 8.688 us median over
10 launches for topk2048/prefix8192/rows1024.

Rejected candidate: capping the fused helper at 32 launch threads regressed
`topk2048/prefix8192/rows1024` from the accepted 0.015136 ms to 0.029600 ms
(1.96x slower). Rejection artifact:
`/root/agent-runs/hisa-indexcache-round4-threads32-reject-gpu1.jsonl`.

DeepGEMM blocker status: 64-head FP4 MQA compiles and runs; artifact
`/root/agent-runs/hisa-indexcache-round3-deepgemm-head64-gpu0.txt` measured
64-head precomputed HISA at topk2048/prefix8192/query_rows1 as 0.052403 ms
versus ordinary NVFP4 IndexCache 0.154877 ms (2.96x). The 32-head artifacts
`/root/agent-runs/hisa-indexcache-round3-deepgemm-head32-gpu1.txt` and
`/root/agent-runs/hisa-indexcache-restart-deepgemm-head32-gpu0.txt` fail at
DeepGEMM JIT compile time with `Unsupported TMEM load size`, because the kernel
instantiates `kNumHeads / 2 == 16` and its TMEM loader only supports widths 32 or
64. Direct guarded fallback proof is
`/root/agent-runs/hisa-indexcache-round3-direct-head32-fallback-gpu1.txt`, where
the 32-head HISA DeepGEMM call returned `None`.

Verification: focused pytest artifacts
`/root/agent-runs/hisa-indexcache-round1-fused-mapall-pytest-gpu1.txt`,
`/root/agent-runs/hisa-indexcache-round2-tail-clear-pytest-gpu1.txt`, and
`/root/agent-runs/hisa-indexcache-round3-deepgemm-guard-pytest-gpu1.txt` passed
`16 passed`, `16 passed`, and `17 passed`, respectively.
