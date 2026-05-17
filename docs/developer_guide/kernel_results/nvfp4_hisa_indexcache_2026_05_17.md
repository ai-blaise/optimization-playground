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
outcome is path correctness: supported 32-head and 64-head packed NVFP4 HISA
queries stay eligible for the DeepGEMM tensor-op path, while unsupported head
counts return `None` and let runtime fallback/skip logic handle the request
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
future HISA optimization pass should benchmark supported 32-head and 64-head
DeepGEMM paths against the Torch reference and only document speedups from
those supported shapes.
