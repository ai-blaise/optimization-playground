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

## Follow-Up Optimization Rounds

Follow-up branch state: `c007b8ae4` merged current `origin/main` into
`codex/indexcache-nvfp4-loop-20260517` before measurement. The accepted
incumbent for these rounds remained `40afbe8af`, because none of the
profiler-guided source candidates beat it across the measured row sizes.

Locked incumbent rerun after the merge:

| Rows | Incumbent JIT dequant median ms | Torch expression median ms | Speedup |
| ---: | ---: | ---: | ---: |
| 8,192 | 0.011184 | 0.148192 | 13.25x |
| 65,536 | 0.016928 | 0.257328 | 15.20x |
| 131,072 | 0.023248 | 0.428384 | 18.43x |
| 262,144 | 0.041504 | 0.742720 | 17.90x |

Direct IKP attribution was feasible and was used for the dequant follow-up.
An instrumented copy of the kernel (`/root/agent-runs/nvfp4_indexcache_ikp_dequant_probe.cu`)
recorded regions around load, decode, and store. On 131,072 rows, the sampled
region means were:

| IKP region | Mean ns | p50 ns | Interpretation |
| --- | ---: | ---: | --- |
| load | 1,724.0 | 1,796.0 | Packed value and scale global loads dominate the local arithmetic. |
| decode | 479.5 | 480.9 | E2M1 table decode/sign application is smaller than load/store. |
| store | 1,025.5 | 837.0 | Expanded FP32 `float4` stores are a major fixed cost. |

Round decisions compared each candidate with the accepted incumbent, not the
original Torch-expression baseline:

| Round | Profiler evidence | Candidate | Candidate median ms at 8K / 64K / 128K / 256K | Decision |
| ---: | --- | --- | --- | --- |
| 1 | IKP load/store dominance suggested reducing scale loads. | Half-warp scale-word broadcast: one lane loads the scale and shuffles within the half warp. | 0.018064 / 0.020160 / 0.028336 / 0.045888 | Rejected. Correct but slower than incumbent at every size. |
| 2 | IKP store/decode shape suggested testing a lower per-lane decode width with a full-warp row. | Full-warp row mapping: `uint16` packed loads, four decoded values per lane, one `float4` store per lane. | 0.011136 / 0.017184 / 0.025440 / 0.047840 | Rejected. Tied 8K within noise but regressed larger bandwidth-shaped cases. |
| 3 | IKP showed decode was smaller but still the only arithmetic region; tested whether sign application was compiler-limited. | Branchless sign-bit XOR in `decode_e2m1_nibble`. | First pass: 0.011392 / 0.015344 / 0.023392 / 0.041600. Repeat: 0.011152 / 0.018000 / 0.024576 / 0.043360. | Rejected. Mixed/noisy at 8K and 64K, and repeated larger-shape regressions. |

No follow-up source candidate was accepted, so no new dequant source commit was
made. Rejected candidates were removed from the source tree. The stopping
evidence is that the accepted half-warp kernel is already near the load/store
floor for this standalone unpack path; attempts to reduce one part either added
shuffle overhead or reduced memory efficiency. A tensor-op rewrite is not
appropriate for this standalone dequant expansion, because it has no
GEMM-shaped computation. The HISA candidate-scoring path remains the
DeepGEMM/tensor-op route documented in `nvfp4_hisa_indexer.md`.

Follow-up artifacts:

- `/root/agent-runs/nvfp4-indexcache-round1-incumbent-ikp.json`
- `/root/agent-runs/nvfp4-indexcache-round1-incumbent-ikp_summary.json`
- `/root/agent-runs/nvfp4-indexcache-round1-scale-broadcast-bench.json`
- `/root/agent-runs/nvfp4-indexcache-round2-full-warp-row-bench.json`
- `/root/agent-runs/nvfp4-indexcache-round3-branchless-sign-bench.json`
- `/root/agent-runs/nvfp4-indexcache-round3-branchless-sign-repeat.json`
- `/root/agent-runs/nvfp4-indexcache-followup-final-incumbent-bench.json`

Final clean-source verification after removing rejected candidates:

```text
python -m compileall -q python/sglang/jit_kernel/nvfp4_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py: passed
git diff --check: passed
python -m pytest -q python/sglang/jit_kernel/tests/test_nvfp4_indexer.py \
  test_nvfp4_hisa_indexer.py::{defaults_to_ordinary_indexcache_when_t_le_k[2048],
  precomputed_reps_match_dequantized_mean_pool}: 5 passed, 1 skipped
```

Final clean-source dequant medians were 0.011040 / 0.016416 / 0.023184 /
0.041216 ms at 8,192 / 65,536 / 131,072 / 262,144 rows.
