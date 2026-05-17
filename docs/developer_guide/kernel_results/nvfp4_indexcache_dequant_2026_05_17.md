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

## Restart Under Stricter Closing Premise

Restart branch state: `3f3bbcf0c` was the clean starting point and the best
accepted source incumbent was still `40afbe8af` plus the follow-up docs.  The
renewed loop ran on `root@31.22.104.123` in `/root/work/op-kernel-indexcache`
with explicit GPU locking.  JIT compilation/build steps were run before taking
the GPU lock; CUDA benchmarks, tests, and IKP probes used
`CUDA_VISIBLE_DEVICES={0,1} /root/agent-runs/gpu_locked.sh ...` on the B200 VM.

The benchmark timed only the standalone CUDA dequant call with CUDA events after
warmup.  It checked exact correctness once per run against the CPU/Torch fallback
and avoided per-iteration output checksums or reductions in the timed section.
The row byte model for effective bandwidth is `rows * (64 value bytes + 4 scale
bytes + 128 * sizeof(float) output bytes)`.

### Restart Round 0 Incumbent

Artifact: `/root/agent-runs/indexcache-restart-round0-incumbent-gpu1.jsonl`

| Rows | Incumbent median ms | Min ms | Max ms | Effective GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.011648 | 0.011104 | 0.098144 | 407.91 |
| 65,536 | 0.015808 | 0.015264 | 0.026880 | 2404.53 |
| 131,072 | 0.023616 | 0.022144 | 0.029280 | 3219.08 |
| 262,144 | 0.041888 | 0.039808 | 0.042720 | 3629.76 |
| 524,288 | 0.072864 | 0.072320 | 0.079040 | 4173.35 |
| 1,048,576 | 0.136544 | 0.135392 | 0.142560 | 4454.05 |

Direct IKP was feasible and was used.  The fresh incumbent IKP probe at
1,048,576 rows is recorded in
`/root/agent-runs/indexcache-restart-round0-incumbent-ikp-gpu1.json` and
`/root/agent-runs/indexcache-restart-round0-incumbent-ikp-gpu1_summary.json`.
The instrumented event timing printed 0.142347 ms/iter.  Sampled region means
were: total 4,444.5 ns, load 1,863.5 ns, decode 1,046.0 ns, store 811.5 ns.

### Restart Optimization Rounds

Each round compared against the latest accepted source incumbent.  Accepted
source commits were made before the next source candidate.

| Round | Candidate | Artifact | 8K / 64K / 128K / 256K / 512K / 1M medians ms | Decision |
| ---: | --- | --- | --- | --- |
| 1 | Dequant launch block size 512 instead of 256. | `/root/agent-runs/indexcache-restart-round1-block512-gpu1.jsonl` | 0.011616 / 0.017600 / 0.024000 / 0.042944 / 0.076800 / 0.142592 | Rejected: tied 8K but regressed 64K-1M versus 0.011648 / 0.015808 / 0.023616 / 0.041888 / 0.072864 / 0.136544. |
| 2 | `__ldg` read-only hints for packed value and scale loads. | `/root/agent-runs/indexcache-restart-round2-ldg-gpu1.jsonl` | 0.011840 / 0.015584 / 0.023712 / 0.041824 / 0.072992 / 0.136576 | Rejected: 64K/256K were noise-scale improvements, 8K and 128K regressed, and 512K/1M did not improve. |
| 3 | 8-lane row mapping: four rows per warp, each lane decodes two packed words. | `/root/agent-runs/indexcache-restart-round3-8lane-gpu1.jsonl` | 0.011712 / 0.015904 / 0.022240 / 0.036768 / 0.063456 / 0.116864 | Accepted in `029407226` because 128K-1M improved by 5.83% / 12.22% / 12.91% / 14.41%; 8K/64K were within small-shape noise. |
| 4 | Branchless sign decode for all rows, then guarded to large rows only. | `/root/agent-runs/indexcache-restart-round4-branchless-sign-gpu1.jsonl`, `/root/agent-runs/indexcache-restart-round4-branchless-large-gpu1.jsonl` | Guarded: 0.011552 / 0.015616 / 0.021888 / 0.036128 / 0.061760 / 0.112160 | Accepted in `1c37192e0`. The unguarded variant regressed 64K/128K; the guarded source kept original decode below 262,144 rows and branchless decode at 262,144+ rows. |
| 5 | 4-lane row mapping: eight rows per warp, each lane decodes four packed words. | `/root/agent-runs/indexcache-restart-round5-4lane-gpu1.jsonl` | 0.011904 / 0.017568 / 0.023680 / 0.038976 / 0.067008 / 0.122112 | Rejected: slower than the current accepted incumbent at every measured row count. |
| 6 | Contiguous `uint2` word-pair load per lane. | `/root/agent-runs/indexcache-restart-round6-pairload-gpu1.jsonl` | 0.011680 / 0.019776 / 0.028000 / 0.046656 / 0.081952 / 0.150688 | Rejected: packed-load vectorization worsened store address order; 1M regressed by 34.36% versus 0.112160 ms. |

### Final Restart Result

Final source commits from the restart:

- `029407226 Optimize NVFP4 IndexCache dequant row mapping`
- `1c37192e0 Gate branchless NVFP4 dequant decode for large rows`

Final benchmark artifact:
`/root/agent-runs/indexcache-restart-final-incumbent-gpu1.jsonl`.

| Rows | Round 0 incumbent ms | Final median ms | Speedup | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.011648 | 0.011552 | 1.0083x | 0.82% |
| 65,536 | 0.015808 | 0.015712 | 1.0061x | 0.61% |
| 131,072 | 0.023616 | 0.022080 | 1.0696x | 6.50% |
| 262,144 | 0.041888 | 0.036160 | 1.1584x | 13.67% |
| 524,288 | 0.072864 | 0.061920 | 1.1767x | 15.02% |
| 1,048,576 | 0.136544 | 0.112160 | 1.2174x | 17.86% |

Final direct IKP profile for the accepted large-row branchless path used the
instrumented standalone probe
`/root/agent-runs/indexcache_restart_final_branchless_ikp_dequant_probe.cu` and
wrote `/root/agent-runs/indexcache-restart-final-branchless-ikp-gpu1.json` plus
`/root/agent-runs/indexcache-restart-final-branchless-ikp-gpu1_summary.json`.
At 1,048,576 rows, the instrumented event loop printed 0.139575 ms/iter.  IKP
sampled region means were: total 3,215.0 ns, load 78.5 ns, decode 2,247.5 ns,
and store 298.5 ns.  The load collapse versus the round-0 1,863.5 ns load mean
matches the accepted 8-lane mapping, while the remaining decode-heavy profile
matches the final rejection of more aggressive row mapping and load-vectorized
word-pair stores.

### Final Restart Verification

```text
python -m py_compile \
  python/sglang/jit_kernel/nvfp4_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_indexer.py \
  python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py:
  passed

git diff --check: passed

CUDA_VISIBLE_DEVICES=0 /root/agent-runs/gpu_locked.sh \
  python -m pytest -q python/sglang/jit_kernel/tests/test_nvfp4_indexer.py -rs:
  3 passed, 1 skipped in 3.21s

CUDA_VISIBLE_DEVICES=1 /root/agent-runs/gpu_locked.sh \
  python -m pytest -q \
  python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py::test_nvfp4_hisa_defaults_to_ordinary_indexcache_when_t_le_k[2048] \
  python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py::test_nvfp4_hisa_precomputed_reps_match_dequantized_mean_pool -rs:
  2 passed in 8.33s
```

### Stop Rationale

The final accepted kernel is still a standalone unpack/expand path, not a
GEMM-shaped operation.  A tensor-op or UMMA rewrite is inappropriate for this
specific dequant lane because the work is packed E2M1/UE8M0 decode plus FP32
stores with no matrix multiply, dot product, or reduction.  CuTe/CZS proof was
therefore not produced for the accepted source because the source remains direct
CUDA JIT; CZS remains required for future CuTe kernels.

The renewed loop stops after six empirical rounds because the plausible remaining
surfaces were measured against the best accepted incumbent and either accepted
or rejected with concrete timings: larger CTA geometry regressed, read-only load
hints did not improve the large rows, 4-lane row mapping regressed all rows, and
contiguous pair-load vectorization regressed 1M rows from 0.112160 ms to
0.150688 ms.  Final IKP at 1M rows shows the packed load region is already only
78.5 ns mean in the instrumented sample, so further load-side tweaks are unlikely
to move end-to-end latency without hurting the store order.  The final event
benchmark reaches 5,422.38 effective GB/s at 1,048,576 rows for 580 useful bytes
per row, and repeated attempts to trade coalesced store order for fewer loads
lost by 8.22%-34.36% on 512K-1M rows.
