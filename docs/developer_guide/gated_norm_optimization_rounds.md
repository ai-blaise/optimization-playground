# GatedNorm Optimization Rounds

Hardware: NVIDIA B200 (SM100), single GPU via `CUDA_VISIBLE_DEVICES=0` under `/root/agent-runs/gpu_locked.sh`.
Software: branch `codex/gatednorm-loop-20260517`, CUDA runtime from the shared SGLang venv. Benchmark tensors are BF16, hidden size 7168.

## Round 0: incumbent

Incumbent: `origin/codex/gatednorm-cute` at `8caf98336` (optimization-playground state before this worker). Dispatch used Triton for rank 8/16 until 2048 tokens, rank 32 until 512 tokens, and torch/cuBLAS GEMMs above those thresholds.

Correctness baseline: `python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py` -> 21 passed.

## Round 1: lower GEMM dispatch thresholds

Profiler/instrumentation: CUDA-event path sweep comparing forced Triton vs torch/cuBLAS GEMM path. IKP was not used in this round because the hotspot was Python dispatch selection between existing paths, not a single CUDA kernel region; CUDA-event path isolation is the direct measurement.

Hotspot/result: production rank 16 stayed on the scalar/reduction Triton fallback at 64-1024 tokens, while torch/cuBLAS GEMMs were faster.

Candidate: change `_default_torch_mm_min_tokens` to dispatch rank >=16 to GEMMs from 64 tokens and rank >=8 from 512 tokens.

Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc 'source /root/work/optimization-playground/.venv/bin/activate; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:$PYTHONPATH; CUDA_VISIBLE_DEVICES=0 python <cuda-event sweep>'
```

Measured before/after versus the Round 0 incumbent:

| rank | tokens | incumbent ms | candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 8 | 512 | 0.03706 | 0.02887 | 1.28x | keep |
| 8 | 1024 | 0.04118 | 0.03607 | 1.14x | keep |
| 16 | 64 | 0.03914 | 0.02863 | 1.37x | keep |
| 16 | 128 | 0.04088 | 0.02842 | 1.44x | keep |
| 16 | 256 | 0.04938 | 0.02857 | 1.73x | keep |
| 16 | 512 | 0.05347 | 0.02808 | 1.90x | keep |
| 16 | 1024 | 0.06169 | 0.03603 | 1.71x | keep |
| 32 | 64 | 0.04528 | 0.02827 | 1.60x | keep |
| 32 | 128 | 0.04536 | 0.02819 | 1.61x | keep |
| 32 | 256 | 0.05957 | 0.02822 | 2.11x | keep |

Neutral cases where both incumbent and candidate already used GEMM: rank 8/16/32 at 2048+ tokens and rank 32 at 512+ tokens were within about 1%.

Correctness verification: candidate output was checked against the incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row. Full pytest: `python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py`.

New incumbent commit: `bb4aefed5` (`Tune GatedNorm GEMM thresholds`).

## CZS proof surface

Artifact: `docs/proofs/gated_norm_mma_czs_module.json`.

Command:

```bash
python scripts/playground/verify_gated_norm_czs.py
```

Result: 15 CZS obligations proved (9 layout legality, 2 vectorization, 4 ldmatrix legality). Caveat: CZS 0.4.1 does not model the inline `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` atom used by the hand-written CUDA path, so this proves the available shared-memory/vectorization/ldmatrix surface, not the legacy inline MMA atom itself.

## Round 2: fuse GEMM epilogue sigmoid and multiply

Incumbent: `bb4aefed5` (Round 1 GEMM threshold tuning).

Profiler/instrumentation: Nsight Systems kernel summary on rank 16, 2048 tokens after Round 1. IKP was not used for this round because the hotspot spans PyTorch/cuBLAS launches and a PyTorch elementwise epilogue rather than a source-instrumentable in-repo CUDA kernel. Replacement evidence: `nsys stats --report cuda_gpu_kern_sum /root/agent-runs/gatednorm-nsys-round1.sqlite`.

Hotspot/result: after threshold tuning, the torch/cuBLAS path still launched separate sigmoid and multiply kernels after the second GEMM. The epilogue accounted for a material share of GPU time; a single fused elementwise pass should remove one launch and one memory round trip.

Candidate: add a Triton `_sigmoid_mul_kernel` and dispatch it only for torch-MM shapes with at least 1024 tokens. Keep the incumbent PyTorch epilogue for smaller shapes where the fused launch was previously measured as neutral or slower.

Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '. /root/.cargo/env; source /root/work/optimization-playground/.venv/bin/activate; export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:/root/work/op-kernel-gatednorm/sgl-kernel/python:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 python /tmp/gatednorm_round2_bench_fair.py'
```

Measured before/after versus the Round 1 incumbent with the same public `gated_norm_forward` entrypoint, toggling only the fusion threshold:

| rank | tokens | incumbent ms | candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 8 | 512 | 0.028341 | 0.027942 | 1.014x | same, keep incumbent path |
| 8 | 1024 | 0.037190 | 0.031888 | 1.166x | keep |
| 8 | 2048 | 0.045937 | 0.032865 | 1.398x | keep |
| 8 | 4096 | 0.086844 | 0.059510 | 1.459x | keep |
| 16 | 512 | 0.027783 | 0.027777 | 1.000x | same, keep incumbent path |
| 16 | 1024 | 0.035941 | 0.031191 | 1.152x | keep |
| 16 | 2048 | 0.045909 | 0.032203 | 1.426x | keep |
| 16 | 4096 | 0.085015 | 0.057963 | 1.467x | keep |
| 32 | 512 | 0.027943 | 0.027736 | 1.007x | same, keep incumbent path |
| 32 | 1024 | 0.035961 | 0.031127 | 1.155x | keep |
| 32 | 2048 | 0.047616 | 0.033642 | 1.415x | keep |
| 32 | 4096 | 0.084625 | 0.057524 | 1.471x | keep |

Correctness verification: candidate output matched the incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row.

Decision: accept. New incumbent commit: `fbfb08dc9` (`Fuse GatedNorm torch-MM epilogue`).

## Round 3: epilogue tile size 2048

Incumbent: `fbfb08dc9` (Round 2 fused epilogue with `BLOCK=1024`).

Profiler/instrumentation: CUDA-event candidate sweep against the accepted incumbent. IKP source markers were not feasible for this Triton/cuBLAS path: Triton JIT kernels cannot include IKP C++ trace macros, and the CUDA extension path could not be loaded on this VM because `sgl-kernel` has no local `common_ops` build.

Hotspot/result: final profiling still showed `_sigmoid_mul_kernel` as the largest individual kernel at rank 16, 2048 tokens, so a larger one-dimensional epilogue tile was tested to reduce program count.

Candidate: change the fused epilogue launch from `BLOCK=1024` to `BLOCK=2048`.

Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '. /root/.cargo/env; source /root/work/optimization-playground/.venv/bin/activate; export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:/root/work/op-kernel-gatednorm/sgl-kernel/python:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 python /tmp/gatednorm_round3_block2048_fair.py'
```

Measured before/after versus the Round 2 incumbent using identical direct torch-MM pipelines except for epilogue `BLOCK`:

| rank | tokens | incumbent ms | candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 8 | 1024 | 0.029500 | 0.029466 | 1.001x | reject |
| 8 | 2048 | 0.032888 | 0.032826 | 1.002x | reject |
| 8 | 4096 | 0.059647 | 0.059487 | 1.003x | reject |
| 16 | 1024 | 0.028697 | 0.028566 | 1.005x | reject |
| 16 | 2048 | 0.031569 | 0.031521 | 1.002x | reject |
| 16 | 4096 | 0.058484 | 0.057827 | 1.011x | reject/no broad win |
| 32 | 1024 | 0.027467 | 0.027450 | 1.001x | reject |
| 32 | 2048 | 0.033524 | 0.033129 | 1.012x | reject/no broad win |
| 32 | 4096 | 0.057885 | 0.057498 | 1.007x | reject |

Correctness verification: candidate output matched the incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row.

Decision: reject. The only rows above 1% were isolated and the broad sweep was within measurement noise. Source remains at the Round 2 incumbent.

## Round 4: Triton `tl.sigmoid` intrinsic in epilogue

Incumbent: `fbfb08dc9` (Round 2 fused epilogue).

Profiler/instrumentation: CUDA-event candidate sweep; IKP source markers remained infeasible for the same Triton/cuBLAS reasons as Round 3.

Hotspot/result: `_sigmoid_mul_kernel` remained the largest individual kernel, and the epilogue computes sigmoid explicitly as `1 / (1 + exp(-x))`.

Candidate: replace the explicit sigmoid expression with Triton's `tl.sigmoid(logits)` intrinsic in a temporary candidate kernel.

Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '. /root/.cargo/env; source /root/work/optimization-playground/.venv/bin/activate; export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:/root/work/op-kernel-gatednorm/sgl-kernel/python:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 python /tmp/gatednorm_round4_tl_sigmoid.py'
```

Measured before/after versus the Round 2 incumbent epilogue:

| rank | tokens | incumbent ms | candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 8 | 1024 | 0.029396 | 0.029138 | 1.009x | reject |
| 8 | 2048 | 0.032979 | 0.032938 | 1.001x | reject |
| 8 | 4096 | 0.059603 | 0.059476 | 1.002x | reject |
| 16 | 1024 | 0.028699 | 0.028706 | 1.000x | reject |
| 16 | 2048 | 0.031774 | 0.031390 | 1.012x | reject/no broad win |
| 16 | 4096 | 0.057874 | 0.057792 | 1.001x | reject |
| 32 | 1024 | 0.026736 | 0.026684 | 1.002x | reject |
| 32 | 2048 | 0.033502 | 0.032988 | 1.016x | reject/no broad win |
| 32 | 4096 | 0.057593 | 0.057567 | 1.000x | reject |

Correctness verification: candidate output matched the incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row.

Decision: reject. The intrinsic form did not produce a broad or stable speedup. No source change.

## Round 5: Triton epilogue `num_warps` launch tuning

Incumbent: `fbfb08dc9` (Round 2 fused epilogue).

Profiler/instrumentation: CUDA-event launch-configuration sweep; IKP source markers remained infeasible for the Triton/cuBLAS path.

Hotspot/result: the fused epilogue remained visible in Nsight/IKP-imported system attribution, so launch configuration was the remaining low-risk candidate before a larger CUTLASS/CuTe rewrite.

Candidate: explicitly launch `_sigmoid_mul_kernel` with `num_warps=4` and `num_warps=8` instead of the default.

Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '. /root/.cargo/env; source /root/work/optimization-playground/.venv/bin/activate; export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:/root/work/op-kernel-gatednorm/sgl-kernel/python:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 python /tmp/gatednorm_round5_numwarps.py'
```

Measured before/after versus the Round 2 incumbent default launch:

| rank | tokens | incumbent ms | best candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 8 | 1024 | 0.029150 | 0.029222 | 0.998x | reject |
| 8 | 2048 | 0.032946 | 0.032895 | 1.002x | reject |
| 8 | 4096 | 0.059512 | 0.059697 | 0.997x | reject |
| 16 | 1024 | 0.028731 | 0.028692 | 1.001x | reject |
| 16 | 2048 | 0.031485 | 0.031279 | 1.007x | reject |
| 16 | 4096 | 0.058203 | 0.057609 | 1.010x | reject/no broad win |
| 32 | 1024 | 0.026794 | 0.026721 | 1.003x | reject |
| 32 | 2048 | 0.033337 | 0.033161 | 1.005x | reject |
| 32 | 4096 | 0.057834 | 0.057562 | 1.005x | reject |

Correctness verification: both `num_warps=4` and `num_warps=8` outputs matched the incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row.

Decision: reject. `num_warps=8` regressed larger rows and `num_warps=4` was effectively the default/noise floor. No source change.

## Round 6: rank >=32 decode GEMM dispatch

Incumbent: `11922dcd9` (post-merge chain with Round 2 accepted threshold and epilogue fusion).

Profiler/instrumentation: CUDA-event path sweep on both B200s using the updated per-GPU locks (`gpu_locked_any.sh`, then explicit GPU 1 and GPU 0 repeats). IKP source markers remained infeasible for this dispatch-level candidate because the compared path chooses between Triton JIT and PyTorch/cuBLAS launches rather than a single in-repo CUDA kernel region. Replacement evidence is direct same-device CUDA-event timing against the `11922dcd9` dispatch thresholds.

Hotspot/result: the stricter restart sweep found the previous closure did not cover high-rank decode shapes below the accepted GEMM thresholds. Forced torch/cuBLAS was materially faster for rank 32/40/48 at 1-32 tokens and rank 64 at 1-8 tokens, while the old/current paths tied once the incumbent already used torch/cuBLAS.

Candidate: change `_default_torch_mm_min_tokens` so `rank >= 32` dispatches to torch/cuBLAS from 1 token. Keep rank 16 at 64 tokens, rank 8 at 512 tokens, and rank 1 at 4096 tokens.

Exploratory command and artifact:

```bash
/root/agent-runs/gpu_locked_any.sh bash -lc '<env>; python <path sweep>'
# /root/agent-runs/gatednorm-restart-round6-path-sweep.jsonl
```

Accepted comparison command and artifact:

```bash
CUDA_VISIBLE_DEVICES=0 /root/agent-runs/gpu_locked.sh bash -lc '<env>; python <candidate-vs-old-threshold sweep>'
# /root/agent-runs/gatednorm-restart-round6-accepted-bench.jsonl
```

Measured candidate default versus old `11922dcd9` thresholds, median of 5 same-device CUDA-event repeats:

| rank | tokens | incumbent ms | candidate ms | speedup | decision |
|---:|---:|---:|---:|---:|---|
| 32 | 1 | 0.044897 | 0.028095 | 1.60x | keep |
| 32 | 4 | 0.045130 | 0.027973 | 1.61x | keep |
| 32 | 8 | 0.045152 | 0.028088 | 1.61x | keep |
| 32 | 16 | 0.045162 | 0.027854 | 1.62x | keep |
| 32 | 32 | 0.045130 | 0.028171 | 1.60x | keep |
| 32 | 64 | 0.028101 | 0.028006 | 1.00x | tie |
| 40 | 1 | 0.053328 | 0.028374 | 1.88x | keep |
| 40 | 4 | 0.053353 | 0.028011 | 1.90x | keep |
| 40 | 8 | 0.053334 | 0.028168 | 1.89x | keep |
| 40 | 16 | 0.054215 | 0.028154 | 1.93x | keep |
| 40 | 32 | 0.053534 | 0.027807 | 1.93x | keep |
| 40 | 64 | 0.028059 | 0.028006 | 1.00x | tie |
| 48 | 1 | 0.055366 | 0.028534 | 1.94x | keep |
| 48 | 4 | 0.053773 | 0.028351 | 1.90x | keep |
| 48 | 8 | 0.053715 | 0.028367 | 1.89x | keep |
| 48 | 16 | 0.054182 | 0.028147 | 1.92x | keep |
| 48 | 32 | 0.055249 | 0.028144 | 1.96x | keep |
| 48 | 64 | 0.028558 | 0.028316 | 1.01x | tie |
| 64 | 1 | 0.055746 | 0.028197 | 1.98x | keep |
| 64 | 4 | 0.056225 | 0.028227 | 1.99x | keep |
| 64 | 8 | 0.056503 | 0.028234 | 2.00x | keep |
| 64 | 16 | 0.028016 | 0.027801 | 1.01x | tie |
| 64 | 32 | 0.028037 | 0.027890 | 1.01x | tie |
| 64 | 64 | 0.028194 | 0.028047 | 1.01x | tie |

Correctness verification: candidate output matched the old-threshold incumbent output with `torch.testing.assert_close(..., atol=2e-2, rtol=2e-2)` for every measured row. Focused pytest: `python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py` -> 22 passed.

Decision: accept. New incumbent commit: this commit.

## Pre-restart stop evidence (superseded)

Accepted incumbent before the stricter restart: `fbfb08dc9`, merged into `11922dcd9`. Final pre-restart profile command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '. /root/.cargo/env; source /root/work/optimization-playground/.venv/bin/activate; export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}; export PYTHONPATH=/root/work/op-kernel-gatednorm/python:/root/work/op-kernel-gatednorm/sgl-kernel/python:${PYTHONPATH:-}; CUDA_VISIBLE_DEVICES=0 nsys profile --force-overwrite=true -o /root/agent-runs/gatednorm-final-r16-t2048 python /tmp/gatednorm_profile_final.py'
python3 /root/work/rule7-refs/intra-kernel-profiler/scripts/ikp_nsys_import.py --nsys-rep /root/agent-runs/gatednorm-final-r16-t2048.nsys-rep --out-dir /root/agent-runs/gatednorm-final-ikp --skip-export
```

Artifacts: `/root/agent-runs/gatednorm-final-r16-t2048.nsys-rep`, `/root/agent-runs/gatednorm-final-r16-t2048.sqlite`, `/root/agent-runs/gatednorm-final-r16-t2048-kernsum.txt`, `/root/agent-runs/gatednorm-final-ikp/nsys_kernels.json`.

Final rank 16, 2048-token kernel summary after Round 2: `_sigmoid_mul_kernel` 36.6% (11.62 us average), first cuBLAS GEMM 24.3% (7.70 us), second cuBLAS GEMM 19.5% (6.19 us), cuBLAS split-K reduce 12.1% (3.83 us), PyTorch SiLU 6.9% (2.19 us). The remaining low-risk epilogue candidates were rejected within noise; further material improvement likely requires a larger tensor-op-first CUTLASS/CuTe rewrite that fuses the GEMM epilogue or replaces the PyTorch/cuBLAS multi-launch path. The legacy `gated_norm_cute_forward` name remains historical: the checked source is hand-written CUDA inline MMA/ldmatrix/cp.async, not CuTe-generated code.

