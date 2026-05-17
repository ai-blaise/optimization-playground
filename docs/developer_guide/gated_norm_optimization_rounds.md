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

New incumbent commit: pending in this branch as "Tune GatedNorm GEMM thresholds".

## CZS proof surface

Artifact: `docs/proofs/gated_norm_mma_czs_module.json`.

Command:

```bash
python scripts/playground/verify_gated_norm_czs.py
```

Result: 15 CZS obligations proved (9 layout legality, 2 vectorization, 4 ldmatrix legality). Caveat: CZS 0.4.1 does not model the inline `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` atom used by the hand-written CUDA path, so this proves the available shared-memory/vectorization/ldmatrix surface, not the legacy inline MMA atom itself.
