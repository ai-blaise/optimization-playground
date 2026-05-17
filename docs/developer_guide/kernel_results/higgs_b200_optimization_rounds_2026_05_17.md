# HIGGS B200 Optimization Rounds, 2026-05-17

Branch: `codex/higgs-tensor-op-loop-20260517`

Base merge: `04d2704ed Merge remote-tracking branch 'origin/main' into codex/higgs-tensor-op-loop-20260517`

Hardware: 1x NVIDIA B200 on `root@31.22.104.123`, CUDA 12.8 driver stack.

IKP source: `/root/work/rule7-refs/intra-kernel-profiler`.

GPU commands were run through `/root/agent-runs/gpu_locked.sh` with
`CUDA_VISIBLE_DEVICES=0` and `/root/work/optimization-playground/.venv`.

## Round 0: Incumbent

Incumbent commit: `04d2704ed`

Hotspot/profile evidence:

* IKP import of Nsight capture `/root/agent-runs/higgs-nsys-baseline-16.nsys-rep`
  into `/root/agent-runs/higgs-ikp-baseline-16` matched 168 HIGGS kernels.
* Shape: `num_slots=4096`, `rows=4`, `heads=8`, `topk=1024`, default
  `higgs_mla_decode_num_splits=16`.
* IKP kernel summary: stage1 split kernel `45.591 us` mean and 85.8% of
  HIGGS GPU time; stage2 `4.805 us`; rotate-query `1.910 us`.

Baseline measurement:

| Shape | Splits | Decode ms | Correctness |
| --- | ---: | ---: | --- |
| r4 h8 topk1024 | 16 | 0.057862 | incumbent |

Decision: stage1 topk work dominates, so the next candidate should increase
split-K parallelism before attempting a larger CuTe/tensor-op rewrite. This
follows the Rule 7 Blackwell guidance to use profiling before choosing a
concrete kernel candidate; the existing HIGGS packed layout is still scalar
codec glue and not yet a CuTe fast path.

## Round 1: B200 Split-K Default

Incumbent: `04d2704ed` default `16` splits.

Candidate: raise the HIGGS fused dense MLA decode default to `32` splits in
`ServerArgs`, `HiggsDense2BitNSATokenToKVPool`, and the HiSparse wrapper. A
pre-edit split sweep over rows `1,4,16` and topk `512,1024,2048` showed `32`
as the robust B200 choice; `64` only won in some corners and regressed common
rows=4 shapes.

Before/after measurement against incumbent:

| Shape | Incumbent | Candidate | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| r4 h8 topk1024 | 0.057862 ms | 0.041285 ms | 1.40x | Keep |

Correctness verification:

* `test/srt/test_higgs_dense_2bit_kv_integration.py::test_higgs_split_k_matches_single_pass`: passed.
* `test/srt/test_quantization_config_dispatch.py`: 33 passed.

New incumbent commit: this round-1 commit.
