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

## Round 2: Pair-Lane Packed-Byte Broadcast

Incumbent: round-1 commit `24b868a7a`/`eb70a3553` lineage, default `32`
splits. The final round-1 commit is the parent of this round-2 commit.

Hotspot/profile evidence: the same IKP stage1 attribution from round 0 still
applies after the round-1 split retune: the dominant kernel is the stage1
topk loop. Within that loop, both coordinate lanes for a 2-D HIGGS pair loaded
the same packed byte from each of the four 32-byte latent groups.

Candidate: only the even coordinate lane loads each packed byte, then broadcasts
it to its paired odd lane using warp shuffle. This follows the Blackwell/CuTe
guidance to reduce redundant scalar memory traffic in codec glue while leaving
the larger tensor-op/CuTe rewrite for a layout-compatible path.

Before/after measurement against incumbent, using the direct HIGGS JIT harness
because the shared venv's `sgl_kernel` package currently lacks SM100
`common_ops` and cannot import the pool stack:

| Shape | Incumbent avg | Candidate avg | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| r4 h8 topk1024 splits32 | 0.040624 ms | 0.040000 ms | 1.016x | Keep |

Correctness verification: seeded direct-HIGGS output checksum matched the
