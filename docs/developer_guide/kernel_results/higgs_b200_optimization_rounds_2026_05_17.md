# HIGGS B200 Optimization Rounds, 2026-05-17

Branch: `codex/higgs-tensor-op-loop-20260517`

Base merge: `04d2704ed Merge remote-tracking branch 'origin/main' into codex/higgs-tensor-op-loop-20260517`

Hardware: 1x NVIDIA B200 on `root@31.22.104.123`, CUDA 12.8 driver stack.

IKP source: `/root/work/rule7-refs/intra-kernel-profiler`.

Initial GPU commands were run through `/root/agent-runs/gpu_locked.sh` with
explicit `CUDA_VISIBLE_DEVICES`; reopened runs used the per-GPU locks in
`/root/agent-runs/gpu_locked.sh` and `/root/agent-runs/gpu_locked_any.sh` so
both B200s could stay busy without sharing one measurement lock.

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

New incumbent commit: `24b868a7a`.

## Round 2: Pair-Lane Packed-Byte Broadcast

Incumbent: round-1 commit `24b868a7a`, default `32` splits.

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
incumbent (`-21.94557762145996`) and the candidate output was finite. The first
draft used a conditional warp collective and was rejected immediately after an
illegal memory access; the committed candidate makes all lanes participate in
the shuffle.

New incumbent commit: `21805592c`.

## Round 3: Read-Only Packed-Byte Loads

Incumbent: round-2 commit `21805592c`.

Hotspot/profile evidence: stage1 remains the only material target after round
2. The paired-lane broadcast reduced duplicated byte loads, leaving the even
lane packed-slot reads in the same stage1 loop.

Candidate: use explicit `__ldg` for the four packed-byte reads performed by
the loading lane. This is still scalar codec glue, but it is a minimal
Blackwell-safe candidate before considering larger CuTe/CZS layout work.

Before/after measurement against incumbent, direct HIGGS JIT harness:

| Shape | Incumbent avg | Candidate avg | Speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| r4 h8 topk1024 splits32 | 0.040000 ms | 0.039861 ms | 1.0035x | Keep |

Correctness verification: seeded checksum matched the incumbent
(`-21.94557762145996`) and output was finite.

New incumbent commit: `f2b9395d6`.

## Round 4: Split-64 Default Candidate

Incumbent: round-3 commit `f2b9395d6`.

Hotspot/profile evidence: IKP still points at stage1 topk work as the expensive
region, and split count directly controls stage1 parallelism. The round-1
pre-sweep hinted that 64 splits can help batch-1 long-topk cases, so this
candidate was re-tested after the round-2 and round-3 stage1 micro-optimizations.

Candidate: make 64 splits the default. No source change was kept while testing;
the direct HIGGS JIT harness passed `--splits 64` against the current source.

Before/after measurement against incumbent:

| Shape | 32-split incumbent | 64-split candidate | Decision |
| --- | ---: | ---: | --- |
| r1 h8 topk512 | 0.026746 ms | 0.026777 ms | Reject |
| r1 h8 topk1024 | 0.036982 ms | 0.032908 ms | Improves |
| r1 h8 topk2048 | 0.057457 ms | 0.043188 ms | Improves |
| r4 h8 topk512 | 0.028902 ms | 0.034955 ms | Reject |
| r4 h8 topk1024 | 0.039838 ms | 0.047196 ms | Reject |
| r4 h8 topk2048 | 0.063759 ms | 0.069809 ms | Reject |
| r16 h8 topk512 | 0.057416 ms | 0.061615 ms | Reject |
| r16 h8 topk1024 | 0.094287 ms | 0.094782 ms | Reject |
| r16 h8 topk2048 | 0.166992 ms | 0.162284 ms | Improves |

Correctness verification: all split-64 runs produced finite output. Checksums
differ slightly from split 32 because split-K merge order changes BF16 rounding,
matching the existing split-vs-single tolerance model.

Decision: reject 64 as a global default because it regresses the rows 4 and
rows 16 decode-like shapes that motivated the B200 default. A future adaptive
runtime split policy could target the batch-1 long-topk cases, but the current
pool-level import path is blocked by the shared venv missing SM100
`sgl_kernel.common_ops`, so this branch leaves no adaptive source change.

New incumbent commit: unchanged, `f2b9395d6`.

## Restart Round 5: Auto Split Policy

Incumbent: round-4 branch head `01f37e9ad`, code incumbent `f2b9395d6`
with a fixed default of 32 splits.

Hotspot/profile evidence: prior IKP/nsys still attributes the dominant
work to the stage1 split kernel (`45.591 us` mean and 85.8% of HIGGS GPU
time for the r4 h8 topk1024 default-16 baseline). Split count directly
changes stage1 parallelism, while stage2 merge overhead grows with the
number of splits. The round-4 rejection showed 64 was not a safe global
default, so the reopened candidate first expanded the split probe to
cover 32/64/128 splits, rows `1/4/8/16/32`, and topk `512/1024/2048/4096`.

Candidate: make `--higgs-mla-decode-num-splits=0` the default auto mode
and keep positive values as fixed overrides. The auto policy chooses:

* 128 splits for `rows*heads <= 8` and `topk >= 4096`.
* 64 splits for `rows*heads <= 8` and `topk >= 1024`.
* 64 splits for `rows*heads == 64` and `topk >= 1024`.
* 64 splits for `64 <= rows*heads <= 128` and `topk >= 2048`.
* 64 splits for `64 <= rows*heads <= 256` and `topk >= 4096`.
* 32 splits otherwise.

Direct HIGGS JIT split-probe evidence against the fixed-32 incumbent:

| Shape | 32 split | 64 split | 128 split | Auto decision |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk512 | 0.026742 ms | 0.026725 ms | 0.034943 ms | 32 |
| r1 h8 topk1024 | 0.036938 ms | 0.032844 ms | 0.038996 ms | 64 |
| r1 h8 topk2048 | 0.057444 ms | 0.043131 ms | 0.045149 ms | 64 |
| r1 h8 topk4096 | 0.098394 ms | 0.064702 ms | 0.056004 ms | 128 |
| r4 h8 topk1024 | 0.039806 ms | 0.046937 ms | 0.055129 ms | 32 |
| r4 h8 topk2048 | 0.063600 ms | 0.069696 ms | 0.073488 ms | 32 |
| r4 h8 topk4096 | 0.109355 ms | 0.114911 ms | 0.110337 ms | 32 |
| r8 h8 topk1024 | 0.064491 ms | 0.061534 ms | 0.073738 ms | 64 |
| r8 h8 topk2048 | 0.109683 ms | 0.098652 ms | 0.106674 ms | 64 |
| r8 h8 topk4096 | 0.200518 ms | 0.171064 ms | 0.174230 ms | 64 |
| r16 h8 topk1024 | 0.094365 ms | 0.094485 ms | 0.108833 ms | 32 |
| r16 h8 topk2048 | 0.167424 ms | 0.161996 ms | 0.174552 ms | 64 |
| r16 h8 topk4096 | 0.313301 ms | 0.296960 ms | 0.305236 ms | 64 |
| r32 h8 topk2048 | 0.292929 ms | 0.294818 ms | 0.315097 ms | 32 |
| r32 h8 topk4096 | 0.562323 ms | 0.555752 ms | 0.571687 ms | 64 |

Pool-level before/after measurement using `splits=0,32`:

| Shape | Fixed-32 incumbent | Auto candidate | Speedup | Correctness |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk1024 | 0.037023 ms | 0.032945 ms | 1.124x | max_abs 2.98e-08, min_cos 0.99999988 |
| r1 h8 topk2048 | 0.057498 ms | 0.043172 ms | 1.332x | max_abs 1.53e-05, min_cos 0.99999988 |
| r1 h8 topk4096 | 0.098443 ms | 0.057543 ms | 1.711x | max_abs 6.10e-05, min_cos 0.99999994 |
| r4 h8 topk1024 | 0.040662 ms | 0.040420 ms | 1.006x | exact vs auto ref |
| r4 h8 topk2048 | 0.064948 ms | 0.064624 ms | 1.005x | exact vs auto ref |
| r4 h8 topk4096 | 0.110449 ms | 0.109948 ms | 1.005x | exact vs auto ref |
| r8 h8 topk1024 | 0.064970 ms | 0.061618 ms | 1.054x | max_abs 2.44e-04, min_cos 0.99999988 |
| r8 h8 topk2048 | 0.109852 ms | 0.098540 ms | 1.115x | max_abs 2.44e-04, min_cos 0.99999982 |
| r8 h8 topk4096 | 0.200157 ms | 0.171649 ms | 1.166x | max_abs 2.44e-04, min_cos 0.99999982 |
| r16 h8 topk2048 | 0.167461 ms | 0.162080 ms | 1.033x | max_abs 2.44e-04, min_cos 0.99999982 |
| r16 h8 topk4096 | 0.313612 ms | 0.297313 ms | 1.055x | max_abs 2.44e-04, min_cos 0.99999982 |
| r32 h8 topk4096 | 0.562692 ms | 0.555700 ms | 1.013x | max_abs 2.44e-04, min_cos 0.99999982 |

Rejected sub-candidates: a global split-64 default remains rejected because
rows=4 regresses; 128 splits are rejected except for the single-row topk4096
corner because they regress topk512/1024 and rows>=8 shapes.

Decision: keep. The candidate improves the long-topk and rows=8/16 cases
without sacrificing the rows=4 fixed-32 guardrail, and manual fixed split
counts remain deployable by passing a positive value.
