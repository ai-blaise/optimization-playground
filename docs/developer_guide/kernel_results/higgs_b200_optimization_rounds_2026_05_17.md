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

New incumbent commit: `9dae7c63d`.

## Restart Round 6: Fractional Split Policy

Incumbent: round-5 commit `9dae7c63d`, auto policy using 32/64/128 splits.

Hotspot/profile evidence: the same stage1-vs-stage2 tradeoff remained after
round 5. Stage1 still benefits from more parallel splits on long topk, but
stage2 merge cost and launch/CTA overhead make powers of two too coarse for
several shapes. Because the kernel accepts arbitrary positive split counts,
the next empirical candidate tested fractional counts `48` and `96` alongside
`32/64/128`; all CUDA runs used `/root/agent-runs/gpu_locked_any.sh` only
around the benchmark process.

Candidate: refine the auto policy to include 48 and 96 splits:

* topk >= 4096: 128 for `rows*heads <= 8`, 96 for `<= 16`, 48 for
  `32/64/256`, 64 for `128`, otherwise 32.
* topk >= 2048: 96 for `rows*heads <= 8`, 64 for `<= 16`, 48 for
  `32/64/128`, otherwise 32.
* topk >= 1024: 64 for `rows*heads <= 16`, 48 for `rows*heads == 64`,
  otherwise 32.

Direct HIGGS JIT evidence for the refined split choices:

| Shape | Prior auto | Fractional best | Best split | Decision |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk2048 | 0.043112 ms | 0.042360 ms | 96 | keep 96 |
| r2 h8 topk4096 | 0.069743 ms | 0.069273 ms | 96 | keep 96 |
| r4 h8 topk2048 | 0.063563 ms | 0.061496 ms | 48 | keep 48 |
| r4 h8 topk4096 | 0.109041 ms | 0.100508 ms | 48 | keep 48 |
| r8 h8 topk1024 | 0.061551 ms | 0.058968 ms | 48 | keep 48 |
| r8 h8 topk2048 | 0.098360 ms | 0.094531 ms | 48 | keep 48 |
| r8 h8 topk4096 | 0.171236 ms | 0.167478 ms | 48 | keep 48 |
| r16 h8 topk2048 | 0.162186 ms | 0.160794 ms | 48 | keep 48 |
| r32 h8 topk4096 | 0.555462 ms | 0.550627 ms | 48 | keep 48 |

Guardrails:

| Shape | 32 | 48 | 64 | 96 | 128 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| r4 h8 topk512 | 0.028889 | 0.030845 | 0.034910 | 0.038954 | 0.045956 | stay 32 |
| r4 h8 topk1024 | 0.039705 | 0.041084 | 0.046812 | 0.049148 | 0.054693 | stay 32 |
| r16 h8 topk4096 | 0.313044 | 0.297348 | 0.297113 | 0.299223 | 0.305123 | stay 64 |

Post-change pool-level validation with `splits=0,32,48,64,96,128`:

| Shape | Prior round-5 auto | New auto | Fixed-32 | Correctness |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk2048 | 0.043172 ms | 0.043496 ms | 0.057510 ms | max_abs 0, min_cos 0.99999988 |
| r2 h8 topk2048 | 0.059334 ms | 0.046160 ms | 0.058945 ms | max_abs 0, min_cos 0.99999982 |
| r2 h8 topk4096 | 0.100889 ms | 0.069737 ms | 0.100596 ms | max_abs 0, min_cos 0.99999988 |
| r4 h8 topk1024 | 0.040389 ms | 0.040925 ms | 0.040612 ms | exact vs auto ref |
| r4 h8 topk2048 | 0.064746 ms | 0.061598 ms | 0.063921 ms | exact vs auto ref |
| r4 h8 topk4096 | 0.110221 ms | 0.100655 ms | 0.109987 ms | exact vs auto ref |
| r8 h8 topk1024 | 0.061622 ms | 0.058716 ms | 0.064622 ms | exact vs auto ref |
| r8 h8 topk2048 | 0.098472 ms | 0.094605 ms | 0.109808 ms | exact vs auto ref |
| r8 h8 topk4096 | 0.171134 ms | 0.167687 ms | 0.200050 ms | exact vs auto ref |
| r16 h8 topk2048 | 0.162415 ms | 0.160780 ms | 0.166925 ms | exact vs auto ref |
| r16 h8 topk4096 | 0.297146 ms | 0.297382 ms | 0.312969 ms | max_abs 0, min_cos 0.99999982 |
| r32 h8 topk4096 | 0.555454 ms | 0.550546 ms | 0.562262 ms | exact vs auto ref |

Rejected sub-candidates: blanket 48 splits is rejected because it regresses
rows=4 topk512/1024; 96 splits are rejected outside the low-CTA long-topk
cases because stage2 overhead dominates; 128 remains useful only for the
single-row topk4096 corner.

Decision: keep. The candidate improves several shapes left on the table by
round 5, preserves fixed positive override deployability, and leaves the
short/topk guardrails on the fixed-32 incumbent.

New incumbent commit: `6eddf70cc`.

## Restart Round 7: Split-Count Neighborhood

Incumbent: round-6 commit `6eddf70cc`, fractional auto policy using
32/48/64/96/128 splits.

Hotspot/profile evidence: after the round-6 fractional policy, remaining
headroom was constrained to split-count granularity. Stage1 parallelism and
stage2 merge overhead were close enough that direct probes around each chosen
split count were still plausible. The candidate tested `40/56/72` around
48/64 and `80/112/160` around 96/128; parsing and source/doc work were done
outside GPU locks, with `gpu_locked_any.sh` used only for CUDA benchmark
processes.

Direct HIGGS JIT neighborhood evidence:

| Shape | Round-6 choice | Better neighbor | Decision |
| --- | ---: | ---: | --- |
| r1 h8 topk2048 | 96: 0.042376 ms | 80: 0.042146 ms | reject, pool did not confirm |
| r2 h8 topk4096 | 96: 0.069542 ms | 80: 0.068143 ms | keep 80 |
| r4 h8 topk2048 | 48: 0.061509 ms | 56: 0.060752 ms | keep 56 |
| r4 h8 topk4096 | 48: 0.100616 ms | 56: 0.097726 ms | keep 56 |
| r8 h8 topk4096 | 48: 0.167671 ms | 72: 0.166087 ms | keep 72 |
| r16 h8 topk2048 | 48: 0.160723 ms | 40: 0.160091 ms | keep 40 |
| r32 h8 topk4096 | 48: 0.550509 ms | 40: 0.550186 ms | keep 40 |

Pool-level validation for the candidate deltas:

| Shape | Round-6 auto | Candidate split | Candidate ms | Correctness |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk2048 | 0.042193 ms | 80 | 0.042568 ms | reject; keep 96 |
| r2 h8 topk4096 | 0.069157 ms | 80 | 0.067762 ms | max_abs 3.81e-06, min_cos 0.99999988 |
| r4 h8 topk2048 | 0.061558 ms | 56 | 0.061048 ms | max_abs 1.53e-05, min_cos 0.99999982 |
| r4 h8 topk4096 | 0.100529 ms | 56 | 0.097878 ms | max_abs 3.05e-05, min_cos 0.99999988 |
| r8 h8 topk4096 | 0.167815 ms | 72 | 0.166221 ms | max_abs 6.10e-05, min_cos 0.99999982 |
| r16 h8 topk2048 | 0.160952 ms | 40 | 0.160153 ms | max_abs 1.22e-04, min_cos 0.99999982 |
| r32 h8 topk4096 | 0.551107 ms | 40 | 0.550574 ms | max_abs 4.88e-04, min_cos 0.99999982 |

Rejected sub-candidates: 80 splits for r1/topk2048 was rejected because the
pool path stayed marginally better at the incumbent 96-split choice. 112 and
160 splits regressed low-CTA topk4096. 40/56/72 are only applied where both
direct and pool measurements indicated a win.

Decision: keep the confirmed neighborhood refinements. Gains are smaller than
rounds 5-6 but still measurable on shapes that remain relevant to long-topk
decode, and the policy still preserves fixed positive split overrides.

New incumbent commit: `c9967a43a`.

## Restart Round 8: Saturation Neighborhood

Incumbent: round-7 commit `c9967a43a`.

Hotspot/profile evidence: after round 7, the only remaining plausible knob in
the current scalar split-K implementation was one-step split-count adjustment
around the selected values. The probe tested `80/88/96` for r1 topk2048,
`120/128/136` for r1 topk4096, `72/80/88` for r2 topk4096,
`52/56/60` for rows=4 long-topk, `68/72/76` for rows=8 topk4096, and
`36/40/44` for rows=16 topk2048 and rows=32 topk4096.

Direct HIGGS JIT saturation evidence:

| Shape | Incumbent split | Best tested neighbor | Decision |
| --- | ---: | ---: | --- |
| r1 h8 topk2048 | 96: 0.041294 ms | 80: 0.041863 ms | reject |
| r1 h8 topk4096 | 128: 0.056459 ms | 120: 0.056062 ms | pool rejected |
| r2 h8 topk4096 | 80: 0.067680 ms | 72: 0.067825 ms | reject |
| r4 h8 topk2048 | 56: 0.061293 ms | 52: 0.062072 ms | reject |
| r4 h8 topk4096 | 56: 0.097865 ms | 52: 0.101092 ms | reject |
| r8 h8 topk4096 | 72: 0.166117 ms | 76: 0.166420 ms | reject |
| r16 h8 topk2048 | 40: 0.160097 ms | 36: 0.159834 ms | pool repeat accepted |
| r32 h8 topk4096 | 40: 0.550106 ms | 44: 0.552913 ms | reject |

Pool-level follow-up:

| Shape | Incumbent auto | Candidate | Candidate ms | Decision |
| --- | ---: | ---: | ---: | --- |
| r1 h8 topk4096 | 128 | 120 | 0.057564 ms vs auto 0.056009 ms | reject |
| r16 h8 topk2048 repeat 1 | 40 | 36 | 0.159847 ms vs auto 0.159989 ms | keep |
| r16 h8 topk2048 repeat 2 | 40 | 36 | 0.159832 ms vs auto 0.160076 ms | keep |
| r16 h8 topk2048 repeat 3 | 40 | 36 | 0.159829 ms vs auto 0.159990 ms | keep |

Decision: accept only the repeat-confirmed r16/topk2048 36-split adjustment.
All other neighbors either regressed directly, failed pool validation, or were
within measurement noise without a consistent advantage.

Stop rationale for the split-policy lane: rounds 5-8 exhausted the useful
split-count search from coarse powers of two through fractional counts and
immediate local neighborhoods. Remaining improvements from split retuning are
at or below noise, and the accepted policy now sits at local minima for the
measured B200 decode shapes. Further material gains require a different kernel
design rather than more split-count tuning.

New incumbent commit: `767a15ae3`.

## Final Profile And Stop Point

Current incumbent: `767a15ae3`.

Final nsys/IKP import:

* Profile command wrote `/root/agent-runs/higgs-final-current-r4k2048.nsys-rep`
  and CUDA-event JSON `/root/agent-runs/higgs-final-current-r4k2048.json`.
* IKP nsys import wrote `/root/agent-runs/higgs-ikp-final-current-r4k2048`.
* Shape: `num_slots=8192`, `rows=4`, `heads=8`, `topk=2048`,
  auto split policy (`0`, selecting 56 splits).
* CUDA-event decode time: `0.062161600589752196 ms`, finite output, min cosine
  `0.9999998211860657`.

Final IKP/nsys kernel attribution:

| Kernel | Count | Avg us | Total us |
| --- | ---: | ---: | ---: |
| `higgs_dense_2bit_mla_decode_stage1_split_kernel` | 27 | 41.675 | 1125.234 |
| `higgs_dense_2bit_mla_decode_stage2_kernel` | 27 | 11.771 | 317.820 |
| `higgs_dense_2bit_store_kernel` | 1 | 66.911 | 66.911 |
| `higgs_dense_2bit_mla_rotate_query_kernel` | 27 | 1.921 | 51.872 |

The final profile agrees with the earlier IKP hotspot story: stage1 is still
the dominant cost, but the accepted split policy already balances stage1
parallelism against stage2 merge overhead for the measured shapes. The
remaining stage1 work is the scalar HIGGS codec loop over packed EDEN2-16
indices, scale, rope, and online-softmax accumulation. The packed 258 B slot
layout is not tensor-core tile shaped; further material speedup would require
a new layout or staging design for a CuTe/CZS tensor-op path, not additional
split-count retuning.

Final stop rationale: stop after rounds 5-8 because the plausible scalar
split-K search space is saturated by empirical CUDA-event and IKP/nsys
evidence. Coarse split defaults, fractional split counts, and immediate
neighborhoods have either been accepted or rejected, and the only surviving
round-8 gain was a repeat-confirmed ~0.1% local tweak. More local split probes
are unlikely to survive noise; the next meaningful candidate would be a larger
layout/tensor-op rewrite with CZS proof artifacts, which is outside a safe
incremental patch for this packed compatibility path.
