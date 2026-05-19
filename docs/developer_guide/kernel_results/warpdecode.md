# WarpDecode

## Scope

WarpDecode MoE decode on Blackwell for the production optimization-playground shape:
hidden size 7168, intermediate size 2048, top-k 8, BF16 packed `w13`, and the
CuTe WarpDecode path. WarpDecode has no direct Megatron flashtraining equivalent;
flashtraining comparisons are not used for this kernel.

## Current Result

Accepted close-out change: remove the terminal `__syncthreads()` and
`__threadfence()` from gate/up kernels when `WD_PDL_ENABLED` is not set. The PDL
build still keeps the sync, fence, and `cudaTriggerProgrammaticLaunchCompletion`
ordering.

Clean B200 final coverage, GPU0 idle-gated, artifact
`artifacts/final_round_noepilogue_20260519T002431Z`:

| Batch | Prior OP incumbent us | Accepted us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 121.695 | 118.657 | 1.0256x |
| 4 | 397.547 | 395.535 | 1.0051x |
| 8 | 774.164 | 771.802 | 1.0031x |
| 16 | 1543.030 | 1532.080 | 1.0071x |
| 32 | 3071.980 | 3056.810 | 1.0050x |
| 64 | 6136.150 | 6099.060 | 1.0061x |

Both production and accepted binaries passed correctness in the final coverage
sweep. The same final round's isolated autoinfer B1 metric was noisier and below
the close-out threshold: 113.668 us to 112.858 us, 1.0072x. Earlier clean
autoinfer for the same no-epilogue candidate measured 113.988 us to 112.678 us,
1.0116x, in `artifacts/ikp_autoinfer_noepilogue_20260518T234740Z`.

## Profile Evidence

IKP/Nsight was used first to choose the target. Gate/up was the dominant B1
kernel, so the accepted candidate removes only non-PDL gate/up epilogue work:

| Kernel | Prior OP incumbent mean us | Accepted mean us | Delta |
| --- | ---: | ---: | ---: |
| gate/up packed | 69.862 | 68.188 | -1.674 us |
| down | 38.347 | 38.507 | +0.160 us |

The improvement is a control-path cleanup, not a new tensor/layout path. It is
safe for the default non-PDL stream-ordered launch because the next kernel sees
prior writes through normal same-stream kernel ordering. PDL remains explicitly
ordered by the existing sync/fence/trigger path.

## Candidate Decisions

| Candidate family | Decision | Evidence |
| --- | --- | --- |
| non-PDL no-epilogue | accepted | Correct; final coverage B1 1.0256x and all B1-B64 batches positive; prior clean autoinfer 1.0116x. |
| PDL chain | rejected | Correct but neutral: 113.595 us to 113.651 us, 0.9995x. |
| no-threadfence / gateflush variants | rejected | Correct but under the old 3% gate and not better than no-epilogue: 1.0026x-1.0094x. |
| B1 no-tail/down specializations | rejected | Correct but below gate or profile-regressive; `b1_special_notail_auto` made down 38.623 us to 50.260 us. |
| gate N=8 / gate512 policy | rejected | Correct but full-path losses: 0.8772x and 0.9917x respectively. |
| large gate K-tile redesign | rejected | Correct smoke, but gate1792 0.8687x and gate3584 0.6312x; larger tiles increased gate/up time. |
| tensor/CuTe/CZS redesign | not promoted | Larger CuTe current-shape probes regressed. The grouped tensor-op idea remains proposal-only; `docs/proofs/warpdecode_b200_optin_czs_module.json` is valid JSON but status `unproved`. |

## Verification

- Final round stopped after the user-requested extra round; no continuous loop is
  left running for WarpDecode.
- GPU-idle-gated B200 coverage: `coverage_summary.json` in
  `artifacts/final_round_noepilogue_20260519T002431Z`.
- IKP/Nsight profile: `profile_summary.json` in the same artifact.
- Post-cleanup source compile/smoke: `compile_smoke.log` reports
  `mismatches=0 checked=7168` for the current header.
- CZS/proof check: `python3 -m json.tool docs/proofs/warpdecode_b200_optin_czs_module.json`
  passed; no CZS promotion applies to this epilogue-only change.

## Notes

Overlapped earlier parent attempts remain preserved as exploratory evidence only.
Acceptance numbers above come from clean per-GPU idle-gated artifacts.
