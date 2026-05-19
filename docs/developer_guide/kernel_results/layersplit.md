# LayerSplit

## Scope

LayerSplit staging and fused materialize memory-copy kernels on B200. This lane
is copy-path work for HIGGS dense KV plus NVFP4 IndexCache/HISA, not a
tensor-core compute kernel.

There is no direct LayerSplit equivalent in the Megatron flashtraining branch.
The flashtraining comparison column is therefore not applicable for this lane.

## Current Result

| Case | Optimization-playground Baseline | Current | Flashtraining Equivalent |
| --- | --- | --- | --- |
| HIGGS 258-byte rows | `stage_for_broadcast` rejected `row_bytes=258` | Correct for 258-byte rows | None |
| 258-byte row stage copy | Launch failure | 1.649x median vs `torch.copy_`, 1.426x min, 1.701x max | None |
| 264-byte padded sidecar | Existing aligned path | 1.637x median vs `torch.copy_`, 1.421x min, 1.712x max | None |
| Final trimmed gate | Launch failure at 258-byte rows | 1.516x median vs `torch.copy_` across the accepted cells | None |
| Owner staging path | Direct `Tensor.copy_` in `prefetch_layersplit_kv_buffer(..., use_staging=True)` | Unchanged; packaged-op helper rejected | None |

The accepted implementation preserves the existing `uint4` vector prefix copy
and adds byte-tail handling for unaligned contiguous stage copies and fused
materialize rows. The non-contiguous stage fallback now delegates to PyTorch
`copy_` on the active row prefix so it is stride-correct instead of silently
dropping byte tails.

## Optimization Rounds

| Round | Candidate | Decision | Evidence |
| --- | --- | --- | --- |
| 1 | Tail-safe stage and materialize copy | Accepted | Fixed the 258-byte HIGGS deployability failure; correctness passed with inactive-row sentinels for 258- and 264-byte rows. |
| 2 | Small-copy CTA-count variants | Rejected | 32/64 block variants helped isolated tiny prefixes but did not beat the 148-block path consistently. |
| 3 | Larger stage-copy threshold/window | Rejected | 768 KiB and 1 MiB windows helped some 2048-row cells but regressed or tied other cells; no stable gain over the 512 KiB default. |
| 4 | Pool-threshold sweep | Rejected as promotion evidence | The pool benchmark path did not call `torch.ops.layersplit_cute.stage_for_broadcast`, so those numbers are not evidence for this kernel. |
| 5 | Owner-staging helper through packaged op | Rejected | The actual Python helper path was correct but slower than direct `Tensor.copy_`; the final round measured helper speedups of only 0.531x-0.574x. Direct precomputed op calls had isolated wins up to 1.076x but lost on other cells, so the production staging path stays unchanged. |

## Verification

- `test/srt/test_nsa_layersplit.py`: passed in the worker run.
- `bench_layersplit_stage.py --row-bytes 258,264 --padding-rows 3`: passed
  active-prefix equality and inactive-row sentinel checks.
- Parent close-out rebuilt the local extension from source and re-ran rows
  `1,4,8,128,512,2048`; 258-byte rows were 1.398x-2.080x faster than
  `torch.copy_`, and 264-byte rows were 1.390x-1.693x faster.
- After removing the dead non-contiguous flat-kernel fallback, a final
  from-source smoke on rows `1,128,2048` kept 258-byte rows 1.203x-1.662x
  faster and 264-byte rows 1.444x-1.891x faster than `torch.copy_`.
- Parent close-out also exercised non-contiguous rows with 258, 259, and 264
  byte widths and verified active rows, inactive rows, and stride gaps.
- The owner-staging helper dispatch/fallback smoke verified correctness, but
  the helper was rejected for performance and removed from the production path.
- A 2-rank distributed pool smoke passed on the B200 VM with 4 interleaved
  LayerSplit layers, `dense_bytes_sum=299520`, and
  `tq_bytes_sum=387840`.
- IKP/NSys import matched 72 LayerSplit kernels under
  `/root/b200-run-20260518/workers/layersplit_kernel_loop/profiles/round1_tail_nsys`;
  active small-copy kernels averaged about 1.6 us on the profiled cells.
- CUPTI sassmetrics preload ran but did not emit JSON on this target, so it was
  not used for promotion.

## Notes

No new CuTe kernel was introduced in this pass, so no CZS proof module is
attached to the accepted candidate. A future CuTe/TMA rewrite still needs CZS
proof and a measured win over this vector-prefix plus byte-tail implementation.
