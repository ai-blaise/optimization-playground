# GatedNorm

## Scope

Forward-only BF16 GatedNorm inference for DeepSeek-V3.2-REAP. The path computes
`normed * sigmoid(silu(normed @ w_down.T) @ w_up.T)` for hidden size 7168 and
rank <= 64.

## Current Result

B200 acceptance-grade validation used idle-gated `CUDA_VISIBLE_DEVICES=0`
and `CUDA_VISIBLE_DEVICES=1` runs. The current selector keeps the accepted
480-token fused sigmoid*mul floor, tries the CZS-proved CuTe path before
torch.mm for rank <= 32 through 1024 tokens, rank40 through 1024 tokens, rank48
through 512 tokens, and rank64 only when tokens < 16. Rank64 tokens >= 16 and
large rank40/48 prefill stay on torch.mm/cuBLAS; the 480-511 token boundary now
still uses the SGLang torch.mm dispatch path but fuses the final sigmoid*mul
scalar launch. The final close-out round also retunes that fused epilogue to a
2048-element block only for rank64 tokens 480 through 528; all other ranks and
token counts keep the 1024-element default block.

The continuation started from commit `ad043e9a7f518a972d2c2cef2c1e17ac5545b8ad`.
That incumbent lost to the equivalent ai-blaise/Megatron-LM flashtraining
GatedNorm at ref `844bf42af7ce73a1b80e4b1ccb3c221dd63de35d` on
rank64/tokens1 and rank64/tokens8, so the matching rank64 tiny CuTe policy was
ported from flashtraining Megatron into optimization-playground first and
treated as the new OP baseline.
After the full loop, all 72 shapes in the final flashtraining matrix were
correct for both implementations and current optimization-playground was faster
than flashtraining on every measured shape. For OP-winning shapes, no
flashtraining->OP port was required.

| Bucket | Prior OP incumbent | Accepted OP policy | Speedup | Decision driver |
| --- | ---: | ---: | ---: | --- |
| rank64, tokens1 | 0.027979 ms | 0.016788 ms | 1.667x | flashtraining->OP CuTe tiny port |
| rank64, tokens8 | 0.027710 ms | 0.020848 ms | 1.329x | flashtraining->OP CuTe tiny port |
| rank40, tokens512 | 0.031822 ms | 0.026276 ms | 1.211x | Narrowed CuTe threshold |
| rank40, tokens1024 | 0.036925 ms | 0.035144 ms | 1.051x | Narrowed CuTe threshold |
| rank48, tokens512 | 0.030978 ms | 0.026948 ms | 1.150x | Narrowed CuTe threshold |
| rank64, tokens480 | 0.032789 ms | 0.030244 ms | 1.084x | Lower fused sigmoid*mul floor |
| rank64, tokens496 | 0.032786 ms | 0.030130 ms | 1.088x | Lower fused sigmoid*mul floor |
| rank64, tokens511 | 0.033710 ms | 0.030113 ms | 1.119x | Lower fused sigmoid*mul floor |

Final close-out epilogue retune, measured against the strongest accepted OP
policy before this retune:

| Bucket | Candidate | Manual median delta | Autoinfer delta | Decision |
| --- | --- | ---: | ---: | --- |
| rank64, tokens480 | block2048 | +2.0608% | +2.5366% | accept |
| rank64, tokens496 | block2048 | +5.4998% | +1.4723% | accept |
| rank64, tokens511 | block2048 | +1.0615% | +0.5505% | accept as positive inside the contiguous accepted window |
| rank64, tokens512 | block2048 | +2.7129% | +3.7374% | accept |
| rank64, tokens528 | block2048 | +2.2185% | +1.5570% | accept |

Guard rows stayed outside the production selector: rank64/tokens1024 was
-0.3046% manual and -0.0911% autoinfer, rank48/tokens1024 was -0.0014% manual
and +0.0046% autoinfer, and rank40/tokens2048 was -0.0950% manual.

Flashtraining Megatron comparator highlights at ref `844bf42af7ce73a1b80e4b1ccb3c221dd63de35d`:

| Bucket | Current | Flashtraining | Delta vs flashtraining |
| --- | ---: | ---: | ---: |
| rank40, tokens512 | 0.026869 ms | 0.045430 ms | +40.86% |
| rank40, tokens1024 | 0.035464 ms | 0.053513 ms | +33.73% |
| rank48, tokens512 | 0.027267 ms | 0.043226 ms | +36.92% |
| rank64, tokens1 | 0.017051 ms | 0.019883 ms | +14.24% |
| rank64, tokens8 | 0.021095 ms | 0.021218 ms | +0.58% |
| rank64, tokens480 | 0.041635 ms | 0.055837 ms | +25.43% |
| rank64, tokens496 | 0.031838 ms | 0.042007 ms | +24.21% |
| rank64, tokens511 | 0.029875 ms | 0.043357 ms | +31.09% |

The block2048 retune is an additional positive OP-only epilogue improvement on
rank64/tokens480-528 after the flashtraining comparison above. Because those
rows were already faster than the equivalent flashtraining GatedNorm and the
retune only applies to accepted positive OP rows, no flashtraining-to-OP port
was needed for the final close-out round.

The parent >=3% gate was applied against the strongest current incumbent or
candidate for the relevant policy. Rank64/tokens8 only has a small positive
edge over flashtraining because both implementations now use the same CuTe
strategy, but it clears the gate versus the losing optimization-playground
incumbent and no longer loses to the required flashtraining comparator. Later
OP-winning comparator rows required no flashtraining->OP port.

System-path applicability: this change is in `sglang.jit_kernel.gated_norm`,
the runtime entry point used by the optimization-playground SGLang path after
HF config/model wiring chooses the GatedNorm module. The environment override
`SGLANG_GATED_NORM_SIGMOID_MUL_FUSE_MIN_TOKENS` still preserves fallback and
A/B control. The accepted 480 floor affects torch.mm/cuBLAS fallback shapes in
the 480-511 token band, most relevant to near-512 chunk/prefill boundaries,
without changing CuTe eligibility, hidden-size assumptions, dtype guards, or
unsupported-shape fallbacks.

## Optimization History

- Seed candidates remained script-local in `scripts/playground`: CuTe-first,
  torch.mm-forced launch policies, fused sigmoid*mul floors, and no-decline CuTe
  probes. Production imports no candidate registry.
- CZS proved the existing CuTe module before CuTe dispatch expansion: 15 proved,
  0 disproved, 0 unknown.
- IKP/NSys for rank64/tokens8 showed old production at 400 kernels over 80
  calls: two nvjet/cuBLAS-backed GEMMs plus SiLU, sigmoid, and mul elementwise
  launches. The flashtraining->OP ported CuTe path used 240 kernels: `gated_norm_pass1_mma`,
  `gated_norm_pass2_mma_n_warps`, and `zero_workspace_kernel`. Autoinfer kept
  the change at 0.027821 ms to 0.020961 ms, +24.66%.
- IKP/NSys for rank40/tokens512 and rank48/tokens512 showed torch.mm production
  at 320 kernels over 80 calls, including two tensor-op GEMMs, SiLU, and the
  fused epilogue. Narrow CuTe used 240 kernels over 80 calls with
  `gated_norm_pass1_mma`, `gated_norm_pass2_mma`, and workspace zeroing.
- Autoinfer kept the narrowed CuTe thresholds: rank40/tokens512 +15.90%,
  rank40/tokens1024 +4.60%, and rank48/tokens512 +13.06%. It rejected broad
  rank48/tokens1024 CuTe at -7.11%.
- Lowering the fused sigmoid*mul floor to 256 tokens was rejected again:
  rank64/tokens256 regressed 8.67% by autoinfer and 11.53% in the sweep.
- Reopening the sigmoid*mul floor from profile data preserved useful rejections:
  a 384 floor was rejected because rank64/tokens384 only reached +2.89% in the
  clean profile, and a 448 floor was rejected because rank64/tokens448 only
  reached +2.86%. The 480 floor was accepted after idle-gated rank64/tokens480,
  496, and 511 profiles cleared +7.76%, +8.10%, and +10.67%.
- IKP/NSys for rank64/tokens480 showed the 480 floor keeps the same torch.mm
  tensor-op GEMMs and removes one scalar launch per call: 2500 kernels across
  500 captured old-production calls became 2000 kernels across 500 captured
  candidate calls.
- The final round accepted the narrow rank64 480-528 block2048 epilogue retune
  under the updated >1% close-out gate. Block4096 was rejected because it
  regressed rank64/tokens480 by 0.6657%, rank64/tokens512 by 1.1671%, and guard
  rows including rank48/tokens1024 by 0.1404%, despite isolated wins at
  rank64/tokens496, 511, and 528.
- Bypassing the rank64 CuTe decline guard was rejected: rank64 tokens16 through
  1024 regressed 40% to 81% versus production.
- Forcing torch.mm earlier for low/mid-rank decode and prefill was rejected:
  geomean speedup was 0.639x and max regression was 95.51%.
- Final broad CuTe guardrails were rejected above the promoted cutoffs:
  rank40/tokens2048 regressed 26.3%, rank48/tokens1024 regressed 5.1%, and
  rank48/tokens2048 regressed 42.9%.

## Tensor-Op Audit

Accepted hot-path changes use the CZS-proved CuTe/tensor-op path or remove
scalar launch overhead around tensor-op work. The new rank64 tiny, rank40, and
rank48 promotions all dispatch into the CuTe MMA kernels. The retained
480-token fused sigmoid*mul policy is scalar CUDA, but it remains eligible
because it clears the gate and fuses launch overhead after the torch.mm/cuBLAS
GEMMs on the production SGLang dispatch path. Credible CuTe alternatives for
those epilogue shapes were tested and rejected where they regressed or failed
the >=3% gate.

## Rejected Probes

Experimental modes are script-local in `scripts/playground`; they are not part
of the production runtime contract.

| Probe | Decision |
| --- | --- |
| Force torch.mm/cuBLAS | Rejected: low-token regressions |
| Rank-8 decode torch.mm | Rejected: decode regressions |
| Earlier low-rank prefill torch.mm | Rejected: decode regressions |
| Lower fused sigmoid*mul floor to 256 | Rejected: rank64/tokens256 regression |
| Lower fused sigmoid*mul floor to 384 | Rejected: rank64/tokens384 clean profile was +2.89%, below gate |
| Lower fused sigmoid*mul floor to 448 | Rejected: rank64/tokens448 clean profile was +2.86%, below gate |
| Broad CuTe/no-decline expansion | Rejected: rank48/rank64 prefill regressions |
| Rank64 CuTe for tokens >=16 | Rejected: no-decline probe regressed 40-81% |
| Fused epilogue block4096 sweep | Rejected: selected-window and guard regressions |

Use `python scripts/playground/bench_gated_norm.py --list-modes` for local
reproduction modes. Use
`python scripts/playground/verify_gated_norm_czs.py --candidate production` for
CuTe proof gating.

## Artifacts

- Flashtraining public matrix before OP-port audit: `/root/b200-run-20260518/metrics/gated_norm_flashtraining_compare_round8_public.jsonl`.
- Rank64 OP-port repeat/order: `/root/b200-run-20260518/metrics/gated_norm_cont_round8_rank64_tiny_repeated_order_preport.jsonl`.
- Rank64 OP-port IKP: `/root/b200-run-20260518/metrics/gated_norm_ikp_round8_rank64_t8_production_default` and `/root/b200-run-20260518/metrics/gated_norm_ikp_round8_rank64_t8_candidate_cute_first_supported`.
- Narrow CuTe repeat/order: `/root/b200-run-20260518/metrics/gated_norm_cont_round9_narrow_cute_repeated_order.jsonl`.
- Narrow CuTe IKP: `/root/b200-run-20260518/metrics/gated_norm_ikp_round9_r40_t512_*` and `/root/b200-run-20260518/metrics/gated_norm_ikp_round9_r48_t512_*`.
- Autoinfer round8/9/10 logs: `/root/b200-run-20260518/metrics/gated_norm_autoinfer_round8_rank64_t8_cute_first.log`, `/root/b200-run-20260518/metrics/gated_norm_autoinfer_round9_cute_thresholds.log`, and `/root/b200-run-20260518/metrics/gated_norm_autoinfer_round10_sigmoid_floor_recheck.log`.
- Final flashtraining matrix: `/root/b200-run-20260518/metrics/gated_norm_flashtraining_compare_final_full.jsonl`.
- Sigmoid floor continuation ledger: `/root/b200-run-20260518/metrics/gated_norm_continuation_ledger.jsonl`.
- 480-floor acceptance profiles: `/root/b200-run-20260518/metrics/gated_norm_accept_round20_sigmoid480.jsonl` and `/root/b200-run-20260518/metrics/gated_norm_accept_round21_sigmoid480_gpu0.jsonl`.
- 480-floor IKP/NSys: `/root/b200-run-20260518/metrics/gated_norm_ikp_round21_sigmoid480_r64_t480`.
- 480-floor changed-scope flashtraining comparator, no flashtraining->OP port required: `/root/b200-run-20260518/metrics/gated_norm_flashtraining_round22_sigmoid480.jsonl`.
- Post-acceptance system-path continuation loop: `/root/b200-run-20260518/metrics/gated_norm_round23_system_guard_loop.log` and `/root/b200-run-20260518/metrics/gated_norm_round23_system_guard_loop.jsonl`.
- Round24b serving-shaped dispatch loop: `/root/b200-run-20260518/metrics/gated_norm_round24b_triton_dispatch_system_loop.log` and `/root/b200-run-20260518/metrics/gated_norm_round24b_triton_dispatch_system_loop.jsonl`.
- Runtime wiring audit: `/root/b200-run-20260518/metrics/gated_norm_runtime_wiring_round23_sigmoid480.log`.
- Final block retune: `/root/b200-run-20260518/metrics/gated_norm_round28_rank64_block_confirm.jsonl` and `/root/b200-run-20260518/metrics/gated_norm_autoinfer_round28_rank64_block_confirm_compare.jsonl`.
- CUPTI PC sampling remains blocked by the prior IKP collector double-free;
  NSys import artifacts are the active IKP evidence.

## Verification

- `MAX_MEMORY_MIB=1024 MAX_UTILIZATION=5 STABLE_SECONDS=20 POLL_SECONDS=5 TIMEOUT_SECONDS=1200 /usr/local/bin/prime-b200-wait-gpu-idle 0 -- env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python -m pytest python/sglang/jit_kernel/tests/test_gated_norm.py -q`: 24 passed; artifact `/root/b200-run-20260518/metrics/gated_norm_verify_pytest_round23_sigmoid480.log`.
- `PYTHONPATH=python .venv/bin/python -m py_compile python/sglang/jit_kernel/gated_norm.py scripts/playground/bench_gated_norm.py scripts/playground/profile_gated_norm_case.py scripts/playground/verify_gated_norm_czs.py scripts/playground/compare_gated_norm_flashtraining.py`: passed; artifact `/root/b200-run-20260518/metrics/gated_norm_verify_pycompile_round23_sigmoid480.log`.
- `CZS_BIN=/root/b200-run-20260518/workers/higgs/tools/czs PYTHONPATH=python .venv/bin/python scripts/playground/verify_gated_norm_czs.py --candidate production`: 15 proved, 0 disproved, 0 unknown; artifact `/root/b200-run-20260518/metrics/gated_norm_verify_czs_round23_sigmoid480.log`.
- `git diff --check`: passed; artifact `/root/b200-run-20260518/metrics/gated_norm_verify_diffcheck_round23_sigmoid480.log`.
- Runtime wiring audit confirmed `DeepseekV2DecoderLayer` imports `gated_norm_forward`, enables it only when HF config has `gated_norm`/weights, falls back to the reference torch ops for non-BF16/CUDA/contiguous unsupported cases, and applies it on both post-RMSNorm SGLang sites; artifact `/root/b200-run-20260518/metrics/gated_norm_runtime_wiring_round23_sigmoid480.log`.
- Production cleanup sweep: `python/sglang/jit_kernel/gated_norm.py` has no
  runtime candidate selector dependency; experiment modes live only in
  `scripts/playground`.
- Parent close-out accepted the rank64/tokens480-528 block2048 retune after the
  worker stopped on `round_stop_after_cycle`; all broader block-size variants
  remain rejected.
