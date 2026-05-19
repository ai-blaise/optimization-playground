# FlashSampling

## Scope

TP-sharded decode sampling on Blackwell B200. The production-relevant shape is
`V=16160`, `D=7168`, BF16 weights/hidden states, and greedy or exact sampling.
FlashSampling has no true flashtraining/Megatron equivalent comparator; accepted
results compare against the optimization-playground incumbent and local dense
logits reference where applicable.

## Current Result

The current accepted baseline is commit `543a70499` (`Make FlashSampling IKP
profile harness self-contained`). No later candidate cleared the close-out gate
of more than 1% over this accepted optimization-playground baseline while also
meeting correctness and deployability requirements.

| Case | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| Target provider BS1 | 0.049059 ms generic target | 0.047400 ms Blackwell target | 1.035x |
| Target provider BS32 | 0.048544 ms generic target | 0.047206 ms Blackwell target | 1.028x |
| Greedy H2/H4/H8 | 0.045983 / 0.045770 / 0.045824 ms | 0.045243 / 0.045088 / 0.045118 ms | 1.015x-1.016x |
| Non-greedy H1 | 0.047392 ms | 0.045498 ms | 1.042x |
| Non-greedy two-wave H72/H80/H96/H112/H128 | 0.110197 / 0.102981 / 0.100250 / 0.097888 / 0.099019 ms stage-auto target | 0.103018 / 0.098349 / 0.095473 / 0.094516 / 0.094490 ms stage-2 target | 1.036x-1.070x |
| Serving greedy V151936/D2048 BS1/2/4/8/16/32/64/128 | 0.115532 / 0.116282 / 0.116782 / 0.119260 / 0.126716 / 0.138460 / 0.157964 / 0.202436 ms dense logits | 0.110971 / 0.105965 / 0.107206 / 0.106234 / 0.107256 / 0.111123 / 0.124744 / 0.185723 ms compile-safe Triton FlashSampling | 1.041x-1.263x |
| REAP greedy V16160/D7168 H72/H80/H96/H112/H128/H160 | 0.080265 / 0.073966 / 0.069403 / 0.067940 / 0.069843 / 0.092318 ms compile-safe FlashSampling fallback | 0.053833 / 0.055989 / 0.057716 / 0.057428 / 0.058060 / 0.066242 ms dense logits dispatch | 1.183x-1.491x |

The current target path also beats the dense matmul plus argmax floor by
1.057x, 1.088x, and 1.067x at batches 1, 32, and 64. The non-greedy two-wave
result is measured against the previously accepted Blackwell target policy, not
an old baseline.

## Optimization History

- Routed the `target` provider to the Blackwell kernel on SM100.
- Retuned `BLOCK_H=8` for greedy `2 <= H <= 8` and non-greedy `H=1`.
- Accepted a non-greedy two-wave Blackwell target gate for H=72..128; greedy
  stays single-wave because the same policy regressed by at least 20%.
- Accepted a 2-stage pipeline only for the non-greedy two-wave path after
  round36 confirmed 3.4%-6.5% over the current stage-auto target.
- Disabled FlashSampling persistent-kernel warp specialization on Blackwell.
  Round41 showed default warp-specialized Triton/target paths hit
  `PassManager::run` on serving and fallback shapes; round42 confirmed the
  compile-safe path remains 4.1%-26.3% faster than dense logits on serving
  greedy BS1..128.
- Added a Blackwell runtime threshold that rejects REAP-like greedy batches above
  H=64 to the normal dense logits path. Round42 showed dense greedy dispatch is
  18.3%-49.1% faster than compile-safe FlashSampling for H72..160, while H64
  remains a FlashSampling win and non-greedy remains eligible.
- Made the IKP/provider smoke harness self-contained enough for the close-out
  environment by resolving direct FlashSampling module imports relative to the
  repository instead of an old absolute checkout path.

## Candidate Decisions

- Rejected round48 order-balanced confirmation of apparent H80/H112 wins.
  H80 best was `target-s5` at 0.097804 ms versus target 0.098054 ms, only
  +0.26%; H112 best was `target-s5` at 0.094645 ms versus target 0.094667 ms,
  only +0.02%. The neighboring `target-wave4` candidate regressed by 4.48% at
  H80 and 4.17% at H112.
- Rejected final current-shape IKP/autoinfer pass `20260519T002142`. IKP showed
  `flashsample_blackwell_kernel` at 120 calls, 11.133 ms total, 92.777 us mean;
  `_local_reduce_samples_kernel` was 2.420 us mean. Best stage neighbors were
  H72 +0.35%, H80 +0.50%, H96 +0.09%, H112 +0.17%, and H128 +0.00%. Serving
  compile reliability had 24/24 no-error rows.
- Rejected final additional current-shape IKP/autoinfer pass `20260519T002602`.
  IKP showed `flashsample_blackwell_kernel` at 120 calls, 11.149 ms total,
  92.907 us mean; `_local_reduce_samples_kernel` was 2.452 us mean. Best stage
  neighbors were H72 +0.24%, H80 +0.48%, H96 +0.40%, H112 +0.13%, and H128
  +0.17%. These do not clear the close-out gate of more than 1% over the current
  accepted baseline.
- Rejected the CuTe/CZS successor as not production-ready. CZS now builds, the
  abstract FlashSampling CuTe plan proved 19/19 checks, and the dense GEMM core
  reached 0.053984 ms for H128/V16160/D7168. The full CuTeDSL GEMM plus argmax
  path failed equivalence against accepted dispatch: round47d matched 113/128
  samples and timed 0.067715 ms versus accepted dispatch 0.068072 ms and torch
  dense argmax 0.065698 ms. A direct FlashSampling tensor rerun improved to
  127/128 matches but still failed exact equivalence and timed 0.067745 ms
  versus accepted dispatch 0.068102 ms. The remaining blocker is a fused,
  numerically equivalent max/index epilogue plus FlashSampling API integration;
  materializing full FP32 logits is also a different workspace contract from the
  production hot path.
- Rejected larger non-greedy wave gates, greedy two-wave/2-stage variants,
  serving-shape larger V tiles, `BLOCK_H=128`, local-reduce replacement,
  workspace-only, warp-count, and extended `BLOCK_D`/`BLOCK_V` variants because
  they tied, regressed, failed correctness, or hit Triton PassManager failures.

## Verification

- Direct kernel tests cover greedy equality, logits debug mode, sampled-id
  range, seed sensitivity, shard offsets, compact local-index workspace, and
  Blackwell launch policy helpers.
- B200 artifact `round36_confirm_s2_nongreedy_h72_128.json` confirms the accepted
  stage-2 two-wave candidate with in-range non-greedy samples.
- B200 artifacts `round35_s2_greedy_guard.json` and
  `round38_greedy_deployability_guard.json` confirm the stage-2 change must not
  apply to greedy. Greedy H=72..160 target fallback paths still hit Triton
  `PassManager::run` failures and remain a deployability item, not benchmark
  noise.
- B200 artifact `round42_post_ws_guard_serving_greedy.json` confirms default
  Triton and target providers compile after the Blackwell warp-specialization
  guard, and the Triton provider beats dense logits by 4.1%-26.3% for serving
  greedy BS1..128.
- B200 artifact `round42_post_ws_guard_serving_nongreedy.json` confirms serving
  non-greedy default Triton and target providers compile and return in-range
  samples after the guard.
- B200 artifact `round42_post_ws_guard_reap_greedy.json` shows the REAP greedy
  H72..160 fallback is compile-safe but slower than dense logits; the runtime
  threshold uses that rejection to route those batches to normal logits.
- The runtime threshold remains shape-scoped to Blackwell greedy batches with
  `valid_vocab_size <= 32768`, `hidden_size >= 4096`, and `batch_size > 64`; it
  does not affect the accepted non-greedy two-wave path or serving V151936/D2048.
- Close-out artifact `closeout_nongreedy_20260519T003409Z.json` rechecked the
  accepted non-greedy H72/H80/H96/H112/H128 shapes with workspaces enabled. The
  target path returned in-range samples and measured 0.103383 / 0.098205 /
  0.096207 / 0.094772 / 0.094675 ms, faster than default Triton at 0.118510 /
  0.113401 / 0.112123 / 0.112077 / 0.112415 ms.
- Close-out artifact `closeout_serving_greedy_20260519T003409Z.json` rechecked
  serving-like greedy V151936/D2048 at BS1/64/128. Target, Triton, and torch_ref
  were all correct versus torch. Triton measured 0.112554 / 0.123271 /
  0.186212 ms versus torch_ref 0.115988 / 0.158685 / 0.199159 ms.
- Close-out artifact `closeout_provider_greedy_direct_20260519T003711Z.json`
  confirms the repo-relative direct-import provider harness runs and reports
  greedy correctness for target and Triton at BS1/32/64/72/128.
- Close-out syntax checks passed for `python/sglang/srt/layers/flashsampling/*.py`,
  `scripts/playground/bench_flashsampling_provider.py`,
  `artifacts/deep_loop_cont/bench_flashsampling_cont.py`, and the IKP profile
  shell harnesses. `git diff --check` passed after the final doc and harness
  cleanup.

## Closeout

The required loop stop was honored: the active round finished, exactly one final
additional round ran, and the VM loop was stopped with GPUs idle. No post-baseline
candidate is accepted under the updated more-than-1% gate. The current accepted
optimization-playground baseline remains `543a70499`.
