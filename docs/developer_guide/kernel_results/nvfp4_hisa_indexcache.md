# NVFP4 HISA + IndexCache

## Scope

Combined NVFP4 IndexCache + HISA Indexer dispatch on B200. Acceptance is only
at the full combined inference boundary. Standalone IndexCache or standalone
HISA measurements are preserved as subcomponent evidence.

## Current Result

Accepted candidate: `hf_config_min_seq_8192`, committed as `fcf003560`.
When a model-card NVFP4 IndexCache+HISA config sets `hisa.enabled=true` but
omits `hisa.min_seq_len`, the runtime sequence gate defaults to `8192` instead
of the generic HISA `65536` gate. The query-row guard accepted in `177532492`
still controls the exact admission boundary.

Acceptance matrix: `topk=1024,2048`, prefixes
`8193,16384,20480,24576,28672,32767,32768,65536`, query rows `256,512,1024`,
64 heads, 4:1 HISA, DeepGEMM scorer, clean idle-gated B200 run.

| Slice | Prior OP incumbent | Current OP | Improvement |
| --- | ---: | ---: | ---: |
| All shapes geomean | 0.414783536 ms | 0.308526836 ms | +34.440018% |
| TopK=1024 geomean | 0.413926497 ms | 0.308861465 ms | +34.016879% |
| TopK=1024 weighted 3x | - | - | +34.228282% |

The prior OP incumbent was the production model-card omission path where
`hisa_min_seq_len` remained at the generic `65536` default, so HISA was not
reachable for many shapes that the accepted query guard already proved safe.
Clean acceptance artifact:
`artifacts/nvfp4_hisa_deep_loop/autoinfer/combined_boundary/hf_config_min_seq_8192_accept_20260518T223909Z_seed601_eval.json`.

## Flashtraining Comparator

The strongest measured flashtraining equivalent is the exact-env full combined
NVFP4 IndexCache-fed HISA comparator at `ai-blaise/Megatron-LM` ref
`844bf42af7ce73a1b80e4b1ccb3c221dd63de35d` with
`MEGATRON_HISA_SELECTOR_BACKEND=bmm`,
`MEGATRON_HISA_SELECTOR_ROW_CHUNK=512`,
`MEGATRON_HISA_CANDIDATE_SLOT_GROUP=16`, and
`MEGATRON_HISA_TARGET_TRITON=1`.

Closest exact-env full-combo matrix: `topk=1024,2048`, prefixes
`8193,16384,32768`, query rows `32,64,256`, 64 heads.

| Comparator | Geomean |
| --- | ---: |
| OP current path in comparator artifact | 0.135254023 ms |
| OP candidate path in comparator artifact | 0.117169933 ms |
| Flashtraining prior BMM full combo | 0.678317805 ms |
| Flashtraining candidate BMM full combo | 1.084122101 ms |

Flashtraining's candidate path was +825.256229% slower than the OP candidate in
that artifact, and the low-query flashtraining policy replay regressed by
-37.431604% on the BMM full-combo path. No
flashtraining-to-optimization-playground port required.

Boundary note: this is the full flashtraining NVFP4 IndexCache+HISA combination
available at the requested ref, but its selector is the production BMM path over
NVFP4 fake-dequantized K from `IndexCacheHISAConfig`; the packed/cuBLASDx HISA
extension was unavailable in that checkout, and Megatron does not have SGLang's
paged DeepGEMM logits/layout or the SGLang HF config admission gate. Artifacts:
`artifacts/nvfp4_hisa_deep_loop/flashtraining/flashtraining_exact_env_policy_compare_20260518T222552Z.json`,
`flashtraining_exact_env_target_triton_20260518T222552Z.json`, and
`flashtraining_exact_env_metadata_20260518T222552Z.json`.

## IKP Evidence

Empirical bottleneck order came first. Combined-boundary IKP attributed the
accepted shape's time to mean-pool/precompute at about 40.7%, block TopK at
about 31.7%, block-score at about 11.1%, candidate DeepGEMM at about 6.4%,
block-rep quant at about 6.0%, and fused mask/topK/map at about 2.3%.
Artifact:
`artifacts/nvfp4_hisa_deep_loop/autoinfer/cute_followup/ikp_bottleneck_summary_20260518T2254Z.json`.

That evidence justified moving from query-gate tweaks into block-rep and
IndexCache-amortization candidates. The follow-up search used the flashtraining
target Triton path as a reference baseline and attempted CZS/CuTe-compatible
page/block-rep successors rather than accepting a Triton-derived shape alone.

## Close-Out Rejections

| Candidate | Correctness | Metrics | Decision |
| --- | --- | --- | --- |
| Current-shape policy/kernel tweaks | Correct when applicable | `block_topk_prefix_trim`: -5.315% all / -5.205% TopK=1024. No-fused mask/topK: -22%. Runtime precompute: +1% but raw order mismatch. q28672 threshold: +1.08% under the prior 3% gate. | Rejected as local maximum evidence for the current implementation shape. Artifact: `artifacts/nvfp4_hisa_deep_loop/autoinfer/current_shape/current_shape_local_max_20260518T2313Z.json`. |
| Page-rep storage tied to IndexCache update | CZS static layout proof passed, but exact raw TopK did not hold | Initial production probe had raw TopK mismatch. | Rejected. Set-equal/order-mismatched results are not exact correctness. Artifacts: `artifacts/nvfp4_hisa_deep_loop/autoinfer/production_page_reps/page_rep_exact_semantics_rejection_20260518T235024Z.json` and `artifacts/nvfp4_hisa_deep_loop/autoinfer/production_page_reps/nvfp4_hisa_page_rep_cache_czs_*.log`. |
| Raw-order block-rep cache | Set-equal only | Incumbent inline-vs-inline at `prefix=8193, topk=1024, qrows=256` had raw position mismatches `[161172,161096]` but zero set-mismatch rows; precomputed-vs-inline had `162885` raw mismatches with same sets. | Rejected. The lane cannot promote set-equal/order-mismatched semantics. Artifact: `artifacts/nvfp4_hisa_deep_loop/autoinfer/production_block_reps/exact_block_raw_order_semantics_blocker_20260518T235719Z.json`. |
| Canonical sorted block-rep cache, final bounded round | Canonical order exact for all measured rows | TopK=1024 q512 geomean one-use -4.493285%, reuse2 +5.266374%. TopK=1024 q1024 geomean one-use -4.025122%, reuse2 +4.342897%. | Rejected for production acceptance. Reuse is promising, but there is no exact deployable reuse-admission proof; one-use production boundary loses. Artifact: `artifacts/nvfp4_hisa_deep_loop/autoinfer/production_block_reps/final_one_round_20260519T002313Z.summary.json`. |

The close-out acceptance gate was relaxed to `>1%`, but no new candidate was
both exact and production-deployable above that gate. The accepted baseline
therefore remains `fcf003560`.

## CZS/CuTe Status

The current NVFP4 IndexCache layout proof is preserved in
`docs/proofs/nvfp4_indexcache_czs_layout.json`; its log records legal layout
coverage and rejected vectorization obligations. The page/block-rep redesign
also received CZS layout probes, including a static `b1/m512` pass, but the
candidate was not retained in production code because exact semantics and
one-use performance did not clear acceptance.

No CuTe/CZS successor was promoted. The evidence says the next viable design is
not a local wrapper around the current page-rep probe; it needs a deterministic
TopK-order contract plus an inference-system reuse/admission policy that proves
precompute is amortized.

## Verification

- Final one-shot B200 round: `artifacts/nvfp4_hisa_deep_loop/autoinfer/production_block_reps/final_one_round_20260519T002313Z.summary.json`.
- `python3 -m py_compile benchmark/nsa/bench_nvfp4_hisa_indexer.py python/sglang/jit_kernel/nvfp4_indexer.py python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py`: passed.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=test_shim:python venv/bin/python -m pytest -q python/sglang/jit_kernel/tests/test_nvfp4_hisa_indexer.py -k "compression_ratio_4to1_block_budget_table or precomputed_deepgemm_matches_inline_deepgemm or deepgemm_skips_unsupported_head_count"`: 5 passed, 12 deselected.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=test_shim:python venv/bin/python -m pytest python/sglang/test/test_nsa_indexer_policy.py -q`: 4 passed, 2 warnings.
- `python3 -m json.tool docs/proofs/nvfp4_indexcache_czs_layout.json` and `python3 -m json.tool artifacts/nvfp4_hisa_deep_loop/autoinfer/production_block_reps/final_one_round_20260519T002313Z.summary.json`: passed.
- Exact-env flashtraining target Triton probe: supported with env enabled,
  unsupported after env disabled; 0.377728 ms Triton versus 1.988384 ms
  fallback, max absolute diff 5.82e-10.

## Close-Out

Unaccepted production page/block-rep integration code was removed; all useful
rejection and proof artifacts remain in `artifacts/nvfp4_hisa_deep_loop/`.
