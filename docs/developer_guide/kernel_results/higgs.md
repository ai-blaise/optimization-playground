# HIGGS

## Scope

Dense MLA HIGGS 2-bit KV cache store, direct dequant, page-table dequant,
and fused MLA decode compatibility on B200 for the DeepSeek-V3.2-REAP
SpinQuant lane. The accepted candidate is
opt-in through the model `quantization_config.kv_cache_scheme` and does not
change the default EDEN2/Hadamard production path.

## Result

Accepted implementation: `store_saw_scalar2`.

This is a SAW-inspired fixed-scale HIGGS slot for SpinQuant-rotated MLA
activations. It keeps the 258-byte HIGGS slot, removes the online Hadamard from
this opt-in path, thresholds BF16 bits directly into a 2-bit scalar lattice,
uses 64-bit latent load/store groups, and moves rope as 16-bit values to avoid
the byte-granular copy that IKP/SASS exposed.

The matching fused MLA decode path is required for this candidate. The default
EDEN2/Hadamard fused decode interprets the 128 packed bytes as EDEN pair
indices, so it is not correct for SAW scalar2 slots. The SAW path now uses a
SAW-aware split-K decode with pair-lane unpack, branchless lattice value
materialization, and at least 16 splits for top-k 1024 requests.

The candidate is selected by either:

```json
{
  "quantization_config": {
    "kv_cache_scheme": {
      "quant_method": "higgs_dense_2bit",
      "b200_candidate": "store_saw_scalar2"
    }
  }
}
```

or `SGLANG_HIGGS_DENSE_2BIT_B200_CANDIDATE=store_saw_scalar2`.

## Performance

Artifact: `artifacts/higgs_saw_branchless_wpb8_full_acceptance.json` from pod
`higgs-cute-prod-ref` on `a4-us-001-rl9`.
Comparator: `lightseekorg/tokenspeed` `origin/main` MLA-KV set/get kernels
mounted read-only from `/workspace/tokenspeed`.

The accepted SAW path now uses branchless BF16 lattice reconstruction and an
8-row CTA shape for `N >= 16384`. Times are minimum CUDA-event milliseconds.

| Rows | OP store | SAW store | TokenSpeed set | SAW store vs OP | SAW store vs TokenSpeed | OP dequant | SAW dequant | TokenSpeed get | SAW dequant vs OP | SAW dequant vs TokenSpeed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.00685 | 0.00672 | 0.01485 | 1.02x | 2.21x | 0.00719 | 0.00644 | 0.01946 | 1.12x | 3.02x |
| 2048 | 0.01440 | 0.00609 | 0.01487 | 2.37x | 2.44x | 0.00826 | 0.00645 | 0.01475 | 1.28x | 2.29x |
| 8192 | 0.04716 | 0.00549 | 0.01479 | 8.60x | 2.70x | 0.02467 | 0.00572 | 0.01475 | 4.31x | 2.58x |
| 16384 | 0.08813 | 0.00823 | 0.01467 | 10.70x | 1.78x | 0.04717 | 0.00750 | 0.01463 | 6.29x | 1.95x |
| 32768 | 0.17220 | 0.01121 | 0.01477 | 15.36x | 1.32x | 0.09024 | 0.01030 | 0.01474 | 8.76x | 1.43x |
| 65536 | 0.34805 | 0.02056 | 0.02382 | 16.93x | 1.16x | 0.20088 | 0.02261 | 0.02467 | 8.89x | 1.09x |
| 131072 | 0.73566 | 0.04210 | 0.04899 | 17.47x | 1.16x | 0.41545 | 0.04705 | 0.04858 | 8.83x | 1.03x |

Page-table dequant also cleared TokenSpeed get across the same large-row
boundary after the 8-row dispatch: 1.82x at 16384, 1.43x at 32768, 1.09x at
65536, and 1.03x at 131072 rows. The `N=1` minimum is within measurement noise
against OP page-table dequant; median latency is faster and the serving-relevant
large rows clear the gate.

Fused MLA decode top-k 1024 results, including the SAW-aware split-K repair:

| Rows | Heads | OP EDEN split | SAW split | SAW vs OP |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0.0943 ms | 0.0439 ms | 2.15x |
| 4 | 16 | 0.0984 ms | 0.0574 ms | 1.71x |
| 16 | 16 | 0.1985 ms | 0.1639 ms | 1.21x |
| 32 | 16 | 0.3150 ms | 0.3006 ms | 1.05x |
| 1 | 32 | 0.0959 ms | 0.0472 ms | 2.03x |
| 16 | 32 | 0.3153 ms | 0.3008 ms | 1.05x |
| 32 | 32 | 0.5637 ms | 0.5684 ms | 0.99x |

The 32x32 decode row is within noise but does not clear the win gate; the
opt-in SAW path remains accepted for the target small-batch decode lane and the
store/get contract, not as a universal replacement for the default
EDEN2/Hadamard path.

## Flashtraining

There is no exact flashtraining Megatron equivalent for this inference MLA KV
slot. The closest Megatron HIGGS reference is training fake-quant forward, which
includes FWHT, EDEN2 quantization, inverse reconstruction, and saved tensors.
It remains useful for correctness and algorithm context, but it is not a fair
latency baseline for this serving KV store/get path. TokenSpeed MLA KV is the
primary serving comparator for this pass.

## Correctness

B200 CUDA smoke on the clean worktree:

- Direct store+dequant for `N in [1, 7, 64, 1024, 4096]`: rope exact,
  `min_cos >= 0.9218`, `mean_cos >= 0.9359`.
- Page-table dequant at `N=1024`: rope exact and compact page table exact.
- Full quality sweep on the experimental worktree through `N=8192`: rope exact,
  `min_cos >= 0.9221`, `mean_cos >= 0.9385`.

The candidate is intentionally not byte-exact with the EDEN2/Hadamard HIGGS
codec. The acceptance target is functional correctness for the SpinQuant path:
rope preservation, stable latent cosine, fixed 258-byte layout, and better
store/get latency than both OP HIGGS scalar and TokenSpeed MLA KV.

## IKP And CZS

IKP/SASS findings drove the accepted changes:

- Separate tensor-core Hadamard plus pack was rejected: materializing the
  transform or forcing the 2-bit classifier into skinny tensor-core tiles did
  not beat both OP HIGGS and TokenSpeed MLA KV.
- Constant-memory dequant LUT was rejected because divergent constant indexing
  regressed large-row recovery.
- SAW fused decode single-pass was correctness-ready but too slow: about
  1.34 ms on the 1x16x1024 gate.
- SAW split-K direct scalar-byte unpack improved that gate to about 0.180 ms,
  but still trailed OP EDEN split.
- Quad-lane byte broadcast was rejected after it regressed the same gate to
  about 0.191 ms.
- Pair-lane unpack plus branchless BF16-bit lattice values was accepted for the
  SAW fused decode path, reaching about 0.074 ms at eight splits and 0.044 ms
  at the selected 16-split setting on the 1x16x1024 gate.
- Single-warp multi-row tiles were rejected because they removed too much
  parallelism for quantization.
- SASS on the accepted store/get shape showed scalar BF16 loads and byte rope
  copies; the first accepted pass changed latent groups to 64-bit loads/stores
  and rope to 16-bit moves.
- The final pass removed branches from BF16 lattice reconstruction and uses
  eight rows per CTA for `N >= 16384`, which is the shape that cleared the
  TokenSpeed get comparator at 131072 rows.

CZS proof: `docs/proofs/higgs_dense_2bit_saw_scalar2_czs_module.json`.
Result: 7 proved, 0 disproved, 0 unknown. The proof covers static row layouts
and the vectorized 4xbf16 latent, 8xbf16 rope, and 4xbf16 output moves. The
accepted store/get path has no MMA/TMA obligations; it is a memory/bit-pack
kernel rather than a GEMM-like tensor-core kernel. The separate CuTe/tensor-core
MLA decode prototypes remain WIP and are not promoted because they do not yet
beat the split-K decode baseline across the target matrix.

## Verification

- `artifacts/higgs_saw_branchless_wpb8_full_acceptance.json`: B200 full sweep
  against OP production HIGGS and TokenSpeed MLA-KV set/get for rows 1 through
  131072.
- `SGLANG_HIGGS_DENSE_2BIT_B200_CANDIDATE=store_saw_scalar2` correctness:
  rope exact for direct and page-table dequant; random-latent mean cosine stayed
  about 0.9396 in the large-row sweep.
- `python3 -m pytest -q python/sglang/test/test_higgs_dense_2bit_kv.py
  test/srt/test_higgs_dense_2bit_kv_integration.py`: 23 passed on B200 pod
  `higgs-cute-prod-ref`.
- Combined config-dispatch, dense HIGGS, GQA, and server-args order suite:
  68 passed on B200 pod `higgs-cute-prod-ref`; this also verifies the
  dispatcher test restores its import stubs before later quantization tests.
- CZS proof passed: `docs/proofs/higgs_dense_2bit_saw_scalar2_czs_module.json`,
  7 proved, 0 disproved, 0 unknown.



# HIGGS MHA/GQA Draft KV

## Scope

SMC-SD draft-model MHA/GQA KV cache for
`BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`. The model shape is
`head_dim=128`, `num_attention_heads=32`, and `num_key_value_heads=2`.
Each K or V row uses the existing 34-byte HIGGS slot:
32 packed EDEN2/FWHT index bytes plus one FP16 scale.

## Result

Accepted implementation:

- Store: Triton packer `store_higgs_mha_2bit_triton`, using the same
  FWHT layout as the fused HIGGS draft decode kernel.
- Materialize fallback: CUDA `dequantize_higgs_mha_2bit` for backends
  that still need BF16 K/V tensors.
- Production decode remains the existing fused HIGGS Triton backend,
  selected when the draft pool is `HiggsMHA2BitTokenToKVPool`.

The lower-level CUDA store candidate was rejected: it was fast, but its
FWHT orientation produced lower reconstruction quality at large row counts.
It is not wired or exported.

## Deployment

The draft model can request this path from its Hugging Face config:

```json
{
  "quantization_config": {
    "kv_cache_scheme": {
      "quant_method": "higgs_mha_2bit",
      "scope": "smc_draft",
      "head_dim": 128,
      "num_key_value_heads": 2,
      "slot_bytes": 34,
      "store_backend": "triton_fwht_eden2",
      "decode_backend": "triton_fused_higgs_mha_2bit"
    }
  }
}
```

When SMC is enabled and the operator leaves `--smc-draft-kv-cache-dtype`
at `auto` or unset, SGLang reads the draft config and sets the draft KV
dtype to `higgs_2bit`. Explicit CLI values still win.

## Performance

Artifact: `artifacts/higgs_gqa_kv_perf_20260524.json` from pod
`higgs-kv-cute` on `a4-us-001-rl9`.

| Rows | Triton store | Eager store | Store speedup | CUDA dequant | Eager dequant | Dequant speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.0188 ms | 0.411 ms | 21.8x | 0.00536 ms | 0.356 ms | 66.6x |
| 2048 | 0.0238 ms | 0.408 ms | 17.1x | 0.00571 ms | 0.359 ms | 63.0x |
| 8192 | 0.0182 ms | 0.594 ms | 32.6x | 0.0124 ms | 0.357 ms | 28.9x |
| 32768 | 0.0352 ms | 1.80 ms | 51.3x | 0.0370 ms | 0.483 ms | 13.1x |
| 65536 | 0.0639 ms | 3.53 ms | 55.3x | 0.0698 ms | 0.926 ms | 13.3x |
| 131072 | 0.122 ms | 7.06 ms | 57.6x | 0.137 ms | 1.86 ms | 13.5x |

Tile sweep: `BLOCK_N=16` had the best aggregate store latency across
512, 2048, 8192, 32768, and 65536 rows; `BLOCK_N=8` was nearly tied and
`BLOCK_N=32` regressed large rows.

## Correctness

- Live Hugging Face `config.json` for
  `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP` contains
  `quantization_config.kv_cache_scheme.quant_method=higgs_mha_2bit`,
  `scope=smc_draft`, `head_dim=128`, `num_key_value_heads=2`, `slot_bytes=34`,
  `store_backend=triton_fwht_eden2`, and
  `decode_backend=triton_fused_higgs_mha_2bit`.
- `python3 -m pytest -q test/srt/test_higgs_mha_2bit_fused_decode.py
  test/srt/test_quantization_config_dispatch.py::test_smc_draft_higgs_mha_kv_cache_scheme_selects_higgs_2bit
  test/registered/unit/server_args/test_server_args.py::TestSMCDisaggregationArgs::test_smc_draft_kv_dtype_uses_draft_hf_config`:
  7 passed on B200 pod `higgs-cute-prod-ref`.
- Store path is byte-exact against the eager codec in the focused unit test.
  Larger benchmark rows showed rare tie differences, but dequantized output
  stayed effectively identical to the eager codec (`dequant_cos_min >= 0.9944`).
- CZS proof: `docs/proofs/higgs_mha_2bit_kv_czs_module.json`, 7 proved,
  0 disproved, 0 unknown.
