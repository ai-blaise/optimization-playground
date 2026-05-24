# HIGGS

## Scope

Dense MLA HIGGS 2-bit KV cache store, direct dequant, and page-table dequant on
B200 for the DeepSeek-V3.2-REAP SpinQuant lane. The accepted candidate is
opt-in through the model `quantization_config.kv_cache_scheme` and does not
change the default EDEN2/Hadamard production path.

## Result

Accepted implementation: `store_saw_scalar2`.

This is a SAW-inspired fixed-scale HIGGS slot for SpinQuant-rotated MLA
activations. It keeps the 258-byte HIGGS slot, removes the online Hadamard from
this opt-in path, thresholds BF16 bits directly into a 2-bit scalar lattice,
uses 64-bit latent load/store groups, and moves rope as 16-bit values to avoid
the byte-granular copy that IKP/SASS exposed.

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

Artifact: `artifacts/higgs_kv_cute/higgs_saw_scalar2_clean_perf_20260524.json`.
Comparator: `lightseekorg/tokenspeed` `origin/main` MLA-KV set/get kernels.

| Rows | OP HIGGS scalar store | Candidate store | TokenSpeed set | Store vs OP | Store vs TokenSpeed | Candidate get | TokenSpeed get | Get vs TokenSpeed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 0.172 ms | 0.0103 ms | 0.0177 ms | 16.8x | 1.72x | 0.0123 ms | 0.0153 ms | 1.24x |
| 65536 | 0.350 ms | 0.0196 ms | 0.0265 ms | 17.2x | 1.35x | 0.0246 ms | 0.0267 ms | 1.09x |
| 131072 | 0.737 ms | 0.0410 ms | 0.0489 ms | 18.0x | 1.19x | 0.0471 ms | 0.0493 ms | 1.04x |
| 262144 | n/a | 0.0767 ms | 0.0925 ms | n/a | 1.21x | 0.0901 ms | 0.0929 ms | 1.03x |

Full sweep on the experimental worktree also covered 512, 2048, 8192, and
16384 rows; every measured row count beat TokenSpeed for both store and get.

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

- Separate tensor-core Hadamard plus pack was rejected: the Hadamard was not the
  limiting cost once EDEN2 scoring/packing was included.
- Constant-memory dequant LUT was rejected because divergent constant indexing
  regressed large-row recovery.
- Single-warp multi-row tiles were rejected because they removed too much
  parallelism for quantization.
- SASS on the accepted shape showed scalar BF16 loads and byte rope copies; the
  accepted follow-up changed latent groups to 64-bit loads/stores and rope to
  16-bit moves.

CZS proof: `docs/proofs/higgs_dense_2bit_saw_scalar2_czs_module.json`.
Result: 7 proved, 0 disproved, 0 unknown. The proof covers static row layouts
and the vectorized 4xbf16 latent, 8xbf16 rope, and 4xbf16 output moves. This
path has no MMA/TMA obligations; it is a memory/bit-pack kernel rather than a
GEMM-like tensor-core kernel.

## Verification

- `SGLANG_HIGGS_DENSE_2BIT_B200_CANDIDATE=store_saw_scalar2` CUDA correctness
  smoke passed in pod `higgs-kv-cute` on `a4-us-001-rl9`.
- Clean-worktree large-row perf sweep passed against TokenSpeed MLA KV at
  32768, 65536, 131072, and 262144 rows.
- CZS proof passed: 7 proved, 0 disproved, 0 unknown.


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

- Full fused decode test file passed on B200: `5 passed`.
- Config-dispatch tests passed on B200 pod: `39 passed`.
- Store path is byte-exact against the eager codec in the focused unit
  test. Larger benchmark rows showed rare tie differences, but dequantized
  output stayed effectively identical to the eager codec
  (`dequant_cos_min >= 0.9944`).
- CZS proof: `docs/proofs/higgs_mha_2bit_kv_czs_module.json`, 7 proved,
  0 disproved, 0 unknown.
