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
