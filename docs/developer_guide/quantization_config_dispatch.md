# Config-Side Dispatch for Custom KV / Indexer Kernels

A model checkpoint can opt into the TurboQuant 2.5-bit dense KV kernel,
HIGGS 2-bit dense KV kernel, NSA IndexCache mode, and the IndexerK8
FP8/NVFP4 cache formats declaratively via fields in `config.json`'s
`quantization_config` block, without requiring the operator to pass CLI
flags at server launch.

The dispatcher lives in
`python/sglang/srt/layers/quantization/quantization_config_dispatch.py`
and is invoked from `ServerArgs._handle_model_specific_adjustments`
right after the model's `hf_config` is resolved.

## Recognized fields

### 1. TurboQuant 2.5-bit dense KV

Extend the existing `kv_cache_scheme` dict with the
`turboquant_dense` `quant_method`:

```json
{
  "quantization_config": {
    "quant_algo": "...",
    "kv_cache_scheme": {
      "quant_method": "turboquant_dense",
      "preset": "latent_2p5bit_nc",
      "slot_bytes": 274,
      "packed_bits": 2.5,
      "kv_dim": 576,
      "latent_dim": 512,
      "rope_dim": 64
    }
  }
}
```

Effect on `server_args`:

| Field            | Mutation                                              |
|------------------|-------------------------------------------------------|
| `quant_method`   | If `"turboquant_dense"`, `enable_turboquant_dense_kv_cache` becomes `True`. |
| `preset`         | Copied into `turboquant_dense_kv_preset` if it is a known preset and the operator did not override the preset on the CLI. |

The `slot_bytes`, `packed_bits`, `kv_dim`, `latent_dim`, and `rope_dim`
fields are informational at this layer; they document the layout
expected by the kernel and are validated downstream by
`TurboQuantDenseKVConfig`.

Recognized presets: `latent_k8`, `latent_4bit_nc`, `latent_k3_nc`,
`latent_2p5bit_nc`. Unknown presets are ignored (logged at INFO) so the
checkpoint still loads on a stock SGLang build.

### 1b. HIGGS 2-bit dense MLA KV (alternative to TurboQuant 2.5-bit)

A second `kv_cache_scheme` flavor uses the HIGGS lattice-quantizer
(Pletka et al., 2025; reference: the AquaKV
`HiggsQuantizer` at https://github.com/goodevening13/aquakv) and
yields a slightly smaller per-token slot than the 2.5-bit TurboQuant
preset:

```json
{
  "quantization_config": {
    "kv_cache_scheme": {
      "quant_method": "higgs_dense_2bit",
      "slot_bytes": 258,
      "packed_bits": 2,
      "kv_dim": 576,
      "latent_dim": 512,
      "rope_dim": 64
    }
  }
}
```

Effect on `server_args`:

| Field          | Mutation                                              |
|----------------|-------------------------------------------------------|
| `quant_method` | If `"higgs_dense_2bit"`, `enable_higgs_dense_2bit_kv_cache` becomes `True`. |

Slot layout (258 bytes / token, vs 274 for `latent_2p5bit_nc`):

```
[128 B packed 4-bit pair indices] [2 B fp16 block scale] [128 B bf16 rope]
```

* Codebook: EDEN2-16 (16 entries of 2-d float32, from the public
  AquaKV grid). Quantization is by nearest-neighbor in 2-d
  (`argmax 2 x.G^T - ||G||^2`).
* Rotation: a single orthonormal block-Hadamard of order 512 per
  token (the "block-diagonal Hadamard rotation" mandated by
  SAW-INT4); since `latent_dim == hadamard_groupsize`, this is one
  Hadamard block per token.
* Per-token scale: `||FWHT(latent)|| / sqrt(512)` stored as fp16, so
  the decode formula is `latent_recon = InvFWHT(scale * G[idx])`.

The TurboQuant and HIGGS dense KV paths are mutually exclusive;
declaring or enabling both raises `ValueError` loudly. The HIGGS
path requires `--kv-cache-dtype=bfloat16` (set automatically when
the auto default is used).

#### Split-K decode (`--higgs-mla-decode-num-splits`)

The HIGGS fused dense MLA decode kernel uses the same split-K
parallelization pattern as TurboQuant's `decode_2p5_split_rotated`.
The topk loop is sharded across `num_splits` blocks per
`(row, head)`; each block produces a partial online-softmax tuple
`(m, l, acc[0..511])`, and a merge stage combines partials via the
log-sum-exp identity, normalizes by the global `l`, and runs the
inverse FWHT_512.

The `--higgs-mla-decode-num-splits` flag (default `16`, matching
TurboQuant's tuned default) controls how aggressively the topk loop
is parallelized:

* `--higgs-mla-decode-num-splits=16` (default) — split-K decode.
  Restores throughput at small batch sizes (b=1..4) where the
  topk-reduction loop in the single-pass kernel would otherwise
  starve SMs. Required for TTFT-sensitive paths and single-user
  decode.
* `--higgs-mla-decode-num-splits=1` — single-pass kernel. Acceptable
  when `num_rows * num_heads` already saturates the GPU (typically
  b >= 8 on H200); avoids the small `mid` scratch
  (`R*H*num_splits*514*4B`) and `q_rotated` scratch
  (`R*H*512*4B`).

Surrogate gate (H200, kv_lora_rank=512, num_heads=128,
pool=2 M tokens, RUNS=8, WARMUP=3); HIGGS row uses the split-K
default:

| batch | topk | HIGGS med (ms) | HIGGS p95 (ms) | TQ med (ms) | TQ p95 (ms) | d_med (%) |
|------:|-----:|---------------:|---------------:|------------:|------------:|----------:|
|     1 | 2048 |          0.185 |          0.191 |       0.299 |       0.302 |    -37.95 |
|     1 | 4096 |          0.344 |          0.347 |       0.567 |       0.568 |    -39.32 |
|     1 | 8192 |          0.662 |          0.663 |       1.107 |       1.107 |    -40.15 |
|     2 | 2048 |          0.314 |          0.316 |       0.541 |       0.543 |    -42.05 |
|     4 | 2048 |          0.574 |          0.576 |       0.993 |       0.994 |    -42.21 |
|     8 | 2048 |          1.100 |          1.102 |       1.901 |       1.906 |    -42.11 |
|    16 | 2048 |          2.140 |          2.141 |       3.710 |       3.712 |    -42.32 |
|    32 | 2048 |          4.206 |          4.208 |       7.333 |       7.342 |    -42.64 |

Before this kernel landed, the HIGGS single-pass decode was +344-379%
slower than TurboQuant at b=1 (topk 2048-8192). The split-K port
recovers the small-batch regression and adds a uniform ~38-43% win
across the sweep on top of the already-favorable memory profile.

### 2. NSA indexer quantization

A new top-level `indexer_quantization` field declares that the
checkpoint expects a specific NSA indexer cache format:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "fp8_e4m3",
      "scale_strategy": "per_token",
      "scale_bytes": 4
    }
  }
}
```

For the Blackwell NVFP4 indexer path, use:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "value_format": "e2m1",
      "scale_format": "ue8m0",
      "scale_block_size": 32
    }
  }
}
```

To enable ordinary IndexCache from the model card without enabling HISA,
add the nested `indexcache` declaration:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "indexcache": {
        "enabled": true,
        "freq": 4,
        "pattern": "FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF"
      }
    }
  }
}
```

For the production NVFP4 IndexCache+HISA deployment, set
`hisa.enabled=true` with `mode: "indexcache-hisa"`. A nested
`indexcache` block is optional: include it when the model card needs a
non-default frequency or explicit pattern, otherwise the dispatcher uses
the normal `nsa_indexcache_freq` default.

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "value_format": "e2m1",
      "scale_format": "ue8m0",
      "scale_block_size": 32,
      "indexcache": {
        "enabled": true,
        "freq": 4,
        "pattern": "FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF"
      },
      "hisa": {
        "enabled": true,
        "mode": "indexcache-hisa",
        "block_size": 128,
        "compression_ratio": 4.0,
        "min_seq_len": 8192,
        "execution_mode": "optimized"
      }
    }
  }
}
```

The shorter form used by the active HIGGS/NVFP4-HISA checkpoint is also
valid when the default IndexCache frequency is acceptable:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "hisa": {
        "enabled": true,
        "mode": "indexcache-hisa",
        "block_size": 128,
        "block_topk": 64,
        "compression_ratio": 4.0,
        "execution_mode": "optimized"
      }
    }
  }
}
```

If `indexcache.enabled=true` is combined with `hisa.enabled=true`, the
dispatcher requires the combined `indexcache-hisa` mode. A contradictory
`hisa.mode: "hisa"` declaration raises `ValueError` instead of silently
dropping the IndexCache half of the production path.

`indexer_mode: "indexcache"` is also accepted directly under
`indexer_quantization`. The nested `indexcache` block may use either the
launcher names (`freq`, `pattern`) or the model override names
(`index_topk_freq`, `index_topk_pattern`). CLI/deployment values still
take precedence: the config only fills in `nsa_indexer_mode`,
`nsa_indexcache_freq`, and `nsa_indexcache_pattern` when those values are
still at their defaults.

Effect on `server_args`:

| Field | Mutation |
| --- | --- |
| `quant_method` | If `"fp8_e4m3"`, `"nvfp4_e2m1_ue8m0"`, or `"disabled"`, the dict is recorded on `server_args.indexer_quantization_declared`. |
| `indexer_mode` | If `"indexcache"` and the launch did not already select another NSA indexer mode, `nsa_indexer_mode` becomes `"indexcache"`. |
| `indexcache.enabled` | If `true` and `indexer_mode` is absent, selects `"indexcache"` under the same CLI-precedence rule. |
| `indexcache.freq` / `index_topk_freq` | Copied into `nsa_indexcache_freq` when the launch still uses the default frequency. |
| `indexcache.pattern` / `index_topk_pattern` | Copied into `nsa_indexcache_pattern` when the launch did not already supply a pattern. |
| `hisa.enabled` with NVFP4 | Enables `enable_nsa_nvfp4_hisa`; defaults to `nsa_indexer_mode="indexcache-hisa"` when launch flags did not already select a mode. |

This dispatch runs after the model config is loaded from Hugging Face. The
promoted indexer fields are copied back onto the already-loaded
`hf_config`, so the DeepSeek `Indexer` construction sees
`nsa_indexer_mode="indexcache-hisa"` and the HISA knobs without a launcher
override.

Effect at runtime: the fused-store dispatch in
`python/sglang/srt/layers/attention/nsa/nsa_indexer.py` calls
`should_use_nsa_fused_store(server_args, key.dtype, indices.dtype,
page_size, ...)`. That helper consults
`server_args.indexer_quantization_declared` and:

* `quant_method == "fp8_e4m3"` forces the FP8 fused-store path. If the
  kernel's compatibility check fails for the runtime tensor shapes the
  helper raises `RuntimeError` rather than silently falling back, so
  the mismatch surfaces loudly.
* `quant_method == "disabled"` forces the fallback path even if the
  dtypes are compatible. Useful for A/B comparisons or for disabling
  the fused store on a specific deployment without rebuilding the
  checkpoint.
* `quant_method == "nvfp4_e2m1_ue8m0"` disables the FP8 fused-store
  gate and selects the Blackwell-only NVFP4 indexer route. That route
  stores 64 packed E2M1 bytes plus one packed int32 UE8M0 scale word per
  token, matching the DeepGEMM FP4 MQA indexer contract for
  `head_dim=128`.
* Field absent (`indexer_quantization_declared is None`): the helper
  falls through to the historical platform/dtype auto-detection
  (`_is_cuda and not _is_fp8_fnuz and can_use_nsa_fused_store(...)`).
  Backward-compatible — zero behavior change for stock checkpoints.

Other `quant_method` values (e.g. `"int4"`) are ignored at config-load
time.

#### How to force-disable IndexCache FP8

Add to the model's `config.json`:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "disabled"
    }
  }
}
```

The runtime always takes the fallback `act_quant` path even if every
other auto-detection signal is positive.

The same selection can be forced at launch with
`--nsa-indexer-quantization`. The default value is `auto`, which uses the
checkpoint declaration when present and otherwise keeps the FP8 path.

The selected layout is passed through the NSA pool constructors, including the
IndexCache, dense TurboQuant, HIGGS dense KV, HiSparse allocation, LayerSplit
index-buffer, and hierarchical-cache host mirror paths. That keeps production
memory sizing and cache offsets aligned for the full custom stack. The current
HiSparse TileLang indexer kernels are FP8-layout-specific; when
`nvfp4_e2m1_ue8m0` is selected, runtime dispatch uses the DeepGEMM FP4
IndexCache logits path instead of the HiSparse FP8 indexer kernels.

## Precedence

CLI flags always take precedence; `quantization_config` fields fill in
when no CLI override is supplied.

For TurboQuant 2.5-bit dense KV:

1. If the operator passes `--enable-turboquant-dense-kv-cache`, the flag
   is already `True` before the dispatcher runs and the dispatcher's
   check `if not server_args.enable_turboquant_dense_kv_cache` is
   short-circuited. The CLI value is preserved.
2. If the operator passes `--turboquant-dense-kv-preset` with a value
   different from the dataclass default `latent_2p5bit_nc`, the
   dispatcher does not overwrite it.
3. Absent CLI flags, the config field promotes the corresponding
   `server_args` value.

For the IndexCache FP8 fused-store dispatch (high to low):

1. CLI flag `--nsa-indexer-quantization`.
2. `quantization_config.indexer_quantization.quant_method`:
   `"fp8_e4m3"` → force fused-store; `"disabled"` → force fallback.
3. Auto-detection by platform and `can_use_nsa_fused_store(...)`.

For IndexCache mode selection (high to low):

1. CLI/deployment flags `--nsa-indexer-mode`,
   `--nsa-indexcache-freq`, and `--nsa-indexcache-pattern`.
2. `quantization_config.indexer_quantization.indexer_mode` or the
   nested `indexcache` declaration.
3. Dataclass defaults (`vanilla`, frequency `4`, no explicit pattern).

This matches the precedence pattern used elsewhere in `server_args.py`
(e.g. `nsa_prefill_backend`, `nsa_decode_backend`).

## Worked example

CLI form for the legacy TurboQuant path:

```bash
python -m sglang.launch_server \
  --model-path <checkpoint-without-higgs-config-dispatch> \
  --enable-turboquant-dense-kv-cache \
  --turboquant-dense-kv-preset latent_2p5bit_nc
```

Config form for the active HIGGS plus NVFP4 IndexCache+HISA path: no dense KV
or indexer flags are required if the model's `config.json` contains both
fields above.

```bash
python -m sglang.launch_server \
  --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4
```

The config form produces the same runtime kernel dispatch as supplying the
equivalent HIGGS and NVFP4 IndexCache+HISA flags explicitly.

## Tests

`test/srt/test_quantization_config_dispatch.py` verifies:

- No-op when `quantization_config` is absent.
- TurboQuant `kv_cache_scheme` enables the flag and copies the preset.
- HIGGS `kv_cache_scheme` (`quant_method=higgs_dense_2bit`) enables
  `enable_higgs_dense_2bit_kv_cache`; a bare declaration with no
  layout fields still enables; the CLI flag short-circuits a redundant
  config-side promotion.
- TurboQuant ↔ HIGGS mutual exclusion is enforced in both directions:
  a config that declares one path while the CLI already enabled the
  other raises `ValueError` loudly.
- The CLI-supplied preset survives a config that asks for a different
  one.
- Unknown `quant_method`, unknown preset, and missing fields all fall
  back cleanly. The unknown-method check is a regression guard for
  both `enable_turboquant_dense_kv_cache` and
  `enable_higgs_dense_2bit_kv_cache`.
- The two fields compose (a config with both set in the same
  `quantization_config`).
- Plain IndexCache can be selected from HF config through
  `indexer_quantization.indexcache.enabled=true` or
  `indexer_quantization.indexer_mode="indexcache"`, including frequency
  and pattern propagation, while preserving CLI/deployment overrides.
- The production NVFP4 IndexCache+HISA config composes these fields:
  `quant_method="nvfp4_e2m1_ue8m0"` and `hisa.enabled=true` resolve to
  `nsa_indexer_mode="indexcache-hisa"`, with or without an explicit
  `indexcache` block.
  A contradictory `indexcache.enabled=true` plus `hisa.mode="hisa"`
  declaration raises `ValueError`.
- A wrapper object that exposes `to_dict()` is coerced and applied
  (both TurboQuant and HIGGS paths).
- `should_use_nsa_fused_store`:
  - With no declaration, behavior matches the auto-detect path
    (regression guard for backward compatibility).
  - Declared `fp8_e4m3` + compatible runtime shapes → forces the
    fused-store path.
  - Declared `fp8_e4m3` + incompatible shapes → raises `RuntimeError`.
  - Declared `disabled` + compatible shapes → forces the fallback.
  - Unrecognized declared method → falls through to auto-detection.

## Adding a new dispatch field

To extend the dispatcher with another declarative field:

1. Add a `_maybe_apply_<feature>(server_args, quant_cfg)` helper to
   `quantization_config_dispatch.py` following the existing pattern.
2. Call it from `apply_quantization_config_dispatch`.
3. Add a corresponding test in
   `test/srt/test_quantization_config_dispatch.py`.
4. Document the new field's JSON shape and effect here.
