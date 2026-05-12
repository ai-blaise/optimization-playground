# Config-Side Dispatch for Custom KV / Indexer Kernels

A model checkpoint can opt into the TurboQuant 2.5-bit dense KV kernel
and the IndexerK8 FP8 fused-store path declaratively via fields in
`config.json`'s `quantization_config` block, without requiring the
operator to pass CLI flags at server launch.

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

Effect on `server_args`:

| Field | Mutation |
| --- | --- |
| `quant_method` | If `"fp8_e4m3"`, `"nvfp4_e2m1_ue8m0"`, or `"disabled"`, the dict is recorded on `server_args.indexer_quantization_declared`. |

This dispatch runs after the model config is loaded from Hugging Face, so a
Hub checkpoint can select the indexer format from its `config.json` without a
launcher override.

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
IndexCache, dense TurboQuant, HiSparse allocation, LayerSplit index-buffer, and
hierarchical-cache host mirror paths. That keeps production memory sizing and
cache offsets aligned for the full custom stack. The current HiSparse TileLang
indexer kernels are FP8-layout-specific; when `nvfp4_e2m1_ue8m0` is selected,
runtime dispatch uses the DeepGEMM FP4 IndexCache logits path instead of the
HiSparse FP8 indexer kernels.

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

1. CLI flag (none today; reserved for a future
   `--nsa-fused-store-mode={force,disable,auto}` knob).
2. `quantization_config.indexer_quantization.quant_method`:
   `"fp8_e4m3"` → force fused-store; `"disabled"` → force fallback.
3. Auto-detection by platform and `can_use_nsa_fused_store(...)`.

This matches the precedence pattern used elsewhere in `server_args.py`
(e.g. `nsa_prefill_backend`, `nsa_decode_backend`).

## Worked example

CLI form (existing, unchanged):

```bash
python -m sglang.launch_server \
  --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
  --enable-turboquant-dense-kv-cache \
  --turboquant-dense-kv-preset latent_2p5bit_nc
```

Config form (new): no flags required if the model's `config.json`
contains both fields above.

```bash
python -m sglang.launch_server \
  --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1
```

The two forms produce identical `server_args` (and therefore identical
runtime kernel dispatch).

## Tests

`test/srt/test_quantization_config_dispatch.py` verifies:

- No-op when `quantization_config` is absent.
- TurboQuant `kv_cache_scheme` enables the flag and copies the preset.
- The CLI-supplied preset survives a config that asks for a different
  one.
- Unknown `quant_method`, unknown preset, and missing fields all fall
  back cleanly.
- The two fields compose (a config with both set in the same
  `quantization_config`).
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
