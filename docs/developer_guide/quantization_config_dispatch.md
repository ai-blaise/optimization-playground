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

### 2. IndexerK8 FP8 fused-store

A new top-level `indexer_quantization` field declares that the
checkpoint expects the IndexerK8 FP8 fused-store path:

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

Effect on `server_args`:

| Field                 | Mutation                                              |
|-----------------------|-------------------------------------------------------|
| `quant_method`        | If `"fp8_e4m3"`, the dict is recorded on `server_args.indexer_quantization_declared`. |

The runtime fused-store dispatch in
`python/sglang/srt/layers/attention/nsa/nsa_indexer.py` selects the FP8
path based on the indexer key tensor dtype. The
`indexer_quantization_declared` attribute is a declarative metadata
record: it documents that the model intends FP8 indexer keys and is
available to startup checks and logging without forcing a kernel
override.

Other `quant_method` values (e.g. `"int4"`) are ignored.

## Precedence

CLI flags always take precedence:

1. If the operator passes `--enable-turboquant-dense-kv-cache`, the flag
   is already `True` before the dispatcher runs and the dispatcher's
   check `if not server_args.enable_turboquant_dense_kv_cache` is
   short-circuited. The CLI value is preserved.
2. If the operator passes `--turboquant-dense-kv-preset` with a value
   different from the dataclass default `latent_2p5bit_nc`, the
   dispatcher does not overwrite it.
3. Absent CLI flags, the config field promotes the corresponding
   `server_args` value.

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

## Adding a new dispatch field

To extend the dispatcher with another declarative field:

1. Add a `_maybe_apply_<feature>(server_args, quant_cfg)` helper to
   `quantization_config_dispatch.py` following the existing pattern.
2. Call it from `apply_quantization_config_dispatch`.
3. Add a corresponding test in
   `test/srt/test_quantization_config_dispatch.py`.
4. Document the new field's JSON shape and effect here.
