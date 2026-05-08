# BUMKC Artifact Integration

`optimization-playground` can opt in to BUMKC artifacts without replacing the
existing IndexCache, dense TurboQuant, SMC-SD, or backend-specific optimized
paths.

Use:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-bumkc \
  --bumkc-plan-path <artifact-root-or-plan-dir>
```

The loader accepts either a concrete BUMKC plan directory containing
`manifest.json`, or an artifact root containing exactly one plan directory. It
validates:

- matching plan and program IDs across manifest, runtime, engine, and tensor
  smoke artifacts,
- `schema_version == "bumkc.optimization_playground.v5"`,
- `engine == "sglang"`,
- `engine_profile == "optimization_playground"`,
- matching target architecture between manifest and engine export,
- checked fallback mode,
- preservation of custom optimizations,
- matching engine/runtime executable flags,
- runtime summary fields against the runtime descriptor, including
  communication-plan collective counts,
- and the required REAP validation model contract.

Add `--bumkc-require-executable` when startup must fail unless the runtime
descriptor is executable. Without that flag, non-executable BUMKC artifacts are
accepted only as checked fallback metadata.

The current integration is intentionally a narrow contract reader. It does not
alter scheduling, kernel selection, KV-cache policy, or custom optimization
defaults unless a later runtime path consumes the validated artifact summary.
