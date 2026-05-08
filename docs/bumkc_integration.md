# BUMKC Artifact Integration

`optimization-playground` can opt in to BUMKC artifacts without replacing the
existing IndexCache, dense TurboQuant, SMC-SD, or backend-specific optimized
paths.

Use:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-bumkc \
  --bumkc-plan-path <artifact-root-or-plan-dir> \
  --bumkc-fallback-mode checked
```

The loader accepts either a concrete BUMKC plan directory containing
`manifest.json`, or an artifact root containing exactly one plan directory. It
validates:

- matching plan and program IDs across manifest, HVM compiler, runtime,
  engine, simulation, and tensor smoke artifacts,
- manifest `schema_version == "bumkc.plan.v1"`,
- manifest `capability_level == "hvm_rooted_runtime_descriptor"`,
- engine export `schema_version == "bumkc.optimization_playground.v14"`,
- engine-exported manifest schema/capability fields matching `manifest.json`,
- `runtime_abi_version == "bumkc.runtime.v1"`,
- runtime smoke schema `bumkc.cuda_smoke.v11`,
- `engine == "sglang"`,
- `engine_profile == "optimization_playground"`,
- exported serving CLI flags,
- `reports/artifact-digests.json` byte counts and SHA-256 hashes before
  runtime metadata is trusted,
- canonical artifact paths for the HVM core book, tensor islands, block
  pipelines, Event Tensor plan, SM task runtime, runtime descriptor, CPU
  reference, and generated CUDA smoke plans/sources,
- matching target architecture between manifest and engine export,
- checked fallback mode,
- `--bumkc-fallback-mode checked` when BUMKC is enabled,
- preservation of custom optimizations,
- matching engine/runtime executable flags,
- compiler summary fields against the HVM tensor island, block pipeline, Event
  Tensor, and simulation artifacts, including native/fallback coverage,
  fallback bridges, side-effect counts and code sums, Event Tensor edge counts,
  dependency tensor counts, notifications, execution count, and violation
  count,
- runtime summary fields against the runtime descriptor, including
  communication-plan collective counts, side-effect counts and code sums,
  serving-state dependency counts, dependency tensor count, dependency
  descriptor count/hash including dependency tensor IDs, and runtime
  substitution bounds,
- generated CUDA runtime-smoke metadata against the runtime summary, compiler
  summary, runtime ABI, expected source path, expected binary name, task
  descriptors, and Event Tensor descriptors, including descriptor-row
  aggregate recomputation for task, dependency, dependency tensor,
  launch-domain, side-effect, serving-state, communication, and rank-topology
  fields plus schema, runtime ABI, plan, program, descriptor-table, and source
  contract hashes,
- and the required REAP validation model contract.

Before invoking any BUMKC runtime entrypoint, the serving path must call
`BumkcArtifactSummary.validate_runtime_launch()` with concrete dynamic shape
values and serving-state keys. The guard rejects missing, unknown,
out-of-range, or out-of-bucket shape substitutions and missing or unknown
serving-state bindings. Startup also validates the artifact's default launch
context through `validate_default_runtime_launch()` when `--enable-bumkc` loads
the artifact.

Add `--bumkc-require-executable` when startup must fail unless the runtime
descriptor is executable. Without that flag, non-executable BUMKC artifacts are
accepted only as checked fallback metadata.

The current integration is intentionally a narrow contract reader. It does not
alter scheduling, kernel selection, KV-cache policy, or custom optimization
defaults unless a later runtime path consumes the validated artifact summary.
