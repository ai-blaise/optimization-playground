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
- engine export `schema_version == "bumkc.optimization_playground.v25"`; the
  loader can still read legacy `v20` artifacts by deriving the engine
  quantization summary from tensor islands and previous `v21` artifacts by
  deriving the engine scale-up summary from the runtime descriptor, while `v22`
  artifacts derive serving hints from quantization metadata and `v23`
  artifacts derive the launch summary from the runtime substitution plan; `v24`
  artifacts derive runtime mode from the manifest/model-source contract,
- model-source export `schema_version == "bumkc.source.v11"`,
- engine-exported manifest schema/capability fields matching `manifest.json`,
- engine-exported source schema matching `source/model-source.json`,
- `runtime_abi_version == "bumkc.runtime.v1"`,
- runtime smoke schema `bumkc.cuda_smoke.v13`,
- manifest runtime mode in `{debug, trace, profile, production}`,
- engine-exported runtime mode matching the manifest for `v25` artifacts,
- `engine == "sglang"`,
- `engine_profile == "optimization_playground"`,
- exported serving CLI flags,
- `reports/artifact-digests.json` byte counts and SHA-256 hashes before
  runtime metadata is trusted,
- canonical artifact paths for the model-source export, HVM core book, tensor
  islands, block pipelines, Event Tensor plan, SM task runtime, runtime
  descriptor, CPU reference, and generated CUDA smoke plans/sources,
- canonical HVM Core source shape, including the `@main` program reference,
  required BUMKC HVM references, HVM tensor-island/source counts, HVM shape
  symbols, fallback bridges, root region, model entry, and tensor-island node
  ownership,
- HVM Core structural summaries in `source/model-source.json` against
  `ir/hvm-core-book.json`,
- model-source provenance, quantization metadata including scale layout, engine
  quantization summary fields, and source summary fields against the manifest,
  engine export, and HVM tensor island artifact,
- matching target architecture between manifest and engine export,
- checked fallback mode,
- `--bumkc-fallback-mode checked` when BUMKC is enabled,
- preservation of custom optimizations,
- matching engine/runtime executable flags,
- production runtime mode only with executable artifacts,
- compiler summary fields against the HVM tensor island, block pipeline, Event
  Tensor, and simulation artifacts, including native/fallback coverage,
  fallback bridges, MoE dispatch counts across tensor islands, block ops, and
  Event Tensors, side-effect counts and code sums, Event Tensor edge counts,
  dependency tensor counts, notifications, execution count, and violation
  count,
- runtime summary fields against the runtime descriptor, including
  communication-plan collective counts, side-effect counts and code sums,
  serving-state dependency counts, dependency tensor count, dependency
  descriptor count/hash including dependency tensor IDs, dependency scope,
  scope-specific wait expressions, runtime substitution bounds, default
  shape-bucket validity, serving-state kind enums, serving-state shape-symbol
  ownership, diagnostic slots, and watchdog timing,
- engine launch summary fields against the runtime substitution plan, including
  shape-symbol count, min/max/bucket aggregates, default substitution
  aggregates, serving-state binding counts, serving-state kind-code summary,
  and serving-state symbol count,
- generated CUDA runtime-smoke metadata against the runtime summary, compiler
  summary, runtime ABI, expected source path, expected binary name, task
  descriptors, and Event Tensor descriptors, including descriptor-row
  aggregate recomputation for task, dependency, dependency tensor,
  launch-domain, side-effect, serving-state, communication, and rank-topology
  fields, runtime diagnostic fields, launch-benchmark metadata, schema, runtime
  ABI, plan, program, descriptor-table, and source contract hashes,
- and the required REAP validation model contract.

Before invoking any BUMKC runtime entrypoint, the serving path must call
`BumkcArtifactSummary.validate_scale_up_domain()` and
`BumkcArtifactSummary.validate_target_architecture()` and
`BumkcArtifactSummary.validate_runtime_launch()`. The scale-up guard rejects
artifacts whose compiled GPU count does not match the serving domain
(`tp_size * pp_size`). The architecture guard rejects artifacts whose target SM
architecture does not match the serving device. Loading also rejects artifacts
whose runtime target GPU count, scale-up rank count, or target architecture
disagree with the engine contract. The runtime-launch guard rejects missing,
unknown, out-of-range, or out-of-bucket shape substitutions and missing or
unknown serving-state bindings. Startup also validates the artifact's default
launch context through `validate_default_runtime_launch()` when `--enable-bumkc`
loads the artifact.

`BumkcArtifactSummary.as_log_dict()` includes the accepted manifest schema,
capability level, source schema, model-source frontend, HVM capture status,
engine schema, scale-up summary, launch summary, and quantization summary, plus
runtime diagnostic and watchdog summary fields, so startup logs can audit the
exact BUMKC contract that was loaded.

When `--enable-bumkc` is active, the validated artifact can supply serving
hints for existing SGLang controls. NVFP4 weight artifacts default unset
`--quantization` to `modelopt_fp4`, and FP8 weight artifacts default it to
`fp8`. If the MoE runner backend is still `auto`, those quantized paths default
to `flashinfer_trtllm`. Explicit user settings, including `--quantization
unquant`, are preserved.

Add `--bumkc-require-executable` when startup must fail unless the runtime
descriptor is executable. Without that flag, non-executable BUMKC artifacts are
accepted only as checked fallback metadata outside production runtime mode.
Production-mode artifacts are always required to be executable.

The current integration is intentionally narrow. It validates artifacts, applies
only the serving hints above when the corresponding user controls are unset,
and otherwise does not alter scheduling, KV-cache policy, or custom
optimization defaults unless a later runtime path consumes the validated
artifact summary.
