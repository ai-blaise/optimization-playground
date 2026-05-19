# G1 Gate

## Scope

G1 attention output gating for DeepSeek-V3.2-REAP. The standalone operation is:

```text
gate = sigmoid(linear_out)
output = attn_out * gate
```

The production SGLang path applies output-only G1 before `o_proj`. The
committed optimization-playground standalone incumbent remains
`fused_adaptive16wave` from local commit
`63cc94c112c8f3e77c9742104d5a9b6f679d01b3`.

Close-out acceptance used the updated >1% gate against the current accepted OP
baseline, with exact correctness and production deployability still mandatory.

## Current Decision

Accepted implementation:

- Candidate: `g1_nvfp4_o_proj_input_prologue`
- Benchmark label: `production_optin_g1_nvfp4_o_proj_input_prologue`
- Runtime gate: `SGLANG_DEEPSEEK_V2_G1_FUSED_FP4_O_PROJ=1`
- Dispatch scope: DeepSeek G1 `o_proj`, NVFP4 quant methods
  `CompressedTensorsW4A4Fp4` and `ModelOptFp4LinearMethod`, BF16 CUDA 2D
  activations, `input_is_parallel`, `M >= 1024`

The candidate fuses SGLang's output-only G1 sigmoid/multiply into the
FlashInfer-compatible CuTe-DSL NVFP4 activation quantization prologue that runs
immediately before the existing FP4 `o_proj` GEMM:

```text
incumbent:
  gated_bf16 = g1_gate_forward_fused(gate_bf16, attn_bf16)
  x_fp4, x_scale = fp4_quantize(gated_bf16, input_scale)
  out = fp4_gemm(x_fp4, weight_fp4, x_scale, weight_scale, alpha)

candidate:
  x_fp4, x_scale = g1_nvfp4_quantize(attn_bf16, gate_bf16, input_scale)
  out = fp4_gemm(x_fp4, weight_fp4, x_scale, weight_scale, alpha)
```

Unsupported shapes, dtypes, devices, quant methods, training/full-forward
semantics, and non-opt-in runs fall back to the accepted standalone
`fused_adaptive16wave` materialization path.

## Flashtraining Comparator

Baseline repository:
`/root/b200-run-20260518/repos/Megatron-LM`, branch `flashtraining-rdep`,
commit `844bf42af7ce73a1b80e4b1ccb3c221dd63de35d`.

Comparator kernel:
`megatron/core/fusions/fused_g1_gate.cu`.

The flashtraining kernel is the real full-forward semantic comparator for G1.
It stores both output and gate for training, while SGLang inference only needs
output. The prior-round comparator remains valid for standalone G1. The new
NVFP4 `o_proj` input prologue has no Megatron flashtraining equivalent because
it is an SGLang-only inference boundary optimization. No
flashtraining-to-optimization-playground port required.

## Standalone Metrics

Median of 5 repeats, B200, Torch `2.12.0+cu130`,
`sglang-kernel 0.4.2.post2`, hidden size `7168`, warmup 100, iters 1000.

| Tokens | Flashtraining full (ms) | Prior OP incumbent `fused_mid8wave` (ms) | Current OP baseline `fused_adaptive16wave` (ms) | Delta vs prior OP incumbent | Delta vs flashtraining | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 0.008191888 | 0.006144128 | 0.005121728 | +16.64% | +37.48% | accepted in prior round |
| 896 | 0.010248384 | 0.006150912 | 0.006149760 | +0.02% | +39.99% | tie |
| 1024 | 0.011526752 | 0.007149920 | 0.006287040 | +12.07% | +45.46% | accepted in prior round |
| 1408 | 0.014349264 | 0.009027168 | 0.008198656 | +9.18% | +42.86% | accepted in prior round |
| 4096 | 0.041000143 | 0.028713568 | 0.028668800 | +0.16% | +30.08% | no material change |
| 8192 | 0.077729469 | 0.056713089 | 0.056674656 | +0.07% | +27.09% | no material change |

Artifacts:

- `/root/b200-run-20260518/workers/g1_gate/bench/round2_medians_20260518T203927Z/summary.json`
- `/root/b200-run-20260518/workers/g1_gate/bench/flashtraining_20260518T203811Z/quick_fused_mid8wave.json`
- `/root/b200-run-20260518/metrics/g1_gate_result.json`

## SGLang Boundary Metrics

Final B200 verification for `g1_nvfp4_o_proj_input_prologue`, hidden size
`7168`, warmup 30, iters 120, 3 repeats. The OP incumbent is the accepted
standalone G1 materialization followed by FlashInfer `fp4_quantize`.

| Tokens | OP incumbent chain (ms) | Candidate (ms) | Delta vs OP incumbent | Fused dispatch | FP4 bytes/scales exact | Decision |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 512 | 0.012208800 | 0.011971467 | +1.94% | no, fallback | yes | no fused promotion at this shape |
| 1024 | 0.016439467 | 0.012249600 | +25.49% | yes | yes | accept |
| 1408 | 0.018469866 | 0.011994933 | +35.06% | yes | yes | accept |
| 4096 | 0.042981601 | 0.029477066 | +31.42% | yes | yes | accept |
| 8192 | 0.081766136 | 0.055622399 | +31.97% | yes | yes | accept |

Final verification artifact:
`/root/b200-run-20260518/workers/g1_gate/bench/final_verify_20260519T0027Z/summary.json`.

Earlier accepted-loop artifact:
`/root/b200-run-20260518/workers/g1_gate/bench/g1_prod_fp4_prologue_20260519T001748Z/summary.json`.

Autoinfer artifact:
`/root/b200-run-20260518/workers/g1_gate/autoinfer_runs/1f233d1d-6b44-4405-a0f5-268d12aae0f5.json`.

## IKP And CZS

Empirical-first progression:

1. Round2 IKP showed standalone G1 was launch/CTA-geometry limited, not an MMA
   or memory-instruction-count problem.
2. Round3 autoinfer plus IKP tested neighboring launch shapes and rejected
   them, establishing a local maximum for the current standalone shape.
3. The larger redesign then targeted the SGLang inference boundary: eliminate
   the separate BF16 gated materialization and FP4 quantization read/launch.

Round3 rejection evidence:

- Benchmark:
  `/root/b200-run-20260518/workers/g1_gate/bench/round3_candidates_20260518T215857Z/summary.json`
- Autoinfer:
  `/root/b200-run-20260518/workers/g1_gate/autoinfer_runs/33b5038b-8b03-4537-bfb8-0830f68b2b15.json`
- IKP:
  `/root/b200-run-20260518/workers/g1_gate/ikp/round3_reject_20260518T234331Z/nsys_kernel_compact_summary.json`

Accepted prologue IKP:

| Shape | Kernel | Median duration |
| ---: | --- | ---: |
| 1024 x 7168 | `G1NVFP4QuantizeSwizzledKernel` | 8288 ns |
| 4096 x 7168 | `G1NVFP4QuantizeSwizzledKernel` | 31648 ns |

Boundary IKP artifact:
`/root/b200-run-20260518/workers/g1_gate/ikp/g1_prod_fp4_prologue_20260519T001748Z/summary.json`.

CZS proof artifact:
`docs/proofs/g1_nvfp4_o_proj_prologue_czs_module.json`.

CZS result:
`8 Proved | 0 Disproved | 0 Unknown` for BF16 load, FP4 packed store, and
scale store layout/vectorization obligations.

Proof log:
`/root/b200-run-20260518/workers/g1_gate/logs/czs_g1_nvfp4_prologue_final_20260519T0027Z.log`.

## Candidate Ledger

| Candidate | Type | Result | Reasoning |
| --- | --- | --- | --- |
| `fused_adaptive16wave` | standalone scalar/control CUDA | accepted prior baseline | IKP showed launch underfill; adaptive block/wave policy beat prior OP incumbent and flashtraining full-forward comparator on key production shapes. |
| `round3_launch_neighbors` | standalone launch policy | rejected | Autoinfer and IKP found nearby B128 and 17-wave variants did not clear the gate and introduced regressions; current standalone shape is locally saturated. |
| `tensor_o_proj_epilogue` | tensor-op fusion sketch | rejected | An `o_proj` output epilogue cannot apply input-side G1 after GEMM without changing semantics. |
| `g1_nvfp4_o_proj_input_prologue` | CuTe-DSL input prologue | accepted | Byte-exact FP4 payload/scales; +25.49% to +35.06% on 1024..8192 tokens in final B200 verification; CZS proof passed. |
| `g1_nvfp4_o_proj_input_prologue_threshold512` | dispatch threshold change | rejected | Final additional round did not complete correctness/benchmark validation, so it is not production-ready. The accepted threshold remains `M >= 1024`. |

## Tensor And CuTe Audit

- Standalone G1 is elementwise sigmoid/multiply; there is no standalone MMA or
  tensor-core computation to map directly to tensor ops.
- The credible tensor-adjacent route is pre-GEMM input-side fusion into
  `o_proj` activation quantization. This keeps tensor cores in the existing FP4
  GEMM and removes an upstream launch plus BF16 traffic.
- The accepted prologue is CuTe-DSL and uses CZS as the proof path. It does not
  claim a new MMA mainloop.
- Triton was not used as the production implementation for this candidate. If a
  future G1 path starts from Triton, it should remain a baseline/reference until
  a CuTe/CZS successor is implemented and measured.

## Verification

- Focused G1/DeepSeek correctness tests: `17 passed`.
- CZS final proof: `8 Proved | 0 Disproved | 0 Unknown`.
- Final B200 benchmark verification:
  `/root/b200-run-20260518/workers/g1_gate/bench/final_verify_20260519T0027Z/summary.json`.
- FP4 payload and scale tensors are byte-exact against
  `g1_gate_forward_fused + fp4_quantize` for
  `512,1024,1408,4096,8192`.
- Final additional round was run exactly once after the active round, as
  requested; its `threshold512` candidate was rejected and no further
  optimization rounds were run.
