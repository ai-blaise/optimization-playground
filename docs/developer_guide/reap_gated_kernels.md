# REAP Gated Kernels

This repository carries two REAP-specific gated kernels for the
`BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4` deployment lane.

## G1 Attention Gate

The `sgl_kernel.g1_gate_forward` op applies the G1 attention gate:

```text
gate = sigmoid(linear_out)
output = attn_out * gate
```

The CUDA op is registered in `sgl-kernel` and currently supports BF16 tensors.
It is sourced from the final G1 SGLang integration commit
`6478dd7cdd322d5c73370802386cb6c0a0780eed`; earlier commits in the same series
are superseded by that implementation.

Validation:

```bash
python -m pytest -q sgl-kernel/tests/test_g1_attention.py
```

### G1 B200 Iteration Log

Round 0, deployability incumbent gate, 2026-05-17:

- Incumbent: `0ea8076a1`, optimization-playground main. On NVIDIA B200
  (`sm100`) the installed `sgl_kernel` loader selected `sgl_kernel/sm100`, but
  the editable build only installed `sm90/common_ops.abi3.so`; a validation
  symlink to the SM90 library contained no `sm_100` cubin and left
  `g1_gate_forward` outputs unchanged/all-zero in the smoke probe.
- Candidate: add generic `SGL_KERNEL_ENABLE_SM100` for CUDA compiler 12.8+
  and install the common ops artifact into `sgl_kernel/sm100` when SM100A is
  not requested. This is a packaging/runtime-dispatch fix; it is not claimed as
  a kernel throughput optimization.
- Result: accepted as the new incumbent because B200 now loads
  `sgl_kernel/sm100/common_ops.abi3.so` containing `sm_100` cubins and the G1
  smoke probe matches the BF16 torch reference exactly (`maxdiff output=0.0`,
  `maxdiff gate=0.0`). Focused test result: `5 passed`.
- Command: `. /root/work/optimization-playground/.venv/bin/activate && CUDA_VISIBLE_DEVICES=0 /root/agent-runs/gpu_locked.sh pytest -q -s sgl-kernel/tests/test_g1_attention.py`.
- Caveat: full editable rebuild needed `CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSGL_KERNEL_COMPILE_THREADS=1 -DENABLE_BELOW_SM90=OFF"` in this shared CMake 4.3 validation environment.

Round 1, output-only fused dispatch, 2026-05-17:

- Incumbent: `d15e54257`, generic SM100 common ops deployability fix.
- Hotspot/result: IKP NSys import for the incumbent full G1 path at 256 tokens x
  hidden 7168 matched 200 launches of `g1_gate_cute_kernel_v2`, mean 3.152 us,
  grid `[592,1,1]`, block `[256,1,1]`; model-side dispatch allocated and wrote
  a gate tensor that is immediately discarded by `_apply_g1_gate`.
- Candidate: expose the existing output-only CUDA entry point as
  `g1_gate_forward_fused`, add Python binding/tests, and use it in the DeepSeek
  G1 hook when available.
- Result: accepted. On NVIDIA B200, `bench_g1_gate.py --tokens 1,16,64,256,1024
  --warmup 100 --iters 500 --json` measured model-style allocation path
  speedups of 19.9%, 20.1%, 20.7%, 20.6%, and 0.1% respectively; preallocated
  path was neutral to mixed because both variants are launch-floor dominated.
  IKP NSys import for fused 256-token path matched 200 launches of
  `g1_gate_cute_fused_kernel_v2`, mean 2.894 us. Correctness maxdiff matched
  the full kernel (`0.0` through 64 tokens, `0.0009765625` at 256 tokens,
  `0.00390625` at 1024 tokens).
- Correctness: `CUDA_VISIBLE_DEVICES=0 /root/agent-runs/gpu_locked.sh pytest -q
  -s sgl-kernel/tests/test_g1_attention.py` -> `10 passed`.
- Artifacts: `/root/agent-runs/g1-round1-fused-bench.json`,
  `/root/agent-runs/g1-round1-incumbent-nsys`,
  `/root/agent-runs/g1-round1-fused-nsys`.


Round 2, launch threshold 3M candidate, 2026-05-17:

- Incumbent: `1b0b0bbbb`, output-only fused dispatch.
- Hotspot/result: direct IKP trace probe at fused 256x7168 attributed the loop
  to `sigmoid_mul` (0.762 us/warp mean), then loads (0.303 us/warp), with
  stores small (0.043 us/warp). The committed CuTe path already uses approximate
  `ex2.approx.ftz.f32` + `rcp.approx.ftz.f32`, so launch geometry was the next
  low-risk lever.
- Candidate: raise `G1_BLOCK128_N_THRESHOLD` from 1.5M to 3M elements so the
  256-token production shape uses the smaller `BLOCK=128, sm_count*8` launch.
- Result: rejected. On B200, fused allocation timings for tokens
  1/16/64/256/1024 were 0.004214/0.004153/0.004113/0.004156/0.008200 ms
  versus incumbent 0.004224/0.004146/0.004189/0.004178/0.008200 ms. The only
  newly affected shape, 256 tokens, improved by ~0.5%, within measurement noise;
  preallocated timing at 256 improved by ~0.09%. Correctness still passed.
- Correctness: `pytest -q -s sgl-kernel/tests/test_g1_attention.py` ->
  `10 passed`.
- Decision: reject and revert source; no new incumbent.
- Artifacts: `/root/agent-runs/g1-round2-direct-ikp-trace_summary.json`,
  `/root/agent-runs/g1-round2-threshold3m-bench.json`,
  `/root/agent-runs/g1-round2-threshold3m-pytest.txt`.


Round 3, launch geometry sweep, 2026-05-17:

- Incumbent: `1b0b0bbbb`, output-only fused dispatch.
- Hotspot/result: Round 2 direct IKP showed compute-dominated sigmoid work and
  a small store region, so the next CuTe/Blackwell candidate was launch
  geometry rather than more memory stores. A standalone SM100 probe swept
  `BLOCK={64,128,256,512}` with grid multipliers `{16,8,4,2}` for the fused
  output-only loop.
- Candidate: replace the incumbent two-bucket launch policy with a different
  block/grid pair if one beat the current `BLOCK=128` small-shape and
  `BLOCK=256` large-shape policy.
- Result: rejected. On B200 the sweep measured launch-floor timings through 256
  tokens and about 8.196 us at 1024 tokens for all useful variants; the
  256-token results were 4.099264/4.099248/4.098368/4.098624 us for
  `b64x16`/`b128x8`/`b256x4`/`b512x2`, which is a tie within noise and keeps
  the current large-shape `BLOCK=256` choice.
- Correctness: no repo source changed; the accepted incumbent correctness
  remains Round 1's `10 passed` focused G1 test.
- Decision: reject; no new incumbent.
- Artifacts: `/root/agent-runs/g1_geometry_probe.cu`,
  `/root/agent-runs/g1-round3-geometry-probe.csv`.

Round 4, SM100 cache-streaming load/store hints, 2026-05-17:

- Incumbent: `1b0b0bbbb`, output-only fused dispatch.
- Hotspot/result: direct IKP put stores at only 0.043 us/warp mean, but the SM90
  fallback uses cache-streaming hints. A standalone SM100 probe compared normal
  `__ldg`/stores against `__ldcs`, `__stcs`, and both together in the
  output-only loop.
- Candidate: add cache-streaming load/store hints to the SM100 fused kernel if
  they reduced the current accepted timings.
- Result: rejected. On B200 the normal/cs_load/cs_store/cs_load_store timings
  for tokens 1/16/64/256/1024 were 4.094976/4.095472/4.095056/4.095392 us,
  4.095664/4.095744/4.095712/4.095616 us,
  4.095600/4.095760/4.095792/4.096336 us,
  4.099408/4.098368/4.098544/4.098496 us, and
  8.195712/8.196560/8.196560/8.196464 us respectively. The small 256-token
  difference was below noise and 1024 tokens regressed slightly.
- Correctness: no repo source changed; probe used the same output formula as the
  accepted kernel.
- Decision: reject; no new incumbent.
- Artifacts: `/root/agent-runs/g1_streaming_probe.cu`,
  `/root/agent-runs/g1-round4-streaming-probe.csv`.

Round 5, GEMM-adjacent FP4 quantization fusion screen, 2026-05-17:

- Incumbent: `1b0b0bbbb`, output-only fused dispatch.
- Hotspot/result: model-side inspection shows `_g1_gate_pre_hook` materializes a
  gated BF16 tensor before `o_proj`; `RowParallelLinear.forward` then delegates
  to the selected quant method. For the production ModelOpt NVFP4 path,
  `ModelOptFp4LinearMethod.apply` immediately calls FlashInfer/CUTLASS
  `fp4_quantize` before FP4 GEMM. A CUDA-event boundary probe measured
  `g1_gate_forward_fused`, `fp4_quantize`, and the sequence.
- Candidate: fuse G1 sigmoid-multiply into activation FP4 quantization so the
  BF16 gated intermediate is not materialized before the tensor-op GEMM input
  path.
- Result: blocked/rejected as a local G1 patch. On B200 with tokens
  1/16/64/256/1024, G1 alone measured
  0.003173/0.003047/0.003279/0.004104/0.008199 ms, FP4 activation quantization
  measured 0.014196/0.013788/0.013702/0.013634/0.013734 ms, and the sequence
  measured 0.016917/0.017262/0.016803/0.016817/0.016912 ms. Nsight for the
  256-token sequence showed separate `g1_gate_cute_fused_kernel_v2` and
  FlashInfer `NVFP4QuantizeSwizzledKernel` launches. This identifies a real
  future fusion target, but implementing it correctly requires a new
  FlashInfer-compatible swizzled-scale FP4 quantizer/contract rather than a
  safe change to the local G1 kernel.
- Correctness: no repo source changed; the candidate was not implemented.
- Decision: reject/block; no new incumbent.
- Artifacts: `/root/agent-runs/g1_fp4_boundary_probe.py`,
  `/root/agent-runs/g1-round5-fp4-boundary-probe.json`,
  `/root/agent-runs/g1-round5-fp4-boundary-nsys.nsys-rep`,
  `/root/agent-runs/g1-round5-fp4-boundary-nsys-kernels.csv`.

Round 6, more aggressive sigmoid approximation, 2026-05-17:

- Incumbent: `1b0b0bbbb`, output-only fused dispatch.
- Hotspot/result: direct IKP still points at `sigmoid_mul`; the accepted SM100
  kernel already uses `ex2.approx.ftz.f32` and `rcp.approx.ftz.f32`, so the
  only remaining local math lever is a lower-accuracy approximation.
- Candidate: replace the current exponential sigmoid with a cheaper clipped
  rational approximation in the output-only loop.
- Result: rejected. On B200, rational timings for tokens 1/16/64/256/1024 were
  3.604320/4.096304/4.096400/4.098336/8.191552 us versus incumbent
  4.095696/4.095536/4.096336/4.098416/8.191568 us. Only the 1-token case
  improved, but max absolute BF16 output difference versus the incumbent was
  0.21142578, far outside the existing G1 correctness tolerance and not
  numerically acceptable for the Megatron-equivalent gate.
- Correctness: rejected by numerical check; no repo source changed.
- Decision: reject; no new incumbent.
- Artifacts: `/root/agent-runs/g1_approx_probe.cu`,
  `/root/agent-runs/g1-round6-approx-probe.csv`.

Current saturation evidence, 2026-05-17:

- Best accepted incumbent remains `1b0b0bbbb`. Further local G1 kernel work is
  launch-floor limited at small/medium token counts, has no demonstrated memory
  hint win, and cannot safely reduce sigmoid math without violating numerical
  equivalence. The remaining plausible performance work is cross-operator
  fusion into ModelOpt/FlashInfer FP4 activation quantization before `o_proj`,
  which is outside a local G1 scalar/vector kernel patch and should be handled
  as a separate tensor-op-adjacent quantization contract change.


## GatedNorm Forward

The `sglang.jit_kernel.gated_norm.gated_norm_forward` kernel applies the
forward-only BF16 GatedNorm formula:

```text
z = normed @ w_down.T
gate = sigmoid(silu(z) @ w_up.T)
output = normed * gate
```

This is intentionally inference-only. The Megatron-LM source commit also
contains backward/autograd and transformer training integration, but those paths
are not ported here.

The deployment contract for both G1 attention gating and GatedNorm is BF16.
Other input dtypes are rejected by the inference wrapper.

GatedNorm uses three BF16 execution paths:

- a Blackwell `sgl_kernel.gated_norm_cute_forward` tensor-op path when the
  local `sgl-kernel` wheel includes the registered op and the shape is in the
  kernel's supported range. The public name is historical: the current forward
  implementation is hand-written CUDA using `mma.sync`, `ldmatrix`, and
  `cp.async`, not a CuTe-generated source file;
- a fused Triton per-token fallback for decode and small batches;
- a torch/cuBLAS BF16 GEMM fallback for larger prefill batches.

The default dispatch thresholds were measured on B200 for hidden size 7168:
rank >= 64 uses GEMMs from 256 tokens, rank >= 32 from 512 tokens,
rank >= 8 from 2048 tokens, and rank 1 from 4096 tokens. Set
`SGLANG_GATED_NORM_TORCH_MM_MIN_TOKENS=-1` to keep small and medium shapes on
the fused-kernel path, `SGLANG_GATED_NORM_USE_TRITON=1` to bypass the CuTe op,
`SGLANG_GATED_NORM_DISABLE_CUTE=1` to use the non-CuTe dispatch, or
`SGLANG_GATED_NORM_TORCH_MM_R{1,8,32,64}_MIN_TOKENS` to tune rank-specific
thresholds during autoinfer runs.

Validation:

```bash
python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py
PYTHONPATH=python python scripts/playground/bench_gated_norm.py
```

Both kernels should be validated on Blackwell before using the full Dynamo
deployment profile.

## Model-side wiring

The model-side glue lives in `sglang.srt.models.deepseek_v2`:

- `_apply_g1_gate(attn_output, gate)` dispatches BF16 CUDA inputs to the
  output-only `sgl_kernel.g1_gate_forward_fused` op when available and falls
  back to `g1_gate_forward` or a sigmoid-times in fp32 otherwise.
- `_g1_gate_pre_hook(module, args, kwargs)` is registered as a
  `forward_pre_hook(with_kwargs=True)` on `o_proj`. A `Module` wrapper would
  re-key the loaded parameters under an `_inner.` prefix, so the hook approach
  is what keeps `o_proj.weight_packed` / `weight_scale` /
  `weight_global_scale` discoverable by the checkpoint loader. The hook
  consumes the per-call gate stashed on the owner as `_g1_pending_gate`,
  clears it (so a re-entrant call cannot re-apply a stale gate), and
  substitutes `attn_output * sigmoid(gate)` as `args[0]`.
- `DeepseekV2DecoderLayer._maybe_apply_gated_norm(x, w_down, w_up)` runs
  after each `prepare_attn` / `prepare_mlp` and dispatches to the BF16 fused
  kernel when `x.is_cuda and x.dtype == torch.bfloat16`; otherwise an
  fp32 reference matmul exactly mirrors
  `ai-blaise/Megatron-LM/megatron/core/fusions/gated_norm.py
  ._gated_norm_torch_mm_forward`.

Construction is gated by `attention_output_gate=True` /
`gated_norm=True` in `config.json`; the active Blaise checkpoint declares both
flags and `gated_norm_rank=16`. Developers can also force construction by setting
`SGLANG_DEEPSEEK_V2_ENABLE_GATED_ATTN=1` /
`SGLANG_DEEPSEEK_V2_ENABLE_GATED_NORM=1`. When the flags are off the
modules become `None` and the helpers short-circuit.

### Wiring tests

```bash
python -m pytest -q test/srt/models/test_deepseek_v2_g1_gated_norm_wiring.py
```

13 cases covering the BF16 fast paths, the fp32 fallback, the
`forward_pre_hook` substitution + clearing semantics, the
`Megatron-LM` reference for `_maybe_apply_gated_norm` at production
shapes (hidden 7168, rank 16), no-op semantics for `None` projections,
and the tuple-input passthrough used by the quantized
`LayerCommunicator` path. Tested on B200 inside the
`reap-nvfp4-fix13` runtime image (single GPU is enough — the helpers
process one decoder layer at a time).
