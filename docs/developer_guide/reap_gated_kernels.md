# REAP Gated Kernels

This repository carries two REAP-specific gated kernels for the
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1` deployment lane.

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

GatedNorm uses two BF16 execution paths:

- a fused Triton per-token path for decode and small batches, where avoiding
  extra launches and intermediate tensors is fastest;
- a torch/cuBLAS BF16 GEMM path for larger prefill batches, where tensor cores
  are faster than scalar per-token reductions.

The default dispatch thresholds were measured on B200 for hidden size 7168:
rank >= 64 uses GEMMs from 256 tokens, rank >= 32 from 512 tokens,
rank >= 8 from 2048 tokens, and rank 1 from 4096 tokens. Set
`SGLANG_GATED_NORM_TORCH_MM_MIN_TOKENS=-1` to force the Triton path, or use
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
  `sgl_kernel.g1_gate_forward` op and falls back to a sigmoid-times in fp32
  otherwise; numerics match `Megatron-LM`'s `Attention._apply_output_gate`.
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
`gated_norm=True` in `config.json`, or by setting
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
