# REAP Gated Kernels

This repository carries two REAP-specific gated kernels for the
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
deployment lane:

- G1 attention output gating
- GatedNorm forward inference

Canonical B200 optimization results are tracked in:

- [G1 gate B200, 2026-05-17](kernel_results/g1_gate_b200_2026_05_17.md)
- [GatedNorm B200, 2026-05-17](kernel_results/gated_norm_b200_2026_05_17.md)

## G1 Attention Gate

The G1 gate applies:

```text
gate = sigmoid(linear_out)
output = attn_out * gate
```

The CUDA op is registered in `sgl-kernel` and supports BF16 tensors. Model-side
wiring uses the output-only `sgl_kernel.g1_gate_forward_fused` op when it is
available, falling back to `g1_gate_forward` or a PyTorch reference path when
needed.

## GatedNorm Forward

GatedNorm applies the inference-only BF16 forward formula:

```text
z = normed @ w_down.T
gate = sigmoid(silu(z) @ w_up.T)
output = normed * gate
```

The Megatron-LM source branch also contains backward/autograd integration, but
only the inference forward path is ported here.

GatedNorm dispatches through three BF16 paths:

- a Blackwell native op when `sgl-kernel` exposes
  `torch.ops.sgl_kernel.gated_norm_cute_forward`;
- a fused Triton per-token fallback for decode and small batches;
- a torch/cuBLAS BF16 GEMM fallback for larger prefill batches.

If the Python wrapper is newer than the installed `sgl-kernel` binary and the
native op is absent, runtime falls back instead of failing the request.

## Model-Side Wiring

The model glue lives in `sglang.srt.models.deepseek_v2`:

- `_apply_g1_gate(attn_output, gate)` dispatches BF16 CUDA inputs to the
  output-only fused op when available.
- `_g1_gate_pre_hook(module, args, kwargs)` applies the gate at `o_proj` without
  changing checkpoint parameter names.
- `DeepseekV2DecoderLayer._maybe_apply_gated_norm(x, w_down, w_up)` applies the
  fused BF16 GatedNorm path or the fp32 reference fallback.

Construction is enabled by `attention_output_gate=True` and `gated_norm=True`
in `config.json`. Developers can force construction with:

```bash
SGLANG_DEEPSEEK_V2_ENABLE_GATED_ATTN=1
SGLANG_DEEPSEEK_V2_ENABLE_GATED_NORM=1
```

## Dispatch Controls

```bash
SGLANG_GATED_NORM_DISABLE_CUTE=1
SGLANG_GATED_NORM_USE_TRITON=1
SGLANG_GATED_NORM_TORCH_MM_MIN_TOKENS=-1
SGLANG_GATED_NORM_TORCH_MM_R16_MIN_TOKENS=2048
```

The rank-specific threshold variables are intended for measurement and
autoinfer runs. Production defaults were measured on B200 for hidden size 7168.

## Verification

```bash
python -m pytest -q sgl-kernel/tests/test_g1_attention.py
python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py
python -m pytest -q test/srt/models/test_deepseek_v2_g1_gated_norm_wiring.py
PYTHONPATH=python python scripts/playground/bench_gated_norm.py
```
