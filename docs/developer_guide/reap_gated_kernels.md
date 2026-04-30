# REAP Gated Kernels

This repository carries two REAP-specific gated kernels for the
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1` deployment lane.

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
rank >= 64 uses GEMMs from 256 tokens, rank >= 32 from 512 tokens, and
rank >= 8 from 2048 tokens. Set
`SGLANG_GATED_NORM_TORCH_MM_MIN_TOKENS=-1` to force the Triton path, or use
`SGLANG_GATED_NORM_TORCH_MM_R{8,32,64}_MIN_TOKENS` to tune rank-specific
thresholds during autoinfer runs.

Validation:

```bash
python -m pytest -q python/sglang/jit_kernel/tests/test_gated_norm.py
PYTHONPATH=python python scripts/playground/bench_gated_norm.py
```

Both kernels should be validated on Blackwell before using the full Dynamo
deployment profile.
