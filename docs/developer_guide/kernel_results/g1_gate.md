# G1 Gate

## Scope

G1 attention output gating for DeepSeek-V3.2-REAP. The kernel computes
`attn_out * sigmoid(gate)` before `o_proj`.

## Current Result

| Tokens | Result |
| ---: | --- |
| 1 / 16 / 64 / 256 | Output-only fused path is 19.9%, 20.1%, 20.7%, and 20.6% faster than the model-style full path. |
| 1024 | Output-only fused path is 0.1% faster, effectively tied. |

IKP measured the 256-token full path at 3.152 us mean and the output-only
`g1_gate_cute_fused_kernel_v2` path at 2.894 us mean.

## Optimization History

- Added SM100 packaging/runtime dispatch so B200 loads an `sm_100` common-ops
  artifact.
- Added and wired `g1_gate_forward_fused` for the output-only model hook.
- Rejected launch-threshold, geometry, cache-hint, local FP4-fusion, and
  rational-sigmoid candidates because they tied, regressed, or failed accuracy.

## Verification

- Focused G1 test: 10 passed.
- Correctness matched the full kernel through 64 tokens and stayed within the
  existing BF16 tolerance at larger token counts.
- Final GPU1 integration bundle passed 64 focused tests.

## Next

The remaining local kernel is launch-floor limited. The next material candidate
is cross-operator fusion into FP4 activation quantization before `o_proj`.
