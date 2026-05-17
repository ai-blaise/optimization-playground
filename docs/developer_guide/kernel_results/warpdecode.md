# WarpDecode

## Scope

WarpDecode MoE kernels for DeepSeek-style decode on Blackwell. Production
dimensions are hidden size 7168, intermediate size 2048, top-k 8, and 128
experts.

## Current Result

| Comparison | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| Original Triton fallback tile chain, B1 harness | 1567.8 us | 322.6 us | 4.86x |
| Production B1 Triton fallback | 307.9 us | 120.2 us CuTe path | 2.56x |
| Final gate/up retune over direct CuTe control | 114.8-6156.1 us | 113.6-6095.8 us | 1.004x-1.016x |

Final B1 Nsight measured gate/up `<4,1024,4>` at 69.3 us average and down
`<8,2048,8>` at 38.5 us average.

## Optimization History

- Promoted the CuTe WarpDecode path over the Triton fallback for the supported
  production shape.
- Accepted the 4-warp gate/up retune because it beat direct CuTe control across
  batches 1, 4, 8, 16, 32, and 64.
- Rejected later tile and launch probes that tied or regressed.

## Verification

- WarpDecode focused tests passed in the final integration bundle.
- Quantization dispatch was covered in the GPU1 integration group.
- Final GPU1 bundle: 64 passed.

## Next

Prioritize tensor-op work in any W4A4/NVFP4 path still relying on scalar FP32
inner loops.
