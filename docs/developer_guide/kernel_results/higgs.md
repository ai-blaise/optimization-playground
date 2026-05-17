# HIGGS

## Scope

HIGGS dense MLA 2-bit KV cache and fused dense MLA decode for the active
DeepSeek-V3.2-REAP lane. HIGGS is the dense KV path replacing TurboQuant.

## Current Result

| Candidate | Bytes/token/layer | Store | Selected Recover | Latent Cosine | Rope Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| NVFP4 dense MLA | 324 | 0.016355 ms | 0.029978 ms | 0.995470 | false |
| TurboQuant dense MLA 2.5-bit | 274 | 0.007296 ms | 0.006256 ms | 0.950731 | true |
| HIGGS dense MLA 2-bit | 258 | 0.008886 ms | 0.005587 ms | 0.944873 | true |

| Decode Shape | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| r4 h8 topk1024, fixed16 to fixed32 | 0.057862 ms | 0.041285 ms | 1.40x |
| r1 h8 topk2048, fixed32 to auto | 0.057498 ms | 0.043172 ms | 1.332x |
| r1 h8 topk4096, fixed32 to auto | 0.098443 ms | 0.057543 ms | 1.711x |
| r8 h8 topk4096, fixed32 to auto | 0.200157 ms | 0.171649 ms | 1.166x |

Final `rows=4`, `heads=8`, `topk=2048`, auto split 56 decode time was
0.0621616 ms with minimum cosine 0.9999998211860657.

## Optimization History

- Raised B200 default split count from 16 to 32.
- Added pair-lane packed-byte broadcast and read-only packed-byte loads.
- Added measured auto split choices across 32, 36, 48, 56, 64, 72, 80, 96,
  and 128 splits.
- Rejected global split-64, blanket fractional policies, and further
  split-neighborhood probes that did not survive pool validation.

## Verification

- HIGGS integration and quantization dispatch tests passed in lane validation.
- Final GPU0 integration bundle covering NVFP4, HISA, and HIGGS passed 27 tests
  with 1 skip.

## Next

The packed 258 B slot layout is scalar codec work. Further material speedup
needs a new CuTe/CZS tensor-op or staging design rather than split retuning.
