# LayerSplit

## Scope

LayerSplit stage-copy and fused materialize kernels on B200. This is raw
device-to-device memory movement, not a tensor-core workload.

## Current Result

| Case | Baseline | Current | Speedup |
| --- | ---: | ---: | ---: |
| <=116 KiB stage-copy payloads | `Tensor.copy_` | custom stage kernel | 1.058x average |
| 128 KiB threshold cells | 0.800x vs direct `Tensor.copy_` | 1.048x vs direct `Tensor.copy_` | 1.31x relative |
| 128 KiB `cute_ms` | 3.681 us | 2.849 us | 1.292x |
| 512 KiB threshold common matrix | previous threshold | final threshold | 1.083x average, up to 1.150x |

The direct-prefix `cudaMemcpyAsync` delegated path reduced the >128 KiB loss
from about 0.80x to 0.89x-0.93x versus direct Python copy.

## Optimization History

- Packaged the existing LayerSplit stage kernel in `sgl-kernel`.
- Raised the small-copy threshold to 128 KiB, then to 512 KiB after same-GPU
  validation.
- Used direct-prefix `cudaMemcpyAsync` in the delegated contiguous path.
- Rejected Python helper dispatch, dynamic CTA count, larger small-copy
  thresholds, fused materialize CTA-threshold sweeps, and unroll4.

## Verification

- Every benchmark cell checked active-prefix equality.
- Padding verification checked inactive rows stayed zero.
- CPU/config integration group including LayerSplit passed 64 tests.

## Next

Future gains need producer/consumer fusion outside this copy kernel; local
threshold and loop-shape probes are saturated.
