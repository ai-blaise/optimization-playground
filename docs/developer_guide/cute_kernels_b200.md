# B200 Kernel Results

This page is the compact B200 kernel index. Detailed per-kernel audit records
live in [Kernel Optimization Results](kernel_results/README.md).

The active deployment lane is
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1` on
NVIDIA Blackwell B200 (`sm100`). CuTe kernels must use CZS as the CuTe compiler
path. Existing non-CuTe kernels remain valid when they are already production
wired and verified.

## Canonical Files

| Kernel | File | Type | Headline |
| --- | --- | --- | --- |
| FlashSampling | [flashsampling.md](kernel_results/flashsampling.md) | Triton target provider | Blackwell target route improves BS1/BS32 by 3.38%/2.76%; small-H retunes add 1.5-4.0%. |
| G1 gate | [g1_gate.md](kernel_results/g1_gate.md) | CuTe/native CUDA | Output-only fused dispatch improves model-style allocation path by about 20% through 256 tokens. |
| GatedNorm | [gated_norm.md](kernel_results/gated_norm.md) | Native CUDA/Triton/cuBLAS dispatch | Low-rank threshold retunes improve important decode buckets by 1.15x-4.07x. |
| LayerSplit | [layersplit.md](kernel_results/layersplit.md) | Native CUDA copy kernels | 512 KiB staging threshold improves the common stage-copy matrix by 1.083x average, up to 1.150x. |
| WarpDecode | [warpdecode.md](kernel_results/warpdecode.md) | CuTe/native CUDA | CuTe path is 2.56x over Triton at B1; final gate/up retune adds 1.004x-1.016x. |
| HIGGS | [higgs.md](kernel_results/higgs.md) | Direct CUDA JIT | Auto split policy improves long-topk shapes up to 1.711x and uses 258 B/token/layer. |
| NVFP4 IndexCache dequant | [nvfp4_indexcache_dequant.md](kernel_results/nvfp4_indexcache_dequant.md) | Direct CUDA JIT | Initial JIT path is 12.63x-17.58x over Torch expansion; restart retunes add up to 1.217x. |
| NVFP4 HISA + IndexCache | [nvfp4_hisa_indexcache.md](kernel_results/nvfp4_hisa_indexcache.md) | Direct CUDA JIT + DeepGEMM | Exact-pool mapping improves by 1.19x-1.50x; supported 64-head DeepGEMM path is 2.96x. |

## Documentation Rule

Keep one stable file per kernel in `kernel_results/`. Later optimization passes
append a short entry to that kernel's `Optimization History` section; they do
not create new per-date or per-branch result files.
