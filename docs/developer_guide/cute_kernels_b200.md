# B200 Kernel Results

This page is the compact B200 kernel index. Detailed per-kernel audit records
live in [Kernel Optimization Results](kernel_results/README.md).

The active deployment lane is
`BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1` on
NVIDIA Blackwell B200 (`sm100`). CuTe kernels must use CZS as the CuTe compiler
path. Existing non-CuTe kernels remain valid when they are already production
wired and verified.

## Canonical Results

| Lane | Canonical Result | Kernel Type | Headline |
| --- | --- | --- | --- |
| FlashSampling | [FlashSampling B200](kernel_results/flashsampling_b200_2026_05_17.md) | Triton target provider | Blackwell target route improves BS1/BS32 by 3.38%/2.76%; small-H and non-greedy H1 retunes add 1.5-4.0% in covered buckets. |
| G1 gate | [G1 Gate B200](kernel_results/g1_gate_b200_2026_05_17.md) | CuTe/native CUDA | Output-only fused dispatch improves model-style allocation path by about 20% through 256 tokens. |
| GatedNorm | [GatedNorm B200](kernel_results/gated_norm_b200_2026_05_17.md) | Native CUDA/Triton/cuBLAS dispatch | Low-rank threshold retunes improve important decode buckets by 1.15x-4.07x. |
| LayerSplit | [LayerSplit B200](kernel_results/layersplit_b200_2026_05_17.md) | Native CUDA copy kernels | 512 KiB staging threshold improves the common stage-copy matrix by 1.083x average, up to 1.150x. |
| WarpDecode | [WarpDecode B200](kernel_results/warpdecode_b200_2026_05_17.md) | CuTe/native CUDA | CuTe path is 2.56x over Triton at B1; final gate/up 4-warp retune adds 1.004x-1.016x over direct CuTe control. |
| HIGGS | [HIGGS B200](kernel_results/higgs_b200_2026_05_17.md) | Direct CUDA JIT | Auto split policy improves long-topk shapes up to 1.711x vs fixed-32 and uses 258 B/token/layer. |
| NVFP4 IndexCache dequant | [NVFP4 IndexCache Dequant B200](kernel_results/nvfp4_indexcache_dequant_b200_2026_05_17.md) | Direct CUDA JIT | Initial JIT path is 12.63x-17.58x over Torch expansion; restart retunes add up to 1.217x over fresh JIT incumbent. |
| NVFP4 HISA + IndexCache | [NVFP4 HISA + IndexCache B200](kernel_results/nvfp4_hisa_indexcache_b200_2026_05_17.md) | Direct CUDA JIT + DeepGEMM | Exact-pool fused mapping improves by 1.19x-1.50x; supported 64-head DeepGEMM path is 2.96x over ordinary NVFP4 IndexCache. |

## Documentation Rules

Each canonical result file uses the same structure:

1. Scope
2. Environment
3. Current Result
4. Accepted Changes
5. Rejected Candidates
6. Verification
7. Limits And Next Work

Feature documentation should link to the canonical result files instead of
embedding full optimization logs.
