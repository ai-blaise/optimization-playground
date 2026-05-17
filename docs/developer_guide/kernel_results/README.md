# Kernel Optimization Results

This directory is the single canonical location for B200 kernel optimization
results. Keep exactly one stable file per kernel. Do not add date-stamped,
branch-stamped, or run-stamped result files for later optimization passes;
append the brief outcome to the kernel's existing file instead.

## File Format

Each kernel file uses the same slim structure:

1. Scope
2. Current Result
3. Optimization History
4. Verification
5. Next

Use measured performance only. Do not claim a speedup for a deployability fix,
a correctness fix, or an unsupported shape.

## Kernels

| Kernel | File | Headline |
| --- | --- | --- |
| FlashSampling | [flashsampling.md](flashsampling.md) | Blackwell target route improves BS1/BS32 by 3.38%/2.76%; small-H and non-greedy H1 retunes add 1.5-4.0% in covered buckets. |
| G1 gate | [g1_gate.md](g1_gate.md) | Output-only fused dispatch improves the model-style allocation path by about 20% through 256 tokens. |
| GatedNorm | [gated_norm.md](gated_norm.md) | Low-rank dispatch threshold retunes improve important decode buckets by 1.15x-4.07x. |
| LayerSplit | [layersplit.md](layersplit.md) | 512 KiB staging threshold improves the common stage-copy matrix by 1.083x average, up to 1.150x. |
| WarpDecode | [warpdecode.md](warpdecode.md) | CuTe path is 2.56x over Triton at B1; final gate/up retune adds 1.004x-1.016x over direct CuTe control. |
| HIGGS | [higgs.md](higgs.md) | Auto split policy improves long-topk shapes up to 1.711x vs fixed-32; HIGGS uses 258 B/token/layer. |
| NVFP4 IndexCache dequant | [nvfp4_indexcache_dequant.md](nvfp4_indexcache_dequant.md) | Initial JIT path is 12.63x-17.58x over Torch expansion; restart retunes add up to 1.217x over fresh JIT incumbent. |
| NVFP4 HISA + IndexCache | [nvfp4_hisa_indexcache.md](nvfp4_hisa_indexcache.md) | Exact-pool fused mapping improves by 1.19x-1.50x; supported 64-head DeepGEMM path is 2.96x over ordinary NVFP4 IndexCache. |
