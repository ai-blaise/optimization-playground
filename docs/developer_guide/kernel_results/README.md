# Kernel Optimization Results

This directory is the canonical audit location for B200 kernel optimization
results. Feature docs should link here instead of embedding round-by-round
performance logs.

## Required Structure

Each result file uses these sections:

1. Scope
2. Environment
3. Current Result
4. Accepted Changes
5. Rejected Candidates
6. Verification
7. Limits And Next Work

Use measured performance only. Do not claim a speedup for a deployability fix,
a correctness fix, or an unsupported shape.

## Result Index

| Lane | Canonical Result | Status | Headline |
| --- | --- | --- | --- |
| FlashSampling | [FlashSampling B200](flashsampling_b200_2026_05_17.md) | Accepted | Blackwell target route improves BS1/BS32 by 3.38%/2.76%; small-H and non-greedy H1 retunes add 1.5-4.0% in covered buckets. |
| G1 gate | [G1 Gate B200](g1_gate_b200_2026_05_17.md) | Accepted | Output-only fused dispatch improves the model-style allocation path by about 20% through 256 tokens. |
| GatedNorm | [GatedNorm B200](gated_norm_b200_2026_05_17.md) | Accepted | Low-rank dispatch threshold retunes improve important decode buckets by 1.15x-4.07x. |
| LayerSplit | [LayerSplit B200](layersplit_b200_2026_05_17.md) | Accepted | 512 KiB staging threshold improves the common stage-copy matrix by 1.083x average, up to 1.150x. |
| WarpDecode | [WarpDecode B200](warpdecode_b200_2026_05_17.md) | Accepted | CuTe path is 2.56x over Triton at B1; final gate/up retune adds 1.004x-1.016x over direct CuTe control. |
| HIGGS | [HIGGS B200](higgs_b200_2026_05_17.md) | Accepted with next-design work | Auto split policy improves long-topk shapes up to 1.711x vs fixed-32; HIGGS uses 258 B/token/layer. |
| NVFP4 IndexCache dequant | [NVFP4 IndexCache Dequant B200](nvfp4_indexcache_dequant_b200_2026_05_17.md) | Accepted | Initial JIT path is 12.63x-17.58x over Torch expansion; restart retunes add up to 1.217x over fresh JIT incumbent. |
| NVFP4 HISA + IndexCache | [NVFP4 HISA + IndexCache B200](nvfp4_hisa_indexcache_b200_2026_05_17.md) | Accepted with guarded fallback | Exact-pool fused mapping improves by 1.19x-1.50x; supported 64-head DeepGEMM path is 2.96x over ordinary NVFP4 IndexCache. |

## Historical Paths

The older long-form log paths are retained as small redirect pages to avoid
breaking external links. New measurements belong in the canonical files above.
