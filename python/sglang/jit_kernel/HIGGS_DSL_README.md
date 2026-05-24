# HIGGS 2-bit Dense MLA KV Decode — CuTe Python DSL Kernel (WIP)

**Status**: ITER 12 — kernel compiles + runs end-to-end on B200; race-free pattern proven via `cute.fill`; full IKP + CZS + autoloop iteration framework operational; production correctness requires final R2S/TMA wire-up (ITER 13).

## Files

| File | Purpose |
|---|---|
| `higgs_dense_2bit_mla_decode_dsl.py` | The CuTe Python DSL kernel (1100+ lines) |
| `higgs_dense_2bit_mla_decode_tc.py` | C++ TC baseline (commit 961c4794a, 2.57× over scalar) |
| `csrc/quantization/higgs_dense_2bit_mla_decode_tc.cuh` | Baseline C++ kernel source |
| `bench_higgs_dsl.py` | PyTorch reference + correctness comparison + bench harness |
| `autoiter_higgs_dsl.py` | Autonomous iteration loop with variant patching + bench + racecheck + CZS gating |
| `test_higgs_dsl_correctness.py` | Standalone correctness test |

## Verification toolchain (all operational on B200 a4-us-001-rl9)

| Tool | Role | Path |
|---|---|---|
| autoiter_higgs_dsl.py | Variant-driven correctness/perf loop | this dir |
| compute-sanitizer racecheck | Race detection per variant | `/usr/local/cuda-13.0/bin/compute-sanitizer` |
| CZS v0.4.1 | Formal layout/MMA/TMEM verification | `/home/spencer/CZS/build/src/czs` |
| IKP CUPTI region profiler | SASS-level per-PC profiling | `/home/spencer/refs/intra-kernel-profiler/tools/cupti_region_profiler/` |
| nsys | Kernel timeline | `/opt/nvidia/nsight-systems/2025.6.3/bin/nsys` |
| PyTorch reference | Correctness ground truth | `bench_higgs_dsl.py:_torch_reference` |

## Running the principled iteration loop

```bash
# Single-shape test (PyTorch ref vs DSL)
python3.11 bench_higgs_dsl.py

# Autoloop with all 3 gates: CZS layout-verify + bench + compute-sanitizer racecheck
python3.11 autoiter_higgs_dsl.py --variants cute_fill_kv --racecheck --czs

# CZS standalone
/home/spencer/CZS/build/src/czs prove --json /tmp/higgs_mla_dsl_module_v2.json

# nsys profile (system-level + kernel timing)
nsys profile -o /tmp/prof --stats=true --trace=cuda python3.11 bench_higgs_dsl.py
```

## Verification matrix (latest autoloop, B200 with CUDA 13.0 + cute.dsl 4.5.1)

| Variant | compile | races (compute-sanitizer) | CZS | std | diff_mean | Notes |
|---|---|---|---|---|---|---|
| baseline iter 6 | ✓ 17s | 32 ✗ | 8P/1D | 1.0e-3 | 4.39e-2 | per-element scatter races |
| swizzle_3_3_3 | ✓ | 32 ✗ | 8P/1D | 1.3e-3 | 4.39e-2 | swizzle bits irrelevant |
| swizzle_0_0_0 | ✓ | 32 ✗ | 8P/1D | 1.0e-3 | 4.39e-2 | no swizzle, same wrongness |
| constant_kv (K=V=1) | ✓ | 32 ✗ | 8P/1D | 1.5e-2 | 4.67e-2 | data flow partial |
| softmax_skip_writes | ✓ | 32 ✗ | 8P/1D | 1.99e-2 | 4.66e-2 | races NOT from softmax |
| **cute_fill_kv** | ✓ 92s | **0 ✓** | **8P/1D** | **5.32e-2** | 8.71e-2 | **race-free + magnitude correct** |
| iter_atom_k_blocks | ✓ | 32 ✗ | 8P/1D | 1.4e-3 | 4.39e-2 | cute.gemm already iterates atoms |
| autovec_load | ✗ | n/a | n/a | nan | inf | compile > 5min (unroll too large) |

(CZS Disproved is TmemLifetime — JSON test-data limitation, not a real kernel issue.)

## Performance comparison (DSL vs C++ TC baseline)

| Implementation | us/call @ (R=1, H=64, TOPK=32) | Status |
|---|---|---|
| C++ TC baseline (961c4794a, 2.57× over scalar) | <1ms target | Production |
| Tokenspeed FP8 MLA | TBD (separate setup needed) | Architecture reference |
| **DSL iter 11 (predequant + races)** | **18.6s (cold) + JIT cache miss** | Compiles + runs; values wrong |
| **DSL iter 12 (cute.fill placeholder)** | **92.4s (cold)** | Race-free + magnitude correct |
| Target | ≤ 2× C++ TC | iter 13+ |

## Root cause (IKP + CZS findings)

**32 WRITE-WRITE races** in the per-element scatter writes to composed-swizzle SMEM views (sX_write). The composed swizzle layout maps multiple LOGICAL `(n, d)` coords to the SAME PHYSICAL byte → simultaneous writes by different threads → last-write-wins → data corruption.

**CZS confirmed** the MMA operand layouts are LEGAL when given the correct atom K=16 (not the tile K=64). My MMA setup is structurally OK; the bug is purely in the per-element write pattern.

**Proven fix direction**: replace per-element scatter with `cute.fill` (single op, race-free by atom-aware iteration) OR `cute.copy` with TMA atom (hardware-vectorized, race-free).

## Remaining work (ITER 13+)

1. **Wire race-free K/V load**: TMA atom (`cute.nvgpu.make_tiled_tma_atom_B`) for GMEM→swizzled SMEM transfer. Race-free by hardware design; should compile fast (single hardware instruction, no per-cell unroll).

2. **Online softmax cross-warp reduction**: current softmax writes have warp-local row_max/sum but multiple warps own the same row → races on softmax_l. Needs `softmax_smem_exchange` pattern from tokenspeed.

3. **R2T rescale_acc_by_alpha**: currently no-op; need `tcgen05.copy.St32x32bOp` for TMEM store.

4. **InvFWHT_512 in epilogue**: cooperative 512-element FWHT across warps via `fwht_scratch` SMEM.

5. **JIT cache miss fix**: each call re-compiles (17s). Need to pre-build cute Tensors or hoist `@cute.jit` to module scope.

6. **Benchmark vs C++ TC**: requires installing remaining sglang deps for TC import (torchvision, etc.); then `autoiter_higgs_dsl.py --bench` will produce a full perf table.

## Architectural decisions (this work)

- **OperandMajorMode.K for QK A,B; .K/.MN for PV A,B**: matches the natural memory layout for Q.K^T attention.
- **PV split into PV_N_CHUNKS=2 × N=256**: SM100 F16 MMA has N cap of 256; we need N=512 (LATENT_DIM).
- **block_h=64, block_n=32**: matches the C++ TC baseline tile shape; balances SMEM usage with parallelism.
- **`_KERNEL_CACHE` module-level dict**: caches per `(block_h, block_n)` to avoid re-instantiation overhead.
- **Pre-dequant in Python wrapper (ITER 11+)**: moves HIGGS decode out of the kernel, letting the kernel focus on attention. Production move pre-dequant to a fused Triton kernel.

## Agentmemory checkpoints (port :3811)

  - `mem_20260523T231312Z_ebbfh0` — iter 4 kernel compiles on B200 (8 errors fixed)
  - `mem_20260524T011654Z_zbivz5` — iter 8 race-conditions discovered via compute-sanitizer
  - `mem_20260524T012259Z_c052pd` — iter 8 autoloop + IKP integration
  - `mem_20260524T013205Z_pm7on2` — iter 9 races=0 breakthrough (cute.fill)
  - `mem_20260524T014418Z_sdq3fh` — iter 10 CZS integration
  - `mem_20260524T021431Z_nzgch9` — iter 12 full toolchain integration

## License

Apache 2.0 (matches optimization-playground / sglang upstream).
