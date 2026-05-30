# SGL external kernel patches

This directory hosts patches and vendored sources for kernels we ship
outside the upstream package. The goal is to keep upstream packages
untouched so a future version-bump does not silently revert a fix, while
still landing the patch into the SGL deploy.

## flashinfer/

### patches/

Unified-diff patches against pinned upstream files. Each patch carries:
  * the upstream commit/tag it applies against (in the header)
  * the iter/ticket reference (in the body)
  * a "test: " line listing the bench / correctness test that gates it

Pin: flashinfer wheel at /usr/local/lib/python3.12/dist-packages/flashinfer
     (container-resolved path; user-mode path at
      /home/spencer/.local/lib/python3.11/site-packages/flashinfer)
     Embeds batched_gemm-b3c1646-c111d7c CUBIN export.

#### 0001-relax-bf16-act-e2m1-weight.patch (#15 iter7 PRIMARY)

Relaxes FP4BlockScaleLauncher::check_moe() to accept (Bfloat16 act +
E2m1 weight) tuple. Unlocks the 58 Bmm_Bfloat16_E2m1E2m1_*.cubin
variants for DSv3.2-REAP NVFP4 MoE. See
notes/nvfp4_moe_iter7_recon.md.

  test: test/srt/test_nvfp4_moe_bmm_bf16_act_bench.py (forthcoming, Stage D)
  size: +17 -6 lines
  surface: check_moe() in FP4BlockScaleLauncher, scoped to dtype gate

#### 0002-python-mirror.patch (#15 iter7 PRIMARY companion)

Mirrors the C++ relaxation in flashinfer/fused_moe/core.py
is_trtllm_moe_supported() so the python guard does not reject the same
combo at dispatch time.

  test: same as 0001
  size: +8 -5 lines
  surface: is_trtllm_moe_supported(), scoped to (E2m1 weights) branch

### csrc/

Vendored source copies (verbatim from upstream pinned version) that we
either:
  * apply patches to (build glue rebuilds the .so for the patched copy)
  * include for IDE navigation when patches are non-trivial

For the iter7 patch series above, the corresponding vendored sources
are pulled at build time (see flashinfer/__init__.py build glue).
The patches in patches/ are the source of truth for the diff; the
vendored .cu files are derived artifacts.

### __init__.py

Build glue. Applies the patches/ series to a working copy of the
upstream sources, JIT-compiles them via the same TVM-FFI toolchain
flashinfer itself uses, and registers the patched symbols under
sglang's external_kernels namespace.

Loading order in compressed_tensors_w4a4_nvfp4_moe.py:

  try:
      from sglang.srt.external_kernels.flashinfer import (
          trtllm_fp4_block_scale_moe_bf16_act,
      )
      _HAS_BF16_ACT_BMM = True
  except ImportError:
      _HAS_BF16_ACT_BMM = False

The env-gate SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE must be set to opt in
to the patched path. Default-off until iter8 wire bench resolves
bit-exactness.

## Status

iter7 (this commit): Stage A — patches + vendored README. Build glue +
runtime wire follow in iter8. Microbench is forthcoming as
test/srt/test_nvfp4_moe_bmm_bf16_act_bench.py once Stage B+C land.
