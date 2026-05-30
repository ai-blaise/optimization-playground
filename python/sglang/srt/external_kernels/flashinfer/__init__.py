"""Vendored flashinfer kernels with SGL-side patches.

iter7 (this checkpoint): scaffold only. The patches in patches/ are
not yet applied to a vendored copy; the symbol
``trtllm_fp4_block_scale_moe_bf16_act`` is **NOT** exported by this
module yet. See ``notes/nvfp4_moe_iter7_recon.md`` for the 4-stage
implementation plan (Stages B-E follow in iter8+).

Future loading order (iter8+):

  1. At import, check that flashinfer is installed and the upstream
     wheel version matches the pin (``flashinfer == X.Y.Z``).
  2. Stage the vendored .cu files into a per-process temp dir.
  3. Apply patches/*.patch via ``git apply --3way`` (no-op if patch
     already applied to current vendored copy).
  4. Compile via the same TVM-FFI toolchain flashinfer itself uses
     (``flashinfer.jit.compile_aot``).
  5. Register the exported ``trtllm_fp4_block_scale_moe_bf16_act``
     symbol on the global op table.

Until iter8 wires Stages B-D, importing this module is a no-op and
``HAS_BF16_ACT_BMM`` stays False so consumers fall through to the
iter1-3 + iter4 PRIMARY production path.
"""
from __future__ import annotations

# The Stage B vendored launcher will replace this False with True once
# the patched .so is loadable. Consumers in
# compressed_tensors_w4a4_nvfp4_moe.py read this flag at startup.
HAS_BF16_ACT_BMM: bool = False

__all__ = ["HAS_BF16_ACT_BMM"]
