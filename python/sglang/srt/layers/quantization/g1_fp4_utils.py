# SPDX-License-Identifier: Apache-2.0
"""CuTe-DSL G1 + NVFP4 activation quantization helper.

This mirrors the FlashInfer non-TMA swizzled NVFP4 quantizer for BF16
row-major inputs and replaces the input load with:

    bf16_round(attn * sigmoid_approx(gate)) -> NVFP4 pack + scale

The explicit BF16 round-trip is required to match current SGLang semantics:
today `g1_gate_forward_fused` writes a BF16 tensor before `fp4_quantize`.
"""

from __future__ import annotations

import functools
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from flashinfer.cute_dsl.fp4_common import (
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    nvfp4_compute_output_scale,
    quantize_and_pack_16,
    rcp_approx_ftz,
    st_global_u64,
)
from flashinfer.cute_dsl.utils import get_num_sm
from flashinfer.quantization.kernels.nvfp4_quantize import (
    SF_LAYOUT_128x4,
    SF_LAYOUT_8x4,
)
from flashinfer.quantization.quantization_cute_dsl_utils import (
    NVFP4_SF_VEC_SIZE,
    ROW_TILE_SIZE,
    bfloat2_to_float2_scaled,
    compute_sf_index_swizzled_128x4_gpu,
    compute_sf_index_swizzled_8x4_gpu,
)


_BLOCKS_PER_SM = 4
_MAX_THREADS_PER_BLOCK = 1024
_MIN_THREADS = 128
_MAX_THREADS = 512


def _compute_optimal_threads(k: int) -> int:
    threads_per_row = k // NVFP4_SF_VEC_SIZE
    if threads_per_row > _MAX_THREADS:
        return _MAX_THREADS
    largest = (_MAX_THREADS // threads_per_row) * threads_per_row
    if largest >= _MIN_THREADS:
        return largest
    candidate = threads_per_row
    while candidate < _MIN_THREADS:
        candidate += threads_per_row
    if candidate <= _MAX_THREADS:
        return candidate
    return _MAX_THREADS


@dsl_user_op
def ex2_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16_round_to_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """Round f32 to bf16 and return the bf16 value as f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
              .reg .b16 b;
              .reg .b32 u;
              cvt.rn.bf16.f32 b, $1;
              cvt.u32.u16 u, b;
              shl.b32 u, u, 16;
              mov.b32 $0, u;
            }
            """,
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def max_abs_16(values: cute.Tensor) -> Float32:
    result = fabs_f32(values[0])
    for i in cutlass.range_constexpr(1, 16):
        result = fmax_f32(result, fabs_f32(values[i]))
    return result


@cute.jit
def g1_sigmoid_mul_16(attn_pairs: cute.Tensor, gate_pairs: cute.Tensor) -> cute.Tensor:
    out = cute.make_rmem_tensor((16,), Float32)
    log2e = Float32(1.4426950408889634)
    one = Float32(1.0)
    for i in cutlass.range_constexpr(8):
        att0, att1 = bfloat2_to_float2_scaled(attn_pairs[i], one)
        gate0, gate1 = bfloat2_to_float2_scaled(gate_pairs[i], one)
        sig0 = rcp_approx_ftz(one + ex2_approx_ftz(-gate0 * log2e))
        sig1 = rcp_approx_ftz(one + ex2_approx_ftz(-gate1 * log2e))
        out[i * 2] = bf16_round_to_f32(att0 * sig0)
        out[i * 2 + 1] = bf16_round_to_f32(att1 * sig1)
    return out


@cute.jit
def process_g1_nvfp4_block_bfloat(
    attn_row, gate_row, elem_base: Int32, global_scale: Float32
) -> tuple:
    attn_pairs = cute.make_rmem_tensor((8,), Uint32)
    gate_pairs = cute.make_rmem_tensor((8,), Uint32)

    attn_ptr0 = get_ptr_as_int64(attn_row, elem_base)
    attn_ptr1 = get_ptr_as_int64(attn_row, elem_base + Int32(8))
    gate_ptr0 = get_ptr_as_int64(gate_row, elem_base)
    gate_ptr1 = get_ptr_as_int64(gate_row, elem_base + Int32(8))

    attn_pairs[0], attn_pairs[1], attn_pairs[2], attn_pairs[3] = ld_global_v4_u32(attn_ptr0)
    attn_pairs[4], attn_pairs[5], attn_pairs[6], attn_pairs[7] = ld_global_v4_u32(attn_ptr1)
    gate_pairs[0], gate_pairs[1], gate_pairs[2], gate_pairs[3] = ld_global_v4_u32(gate_ptr0)
    gate_pairs[4], gate_pairs[5], gate_pairs[6], gate_pairs[7] = ld_global_v4_u32(gate_ptr1)

    values = g1_sigmoid_mul_16(attn_pairs, gate_pairs)
    block_max = max_abs_16(values)

    fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)
    packed64 = quantize_and_pack_16(values, output_scale)
    return scale_fp8, packed64


class G1NVFP4QuantizeSwizzledKernel:
    def __init__(self, k: int, sf_layout: int = SF_LAYOUT_128x4, enable_pdl: bool = False):
        self.k = k
        self.enable_pdl = enable_pdl
        self.sf_layout = sf_layout
        self.sf_is_128x4 = sf_layout == SF_LAYOUT_128x4
        assert k % NVFP4_SF_VEC_SIZE == 0
        self.num_sf_blocks_per_row = k // NVFP4_SF_VEC_SIZE
        self.padded_sf_cols = ((self.num_sf_blocks_per_row + 3) // 4) * 4
        self.num_threads = _compute_optimal_threads(k)
        self.threads_per_row = self.num_sf_blocks_per_row
        if self.threads_per_row <= self.num_threads:
            self.rows_per_block = self.num_threads // self.threads_per_row
            self.needs_col_loop = False
        else:
            self.rows_per_block = 1
            self.needs_col_loop = True

    @cute.jit
    def _compute_sf_offset(
        self, row_idx: Int32, col_idx: Int32, padded_cols: Int32
    ) -> Int32:
        if cutlass.const_expr(self.sf_is_128x4):
            return compute_sf_index_swizzled_128x4_gpu(row_idx, col_idx, padded_cols)
        return compute_sf_index_swizzled_8x4_gpu(row_idx, col_idx, padded_cols)

    @cute.jit
    def __call__(
        self,
        mAttn: cute.Tensor,
        mGate: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        num_blocks: Int32,
        mGlobalScale: cute.Tensor,
        stream,
    ):
        self.kernel(mAttn, mGate, mOutput, mScales, M, padded_M, mGlobalScale).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            max_number_threads=[_MAX_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mAttn: cute.Tensor,
        mGate: cute.Tensor,
        mOutput: cute.Tensor,
        mScales: cute.Tensor,
        M: Int32,
        padded_M: Int32,
        mGlobalScale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        global_scale = Float32(mGlobalScale[Int32(0)])
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        padded_sf_cols = self.padded_sf_cols
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block

        if cutlass.const_expr(self.needs_col_loop):
            num_threads = self.num_threads
            row_idx = bidx
            while row_idx < padded_M:
                if row_idx >= M:
                    sf_col_idx = tidx
                    while sf_col_idx < padded_sf_cols:
                        sf_offset = self._compute_sf_offset(row_idx, sf_col_idx, padded_sf_cols)
                        mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + num_threads
                else:
                    sf_col_idx = tidx
                    while sf_col_idx < num_sf_blocks_per_row:
                        elem_base = sf_col_idx * NVFP4_SF_VEC_SIZE
                        scale_fp8, packed64 = process_g1_nvfp4_block_bfloat(
                            mAttn[row_idx, None],
                            mGate[row_idx, None],
                            elem_base,
                            global_scale,
                        )
                        sf_offset = self._compute_sf_offset(row_idx, sf_col_idx, padded_sf_cols)
                        mScales[sf_offset] = scale_fp8
                        out_base = sf_col_idx * (NVFP4_SF_VEC_SIZE // 2)
                        out_ptr = get_ptr_as_int64(mOutput[row_idx, None], out_base)
                        st_global_u64(out_ptr, packed64)
                        sf_col_idx = sf_col_idx + num_threads

                    sf_col_idx = num_sf_blocks_per_row + tidx
                    while sf_col_idx < padded_sf_cols:
                        sf_offset = self._compute_sf_offset(row_idx, sf_col_idx, padded_sf_cols)
                        mScales[sf_offset] = Uint8(0)
                        sf_col_idx = sf_col_idx + num_threads
                row_idx = row_idx + grid_dim_x
        else:
            row_in_block = tidx // threads_per_row
            sf_idx_in_row = tidx % threads_per_row
            row_batch_idx = bidx
            row_idx = row_batch_idx * rows_per_block + row_in_block
            while row_batch_idx * rows_per_block < padded_M:
                if row_idx < padded_M:
                    if row_idx >= M:
                        local_sf_idx = sf_idx_in_row
                        while local_sf_idx < padded_sf_cols:
                            sf_offset = self._compute_sf_offset(row_idx, local_sf_idx, padded_sf_cols)
                            mScales[sf_offset] = Uint8(0)
                            local_sf_idx = local_sf_idx + threads_per_row
                    else:
                        if sf_idx_in_row < num_sf_blocks_per_row:
                            elem_base = sf_idx_in_row * NVFP4_SF_VEC_SIZE
                            scale_fp8, packed64 = process_g1_nvfp4_block_bfloat(
                                mAttn[row_idx, None],
                                mGate[row_idx, None],
                                elem_base,
                                global_scale,
                            )
                            sf_offset = self._compute_sf_offset(
                                row_idx, sf_idx_in_row, padded_sf_cols
                            )
                            mScales[sf_offset] = scale_fp8
                            out_base = sf_idx_in_row * (NVFP4_SF_VEC_SIZE // 2)
                            out_ptr = get_ptr_as_int64(mOutput[row_idx, None], out_base)
                            st_global_u64(out_ptr, packed64)

                        if cutlass.const_expr(self.num_sf_blocks_per_row != self.padded_sf_cols):
                            pad_col = num_sf_blocks_per_row + sf_idx_in_row
                            while pad_col < padded_sf_cols:
                                sf_offset = self._compute_sf_offset(row_idx, pad_col, padded_sf_cols)
                                mScales[sf_offset] = Uint8(0)
                                pad_col = pad_col + threads_per_row
                row_batch_idx = row_batch_idx + grid_dim_x
                row_idx = row_batch_idx * rows_per_block + row_in_block

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _get_compiled_g1_nvfp4(k: int, sf_layout: int, enable_pdl: bool) -> Tuple[Callable, int]:
    sym_m = cute.sym_int()
    sym_scale_size = cute.sym_int()
    attn_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, k), stride_order=(1, 0), assumed_align=16
    )
    gate_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, k), stride_order=(1, 0), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, k // 2), stride_order=(1, 0), assumed_align=16
    )
    scales_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_scale_size,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_obj = G1NVFP4QuantizeSwizzledKernel(k, sf_layout, enable_pdl)
    compiled = cute.compile(
        kernel_obj,
        attn_fake,
        gate_fake,
        output_fake,
        scales_fake,
        Int32(1),
        Int32(128),
        Int32(1),
        global_scale_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled, kernel_obj.rows_per_block


def g1_nvfp4_quantize_cute_dsl(
    attn: torch.Tensor,
    gate: torch.Tensor,
    global_scale: torch.Tensor,
    sf_layout: int = SF_LAYOUT_128x4,
    enable_pdl: bool | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert attn.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
    assert attn.is_cuda and gate.is_cuda
    assert attn.device == gate.device
    assert attn.shape == gate.shape and attn.dim() == 2
    assert attn.is_contiguous() and gate.is_contiguous()
    m, k = attn.shape
    assert k % NVFP4_SF_VEC_SIZE == 0
    enable_pdl = (
        torch.cuda.get_device_capability(attn.device)[0] >= 9
        if enable_pdl is None
        else enable_pdl
    )
    if sf_layout == SF_LAYOUT_8x4:
        row_tile_size = 8
    else:
        row_tile_size = ROW_TILE_SIZE
    padded_m = ((m + row_tile_size - 1) // row_tile_size) * row_tile_size
    num_sf_blocks_per_row = k // NVFP4_SF_VEC_SIZE
    padded_sf_cols = ((num_sf_blocks_per_row + 3) // 4) * 4
    scale_output_size = padded_m * padded_sf_cols
    kernel_fn, rows_per_block = _get_compiled_g1_nvfp4(k, sf_layout, bool(enable_pdl))
    num_sm = get_num_sm(attn.device)
    target_grid = num_sm * _BLOCKS_PER_SM
    num_blocks = min((padded_m + rows_per_block - 1) // rows_per_block, target_grid)
    global_scale_tensor = global_scale.float().reshape(1).contiguous().to(attn.device)
    fp4_output = torch.empty(m, k // 2, dtype=torch.uint8, device=attn.device)
    scale_output = torch.empty(scale_output_size, dtype=torch.uint8, device=attn.device)
    kernel_fn(
        attn,
        gate,
        fp4_output,
        scale_output,
        m,
        padded_m,
        num_blocks,
        global_scale_tensor,
    )
    return fp4_output, scale_output.reshape(-1, padded_sf_cols)


def maybe_g1_nvfp4_quantize(
    x: torch.Tensor,
    gate: torch.Tensor | None,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused G1+NVFP4 quantization output when the safe path applies."""
    if gate is None:
        return None
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return None
    if not x.is_cuda or not gate.is_cuda:
        return None
    if x.device != gate.device:
        return None
    if x.dim() != 2 or x.shape != gate.shape:
        return None
    if not x.is_contiguous():
        return None
    gate = gate.contiguous().view_as(x)
    if x.shape[-1] % NVFP4_SF_VEC_SIZE != 0:
        return None
    return g1_nvfp4_quantize_cute_dsl(x, gate, global_scale)


def apply_g1_gate_for_fp4_fallback(
    x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Materialize the standard BF16 G1 result before FP4 quantization."""
    gate = gate.contiguous().view_as(x)
    if (
        x.is_cuda
        and gate.is_cuda
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
    ):
        try:
            from sgl_kernel import g1_gate_forward_fused
        except (ImportError, OSError):
            g1_gate_forward_fused = None
        if g1_gate_forward_fused is not None:
            out = torch.empty_like(x)
            g1_gate_forward_fused(gate, x, out)
            return out
    return (x * torch.sigmoid(gate.float())).to(x.dtype)
