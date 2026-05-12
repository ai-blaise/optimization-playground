import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    try:
        from sglang.jit_kernel.nvfp4_indexer import (
            can_use_nsa_nvfp4_indexer,
            fused_store_index_k_cache_nvfp4,
            quantize_indexer_q_nvfp4,
        )
    except ModuleNotFoundError:
        can_use_nsa_nvfp4_indexer = None
        fused_store_index_k_cache_nvfp4 = None
        quantize_indexer_q_nvfp4 = None
else:
    can_use_nsa_nvfp4_indexer = None
    fused_store_index_k_cache_nvfp4 = None
    quantize_indexer_q_nvfp4 = None

pytestmark = pytest.mark.skipif(
    torch is None or can_use_nsa_nvfp4_indexer is None,
    reason="torch and SGLang runtime dependencies are required",
)


def _nvfp4_supported() -> bool:
    if torch is None:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def _quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    idx = torch.where(ax > 0.25, 1, idx)
    idx = torch.where(ax >= 0.75, 2, idx)
    idx = torch.where(ax > 1.25, 3, idx)
    idx = torch.where(ax >= 1.75, 4, idx)
    idx = torch.where(ax > 2.5, 5, idx)
    idx = torch.where(ax >= 3.5, 6, idx)
    idx = torch.where(ax > 5.0, 7, idx)
    sign = (x < 0) & (idx != 0)
    return idx | (sign.to(torch.uint8) << 3)


def _ref_indexer_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    assert cols == 128
    x_view = x.view(rows, 4, 32)
    scale = _ceil_to_ue8m0(x_view.abs().float().amax(dim=2).clamp_min(1e-4) / 6.0)
    codes = _quantize_to_e2m1(x_view.float() / scale.unsqueeze(-1)).view(rows, cols)
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    return packed.contiguous(), _pack_ue8m0_to_int(scale.contiguous())


def test_nvfp4_indexer_jit_reports_unavailable_without_blackwell():
    if _nvfp4_supported():
        pytest.skip("Blackwell runtime covers the executable NVFP4 tests.")
    assert not can_use_nsa_nvfp4_indexer(torch.bfloat16, torch.int64, 64)


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_quantize_indexer_q_nvfp4_matches_reference():
    torch.manual_seed(0)
    query = (torch.randn((7, 64, 128), device="cuda") * 0.5).to(torch.bfloat16)
    values, scales = quantize_indexer_q_nvfp4(query)
    ref_values, ref_scales = _ref_indexer_nvfp4(query.view(-1, 128))
    torch.testing.assert_close(values.view(-1, 64), ref_values)
    torch.testing.assert_close(scales.view(-1), ref_scales.view(-1))


@pytest.mark.skipif(not _nvfp4_supported(), reason="NVFP4 requires Blackwell.")
def test_fused_store_index_k_cache_nvfp4_matches_reference():
    torch.manual_seed(1)
    key = (torch.randn((11, 128), device="cuda") * 0.5).to(torch.bfloat16)
    loc = torch.tensor([0, 3, 7, 64, 65, 127, 128, 129, 191, 255, 319], device="cuda")
    page_size = 64
    buf = torch.zeros((5, page_size * 68), dtype=torch.uint8, device="cuda")

    fused_store_index_k_cache_nvfp4(key, buf, loc, page_size=page_size)

    ref_values, ref_scales = _ref_indexer_nvfp4(key)
    for row, token in enumerate(loc.tolist()):
        page = token // page_size
        offset = token % page_size
        value_start = offset * 64
        scale_start = page_size * 64 + offset * 4
        torch.testing.assert_close(
            buf[page, value_start : value_start + 64], ref_values[row]
        )
        torch.testing.assert_close(
            buf[page, scale_start : scale_start + 4].view(torch.int32),
            ref_scales.view(-1)[row : row + 1],
        )
