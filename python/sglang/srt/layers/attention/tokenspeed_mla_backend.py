from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend
from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_INSTALL_HINT = (
    "tokenspeed_mla package is not installed. Install it with "
    "`uv pip install tokenspeed-mla`."
)
_MAX_WORKSPACE_Q_LEN = 8
_WORKSPACE_BUFFERS: dict[torch.device, torch.Tensor] = {}


@dataclass
class TokenSpeedMLADecodeMetadata:
    block_kv_indices: torch.Tensor
    max_seq_len: int


def _require_tokenspeed_mla():
    try:
        import tokenspeed_mla
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return tokenspeed_mla


class TokenSpeedMLABackend(FlashInferMLAAttnBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        if not is_sm100_supported():
            raise ValueError("TokenSpeed MLA backend is only supported on SM100 GPUs.")

        _require_tokenspeed_mla()

        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.forward_metadata: Optional[TokenSpeedMLADecodeMetadata] = None
        self.cuda_graph_kv_indices = None

        if (
            self.qk_nope_head_dim != 128
            or self.qk_rope_head_dim != 64
            or self.v_head_dim != 128
        ):
            raise ValueError(
                "TokenSpeed MLA requires DeepSeek MLA dimensions "
                "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)."
            )
        if self.page_size not in (32, 64):
            raise ValueError("TokenSpeed MLA requires page_size 32 or 64.")

    def _get_workspace(self, device: torch.device, q_len: int) -> torch.Tensor:
        tokenspeed_mla = _require_tokenspeed_mla()
        needed = (
            tokenspeed_mla.get_num_sm(device)
            * self.num_local_heads
            * max(q_len, _MAX_WORKSPACE_Q_LEN)
            * (self.kv_lora_rank + 1)
            * 4
        )
        existing = _WORKSPACE_BUFFERS.get(device)
        if existing is None or existing.numel() < needed:
            existing = torch.empty(needed, dtype=torch.int8, device=device)
            _WORKSPACE_BUFFERS[device] = existing
        return existing

    def _make_block_kv_indices(
        self,
        bs: int,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        max_blocks = triton.cdiv(max_seq_len, self.page_size)
        if out is None:
            out = torch.full(
                (bs, max_blocks),
                -1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
        create_flashmla_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            out,
            self.req_to_token.stride(0),
            out.stride(0),
            PAGED_SIZE=self.page_size,
        )
        return out[:, :max_blocks]

    def _decode_scales(self, layer: RadixAttention) -> tuple[float, float]:
        if self.data_type == torch.float8_e4m3fn:
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            return layer.scaling * k_scale, k_scale
        return layer.scaling, 1.0

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_decode_or_idle():
            bs = forward_batch.batch_size
            max_seq_len = int(forward_batch.seq_lens_cpu.max().item())
            block_kv_indices = self._make_block_kv_indices(
                bs,
                max_seq_len,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
            )
            self.forward_metadata = TokenSpeedMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len=max_seq_len,
            )
        else:
            super().init_forward_metadata(forward_batch)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        block_kv_indices: Optional[torch.Tensor] = None,
    ):
        if block_kv_indices is None:
            block_kv_indices = torch.full(
                (max_bs, triton.cdiv(self.max_context_len, self.page_size)),
                1,
                dtype=torch.int32,
                device="cuda",
            )
        self.cuda_graph_kv_indices = block_kv_indices

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            max_seq_len = int(seq_lens.max().item())
            block_kv_indices = self._make_block_kv_indices(
                bs,
                max_seq_len,
                req_pool_indices,
                seq_lens,
                self.cuda_graph_kv_indices,
            )
            self.forward_metadata = TokenSpeedMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len=max_seq_len,
            )
        else:
            super().init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_decode_or_idle():
            assert seq_lens_cpu is not None
            seq_lens = seq_lens[:bs]
            max_seq_len = int(seq_lens_cpu[:bs].max().item())
            block_kv_indices = self._make_block_kv_indices(
                bs,
                max_seq_len,
                req_pool_indices[:bs],
                seq_lens,
                self.cuda_graph_kv_indices,
            )
            self.forward_metadata = TokenSpeedMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len=max_seq_len,
            )
        else:
            super().init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _merge_query(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        q_rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if q_rope is None:
            return q.view(-1, layer.tp_q_head_num, layer.head_dim)

        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
        )
        return torch.cat([q_nope, q_rope], dim=-1)

    def _merge_key(
        self,
        k: torch.Tensor,
        layer: RadixAttention,
        k_rope: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if k_rope is None:
            return k.view(-1, layer.tp_k_head_num, layer.head_dim)

        k_nope = k.view(-1, layer.tp_k_head_num, layer.v_head_dim)
        k_rope = k_rope.view(
            -1, layer.tp_k_head_num, layer.head_dim - layer.v_head_dim
        )
        return torch.cat([k_nope, k_rope], dim=-1)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None and save_kv_cache:
            if k_rope is None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )

        tokenspeed_mla = _require_tokenspeed_mla()
        query = self._merge_query(q, layer, q_rope).to(self.data_type)
        bs = forward_batch.batch_size
        q_len = query.shape[0] // bs
        query = query.view(bs, q_len, layer.tp_q_head_num, layer.head_dim)
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = kv_cache.view(-1, self.page_size, self.kv_cache_dim)
        workspace = self._get_workspace(query.device, q_len)
        softmax_scale, output_scale = self._decode_scales(layer)

        out = tokenspeed_mla.tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=self.forward_metadata.block_kv_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            max_seq_len=self.forward_metadata.max_seq_len,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            enable_pdl=False,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if not getattr(self.forward_metadata, "use_ragged", False):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                q_rope,
                k_rope,
            )

        if k is not None and save_kv_cache:
            if k_rope is None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, k_rope
                )

        tokenspeed_mla = _require_tokenspeed_mla()
        query = self._merge_query(q, layer, q_rope)
        key = self._merge_key(k, layer, k_rope)
        value = v.view(-1, layer.tp_k_head_num, layer.v_head_dim).contiguous()
        bs = forward_batch.batch_size
        cum_seq_lens = self.qo_indptr[: bs + 1]
        max_seq_len = int((cum_seq_lens[1:] - cum_seq_lens[:-1]).max().item())

        out = tokenspeed_mla.tokenspeed_mla_prefill(
            query=query,
            key=key,
            value=value,
            seq_lens=cum_seq_lens[1:] - cum_seq_lens[:-1],
            cum_seq_lens=cum_seq_lens,
            max_seq_len=max_seq_len,
            batch_size=bs,
            softmax_scale=layer.scaling,
            is_causal=True,
            return_lse=False,
            enable_pdl=False,
        )
        return out.view(-1, layer.tp_q_head_num * layer.v_head_dim)
