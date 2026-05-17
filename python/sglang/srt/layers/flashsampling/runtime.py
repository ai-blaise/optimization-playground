import dataclasses
import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.flashsampling.tp_info import TPInfo
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FlashSamplingInfo:
    hidden_states: torch.Tensor
    lm_head_weight: torch.Tensor
    vocab_start_index: int
    valid_vocab_size: int
    use_attn_tp_group: bool
    logit_scale: Optional[float] = None


def get_flashsampling_info(
    *,
    hidden_states: torch.Tensor,
    lm_head,
    sampling_info: Optional[SamplingBatchInfo],
    server_args,
    forward_mode,
    extend_return_logprob: bool,
    final_logit_softcapping: Optional[float],
    logit_scale: Optional[float],
    use_fp32_lm_head: bool,
    use_attn_tp_group: bool,
    do_dp_attention_lm_head_gather: bool,
) -> Optional[FlashSamplingInfo]:
    if not getattr(server_args, "enable_flashsampling", False):
        return None
    if sampling_info is None:
        return None
    if not forward_mode.is_decode():
        return None
    min_batch_size = getattr(server_args, "flashsampling_min_batch_size", 1)
    if hidden_states.shape[0] < min_batch_size:
        return None
    max_batch_size = getattr(server_args, "flashsampling_max_batch_size", None)
    if max_batch_size is not None and hidden_states.shape[0] > max_batch_size:
        return None
    if not is_flashsampling_sampling_info_supported(sampling_info):
        _handle_rejected_batch(server_args, "sampling parameters require logits")
        return None
    if extend_return_logprob:
        _handle_rejected_batch(server_args, "input logprobs require logits")
        return None
    if final_logit_softcapping is not None:
        _handle_rejected_batch(server_args, "final logit softcapping is unsupported")
        return None
    if use_fp32_lm_head:
        _handle_rejected_batch(server_args, "fp32 lm_head path is unsupported")
        return None
    if do_dp_attention_lm_head_gather:
        _handle_rejected_batch(server_args, "DP-attention lm_head gather is unsupported")
        return None
    if not torch.cuda.is_available():
        _handle_rejected_batch(server_args, "FlashSampling requires CUDA")
        return None
    if not hasattr(lm_head, "weight") or hasattr(lm_head, "apply_lora"):
        _handle_rejected_batch(server_args, "lm_head type is unsupported")
        return None
    metadata = get_flashsampling_lm_head_metadata(lm_head)
    if metadata is None:
        _handle_rejected_batch(server_args, "lm_head layout is unsupported")
        return None
    weight, vocab_start_index, valid_vocab_size = metadata

    return FlashSamplingInfo(
        hidden_states=hidden_states,
        lm_head_weight=weight,
        vocab_start_index=vocab_start_index,
        valid_vocab_size=valid_vocab_size,
        use_attn_tp_group=use_attn_tp_group,
        logit_scale=logit_scale,
    )


def is_flashsampling_sampling_info_supported(
    sampling_info: SamplingBatchInfo,
) -> bool:
    if sampling_info.has_custom_logit_processor:
        return False
    if sampling_info.grammars or sampling_info.vocab_mask is not None:
        return False
    if sampling_info.acc_additive_penalties is not None:
        return False
    if sampling_info.acc_scaling_penalties is not None:
        return False
    if sampling_info.logit_bias is not None:
        return False
    if sampling_info.sampling_seed is not None:
        return False
    if sampling_info.need_min_p_sampling or sampling_info.need_top_p_sampling:
        return False
    if sampling_info.is_all_greedy:
        return True
    if sampling_info.need_top_k_sampling:
        return False
    return _temperature_is_uniform(sampling_info)


def _temperature_is_uniform(sampling_info: SamplingBatchInfo) -> bool:
    temperature_is_uniform = getattr(sampling_info, "temperature_is_uniform", None)
    if temperature_is_uniform is not None:
        return temperature_is_uniform

    temperatures = sampling_info.temperatures.reshape(-1)
    return bool(torch.all(temperatures == temperatures[0]).item())


def get_flashsampling_lm_head_metadata(
    lm_head,
) -> Optional[tuple[torch.Tensor, int, int]]:
    if not hasattr(lm_head, "weight") or hasattr(lm_head, "apply_lora"):
        return None
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None and not isinstance(
        quant_method, UnquantizedEmbeddingMethod
    ):
        return None
    if getattr(lm_head, "num_added_embeddings", 0) != 0:
        return None

    weight = lm_head.weight
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return None

    vocab_start_index, valid_vocab_size = _lm_head_vocab_range(lm_head, weight)
    if valid_vocab_size <= 0:
        return None
    return weight, vocab_start_index, valid_vocab_size


def _lm_head_vocab_range(lm_head, weight: torch.Tensor) -> tuple[int, int]:
    shard_indices = getattr(lm_head, "shard_indices", None)
    if shard_indices is None:
        return 0, min(getattr(lm_head, "num_embeddings", weight.shape[0]), weight.shape[0])

    start = shard_indices.padded_org_vocab_start_index
    valid_vocab_size = shard_indices.org_vocab_end_index - start
    return start, min(valid_vocab_size, weight.shape[0])


def _handle_rejected_batch(server_args, reason: str) -> None:
    if getattr(server_args, "flashsampling_fallback", "auto") == "error":
        raise RuntimeError(f"FlashSampling rejected this batch: {reason}.")
    logger.debug("FlashSampling fallback: %s", reason)


class FlashSamplingRuntime:
    def __init__(self):
        self._seed = 0
        self._local_workspace_cache: dict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def sample(
        self,
        info: FlashSamplingInfo,
        sampling_info: SamplingBatchInfo,
    ) -> torch.Tensor:
        group = (
            get_attention_tp_group()
            if info.use_attn_tp_group
            else get_tp_group()
        )
        tp_rank = group.rank_in_group
        tp_size = group.world_size

        hidden_states = info.hidden_states
        if hidden_states.dtype != info.lm_head_weight.dtype:
            hidden_states = hidden_states.to(info.lm_head_weight.dtype)
        hidden_states = hidden_states.contiguous()

        temperature = self._temperature_tensor(info, sampling_info)
        seed = self._next_seed()
        from sglang.srt.layers.flashsampling.core import MIN_BLOCK_SIZE_V

        server_args = get_global_server_args()
        use_target = getattr(server_args, "flashsampling_provider", "triton") == "target"
        if use_target:
            device_index = info.lm_head_weight.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            if torch.cuda.get_device_capability(device_index)[0] >= 10:
                from sglang.srt.layers.flashsampling.target_kernel_blackwell import (
                    fused_mm_sample_blackwell as _fused_mm_sample,
                )
            else:
                from sglang.srt.layers.flashsampling.target_kernel import (
                    fused_mm_sample_target as _fused_mm_sample,
                )
        else:
            from sglang.srt.layers.flashsampling.core import (
                fused_mm_sample_triton as _fused_mm_sample,
            )

        need_scores = tp_size > 1
        workspaces = (
            {}
            if need_scores
            else self._local_workspaces(
                info.lm_head_weight.device,
                (info.valid_vocab_size + MIN_BLOCK_SIZE_V - 1) // MIN_BLOCK_SIZE_V,
                hidden_states.shape[0],
            )
        )
        result = _fused_mm_sample(
            weights=info.lm_head_weight,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temperature,
            seed=seed,
            greedy_sampling=sampling_info.is_all_greedy,
            tp=TPInfo(rank=tp_rank, size=1),
            return_scores=need_scores,
            valid_vocab_size=info.valid_vocab_size,
            vocab_start_index=info.vocab_start_index,
            **workspaces,
        )

        if need_scores:
            samples, scores = result
            samples = self._reduce_tp_samples(samples, scores, group, tp_size)
        else:
            samples = result

        return samples.view(-1).to(torch.int32)

    def _next_seed(self) -> int:
        self._seed = (self._seed + 1) & 0x7FFFFFFF
        return self._seed

    def _local_workspaces(
        self,
        device: torch.device,
        max_grid_size_v: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        from sglang.srt.layers.flashsampling.core import LOCAL_INDEX_DTYPE

        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (device_index, max_grid_size_v, batch_size, 1)
        cached = self._local_workspace_cache.get(key)
        if cached is None:
            maxs = torch.empty(
                (1, max_grid_size_v, batch_size),
                dtype=torch.bfloat16,
                device=device,
            )
            maxs_idx = torch.empty_like(maxs, dtype=LOCAL_INDEX_DTYPE)
            logits_out = torch.empty((1,), dtype=torch.float32, device=device)
            cached = (maxs, maxs_idx, logits_out)
            self._local_workspace_cache[key] = cached
        maxs, maxs_idx, logits_out = cached
        return {
            "maxs_workspace": maxs,
            "maxs_idx_workspace": maxs_idx,
            "logits_out_workspace": logits_out,
        }

    @staticmethod
    def _temperature_tensor(
        info: FlashSamplingInfo,
        sampling_info: SamplingBatchInfo,
    ) -> torch.Tensor:
        if sampling_info.is_all_greedy:
            return torch.empty((), dtype=torch.float32, device=info.hidden_states.device)

        temperature = sampling_info.temperatures.reshape(-1)[0]
        if info.logit_scale is not None:
            temperature = temperature / info.logit_scale
        if temperature.dtype != torch.float32:
            temperature = temperature.float()
        return temperature.reshape(())

    @staticmethod
    def _reduce_tp_samples(
        samples: torch.Tensor,
        scores: torch.Tensor,
        group,
        tp_size: int,
    ) -> torch.Tensor:
        batch_size = samples.shape[0]
        all_scores = group.all_gather(scores.contiguous(), dim=0).view(
            tp_size, batch_size, 1
        )
        all_samples = group.all_gather(samples.contiguous(), dim=0).view(
            tp_size, batch_size, 1
        )
        best_rank = all_scores.argmax(dim=0).view(1, batch_size, 1)
        return all_samples.gather(0, best_rank).squeeze(0)
