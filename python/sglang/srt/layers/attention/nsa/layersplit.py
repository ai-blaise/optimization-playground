from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

from sglang.srt.layers.attention.nsa.indexer_policy import validate_indexcache_pattern

NSA_PREFILL_CP_KV_STORAGE_CHOICES = ("replicated", "layersplit")
NSA_PREFILL_CP_LAYERSPLIT_LAYOUT_CHOICES = ("interleaved", "contiguous")


@dataclass(frozen=True)
class LayerSplitPolicy:
    cp_rank: int
    cp_size: int
    start_layer: int
    end_layer: int
    layout: str = "interleaved"

    def __post_init__(self) -> None:
        if self.cp_size <= 0:
            raise ValueError("cp_size must be positive.")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(
                f"cp_rank must be in [0, {self.cp_size}), got {self.cp_rank}."
            )
        if self.start_layer < 0 or self.end_layer <= self.start_layer:
            raise ValueError(
                f"invalid layer range [{self.start_layer}, {self.end_layer})."
            )
        if self.layout not in NSA_PREFILL_CP_LAYERSPLIT_LAYOUT_CHOICES:
            raise ValueError(f"unsupported LayerSplit layout: {self.layout}.")

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    def owner_rank(self, layer_id: int) -> int:
        self._check_layer_id(layer_id)
        layer_offset = layer_id - self.start_layer
        if self.layout == "interleaved":
            return layer_offset % self.cp_size
        return min(layer_offset * self.cp_size // self.num_layers, self.cp_size - 1)

    def owns_layer(self, layer_id: int) -> bool:
        return self.owner_rank(layer_id) == self.cp_rank

    def owned_layer_ids(self) -> Tuple[int, ...]:
        return tuple(
            layer_id
            for layer_id in range(self.start_layer, self.end_layer)
            if self.owns_layer(layer_id)
        )

    def _check_layer_id(self, layer_id: int) -> None:
        if not self.start_layer <= layer_id < self.end_layer:
            raise ValueError(
                f"layer_id {layer_id} is outside "
                f"[{self.start_layer}, {self.end_layer})."
            )


@dataclass(frozen=True)
class LayerSplitTransferDescriptor:
    layer_id: int
    owner_rank: int
    indexcache_source_layer_id: int


@dataclass(frozen=True)
class LayerSplitMetadata:
    policy: LayerSplitPolicy
    transfer_descriptors: Tuple[LayerSplitTransferDescriptor, ...]

    def owns_layer(self, layer_id: int) -> bool:
        return self.policy.owns_layer(layer_id)

    @property
    def owned_layer_ids(self) -> Tuple[int, ...]:
        return self.policy.owned_layer_ids()


def get_indexcache_source_layer_id(
    layer_id: int,
    *,
    num_layers: int,
    freq: int,
    pattern: Optional[str],
) -> int:
    if not 0 <= layer_id < num_layers:
        raise ValueError(f"layer_id must be in [0, {num_layers}), got {layer_id}.")
    if pattern is not None:
        validate_indexcache_pattern(pattern, num_layers)
        if pattern[layer_id] == "F":
            return layer_id
        for source_layer_id in range(layer_id - 1, -1, -1):
            if pattern[source_layer_id] == "F":
                return source_layer_id
        raise ValueError("NSA IndexCache pattern must keep layer 0 as 'F'.")
    if freq < 1:
        raise ValueError("NSA IndexCache frequency must be at least 1.")
    if max(layer_id - 1, 0) % freq == 0:
        return layer_id
    for source_layer_id in range(layer_id - 1, -1, -1):
        if max(source_layer_id - 1, 0) % freq == 0:
            return source_layer_id
    return 0


def build_layersplit_transfer_descriptors(
    policy: LayerSplitPolicy,
    *,
    num_layers: int,
    indexcache_freq: int,
    indexcache_pattern: Optional[str],
) -> Tuple[LayerSplitTransferDescriptor, ...]:
    return tuple(
        LayerSplitTransferDescriptor(
            layer_id=layer_id,
            owner_rank=policy.owner_rank(layer_id),
            indexcache_source_layer_id=get_indexcache_source_layer_id(
                layer_id,
                num_layers=num_layers,
                freq=indexcache_freq,
                pattern=indexcache_pattern,
            ),
        )
        for layer_id in range(policy.start_layer, policy.end_layer)
    )


def filter_layers_for_cp_owner(
    layer_ids: Iterable[int],
    policy: LayerSplitPolicy,
) -> Tuple[int, ...]:
    return tuple(layer_id for layer_id in layer_ids if policy.owns_layer(layer_id))


def prepare_layersplit_metadata(
    *,
    cp_rank: int,
    cp_size: int,
    start_layer: int,
    end_layer: int,
    layout: str,
    num_layers: int,
    indexcache_freq: int,
    indexcache_pattern: Optional[str],
) -> LayerSplitMetadata:
    policy = LayerSplitPolicy(
        cp_rank=cp_rank,
        cp_size=cp_size,
        start_layer=start_layer,
        end_layer=end_layer,
        layout=layout,
    )
    return LayerSplitMetadata(
        policy=policy,
        transfer_descriptors=build_layersplit_transfer_descriptors(
            policy,
            num_layers=num_layers,
            indexcache_freq=indexcache_freq,
            indexcache_pattern=indexcache_pattern,
        ),
    )


def build_layersplit_mla_transfer_params(
    src_data_ptrs: Sequence[int],
    dst_data_ptrs: Sequence[int],
    item_lens: Sequence[int],
    policy: LayerSplitPolicy,
) -> list[tuple[int, int, int]]:
    layer_count = len(src_data_ptrs)
    if layer_count != policy.num_layers:
        raise ValueError(
            f"expected {policy.num_layers} source layer pointers, got "
            f"{layer_count}."
        )
    if len(item_lens) != layer_count:
        raise ValueError(
            f"expected {layer_count} item lengths, got {len(item_lens)}."
        )
    if len(dst_data_ptrs) != layer_count and len(dst_data_ptrs) < policy.end_layer:
        raise ValueError(
            "destination layer pointers must describe either the local PP stage "
            "or the global decode layer range."
        )

    layers_params = []
    for local_layer_idx in range(layer_count):
        layer_id = policy.start_layer + local_layer_idx
        if not policy.owns_layer(layer_id):
            continue
        if len(dst_data_ptrs) == layer_count:
            dst_ptr = dst_data_ptrs[local_layer_idx]
        else:
            dst_ptr = dst_data_ptrs[layer_id]
        layers_params.append(
            (
                src_data_ptrs[local_layer_idx],
                dst_ptr,
                item_lens[local_layer_idx],
            )
        )
    return layers_params
