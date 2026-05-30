from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

NSA_PREFILL_CP_KV_STORAGE_CHOICES = ("replicated", "layersplit")
NSA_PREFILL_CP_LAYERSPLIT_LAYOUT_CHOICES = ("interleaved", "contiguous")
LAYERSPLIT_B200_CANDIDATE_ENV = "SGLANG_LAYERSPLIT_B200_CANDIDATE"
LAYERSPLIT_STAGE_SMALL_BYTES_ENV = "SGLANG_LAYERSPLIT_STAGE_SMALL_BYTES"
LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES = 512 * 1024
# Iter4 (#16) bumped the HIGGS dense 2-bit slot stride from 258 to
# 272 (258 B payload + 14 B 16-align pad) so ``cp.async.16`` from the
# slot base is legal in the split-K decode kernel. LayerSplit dense-KV
# transfer chunks size against this slot stride.
LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES = 272


@dataclass(frozen=True)
class LayerSplitB200Candidate:
    """Opt-in LayerSplit B200 validation candidate.

    The production candidate preserves incumbent behavior. Non-production
    entries are metadata-backed probes so B200 runs can select and report a
    single concrete variant without promoting it by default.
    """

    name: str
    summary: str
    category: str
    stage_small_bytes: Optional[int] = None
    descriptor_cp_size: Optional[int] = None
    descriptor_b200_count: Optional[int] = None
    producer_stage_fusion: bool = False
    higgs_dense_kv_transfer_sizing: bool = False
    owner_local_validation: bool = False
    runtime_env: Tuple[Tuple[str, str], ...] = ()
    requires_b200: bool = True
    requires_higgs_dense_kv: bool = True
    requires_nvfp4_index_cache: bool = True
    requires_hisa: bool = True
    requires_ikp: bool = True
    requires_czs: bool = False
    proof_modules: Tuple[str, ...] = ()
    promotion_blockers: Tuple[str, ...] = ()
    production_default: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "summary": self.summary,
            "category": self.category,
            "stage_small_bytes": self.stage_small_bytes,
            "descriptor_cp_size": self.descriptor_cp_size,
            "descriptor_b200_count": self.descriptor_b200_count,
            "producer_stage_fusion": self.producer_stage_fusion,
            "higgs_dense_kv_transfer_sizing": self.higgs_dense_kv_transfer_sizing,
            "owner_local_validation": self.owner_local_validation,
            "runtime_env": dict(self.runtime_env),
            "requires_b200": self.requires_b200,
            "requires_higgs_dense_kv": self.requires_higgs_dense_kv,
            "requires_nvfp4_index_cache": self.requires_nvfp4_index_cache,
            "requires_hisa": self.requires_hisa,
            "requires_ikp": self.requires_ikp,
            "requires_czs": self.requires_czs,
            "proof_modules": list(self.proof_modules),
            "promotion_blockers": list(self.promotion_blockers),
            "production_default": self.production_default,
        }


_COMMON_B200_PROMOTION_BLOCKERS = (
    "B200 correctness sweep with HIGGS dense KV plus NVFP4 IndexCache/HISA",
    "B200 IKP trace covering stage-copy, descriptor build, NCCL, and HISA overlap",
    "1-2 B200 throughput/TTFT comparison against the incumbent LayerSplit default",
)


LAYERSPLIT_B200_CANDIDATES: Tuple[LayerSplitB200Candidate, ...] = (
    LayerSplitB200Candidate(
        name="production",
        summary="Current LayerSplit behavior: 512 KiB stage-copy threshold, "
        "generic descriptors, producer copy/stage as separate steps, and "
        "incumbent validation.",
        category="baseline",
        stage_small_bytes=LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES,
        requires_b200=False,
        requires_higgs_dense_kv=False,
        requires_nvfp4_index_cache=False,
        requires_hisa=False,
        requires_ikp=False,
        production_default=True,
    ),
    LayerSplitB200Candidate(
        name="stage_copy_threshold_256k",
        summary="Lower the custom stage-copy cutoff to 256 KiB to test whether "
        "copy-engine delegation should start earlier for HIGGS slot rows.",
        category="stage-copy threshold",
        stage_small_bytes=256 * 1024,
        runtime_env=((LAYERSPLIT_STAGE_SMALL_BYTES_ENV, str(256 * 1024)),),
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="stage_copy_threshold_768k",
        summary="Raise the custom stage-copy cutoff to 768 KiB to test whether "
        "B200 keeps the vectorized stage kernel ahead above the incumbent limit.",
        category="stage-copy threshold",
        stage_small_bytes=768 * 1024,
        runtime_env=((LAYERSPLIT_STAGE_SMALL_BYTES_ENV, str(768 * 1024)),),
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="cp2_descriptor_1b200",
        summary="Precompute owner/peer descriptors for CP=2 on one B200, where "
        "each rank owns alternating layers and one local NVLink domain.",
        category="descriptor specialization",
        descriptor_cp_size=2,
        descriptor_b200_count=1,
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="cp4_descriptor_2b200",
        summary="Precompute owner/peer descriptors for CP=4 on two B200s, "
        "keeping the interleaved owner map explicit for two local NVLink domains.",
        category="descriptor specialization",
        descriptor_cp_size=4,
        descriptor_b200_count=2,
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="producer_stage_fusion",
        summary="Allow the producer rank to fuse owner-local descriptor emission "
        "with stage-copy when the active prefix fits the selected threshold.",
        category="producer fusion",
        producer_stage_fusion=True,
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="higgs_dense_kv_transfer_sizing",
        summary="Size LayerSplit dense-KV transfer chunks from the HIGGS 272 "
        "B/token slot (258 B payload + 14 B 16-align pad; iter4 #16) "
        "instead of BF16 dense row width.",
        category="HIGGS transfer sizing",
        higgs_dense_kv_transfer_sizing=True,
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
    LayerSplitB200Candidate(
        name="owner_local_validation",
        summary="Constrain validation to owner-local layers plus an explicit "
        "scratch-layer sentinel per rank before considering wider promotion.",
        category="validation policy",
        owner_local_validation=True,
        promotion_blockers=_COMMON_B200_PROMOTION_BLOCKERS,
    ),
)

_LAYERSPLIT_B200_CANDIDATE_BY_NAME = {
    candidate.name: candidate for candidate in LAYERSPLIT_B200_CANDIDATES
}


def iter_layersplit_b200_candidates(
    *, include_production: bool = True
) -> Tuple[LayerSplitB200Candidate, ...]:
    if include_production:
        return LAYERSPLIT_B200_CANDIDATES
    return tuple(
        candidate
        for candidate in LAYERSPLIT_B200_CANDIDATES
        if not candidate.production_default
    )


def get_layersplit_b200_candidate(
    name: Optional[str] = None,
) -> LayerSplitB200Candidate:
    selected = name
    if selected is None:
        selected = os.environ.get(LAYERSPLIT_B200_CANDIDATE_ENV)
    if selected is None or selected == "":
        selected = "production"
    try:
        return _LAYERSPLIT_B200_CANDIDATE_BY_NAME[selected]
    except KeyError as exc:
        valid = ", ".join(sorted(_LAYERSPLIT_B200_CANDIDATE_BY_NAME))
        raise ValueError(
            f"Unknown LayerSplit B200 candidate {selected!r}. "
            f"Valid candidates: {valid}."
        ) from exc


def layersplit_b200_candidate_metadata() -> dict[str, object]:
    return {
        "selector_env": LAYERSPLIT_B200_CANDIDATE_ENV,
        "stage_threshold_env": LAYERSPLIT_STAGE_SMALL_BYTES_ENV,
        "candidates": [candidate.as_dict() for candidate in LAYERSPLIT_B200_CANDIDATES],
    }


def _parse_stage_small_bytes_override(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{LAYERSPLIT_STAGE_SMALL_BYTES_ENV} must be an integer byte count."
        ) from exc
    if parsed < 0:
        raise ValueError(f"{LAYERSPLIT_STAGE_SMALL_BYTES_ENV} must be non-negative.")
    return parsed


def select_layersplit_stage_small_bytes(
    candidate_name: Optional[str] = None,
) -> int:
    """Return the opt-in stage-copy threshold, keeping production as default."""

    override = os.environ.get(LAYERSPLIT_STAGE_SMALL_BYTES_ENV)
    if override is not None and override != "":
        return _parse_stage_small_bytes_override(override)
    candidate = get_layersplit_b200_candidate(candidate_name)
    if candidate.stage_small_bytes is not None:
        return candidate.stage_small_bytes
    return LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES


def validate_layersplit_server_args(
    *,
    enable_nsa_prefill_context_parallel: bool,
    attn_cp_size: int,
    disaggregation_mode: str,
    disaggregation_transfer_backend: str,
    all_cp_ranks_transfer: bool,
    is_deepseek_nsa_model: bool,
    enable_turboquant_dense_kv_cache: bool,
    turboquant_skip_layers,
) -> None:
    if not enable_nsa_prefill_context_parallel:
        raise ValueError(
            "--nsa-prefill-cp-kv-storage-mode=layersplit requires "
            "--enable-nsa-prefill-context-parallel."
        )
    if attn_cp_size <= 1:
        raise ValueError(
            "--nsa-prefill-cp-kv-storage-mode=layersplit requires an "
            "effective attention CP size greater than 1. Check --tp, "
            "--dp, and --attn-cp-size; a topology with CP size 1 would "
            "not split KV ownership across ranks."
        )
    if disaggregation_mode == "decode":
        raise ValueError(
            "LayerSplit is a prefill CP KV storage policy and should not "
            "be enabled on decode workers."
        )
    if disaggregation_mode == "prefill" and not all_cp_ranks_transfer:
        raise ValueError(
            "--nsa-prefill-cp-kv-storage-mode=layersplit with "
            "disaggregated prefill requires "
            "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1."
        )
    if disaggregation_mode == "prefill" and disaggregation_transfer_backend not in (
        "mooncake",
        "nixl",
    ):
        raise ValueError(
            "--nsa-prefill-cp-kv-storage-mode=layersplit with "
            "disaggregated prefill requires the mooncake or nixl transfer backend."
        )
    if not is_deepseek_nsa_model:
        raise ValueError(
            "--nsa-prefill-cp-kv-storage-mode=layersplit is only "
            "supported for DeepSeek Sparse Attention models."
        )
    if enable_turboquant_dense_kv_cache and turboquant_skip_layers:
        raise ValueError(
            "LayerSplit with dense TurboQuant requires a uniform dense KV "
            "row width and does not support --turboquant-skip-layers."
        )


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
class LayerSplitDescriptor:
    layer_id: int
    local_layer_idx: int
    owner_rank: int
    owned_by_rank: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "local_layer_idx": self.local_layer_idx,
            "owner_rank": self.owner_rank,
            "owned_by_rank": self.owned_by_rank,
        }


@dataclass(frozen=True)
class LayerSplitDescriptorPlan:
    candidate_name: str
    cp_rank: int
    cp_size: int
    b200_count: int
    owned_layer_ids: Tuple[int, ...]
    peer_layer_ids: Tuple[int, ...]
    descriptors: Tuple[LayerSplitDescriptor, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "cp_rank": self.cp_rank,
            "cp_size": self.cp_size,
            "b200_count": self.b200_count,
            "owned_layer_ids": list(self.owned_layer_ids),
            "peer_layer_ids": list(self.peer_layer_ids),
            "descriptors": [descriptor.as_dict() for descriptor in self.descriptors],
        }


@dataclass(frozen=True)
class LayerSplitProducerStageDecision:
    should_fuse: bool
    reason: str
    owner_rank: int
    active_bytes: int
    stage_small_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "should_fuse": self.should_fuse,
            "reason": self.reason,
            "owner_rank": self.owner_rank,
            "active_bytes": self.active_bytes,
            "stage_small_bytes": self.stage_small_bytes,
        }


@dataclass(frozen=True)
class LayerSplitHiggsTransferSizing:
    active_rows: int
    page_size: int
    slot_bytes: int
    active_bytes: int
    transfer_rows: int
    page_item_bytes: int
    transfer_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "active_rows": self.active_rows,
            "page_size": self.page_size,
            "slot_bytes": self.slot_bytes,
            "active_bytes": self.active_bytes,
            "transfer_rows": self.transfer_rows,
            "page_item_bytes": self.page_item_bytes,
            "transfer_bytes": self.transfer_bytes,
        }


@dataclass(frozen=True)
class LayerSplitOwnerLocalValidationPlan:
    candidate_name: str
    owned_layer_ids: Tuple[int, ...]
    scratch_layer_ids: Tuple[int, ...]
    validation_layer_ids: Tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "owned_layer_ids": list(self.owned_layer_ids),
            "scratch_layer_ids": list(self.scratch_layer_ids),
            "validation_layer_ids": list(self.validation_layer_ids),
        }


def build_layersplit_descriptor_plan(
    policy: LayerSplitPolicy,
    *,
    b200_count: int,
    candidate_name: Optional[str] = None,
) -> Optional[LayerSplitDescriptorPlan]:
    """Build an opt-in CP/topology-specialized descriptor plan.

    Returns ``None`` for production/default and for candidates that do not
    match the provided CP size and B200 count.
    """

    candidate = get_layersplit_b200_candidate(candidate_name)
    if candidate.descriptor_cp_size is None:
        return None
    if policy.cp_size != candidate.descriptor_cp_size:
        return None
    if b200_count != candidate.descriptor_b200_count:
        return None

    descriptors = tuple(
        LayerSplitDescriptor(
            layer_id=layer_id,
            local_layer_idx=layer_id - policy.start_layer,
            owner_rank=policy.owner_rank(layer_id),
            owned_by_rank=policy.owns_layer(layer_id),
        )
        for layer_id in range(policy.start_layer, policy.end_layer)
    )
    owned = tuple(
        descriptor.layer_id for descriptor in descriptors if descriptor.owned_by_rank
    )
    peer = tuple(
        descriptor.layer_id
        for descriptor in descriptors
        if not descriptor.owned_by_rank
    )
    return LayerSplitDescriptorPlan(
        candidate_name=candidate.name,
        cp_rank=policy.cp_rank,
        cp_size=policy.cp_size,
        b200_count=b200_count,
        owned_layer_ids=owned,
        peer_layer_ids=peer,
        descriptors=descriptors,
    )


def should_fuse_layersplit_producer_stage(
    policy: LayerSplitPolicy,
    *,
    layer_id: int,
    active_rows: int,
    row_bytes: int,
    candidate_name: Optional[str] = None,
) -> LayerSplitProducerStageDecision:
    """Decide whether an opt-in producer-side stage-copy fusion is eligible."""

    owner_rank = policy.owner_rank(layer_id)
    active_bytes = int(active_rows) * int(row_bytes)
    stage_small_bytes = select_layersplit_stage_small_bytes(candidate_name)
    candidate = get_layersplit_b200_candidate(candidate_name)
    if not candidate.producer_stage_fusion:
        return LayerSplitProducerStageDecision(
            False,
            "candidate does not enable producer-stage fusion",
            owner_rank,
            active_bytes,
            stage_small_bytes,
        )
    if owner_rank != policy.cp_rank:
        return LayerSplitProducerStageDecision(
            False,
            "rank is not the owner for this layer",
            owner_rank,
            active_bytes,
            stage_small_bytes,
        )
    if active_rows <= 0:
        return LayerSplitProducerStageDecision(
            False,
            "active_rows must be positive",
            owner_rank,
            active_bytes,
            stage_small_bytes,
        )
    if row_bytes % 8 != 0:
        return LayerSplitProducerStageDecision(
            False,
            "row_bytes must be divisible by 8",
            owner_rank,
            active_bytes,
            stage_small_bytes,
        )
    if active_bytes > stage_small_bytes:
        return LayerSplitProducerStageDecision(
            False,
            "active prefix exceeds selected stage-copy threshold",
            owner_rank,
            active_bytes,
            stage_small_bytes,
        )
    return LayerSplitProducerStageDecision(
        True,
        "owner-local active prefix is eligible for fused descriptor/stage-copy",
        owner_rank,
        active_bytes,
        stage_small_bytes,
    )


def plan_layersplit_higgs_dense_kv_transfer_sizing(
    *,
    active_rows: int,
    page_size: int,
    slot_bytes: int = LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES,
    candidate_name: Optional[str] = None,
) -> Optional[LayerSplitHiggsTransferSizing]:
    """Plan HIGGS dense-KV transfer bytes for an opt-in LayerSplit probe."""

    candidate = get_layersplit_b200_candidate(candidate_name)
    if not candidate.higgs_dense_kv_transfer_sizing:
        return None
    if active_rows < 0:
        raise ValueError("active_rows must be non-negative.")
    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if slot_bytes <= 0:
        raise ValueError("slot_bytes must be positive.")

    transfer_pages = (active_rows + page_size - 1) // page_size
    transfer_rows = transfer_pages * page_size
    page_item_bytes = page_size * slot_bytes
    return LayerSplitHiggsTransferSizing(
        active_rows=active_rows,
        page_size=page_size,
        slot_bytes=slot_bytes,
        active_bytes=active_rows * slot_bytes,
        transfer_rows=transfer_rows,
        page_item_bytes=page_item_bytes,
        transfer_bytes=transfer_pages * page_item_bytes,
    )


def build_layersplit_owner_local_validation_plan(
    policy: LayerSplitPolicy,
    *,
    scratch_layer_count: int = 1,
    candidate_name: Optional[str] = None,
) -> Optional[LayerSplitOwnerLocalValidationPlan]:
    """Return owner-local validation layers for the opt-in validation probe."""

    candidate = get_layersplit_b200_candidate(candidate_name)
    if not candidate.owner_local_validation:
        return None
    if scratch_layer_count < 0:
        raise ValueError("scratch_layer_count must be non-negative.")

    owned = policy.owned_layer_ids()
    scratch = tuple(
        layer_id
        for layer_id in range(policy.start_layer, policy.end_layer)
        if layer_id not in owned
    )[:scratch_layer_count]
    return LayerSplitOwnerLocalValidationPlan(
        candidate_name=candidate.name,
        owned_layer_ids=owned,
        scratch_layer_ids=scratch,
        validation_layer_ids=owned + scratch,
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
            f"expected {policy.num_layers} source layer pointers, got {layer_count}."
        )
    if len(item_lens) != layer_count:
        raise ValueError(f"expected {layer_count} item lengths, got {len(item_lens)}.")
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
