import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for module_name in [
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.attention",
    "sglang.srt.layers.attention.nsa",
]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

layersplit = load_module(
    "sglang.srt.layers.attention.nsa.layersplit",
    ROOT / "python/sglang/srt/layers/attention/nsa/layersplit.py",
)

LAYERSPLIT_B200_CANDIDATE_ENV = layersplit.LAYERSPLIT_B200_CANDIDATE_ENV
LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES = (
    layersplit.LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES
)
LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES = (
    layersplit.LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES
)
LAYERSPLIT_STAGE_SMALL_BYTES_ENV = layersplit.LAYERSPLIT_STAGE_SMALL_BYTES_ENV
LayerSplitPolicy = layersplit.LayerSplitPolicy
build_layersplit_descriptor_plan = layersplit.build_layersplit_descriptor_plan
build_layersplit_owner_local_validation_plan = (
    layersplit.build_layersplit_owner_local_validation_plan
)
build_layersplit_mla_transfer_params = layersplit.build_layersplit_mla_transfer_params
get_layersplit_b200_candidate = layersplit.get_layersplit_b200_candidate
iter_layersplit_b200_candidates = layersplit.iter_layersplit_b200_candidates
layersplit_b200_candidate_metadata = layersplit.layersplit_b200_candidate_metadata
plan_layersplit_higgs_dense_kv_transfer_sizing = (
    layersplit.plan_layersplit_higgs_dense_kv_transfer_sizing
)
select_layersplit_stage_small_bytes = layersplit.select_layersplit_stage_small_bytes
should_fuse_layersplit_producer_stage = layersplit.should_fuse_layersplit_producer_stage
validate_layersplit_server_args = layersplit.validate_layersplit_server_args


def valid_layersplit_args(**overrides):
    args = {
        "enable_nsa_prefill_context_parallel": True,
        "attn_cp_size": 2,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "mooncake",
        "all_cp_ranks_transfer": True,
        "is_deepseek_nsa_model": True,
        "enable_turboquant_dense_kv_cache": True,
        "turboquant_skip_layers": None,
    }
    args.update(overrides)
    return args


def test_layersplit_interleaved_owner_mapping():
    policy = LayerSplitPolicy(
        cp_rank=1,
        cp_size=4,
        start_layer=0,
        end_layer=10,
        layout="interleaved",
    )

    assert policy.owned_layer_ids() == (1, 5, 9)
    assert policy.owner_rank(6) == 2
    assert not policy.owns_layer(8)


def test_layersplit_contiguous_owner_mapping():
    policy = LayerSplitPolicy(
        cp_rank=2,
        cp_size=4,
        start_layer=0,
        end_layer=10,
        layout="contiguous",
    )

    assert policy.owned_layer_ids() == (5, 6, 7)
    assert policy.owner_rank(9) == 3


def test_layersplit_default_owner_mapping_is_interleaved():
    policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=0, end_layer=4)

    assert policy.owned_layer_ids() == (0, 2)


def test_layersplit_b200_candidate_registry_has_opt_in_candidates():
    candidates = iter_layersplit_b200_candidates(include_production=False)

    assert len(candidates) >= 4
    assert {
        "stage_copy_threshold_256k",
        "stage_copy_threshold_768k",
        "cp2_descriptor_1b200",
        "cp4_descriptor_2b200",
        "producer_stage_fusion",
        "higgs_dense_kv_transfer_sizing",
        "owner_local_validation",
    }.issubset({candidate.name for candidate in candidates})
    assert all(candidate.requires_b200 for candidate in candidates)
    assert all(candidate.requires_higgs_dense_kv for candidate in candidates)
    assert all(candidate.requires_nvfp4_index_cache for candidate in candidates)
    assert all(candidate.requires_hisa for candidate in candidates)
    assert all(candidate.requires_ikp for candidate in candidates)


def test_layersplit_b200_candidate_selector_defaults_to_production(monkeypatch):
    monkeypatch.delenv(LAYERSPLIT_B200_CANDIDATE_ENV, raising=False)
    monkeypatch.delenv(LAYERSPLIT_STAGE_SMALL_BYTES_ENV, raising=False)

    assert get_layersplit_b200_candidate().name == "production"
    assert select_layersplit_stage_small_bytes() == (
        LAYERSPLIT_PRODUCTION_STAGE_SMALL_BYTES
    )

    monkeypatch.setenv(LAYERSPLIT_B200_CANDIDATE_ENV, "stage_copy_threshold_256k")
    assert get_layersplit_b200_candidate().stage_small_bytes == 256 * 1024
    assert select_layersplit_stage_small_bytes() == 256 * 1024

    monkeypatch.setenv(LAYERSPLIT_STAGE_SMALL_BYTES_ENV, str(384 * 1024))
    assert select_layersplit_stage_small_bytes() == 384 * 1024


def test_layersplit_b200_candidate_selector_rejects_unknown(monkeypatch):
    monkeypatch.setenv(LAYERSPLIT_B200_CANDIDATE_ENV, "not_a_candidate")

    with pytest.raises(ValueError, match="Unknown LayerSplit B200"):
        get_layersplit_b200_candidate()


def test_layersplit_stage_threshold_env_rejects_bad_values(monkeypatch):
    monkeypatch.setenv(LAYERSPLIT_STAGE_SMALL_BYTES_ENV, "nope")

    with pytest.raises(ValueError, match=LAYERSPLIT_STAGE_SMALL_BYTES_ENV):
        select_layersplit_stage_small_bytes()

    monkeypatch.setenv(LAYERSPLIT_STAGE_SMALL_BYTES_ENV, "-1")
    with pytest.raises(ValueError, match="non-negative"):
        select_layersplit_stage_small_bytes()


def test_layersplit_b200_candidate_metadata_is_json_ready():
    metadata = layersplit_b200_candidate_metadata()

    assert metadata["selector_env"] == LAYERSPLIT_B200_CANDIDATE_ENV
    assert metadata["stage_threshold_env"] == LAYERSPLIT_STAGE_SMALL_BYTES_ENV
    assert isinstance(metadata["candidates"], list)
    assert any(
        candidate["name"] == "production" and candidate["production_default"]
        for candidate in metadata["candidates"]
    )
    assert any(
        candidate["name"] == "stage_copy_threshold_768k"
        and candidate["runtime_env"][LAYERSPLIT_STAGE_SMALL_BYTES_ENV]
        == str(768 * 1024)
        for candidate in metadata["candidates"]
    )


def test_layersplit_cp_descriptor_specialization_candidates_are_opt_in():
    cp2_policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=0, end_layer=6)
    assert (
        build_layersplit_descriptor_plan(
            cp2_policy,
            b200_count=1,
        )
        is None
    )

    cp2_plan = build_layersplit_descriptor_plan(
        cp2_policy,
        b200_count=1,
        candidate_name="cp2_descriptor_1b200",
    )
    assert cp2_plan is not None
    assert cp2_plan.owned_layer_ids == (0, 2, 4)
    assert cp2_plan.peer_layer_ids == (1, 3, 5)
    assert [d.owner_rank for d in cp2_plan.descriptors] == [0, 1, 0, 1, 0, 1]

    cp4_policy = LayerSplitPolicy(
        cp_rank=3,
        cp_size=4,
        start_layer=8,
        end_layer=16,
    )
    cp4_plan = build_layersplit_descriptor_plan(
        cp4_policy,
        b200_count=2,
        candidate_name="cp4_descriptor_2b200",
    )
    assert cp4_plan is not None
    assert cp4_plan.owned_layer_ids == (11, 15)
    assert cp4_plan.as_dict()["b200_count"] == 2

    assert (
        build_layersplit_descriptor_plan(
            cp4_policy,
            b200_count=1,
            candidate_name="cp4_descriptor_2b200",
        )
        is None
    )


def test_layersplit_producer_stage_fusion_candidate_is_owner_local():
    policy = LayerSplitPolicy(cp_rank=1, cp_size=2, start_layer=0, end_layer=4)

    assert not should_fuse_layersplit_producer_stage(
        policy,
        layer_id=1,
        active_rows=32,
        # Match the iter4 (#16) HIGGS slot stride.
        row_bytes=272,
    ).should_fuse

    decision = should_fuse_layersplit_producer_stage(
        policy,
        layer_id=1,
        active_rows=32,
        row_bytes=264,
        candidate_name="producer_stage_fusion",
    )
    assert decision.should_fuse
    assert decision.owner_rank == 1

    too_large = should_fuse_layersplit_producer_stage(
        policy,
        layer_id=1,
        active_rows=4096,
        row_bytes=264,
        candidate_name="producer_stage_fusion",
    )
    assert not too_large.should_fuse
    assert "threshold" in too_large.reason

    remote = should_fuse_layersplit_producer_stage(
        policy,
        layer_id=0,
        active_rows=32,
        row_bytes=264,
        candidate_name="producer_stage_fusion",
    )
    assert not remote.should_fuse
    assert "not the owner" in remote.reason


def test_layersplit_higgs_dense_kv_transfer_sizing_candidate_is_opt_in():
    assert (
        plan_layersplit_higgs_dense_kv_transfer_sizing(
            active_rows=129,
            page_size=64,
        )
        is None
    )

    sizing = plan_layersplit_higgs_dense_kv_transfer_sizing(
        active_rows=129,
        page_size=64,
        candidate_name="higgs_dense_kv_transfer_sizing",
    )
    assert sizing is not None
    # Iter4 (#16): LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES = 272
    # (258 B payload + 14 B 16-align pad).
    assert sizing.slot_bytes == LAYERSPLIT_HIGGS_DENSE_2BIT_SLOT_BYTES
    assert sizing.active_bytes == 129 * 272
    assert sizing.transfer_rows == 192
    assert sizing.page_item_bytes == 64 * 272
    assert sizing.transfer_bytes == 3 * 64 * 272


def test_layersplit_owner_local_validation_candidate_adds_scratch_layer():
    policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=4, end_layer=10)

    assert build_layersplit_owner_local_validation_plan(policy) is None

    plan = build_layersplit_owner_local_validation_plan(
        policy,
        candidate_name="owner_local_validation",
    )
    assert plan is not None
    assert plan.owned_layer_ids == (4, 6, 8)
    assert plan.scratch_layer_ids == (5,)
    assert plan.validation_layer_ids == (4, 6, 8, 5)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("layout", ["interleaved", "contiguous"])
def test_layersplit_owner_mapping_covers_layers_for_cp_sizes(cp_size, layout):
    layer_count = cp_size * 2 + 1
    start_layer = 3
    owners = {}

    for cp_rank in range(cp_size):
        policy = LayerSplitPolicy(
            cp_rank=cp_rank,
            cp_size=cp_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
            layout=layout,
        )
        for layer_id in policy.owned_layer_ids():
            assert layer_id not in owners
            owners[layer_id] = cp_rank

    expected_layers = set(range(start_layer, start_layer + layer_count))
    assert set(owners) == expected_layers

    for layer_id, cp_rank in owners.items():
        policy = LayerSplitPolicy(
            cp_rank=cp_rank,
            cp_size=cp_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
            layout=layout,
        )
        assert policy.owns_layer(layer_id)
        assert policy.owner_rank(layer_id) == cp_rank


def test_layersplit_mla_transfer_params_use_global_decode_layers():
    policy = LayerSplitPolicy(
        cp_rank=1,
        cp_size=2,
        start_layer=2,
        end_layer=6,
        layout="interleaved",
    )

    assert build_layersplit_mla_transfer_params(
        src_data_ptrs=[100, 101, 102, 103],
        dst_data_ptrs=[200, 201, 202, 203, 204, 205],
        item_lens=[10, 11, 12, 13],
        policy=policy,
    ) == [
        (101, 203, 11),
        (103, 205, 13),
    ]


def test_layersplit_mla_transfer_params_use_local_decode_layers_for_same_pp():
    policy = LayerSplitPolicy(
        cp_rank=0,
        cp_size=2,
        start_layer=2,
        end_layer=6,
        layout="interleaved",
    )

    assert build_layersplit_mla_transfer_params(
        src_data_ptrs=[100, 101, 102, 103],
        dst_data_ptrs=[300, 301, 302, 303],
        item_lens=[10, 11, 12, 13],
        policy=policy,
    ) == [
        (100, 300, 10),
        (102, 302, 12),
    ]


def test_layersplit_transfer_params_reject_mismatched_inputs():
    policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=2, end_layer=6)

    with pytest.raises(ValueError, match="source layer pointers"):
        build_layersplit_mla_transfer_params(
            src_data_ptrs=[100, 101],
            dst_data_ptrs=[300, 301],
            item_lens=[10, 11],
            policy=policy,
        )

    with pytest.raises(ValueError, match="item lengths"):
        build_layersplit_mla_transfer_params(
            src_data_ptrs=[100, 101, 102, 103],
            dst_data_ptrs=[300, 301, 302, 303],
            item_lens=[10, 11],
            policy=policy,
        )

    with pytest.raises(ValueError, match="destination layer pointers"):
        build_layersplit_mla_transfer_params(
            src_data_ptrs=[100, 101, 102, 103],
            dst_data_ptrs=[300, 301, 302],
            item_lens=[10, 11, 12, 13],
            policy=policy,
        )


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_layersplit_transfer_params_cover_all_layers_across_cp_sizes(cp_size):
    start_layer = 2
    layer_count = cp_size * 2
    src_data_ptrs = [1000 + i for i in range(layer_count)]
    dst_data_ptrs = [2000 + i for i in range(start_layer + layer_count)]
    item_lens = [3000 + i for i in range(layer_count)]
    all_params = []

    for cp_rank in range(cp_size):
        policy = LayerSplitPolicy(
            cp_rank=cp_rank,
            cp_size=cp_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_count,
        )
        all_params.extend(
            build_layersplit_mla_transfer_params(
                src_data_ptrs=src_data_ptrs,
                dst_data_ptrs=dst_data_ptrs,
                item_lens=item_lens,
                policy=policy,
            )
        )

    assert sorted(all_params) == sorted(
        (
            src_data_ptrs[local_layer_idx],
            dst_data_ptrs[start_layer + local_layer_idx],
            item_lens[local_layer_idx],
        )
        for local_layer_idx in range(layer_count)
    )


def test_layersplit_rejects_bad_policy():
    with pytest.raises(ValueError, match="cp_rank"):
        LayerSplitPolicy(cp_rank=2, cp_size=2, start_layer=0, end_layer=4)

    with pytest.raises(ValueError, match="invalid layer range"):
        LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=4, end_layer=4)

    with pytest.raises(ValueError, match="unsupported"):
        LayerSplitPolicy(
            cp_rank=0,
            cp_size=2,
            start_layer=0,
            end_layer=4,
            layout="striped",
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"enable_nsa_prefill_context_parallel": False},
            "enable-nsa-prefill-context-parallel",
        ),
        ({"attn_cp_size": 1}, "effective attention CP size greater than 1"),
        ({"disaggregation_mode": "decode"}, "decode workers"),
        (
            {
                "disaggregation_mode": "prefill",
                "all_cp_ranks_transfer": False,
            },
            "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1",
        ),
        (
            {"disaggregation_transfer_backend": "mori"},
            "mooncake or nixl transfer backend",
        ),
        ({"is_deepseek_nsa_model": False}, "DeepSeek Sparse Attention"),
        (
            {"turboquant_skip_layers": {1}},
            "uniform dense KV row width",
        ),
    ],
)
def test_layersplit_server_arg_validation_rejects_invalid_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_layersplit_server_args(**valid_layersplit_args(**overrides))


@pytest.mark.parametrize("attn_cp_size", [2, 4, 8])
def test_layersplit_server_arg_validation_accepts_prefill_cp_configs(
    attn_cp_size,
):
    validate_layersplit_server_args(**valid_layersplit_args(attn_cp_size=attn_cp_size))


def test_layersplit_cute_kernel_is_packaged():
    cmake = (ROOT / "sgl-kernel/CMakeLists.txt").read_text()

    assert '"csrc/kvcacheio/layersplit_cute.cu"' in cmake


def test_layersplit_stage_kernel_has_opt_in_threshold_env():
    source = (ROOT / "sgl-kernel/csrc/kvcacheio/layersplit_cute.cu").read_text()

    assert LAYERSPLIT_B200_CANDIDATE_ENV in source
    assert LAYERSPLIT_STAGE_SMALL_BYTES_ENV in source
    assert "stage_copy_threshold_256k" in source
    assert "stage_copy_threshold_768k" in source
    assert "resolve_small_byte_threshold" in source


def test_layersplit_stage_kernel_accepts_higgs_tail_rows():
    source = (ROOT / "sgl-kernel/csrc/kvcacheio/layersplit_cute.cu").read_text()

    assert "row_bytes must be positive" in source
    assert "row_bytes must be divisible by 8" not in source
    assert "layersplit_small_copy_tail_kernel" in source
    assert "row_dst[i] = row_src[i]" in source
    assert "dst_active.copy_(src_active, true)" in source
