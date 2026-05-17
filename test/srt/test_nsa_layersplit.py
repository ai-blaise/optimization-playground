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

LayerSplitPolicy = layersplit.LayerSplitPolicy
build_layersplit_mla_transfer_params = layersplit.build_layersplit_mla_transfer_params
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
def test_layersplit_server_arg_validation_rejects_invalid_configs(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        validate_layersplit_server_args(**valid_layersplit_args(**overrides))


@pytest.mark.parametrize("attn_cp_size", [2, 4, 8])
def test_layersplit_server_arg_validation_accepts_prefill_cp_configs(
    attn_cp_size,
):
    validate_layersplit_server_args(
        **valid_layersplit_args(attn_cp_size=attn_cp_size)
    )


def test_layersplit_cute_kernel_is_packaged():
    cmake = (ROOT / "sgl-kernel/CMakeLists.txt").read_text()

    assert '"csrc/kvcacheio/layersplit_cute.cu"' in cmake
