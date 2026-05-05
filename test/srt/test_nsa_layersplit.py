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


def test_layersplit_mla_transfer_params_use_global_decode_layers():
    policy = LayerSplitPolicy(cp_rank=1, cp_size=2, start_layer=2, end_layer=6)

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
    policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=2, end_layer=6)

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


def test_layersplit_server_arg_validation_accepts_prefill_config():
    validate_layersplit_server_args(**valid_layersplit_args())
