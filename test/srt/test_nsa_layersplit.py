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

load_module(
    "sglang.srt.layers.attention.nsa.indexer_policy",
    ROOT / "python/sglang/srt/layers/attention/nsa/indexer_policy.py",
)
layersplit = load_module(
    "sglang.srt.layers.attention.nsa.layersplit",
    ROOT / "python/sglang/srt/layers/attention/nsa/layersplit.py",
)

LayerSplitPolicy = layersplit.LayerSplitPolicy
build_layersplit_mla_transfer_params = layersplit.build_layersplit_mla_transfer_params
build_layersplit_transfer_descriptors = (
    layersplit.build_layersplit_transfer_descriptors
)
filter_layers_for_cp_owner = layersplit.filter_layers_for_cp_owner
get_indexcache_source_layer_id = layersplit.get_indexcache_source_layer_id


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


def test_layersplit_indexcache_source_from_pattern():
    pattern = "FSSFFS"

    assert get_indexcache_source_layer_id(
        0, num_layers=6, freq=4, pattern=pattern
    ) == 0
    assert get_indexcache_source_layer_id(
        2, num_layers=6, freq=4, pattern=pattern
    ) == 0
    assert get_indexcache_source_layer_id(
        5, num_layers=6, freq=4, pattern=pattern
    ) == 4


def test_layersplit_indexcache_source_from_frequency():
    assert get_indexcache_source_layer_id(
        0, num_layers=8, freq=4, pattern=None
    ) == 0
    assert get_indexcache_source_layer_id(
        2, num_layers=8, freq=4, pattern=None
    ) == 1
    assert get_indexcache_source_layer_id(
        6, num_layers=8, freq=4, pattern=None
    ) == 5


def test_layersplit_transfer_descriptors_keep_indexcache_source():
    policy = LayerSplitPolicy(cp_rank=0, cp_size=2, start_layer=0, end_layer=6)
    descriptors = build_layersplit_transfer_descriptors(
        policy,
        num_layers=6,
        indexcache_freq=4,
        indexcache_pattern="FSSFFS",
    )

    assert [
        (d.layer_id, d.owner_rank, d.indexcache_source_layer_id)
        for d in descriptors
    ] == [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 0),
        (3, 1, 3),
        (4, 0, 4),
        (5, 1, 4),
    ]


def test_layersplit_disaggregation_layer_owner_filter():
    policy = LayerSplitPolicy(cp_rank=1, cp_size=2, start_layer=0, end_layer=6)

    assert filter_layers_for_cp_owner([0, 1, 2, 3, 4, 5], policy) == (1, 3, 5)


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
