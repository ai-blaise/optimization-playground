import pytest

from sglang.srt.layers.attention.nsa.indexer_policy import (
    get_indexcache_skip_flags,
    validate_indexcache_pattern,
)
from sglang.srt.server_args import prepare_server_args


def test_indexcache_pattern_validation():
    validate_indexcache_pattern("FSSS", 4)

    with pytest.raises(ValueError, match="length"):
        validate_indexcache_pattern("FSS", 4)

    with pytest.raises(ValueError, match="layer 0"):
        validate_indexcache_pattern("SFSS", 4)

    with pytest.raises(ValueError, match="only contain"):
        validate_indexcache_pattern("FSXS", 4)


def test_indexcache_skip_flags_from_pattern():
    pattern = "FSSFFS"
    assert get_indexcache_skip_flags(
        mode="indexcache-hisa",
        layer_id=0,
        num_layers=6,
        freq=4,
        pattern=pattern,
        is_nextn=False,
    ) == (False, True)
    assert get_indexcache_skip_flags(
        mode="indexcache-hisa",
        layer_id=1,
        num_layers=6,
        freq=4,
        pattern=pattern,
        is_nextn=False,
    ) == (True, True)
    assert get_indexcache_skip_flags(
        mode="hisa",
        layer_id=1,
        num_layers=6,
        freq=4,
        pattern=pattern,
        is_nextn=False,
    ) == (False, False)


def test_nsa_indexer_cli_updates_model_override_args():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--nsa-indexer-mode",
            "indexcache-hisa",
            "--nsa-indexcache-pattern",
            "FSSS",
            "--hisa-block-size",
            "128",
            "--hisa-block-topk",
            "64",
        ]
    )
    assert '"nsa_indexer_mode": "indexcache-hisa"' in args.json_model_override_args
    assert '"index_topk_pattern": "FSSS"' in args.json_model_override_args
