from __future__ import annotations

from typing import Optional, Tuple


NSA_INDEXER_MODES = ("vanilla", "indexcache", "hisa", "indexcache-hisa")


def uses_indexcache(mode: str) -> bool:
    return mode in ("indexcache", "indexcache-hisa")


def uses_hisa(mode: str) -> bool:
    return mode in ("hisa", "indexcache-hisa")


def uses_hisa_pruning(
    *,
    seq_len: int,
    index_topk: int,
    block_size: int,
    block_topk: int,
    min_seq_len: int,
) -> bool:
    candidate_capacity = block_size * block_topk
    if candidate_capacity < index_topk:
        return False
    return seq_len > max(min_seq_len, candidate_capacity)


def validate_indexcache_pattern(pattern: Optional[str], num_layers: int) -> None:
    if pattern is None:
        return
    if len(pattern) != num_layers:
        raise ValueError(
            f"NSA IndexCache pattern length must match num_hidden_layers: "
            f"got {len(pattern)}, expected {num_layers}."
        )
    if any(role not in ("F", "S") for role in pattern):
        raise ValueError("NSA IndexCache pattern may only contain 'F' and 'S'.")
    if pattern[0] != "F":
        raise ValueError("NSA IndexCache pattern must keep layer 0 as 'F'.")


def get_indexcache_skip_flags(
    *,
    mode: str,
    layer_id: int,
    num_layers: int,
    freq: int,
    pattern: Optional[str],
    is_nextn: bool,
) -> Tuple[bool, bool]:
    if is_nextn or not uses_indexcache(mode):
        return False, False
    if pattern is not None:
        validate_indexcache_pattern(pattern, num_layers)
        skip_topk = pattern[layer_id] == "S"
        next_skip_topk = layer_id + 1 < num_layers and pattern[layer_id + 1] == "S"
        return skip_topk, next_skip_topk
    if freq < 1:
        raise ValueError("NSA IndexCache frequency must be at least 1.")
    return max(layer_id - 1, 0) % freq != 0, layer_id % freq != 0
