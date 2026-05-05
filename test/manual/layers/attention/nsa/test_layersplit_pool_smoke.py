import os


class Args:
    nsa_prefill_cp_kv_storage_mode = "layersplit"

    def __init__(self):
        self.nsa_prefill_cp_layersplit_layout = os.environ.get(
            "LAYERSPLIT_LAYOUT", "contiguous"
        )


class Group:
    def __init__(self, dist):
        self._dist = dist

    @property
    def world_size(self):
        return self._dist.get_world_size()

    def broadcast(self, tensor, src=0):
        self._dist.broadcast(tensor, src=src)
        return tensor


def patch_sglang_runtime(dist):
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.server_args as server_args

    dp_attention.get_attention_cp_rank = lambda: dist.get_rank()
    dp_attention.get_attention_cp_size = lambda: dist.get_world_size()
    dp_attention.get_attention_cp_group = lambda: Group(dist)
    server_args.get_global_server_args = lambda: Args()


def assert_tensor_value(name, tensor, expected):
    actual = int(tensor.flatten()[0].item())
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def smoke_layer_count(world_size):
    return int(os.environ.get("LAYERSPLIT_SMOKE_LAYERS", max(4, world_size * 2)))


def run_dense_pool(torch, dist, device, layer_count):
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

    pool = NSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=8,
        dtype=torch.bfloat16,
        qk_rope_head_dim=4,
        layer_num=layer_count,
        device=device,
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=12,
        start_layer=0,
        end_layer=layer_count,
        index_buf_size=128,
    )
    rank = dist.get_rank()
    owned_layers = 0
    for layer_id in range(layer_count):
        if pool.layersplit_owns_layer(layer_id):
            owned_layers += 1
            pool.kv_buffer[layer_id].fill_(10 + layer_id)
            pool.index_k_with_scale_buffer[layer_id].fill_(40 + layer_id)
    if owned_layers == 0:
        raise AssertionError(f"rank {rank} owns no layers in smoke test")
    dist.barrier()
    for layer_id in range(layer_count):
        key, value = pool.get_kv_buffer(layer_id)
        assert value.data_ptr() == key.data_ptr()
        assert value.shape[-1] == pool.kv_lora_rank
        assert_tensor_value(
            f"dense kv rank={rank} layer={layer_id}", key, 10 + layer_id
        )
        index_buf = pool.get_index_k_with_scale_buffer(layer_id)
        assert_tensor_value(
            f"index rank={rank} layer={layer_id}", index_buf, 40 + layer_id
        )
    return pool.get_kv_size_bytes()


def run_turboquant_storage(torch, dist, device, layer_count):
    from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool

    pool = TurboQuantNSATokenToKVPool(
        size=128,
        page_size=64,
        kv_lora_rank=128,
        dtype=torch.bfloat16,
        qk_rope_head_dim=16,
        layer_num=layer_count,
        device=device,
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=144,
        start_layer=0,
        end_layer=layer_count,
        index_buf_size=128,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
        turboquant_mla_decode_num_splits=16,
    )
    rank = dist.get_rank()
    owned_layers = 0
    for layer_id in range(layer_count):
        if pool.layersplit_owns_layer(layer_id):
            owned_layers += 1
            pool.kv_buffer[layer_id].fill_(70 + layer_id)
    if owned_layers == 0:
        raise AssertionError(f"rank {rank} owns no TurboQuant layers in smoke test")
    dist.barrier()
    for layer_id in range(layer_count):
        kv = pool._get_layersplit_kv_buffer(layer_id)
        if kv.dtype != torch.uint8:
            raise AssertionError(f"turboquant storage dtype mismatch: {kv.dtype}")
        assert_tensor_value(
            f"turboquant kv rank={rank} layer={layer_id}", kv, 70 + layer_id
        )
    return pool.get_kv_size_bytes()


def main():
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("LayerSplit pool smoke requires CUDA.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(backend="nccl")
    try:
        patch_sglang_runtime(dist)
        layout = os.environ.get("LAYERSPLIT_LAYOUT", "contiguous")
        layer_count = smoke_layer_count(dist.get_world_size())
        dense_bytes = run_dense_pool(torch, dist, device, layer_count)
        tq_bytes = run_turboquant_storage(torch, dist, device, layer_count)
        dense_total = torch.tensor([dense_bytes], dtype=torch.float64, device=device)
        tq_total = torch.tensor([tq_bytes], dtype=torch.float64, device=device)
        dist.all_reduce(dense_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(tq_total, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            print(
                "LayerSplit distributed smoke passed: "
                f"world_size={dist.get_world_size()} "
                f"layers={layer_count} "
                f"layout={layout} "
                f"dense_bytes_sum={int(dense_total.item())} "
                f"tq_bytes_sum={int(tq_total.item())}"
            )
    finally:
        dist.destroy_process_group()


def test_layersplit_pool_smoke():
    import pytest

    pytest.importorskip("torch")
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        pytest.skip("run with torch.distributed.run and at least two CUDA ranks")
    main()


if __name__ == "__main__":
    main()
