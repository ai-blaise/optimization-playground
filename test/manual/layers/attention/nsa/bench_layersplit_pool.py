import argparse
import json
import os


class RuntimeArgs:
    nsa_prefill_cp_kv_storage_mode = "layersplit"

    def __init__(self, layout):
        self.nsa_prefill_cp_layersplit_layout = layout


class Group:
    def __init__(self, dist):
        self._dist = dist

    @property
    def world_size(self):
        return self._dist.get_world_size()

    def broadcast(self, tensor, src=0):
        self._dist.broadcast(tensor, src=src)
        return tensor


def patch_sglang_runtime(dist, layout):
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.server_args as server_args

    dp_attention.get_attention_cp_rank = lambda: dist.get_rank()
    dp_attention.get_attention_cp_size = lambda: dist.get_world_size()
    dp_attention.get_attention_cp_group = lambda: Group(dist)
    server_args.get_global_server_args = lambda: RuntimeArgs(layout)


def parse_matrix(value):
    cells = []
    for cell in value.split(","):
        input_len, output_len = cell.lower().split("x", 1)
        cells.append((parse_count(input_len), parse_count(output_len)))
    return cells


def parse_count(value):
    value = value.strip().lower()
    if value.endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def make_pool(torch, device, storage, input_len, layer_count):
    if storage == "dense":
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        return NSATokenToKVPool(
            size=input_len,
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
            index_buf_size=input_len,
        )

    from sglang.srt.mem_cache.memory_pool import TurboQuantNSATokenToKVPool

    return TurboQuantNSATokenToKVPool(
        size=input_len,
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
        index_buf_size=input_len,
        turboquant_dense_kv_preset="latent_2p5bit_nc",
        turboquant_execution_mode="fused_decode",
        turboquant_mla_decode_num_splits=16,
    )


def fill_owned_layers(pool, layer_count):
    owned_layers = 0
    for layer_id in range(layer_count):
        if not pool.layersplit_owns_layer(layer_id):
            continue
        owned_layers += 1
        pool.kv_buffer[layer_id].fill_(layer_id + 1)
        pool.index_k_with_scale_buffer[layer_id].fill_(layer_id + 17)
    return owned_layers


def touch_layers(pool, layer_count, storage):
    for layer_id in range(layer_count):
        if storage == "dense":
            pool.get_kv_buffer(layer_id)
        else:
            pool._get_layersplit_kv_buffer(layer_id)
        pool.get_index_k_with_scale_buffer(layer_id)


def benchmark_cell(torch, dist, device, args, input_len, output_len):
    pool = make_pool(torch, device, args.storage, input_len, args.layer_count)
    owned_layers = fill_owned_layers(pool, args.layer_count)
    if owned_layers == 0:
        raise RuntimeError(f"rank {dist.get_rank()} owns no layers")

    dist.barrier()
    for _ in range(args.warmup):
        touch_layers(pool, args.layer_count, args.storage)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        touch_layers(pool, args.layer_count, args.storage)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / args.iters
    latency = torch.tensor([elapsed_ms], dtype=torch.float64, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)

    bytes_per_rank = torch.tensor(
        [pool.get_kv_size_bytes()], dtype=torch.float64, device=device
    )
    dist.all_reduce(bytes_per_rank, op=dist.ReduceOp.SUM)
    del pool
    torch.cuda.empty_cache()

    avg_ms = float(latency.item())
    return {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "storage": args.storage,
        "layout": args.layout,
        "cp_size": dist.get_world_size(),
        "layer_count": args.layer_count,
        "avg_latency_ms": avg_ms,
        "tokens_per_second": input_len / max(avg_ms / 1000.0, 1e-9),
        "within_latency_budget": avg_ms <= args.latency_budget_ms,
        "kv_bytes_sum": int(bytes_per_rank.item()),
    }


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        default="8192x1k,16kx1k,32kx1k,64kx1k,128kx1k",
    )
    parser.add_argument(
        "--storage",
        choices=("turboquant", "dense"),
        default="turboquant",
    )
    parser.add_argument(
        "--layout",
        choices=("interleaved", "contiguous"),
        default="contiguous",
    )
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--latency-budget-ms", type=float, default=2000.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("LayerSplit benchmark requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(backend="nccl")
    try:
        if args.layer_count < dist.get_world_size():
            raise ValueError("--layer-count must be at least WORLD_SIZE")
        patch_sglang_runtime(dist, args.layout)
        results = []
        for input_len, output_len in parse_matrix(args.matrix):
            result = benchmark_cell(torch, dist, device, args, input_len, output_len)
            if dist.get_rank() == 0:
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
        if dist.get_rank() == 0:
            max_latency_ms = max(result["avg_latency_ms"] for result in results)
            min_tokens_per_second = min(
                result["tokens_per_second"] for result in results
            )
            total_input_tokens = sum(result["input_tokens"] for result in results)
            total_latency_ms = sum(result["avg_latency_ms"] for result in results)
            aggregate_tokens_per_second = total_input_tokens / max(
                total_latency_ms / 1000.0, 1e-9
            )
            max_kv_bytes_sum = max(result["kv_bytes_sum"] for result in results)
            all_within_budget = int(
                all(result["within_latency_budget"] for result in results)
            )
            print(
                "layersplit_summary "
                f"max_latency_ms={max_latency_ms:.6f} "
                f"min_tokens_per_second={min_tokens_per_second:.6f} "
                f"aggregate_tokens_per_second={aggregate_tokens_per_second:.6f} "
                f"max_kv_bytes_sum={max_kv_bytes_sum} "
                f"all_within_latency_budget={all_within_budget}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
