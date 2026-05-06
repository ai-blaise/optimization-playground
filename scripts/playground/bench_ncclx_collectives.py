#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


@contextlib.contextmanager
def _temp_env(**env_vars: Any):
    old = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_probe_command(cmd: List[str]) -> Dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd, "stdout": "", "stderr": ""}
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "available": True,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def rdma_probe() -> Dict[str, Any]:
    keys = [
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
        "NCCL_NET",
        "NCCL_NET_PLUGIN",
        "NCCL_DEBUG",
        "UCX_NET_DEVICES",
        "TORCHCOMM_BACKEND_LIB_PATH_NCCLX",
        "TORCHCOMM_NCCLX_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD",
    ]
    return {
        "host": socket.gethostname(),
        "env": {key: os.environ.get(key) for key in keys if key in os.environ},
        "commands": {
            "ibv_devices": _run_probe_command(["ibv_devices"]),
            "ibstat": _run_probe_command(["ibstat", "-l"]),
            "ibdev2netdev": _run_probe_command(["ibdev2netdev"]),
            "nvidia_smi_topo": _run_probe_command(["nvidia-smi", "topo", "-m"]),
            "nvidia_peermem": _run_probe_command(
                ["bash", "-lc", "lsmod | grep -E 'nvidia_peermem|nv_peer_mem' || true"]
            ),
            "ip_links": _run_probe_command(["ip", "-br", "link"]),
        },
    }


def _init_distributed():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
    return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")


def _parse_hints(raw_hints: str) -> Dict[str, str]:
    if not raw_hints:
        return {}

    raw_hints = raw_hints.strip()
    if raw_hints.startswith("{"):
        parsed = json.loads(raw_hints)
        if not isinstance(parsed, dict):
            raise ValueError("NCCLX hints JSON must be an object")
        return {str(key): str(value) for key, value in parsed.items()}

    hints = {}
    for item in raw_hints.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError("NCCLX hints must be JSON or key=value pairs")
        hints[key.strip()] = value.strip()
    return hints


def _new_ncclx_comm(device, name: str, hints: Dict[str, str], enable_rdma: bool):
    import torch.distributed as dist
    import torchcomms
    from torch.distributed import PrefixStore, distributed_c10d

    if enable_rdma:
        from torchcomms import ncclx

        ncclx.init_caching_allocator_hook()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    store = PrefixStore(name, distributed_c10d._get_default_store())
    with _temp_env(
        TORCHCOMM_RANK=rank,
        TORCHCOMM_SIZE=world_size,
        TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD="env",
    ):
        return torchcomms.new_comm(
            "ncclx",
            device,
            store=store,
            name=name,
            hints=hints or None,
        )


def _mb_to_numel(size_mb: int, dtype) -> int:
    import torch

    return max(1, size_mb * 1024 * 1024 // torch.tensor([], dtype=dtype).element_size())


def _time_op(fn, warmup: int, iters: int):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _bench_default(op: str, size_mb: int, dtype, warmup: int, iters: int, device):
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    numel = _mb_to_numel(size_mb, dtype)
    torch.cuda.reset_peak_memory_stats(device)

    if op == "all_reduce":
        tensor = torch.full((numel,), float(rank + 1), dtype=dtype, device=device)

        def run():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    elif op == "all_gather":
        tensor = torch.full((numel,), float(rank), dtype=dtype, device=device)
        output = torch.empty((numel * world_size,), dtype=dtype, device=device)

        def run():
            dist.all_gather_into_tensor(output, tensor)

    elif op == "reduce_scatter":
        tensor = torch.full(
            (numel * world_size,), float(rank + 1), dtype=dtype, device=device
        )
        output = torch.empty((numel,), dtype=dtype, device=device)

        def run():
            dist.reduce_scatter_tensor(output, tensor, op=dist.ReduceOp.SUM)

    else:
        raise ValueError(op)

    elapsed = _time_op(run, warmup, iters)
    return {
        "backend": "default",
        "op": op,
        "size_mb": size_mb,
        "elapsed_s": elapsed,
        "iters": iters,
        "avg_ms": elapsed * 1000 / iters,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _bench_ncclx(op: str, size_mb: int, dtype, warmup: int, iters: int, device, comm):
    import torch
    import torch.distributed as dist
    import torchcomms

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    numel = _mb_to_numel(size_mb, dtype)
    torch.cuda.reset_peak_memory_stats(device)

    if op == "all_reduce":
        tensor = torch.full((numel,), float(rank + 1), dtype=dtype, device=device)

        def run():
            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

    elif op == "all_gather":
        tensor = torch.full((numel,), float(rank), dtype=dtype, device=device)
        output = torch.empty((numel * world_size,), dtype=dtype, device=device)

        def run():
            comm.all_gather_single(output, tensor, async_op=False)

    elif op == "reduce_scatter":
        tensor = torch.full(
            (numel * world_size,), float(rank + 1), dtype=dtype, device=device
        )
        output = torch.empty((numel,), dtype=dtype, device=device)

        def run():
            comm.reduce_scatter_single(
                output, tensor, torchcomms.ReduceOp.SUM, async_op=False
            )

    else:
        raise ValueError(op)

    elapsed = _time_op(run, warmup, iters)
    return {
        "backend": "ncclx",
        "op": op,
        "size_mb": size_mb,
        "elapsed_s": elapsed,
        "iters": iters,
        "avg_ms": elapsed * 1000 / iters,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run_collective_bench(args) -> Dict[str, Any]:
    import torch
    import torch.distributed as dist

    device = _init_distributed()
    dtype = getattr(torch, args.dtype)
    backends = ["default", "ncclx"] if args.backend == "both" else [args.backend]
    results = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "device": str(device),
        "rdma_probe": rdma_probe() if args.include_probe else None,
        "results": [],
    }

    ncclx_comm = None
    if "ncclx" in backends:
        ncclx_comm = _new_ncclx_comm(
            device,
            f"bench_ncclx_{int(time.time())}",
            _parse_hints(args.torchcomms_ncclx_hints),
            args.enable_torchcomms_ncclx_rdma,
        )

    try:
        for size_mb in args.sizes_mb:
            for op in args.ops:
                if "default" in backends:
                    results["results"].append(
                        _bench_default(
                            op, size_mb, dtype, args.warmup, args.iters, device
                        )
                    )
                if "ncclx" in backends:
                    results["results"].append(
                        _bench_ncclx(
                            op,
                            size_mb,
                            dtype,
                            args.warmup,
                            args.iters,
                            device,
                            ncclx_comm,
                        )
                    )
    finally:
        if ncclx_comm is not None:
            ncclx_comm.finalize()
        dist.destroy_process_group()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--include-probe", action="store_true")
    parser.add_argument(
        "--backend", choices=["default", "ncclx", "both"], default="both"
    )
    parser.add_argument(
        "--ops", nargs="+", default=["all_reduce", "all_gather", "reduce_scatter"]
    )
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 8, 64, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--torchcomms-ncclx-hints", default="")
    parser.add_argument("--enable-torchcomms-ncclx-rdma", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.probe_only:
        payload = rdma_probe()
    else:
        payload = run_collective_bench(args)

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
