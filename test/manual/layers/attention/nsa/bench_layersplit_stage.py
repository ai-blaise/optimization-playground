import argparse
import json
import os
from pathlib import Path


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def load_local_extension(repo_root, name):
    from torch.utils.cpp_extension import load

    source = repo_root / "sgl-kernel/csrc/kvcacheio/layersplit_cute.cu"
    build_dir = Path(os.environ.get("LAYERSPLIT_EXT_BUILD_DIR", "/tmp/layersplit_ext"))
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(source)],
        extra_include_paths=[
            str(repo_root / "sgl-kernel/include"),
            str(repo_root / "sgl-kernel/csrc"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_100,code=sm_100",
        ],
        extra_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=False,
        is_python_module=False,
    )


def time_ms(torch, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,4,8,16,32,64,128,256")
    parser.add_argument("--row-bytes", default="288,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--load-local-extension", action="store_true")
    parser.add_argument("--extension-name", default="layersplit_cute_bench")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LayerSplit stage benchmark requires CUDA.")
    repo_root = Path(__file__).resolve().parents[5]
    if args.load_local_extension:
        load_local_extension(repo_root, args.extension_name)
    else:
        import sgl_kernel  # noqa: F401

    op = torch.ops.layersplit_cute.stage_for_broadcast
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    for rows in parse_int_list(args.rows):
        for row_bytes in parse_int_list(args.row_bytes):
            src = torch.randint(
                0, 256, (rows, row_bytes), dtype=torch.uint8, device=device
            )
            dst = torch.empty_like(src)

            def copy_run():
                dst.copy_(src)

            def cute_run():
                op(src, dst, rows, row_bytes)

            copy_ms = time_ms(torch, copy_run, args.warmup, args.iters)
            cute_ms = time_ms(torch, cute_run, args.warmup, args.iters)
            if not torch.equal(src, dst):
                raise AssertionError(
                    f"LayerSplit stage mismatch for rows={rows}, row_bytes={row_bytes}"
                )
            result = {
                "benchmark_kind": "layersplit_stage",
                "rows": rows,
                "row_bytes": row_bytes,
                "total_bytes": rows * row_bytes,
                "copy_ms": copy_ms,
                "cute_ms": cute_ms,
                "speedup_vs_copy": copy_ms / max(cute_ms, 1e-12),
                "extension": "local" if args.load_local_extension else "installed",
            }
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
