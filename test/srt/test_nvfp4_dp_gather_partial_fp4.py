"""Correctness + microbench for the iter5 FP4 allgather collective.

Tests ``sglang.srt.layers.dp_attention.dp_gather_partial_fp4`` — the
parallel NCCL allgather of (FP4 packed bytes, UE8M0 scale bytes) that
unlocks the iter4 SECONDARY communicator.py L1024 wire for the
DSv3.2-REAP DP=TP=8 + attn_tp_size=1 deploy config.

Reference (bit-exact target):

    # BF16 path (current iter4 SECONDARY, BF16 emit wasted because we
    # then run fp4_quantize on the gathered BF16):
    y_local_bf16 = fused_add_rmsnorm(x, residual, weight, eps)
    y_global_bf16 = all_gather_into_tensor(y_local_bf16)
    fp4_ref, sf_ref = scaled_fp4_quant_linear(y_global_bf16, gs)

    # iter5 path (BF16 emit unused after kernel, FP4 + SF allgathered
    # directly):
    y_local_bf16, fp4_local, sf_local = (
        fused_rmsnorm_to_fp4_and_bf16_linear(x, residual, weight, gs, eps)
    )
    fp4_global = all_gather_into_tensor(fp4_local)
    sf_global = all_gather_into_tensor(sf_local)

    assert fp4_global == fp4_ref   # 100%
    assert sf_global == sf_ref     # 100%

Uses 2 spare GPUs (cuda:6, cuda:7) under a torchrun multi-process
launcher so we never disturb the other workloads on cuda:0..5.

Usage:

    PYTHONPATH=/tmp/pyextra:python python3.11 -m torch.distributed.run \\
        --nproc_per_node=2 --master_port=29611 \\
        -- test/srt/test_nvfp4_dp_gather_partial_fp4.py [--bench]
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist


PROD_HIDDEN = 7168
DECODE_BATCHES_LOCAL = [1, 8, 16, 32, 64, 128, 256, 512]


def _suggest_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32).clamp_min_(1e-6)
    return (448.0 * 6.0 / amax).reshape(1)


def _setup_dist() -> Tuple[int, int, torch.device]:
    # Avoid disturbing other workloads on cuda:0..5; pin this test to
    # cuda:6 + cuda:7. The launcher must be torch.distributed.run with
    # --nproc_per_node=2; rank 0 lands on cuda:6, rank 1 on cuda:7.
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, "test designed for world_size=2"

    device_idx = 6 + rank
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, device


def _setup_pynccl(rank: int, world: int, device: torch.device):
    """Build a pynccl communicator on top of the existing default group.

    pynccl ncclAllGather is graph-capturable; torch.distributed's
    higher-level wrapper has stream/event guards that don't capture.
    Returns a ``PyNcclCommunicator`` ready for ``all_gather`` calls.

    PyNcclCommunicator asserts its bootstrap group is non-NCCL so the
    nccl id broadcast doesn't deadlock; create a sidecar gloo group for
    bootstrap.
    """
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    gloo_group = dist.new_group(ranks=list(range(world)), backend="gloo")
    return PyNcclCommunicator(group=gloo_group, device=device)


def _run_correctness(rank: int, world: int, device: torch.device) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm
    from sglang.jit_kernel.nvfp4 import (
        fused_rmsnorm_to_fp4_and_bf16_linear,
        scaled_fp4_quant_linear,
    )

    # Same seed across ranks for the weight tensor; different per-rank
    # for the hidden_states / residual.
    torch.manual_seed(0xCAFE + rank * 17)
    eps = 1e-6
    fail = 0

    for m_local in DECODE_BATCHES_LOCAL:
        m_global = m_local * world

        # Different per-rank input.
        x = torch.randn(
            m_local, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        ) * 0.5
        residual = torch.randn(
            m_local, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        ) * 0.2

        # Replicated weight + global scale across ranks (the input
        # global scale is the post-quant input scale of the next layer,
        # which is replicated weight metadata in deploy).
        gw_seed = torch.Generator(device=device).manual_seed(0xBEEF)
        weight = torch.empty(
            PROD_HIDDEN, dtype=torch.bfloat16, device=device
        ).normal_(generator=gw_seed) * 0.1 + 1.0
        # Global scale must also be replicated (the iter4 stash uses
        # the consumer's w13_input_scale_quant_slice cached on the
        # layer at load time). Use a fixed scalar across ranks.
        gs = torch.tensor([8.0], dtype=torch.float32, device=device)

        # === iter5 path (allgather FP4 + SF) ===
        x_iter5 = x.clone()
        r_iter5 = residual.clone()
        y_local_iter5, fp4_local, sf_local = (
            fused_rmsnorm_to_fp4_and_bf16_linear(
                x_iter5, r_iter5, weight, gs, eps
            )
        )
        # FP4: uint8 [m_local, n//2]
        # SF view: fp8_e4m3fn [m_local, n//16]; underlying int32 [m_local, (n//16)//4]
        sf_int32_local = sf_local.view(torch.int32)

        fp4_global = torch.empty(
            (m_global, PROD_HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
        )
        sf_int32_global = torch.empty(
            (m_global, sf_int32_local.shape[1]),
            dtype=torch.int32,
            device=device,
        )
        dist.all_gather_into_tensor(fp4_global, fp4_local)
        dist.all_gather_into_tensor(sf_int32_global, sf_int32_local)
        sf_global_iter5 = sf_int32_global.view(torch.float8_e4m3fn)

        # === Reference path (allgather BF16, then fp4_quantize on gathered) ===
        x_ref = x.clone()
        r_ref = residual.clone()
        fused_add_rmsnorm(x_ref, r_ref, weight, eps)
        # x_ref is now BF16 post-norm hidden_states.
        y_global_bf16 = torch.empty(
            (m_global, PROD_HIDDEN), dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(y_global_bf16, x_ref)
        fp4_ref, sf_ref = scaled_fp4_quant_linear(y_global_bf16, gs)

        # === Compare ===
        fp4_match = (fp4_global == fp4_ref).float().mean().item()
        sf_match = (
            sf_global_iter5.view(torch.int32) == sf_ref.view(torch.int32)
        ).float().mean().item()
        ok = (fp4_match == 1.0) and (sf_match == 1.0)
        if rank == 0:
            tag = "OK " if ok else "FAIL"
            print(
                f"[corr][rank0] m_local={m_local:4d} m_global={m_global:4d}  "
                f"FP4 match={fp4_match*100:6.2f}%  "
                f"SF match={sf_match*100:6.2f}%  {tag}"
            )
        if not ok:
            fail += 1

    if rank == 0:
        if fail:
            print(f"\n[corr] {fail} m_local size(s) failed bit-exact check.")
        else:
            print("\n[corr] all sizes bit-exact: 100% FP4 + 100% SF match.")


def _time_call(fn, *args, warmup=200, iters=1000, use_cuda_graph=False) -> float:
    # Eager mode bench. CUDA-graph capture of NCCL allgather is
    # supported via pynccl but the test imports raw torch.distributed
    # which goes through ncclx; mixing capture + nccl group barriers
    # across ranks here is brittle. Production captured-mode delta is
    # estimated from the iter4 SECONDARY result (extra collective at
    # graph-mode time = bytes-proportional fraction of BF16 allgather
    # plus ~1us launch per added collective).
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us/call


def _time_call_captured(fn, *args, warmup=50, iters=500) -> float:
    """Capture a CUDA graph of ``fn(*args)`` and time the replay loop.

    NCCL ops are captured under pynccl's ``change_state`` context — the
    test uses torch.distributed.all_gather_into_tensor which dispatches
    to the active pynccl_comm under sglang's GroupCoordinator. For the
    raw torch.distributed path used in this test we manually mark the
    capture region with ``change_state`` via a small monkey patch on
    the default group: see :func:`_run_bench_captured`.
    """
    # Warmup on default stream.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Capture.
    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g):
            fn(*args)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    # Replay.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def _run_bench(rank: int, world: int, device: torch.device) -> None:
    from sglang.jit_kernel.nvfp4 import (
        fused_rmsnorm_to_fp4_and_bf16_linear,
        scaled_fp4_quant_linear,
    )

    # Three configurations:
    #   (a) BF16-only:    iter4 SECONDARY graph baseline. Kernel emits
    #                     BF16 + FP4 + SF, but only BF16 is allgathered.
    #                     Consumer then calls fp4_quantize on the global
    #                     BF16. The FP4 + SF emit is wasted.
    #   (b) BF16 + post-gather fp4_quantize: same as (a) but with the
    #                     downstream fp4_quantize included — this is
    #                     the TRUE deploy baseline cost the iter5 wire
    #                     replaces. The FP4 stash from the local kernel
    #                     is discarded (current iter4 SECONDARY behavior).
    #   (c) iter5 wire:   BF16 + FP4 + SF allgather. The downstream
    #                     fp4_quantize is skipped because the gathered
    #                     stash matches what the consumer would compute.
    #
    # Delta = (b) - (c). Positive = iter5 wins.
    if rank == 0:
        print(
            f"\n[bench] hidden={PROD_HIDDEN}  dp={world}  "
            f"device=cuda:{device.index}\n"
            f"           |  (a) BF16-only  |  (b) BF16 + post-gather q  |"
            f"  (c) iter5 BF16+FP4+SF  |  delta (b)-(c)  |  speedup\n"
            f"-----------+-----------------+----------------------------+"
            f"------------------------+-----------------+----------"
        )

    torch.manual_seed(0xFEED + rank)
    eps = 1e-6

    for m_local in DECODE_BATCHES_LOCAL:
        m_global = m_local * world

        x = torch.randn(
            m_local, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        ) * 0.5
        residual = torch.randn(
            m_local, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        ) * 0.2
        weight = (
            torch.randn(PROD_HIDDEN, dtype=torch.bfloat16, device=device)
            * 0.1 + 1.0
        )
        gs = torch.tensor([8.0], dtype=torch.float32, device=device)

        # Pre-allocate the global buffers (matches the deploy path).
        y_global_bf16 = torch.empty(
            (m_global, PROD_HIDDEN), dtype=torch.bfloat16, device=device
        )
        fp4_global = torch.empty(
            (m_global, PROD_HIDDEN // 2), dtype=torch.uint8, device=device
        )
        sf_int32_global = torch.empty(
            (m_global, (PROD_HIDDEN // 16) // 4),
            dtype=torch.int32,
            device=device,
        )

        # Bench (a): BF16-only allgather — the iter4 SECONDARY graph
        # baseline (the kernel still emits FP4 + SF but they go unused).
        def _bf16_only():
            x_l = x.clone()
            r_l = residual.clone()
            y_l, _, _ = fused_rmsnorm_to_fp4_and_bf16_linear(
                x_l, r_l, weight, gs, eps
            )
            dist.all_gather_into_tensor(y_global_bf16, y_l)

        # Bench (b): true deploy baseline — BF16 allgather then run
        # fp4_quantize on the gathered BF16 (matches the
        # `_USE_SGL_NVFP4_QUANT_LINEAR` path in
        # compressed_tensors_w4a4_nvfp4_moe.py L375 when the stash is
        # absent). This is the cost the iter5 wire replaces.
        def _bf16_then_quant():
            x_l = x.clone()
            r_l = residual.clone()
            y_l, _, _ = fused_rmsnorm_to_fp4_and_bf16_linear(
                x_l, r_l, weight, gs, eps
            )
            dist.all_gather_into_tensor(y_global_bf16, y_l)
            scaled_fp4_quant_linear(y_global_bf16, gs)

        # Bench (c): iter5 wire — BF16 + FP4 + SF allgather. Same kernel
        # emit as (a)/(b), three collectives, downstream fp4_quantize
        # skipped.
        def _iter5():
            x_l = x.clone()
            r_l = residual.clone()
            y_l, fp4_l, sf_l = fused_rmsnorm_to_fp4_and_bf16_linear(
                x_l, r_l, weight, gs, eps
            )
            sf_int32_l = sf_l.view(torch.int32)
            dist.all_gather_into_tensor(y_global_bf16, y_l)
            dist.all_gather_into_tensor(fp4_global, fp4_l)
            dist.all_gather_into_tensor(sf_int32_global, sf_int32_l)

        # Isolate the collective surfaces so we are not dominated by
        # eager-mode kernel-emit overhead in the comparison; the kernel
        # emit cost is identical across (a)/(b)/(c) and is graph-
        # captured in deploy. The relevant question is: how much extra
        # do (FP4 + SF) collectives cost vs (post-gather fp4_quantize)?
        y_l_static = torch.empty_like(
            x, dtype=torch.bfloat16
        )
        fp4_l_static = torch.empty(
            m_local, PROD_HIDDEN // 2, dtype=torch.uint8, device=device
        )
        sf_int32_l_static = torch.empty(
            m_local, (PROD_HIDDEN // 16) // 4, dtype=torch.int32, device=device
        )

        def _collective_bf16():
            dist.all_gather_into_tensor(y_global_bf16, y_l_static)

        def _collective_bf16_then_quant():
            dist.all_gather_into_tensor(y_global_bf16, y_l_static)
            scaled_fp4_quant_linear(y_global_bf16, gs)

        def _collective_iter5():
            dist.all_gather_into_tensor(y_global_bf16, y_l_static)
            dist.all_gather_into_tensor(fp4_global, fp4_l_static)
            dist.all_gather_into_tensor(sf_int32_global, sf_int32_l_static)

        t_a = _time_call(_collective_bf16)
        t_b = _time_call(_collective_bf16_then_quant)
        t_c = _time_call(_collective_iter5)
        delta = t_b - t_c
        speedup = t_b / t_c if t_c > 0 else 0.0

        if rank == 0:
            print(
                f"  m={m_local:4d}  |  {t_a:11.2f}us  |  "
                f"{t_b:22.2f}us  |  "
                f"{t_c:18.2f}us  |  {delta:+11.2f}us  |  "
                f"{speedup:5.2f}x"
            )


def _run_bench_graph_captured(
    rank: int, world: int, device: torch.device
) -> None:
    """Same comparison as ``_run_bench`` but with pynccl directly under
    a captured CUDA graph. This is the deploy-relevant measurement
    because the iter4 PRIMARY / SECONDARY wires fire inside a captured
    decode graph; eager-mode launch overhead dominates the eager bench
    and obscures the bytes-on-the-wire trade-off.
    """
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant_linear

    pynccl_comm = _setup_pynccl(rank, world, device)

    if rank == 0:
        print(
            f"\n[bench-graph] hidden={PROD_HIDDEN}  dp={world}  "
            f"device=cuda:{device.index}  (CUDA-graph captured pynccl)\n"
            f"           |  (a) BF16-only  |  (b) BF16 + post-gather q  |"
            f"  (c) all 3 serial |  (d) iter5 PROD  |  (e) iter6 PROD  |"
            f"  iter5->iter6  |  baseline->iter6  |  speedup\n"
            f"-----------+-----------------+----------------------------+"
            f"-------------------+------------------+------------------+"
            f"---------------+-------------------+----------"
        )

    gs = torch.tensor([8.0], dtype=torch.float32, device=device)

    for m_local in DECODE_BATCHES_LOCAL:
        m_global = m_local * world

        # Static buffers — graph capture requires fixed addresses.
        y_l = torch.randn(
            m_local, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        )
        fp4_l = torch.empty(
            m_local, PROD_HIDDEN // 2, dtype=torch.uint8, device=device
        )
        sf_int32_l = torch.empty(
            m_local, (PROD_HIDDEN // 16) // 4, dtype=torch.int32, device=device
        )
        y_global = torch.empty(
            m_global, PROD_HIDDEN, dtype=torch.bfloat16, device=device
        )
        fp4_global = torch.empty(
            m_global, PROD_HIDDEN // 2, dtype=torch.uint8, device=device
        )
        sf_int32_global = torch.empty(
            m_global, (PROD_HIDDEN // 16) // 4, dtype=torch.int32, device=device
        )

        def _capture_and_time(fn) -> float:
            # Warm up the comm + kernel.
            with pynccl_comm.change_state(enable=True):
                for _ in range(10):
                    fn()
                torch.cuda.synchronize()

                g = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream()
                capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):
                    with torch.cuda.graph(g):
                        fn()
                torch.cuda.current_stream().wait_stream(capture_stream)
                torch.cuda.synchronize()

                # Time replay.
                iters = 2000
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    g.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1e3 / iters  # us/call

        def _bf16():
            pynccl_comm.all_gather(y_global, y_l)

        def _bf16_then_quant():
            pynccl_comm.all_gather(y_global, y_l)
            scaled_fp4_quant_linear(y_global, gs)

        def _iter5_serial():
            pynccl_comm.all_gather(y_global, y_l)
            pynccl_comm.all_gather(fp4_global, fp4_l)
            pynccl_comm.all_gather(sf_int32_global, sf_int32_l)

        def _iter5_prod():
            # Iter5 PRODUCTION wire: BF16 allgather fires as a separate
            # NCCL launch (via reg_all_gather_into_tensor in production;
            # raw pynccl_comm.all_gather here for the bench), then
            # FP4+SF fused under one ncclGroupStart/End.
            pynccl_comm.all_gather(y_global, y_l)
            pynccl_comm.group_start()
            pynccl_comm.all_gather(fp4_global, fp4_l)
            pynccl_comm.all_gather(sf_int32_global, sf_int32_l)
            pynccl_comm.group_end()

        def _iter6_all_grouped():
            # Iter6 PRODUCTION wire: ALL THREE (BF16 + FP4 + SF) fused
            # under one ncclGroupStart/End. Single NCCL launch instead
            # of iter5's two. This is the iter6 PRIMARY vector — see
            # dp_attention.py:dp_gather_partial_bf16_fp4_fused for the
            # production callsite implementation.
            pynccl_comm.group_start()
            pynccl_comm.all_gather(y_global, y_l)
            pynccl_comm.all_gather(fp4_global, fp4_l)
            pynccl_comm.all_gather(sf_int32_global, sf_int32_l)
            pynccl_comm.group_end()

        # Run barrier between captures so ranks stay in lockstep.
        dist.barrier()
        t_a = _capture_and_time(_bf16)
        dist.barrier()
        t_b = _capture_and_time(_bf16_then_quant)
        dist.barrier()
        t_c = _capture_and_time(_iter5_serial)
        dist.barrier()
        t_d = _capture_and_time(_iter5_prod)
        dist.barrier()
        t_e = _capture_and_time(_iter6_all_grouped)
        dist.barrier()

        # iter6 win = iter5_prod - iter6_all_grouped: the launch
        # overhead saved by lifting the BF16 leg into the same
        # ncclGroupStart/End block as FP4+SF.
        delta_5_to_6 = t_d - t_e
        # Cumulative iter5+iter6 vs the true deploy baseline (b):
        delta_b_to_6 = t_b - t_e
        speedup_b_to_6 = t_b / t_e if t_e > 0 else 0.0

        if rank == 0:
            print(
                f"  m={m_local:4d}  |  {t_a:11.2f}us  |  "
                f"{t_b:22.2f}us  |  "
                f"{t_c:11.2f}us  |  {t_d:11.2f}us  |  {t_e:11.2f}us  |"
                f"  {delta_5_to_6:+9.2f}us  |  {delta_b_to_6:+9.2f}us  |  "
                f"{speedup_b_to_6:5.2f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-graph", action="store_true")
    args = parser.parse_args()

    rank, world, device = _setup_dist()

    try:
        _run_correctness(rank, world, device)
        if args.bench:
            dist.barrier()
            _run_bench(rank, world, device)
        if args.bench_graph:
            dist.barrier()
            _run_bench_graph_captured(rank, world, device)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
