# NCCLX Device Collectives

This page documents the optional torchcomms/NCCLX collective path in SGLang.
It is disabled by default. The default path keeps the existing
PyTorch/NCCL, PyNCCL, custom all-reduce, MSCCL++, and symmetric-memory routing.

NCCLX is described in the paper
[Collective Communication for 100k+ GPUs](https://arxiv.org/pdf/2510.20171).
The implementation used here is
[meta-pytorch/torchcomms](https://github.com/meta-pytorch/torchcomms), whose
public API is documented at
[meta-pytorch.org/torchcomms](https://meta-pytorch.org/torchcomms/main/index.html).

## Enabling NCCLX

Install a torchcomms build that includes the `ncclx` backend and matches the
active PyTorch ABI, then launch SGLang with:

```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --tp-size 8 \
  --device-collective-backend ncclx
```

Useful options:

| Argument | Purpose |
| --- | --- |
| `--device-collective-backend default` | Use the existing SGLang collective routing. This is the default. |
| `--device-collective-backend ncclx` | Route supported device collectives through torchcomms NCCLX. |
| `--torchcomms-ncclx-strict` | Fail startup if NCCLX cannot initialize. Without this flag, SGLang falls back to the default collective path and logs a warning. |
| `--torchcomms-ncclx-hints` | Pass torchcomms NCCLX hints as JSON or comma-separated `key=value` pairs. |
| `--enable-torchcomms-ncclx-rdma` | Initialize the NCCLX CUDA allocator registration hook for RDMA-capable transports. Enable this only after the host RDMA probe and a small NCCLX collective smoke test pass. |
| `--torchcomms-ncclx-abort-on-timeout` | Ask torchcomms NCCLX to abort the process on communicator timeout or error. Use this only in controlled production gates where fail-fast behavior is preferred over fallback. |

The NCCLX path is implemented at the `GroupCoordinator` layer. That keeps it
orthogonal to model-specific features such as LayerSplit, IndexCache, dense
TurboQuant, HiSparse, SMC-SD, Gated Attention, and GatedNorm.
When NCCLX initializes for a model-parallel group, SGLang skips the alternate
PyNCCL, custom all-reduce, MSCCL++, and symmetric-memory device communicators
for that group so the selected backend owns the measured collective path.

The SGLang integration exposes the torchcomms operations that are safe to call
from Python and delegates backend-specific availability to the installed
torchcomms build:

- SGLang-routed collectives: all-reduce, out-of-place all-reduce,
  reduce-scatter, all-gather, CP all-gather, broadcast, send, and recv.
- Additional torchcomms public collectives for hardening and future feature
  work: reduce, equal and variable all-to-all, scatter, gather, gather-single,
  barrier, split, batched P2P creation, hook registration, persistent
  all-gather where supported by the backend, RMA window creation, device
  transport handle access, communicator metadata, abort, and reconfigure.
- NCCLX backend-specific escape hatches: GPU-resident split-size all-to-all-v,
  dynamic all-to-all-v dispatch/combine, quantized reduce-scatter, and
  communicator dumps when the installed torchcomms build exposes those methods.

NCCLX paper features such as one-sided RMA, GPU-resident metadata collectives,
fault-tolerant reconfiguration, and backend diagnostics are exposed through
torchcomms APIs where they are public. SGLang does not replace unrelated
application protocols, such as the DeepEP MoE token dispatcher, with NCCLX
internals; those paths remain independently configurable.

## RDMA

NCCLX can use the RDMA-capable transport paths exposed by torchcomms and NCCLX
when the host has the required network and driver stack. The SGLang flag
`--enable-torchcomms-ncclx-rdma` initializes the torchcomms NCCLX CUDA caching
allocator hook so PyTorch GPU allocations can be registered for NCCLX transport.

Before running production tests, capture a host probe:

```bash
python3 scripts/playground/bench_ncclx_collectives.py --probe-only
```

On RDMA-capable systems the probe should show InfiniBand/RoCE devices from
`ibv_devices`, `ibstat -l`, or `ibdev2netdev`, and the intended NCCL/NCCLX
environment variables. On systems without RDMA devices, this probe still
records the absence of RDMA so the run is reproducible.

Do not enable `--enable-torchcomms-ncclx-rdma` on hosts whose probe does not
show the RDMA stack. Keep NCCLX enabled without the allocator hook on those
hosts, then rerun the same commands with
`--enable-torchcomms-ncclx-rdma` when RDMA-capable target hardware is available.

## Collective Microbenchmark

Use `torchrun` to compare default collectives against NCCLX with the same tensor
sizes:

```bash
torchrun --nproc-per-node 8 \
  scripts/playground/bench_ncclx_collectives.py \
  --backend both \
  --ops all_reduce all_gather reduce_scatter all_to_all_single all_to_all_v_single device_alltoallv_single reduce_scatter_quantized \
  --sizes-mb 1 8 64 256 \
  --torchcomms-ncclx-abort-on-timeout \
  --include-probe \
  --output /tmp/ncclx_collectives.json
```

The output includes per-rank latency and CUDA peak allocated/reserved memory for
the selected operations. The default operation set covers `all_reduce`,
`all_gather`, `reduce_scatter`, and equal-split `all_to_all_single`; the command
above adds variable all-to-all and NCCLX's GPU-resident split-size all-to-all.
Use this before full serving tests to verify that torchcomms NCCLX is available
and that the RDMA environment is visible inside the runtime.
Add `--enable-torchcomms-ncclx-rdma` only on RDMA-capable hosts after the probe
and a no-RDMA NCCLX smoke test pass.

## Production P/D Matrix

The production validation target compares `default` against `ncclx` with the
same serving stack:

- 4 GPU prefill worker and 4 GPU decode worker.
- Disaggregated prefill/decode.
- LayerSplit context parallelism on the prefill worker.
- Expert parallelism and DP attention on the decode worker.
- IndexCache, dense 2.5-bit TurboQuant, HiSparse, and decode-side SMC-SD.
- Input/output matrix: `8192/1k`, `16k/1k`, `16k/4k`, `32k/4k`, `64k/4k`.

The helper script emits and runs that matrix:

```bash
MODEL_PATH=BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
SMC_DRAFT_MODEL_PATH=/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
BACKENDS="default ncclx" \
scripts/playground/run-ncclx-pd-matrix.sh
```

The matrix script leaves the NCCLX RDMA allocator hook disabled by default. Set
`ENABLE_TORCHCOMMS_NCCLX_RDMA=1` only on hosts whose RDMA probe and NCCLX
collective smoke test have passed.

The default topology is:

| Worker | Default topology | Notes |
| --- | --- | --- |
| Prefill | `PREFILL_TP_SIZE=4`, `PREFILL_DP_SIZE=1`, `PREFILL_CP_SIZE=4` | LayerSplit CP uses `round-robin-split` and requires `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` for P/D transfer. |
| Decode | `DECODE_TP_SIZE=4`, `DECODE_DP_SIZE=4`, `DECODE_EP_SIZE=4` | DP attention, DeepEP MoE A2A, HiSparse, IndexCache, dense TurboQuant, and SMC-SD are enabled. |

The script exports `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1` by default
so every LayerSplit CP owner can transfer its owned KV layers to the decode
worker.

Set `DRY_RUN=1` to inspect the exact launch and benchmark commands without
starting servers.

## Current Validation Status

The SGLang integration is verified for argument wiring, backend selection,
torchcomms API coverage, production P/D command construction, and reduced
production serving on `instance-20260415-161450`.

That H200 host was rebuilt with 8 GPUs, active RDMA devices `mlx5_0` through
`mlx5_7`, source-built `torchcomms` with `ncclx` backend support, and
source-built `sgl-kernel` coverage for the custom G1 Gated Attention and BF16
GatedNorm model path. The four-rank collective gate passed for both the default
backend and NCCLX with the RDMA allocator hook enabled. The 1 MiB operations
were faster with NCCLX, the 16 MiB operations were effectively equal except for
one small all-to-all noise regression, and CUDA peak allocated/reserved memory
did not increase.

Reduced production P/D serving smokes also passed for both `default` and
`ncclx` with LayerSplit prefill, HiSparse, IndexCache, dense TurboQuant,
decode-side SMC-SD, DP attention, NIXL transfer, Gated Attention, and GatedNorm.
Those H200 smokes used a tiny FP8 proxy model and set
`DECODE_MOE_A2A_BACKEND=none` because the DeepEP/NVSHMEM low-latency transport
fails before SGLang initializes with `Unable to create ah`,
`create DCT share err`, and `nvshmem setup connections failed`.
`nvidia-peermem` also fails to load on this image.

The full target acceptance matrix still needs to be rerun on the production
B200 or equivalent image with working DeepEP/NVSHMEM and the target NVFP4
checkpoint. Keep `--torchcomms-ncclx-strict` enabled for that gate so an NCCLX
fallback cannot be counted as a successful NCCLX run.

The acceptance gate for NCCLX is stricter than a smoke test:

- NCCLX must show lower CUDA communication memory on every matrix point.
- NCCLX must not regress end-to-end throughput on any matrix point.
- Any throughput gain should be reported together with TTFT and peak memory.
- If NCCLX is unavailable on the current hardware, record the probe output and
  rerun the same matrix when the RDMA-capable target hardware is available.

## Failure Modes

If `--torchcomms-ncclx-strict` is not set and torchcomms cannot import, the
wheel does not include `ncclx`, or NCCLX cannot initialize a subgroup, SGLang
logs the failure and falls back to the default collective path. Use strict mode
for benchmark gates so a fallback cannot be mistaken for an NCCLX result.
If torchcomms imports fail with unresolved C++ symbols, rebuild or install a
torchcomms wheel matched to the exact PyTorch version in the runtime image
before running the NCCLX acceptance matrix.

For source builds against PyTorch 2.9-era wheels, current torchcomms main may
also need its Flight Recorder filesystem include updated from the removed
`c10/util/FileSystem.h` header to standard C++ `<filesystem>`. Treat that as a
third-party runtime build compatibility patch; it is not part of the SGLang
NCCLX integration itself.
