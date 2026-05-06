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
| `--enable-torchcomms-ncclx-rdma` | Initialize the NCCLX CUDA allocator registration hook for RDMA-capable transports. |

The NCCLX path is implemented at the `GroupCoordinator` layer. That keeps it
orthogonal to model-specific features such as LayerSplit, IndexCache, dense
TurboQuant, HiSparse, SMC-SD, Gated Attention, and GatedNorm.
When NCCLX initializes for a model-parallel group, SGLang skips the alternate
PyNCCL, custom all-reduce, MSCCL++, and symmetric-memory device communicators
for that group so the selected backend owns the measured collective path.

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

## Collective Microbenchmark

Use `torchrun` to compare default collectives against NCCLX with the same tensor
sizes:

```bash
torchrun --nproc-per-node 8 \
  scripts/playground/bench_ncclx_collectives.py \
  --backend both \
  --sizes-mb 1 8 64 256 \
  --enable-torchcomms-ncclx-rdma \
  --include-probe \
  --output /tmp/ncclx_collectives.json
```

The output includes per-rank latency and CUDA peak allocated/reserved memory for
`all_reduce`, `all_gather`, and `reduce_scatter`. Use this before full serving
tests to verify that torchcomms NCCLX is available and that the RDMA environment
is visible inside the runtime.

## Production P/D Matrix

The production validation target compares `default` against `ncclx` with the
same serving stack:

- 4 GPU prefill worker and 4 GPU decode worker.
- Disaggregated prefill/decode.
- LayerSplit context parallelism on the prefill worker.
- Expert parallelism on the decode worker.
- DP attention on both workers.
- IndexCache, dense 2.5-bit TurboQuant, HiSparse, and decode-side SMC-SD.
- Input/output matrix: `8192/1k`, `16k/1k`, `16k/4k`, `32k/4k`, `64k/4k`.

The helper script emits and runs that matrix:

```bash
MODEL_PATH=BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1 \
SMC_DRAFT_MODEL_PATH=/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
BACKENDS="default ncclx" \
scripts/playground/run-ncclx-pd-matrix.sh
```

Set `DRY_RUN=1` to inspect the exact launch and benchmark commands without
starting servers.

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
