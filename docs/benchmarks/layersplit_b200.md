# LayerSplit B200 Benchmark Notes

## Environment

- VM: Prime Intellect B200, `root@31.22.104.123`
- GPU: NVIDIA B200, CUDA toolkit 12.8, PyTorch CUDA 12.8 runtime
- Branch base: `origin/main` at `44a6d42a9101972191f5b5aca5c32b643922b572`
- GPU isolation: commands run under `/root/agent-runs/gpu_locked.sh` with `CUDA_VISIBLE_DEVICES=0` unless noted
- Privacy: synthetic tensors only; no prompts, token IDs, or request payloads

## Round 0: Package Existing LayerSplit Stage Kernel

- Incumbent: `44a6d42a9101972191f5b5aca5c32b643922b572` (`origin/main`). The `layersplit_cute.cu` source existed but was absent from `sgl-kernel/CMakeLists.txt`, so a clean `sgl_kernel` build did not package `torch.ops.layersplit_cute.stage_for_broadcast`.
- Candidate: add `csrc/kvcacheio/layersplit_cute.cu` to the `sgl-kernel` source list, keep the existing kernel threshold behavior, add a packaging unit test, and add `test/manual/layers/attention/nsa/bench_layersplit_stage.py` for direct B200 stage-copy measurement.
- Hotspot/profiler signal: CUDA event instrumentation in the new benchmark showed the custom stage path is launch-floor dominated but faster than `Tensor.copy_` for intended small staging payloads. Direct IKP instrumentation is planned for a later round; this round establishes packaging and deployability.
- Command:

```bash
cd /root/work/op-kernel-layersplit
. /root/.cargo/env
source /root/work/optimization-playground/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/work/op-kernel-layersplit/python:${PYTHONPATH:-}
export MAX_JOBS=2
export LAYERSPLIT_EXT_BUILD_DIR=/tmp/layersplit_ext_round0
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   python test/manual/layers/attention/nsa/bench_layersplit_stage.py   --load-local-extension --extension-name layersplit_cute_round0   --rows 1,2,4,8,16,32,64,128,256   --row-bytes 288,512,1024,2048 --warmup 50 --iters 1000
```

- Result: for payloads at or below the kernel's 116 KiB small-copy threshold, `layersplit_cute` averaged 1.058x faster than `Tensor.copy_` across 30 cells; range 1.042x to 1.088x. For larger payloads, the current C++ op delegates to `copy_` and is slower because of the extra dispatch layer; those cells averaged 0.802x and are not promoted as runtime evidence.
- Correctness: every benchmark cell checks `torch.equal(src, dst)`.
- Decision: accepted as deployability and benchmark infrastructure. New incumbent: Round 0 commit.

## Rejected Candidate: Python Runtime Helper Dispatch

- Incumbent: Round 0 source candidate before commit.
- Candidate: call `torch.ops.layersplit_cute.stage_for_broadcast` from `MLATokenToKVPool.prefetch_layersplit_kv_buffer` owner-side staging, falling back to `Tensor.copy_` for unaligned or unavailable-op cases.
- Command: same environment as Round 0 with a temporary `--include-helper` benchmark variant.
- Result: helper dispatch measured 4.03-4.88 us on representative 32 KiB-512 KiB staged buffers, while direct `Tensor.copy_` was about 2.94-3.04 us.
- Decision: rejected. The source helper was removed; no runtime behavior change is kept.


## Round 1: Dynamic Small-Copy CTA Count

- Incumbent: `a59198170391a9cfe1c36e539949f279793f2448` (Round 0 accepted packaging commit).
- Hotspot/profiler signal: direct IKP trace probe in `/root/agent-runs/layersplit_ikp_probe.cu` showed the fixed 148-CTA small-copy launch leaves most warps idle on small staged buffers. For 1x288 bytes, fixed launch used 1 active warp and 1183 idle warps; a one-block dynamic launch would use 1 active warp and 7 idle warps. For 64x1024 bytes, fixed launch used 128 active warps and 1056 idle warps; a 16-block dynamic launch would use 128 active warps and 0 idle warps.
- Candidate: choose `min(148, ceil(num_vecs / 256))` CTAs for the 16-byte and 8-byte small-copy kernels.
- Command:

```bash
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   /root/agent-runs/layersplit_ikp_probe --rows=1 --row-bytes=288 --blocks=148   --out=/root/agent-runs/layersplit_ikp_fixed_1x288.json
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   python test/manual/layers/attention/nsa/bench_layersplit_stage.py   --load-local-extension --extension-name layersplit_cute_round1_dynamic   --rows 1,2,4,8,16,32,64,128,256   --row-bytes 288,512,1024,2048 --warmup 50 --iters 1000
```

- Result: rejected. On the same B200 matrix, <=116 KiB payload average changed from 1.058x to 1.054x vs `Tensor.copy_`, and average `cute_ms` regressed from 2.801 us to 2.850 us. Larger delegated payloads were also slightly worse. The fixed 148-CTA launch appears to trade idle warps for lower latency through broad SM residency; the benchmark remains launch-floor dominated.
- Correctness: benchmark cells checked `torch.equal(src, dst)`.
- Decision: reject. Source reverted; incumbent remains `a59198170391a9cfe1c36e539949f279793f2448`.
