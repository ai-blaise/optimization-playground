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


## Round 2: Extend Small-Copy Threshold To 128 KiB

- Incumbent: `a59198170391a9cfe1c36e539949f279793f2448` (Round 0 accepted source; Round 1 was rejected and did not change source).
- Hotspot/profiler signal: CUDA event instrumentation showed all 128 KiB staged-copy cells were falling through to the C++ op's delegated `Tensor.copy_` path and paying an extra dispatch layer. The 128 KiB cells in Round 0 averaged 0.800x vs direct `Tensor.copy_` and 3.681 us in `cute_ms`.
- Candidate: raise `kSmallByteThreshold` from 116 KiB to 128 KiB so exactly-128 KiB payloads use the custom fixed-CTA copy kernel.
- Command:

```bash
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   python test/manual/layers/attention/nsa/bench_layersplit_stage.py   --load-local-extension --extension-name layersplit_cute_round2_threshold   --rows 64,128,256 --row-bytes 512,1024,2048   --warmup 50 --iters 1000
```

- Result: accepted. The 128 KiB cells improved from 0.800x to 1.048x vs direct `Tensor.copy_`; average `cute_ms` improved from 3.681 us to 2.849 us. Larger 256 KiB and 512 KiB delegated cells stayed near the incumbent and remain outside the custom threshold.
- Correctness: every benchmark cell checks `torch.equal(src, dst)`.
- Decision: accept. New incumbent: Round 2 threshold commit.


## Round 3: Direct Prefix `cudaMemcpyAsync` For Delegated Path

- Incumbent: `be1dbd3ab4283b602bbfa3811584fe65bca00f15` (Round 2 accepted 128 KiB threshold commit).
- Hotspot/profiler signal: after Round 2, payloads above 128 KiB still delegated through `Tensor.copy_` inside the C++ op and stayed below direct Python `copy_` because the op adds dispatch overhead before reaching the same copy primitive. The delegated path also copied the whole tensor rather than the `active_rows` prefix, which is wrong for larger staging buffers.
- Candidate: replace the delegated contiguous path with `cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), active_rows * row_bytes, cudaMemcpyDeviceToDevice, stream)` so the C++ op copies the active prefix directly.
- Command:

```bash
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   python test/manual/layers/attention/nsa/bench_layersplit_stage.py   --load-local-extension --extension-name layersplit_cute_round3_memcpy   --rows 64,128,256,512 --row-bytes 512,1024,2048,4096   --warmup 50 --iters 1000
/root/agent-runs/gpu_locked.sh env CUDA_VISIBLE_DEVICES=0   python test/manual/layers/attention/nsa/bench_layersplit_stage.py   --load-local-extension --extension-name layersplit_cute_round3_memcpy   --rows 64,128 --row-bytes 2048,4096 --padding-rows 1   --warmup 10 --iters 50
```

- Result: accepted. On overlapping >128 KiB delegated cells, speedup vs direct Python `copy_` improved from roughly 0.80x to 0.89x-0.93x, and the 262 KiB active-prefix correctness cells measured 1.34x-1.43x when Python `copy_` copied the same active prefix. This path is still not faster than direct `copy_` for all large contiguous payloads, so the win is primarily reduced delegated-path overhead plus correct active-prefix semantics.
- Correctness: every benchmark cell checks `torch.equal(src[:rows], dst[:rows])`; the `--padding-rows 1` run also verifies inactive rows remain zero.
- Decision: accept. New incumbent: Round 3 direct-prefix memcpy commit.
