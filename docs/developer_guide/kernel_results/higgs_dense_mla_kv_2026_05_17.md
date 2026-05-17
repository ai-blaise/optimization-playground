# HIGGS Dense MLA KV B200 Result, 2026-05-17

## Scope

Branch: `codex/higgs-tensor-op-loop-20260517`

Commit: `86ace5044 Add HIGGS dense MLA KV sizing and benchmark coverage`

Hardware: 1x NVIDIA B200 on `root@31.22.104.123`, CUDA 12.8 driver stack.

This iteration fixed the production pool-capacity path for HIGGS dense MLA
2-bit KV cache and added HIGGS to the dense MLA KV candidate benchmark. It did
not promote a new CuTe tensor-op decode kernel; the existing scalar decode path
remains the target for the next optimization loop.

## Benchmark

Command:

```bash
PYTHONPATH=/root/work/op-kernel-higgs/python \
/root/work/optimization-playground/.venv/bin/python \
  benchmark/deepseek_v3/bench_dense_mla_kv_candidates.py \
  --num-tokens 512 \
  --selected-tokens 128 \
  --warmup 3 \
  --iters 10 \
  --json-output /root/agent-runs/higgs-dense-mla-kv-smoke.json
```

Results:

| Candidate | Bytes/token/layer | Store ms | Selected recover ms | Latent cosine mean | Rope exact |
| --- | ---: | ---: | ---: | ---: | --- |
| NVFP4 dense MLA | 324 | 0.016355 | 0.029978 | 0.995470 | false |
| TurboQuant dense MLA 2.5-bit | 274 | 0.007296 | 0.006256 | 0.950731 | true |
| HIGGS dense MLA 2-bit | 258 | 0.008886 | 0.005587 | 0.944873 | true |

## Direct Improvement

HIGGS uses 258 B/token/layer in this production NSA pool path, which is
16 B/token/layer smaller than TurboQuant dense MLA 2.5-bit (5.839% less
storage). In the same smoke shape, HIGGS selected recover is 1.1197x faster
than TurboQuant (`0.006256 / 0.005587`), while HIGGS store is 0.8210x the
TurboQuant store rate (`0.007296 / 0.008886`), so store is not improved in this
shape.

## Verification

```text
test/registered/unit/model_executor/test_pool_configurator.py: 20 passed
test/srt/test_quantization_config_dispatch.py: 33 passed
python/sglang/test/test_higgs_dense_2bit_kv.py
test/srt/test_higgs_dense_2bit_kv_integration.py: 15 passed
git diff --check: passed
```

## Caveat

The measured win here is pool sizing, memory footprint, and selected-recover
coverage. The HIGGS fused MLA decode kernel still needs a tensor-op/CuTe design
before this branch can claim decode-kernel throughput optimization.
