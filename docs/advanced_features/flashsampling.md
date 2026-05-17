# FlashSampling

FlashSampling is an optional decode-time sampler that fuses exact sampling into
the LM-head matmul. Instead of materializing the full logits tensor in HBM, it
computes logits tile by tile, applies the same greedy or Gumbel-max sampling
rule on chip, writes per-tile winners, and reduces those winners to token ids.

The implementation is based on the FlashSampling paper and reference project:

- [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854)
- [FlashSampling/FlashSampling](https://github.com/FlashSampling/FlashSampling)

## Enabling

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-1.7B \
  --enable-flashsampling
```

FlashSampling is opt-in and falls back to the normal logits path when a request
batch is not eligible.

## Kernel Providers

```bash
--flashsampling-provider triton   # persistent fused kernel (default)
--flashsampling-provider target   # non-persistent kernel for TP-sharded shapes
--flashsampling-provider auto     # selects triton
```

The `target` provider uses a non-persistent grid-launch kernel optimized for
shapes where the total number of (V, H) tiles fits in a single SM wave
(tiles <= NUM_SMS). This is the typical case for TP-sharded vocab sizes
(e.g. V=16160 on TP=8 DeepSeek-V3.2-REAP). On Blackwell (SM100/B200),
`target` dispatches to the Blackwell-tuned variant with adaptive pipeline
stages; older CUDA devices use the generic target kernel. When tiles exceed
NUM_SMS, the target provider falls back to the persistent kernel automatically.

IKP-validated performance on H200 (V=16160, D=7168, TP=8):
- Kernel-only: 0.061ms (3.80 TB/s, 79% peak) vs persistent: 0.086ms (2.68 TB/s)
- End-to-end with reduce: 27-34% faster at BS=1..64
- Bit-exact greedy match at all batch sizes

## Supported Batches

The fast path currently supports decode batches with:

- greedy sampling, or uniform-temperature sampling with `top_k=-1` and
  `top_p=1.0`
- BF16 or FP16 unquantized LM-head weights
- no custom logits processor, grammar mask, vocab mask, penalties, logit bias,
  seeded per-request sampling, min-p, top-p, top-k, final logit softcapping,
  logprob return, LoRA LM-head adapter, or quantized LM head

Prefill and extend batches continue through the normal logits path.

## Batch Gates

The default minimum decode batch size is 128:

```bash
--flashsampling-min-batch-size 128
```

This conservative default comes from H200 serving validation on Qwen3-1.7B:
kernel microbenchmarks showed large wins at batch 128 and 256, while
end-to-end serving at concurrency 64 was not a stable output-throughput win.
Keeping the lower batch sizes on the normal logits path avoids that regression
while preserving the high-batch win.

Use the gates below for A/B tests:

```bash
--flashsampling-min-batch-size 1
--flashsampling-max-batch-size 256
```

## CUDA Graphs and Warmup

When FlashSampling and CUDA graphs are both enabled, SGLang captures separate
graph variants for eligible FlashSampling decode batches. The default warmup
covers the Triton batch-shape buckets used by the fused kernel:

```bash
--flashsampling-warmup-batch-sizes 1 32 64
```

Passing `--flashsampling-warmup-batch-sizes` with no values disables this warmup.

## Tensor Parallel and DP Attention

For tensor parallel LM heads, FlashSampling runs over the active sampling group
and reduces local per-rank winners. With DP attention enabled, the server
enables DP LM-head sharding so sampling can avoid materializing full logits
outside the attention tensor-parallel group.

## Validation Harness

The paper-style A/B harness compares the normal sampler to FlashSampling across
request concurrency levels using identical prompts and request bodies:

```bash
OUT_DIR=/tmp/sglang_flashsampling_paper_ab \
MODELS='Qwen/Qwen3-1.7B' \
CONCURRENCIES='1 2 4 8 16 32 64 128 256' \
NUM_RUNS=3 \
WARMUP_REQUESTS=concurrency \
FLASHSAMPLING_MIN_BATCH_SIZE=128 \
scripts/playground/run-flashsampling-paper-ab.sh
```

To compare the Blackwell target provider explicitly, add:

```bash
FLASHSAMPLING_PROVIDER=target
```

For Qwen3-8B paper-style validation on H200, use the paper gate rather than the
conservative production gate:

```bash
MODELS='Qwen/Qwen3-8B' \
CONCURRENCIES='1 2 4 8 16 32 64' \
NUM_RUNS=5 \
WARMUP_REQUESTS=concurrency \
FLASHSAMPLING_MIN_BATCH_SIZE=1 \
FLASHSAMPLING_MAX_BATCH_SIZE=128 \
DISABLE_PIECEWISE_CUDA_GRAPH=1 \
scripts/playground/run-flashsampling-paper-ab.sh
```

On `instance-20260415-161450` with Qwen3-8B and output length 256, that harness
showed median TPOT improvements of 4.08% to 6.30% across concurrency 1 to 64,
with median output throughput improving at every tested concurrency. Piecewise
CUDA graph was disabled because the baseline server hit a FusedAddRMSNorm graph
capture issue before FlashSampling was enabled; ordinary CUDA graph capture
remained enabled.

Qwen3-32B was also validated on the same H200 VM as a larger paper model. The
paper reports low-single-digit Qwen3-32B gains because attention and FFN dominate
decode time at that size. With the same AIME prompt set, output length 256,
ordinary CUDA graphs enabled, and piecewise CUDA graph disabled, the five-run
matrix showed lower median TPOT and higher output throughput at every tested
concurrency:

| Concurrency | Baseline TPOT (ms) | FlashSampling TPOT (ms) | TPOT gain |
| ----------- | ------------------ | ----------------------- | --------- |
| 1 | 18.0475 | 17.8713 | 0.98% |
| 2 | 18.4875 | 18.2819 | 1.11% |
| 4 | 18.6956 | 18.5073 | 1.01% |
| 8 | 18.9468 | 18.7237 | 1.18% |
| 16 | 19.2199 | 18.9511 | 1.40% |
| 32 | 20.4078 | 20.1787 | 1.12% |
| 64 | 22.5932 | 21.4922 | 4.87% |

For paper-style sweeps that force `--flashsampling-min-batch-size 1`, explicit
warmup buckets can reduce high-concurrency first-use overhead:

```bash
FLASHSAMPLING_WARMUP_BATCH_SIZES='1 2 4 8 16 32 64 128'
```

On Qwen3-32B this produced a 3-run candidate-only median of 21.4619 ms at
concurrency 64, a 5.01% TPOT gain over the same baseline median. It was neutral
to slightly mixed at lower concurrencies, so it remains an A/B tuning knob rather
than a production default.

Direct IKP kernel profiling for the Qwen3-32B shape showed the fused kernel is
substantially faster in isolation: 10.44% to 34.14% faster than dense
matmul-plus-sampling for batch sizes 1 through 128. The smaller end-to-end gain
is expected on 32B because the sampler is a smaller fraction of decode TPOT.

The IKP serving profiler captures kernel-level traces for the same baseline and
FlashSampling variants:

```bash
OUT_DIR=/tmp/sglang_flashsampling_serving_ikp \
MODEL=Qwen/Qwen3-1.7B \
CONCURRENCY=128 \
WARMUP_REQUESTS=concurrency \
scripts/playground/profile-flashsampling-serving-ikp.sh
```

### B200 Target-Provider Optimization Log

Hardware: 1x NVIDIA B200 (SM100), `CUDA_VISIBLE_DEVICES=0`. Shape:
`V=16160`, `D=7168`, BF16 weights/hidden states, greedy sampling, 20 warmup
iterations and 200 timed iterations unless noted. Correctness means sampled ids
match dense BF16 matmul argmax. Benchmarks used `/root/agent-runs/gpu_locked.sh`.

Hotspot attribution: IKP imported the Nsight trace
`/root/agent-runs/nsys-flashsampling-target-h64.nsys-rep` into
`/root/agent-runs/ikp-flashsampling-target-h64/nsys_kernels.json`. Matmul/sample
work dominates: `flashsample_blackwell_kernel` was 619.039 us total across 14
calls (44.217 us mean); `_local_reduce_samples_kernel` was 32.992 us total
(2.357 us mean). Optimization rounds therefore targeted the Blackwell matmul
kernel schedule, not the local reduction.

Round 0 incumbent -> candidate -> decision:

- Incumbent: `--flashsampling-provider target` used the generic target kernel on
  B200.
- Candidate: dispatch `target` to `target_kernel_blackwell.py` on SM100 and add
  API parity for `return_scores`/fallback args.
- Command:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '
  cd /root/work/op-kernel-flashsampling &&   source /root/work/optimization-playground/.venv/bin/activate &&   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/work/op-kernel-flashsampling/python:$PYTHONPATH   python scripts/playground/bench_flashsampling_provider.py     --providers target_generic target     --vocab-size 16160 --hidden-size 7168 --batch-sizes 1 32 64     --warmup 20 --iters 200 --direct-module-import     --output-json /root/agent-runs/flashsampling-round0-generic-vs-blackwell.json
'
```

| Batch | Incumbent `target_generic` ms | Candidate `target_blackwell` ms | Change | Decision |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 0.049059 | 0.047400 | 3.38% faster | accept |
| 32 | 0.048544 | 0.047206 | 2.76% faster | accept |
| 64 | 0.049855 | 0.049906 | 0.10% slower, within noise | accept for B200 provider correctness |

New incumbent commit: `2564780c6` (`Route FlashSampling target provider to Blackwell`).

Subsequent rounds compare against this committed incumbent. The command shape was:

```bash
/root/agent-runs/gpu_locked.sh bash -lc '
  cd /root/work/op-kernel-flashsampling &&   source /root/work/optimization-playground/.venv/bin/activate &&   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/work/op-kernel-flashsampling/python:$PYTHONPATH   python scripts/playground/bench_flashsampling_provider.py     --providers target --vocab-size 16160 --hidden-size 7168     --batch-sizes 1 32 64 --warmup 20 --iters 200     --direct-module-import --output-json /root/agent-runs/flashsampling-round-<name>.json     <candidate override>
'
```

| Round | Incumbent | Candidate | BS1 ms | BS32 ms | BS64 ms | Decision |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| 1 | `2564780c6` target Blackwell | `BLOCK_D=64` | 0.052593 | 0.053097 | 0.053800 | reject: slower than incumbent 0.047034 / 0.047219 / 0.049948 |
| 2 | `2564780c6` target Blackwell | `BLOCK_V=256` | 0.060053 | 0.061689 | 0.112338 | reject: slower, especially BS64 |
| 3 | `2564780c6` target Blackwell | `num_warps=4` | 0.047127 | 0.047231 | 0.050833 | reject: tie at BS1/32, slower at BS64 |
| 4 | `2564780c6` target Blackwell | `BLOCK_D=256` | 0.047521 | 0.065580 | 0.068808 | reject: large BS32/64 regression |
| 5 | `2564780c6` target Blackwell | force `num_stages=4` | 0.049225 | 0.048805 | 0.049842 | reject: slower at BS1/32, BS64 tie |
| 6 | `2564780c6` target Blackwell | force `num_stages=3` | 0.052469 | 0.052031 | 0.053871 | reject: slower |
| 7 | `2564780c6` target Blackwell | force `num_stages=2` | 0.080342 | 0.080238 | 0.083285 | reject: much slower |

Restart round 8 accepted under the stricter close premise:

- Incumbent: committed Blackwell target provider at `2564780c6`, with `BLOCK_H=16` for all `H <= 16`.
- Candidate: keep `H=1` on `BLOCK_H=16`, but use `BLOCK_H=8` for small warmup/sweep buckets `2 <= H <= 8`. This avoids doing a 16-column MMA tile for multi-request buckets that only need 2-8 columns, while preserving the incumbent launch shape for `H=1`, `H=32`, and `H=64`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=1 /root/agent-runs/gpu_locked.sh python - <<'PY'
# paired CUDA-event benchmark; artifact:
# /root/agent-runs/flashsampling-restart-round3-blockh2to8-final-paired-gpu1.json
PY
```

| Batch | Incumbent old `BLOCK_H=16` ms | Candidate ms | Change | Decision |
| ---: | ---: | ---: | ---: | :--- |
| 2 | 0.045983 | 0.045243 | 1.61% faster | accept |
| 4 | 0.045770 | 0.045088 | 1.49% faster | accept |
| 8 | 0.045824 | 0.045118 | 1.54% faster | accept |
| 32 | 0.047171 | 0.047173 | tie | neutral |
| 64 | 0.049540 | 0.049684 | 0.29% slower, within noise for unchanged tile | neutral |

Additional restart candidates rejected against the best accepted incumbent:

| Candidate | Evidence | Decision |
| :--- | :--- | :--- |
| Force non-persistent target dispatch beyond one SM wave | H128 0.086338 ms vs incumbent 0.057755 ms; H256 0.161747 ms vs 0.084151 ms | reject: two-wave target grid rereads weights and is much slower |
| Use `BLOCK_H=128` for H128/H256 buckets | H128 0.065801 ms vs incumbent 0.057755 ms; forced H256 0.123797 ms vs 0.084151 ms | reject: accumulator pressure outweighs fewer H tiles |
| Add `maxnreg=255` launch hint | single run was noise-level (H1 0.046647 ms vs 0.046713 ms; H32 0.047221 ms vs 0.047225 ms) and did not explain the paired small-H win | reject: removed from source |

Restart round 9 accepted after `c251cb7bb`:

- Incumbent: `c251cb7bb`, which uses `BLOCK_H=8` for greedy and non-greedy `2 <= H <= 8`, but keeps H1 on `BLOCK_H=16`.
- Candidate: use `BLOCK_H=8` for non-greedy H1 only. The greedy H1 path stays on `BLOCK_H=16` because its earlier paired greedy measurements were noise-level, while non-greedy H1 spends extra work generating Gumbel noise for unused columns.
- Paired CUDA-event artifact: `/root/agent-runs/flashsampling-restart-round5-sampling-h1-blockh8-candidate-gpu-any.json`.

| Batch | Incumbent non-greedy ms | Candidate non-greedy ms | Change | Decision |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 0.047392 | 0.045498 | 4.00% faster | accept |
| 2 | 0.045277 | 0.045281 | tie | neutral |
| 4 | 0.045330 | 0.045285 | 0.10% faster | neutral |
| 8 | 0.045256 | 0.045206 | 0.11% faster | neutral |

Restart candidate rejected after `c251cb7bb`:

| Candidate | Evidence | Decision |
| :--- | :--- | :--- |
| Replace per-H local-reduce CTAs with one small-H CTA | `/root/agent-runs/flashsampling-restart-round4-smallh-reduce-candidate-gpu-any.json`: H2 0.045240 -> 0.045315 ms, H4 0.045117 -> 0.045092 ms, H8 0.045114 -> 0.045105 ms | reject: launch floor dominates and changes are ties/noise |

All rejected candidates matched dense argmax correctness or, for stochastic
sampling probes, produced in-range token ids. No rejected kernel constant changes
are left in source. A dense matmul+argmax floor check on the original greedy
shape measured 0.049364 / 0.051334 / 0.053421 ms for BS1/32/64, while the
post-restart target path measures 0.046680 / 0.047191 / 0.050074 ms for the
same buckets and 0.045159 / 0.045169 / 0.045117 ms for BS2/4/8. The final
non-greedy H1/2/4/8 sweep measured 0.045485 / 0.045252 / 0.045424 /
0.045425 ms. Final IKP/nsys attribution for non-greedy H1 shows
`flashsample_blackwell_kernel` still dominates at 39.937 us mean, while
`_local_reduce_samples_kernel` is 2.415 us mean. Stopping rationale: the
remaining work is the matmul/noise kernel plus one small fixed reduction launch;
Rule 7/CuTe constraints make `BLOCK_H=8` the smallest plausible tensor-core N
tile, and further tested surfaces (two-wave non-persistent dispatch, `BLOCK_H=128`,
local-reduce fusion, `maxnreg=255`, stage/warp/D/V tile changes) regressed or
tied within noise against the best accepted incumbent.

Caveat: the VM shared venv was missing an SM100
`sgl_kernel/common_ops` binary, so standalone kernel profiling used
`--direct-module-import`; server-level B200 TPOT benchmarking is blocked until
that shared install is restored.

The direct kernel tests validate greedy equality, logits debug mode, sampled-id
range, seed sensitivity, vocabulary-shard offsets, and compact local index
workspace dtype.
