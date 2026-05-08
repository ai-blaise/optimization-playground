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

The direct kernel tests validate greedy equality, logits debug mode, sampled-id
range, seed sensitivity, vocabulary-shard offsets, and compact local index
workspace dtype.
