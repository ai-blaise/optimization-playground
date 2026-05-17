# FlashSampling

FlashSampling is an optional decode-time sampler that fuses exact sampling into
the LM-head matmul. Instead of materializing the full logits tensor in HBM, it
computes logits tile by tile, applies the same greedy or Gumbel-max sampling
rule on chip, writes per-tile winners, and reduces those winners to token ids.

The implementation is based on:

- [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854)
- [FlashSampling/FlashSampling](https://github.com/FlashSampling/FlashSampling)

Canonical B200 optimization results are tracked in
[FlashSampling](../developer_guide/kernel_results/flashsampling.md).

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
shapes where the total number of `(V, H)` tiles fits in a single SM wave
(`tiles <= NUM_SMS`). This is the typical TP-sharded DeepSeek-V3.2-REAP shape
with `V=16160` on TP=8. On Blackwell (`sm100`/B200), `target` dispatches to the
Blackwell-tuned variant with adaptive pipeline stages; older CUDA devices use
the generic target kernel. When tiles exceed `NUM_SMS`, the target provider
falls back to the persistent kernel automatically.

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

The default minimum decode batch size is conservative:

```bash
--flashsampling-min-batch-size 128
```

Use these gates for A/B tests:

```bash
--flashsampling-min-batch-size 1
--flashsampling-max-batch-size 256
```

## CUDA Graphs And Warmup

When FlashSampling and CUDA graphs are both enabled, SGLang captures separate
graph variants for eligible FlashSampling decode batches.

```bash
--flashsampling-warmup-batch-sizes 1 32 64
```

Passing `--flashsampling-warmup-batch-sizes` with no values disables this
warmup.

## Tensor Parallel And DP Attention

For tensor-parallel LM heads, FlashSampling runs over the active sampling group
and reduces local per-rank winners. With DP attention enabled, the server
enables DP LM-head sharding so sampling can avoid materializing full logits
outside the attention tensor-parallel group.

## Validation Harness

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

The IKP serving profiler captures kernel-level traces for the same baseline
and FlashSampling variants:

```bash
OUT_DIR=/tmp/sglang_flashsampling_serving_ikp \
MODEL=Qwen/Qwen3-1.7B \
CONCURRENCY=128 \
WARMUP_REQUESTS=concurrency \
scripts/playground/profile-flashsampling-serving-ikp.sh
```
