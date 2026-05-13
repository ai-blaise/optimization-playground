# NVFP4 HISA IndexCache Indexer

This experimental path adds a configurable HISA block-to-token selector for the
NVFP4 IndexCache DSA Indexer. It is an alternative to the incumbent dense
IndexCache selector, not a replacement.

Enablement:

```bash
--nsa-indexer-quantization nvfp4_e2m1_ue8m0 \
--nsa-indexer-mode indexcache-hisa \
--enable-nsa-nvfp4-hisa \
--hisa-compression-ratio 4.0
```

Model config enablement:

```json
{
  "quantization_config": {
    "indexer_quantization": {
      "quant_method": "nvfp4_e2m1_ue8m0",
      "value_format": "e2m1",
      "scale_format": "ue8m0",
      "scale_block_size": 32,
      "hisa": {
        "enabled": true,
        "block_size": 128,
        "compression_ratio": 4.0,
        "execution_mode": "optimized"
      }
    }
  }
}
```

The CLI takes precedence over model-card defaults. Set
`--hisa-compression-ratio 0` to use the fixed `--hisa-block-topk` budget for
debugging or A/B tests; fixed-budget mode also honors `--hisa-min-seq-len`.

## Contract

The default path follows the HISA paper's compression-ratio form:

- Logical block size is fixed to `B=128`.
- For a prefix length `L`, the eligible block count is `M=ceil(L / B)`.
- With `compression_ratio=4.0`, the selected block count is
  `m=max(ceil(M / 4), ceil(k / B))`, capped by `M`.
- The candidate token pool is `m * B`.
- For short contexts where `L <= k`, the caller falls back to the incumbent
  NVFP4 IndexCache selector.
- For dynamic compression-ratio mode, there is no additional sequence-length
  threshold. The 4:1 ratio is applied at every eligible `L > k`.
- The first and last eligible blocks are forced into the selected block set.
- Block representatives are mean-pooled over the NVFP4-dequantized indexing
  keys.
- Block and token scoring use the same weighted ReLU DSA score as the incumbent
  Indexer: `sum_h weight_h * relu(q_h dot k)`.
- Sparse MLA is unchanged; this path only changes the selected index set.
- When the candidate token pool has exactly `k` tokens, the implementation skips
  candidate token scoring and maps the selected blocks directly to output token
  ids. This preserves semantics because the refinement top-k would return every
  candidate token.

For the paper Figure 2 sequence lengths with `k=2048`, the dynamic block budget
is:

| Prefix length | Eligible blocks `M` | Selected blocks `m` | Candidate tokens |
| ---: | ---: | ---: | ---: |
| 8192 | 64 | 16 | 2048 |
| 16384 | 128 | 32 | 4096 |
| 32768 | 256 | 64 | 8192 |
| 65536 | 512 | 128 | 16384 |

## Profiling

The benchmark JSONL profile contains the HISA stages `blockscore_precomputed`,
`block_topk`, `candidate_pages`, `candidate_logits`, `fused_mask_topk_map`, and
`map/store`. Exact-pool cases use `candidate_map_all` instead of candidate
logits/top-k. IKP source is available on the B200 VM under
`/root/b200-phase/refs/intra-kernel-profiler` and should be used when changing
the CUDA kernels.

Paper-shaped kernel comparison:

```bash
python benchmark/nsa/bench_nvfp4_hisa_indexer.py \
  --prefix-lengths 8192,16384,32768,65536 \
  --topks 2048 \
  --heads 64 \
  --query-rows 1024 \
  --hisa-candidate-scorer precomputed \
  --hisa-compression-ratio 4.0 \
  --warmup 5 \
  --iters 20 \
  --json-out /tmp/hisa_nvfp4_4to1_paper_shape.json
```

B200 paper-shaped results from
`/root/b200-phase/logs/worker_a_launchshape_4to1_repeat_20260513T004559Z`:

| Prefix length | Selected blocks `m` | Incumbent DeepGEMM (ms) | HISA 4:1 (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 16 | 0.3545 | 0.0701 | 5.06x |
| 16384 | 32 | 0.5675 | 0.1872 | 3.03x |
| 32768 | 64 | 1.0010 | 0.2567 | 3.90x |
| 65536 | 128 | 1.7773 | 0.4307 | 4.13x |

This accepted path uses precomputed/store-maintained HISA block
representatives, exact four-pass radix fused mask/top-k/map, dynamic
`block_topk` launch widths for the 4:1 chart shapes, and a row-wise
`candidate_pages` kernel. A rejected earlier fused top-k candidate used only
the top radix byte and was not exact; strict B200 tests now compare selected
token sets against `torch.topk`.

Decode-shaped comparison:

```bash
python benchmark/nsa/bench_nvfp4_hisa_indexer.py \
  --prefix-lengths 8192,16384,32768,65536 \
  --topks 2048,1024 \
  --heads 64 \
  --query-rows 1 \
  --hisa-candidate-scorer precomputed \
  --hisa-compression-ratio 4.0 \
  --warmup 5 \
  --iters 20
```

## Acceptance

The 4:1 path must beat incumbent NVFP4 IndexCache on the paper-shaped
multi-query benchmark and pass focused correctness tests before it is used. The
runtime dispatch keeps the decode-shaped one-query path on dense IndexCache by
default because HISA's block-selection overhead does not amortize there.
