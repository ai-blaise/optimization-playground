# NVFP4 HISA IndexCache Indexer

This experimental path adds a configurable HISA block-to-token selector for the
NVFP4 IndexCache DSA Indexer. It is evaluated only against the ordinary NVFP4
IndexCache selector; no other baseline is in scope for this integration.

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

The CLI takes precedence over model-card defaults. The accepted NVFP4
IndexCache+HISA integration uses strict `--hisa-compression-ratio 4.0`
semantics; fixed-budget HISA debug modes are outside this contract.

## Contract

The accepted path follows the HISA paper's compression-ratio form:

- Logical block size is fixed to `B=128`.
- For a prefix length `L`, the eligible block count is `M=ceil(L / B)`.
- With strict `compression_ratio=4.0` (Compression Ratio = 4:1), the selected block count is
  `m=ceil(M / 4)`, capped by `M`.
- The candidate token pool is `m * B`.
- For short contexts where `L <= k`, the caller falls back to the ordinary
  NVFP4 IndexCache selector.
- For dynamic compression-ratio mode, there is no additional sequence-length
  threshold. The 4:1 ratio is applied at every eligible `L > k`.
- The first and last eligible blocks are forced into the selected block set.
- Block representatives are mean-pooled over the NVFP4-dequantized indexing
  keys.
- Block and token scoring use the same weighted ReLU DSA score as the ordinary
  NVFP4 IndexCache Indexer: `sum_h weight_h * relu(q_h dot k)`.
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

Paper-shaped forward comparison versus ordinary NVFP4 IndexCache:

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

B200 `topk=2048` commit-readiness forward facts versus ordinary NVFP4
IndexCache:

| Prefix length | Selected blocks `m` | Ordinary NVFP4 IndexCache (ms) | NVFP4 IndexCache+HISA 4:1 (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 16 | 0.360976 | 0.072824 | 4.9569x |
| 16384 | 32 | 0.531413 | 0.191044 | 2.7816x |
| 32768 | 64 | 1.006156 | 0.259645 | 3.8751x |
| 65536 | 128 | 1.781986 | 0.429308 | 4.1508x |

B200 `topk=1024` forward facts versus ordinary NVFP4 IndexCache, using
`benchmark/nsa/topk1024_forward_hisa_b200_20260513.json`:

| Prefix length | Selected blocks `m` | Ordinary NVFP4 IndexCache (ms) | NVFP4 IndexCache+HISA 4:1 (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 8 | 0.270774 | 0.067825 | 3.9922x |
| 8192 | 16 | 0.356633 | 0.149706 | 2.3822x |
| 16384 | 32 | 0.534138 | 0.186076 | 2.8705x |
| 32768 | 64 | 1.000408 | 0.254425 | 3.9320x |
| 65536 | 128 | 1.778057 | 0.424144 | 4.1921x |

This accepted path uses precomputed/store-maintained HISA block
representatives, exact four-pass radix fused mask/top-k/map, dynamic
`block_topk` launch widths for the 4:1 chart shapes, and a row-wise
`candidate_pages` kernel. A rejected earlier fused top-k candidate used only
the top radix byte and was not exact; strict B200 tests now compare selected
token sets against `torch.topk`.

Decode-shaped sanity check against ordinary NVFP4 IndexCache:

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

The 4:1 path must beat ordinary NVFP4 IndexCache on the paper-shaped
multi-query benchmark and pass focused correctness tests before it is used. The
runtime dispatch keeps the decode-shaped one-query path on ordinary NVFP4 IndexCache by
default because HISA's block-selection overhead does not amortize there.
