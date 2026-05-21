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
      "indexcache": {
        "enabled": true,
        "freq": 4,
        "pattern": "FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF"
      },
      "hisa": {
        "enabled": true,
        "mode": "indexcache-hisa",
        "block_size": 128,
        "compression_ratio": 4.0,
        "min_seq_len": 8192,
        "execution_mode": "optimized"
      }
    }
  }
}
```

The CLI takes precedence over model-card defaults. The production model-card
shape above resolves to the same serving state as
`--nsa-indexer-quantization nvfp4_e2m1_ue8m0 --nsa-indexer-mode indexcache-hisa
--enable-nsa-nvfp4-hisa --hisa-compression-ratio 4.0`, with the optional
IndexCache frequency or pattern copied from the `indexcache` block when launch
flags do not override them. If `indexcache.enabled=true` is present, then
`hisa.mode` must be `indexcache-hisa`; selecting standalone `hisa` is rejected
as a contradictory deployment config. The accepted NVFP4 IndexCache+HISA
integration uses strict `--hisa-compression-ratio 4.0` semantics; fixed-budget
HISA debug modes are outside this contract.

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
`map/store`. Exact-pool cases use `block_topk_map_all` instead of candidate
logits/top-k, fusing block selection and token-id mapping into one CUDA launch. IKP source is available on the B200 VM under
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

The post-acceptance fused-top-k toggle loop kept the exact radix fused
mask/top-k/map path enabled. Disabling it preserved the comparator but slowed
HISA to 0.250560/0.288241/0.373699/0.584942 ms at
8K/16K/32K/64K, versus 0.149637/0.186061/0.254515/0.420654 ms with the fused
path.

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

## Runtime NIXL Validation

The production disaggregated path can route NVFP4 IndexCache+HISA through
HiSparse KV transfer. In that shape, a per-rank HiSparse page slice may be empty
even when the overall request has KV pages. The NIXL sender must not build an
RDMA descriptor list for the empty slice; instead it sends the normal KV arrival
notification without data so the decode-side transfer status can advance.

B200 validation after this guard:

```bash
AUTOINFER_CANDIDATE=setup1_full_custom_r27_nixl_empty_notification_fix_c32 \
AUTOINFER_CELLS=e2e_1k_128_c32:1024:128 \
AUTOINFER_BENCH_SHORT_PROMPTS=32 \
AUTOINFER_MAX_RUNNING_REQUESTS=32 \
/home/spencergarnets/dynamo-reap-production-bundle-20260520-233425/infrastructure/scripts/sglang-reap/run-b200-six-cell-bench.sh
```

Artifact:
`/home/spencergarnets/inference-opt/e2e-merge-full-custom-r27-nixl-empty-notif-fix-20260521T210707Z/summary.json`.
The run completed 32/32 exact-token requests with 32,768 input tokens and 4,096
output tokens. Output throughput was 22.976 tok/s, mean TTFT was 166,248.7 ms,
and mean TPOT was 85.279 ms. This is a functional correctness and transfer
stability gate only; it is not a production throughput result.

## Acceptance

The 4:1 path must beat ordinary NVFP4 IndexCache on the paper-shaped
multi-query benchmark and pass focused correctness tests before it is used. The
runtime dispatch keeps the decode-shaped one-query path on ordinary NVFP4 IndexCache by
default because HISA's block-selection overhead does not amortize there.


## 2026-05-17 Restart Follow-up

Hardware and environment: B200 VM `root@31.22.104.123`, branch
`codex/hisa-indexcache-tensor-op-loop-20260517`, shared venv
`/root/work/optimization-playground/.venv`. CUDA-event microbenchmarks and
pytest used GPU 1 via `/root/agent-runs/gpu_locked.sh`; the final Nsight and
64-head DeepGEMM probes used GPU 0. Build/JIT-load probes were run outside the
GPU lock.

Exact-pool helper artifacts:

- Round 1 fused map-all benchmark:
  `/root/agent-runs/hisa-indexcache-round1-mapall-candidate-gpu1.jsonl`.
- Round 2 tail-clear benchmark:
  `/root/agent-runs/hisa-indexcache-round2-tail-clear-candidate-gpu1.jsonl`.
- Final Nsight report:
  `/root/agent-runs/hisa-indexcache-final-fused-mapall-nsys.nsys-rep`, kernel
  CSV `/root/agent-runs/hisa-indexcache-final-fused-mapall-kernels.csv`.

B200 CUDA-event medians for dynamic 4:1 exact-pool shapes where
`effective_block_topk * 128 == topk_tokens`:

| Shape | Two-step incumbent ms | Fused accepted ms | Tail-clear final ms | Final speedup vs two-step |
| --- | ---: | ---: | ---: | ---: |
| topk1024 prefix4096 rows32 | 0.013760 | 0.011776 | 0.011120 | 1.24x |
| topk1024 prefix4096 rows1024 | 0.015584 | 0.013536 | 0.012864 | 1.21x |
| topk2048 prefix8192 rows32 | 0.015712 | 0.015008 | 0.013216 | 1.19x |
| topk2048 prefix8192 rows1024 | 0.022768 | 0.017696 | 0.015136 | 1.50x |

Nsight Systems measured the final
`hisa_block_topk_map_all_indexer_cache_nvfp4` CUDA kernel itself at 8.688 us
median over 10 launches for topk2048/prefix8192/rows1024 on GPU 0. A launch
geometry candidate that capped the fused helper at 32 threads was rejected using
`/root/agent-runs/hisa-indexcache-round4-threads32-reject-gpu1.jsonl`: it
regressed topk2048/prefix8192/rows1024 from 0.015136 ms to 0.029600 ms
(1.96x slower).

DeepGEMM FP4 MQA support is limited to packed Q tensors with 64 heads in this
venv. Probe artifact `/root/agent-runs/hisa-indexcache-round3-deepgemm-head64-gpu0.txt`
measured 64-head precomputed HISA at topk2048/prefix8192/query_rows1 as
0.052403 ms versus ordinary NVFP4 IndexCache 0.154877 ms (2.96x). The 32-head
artifacts `/root/agent-runs/hisa-indexcache-round3-deepgemm-head32-gpu1.txt`
and `/root/agent-runs/hisa-indexcache-restart-deepgemm-head32-gpu0.txt` fail at
DeepGEMM JIT compile time with `Unsupported TMEM load size`: the current
Blackwell FP4 MQA implementation splits heads into two TMEM loads of
`kNumHeads / 2`, so 32 heads produces an unsupported 16-wide TMEM load. The
HISA DeepGEMM path therefore returns `None` for 32-head tensors; direct fallback
proof is in `/root/agent-runs/hisa-indexcache-round3-direct-head32-fallback-gpu1.txt`.

Verification artifacts: round 1 focused pytest
`/root/agent-runs/hisa-indexcache-round1-fused-mapall-pytest-gpu1.txt` passed
`16 passed`; round 2
`/root/agent-runs/hisa-indexcache-round2-tail-clear-pytest-gpu1.txt` passed
`16 passed`; round 3
`/root/agent-runs/hisa-indexcache-round3-deepgemm-guard-pytest-gpu1.txt` passed
`17 passed`.
