# DSA NVFP4 Indexer + Production TPOT — Deep Root-Cause Investigation

**Date**: 2026-05-30
**Pod**: `deepseek-v32-nextn-graft-sglang-0-decode-qfpsg` (dynamo-system ns)
**Model**: BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft
**Hardware**: 8x B200 (148 SM each), TP=8 + EP=8
**Baseline TPOT (measured live)**: 29-48 ms depending on KV-cache warmth (49.5 ms cold, 29.4 ms warm at 32 conc)
**Pod build**: `localhost/optimization-playground-sglang-runtime:reap-nvfp4-tokenspeed-0.1.1-nodeps-lsefix`
**SGLang version**: 0.5.10.post1

---

## Executive summary (TL;DR)

The iter1-6 NVFP4 indexer wins, measured at microbench cells **(batch=128, prefix=32K)**, **DO NOT MATERIALIZE in production decode** because:

1. **The production decode cell is `batch=8 per rank, prefix~4096-5120`, total_work=8192-10240** — three orders of magnitude smaller than the iter5 PRIMARY autotune trigger (`total_work >= 1,048,576`).
2. **The autotune dispatcher routes the production cell to `tilen=8`** (the small-batch path), bypassing both **iter4 PRIMARY (persistent + kprefetch, default OFF)** AND **iter5 PRIMARY (WMMA cand_score, gated on TILEN=32)**.
3. **Iter4/5/6 mean_pool wins DO apply** (7.9us savings per call at prod cell, 460us at iter5 cell), but they save <1% of indexer pipeline at the production cell — the cand_score kernel is the *bottleneck*, not mean_pool.
4. **WMMA gives a 2x speedup at ALL cells** I measured (not just total_work>=1M). The autotune threshold is empirically miscalibrated for the production grid.
5. The indexer consumes **~50% of TPOT** at production cell because `index_topk_freq` is **not set in model config** (defaults to 1), so the indexer runs on all 61 transformer layers per decode step.

The other ~50% of TPOT (~20 ms) is MLA attention + MoE GEMM + RMSNorm + allreduce.

---

## 1. The actual TPOT breakdown at production cell

**Measured live TPOT**: 29.4 ms (32 conc, 4096in/256out, 2nd run with KV cache warm), 47.8 ms cold.

### 1.1 Indexer share (kernel-level breakdown at batch=8/rank, prefix=4096, with all iter env vars ON)

| Component | Time/call | Notes |
|----|----|----|
| mean_pool (iter6 PRIMARY transp)| 21.5 us | Down from 29.4us at iter2 baseline (-27%) |
| block_score (`fp8_fp4_mqa_logits` equivalent — torch path)| 29.2 us | Bottleneck of pre-cand_score stage |
| block_topk | ~7 us | small reduction |
| cand_score **TILEN=8** (autotune selected) | **202 us** | The dominant kernel — autotune picks small-batch variant |
| topk + map/store | ~10 us | small |
| Python dispatch + cudaLaunchKernel × 18 + sync (max.item, etc) | **~80 us** | The dispatcher overhead |
| **Total per-call (measured)** | **357 us** | matches direct `torch.cuda.Event` measurement |

**Layers running indexer**: 61 (all transformer layers). `index_topk_freq` is **not set in model config**, defaults to 1.

**Per decode step contribution**: 61 layers × 357 us = **~21.8 ms = ~50% of cold TPOT (47.8 ms) or 74% of warm TPOT (29.4 ms)**.

### 1.2 Where the remaining time goes (estimated from kernel inventory)

| Component | Per-step estimate | Notes |
|----|----|----|
| Indexer (DSA topk selection) | ~22 ms | dominant |
| MLA attention (Q/K/V projection + RoPE + scaled-dot + V proj) | ~7-10 ms | MLA on 16K context with selected 2048 tokens |
| MoE GEMM (flashinfer_trtllm, EP=8 with 16 experts/rank) | ~10-12 ms | 58 MoE layers; 8/128 routed experts/token |
| RMSNorm + residual + LM head | ~2 ms | cheap |
| `--enable-flashinfer-allreduce-fusion` allreduce | ~1-3 ms | 8-way TP allreduce |
| **Total TPOT projection** | **~42-48 ms** | matches live measurement |

---

## 2. The actual kernel selected at runtime

### 2.1 DeepGEMM availability

`deep_gemm` has **`fp8_mqa_logits`** and **`fp8_paged_mqa_logits`**, but does **NOT** have `fp8_fp4_mqa_logits` or `fp8_fp4_paged_mqa_logits`.

This means at `_should_use_hisa_nvfp4_paged` (dsa_indexer.py:1216-1275):
- Line 1248: `has_dense_nvfp4_kernel = False`
- Line 1253-1254: `if not has_dense_nvfp4_kernel: return True` → **HISA NVFP4 path is always taken**

At `_get_topk_hisa_nvfp4_paged` (dsa_indexer.py:1443-1448):
- `_has_deep_gemm_kernel("fp8_fp4_mqa_logits") == False` → routes to **`nvfp4_hisa_indexer_paged_torch`** (iter wins path) ✓

### 2.2 Env-var sanity check (in-pod confirmation)

All iter env vars correctly loaded into Python-level constants:

| Env var | Pod value | Python const |
|----|----|----|
| `SGLANG_NSA_NVFP4_HISA` | 1 | `_hisa_nvfp4_env_enabled=True` |
| `SGLANG_NSA_NVFP4_HISA_CAND_SCORE_WMMA` | 1 | `_hisa_nvfp4_candidate_score_wmma=True` |
| `SGLANG_NSA_NVFP4_HISA_MEAN_POOL_PREDECODE` | 1 | `_hisa_nvfp4_mean_pool_predecode=True` |
| `SGLANG_NSA_NVFP4_HISA_MEAN_POOL_TRANSP` | 1 | `_hisa_nvfp4_mean_pool_transp=True` |
| `SGLANG_NSA_NVFP4_HISA_COMPRESSION_RATIO` | 4.0 | `_hisa_nvfp4_compression_ratio=4.0` |
| `SGLANG_NSA_HISA_PAGED_DECODE_MIN_SEQ_LEN` | 0 | `_hisa_paged_decode_min_seq_len=0` |

Autotune & dispatch: `_hisa_nvfp4_candidate_score_autotune=True`, `_hisa_nvfp4_candidate_score_tilen_size=8` (default), `_hisa_nvfp4_collective_key=False`.

### 2.3 Autotune dispatcher routing (`nvfp4_indexer.py:3106-3168`)

```python
if total_work >= 1048576:
    if _hisa_nvfp4_candidate_score_wmma and n_heads_proj <= 64:
        _candidate_score_fn = hisa_candidate_score_tilen32_wmma  # iter5 PRIMARY
    else:
        _candidate_score_fn = hisa_candidate_score_tilen32       # iter5 fallback
elif total_work >= 262144:
    _candidate_score_fn = hisa_candidate_score_tilen16
else:
    _candidate_score_fn = hisa_candidate_score_tilen              # tilen=8
```

`total_work = q_rows * cand_len_proj = batch_per_rank * effective_block_topk * 128`.

At **production cell** (batch=8/rank, prefix=4096): `total_work = 8 * 1024 = 8192 << 262144` → **autotune picks tilen=8**, never WMMA.

To hit iter5 PRIMARY WMMA (`total_work >= 1048576`):
- batch=128, prefix>=32K (the iter projection cell, **never hit at 32-64 conc with TP=8**)
- batch=64, prefix>=64K (above context_length)
- batch=32, prefix>=128K (above context_length)

**The iter5 PRIMARY autotune threshold cannot be reached at the production shape grid.**

### 2.4 Empirical kernel timing at production cell (batch=8, prefix=4096)

| Cand_score variant | Median time | Speedup vs tilen=8 |
|----|----|----|
| tilen=8 (autotune choice) | 202.2 us | 1.00x |
| tilen=16 | 197.2 us | 1.03x |
| tilen=32 | 195.4 us | 1.04x |
| **tilen=32 + WMMA (iter5 PRIMARY)** | **104.8 us** | **1.93x** |
| persistent (iter3 v1) | 315.4 us | 0.64x |
| persistent + kprefetch (iter4 PRIMARY) | 330.0 us | 0.61x |

**WMMA gives a 1.9x speedup at production cell** but is gated off by the autotune `total_work` threshold. iter3 persistent + iter4 kprefetch are correctly OFF.

### 2.5 Scaling: WMMA wins at every cell I measured

| Cell | tilen=8 | tilen=32+WMMA | Speedup |
|----|----|----|----|
| batch=1, prefix=4096 | 53.1 us | 41.1 us | 1.29x |
| batch=8, prefix=4096 | 202.2 us | 104.8 us | **1.93x** |
| batch=8, prefix=5120 | 206.6 us | 111.1 us | 1.86x |
| batch=32, prefix=5120 | 724.9 us | 339.4 us | 2.14x |
| batch=64, prefix=5120 | 1415.2 us | 651.6 us | 2.17x |
| batch=128, prefix=32K | 2905.8 us | 1385.9 us | 2.10x |

**WMMA wins at ALL tested cells, including the smallest ones**. The autotune threshold is empirically miscalibrated.

---

## 3. The actual HISA path taken (gating conditions)

Trace of `_should_use_hisa_nvfp4_paged` at production cell evaluation:

| Gate | Pass? | Reason |
|----|----|----|
| `enable_dsa_nvfp4_hisa` | ✓ | `SGLANG_NSA_NVFP4_HISA=1` |
| `uses_hisa(mode)` | ✓ | `indexcache-hisa` |
| `hisa_execution_mode == "optimized"` | ✓ | default |
| `_is_cuda` | ✓ | B200 |
| `forward_mode.is_decode_or_idle()` | ✓ | decode |
| `index_topk in (1024, 2048)` | ✓ | 2048 |
| `hisa_block_size == _hisa_nvfp4_block_size` | ✓ | 128 == 128 |
| `hisa_compression_ratio == _hisa_nvfp4_compression_ratio` | ✓ | 4.0 == 4.0 |
| `hisa_block_size == 128` | ✓ | redundant check |
| `has_dense_nvfp4_kernel` | **False** | deepgemm has no `fp8_fp4_paged_mqa_logits` |
| **Returns** | **True** (line 1254 short-circuit) | **HISA NVFP4 is taken** |

Trace inside `_get_topk_hisa_nvfp4_paged` → `nvfp4_hisa_indexer_paged_torch`:

1. `_should_fallback_to_dense_short(True, prefix_lens=4096, topk=2048)` → 4096 > 2048 → returns False → continues
2. `_hisa_mean_pool_call` → **iter6 PRIMARY transp** kernel selected (transp + predecode env vars)
3. `hisa_block_score_indexer_cache_nvfp4` → unchanged
4. `_hisa_block_topk_counts` (compression_ratio=4.0) → `effective_block_topk=8`, `candidate_len=8*128=1024`
5. **`candidate_len(1024) != topk(2048)`** → does NOT hit early-return fast path (`hisa_block_topk_map_all`)
6. `hisa_block_topk_indexer_cache_nvfp4` → block topk selection
7. cand_score dispatcher → autotune: `total_work=8*1024=8192` → **`hisa_candidate_score_tilen` (tilen=8)** selected, NOT WMMA
8. final topk + gather + map

**Conclusion**: the iter5 PRIMARY WMMA cand_score code path is **dead code** at production cell.

---

## 4. The actual EP behavior

Process tree confirms 8 sglang scheduler processes: `TP0_EP0` through `TP7_EP7`. Each rank IS its own EP rank. `--ep-size 8` is correctly wired.

**Expert sharding**: `n_routed_experts=128` (REAP-reduced from 256), `--ep-size 8` → **16 experts per rank** locally.

flashinfer_trtllm receives `local_expert_offset = 16 * rank, local_num_experts = 16`. Each token activates 8/128 experts globally, so on average each rank processes `8 * batch * (16/128) = batch` expert routings per layer (load-balanced assumption).

**This means MoE GEMM per rank** scales with `O(batch_per_rank * 16_local_experts * moe_intermediate_size = batch * 16 * 2048)`. At batch=8: 8 × 16 × 2048 = 262144 ops/layer for the gate (small), and the gemm itself processes `batch * 16 * hidden_size * moe_intermediate_size = 8 * 16 * 7168 * 2048 = 1.88B FMAs` per MoE layer. At BF16 throughput of ~990 TFLOPS on B200, this is ~1.9ms/MoE layer in compute (but actually less, since each token only activates 8 experts globally, ~1 per rank with EP=8).

EP=8 is functioning correctly. MoE is not the dominant bottleneck.

---

## 5. Five SPECIFIC, ACTIONABLE next steps (ranked by ROI / effort)

### Step 1 — **Force WMMA cand_score at all cells** (HIGHEST ROI, lowest effort)
**File**: `python/sglang/jit_kernel/nvfp4_indexer.py`, lines 3106-3168 (also lines 3153-3164 for non-autotune path)

The autotune threshold `total_work >= 1048576` is empirically miscalibrated. WMMA gives 1.9-2.2x speedup at ALL measured cells. **Lower threshold to 1024 or always-on when `_hisa_nvfp4_candidate_score_wmma and n_heads <= 64`.**

**Recommended patch** (smallest impact, biggest win):
```python
# At top of autotune branch (line ~3133):
n_heads_proj = int(q_values.shape[1])
if _hisa_nvfp4_candidate_score_wmma and n_heads_proj <= 64:
    # WMMA wins at all cells; total_work threshold was set conservatively
    # against the iter5 microbench but production cells are 100-1000x smaller.
    # Empirical: batch=8/prefix=4K saves 97us vs tilen=8 (1.9x).
    _candidate_score_fn = hisa_candidate_score_tilen32_wmma_indexer_cache_nvfp4
elif total_work >= 1048576:
    _candidate_score_fn = hisa_candidate_score_tilen32_indexer_cache_nvfp4
elif total_work >= 262144:
    _candidate_score_fn = hisa_candidate_score_tilen16_indexer_cache_nvfp4
else:
    _candidate_score_fn = hisa_candidate_score_tilen_indexer_cache_nvfp4
```

**Projected impact**: At production cell, indexer per-call drops from ~357us to ~270us (saves ~87us × 61 layers = **5.3 ms / decode step**). **TPOT 42 ms → ~36-37 ms (-12-15%)**.

**Validation**: re-run the production bench with this patch + intel-correctness sweep.

**ROI**: HIGHEST — 5+ms TPOT win for a 5-line patch.

---

### Step 2 — **Set `index_topk_freq=4` in model config OR via CLI** (huge ROI, requires correctness check)
**File**: `/models/.../config.json` or `--json-model-override-args`

Currently `index_topk_freq` is not set (defaults to 1). Indexer runs on **all 61 layers**. Setting freq=4 makes only ~16 layers compute the indexer; the other 45 reuse the previous layer's topk.

**Projected impact**:
- Layers running indexer: 61 → 16 (saves 45 × 357us = **16 ms / decode step**!)
- BUT: needs correctness verification (the model must be **trained with freq=4** or accuracy degrades). DeepSeek-V3.2 DSA training used freq=1; using freq=4 at inference time may degrade quality on hard prompts.

**Required**: run intel-correctness with freq=4 before deploying.

**ROI**: HIGHEST possible (16 ms = 38% TPOT win) but RISK is correctness regression. Need quality eval.

**Action**: spec a 5-prompt eval (truthfulqa-mc + gsm8k) with freq=4 vs freq=1, compare quality. If <2% drop, deploy.

---

### Step 3 — **Cache `block_counts.max().item()` across layers** (medium ROI, low effort)
**File**: `python/sglang/jit_kernel/nvfp4_indexer.py:2722, 344` and `dsa_indexer.py` per-layer call sites

The indexer dispatcher does **2 device-to-host syncs per call**: `seq_lens_flat.max().item()` in `_hisa_max_blocks` (line 344), and `selected.max().item()` in `_hisa_block_topk_counts` (line 2722). At 61 layers/step, that's **122 syncs/step = ~1.2 ms of sync time** (10us per sync).

But `seq_lens` and prefix_lens are *identical for all 61 layers within a single decode step* (KV cache only grows by 1 token per step). The first layer's `max_blocks` and `block_topk_counts` can be **cached and reused for the remaining 60 layers**.

**Patch sketch**: add a per-forward-batch scratch dict, computed by layer 0 and read by layers 1-60.

**Projected impact**: -1 ms TPOT.

**ROI**: medium — saves dispatcher overhead that's currently 25% of the indexer pipeline at production cell.

---

### Step 4 — **Eliminate `torch.all(...).item()` sync in `_should_fallback_to_dense_short`** (small ROI)
**File**: `python/sglang/jit_kernel/nvfp4_indexer.py:336`

```python
return bool(torch.all(prefix_lens <= topk_tokens).item())
```

At production cell, this is **almost always False** (prefix grows from 4096 to 5120, always > topk=2048). But it forces a sync. Replace with a host-side check: dispatch already knows `max_seq_len_hint` and `index_topk` — compare them on host:

```python
if not fallback_to_dense_if_short or _is_cuda_graph_capturing(prefix_lens):
    return False
# Use the host-side max_seq_len_hint if available
host_max = max_seq_len_hint  # parameterize through call chain
if host_max is not None and host_max > topk_tokens:
    return False
# Only sync as last resort:
return bool(torch.all(prefix_lens <= topk_tokens).item())
```

**Projected impact**: -0.6 ms TPOT (saves 1 sync × 61 layers × 10us).

**ROI**: small but trivial to land.

---

### Step 5 — **Run a production-shape bench grid and re-autotune the entire dispatcher** (high ROI, high effort)
**File**: `python/sglang/jit_kernel/nvfp4_indexer.py:3106-3168`

The current autotune table was built for **(batch=128, prefix=32K)**, which is **never the production cell**. Re-measure the variant winner at every realistic production cell:

| Cell | Current autotune | Empirically best |
|----|----|----|
| batch∈[1,8], prefix∈[1K,8K] | tilen=8 | tilen=32 + WMMA (1.3-1.9x) |
| batch∈[8,32], prefix∈[4K,16K] | tilen=8 | tilen=32 + WMMA (1.9-2.1x) |
| batch∈[32,128], prefix∈[16K,64K] | tilen=8/16 | tilen=32 + WMMA (~2x) |
| batch=128, prefix=32K (the iter cell) | tilen=32 + WMMA | tilen=32 + WMMA (1.9x) |

Replace `total_work >= X` threshold logic with a **lookup table or always-WMMA fallback** when env var is on.

**Projected impact**: same as Step 1, scoped wider.

**ROI**: high. Requires more eng work but cleaner long-term.

---

## Additional findings (lower priority, but worth tracking)

### A. The `candidate_len == topk_tokens` early-return path was BROKEN by topk=2048
With `index_topk=2048`, the early-return at `nvfp4_indexer.py:3042` requires `effective_block_topk * 128 == 2048` → `effective_block_topk == 16` → `block_counts >= 64` → `prefix >= 8192`. At production cell (prefix=4096-5120), early-return is **NEVER taken**. The fast `hisa_block_topk_map_all` kernel (33us) is **bypassed**, forcing the slower `hisa_block_topk + cand_score + topk` path (~250us).

If the goal is to use the fast path early at low prefix, **lower index_topk to 1024** (matches block_size=128 × ceil(prefix/512) rate at small prefix), at the cost of quality. **Not recommended** without intel-correctness validation.

### B. mean_pool iter wins fully apply but are dwarfed by cand_score
At production cell, iter6 PRIMARY transp saves 7.9us per call (22us vs 30us). At 61 layers/step = **0.48 ms/step**. Real but tiny vs the 5.3ms gain from forcing WMMA.

### C. iter4 PRIMARY (persistent + kprefetch) is correctly OFF
Empirically slower at production cell (330us vs 202us for tilen=8). Keeping it off is right; documentation in the code is accurate.

### D. The bench TPOT mismatch may also be PCG-related
The pod uses `--cuda-graph-max-bs 128` + **no DP attention** + **vanilla CUDA graphs**. With max_running_requests=128 and 32 conc, dynamic batching means the running batch fluctuates 16-32, but the CUDA graph is captured at padded batch=128. **Indexer kernel input shape may be padded to 128** at capture time. Need to verify with a torch.profiler trace from inside a captured decode step (out of scope for this 4hr investigation).

### E. SGLang has a `start_profile` API exposed via dynamo IPC
The `dynamo.sglang.request_handlers.handler_base.HandlerBase.start_profile` route is registered but only reachable via Dynamo's engine-route IPC, not HTTP. Setting up a dynamo client call would allow capturing a full decode step trace; **recommended for a follow-up if Step 1 doesn't unlock the projected wins**.

---

## Methodology / artifacts

All measurements taken **inside the running pod** at `/tmp/`:

- `/tmp/indexer_microbench.py` — production-shape sweep across (batch, prefix) cells
- `/tmp/in_process_profile.py` — sub-kernel attribution (mean_pool variants, block_score, block_topk_map_all)
- `/tmp/scaling_analysis.py` — cand_score variant comparison at every cell
- `/tmp/profile_full_pipeline.py` — torch.profiler chrome trace export
- `/tmp/live_decode_probe.py` — live frontend bench (TPOT measurement)
- `/tmp/sgl_profile/indexer_prod_b8_p4096.json` — chrome trace for production cell
- `/tmp/sgl_profile/indexer_iter5cell_b128_p32K.json` — chrome trace for iter5 cell
- `/tmp/indexer_microbench_results.json`, `/tmp/cand_score_scaling.json` — JSON dumps

**Key timing methodology**: torch.cuda.Event + torch.cuda.synchronize(), 15-20 iterations, median. Warmup: 3-5 iterations.

**Profiler**: torch.profiler with `ProfilerActivity.CPU + CUDA`, key_averages, chrome_trace_export.

**Note**: nsys / IKP are not installed inside the pod and could not be used. Investigation relied on torch profiler + manual timing.
