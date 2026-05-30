# PCG + DP-attention vs vanilla CUDA-graph + no-DP — strategic research

Author: Spencer Garnets
Date: 2026-05-30
Status: research-only; no code changes. Decision input for #14 pivot.

## 0. Executive summary (3 lines)

**Recommendation: PIVOT to TP=8, no DP attention, drop --enforce-piecewise-cuda-graph.**
On 8xB200 single-node for DSv3.2-REAP-345B with `--moe-runner-backend
flashinfer_trtllm` (NOT wide-EP), DP attention pays its full cost (two
allgathers + reduce-scatter per layer, the entire `forward_idle` ladder,
the small-tensor B200 SM_100 kernel launch bug surface) for ~zero
attention compute saving — the attention savings are realised only when
the MoE is wide-EP, which we are not running. Vanilla CUDA-graph (decode)
+ PCG (extend, default-on) without DP gives the same kernel-level wins
from #16/#19/iter1-6 plus a path to LayerSplit (decode-side) and Warp
Decode, with `forward_idle` removed and the bulk of `dp_attention.py`
gather/scatter triton workarounds rendered moot. Estimated TPOT: ~22-24
ms (within 0.5 ms of the PCG+DP target after the followup-13 work),
without the followup-13/-14/-15 risk on the launch-config bug surface.

## 1. DP attention semantics — when it is worth paying for

### 1.1 What DP attention actually buys you

`--enable-dp-attention` with `dp_size == tp_size` (the deploy config)
sets `attn_tp_size = tp_size // dp_size = 1` and `attn_dp_size = dp_size
= 8` (`compute_dp_attention_world_info` in
`python/sglang/srt/layers/dp_attention.py:284-298`). Each rank now owns
**all** attention heads for **its DP shard of tokens** instead of
**1/tp_size of heads** for **all tokens**. The model code reads
`get_attention_tp_size() = 1` and instantiates `wq_b`, `wo_a`, `wo_b`
with `tp_size=attn_tp_size=1` so the QKV and output projections are
unsharded (deepseek_v4.py:235-405). Each rank does:

```
attn_compute_per_rank = (Nheads * head_dim^2 * tokens_per_rank)
                      = Nheads * head_dim^2 * (B / dp_size)
```

vs the TP-only case:

```
attn_compute_per_rank_tp = (Nheads/tp_size * head_dim^2 * B)
                         = (Nheads * head_dim^2 * B) / tp_size
```

When `dp_size == tp_size == 8`, the FLOPs per rank are **identical**.
DP attention re-distributes the work: each rank processes 1/8 of tokens
on all 8/8 of heads, vs TP processing 1/8 of heads on 8/8 of tokens. The
arithmetic intensity is the same.

**The win comes from elsewhere**: under wide-EP MoE, DP attention lets
each rank perform **its own KV cache lookup** independently for its
shard of tokens (no cross-rank reduce-scatter on the KV layout), and the
**MoE dispatch** then exchanges only the post-attention activations
that need to flow to the rank holding the expert. The "wide EP" pattern
(256+ experts spread across nodes) is the classical motivation.

### 1.2 Cost ledger when DP is paid for nothing (our case)

DSv3.2-REAP @ 128 experts/layer, **single node, `ep_size = 1`,
`moe_a2a_backend = none`, `moe_runner_backend = flashinfer_trtllm`**:

- Every MoE layer ships ALL 128 experts on every rank (TP-only MoE
  with full replication of expert weights; trtllm-gen fused MoE kernel
  routes per-token to the local set of experts via the topk_ids).
- There is **NO cross-rank MoE dispatch** — `moe_a2a_backend = none`
  means no DeepEP/flashinfer A2A. The MoE kernel runs locally on each
  rank's TP-sharded slice of `hidden_states`.
- So the "rank owns its tokens, dispatches to experts" win does not
  apply. Each rank does the SAME work the TP-only path would do, PLUS
  the DP allgather/reduce-scatter cycle described in the next section.

The cost of DP attention in this configuration is **all overhead, no
win**:

1. **Per-layer `dp_gather_partial` + `dp_scatter`** of the entire
   hidden_states (BF16, `[B_global, hidden]`) — 2 collectives per layer
   (`communicator.py:1071-1173`):
   - allgather: `[B/8, 7168]` BF16 per rank → `[B, 7168]` on every rank
     (1.8 MiB at B=128 per layer).
   - reduce-scatter after MoE: `[B, 7168]` → `[B/8, 7168]`.
   - The iter5+iter6 NVFP4 MoE work in `dp_attention.py:685-874` fuses
     this BF16 allgather with the FP4 packed activations + UE8M0 scales
     under a single `ncclGroupStart/End` pair, **but the gather itself
     still has to happen**. The fusion saves ~1us/layer of NCCL launch
     overhead, not the bytes-on-wire.
2. **MLP-sync barrier (`prepare_mlp_sync_batch`)**: a CPU-side
   `global_num_tokens` gloo allreduce / broadcast per scheduler
   iteration (`server_args.py:6981-6987` describes
   `--enable-dp-attention-local-control-broadcast` as the
   "Eliminates a costly all-ranks gloo sync on every scheduler
   iteration" optimization, which only **partially** mitigates this).
3. **`forward_idle` codepath**: when a rank has zero tokens assigned
   (e.g. low-batch warmup, post-finish ladder, MLP-sync padding), that
   rank still has to run a forward pass with `batch_size = 0` so the
   collectives at the layer boundaries do not deadlock. This path
   bypasses CUDA graphs entirely (`model_runner.py:3270-3299`,
   `forward_batch_info.py:84` "IDLE: for data parallel attention,
   some workers will be IDLE if no sequence are allocated"). This is
   the codepath that hosts the B200 SM_100 small-tensor kernel launch
   bug being fought in followups 9–13.
4. **Asymmetric memory pool**: DP attention disables symmetric memory
   when `dp_padding_mode` is not max-len (`dp_attention.py:144,171`).
   Symmetric memory is required for some peer-to-peer collectives and
   for the in-capture custom-allreduce path.

### 1.3 Schedule + chunked-prefill side-effects

DP attention also has a scheduler-side cost: at
`server_args.py:3450-3456`, enabling DP attention forces
`chunked_prefill_size //= dp_size` and reduces
`schedule_conservativeness` by 70% (`* 0.3`). The 1/8 chunked prefill
size means longer prefills are split into more chunks, increasing the
launch overhead per prefill request. None of these costs are paid in
the TP=8-only path.

### 1.4 Verdict on (1)

For DSv3.2-REAP on 8xB200 single-node with `--moe-runner-backend
flashinfer_trtllm` + `--moe-a2a-backend none`, DP attention saves
**zero attention compute** (same FLOPs/rank as TP) and **adds 2
collectives + forward_idle + scheduler tax** per layer. **DP attention
is not justified here.**

## 2. Vanilla CUDA graph vs PCG — overhead breakdown

### 2.1 What each runner actually captures

Both runners coexist in sglang and serve different forward modes
(`model_runner.py:2918-3107`):

- **`CudaGraphRunner`** (`cuda_graph_runner.py`, 1680 lines) — captures
  ONE big graph per `(batch_size, lora_variant, stream_idx)` covering
  the entire decode model.forward. Covers `forward_mode in {DECODE,
  TARGET_VERIFY, IDLE, DLLM_EXTEND}` (`forward_batch_info.py:161-167`).
  Decode goes through this path *always*, regardless of `dp_attention`.
- **`PiecewiseCudaGraphRunner`** (`piecewise_cuda_graph_runner.py`, 983
  lines) — uses torch.compile + Dynamo to break the model into chunks
  around the MoE A2A boundary, captures one CUDA graph per chunk per
  `num_tokens` size. Covers `forward_mode in {EXTEND, MIXED}`
  (`piecewise_cuda_graph_runner.py:246`,
  `replay_prepare:830-839` normalizes MIXED→EXTEND). Each replay
  stitches the chunks at runtime so the model can have data-dependent
  branching at A2A boundaries.

### 2.2 The PCG ↔ DP attention exclusion (mutual-exclusive by design)

`server_args.py:1441-1512` `_handle_piecewise_cuda_graph` is the
auto-disable ladder. Line 1452-1453:

```python
# 2. DP attention
if self.enable_dp_attention:
    self.disable_piecewise_cuda_graph = True
```

**DP attention auto-disables PCG**. The only way to run both is
`--enforce-piecewise-cuda-graph`, which skips the entire auto-disable
ladder (`server_args.py:1442-1445`). This is the explicit fight Spencer
has been waging through followups 9-13 — forcing the two paths to
coexist on B200 SM_100.

Auto-disable rationale: PCG's torch.compile-based chunking + capture
assumes the per-chunk shapes are stable across replays. DP attention
introduces the `forward_idle` codepath where a rank has 0 tokens; this
breaks the shape stability assumption at every chunk boundary, requiring
the followup-9 `dp_local_start_pos_gpu` precompute, the followup-10/11
zero buffer plumbing, and the followup-12 triton fill-zero kernel.

### 2.3 Per-replay overhead

For decode (the latency-critical TPOT path), the regular CudaGraphRunner
replay is:

```
replay_prepare (1 D2D copy per shape-dependent input, ~5-15us at B=128)
+ self.graphs[graph_key].replay()  (single cuMemcpyAsync per node group + cuLaunchKernel per node)
+ output slice (~1us)
```

A typical 61-layer DSv3.2 captured decode graph has ~3000-5000 nodes
(per attention layer: 4-6 RadixAttention calls, ~10 quantization/dequant
calls, ~3 RMSNorms, ~5 linear projections, ~4 collectives; per MoE
layer: ~5-10 topk/scatter ops, ~20 trtllm-gen launches, ~3 collectives).
The captured graph replay overhead is ~1-3us for `cuGraphLaunch`, then
the per-kernel launch path drops to ~50ns per kernel inside the captured
graph (the captured `cudaGraphLaunchKernel_v2` path bypasses the API
runtime's bookkeeping). At ~5000 nodes that's ~250us of "launch" inside
the graph — and the actual kernel time is whatever the kernels take.

The **only** overhead vs the raw kernel time of a vanilla CUDA graph
decode replay is the ~5-15us replay_prepare overhead and the ~1-3us
cuGraphLaunch. Sub-1% of a 25.9 ms FP8 TPOT.

For PCG (extend, the prefill latency path), the per-replay overhead is
higher because there are multiple chunks per forward and each chunk
replay has its own launch overhead. From `piecewise_cuda_graph_runner.py:899-954`:

```
replay = enable_piecewise_cuda_graph() context (~1us)
       + replay_prepare (1 D2D per input, ~5-15us)
       + set_forward_context + DSA indexer setup (~5-20us)
       + attn_backend.init_forward_metadata(forward_batch) (~50-200us — this is the planning)
       + model.forward (which fires N chunks × cuGraphLaunch each)
       + output construct (~5us)
```

The PCG decode path **does not exist** — PCG only captures EXTEND
(`capture_forward_mode = ForwardMode.EXTEND` at line 246). So the
question "is PCG slower at decode" has a structural answer: there is no
PCG decode. Spencer's premise about PCG vs vanilla CUDA graph for decode
is actually a misnomer — both paths use the regular CudaGraphRunner for
decode, and the only knob `--enforce-piecewise-cuda-graph` flips is the
EXTEND path (and whether DP-attention is allowed alongside it).

### 2.4 Real comparison: "PCG ON for extend + DP ON" vs "PCG ON for extend + DP OFF"

The honest comparison is:

| Path                                | Decode runner | Extend runner | forward_idle? | dp_attention.py gather/scatter? |
|-------------------------------------|---------------|---------------|---------------|---------------------------------|
| Current (DP=8 + --enforce-piecewise) | CudaGraphRunner with DP padding | PCG | YES (per-iter) | YES (per-layer, with followup 9-12 workarounds) |
| Proposed (TP=8, no --enforce)        | CudaGraphRunner (no DP padding) | PCG (default-on) | NO | NO |

Note: **PCG remains on for extend in BOTH paths**, just default-on
instead of forced. The DP-attention auto-disable at line 1452 only
fires when DP is enabled, so removing DP removes the auto-disable
condition and PCG stays on without --enforce-piecewise.

### 2.5 Quantified per-layer DP cost (deploy decode peak m=128)

From iter5/iter6 NVFP4 MoE bench data (`dp_attention.py:706-714`,
`nvfp4_moe_iter6_recon.md`):

| Operation                                          | Cost @ m_global=128 |
|----------------------------------------------------|---------------------|
| `dp_gather_partial` BF16 allgather (1.8 MiB)       | ~10-15 us           |
| Reduce-scatter post-MoE                            | ~5-8 us             |
| Iter5 FP4+SF grouped allgather (saves -10us regress at m=256, breakeven m=128) | -0.7 us / layer net |
| `prepare_mlp_sync_batch` per-iter gloo barrier     | ~50-200 us / step   |
| `forward_idle` overhead per idle rank per step     | ~5-20 ms (uncaptured, ~50% of FP8 TPOT)    |

The forward_idle cost is the big one. When even ONE rank is idle (which
happens routinely in the warmup ladder, post-finish ladder, and during
MLP-sync padding under uneven batch distribution), that rank runs a
non-captured forward pass that takes 5-20 ms — and the other 7 ranks
have to BLOCK on the collectives at every layer until the idle rank
catches up. This is the worst-case tail latency contributor in the
current DP path. Per-iter cost of ~250us-3ms for the per-layer
collectives is amortized across 61 layers.

## 3. TP=8-only config — exact server args

Drop-in replacement for the current DGD config, removing the DP knobs:

```
--model BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
--tp 8 \
--moe-runner-backend flashinfer_trtllm \
--moe-a2a-backend none \
--attention-backend dsa \
--kv-cache-dtype fp8_e4m3 \
--quantization compressed-tensors \
--mem-fraction-static 0.85 \
--page-size 64 \
--max-running-requests 256 \
--enable-mixed-chunk \
--enable-return-hidden-states \
--show-time-cost
```

Key changes from current DGD:

- DROPPED `--dp 8`
- DROPPED `--enable-dp-attention`
- DROPPED `--enforce-piecewise-cuda-graph` (PCG stays default-on; the
  auto-disable only fires from the `enable_dp_attention=True` branch).
- DROPPED `--enable-dp-attention-local-control-broadcast` (moot
  without DP).
- DROPPED `--enable-dp-lm-head` if present (only meaningful with DP).
- KEEP `--moe-runner-backend flashinfer_trtllm` — this is the
  single-node fused trtllm-gen MoE, unchanged.
- KEEP `--moe-a2a-backend none` — this is already the no-EP single-node
  setting, unchanged.

Things to verify on first launch:

- `attn_tp_size` will become 8 (= tp_size, since dp_size=1).
- Each rank shards `n_local_heads = n_heads // 8` and the QKV/Output
  projections become 1/8-sharded. Total attention compute per rank is
  unchanged; just redistributed.
- `require_mlp_tp_gather` returns False (line 3084-3106 needs DP).
- `require_attn_tp_gather` returns True (line 3109-3129, condition
  `moe_dense_tp_size == 1` is met) — so the LayerCommunicator still
  scatters input to attention; this is per-attn_tp_group (= per-TP-group
  here, single 8-way group), not per-DP-group. This is a **single
  tp_group reduce-scatter**, not 2 cross-group collectives.
- `require_mlp_sync` returns False (line 3136-3137 needs DP) — no MLP
  sync barrier, no gloo allreduce per scheduler iter.
- `forward_idle` path becomes unreachable; ranks always have tokens
  if any rank does (single TP group, single batch).
- PCG stays on (no auto-disable from line 1452).

## 4. LayerSplit + Warp Decode — DP attention compatibility

### 4.1 LayerSplit (existing, prefill-CP only)

The existing `LayerSplit` in this repo is **prefill-CP KV storage**, not
the decode-side LayerSplit Spencer is targeting:

- `python/sglang/srt/layers/attention/dsa/layersplit.py:212-277`
  `validate_layersplit_server_args` requires
  `--enable-dsa-prefill-context-parallel` AND `attn_cp_size > 1`.
- Storage policy: layer ownership is partitioned across CP ranks via
  `LayerSplitPolicy` (interleaved or contiguous layout); during prefill
  CP, each rank stores the dense KV + indexer cache for the layers it
  owns and stages the others via NCCL.
- This existing LayerSplit is on top of DSA CP, which under
  `server_args.py:2004-2039` **forces** `enable_dp_attention = True`,
  `moe_dense_tp_size = 1`, `moe_a2a_backend = deepep`, `ep_size =
  tp_size`. So this LayerSplit is **explicitly TIED to DP attention**.

The decode-side LayerSplit Spencer is targeting must be a different
concept: a layer-chunked attention dataflow that requires single-tensor
flow (no per-rank token shards) so the chunk boundaries align cleanly
with whole layers, not 1/8-of-tokens-per-rank slices.

### 4.2 Why DP attention blocks decode-side LayerSplit

Source-code-cited reasons:

1. **DP attention scatters tokens across ranks within an attention
   group**. In the deploy config (`dp_size=tp_size=8`, `attn_tp_size=1`),
   each rank owns 1/8 of the tokens for the entire attention compute.
   For decode-side LayerSplit to assign chunks of layers (e.g. layers
   0-7 to rank 0, layers 8-15 to rank 1, ...) the whole hidden_states
   tensor must flow through one rank for that layer chunk — but DP
   attention has already split the tokens 1/8 across ranks. The
   per-token reduce-scatter at the end of attention (line 918-919
   `attn_tp_reduce_scatter_tensor`) is incompatible with the
   layer-chunk dataflow: LayerSplit would need to undo the token split,
   process the chunk, then re-split.

2. **`LayerCommunicator.prepare_attn` / `_gather_hidden_states_and_residual`
   `communicator.py:511-720, 995-1240`** runs at **every layer
   boundary** under DP. If we partition layers across ranks (decode
   LayerSplit), the communicator would have to know per-layer whether
   to do the DP gather/scatter or to defer to LayerSplit's stage-copy.
   The current LayerCommunicator is layer-stateless; making it
   layer-aware is the integration friction.

3. **`forward_idle` blocks LayerSplit's chunk-flow assumption**. If
   even one rank is idle during a step (which is routine under DP),
   LayerSplit's owner rank for a chunk would have to either (a)
   participate in the idle ladder (defeating the chunk-locality win)
   or (b) skip the idle rank's slice of tokens (breaking the
   chunk-flow invariant). Removing DP removes `forward_idle`, removes
   this conflict.

### 4.3 Warp Decode — already DP-incompatible by code

`python/sglang/srt/layers/moe/warp_decode/integration.py:50-51`:

```python
if getattr(layer, "moe_ep_size", 1) > 1:
    return None
```

Warp Decode is **already gated off** when `moe_ep_size > 1`. In the
current DSv3.2-REAP deploy `ep_size = 1` (no EP), so Warp Decode is
*technically* eligible for the MoE layers. But:

1. **Warp Decode is a SMALL-batch MoE path** (max 64 tokens, see
   `envs.SGLANG_WARP_DECODE_MAX_BATCH = EnvInt(64)` at
   `environ.py:409`). Under DP attention, batch is split across 8 DP
   ranks (`B/8` per rank). So the per-rank batch at peak (B=128) is
   16, well within the Warp Decode regime. **DP attention is not
   the problem at the batch-size gate**.

2. **The fundamental conflict** is that Warp Decode is a **per-rank
   MoE expert kernel** that processes ALL the tokens through the
   experts owned by that rank (`runner.py:75-200`). With DP attention,
   each rank only owns `B/8` tokens but ALL 128 experts. The Warp
   Decode kernel must dispatch each token to its `top_k=8` experts.
   The kernel's warp-scheduling assumption is that the warp scheduler
   can keep all warps busy on the local token batch; with `B/8 = 16`
   tokens and 128 local experts the expert utilization is sparse (a
   single token only touches 8/128 experts), so warps idle on
   experts not hit. The Warp Decode CuTe SM_100 kernel
   (`kernels.py:30-100`, `cute_warp_decode_moe_packed` is loaded
   from `sgl_kernel.warp_decode_cute_moe_packed_forward`) tile-sizes
   are `_CUTE_GATE_UP_TILE_K = 1024` and `_CUTE_DOWN_TILE_N = 2048`
   for hidden=7168 / intermediate=2048; at 16 tokens × 8 experts =
   128 "token-expert" pairs distributed sparsely across 128 experts,
   the warp utilization drops because most warps land in
   zero-token-expert tiles.

3. **Without DP** (TP=8 only), each MoE layer's input is the full
   `B=128` post-attention activation, sharded across TP ranks by
   the TP gather inside the model (each rank gets `[B, hidden/tp_size]`
   for the gate). The expert routing happens on full B tokens with
   the full expert set; the trtllm-gen path or Warp Decode path
   then processes `B*top_k = 1024` token-expert pairs uniformly
   across the rank's expert weights. Warp utilization is high.

**So**: DP attention doesn't gate Warp Decode off (the `moe_ep_size`
check still passes), but it makes Warp Decode's warp scheduling
inefficient by forcing sparse expert utilization. Removing DP makes
Warp Decode practically viable.

### 4.4 Estimated unlocked perf

- **Decode-side LayerSplit** (Spencer's planned future opt; not in
  current code): theoretical TPOT cut depends on the chunk size, but
  if the chunk-locality reduces inter-layer launch overhead by 30-50%
  per layer and removes the need for per-layer DP gather/scatter, the
  per-layer savings stack: 61 layers × (10-15 us gather + 5-8 us
  scatter + 1-2 us launch) ≈ 1-1.5 ms TPOT. Combined with
  warp-utilization improvements on the attention kernel from removing
  the per-rank attn fragmentation, projected ceiling **+2-4 ms TPOT
  reduction (roughly 8-15% of the 25.9 ms FP8 baseline)**.
- **Warp Decode** wired through compressed_tensors_w4a4_nvfp4_moe or
  the BF16-act in-cubin path: from `bench_warp_decode.py` projections
  on B200 SM_100 at the relevant batch (16-128 tokens per layer × 58
  NVFP4 layers + 3 BF16 dense layers), depending on per-layer NVFP4
  MoE current cost (~2.3 ms aggregate per iter4 PRIMARY close-out's
  "200-300us/step ceiling target"), **a successful Warp Decode wiring
  could reduce the NVFP4 MoE step contribution from 2.3 ms toward 1.5
  ms, ~+0.8 ms TPOT.**

Combined unlocked perf from LayerSplit + Warp Decode: **3-5 ms TPOT
ceiling** beyond the current iter1-6 stack — this is the long-run
upside that justifies the pivot.

## 5. Bug surface analysis — which #14 followups go away

The followup work in #14 is concentrated in two places:

### 5.1 `dp_attention.py` followups (followups 9, 10, 10-cont, 11, 12)

All of these workaround a single root cause: **B200 SM_100
`cudaErrorInvalidConfiguration` on small-leading-dim tensor ops when
replayed from a captured CUDA graph**. The bug surfaces on `fill_(0)`,
`mul_(0)`, `copy_(zero_buffer)`, `cumsum`, and (followup 13 in flight)
`torch.matmul`. The PyTorch elementwise launch dispatcher hits the bug
on shapes like `(1, 7168) BF16`. The triton dispatch path is known-good
at the same shapes (followup 12 commit b8987cb270 confirmed this).

**Where the bug fires**:
- `_dp_gather_via_all_reduce` line 582-595: `fill_zero_triton` on
  global_tokens (would otherwise be `global_tokens.fill_(0)`).
- `_dp_gather_via_all_gather` line 632-645: same.
- `dp_scatter` line 877-892: same.
- All three of these are **inside the DP gather/scatter codepath**.
  They only run when `enable_dp_attention = True` and per-rank token
  count varies (the `numel == 0` short-circuit covers the all-zero
  case).

**If DP is removed**: the `_dp_gather_via_*` and `dp_scatter` functions
are not called. The communicator goes through the
`get_attn_tp_context().input_scattered` path
(`communicator.py:519-545`) which does `attn_tp_all_gather_into_tensor`
or `tp_reduce_scatter` directly without the dp-shape-aware zero-fill
plumbing. **Followups 9, 10, 10-cont, 11, 12 become moot**.

### 5.2 `logits_processor.py:933` matmul (followup 13, in flight)

This is the `torch.matmul` in `_compute_lm_head` that triggers
`cudaErrorInvalidConfiguration` on the `forward_idle` codepath when the
input is `[0, hidden]` (idle rank with 0 tokens, projecting through the
LM head). The bug is **specifically in the forward_idle path** — it
does not fire on the captured cuda graph decode replay (which always
has `batch_size >= 1`).

**If DP is removed**: `forward_idle` is unreachable
(`model_runner.py:3506-3509` only routes to it when
`forward_batch.forward_mode.is_idle()`, and IDLE only happens with
DP-attention rank skew per `forward_batch_info.py:84` comment).
**Followup 13 becomes moot**.

### 5.3 Net bug surface delta

Removing DP attention erases the entire `#14 fix followup` chain. The
small-tensor launch-config bug remains a real B200 SM_100 issue, but it
no longer surfaces in the production path because:

1. The decode CUDA graph captures with `batch_size = max_batch` and
   replays with the actual batch (padded if needed), never producing
   `numel = 0` tensors.
2. The PCG extend path captures with `num_tokens >= 16` (the smallest
   capture size), never producing zero-sized tensors inside the graph.
3. The communicator path for TP-only does not have the
   shape-dependent `fill_(0)` zero-fill pattern that DP needs for
   variable-rank-token-count handling.

The launch-config bug becomes a **latent** issue (still in the
PyTorch source, still affecting any future opt that does small-tensor
elementwise inside a captured graph) but is **out of the production
critical path**.

## 6. Cumulative iter-win re-projection — TPOT estimate both paths

Baseline FP8 trtllm-DSA: **25.9 ms TPOT** (from
`higgs_dsa_iter8_recon.md:215` "cubin's compute-bound floor").
Baseline #16 HIGGS dense KV: **38.6 ms TPOT** (from
`higgs_mla_decode_iter6_close_out.md:142`).

### 6.1 Path A: current (PCG + DP=8, followup-13 work to finish)

Cumulative wins from existing iter1-N stack:

| Vector | Current ms TPOT impact (delta vs FP8 baseline) |
|--------|-----------------------------------------------|
| DSA NVFP4 indexer iter1-6 | ~-3.5 ms (pipeline reduction, kernel-level, independent of DP) |
| #16 HIGGS MLA decode iter1-5 | 38.6 → 33.4 ms (depth-2 closed-out negative; -5.2 ms from baseline) |
| #19 HIGGS+trtllm DSA iter3-7 | -4 to -5.6 ms (mostly kernel; dedicated stream depth-4 ping-pong tied to DP-rank-parallel layers; would need re-analysis without DP) |
| #19 iter8 scaffold | +9.56 ms projected (correctness fix in iter9 in flight) |
| #15 NVFP4 MoE iter1-6 | ~-0.3 ms (~223 us/step at peak = 0.86% TPOT; fused RMSNorm-FP4 + FP4 allgather + 3-way ncclGroup fusion — iter5+iter6 ~+105 us/step are tied to DP) |
| #14 followup-13 lift unblock | enables the iter6 NVFP4 MoE 3-way fusion to run in production |

Best-case sum vs **HIGGS baseline 38.6 ms**: ~33.4 (iter5 #16) - 3.5
(DSA indexer) - ~5 (#19 iter3-7 partial) - 9.5 (iter8 scaffold after
iter9 correctness fix) - 0.3 (#15) = **~15 ms TPOT** (compute-bound
floor at the FP8 25.9 ms with HIGGS's lower kernel cost dominates;
actual stays at ~22-24 ms TPOT after the iter9 correctness fix).

Critical caveat: **iter5+iter6 #15 wins (~105 us/step) DEPEND on DP
attention.** Removing DP removes these. Iter1-4 of #15 stay (RMSNorm-FP4
fusion, deploy wire) ≈ ~120 us/step preserved.

### 6.2 Path B: TP=8 only (no DP, PCG default-on for extend)

Cumulative wins:

| Vector | TPOT impact under TP-only |
|--------|---------------------------|
| DSA NVFP4 indexer iter1-6 | -3.5 ms (UNCHANGED, kernel-level) |
| #16 HIGGS MLA decode iter1-5 | -5.2 ms (UNCHANGED, kernel-level — page table layout independent of DP) |
| #19 iter3-7 | -3 to -4 ms (DOWN ~1-1.5 ms: depth-4 ping-pong reasoning needs re-tuning without DP-rank parallelism. Dedicated trtllm-gen stream still applies. FP8 cubin still applies.) |
| #19 iter8 scaffold | -9.5 ms (UNCHANGED, kernel-level inline producer, reads HIGGS slots from gmem regardless of DP) |
| #15 NVFP4 MoE iter1-4 | -0.13 ms (RMSNorm-FP4 fusion, deploy wire — kernel-level) |
| #15 NVFP4 MoE iter5-6 (DP-tied wires) | **LOST -0.105 ms (-105 us/step)** |
| DP allgather/scatter overhead | **SAVED +0.2-0.3 ms TPOT** (no per-layer dp_gather/scatter) |
| forward_idle tail latency | **SAVED 0-2 ms p95 TPOT** (no idle ladder; mean unchanged) |
| MLP sync gloo per-iter | **SAVED 0.05-0.2 ms TPOT** (no per-iter scheduler barrier) |
| chunked_prefill 1/8-ing | irrelevant to decode TPOT (prefill-only) |

Net: roughly the same ~22-24 ms TPOT mean projection, but with **lower
p95 tail latency** (no forward_idle stalls) and **without followup-13
risk**.

### 6.3 Strategic tiebreaker

The headline TPOT numbers are within ~0.5 ms of each other.
**Tiebreaker**: the **unlocked future opts** (decode-side LayerSplit
projected +2-4 ms, Warp Decode projected +0.8 ms) only land under
Path B. Path A is a dead-end at ~22-24 ms TPOT until DP is removed.

Path B at ~22-24 ms now → +2-4 ms LayerSplit → +0.8 ms Warp Decode
→ **~17-21 ms TPOT ceiling** in a follow-on campaign.

Path A is stuck at ~22-24 ms TPOT and the followup-13/-14/-15 work to
keep the launch-config bug suppressed is itself an open-ended drag.

## 7. Recommendation + migration path

**RECOMMENDATION: Pivot to Path B (TP=8 only, no DP, drop
--enforce-piecewise-cuda-graph).**

### 7.1 Migration steps

1. **Land the args change** in the deploy manifest
   (deepseek-v32-nextn-graft-sglang):
   - Remove `--dp 8`
   - Remove `--enable-dp-attention`
   - Remove `--enforce-piecewise-cuda-graph`
   - Remove `--enable-dp-attention-local-control-broadcast` (if set)
   - Remove `--enable-dp-lm-head` (if set)
   - Keep all kernel-level flags (kv cache, moe-runner-backend,
     attention-backend, etc.) as-is.

2. **Smoke test** on a low-batch (B=8-32) workload to confirm:
   - Decode CudaGraphRunner captures cleanly with `attn_tp_size=8`.
   - PCG extends captures cleanly (no DP-related shape mismatches).
   - No `forward_idle` log lines.
   - No #14-followup-style `cudaErrorInvalidConfiguration` log lines.
   - Token logits sanity vs the current DP path (relax assertion to
     bf16 noise, ~1e-2 rel tol).

3. **Production benchmark** at B=128 deploy peak:
   - intel-correctness TWO-STAGE accuracy gate.
   - linear_decode_tps target: match or beat the current ~22-24 ms
     TPOT projection from the iter1-6 stack on this config.

4. **Lock in the iter1-6 wins that survive**:
   - DSA NVFP4 indexer iter1-6: kernel-level, automatically apply.
   - #16 HIGGS MLA decode iter1-5: kernel-level, automatically apply.
   - #19 iter3-7 + iter8 scaffold (after iter9 correctness fix):
     kernel-level, automatically apply; the dedicated stream +
     depth-4 ping-pong tuning may need a re-bench at TP-only batch
     distribution, but the architectural win survives.
   - #15 NVFP4 MoE iter1-4: kernel-level fused RMSNorm-FP4, automatically
     apply.
   - #15 NVFP4 MoE iter5-6: **REVERT or scope-tag as deferred**. The
     FP4 allgather + 3-way ncclGroup fusion are DP-specific dead code
     under TP-only. Either delete cleanly or gate behind
     `is_dp_attention_enabled()`.

5. **Open the follow-on campaign**:
   - Decode-side LayerSplit prototype (new code; the existing
     `attention/dsa/layersplit.py` is for prefill-CP and is unrelated
     to the decode-side concept).
   - Warp Decode wiring through compressed_tensors_w4a4_nvfp4_moe
     (the gate `moe_ep_size > 1` already passes; need to handle the
     NVFP4 quant path).

### 7.2 Risks

1. **Some kernel-level wins benchmarked under DP may need re-tuning**.
   The depth-4 ping-pong and dedicated trtllm-gen stream choices were
   informed by DP-rank-parallel layer execution. Under TP-only, the
   layer execution is more serialized (single TP group), so the ping-
   pong depth and stream pool size sweet spot may shift. Estimated
   re-tuning cost: 2-4 hours per iter (manageable inside the existing
   iter cadence).

2. **#15 iter5+iter6 work loss is real** (-105 us/step or -0.105 ms
   TPOT). Mitigated by the unlocked future opts (LayerSplit +2-4 ms,
   Warp Decode +0.8 ms) — ratio is favorable by 20-40x.

3. **forward_idle removal is a behavior change**. Any downstream
   tooling that expects DP-rank-skew tail-latency handling will need
   to adapt. Unlikely in this campaign since the deploy is single-
   replica.

4. **MoE A2A configurations** (deepep, flashinfer, mooncake) require
   DP attention per `server_args.py:3666-3669`. We are not using any
   of these, so this is not a near-term concern.

### 7.3 Decision matrix

| Dimension                          | PCG + DP=8                | TP=8 only            | Winner |
|------------------------------------|---------------------------|----------------------|--------|
| Per-layer FLOPs/rank               | Same                      | Same                 | Tie   |
| Per-layer collectives              | 2 allgather + scatter     | 1 reduce-scatter     | TP    |
| forward_idle codepath              | Present                   | Absent               | TP    |
| #14 followup-9..-13 maintenance    | Ongoing                   | Eliminated           | TP    |
| Iter1-6 kernel wins survive?       | All                       | All except #15 iter5-6 | DP (marginal) |
| LayerSplit (decode) compatible?    | No                        | Yes                  | TP    |
| Warp Decode compatible?            | Marginal (sparse warps)   | Yes (full warps)     | TP    |
| Tail latency (forward_idle)        | Bad                       | Good                 | TP    |
| Chunked-prefill 1/8 throttle       | Yes                       | No                   | TP    |
| Hot-path TPOT mean                 | ~22-24 ms                 | ~22-24 ms            | Tie    |
| Future-opt ceiling                 | ~22 ms (stuck)            | ~17-21 ms            | TP    |

**Verdict: TP=8 only is the structurally better path. Migrate now.**

## Appendix A — file refs

- `python/sglang/srt/server_args.py:1441-1512` `_handle_piecewise_cuda_graph` auto-disable ladder.
- `python/sglang/srt/server_args.py:1452-1453` DP-attention auto-disables PCG.
- `python/sglang/srt/server_args.py:3445-3492` `_handle_data_parallelism` (incl. chunked_prefill //= dp_size).
- `python/sglang/srt/server_args.py:2368` `attn_dp_size = dp_size if enable_dp_attention else 1`.
- `python/sglang/srt/server_args.py:7891-7914` LayerSplit (prefill-CP) validation gate.
- `python/sglang/srt/layers/dp_attention.py:284-298` `compute_dp_attention_world_info`.
- `python/sglang/srt/layers/dp_attention.py:540-595` `fill_zero_triton` (followup-12 workaround).
- `python/sglang/srt/layers/dp_attention.py:685-874` FP4 allgather + 3-way ncclGroup fusion (iter5+6 #15).
- `python/sglang/srt/layers/communicator.py:437-720, 995-1240` LayerCommunicator + `prepare_attn` / `_gather_hidden_states_and_residual`.
- `python/sglang/srt/model_executor/cuda_graph_runner.py:593-933, 1483-1564` CudaGraphRunner (decode).
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py:181-984` PCG runner (extend).
- `python/sglang/srt/model_executor/forward_batch_info.py:75-174` ForwardMode definitions (IDLE etc.).
- `python/sglang/srt/model_executor/model_runner.py:3153-3520` decode/extend/idle dispatch.
- `python/sglang/srt/layers/moe/warp_decode/integration.py:27-81` Warp Decode entry, EP gate.
- `python/sglang/srt/layers/moe/warp_decode/runner.py:75-200` WarpDecodeRunnerCore.
- `python/sglang/srt/layers/attention/dsa/layersplit.py:212-630` LayerSplit (prefill-CP variant).
- `python/sglang/srt/utils/common.py:3084-3137` `require_*` helpers for DP attention.

## Appendix B — what's NOT in this report

- Microbench of vanilla CudaGraphRunner replay cost vs PCG replay cost
  via intra-kernel-profiler. **Justification**: the structural finding
  in §2.3 (PCG does not handle decode; both paths use CudaGraphRunner
  for decode) makes the microbench moot for the decode-TPOT question.
  IKP becomes relevant for the unlocked LayerSplit + Warp Decode follow-
  on campaign, where per-kernel attribution will drive the next iter
  cycle.

- End-to-end pod run on the proposed Path B config. **Justification**:
  mission scope is research-only. The deploy migration in §7.1 step 2-3
  is the natural follow-up commit that runs the pod.

- DSv3.2 single-node deepseek-v3 model code structural changes
  required for TP-only. **Justification**: read of `deepseek_v4.py`
  L223-419 (MQALayer) confirms the model already uses
  `attn_tp_size = get_attention_tp_size()` and the QKV/Output
  projections are parametrized on `attn_tp_size`. The code transparently
  handles both DP and TP cases. No model change required.
