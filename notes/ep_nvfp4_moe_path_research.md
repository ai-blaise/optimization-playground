# EP=8 path for DeepSeek-V3.2-REAP-345B NVFP4 MoE on 8xB200 — research

Author: Spencer Garnets
Date: 2026-05-30
Status: research only; no production code changes in scope. Decision input
for the EP=8 follow-up to the TP=8 vanilla-CUDA-graph deploy on `1d7deebee`.

## 0. Executive summary

**Recommendation: stop using `--moe-a2a-backend deepep` for the EP=8
follow-up. Switch to `--moe-runner-backend flashinfer_cutlass
--moe-a2a-backend none --ep-size 8`** and fix the latent
`skip_local_expert_mapping` mismatch in
`python/sglang/srt/layers/moe/token_dispatcher/standard.py:96-105` so the
compressed-tensors NVFP4 cutlass-moe-fp4 branch sees correctly-mapped
`topk_ids`.

The blocking finding is structural: the four MoE-runner backends that can
consume `DeepEPLLDispatchOutput` and the four backends that can apply
`compressed-tensors` NVFP4 weights have an **empty intersection** in
sglang HEAD `1d7deebee`. Specifically:

| MoE-runner backend       | Has DeepEP fused func? | Accepts `compressed-tensors` NVFP4? |
|--------------------------|------------------------|-------------------------------------|
| `flashinfer_trtllm`      | **NO** (only `("none", "flashinfer_trtllm")`) | YES |
| `flashinfer_trtllm_routed` | NO                   | NO (server\_args limits to fp8/mxfp8/modelopt\_fp4) |
| `flashinfer_cutlass`     | NO (no fused func registered) | YES |
| `flashinfer_cutedsl`     | **YES** (`("deepep", "flashinfer_cutedsl")`) | **NO** (server\_args limits to modelopt\_fp4) |
| `flashinfer_mxfp4`       | NO                     | NO (MXFP4 only) |
| `deep_gemm`              | YES (via pre/post permute) | NO (FP8-block-scale or MXFP4 only) |
| `triton`                 | NO                     | NO (forces ep\_size==1) |
| `cutlass`                | NO                     | NO (forces ep\_size==1) |
| `aiter`, `marlin`, `warp_decode` | NO             | NO                                |

The only existing kernel-level entry point that handles
`DeepEPLLDispatchOutput` for NVFP4 weights is
`@register_fused_func("deepep", "flashinfer_cutedsl")` at
`python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py:467`. It is
gated to `modelopt_fp4` weights in
`python/sglang/srt/server_args.py:3559-3565`. The REAP-345B checkpoint
ships `compressed-tensors`/`NVFP4A16` and routes to
`CompressedTensorsW4A4Nvfp4MoE` per
`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py:818-819`,
so it cannot reach that fused func.

Three usable paths to EP=8 exist, in decreasing priority:

1. **Path A — `flashinfer_cutlass --moe-a2a-backend none --ep-size 8`**
   (recommended). Latent `skip_local_expert_mapping` bug must be fixed.
   ~2-line server-args repath plus a ~5-line dispatcher fix.
2. **Path B — adapt the compressed-tensors scheme to consume
   `DeepEPLLDispatchOutput` and call `flashinfer_cutedsl_moe_masked`**,
   i.e. port modelopt's `_is_cutedsl_v1_deepep` path into
   `CompressedTensorsW4A4Nvfp4MoE.apply_weights` plus its
   `process_weights_after_loading`. ~80 LOC of code change, weight-layout
   re-derivation, requires backing out the trtllm fast path.
3. **Path C — keep `flashinfer_trtllm` and patch the
   compressed-tensors scheme so it route per-rank by carving
   `local_expert_offset` + `local_num_experts` weights at load time,
   driven by `--ep-size 8` with `--moe-a2a-backend none`**. The trtllm
   kernel already supports EP via `local_expert_offset` (it computes the
   routing internally and ignores experts outside the local range).
   The fast path remains alive. ~20 LOC of scheme change.

Path C is the smallest behavioural delta but does **not** rely on
DeepEP at all (no cross-rank dispatch). Path A is the canonical
"flashinfer_cutlass internal EP" path that modelopt uses. Path B is
the canonical "deepep + per-rank kernel" path that modelopt uses for
`_is_cutedsl_v1_deepep`. **Path C is the lowest-risk, smallest-diff
unlock; Path A is the next-best if a cross-rank A2A is wanted; Path B
is preserved as a fallback if DeepEP A2A is required for the next
multi-node deploy.**

The DeepEP A2A on a **single 8xB200 node** buys very little: there is
no inter-node NVLink boundary to cross, expert tokens still travel
over the same NVLink as the all-reduce would, and the per-layer
dispatch+combine pair is ~6 us each at the production batch (vs
~1 us for the all-reduce path). The reason to want EP=8 here is
**memory** (each rank holds 16 experts instead of 128 routed
experts), not communication. Path C delivers the memory saving
without the DeepEP serialization tax.

## 1. The failing site, in full

`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py:329`:

```python
    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output         # <-- AttributeError under DeepEP
```

The type-hint says `StandardDispatchOutput`, which has a `topk_output`
field (`python/sglang/srt/layers/moe/token_dispatcher/standard.py:59-65`).
Under `--moe-a2a-backend deepep`, the dispatcher returns
`DeepEPLLDispatchOutput`
(`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:99-110`):

```python
class DeepEPLLDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int
```

There is no `topk_output` and the underlying `BypassedTopKOutput`
(carrying `router_logits` + `topk_config` needed by trtllm-gen) has
already been materialized inside `dispatch_a` by the
`isinstance(topk_output, BypassedTopKOutput): topk_output.to_standard()`
line that landed in `1d7deebee`
(`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:633-637`).
Once routing is done host-side and DeepEP has shuffled tokens across
ranks, the trtllm-gen fused kernel **cannot run** — it does its own
routing internally and the per-rank token bag no longer carries the
global `router_logits` it needs.

Two consumers in the scheme need different inputs:

- trtllm fast path (`self.use_flashinfer_trtllm` true,
  `compressed_tensors_w4a4_nvfp4_moe.py:343-454`): needs `router_logits`
  + `topk_config` (the unmaterialized `BypassedTopKOutput` form). All
  routing happens inside `trtllm_fp4_block_scale_moe`.
- cutlass fallback (`else`, `compressed_tensors_w4a4_nvfp4_moe.py:455-475`):
  needs `topk_weights` + `topk_ids`. Routing is host-side.

Under EP+DeepEP the host-side routing has already happened, so the
trtllm fast path is fundamentally incompatible with `--moe-a2a-backend
deepep`. The cutlass fallback would work if `topk_output.topk_weights`
and `topk_output.topk_ids` could be read off the `DeepEPLLDispatchOutput`
directly — but they live on the dispatch tuple, not on a `topk_output`
sub-attribute.

## 2. Runner-backend × dispatcher × NVFP4-scheme matrix

### 2.1 `@register_fused_func` registrations in HEAD `1d7deebee`

| File:line                                                                              | Decorator key                              | What it consumes               |
|----------------------------------------------------------------------------------------|--------------------------------------------|--------------------------------|
| `moe_runner/triton.py:139`                                                             | `("none", "triton")`                       | `StandardDispatchOutput`       |
| `moe_runner/marlin.py:76`                                                              | `("none", "marlin")`                       | `StandardDispatchOutput`       |
| `moe_runner/flashinfer_trtllm.py:1158`                                                 | `("none", "flashinfer_trtllm")`            | `StandardDispatchOutput`       |
| `moe_runner/flashinfer_trtllm.py:1182`                                                 | `("none", "flashinfer_trtllm_routed")`     | `StandardDispatchOutput`       |
| `moe_runner/flashinfer_cutedsl.py:355`                                                 | `("none", "flashinfer_cutedsl")`           | `StandardDispatchOutput`       |
| `moe_runner/flashinfer_cutedsl.py:401`                                                 | `("flashinfer", "flashinfer_cutedsl")`     | `FlashinferDispatchOutput`     |
| `moe_runner/flashinfer_cutedsl.py:467`                                                 | `("deepep", "flashinfer_cutedsl")`         | `DeepEPLLDispatchOutput`       |

`deep_gemm` does not use `register_fused_func`. It uses
`register_pre_permute("deepep_ll", "deep_gemm")` +
`register_post_permute("deep_gemm", "deepep_ll")`
(`moe_runner/deep_gemm.py:657-690`) which run *around* the
`DeepGemmRunnerCore.run`. That path supports `DeepEPLLDispatchOutput`,
but **only** for FP8-block-scale weights (or MXFP4 with
`is_fp4_experts=True` selecting `(1,128)/(1,32)` recipes —
`moe_runner/deep_gemm.py:120-189`), not for NVFP4 W4A4.

### 2.2 `--moe-runner-backend` × `--quantization` allowed combinations

From `python/sglang/srt/server_args.py:3500-3600`:

| Runner backend             | Allowed `--quantization` values                                                                     | EP supported (per server\_args) |
|----------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------|
| `flashinfer_trtllm`        | `modelopt_fp4`, `fp8`, `mxfp8`, `modelopt_fp8`, `modelopt_mixed`, **`compressed-tensors`**, None    | `ep_size ∈ {1, tp_size}`        |
| `flashinfer_trtllm_routed` | `fp8`, `mxfp8`, `modelopt_fp4`, None                                                                | `ep_size ∈ {1, tp_size}`        |
| `flashinfer_cutlass`       | `modelopt_fp4`, `modelopt_fp8`, `modelopt_mixed`, **`compressed-tensors`**, None                    | `ep_size ∈ {1, tp_size}`        |
| `flashinfer_cutedsl`       | `modelopt_fp4` only                                                                                 | `ep_size ∈ {1, tp_size}`        |
| `cutlass`                  | `fp8`, `mxfp8`                                                                                      | `ep_size == 1` (hard assert)    |
| `triton`                   | any                                                                                                 | `ep_size == 1` (server\_args:273-277) |

Only `flashinfer_trtllm` and `flashinfer_cutlass` are both permitted
for `compressed-tensors` AND can run with `ep_size = tp_size = 8`.
`flashinfer_cutedsl` is the only NVFP4 runner with a DeepEP fused
func, but it's locked to `modelopt_fp4` weight format. The REAP-345B
checkpoint is `compressed-tensors` (its `config.json` says
`"quant_method": "compressed-tensors"`,
`"format": "nvfp4_e2m1_ue8m0"`), so without a checkpoint reformat
`flashinfer_cutedsl` is unreachable.

### 2.3 `compressed_tensors` MoE scheme dispatch (what kernel actually runs)

`CompressedTensorsW4A4Nvfp4MoE` reads
`get_moe_runner_backend().is_flashinfer_trtllm()` once in `__init__`
(`compressed_tensors_w4a4_nvfp4_moe.py:69`) and stores the bool on
`self.use_flashinfer_trtllm`. Everything in
`process_weights_after_loading` and `apply_weights` branches on that
bool:

- **`use_flashinfer_trtllm = True`** → builds shuffled w13/w2 weights
  for trtllm-gen layout in `process_weights_after_loading:226-279`,
  calls `flashinfer.trtllm_fp4_block_scale_moe` in `apply_weights:343-454`.
- **else** → swizzles blockscales in
  `process_weights_after_loading:280-296`, builds `cutlass_moe_params`,
  calls **sglang-native** `cutlass_moe_fp4` (NOT flashinfer's
  `cutlass_fused_moe`) in `apply_weights:455-475`.

So with `--moe-runner-backend flashinfer_cutlass`, the scheme
actually runs **sglang-native `cutlass_moe_fp4`**, not flashinfer's
fused MoE. This is documented in
`python/sglang/srt/server_args.py:3510-3515` (Spencer's own note):

> `compressed-tensors` is permitted because
> `CompressedTensorsW4A4Nvfp4MoE` (see
> `schemes/compressed_tensors_w4a4_nvfp4_moe.py`) self-dispatches to its
> `cutlass_moe_fp4` path when the runner backend is not
> `flashinfer_trtllm`.

This decoupling matters because the **flashinfer_cutlass** path in
modelopt
(`modelopt_quant.py:2210-2270`) calls `flashinfer.cutlass_fused_moe`
which natively accepts `tp_size/tp_rank/ep_size/ep_rank/enable_alltoall`
arguments. The compressed-tensors scheme path goes through a
different kernel that does not have those arguments — EP is provided
by upstream weight slicing and topk filtering only.

### 2.4 Latent `skip_local_expert_mapping` bug

`StandardDispatcher.__init__`
(`python/sglang/srt/layers/moe/token_dispatcher/standard.py:96-105`):

```python
self.skip_local_expert_mapping = (
    backend.is_flashinfer_cutlass()      # <-- our target
    or backend.is_flashinfer_cutedsl()
    or backend.is_flashinfer_trtllm()
    or backend.is_flashinfer_trtllm_routed()
    or self.enable_flashinfer_mxfp4_moe
)
```

The dispatcher skips applying `local_expert_mapping` (which would
relabel global `topk_ids ∈ [0, num_experts)` to local
`topk_ids ∈ [0, num_local_experts) ∪ {-1}`) whenever any of these
runners is selected, because *those runners are supposed to handle EP
internally via `local_expert_offset`*. But the compressed-tensors
scheme's `else` branch (`cutlass_moe_fp4`) is the sglang-native
kernel, **not** flashinfer's, and it does **not** accept a
`local_expert_offset` argument. It expects
`topk_ids ∈ [0, num_local_experts)`.

Result: with `--moe-runner-backend flashinfer_cutlass --ep-size 8`,
the compressed-tensors scheme passes **global** `topk_ids`
(`∈ [0, 128)` for the REAP-345B `n_routed_experts=128`) into a
kernel that expects local IDs (`∈ [0, 16)`). The kernel either
asserts in `cutlass_moe.py:415` (`e_w1 == params.num_experts`) or
produces wrong outputs.

So even if we side-step DeepEP, just typing `--moe-runner-backend
flashinfer_cutlass --ep-size 8` against the REAP-345B checkpoint
today will fail at the scheme level. The fix is small but mandatory.

## 3. The three paths, in order

### 3.1 Path A — `flashinfer_cutlass` + standard dispatcher + ep\_size=tp\_size, fix the local-expert mapping

Server args:

```
--tp 8 --ep-size 8 \
--moe-a2a-backend none \
--moe-runner-backend flashinfer_cutlass \
--quantization compressed-tensors
```

Code change (one small dispatcher patch + one small scheme patch):

**Dispatcher**
(`python/sglang/srt/layers/moe/token_dispatcher/standard.py:96-105`):
make `skip_local_expert_mapping` runner-and-scheme-aware. The cleanest
delta is to gate it on whether the scheme actually consumes
`local_expert_offset`. Since the compressed-tensors scheme does not,
restrict the skip to `flashinfer_cutedsl`, `flashinfer_trtllm`,
`flashinfer_trtllm_routed`, and the modelopt-NVFP4 fp4_allgather case:

```python
# OLD (standard.py:96-105):
self.skip_local_expert_mapping = (
    backend.is_flashinfer_cutlass()
    or backend.is_flashinfer_cutedsl()
    or backend.is_flashinfer_trtllm()
    or backend.is_flashinfer_trtllm_routed()
    or self.enable_flashinfer_mxfp4_moe
)

# PROPOSED:
# flashinfer_cutlass is conditionally skip-EP-aware: only the modelopt-
# NVFP4 path drives `flashinfer.cutlass_fused_moe` (which handles EP
# internally). The compressed-tensors NVFP4 scheme falls through to
# sglang-native `cutlass_moe_fp4` which expects local topk_ids.
_compressed_tensors_w4a4_nvfp4 = (
    get_global_quant_config() == "compressed-tensors"
    and is_compressed_tensors_w4a4_nvfp4()
)
self.skip_local_expert_mapping = (
    (backend.is_flashinfer_cutlass() and not _compressed_tensors_w4a4_nvfp4)
    or backend.is_flashinfer_cutedsl()
    or backend.is_flashinfer_trtllm()
    or backend.is_flashinfer_trtllm_routed()
    or self.enable_flashinfer_mxfp4_moe
)
```

A simpler alternative: keep `skip_local_expert_mapping=True` for
`flashinfer_cutlass` and instead patch the compressed-tensors
scheme's `else` branch to use **flashinfer's** `cutlass_fused_moe`
(which does accept `ep_size/ep_rank`), as modelopt does at
`modelopt_quant.py:2230-2265`. That's a ~25 LOC scheme change; the
weight layouts already match (the modelopt path uses
`layer.w13_weight.view(torch.long)` + `w13_blockscale_swizzled.view(torch.int32)`,
and the compressed-tensors scheme already produces
`swizzle_blockscale(layer.w13_weight_scale)` in
`process_weights_after_loading:280-291`).

**Scheme**
(`compressed_tensors_w4a4_nvfp4_moe.py:455-475`): replace
`cutlass_moe_fp4` with `flashinfer.cutlass_fused_moe`. Scaffold:

```python
from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type
from sglang.srt.layers.quantization.modelopt_quant import ActivationType
from sglang.srt.utils import next_power_of_2

# ... inside the `else` branch of apply_weights:
topk_weights = topk_output.topk_weights
topk_ids = topk_output.topk_ids

fi_activation = ActivationType(
    get_activation_type(
        self.moe_runner_config.activation,
        is_gated=self.moe_runner_config.is_gated,
    )
)

with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
    symm_output = torch.empty(
        x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device,
    )

output = flashinfer_cutlass_fused_moe(
    output=symm_output,
    input=x,
    token_selected_experts=topk_ids.to(torch.int),
    token_final_scales=topk_weights,
    fc1_expert_weights=layer.w13_weight.view(torch.long),
    fc2_expert_weights=layer.w2_weight.view(torch.long),
    output_dtype=torch.bfloat16,
    input_sf=None,
    quant_scales=[
        layer.w13_input_scale_quant,
        layer.w13_weight_scale.view(torch.int32),
        layer.g1_alphas,
        layer.w2_input_scale_quant,
        layer.w2_weight_scale.view(torch.int32),
        layer.g2_alphas,
    ],
    ep_size=layer.moe_ep_size,
    ep_rank=layer.moe_ep_rank,
    tp_size=layer.moe_tp_size,
    tp_rank=layer.moe_tp_rank,
    tune_max_num_tokens=next_power_of_2(x.shape[0]),
    activation_type=fi_activation,
    enable_alltoall=False,  # --moe-a2a-backend none
)[0]
```

**Pros**: minimal diff. Standard dispatcher + flashinfer cutlass do
the EP work. No DeepEP at all (so DeepEP-mode/NVSHMEM init issues that
plagued previous tries are moot). The trtllm fast path stays alive
when `--moe-runner-backend flashinfer_trtllm` is selected.

**Cons**: requires per-rank weight slicing in
`process_weights_after_loading` — the compressed-tensors scheme today
keeps **global** `num_experts` worth of weights on every rank,
because trtllm-gen needs the full table (with `local_expert_offset`
selecting the active range). Either we
slice at load time (saving the memory we are after) or we keep the
global table and rely on flashinfer's `ep_rank` to select. The former
is the actual EP memory win we want.

The FusedMoE layer **already** sets `num_local_experts` per rank
during `__init__` and `local_expert_offset = moe_ep_rank *
num_local_experts` (`fused_moe_triton/layer.py:1138` and the
`make_expert_params_mapping` weight loader). Spencer's deploy on
`491a78ed6` already runs with this set up — the existing trtllm
fast-path success has `local_expert_offset` and
`local_num_experts` filled in correctly when `ep_size > 1`. So the
**scheme** doesn't need to re-slice; it just needs the kernel to
accept and respect those arguments. flashinfer's `cutlass_fused_moe`
already does.

**Verification gate**: the same eval suite that signed off the TP=8
deploy (intel-correctness + linear_decode_tps) must clear with
EP=8/flashinfer_cutlass. Concurrency target: ~7-8x current decode
batch size at constant KV memory.

### 3.2 Path B — port modelopt's `_is_cutedsl_v1_deepep` into the compressed-tensors scheme

Server args:

```
--tp 8 --ep-size 8 \
--moe-a2a-backend deepep \
--deepep-mode low_latency \
--moe-runner-backend flashinfer_cutedsl \
--quantization compressed-tensors
```

The unblocking requires:

1. Drop the `flashinfer_cutedsl` server\_args assertion that
   restricts `--quantization` to `modelopt_fp4`
   (`server_args.py:3559-3565`):

   ```python
   # CURRENT:
   if self.moe_runner_backend == "flashinfer_cutedsl":
       assert self.quantization in ["modelopt_fp4"], ...
   # PROPOSED:
   if self.moe_runner_backend == "flashinfer_cutedsl":
       assert self.quantization in ["modelopt_fp4", "compressed-tensors"], ...
   ```

2. Teach `CompressedTensorsW4A4Nvfp4MoE.apply_weights` to consume
   `DeepEPLLDispatchOutput` and call `flashinfer_cutedsl_moe_masked`
   the way modelopt does
   (`modelopt_quant.py:2143-2166`):

   ```python
   from sglang.srt.layers.moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
   from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

   if isinstance(dispatch_output, DeepEPLLDispatchOutput):
       hs, hs_sf, _, _, masked_m, _ = dispatch_output
       output = flashinfer_cutedsl_moe_masked(
           hidden_states=(hs, hs_sf),
           input_global_scale=layer.w13_input_scale_quant,
           w1=layer.w13_weight, w1_blockscale=layer.w13_blockscale_swizzled,
           w1_alpha=layer.g1_alphas,
           w2=layer.w2_weight, w2_blockscale=layer.w2_blockscale_swizzled,
           a2_global_scale=layer.w2_input_scale_quant, w2_alpha=layer.g2_alphas,
           masked_m=masked_m,
       )
       return DeepEPLLCombineInput(
           hidden_states=output,
           topk_ids=dispatch_output.topk_ids,
           topk_weights=dispatch_output.topk_weights,
       )
   ```

3. Mirror modelopt's `process_weights_after_loading` slicing path
   (`modelopt_quant.py:1830-1855`) so that
   `w13_input_scale_quant` and `w2_input_scale_quant` are sliced from
   `[num_experts]` to `[num_local_experts]` for the EP rank, and
   `w13_blockscale_swizzled` / `w2_blockscale_swizzled` are kept
   in the swizzled layout that `flashinfer_cutedsl_moe_masked`
   expects.

**Pros**: lets us keep DeepEP for cross-rank dispatch (relevant if
the deploy expands beyond a single node and we want low-latency A2A
on a future multi-node run).

**Cons**:
- ~80-100 LOC of new code (scheme branch + weight slicing).
- The `flashinfer_cutedsl` runner has **no fused topk fast path**
  for compressed-tensors — the trtllm fast path is gone.
- DeepEP NVSHMEM init has historically been brittle on this cluster
  (see `docs/advanced_features/ncclx_collectives.md:200-213`
  "the DeepEP/NVSHMEM low-latency transport fails before SGLang
  initializes with `Unable to create ah`, `create DCT share err`,
  and `nvshmem setup connections failed`"). Single-node deploy
  doesn't benefit from DeepEP's NVSHMEM-over-IB; it would use
  NVLink-only intra-node A2A but pays the same dispatcher cost.

### 3.3 Path C — keep `flashinfer_trtllm` + EP, slice weights per rank, no a2a

Server args:

```
--tp 8 --ep-size 8 \
--moe-a2a-backend none \
--moe-runner-backend flashinfer_trtllm \
--quantization compressed-tensors
```

The trtllm-gen `flashinfer.trtllm_fp4_block_scale_moe` already
accepts `local_expert_offset` and `local_num_experts`
(`compressed_tensors_w4a4_nvfp4_moe.py:439-441`):

```python
local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
local_num_experts=layer.num_local_experts,
```

When `ep_size > 1`, the FusedMoE weight loader at
`fused_moe_triton/layer.py:1141-1190` (the
`make_expert_params_mapping` chain) loads only the local-rank slice
of `w13_*` / `w2_*` from the checkpoint — meaning each rank already
allocates the EP-sized weight tensors. The trtllm kernel then reads
**only** `[local_expert_offset, local_expert_offset + local_num_experts)`
in its routing decision; everything outside that range gets a zero
contribution from this rank, and the post-MoE all-reduce (when no
a2a is involved) sums up the contributions from each rank.

Today the deploy uses `--ep-size 1`, so the weight loader loads the
full table on every rank. Flipping to `--ep-size 8 --moe-a2a-backend
none` should "just work" if:

- The MoE scheme's `process_weights_after_loading` does not
  hard-assume `num_local_experts == num_experts`. Audit:
  `compressed_tensors_w4a4_nvfp4_moe.py:204-208` reads
  `layer.num_local_experts` for the trtllm-path
  `w13_input_global_scale.expand(layer.num_local_experts)`, which is
  correct.
- `make_expert_params_mapping` knows how to load only the local
  range. `fused_moe_triton/layer.py:1146-1190` does exactly this
  per-rank.
- The all-reduce after the MoE layer covers the EP communication.
  `FusedMoE.forward_impl` does
  `tensor_model_parallel_all_reduce` when `reduce_results and
  (moe_tp_size > 1 or moe_ep_size > 1)` —
  `fused_moe_triton/layer.py:1131-1133`. Confirmed.

The one open question for Path C is whether trtllm-gen's
`trtllm_fp4_block_scale_moe` handles a "this rank has zero
tokens for any local expert" case cleanly. Modelopt's
`enable_flashinfer_trtllm_moe + EP` configuration is exercised in
its own test suite (`docs/basic_usage/deepseek_v32.md:235-239`
recommends `flashinfer_trtllm` for `nvidia/DeepSeek-V3.2-NVFP4`
with `--tp 4`, no `--ep`). To be tested:

1. Boot with `--ep-size 8 --moe-a2a-backend none --moe-runner-backend
   flashinfer_trtllm` and verify weight load → first forward.
2. Cross-check that the all-reduce in `FusedMoE.forward_impl:1131`
   recombines correctly: only `local_num_experts = 16` produce
   non-zero contribution per rank.
3. Run intel-correctness across the EP=1 baseline.

**Pros**:
- Smallest possible diff. Likely zero code changes — pure server-args
  flip. (The latent `skip_local_expert_mapping` is moot for
  `flashinfer_trtllm` because trtllm routes internally.)
- Trtllm fast path stays alive. Iter4 cubin advantages preserved.
- No DeepEP dependency.

**Cons**:
- The all-reduce after MoE still ships the full `hidden_states`
  across all 8 ranks (vs DeepEP's per-token routed reduce).
  But measured on a single node, the all-reduce is ~1-2 us
  (NVLink + SHARP / custom-allreduce) vs DeepEP dispatch+combine
  at ~6-10 us per layer. **Path C is faster in practice** on
  one-node deploys.
- The "EP-internal sparsity" win (each rank only computes for its
  16 local experts' assigned tokens) is real and substantial —
  for top_k=8, expectation is ~8/128 = 6.25% of tokens get any
  given expert. With 16 local experts per rank, ~1.0 expected
  expert per token per rank, vs 8 experts per token in EP=1.
  Compute drops ~8x per rank, and the all-reduce simply zeros
  out the inactive-expert contributions.

## 4. Specific server args (concrete launch for each path)

### Path A: flashinfer_cutlass internal EP

```bash
python -m sglang.launch_server \
    --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
    --quantization compressed-tensors \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend none \
    --moe-runner-backend flashinfer_cutlass \
    --disable-shared-experts-fusion \
    --disable-cuda-graph-padding \
    --max-running-requests 64
```

**Code prereq**: small `skip_local_expert_mapping` patch in
`token_dispatcher/standard.py` + 25 LOC switch to flashinfer's
`cutlass_fused_moe` in the compressed-tensors scheme.

### Path B: DeepEP LL + flashinfer_cutedsl, compressed-tensors enabled

```bash
python -m sglang.launch_server \
    --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
    --quantization compressed-tensors \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-runner-backend flashinfer_cutedsl \
    --disable-shared-experts-fusion \
    --max-running-requests 64
```

**Code prereq**: ~80-100 LOC scheme port + server\_args
allowlist change. Highest risk because of DeepEP NVSHMEM
brittleness on this cluster.

### Path C: trtllm + EP via local\_expert\_offset, no a2a (recommended first try)

```bash
python -m sglang.launch_server \
    --model-path BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
    --quantization compressed-tensors \
    --tp 8 \
    --ep-size 8 \
    --moe-a2a-backend none \
    --moe-runner-backend flashinfer_trtllm \
    --disable-shared-experts-fusion \
    --max-running-requests 64
```

**Code prereq**: likely zero. Try the flip first; if it works,
ship it. If the trtllm kernel asserts on a zero-token edge case,
fall back to Path A.

## 5. Code-change scaffold for each path (compact summary)

### Path A (~30 LOC across 2 files)

`python/sglang/srt/layers/moe/token_dispatcher/standard.py`:
gate `skip_local_expert_mapping` on
`flashinfer_cutlass` only when the active scheme actually consumes
`local_expert_offset` (i.e., not the compressed-tensors NVFP4
self-dispatched cutlass\_moe\_fp4 path). Cleanest is to introduce a
`scheme_handles_ep_internally` flag on the FusedMoE layer set during
`process_weights_after_loading`.

`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`:
in the `else` (non-trtllm) branch of `apply_weights`, replace
`cutlass_moe_fp4` with `flashinfer.cutlass_fused_moe` — pass
`ep_size`, `ep_rank`, `tp_size`, `tp_rank`, `enable_alltoall=False`.

### Path B (~100 LOC across 3 files)

`python/sglang/srt/server_args.py:3559-3565`: extend the
`flashinfer_cutedsl` allowed-quantization list to include
`compressed-tensors`.

`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`:
- in `__init__`, capture
  `use_flashinfer_cutedsl_v1 = is_flashinfer_cutedsl_v1_path()`
  alongside `use_flashinfer_trtllm`.
- in `process_weights_after_loading`, when `use_flashinfer_cutedsl_v1`,
  slice `w13_input_scale_quant` / `w2_input_scale_quant` to
  `num_local_experts` (mirrors `modelopt_quant.py:1830-1855`), and
  keep `w13_weight_scale` / `w2_weight_scale` in the swizzled layout
  produced by `swizzle_blockscale` (already done for non-trtllm path).
- in `apply_weights`, when `isinstance(dispatch_output,
  DeepEPLLDispatchOutput)`, branch to a new
  `_apply_deepep_cutedsl(layer, dispatch_output)` method that calls
  `flashinfer_cutedsl_moe_masked` and returns `DeepEPLLCombineInput`.

`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`:
no changes needed; the scheme registration already routes to the
right `CompressedTensorsW4A4Nvfp4MoE`.

### Path C (~0-5 LOC)

Likely zero. If a trtllm zero-token edge case asserts, the
fix is a guard on `hs_fp4.shape[0] == 0` in the trtllm branch
of `apply_weights` similar to the zero-rows short-circuit at
`compressed_tensors_w4a4_nvfp4_moe.py:333-339` (which already
exists for forward\_idle).

## 6. Performance projection — EP=8 vs EP=1 memory + concurrency

### 6.1 Memory savings (per rank)

Each MoE layer's NVFP4 expert weights (W13 + W2) at:
- `hidden_size = 7168`
- `moe_intermediate_size = 2048`
- `n_routed_experts = 128`
- W13: `[128, 2 * 2048, 7168/2 bytes (FP4 packed)] = [128, 4096, 3584] uint8`
  → 128 × 4096 × 3584 = 1.84 GiB / layer per rank
- W2: `[128, 7168, 2048/2 bytes] = [128, 7168, 1024] uint8`
  → 128 × 7168 × 1024 = 0.94 GiB / layer per rank
- Scales (w13\_weight\_scale, w2\_weight\_scale, FP8):
  ~1/16 of weight bytes ≈ 0.17 GiB / layer per rank
- Per-rank per-layer total: ~2.95 GiB at EP=1.

`first_k_dense_replace=3`, `num_hidden_layers=61` → 58 MoE layers.
- Per-rank MoE weight footprint at EP=1: 58 × 2.95 GiB ≈ **171 GiB**.

At EP=8: each rank holds 1/8 of experts = 16 local experts.
- Per-rank MoE weight footprint at EP=8: 58 × (2.95/8) GiB ≈ **21.4 GiB**.

**Savings per rank: ~150 GiB**. On B200 with 192 GiB HBM3e, this
is the difference between "all MoE weight + KV + activations
barely fit" and "comfortable headroom for larger KV / longer
context / higher batch."

Specifically, the KV cache for DeepSeek-V3.2 dense MLA at 64K
context, head\_dim=64, 128 KV heads per layer:
- per-token KV footprint at FP8 dense = 128 × 64 × 2 bytes × 61
  layers = 1.0 MB / token.
- 192 GiB - 21.4 GiB - 10 GiB (activations/comms) = 160 GiB free
  for KV.
- KV capacity at FP8: 160 GiB / 1.0 MB = ~160K tokens per rank,
  × 8 ranks (DP+EP+TP balance, NOT free in TP=8) = depending on
  attention model, ~160K-1.3M concurrent tokens system-wide.

vs EP=1: 192 GiB - 171 GiB - 10 GiB = ~11 GiB free for KV →
~11K tokens per rank. Concurrency ceiling: ~14x lower.

**Concurrency gain projection: 7-10x more concurrent decoding
seats at the same context length**, bounded by attention
compute and per-batch overhead rather than memory.

### 6.2 Compute savings per layer

At top\_k=8, each token sees 8/128 = 6.25% of experts. Each rank
holds 16/128 = 12.5% of experts. Expected experts-per-token-per-rank:
8 × (16/128) = 1.0.

EP=1 per-rank compute: 8 experts per token (every rank computes
the full top-k path because experts are replicated).

EP=8 per-rank compute: 1.0 expert per token expected; rest of
the MoE pass is skipped via masked GEMM (in DeepEP+cutedsl) or
zeroed out (in trtllm with `local_expert_offset`). The all-reduce
recombines.

**Compute savings: ~8x per MoE layer per rank**. The full pass
gets the standard ~1.5-2x decode TPOT improvement (since MoE
isn't the only cost — attention, RMSNorm, linear projections all
stay constant or scale weakly).

### 6.3 Communication cost

At `B=128, hidden_size=7168`, BF16:
- TP=8 all-reduce per MoE layer: `[128, 7168] × 2 bytes = 1.8 MiB`
  * 1 / 8 partial = 0.23 MiB shipped per rank per all-reduce.
  At NVLink BW ~900 GB/s, this is ~0.26 us per layer.
- DeepEP LL dispatch + combine: per-rank ~7-12 us.
- DeepEP normal dispatch + combine: ~15-25 us.

So Path C (trtllm + no a2a) and Path A (cutlass + no a2a)
**both win on comms** vs DeepEP for this single-node
deployment. The MoE compute itself dominates anyway (~50 us
per layer at top\_k=8).

### 6.4 Expected TPOT delta

At current TP=8 deploy (8xB200, dense BF16 attention + NVFP4
MoE, batch=128, ~58 ms TPOT), the MoE pass is ~30-40% of TPOT
(estimated from iter4 profile). EP=8 compute savings at ~8x
on MoE only would drop the MoE share from ~22 ms to ~3-4 ms,
giving a projected TPOT of **~38-42 ms**. The bigger win is
the headroom for larger batches: at fixed TPOT (~58 ms), the
batch ceiling lifts ~3-5x.

## 7. Acceptance gate

The intel-correctness + linear\_decode\_tps two-stage gate
(per the project measurement-discipline rule) applies as written
in MEMORY.md.

- **Stage 1 — correctness**: intel-correctness eval at the same
  preset that passed for the TP=8 baseline (commit `1d7deebee`).
  Pass criteria: deltas within the existing ±0.5% band on the
  reference rubric.
- **Stage 2 — performance**: linear\_decode\_tps at the production
  batch (B=64, B=128) and prefix lengths (4K, 16K, 64K). Pass
  criteria: ≥1.0x TPOT @ B=128, ≥3.0x batch ceiling at fixed
  TPOT.

A negative correctness result on any of the three paths invalidates
that path; default fallback is Path C → Path A → Path B.

## 8. Open questions / known unknowns

1. **trtllm zero-token edge case at EP=8** (Path C): when a rank
   genuinely has no tokens routed to any of its 16 local experts,
   does the trtllm kernel assert or silently produce a zero output?
   `compressed_tensors_w4a4_nvfp4_moe.py:333-339` short-circuits
   `x.shape[0] == 0` (the DP forward\_idle case) but does NOT
   short-circuit "tokens exist but none route here." Test required.

2. **flashinfer.cutlass\_fused\_moe weight layout** (Path A):
   the modelopt `enable_flashinfer_cutlass_moe` path
   (`modelopt_quant.py:1968-1989`) calls
   `swizzle_blockscale(layer.w13_weight_scale)` and binds to
   `layer.w13_blockscale_swizzled`. The compressed-tensors scheme
   does the same swizzle but binds to `layer.w13_weight_scale`
   itself (`compressed_tensors_w4a4_nvfp4_moe.py:283-291`).
   Both layouts feed `flashinfer.cutlass_fused_moe` per its
   `quant_scales[1] = w1_blockscale.view(torch.int32)`. To verify:
   inspect `flashinfer.fused_moe.core.cutlass_fused_moe` source
   at `/home/spencer/.local/lib/python3.11/site-packages/flashinfer/fused_moe/core.py:775+`
   for the exact `quant_scales[1]` shape and stride contract.

3. **DeepEP NVSHMEM stability** (Path B): on this cluster the
   B200 + RoCE NVSHMEM init has been brittle. `ncclx_collectives.md:200-213`
   notes `DECODE_MOE_A2A_BACKEND=none` was used precisely because
   DeepEP init failed. The 1d7deebee BypassedTopKOutput fix
   resolved a different DeepEP error but not the NVSHMEM probe
   issue.

4. **Path A vs Path C on TPOT**: at B=128 single node, the
   trtllm-gen fused MoE (Path C) should beat sglang-native cutlass
   (Path A's `else` branch even after we re-wire to flashinfer's
   `cutlass_fused_moe`). The trtllm-gen cubins are tuned for B200
   tile shapes (see `notes/nvfp4_moe_iter7_recon.md`). Path C is
   likely faster; Path A is the contingency.

## 9. Decision

**First try: Path C** (`--ep-size 8 --moe-a2a-backend none
--moe-runner-backend flashinfer_trtllm`). Zero code changes.
If trtllm asserts on a zero-token-for-local-experts edge case
on B200, fall through to **Path A** (with the
`skip_local_expert_mapping` + flashinfer `cutlass_fused_moe`
patches). Keep **Path B** as the longer-term option if multi-node
DeepEP is needed later.

If Path C boots and clears intel-correctness, the project's
single-node memory headroom unlock is delivered in zero LOC.
That is the maximally honest research outcome: the work is
**in the server-args choice, not in patching the EP+DeepEP
collision**.
