# #15 NVFP4 MoE iter6 reconnaissance

## Mission

PRIMARY: fuse all 3 allgathers (BF16 + FP4 packed + UE8M0 SF scales)
into a single ncclGroupStart/End block. iter5 already groups FP4 + SF;
extending to BF16 saves an additional NCCL launch per layer.

## Current state (iter5 PRIMARY, commit ab38f9568)

* python/sglang/srt/layers/dp_attention.py:
  - dp_gather_partial_fp4() at L685: groups FP4 + SF under pynccl_comm
    group_start/end. Wins +0.7us/layer vs serial at production peak
    (m_global<=128); regresses past m_global=256.

* python/sglang/srt/layers/communicator.py L1024 wire:
  - L1074: dp_gather_partial(hidden_states, local_hidden_states, ...)
    -> BF16 gather (separate NCCL launch).
  - L1126: dp_gather_partial_fp4(...) -> grouped FP4 + SF.
  - Sequence inside the captured graph: BF16 launch, then FP4+SF
    grouped launch. 2 launches total.

## iter6 implementation plan

1. New function dp_gather_partial_bf16_fp4_fused in dp_attention.py:
   wraps ALL THREE allgathers (BF16 + FP4 + SF) in one
   pynccl_comm.group_start()/group_end() block. Honors the same
   torchcomms_ncclx fallback as the iter5 path.

   Caveat: the existing dp_gather_partial path goes through
   reg_all_gather_into_tensor -> _all_gather_into_tensor ->
   pynccl_comm.all_gather (with change_state). To fuse with the
   FP4+SF group, the BF16 allgather must use the SAME
   pynccl_comm.all_gather call site so all three sit between the
   one group_start/end pair. This means bypassing the
   reg_all_gather_into_tensor custom op for the BF16 leg in the
   fused path.

   Capture-safety: pynccl_comm.all_gather is graph-capturable (test
   /srt/test_nvfp4_dp_gather_partial_fp4.py:_run_bench_graph_captured
   already captures 3 raw pynccl all_gathers under one graph).

   Dynamo-safety: the wire surface is inside _gather_hidden_states_and_residual
   which runs inside the piecewise-cuda-graph traced region. The
   iter5 path already calls pynccl_comm.group_start/group_end inside
   this region without breaking Dynamo. The same pattern transfers.

2. communicator.py L1024 wire refactor:
   Replace the L1074 dp_gather_partial(...) call + the L1126
   dp_gather_partial_fp4(...) call with a single conditional:
     - If _stash is None or m_global > _ITER5_FP4_ALLGATHER_M_GLOBAL_MAX:
         dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
         (no stash to gather)
     - Else: dp_gather_partial_bf16_fp4_fused(
                 hidden_states, local_hidden_states,
                 _global_fp4, _local_fp4,
                 _global_sf_storage, _sf_storage_local,
                 forward_batch,
             )

3. Microbench: extend test_nvfp4_dp_gather_partial_fp4.py with a new
   bench variant _iter6_all_grouped that wraps all 3 in one
   group_start/end pair (this is what _iter5_grouped already does in
   the test!). So we need to compare:
     iter5 production wire (BF16 separate + FP4+SF grouped)  ~= bench (c)+1 launch
     iter6 production wire (all 3 grouped)                   == bench _iter5_grouped (d)
   Need to add a (c2) variant: BF16 + (FP4+SF grouped) two-launch.
   Already have (a) BF16-only and (d) all 3 grouped.

   The bench delta between (c2) and (d) is the iter6 win projection.

## Honest scope warning

Per iter5 commit body: iter6 grouping was projected at ~+1us/layer.
That projection assumed ncclGroupStart/End amortizes per-launch
overhead (~1-2us each launch) by collapsing 2 launches -> 1. The
actual saving will depend on:
  - whether the BF16 reg_all_gather_into_tensor op has any pre/post
    work that breaks the group fusion (e.g. event_record, stream
    sync).
  - whether per-launch overhead at the production bytes count (m=128,
    hidden=7168, BF16 = 14336 B/token x 128 = 1.8 MiB) is actually
    launch-dominated or wire-dominated.

If the bench shows <0.3us/layer delta, this is also an honest negative
and we land it as opt-in only.

## IKP applicability

IKP profiles instructions inside custom CUDA kernels (regions in your
own .cu source). The iter6 vector is purely a host-side NCCL launch
fusion — the wins/losses are visible in CUDA-graph captured microbench
(launch overhead per ncclGroupStart/End boundary). IKP would only
become directly useful for the iter6 TERTIARY vector (flashinfer
bmm_Bfloat16_E2m1E2m1_* CUBIN), where it could attribute per-instruction
cost inside the bmm kernel. Not used for the PRIMARY vector this iter.

## Next steps

1. Implement dp_gather_partial_bf16_fp4_fused.
2. Refactor communicator.py L1074-L1149 to call the fused function.
3. Extend test with iter6 bench variant.
4. Run dp=2 graph-captured bench on cuda:6+cuda:7.
5. Commit with measured numbers in table.
