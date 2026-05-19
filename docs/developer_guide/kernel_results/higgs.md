# HIGGS

## Scope

HIGGS dense MLA 2-bit KV cache store, selected/dequant paths, and fused MLA
decode on B200 for the DeepSeek-V3.2-REAP lane.

## Result

Accepted implementation: `store_const_codebook_warp_pack`.

This is a store-only production candidate. It keeps the fixed EDEN2 codebook
path and replaces the normalized-pair and packed-index shared-memory handoff
with warp shuffles. It is bit-exact against production in the HIGGS tests and
clears the relaxed close-out gate.

The default production selector now uses this store path. No fused MLA decode
candidate was promoted. The best decode variants produced large-shape wins, but
small-row integration shapes were mixed and the functional tolerance envelope
for the changed split-reduction numerics is not yet validated against the HIGGS
and SAW-INT4 reference constraints. Those variants are preserved as rejected
evidence rather than production code.

## Performance

Final B200 verification artifact:
`/root/b200-run-20260518/workers/higgs/artifacts/worker_loop_20260518/round24_final_closeout_verification/round24_summary.json`.

| Path | OP baseline | Accepted candidate | Speedup | Correctness |
| --- | ---: | ---: | ---: | --- |
| Store, 8192 tokens | 0.063603840 ms | 0.047208958 ms | 1.3473x | bit-exact |
| Selected KV, r1 topk2048 | 0.009014000 ms | 0.008909600 ms | 1.0117x | bit-exact decode output |
| Decode, r1 topk2048 after store | 0.042072800 ms | 0.041550800 ms | 1.0126x | bit-exact |
| Decode, r4 topk2048 after store | 0.059747601 ms | 0.060678399 ms | 0.9847x | bit-exact |
| Decode, r16 topk4096 after store | 0.296774793 ms | 0.296769595 ms | 1.0000x | bit-exact |
| Decode, r64 topk4096 after store | 1.070790768 ms | 1.070921230 ms | 0.9999x | bit-exact |

Store IKP, final B200 verification:

| Candidate | Store kernel SASS instructions | Relative |
| --- | ---: | ---: |
| `production` | 544,366,592 | 1.0000x |
| `store_const_codebook_warp_pack` | 469,393,408 | 0.8623x |

Earlier accepted-baseline comparison:
`store_const_codebook_warp_pack` was also measured at about 1.109x versus the
previous accepted `store_const_codebook` store candidate in the round11b
warp-pack-fixed profile, and about 1.333x versus the original production store.

## Flashtraining

Closest comparator artifact:
`/root/b200-run-20260518/workers/higgs/artifacts/worker_loop_20260518/round24_final_closeout_verification/higgs_flashtraining_comparator_store_only_final.json`.

Megatron reference: `ai-blaise/Megatron-LM` ref
`844bf42af7ce73a1b80e4b1ccb3c221dd63de35d`.

There is no exact flashtraining KV-cache store equivalent. The closest real
reference is Megatron HIGGS dense 2-bit fake-quant forward
(`higgs_kv_fwd_kernel`), which includes FWHT, EDEN2 quantization, indices,
scale, inverse FWHT, output, and training saved tensors. The OP comparison
therefore reports store-only timing and store+dequant timing; it is not a
full-equivalence claim.

| Rows | Flashtraining fake-quant forward | OP production store+dequant | Accepted store+dequant | Accepted / flashtraining |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 0.036897200 ms | 0.047206801 ms | 0.038994801 ms | 0.9462x |
| 8192 | 0.069633198 ms | 0.090139598 ms | 0.073762000 ms | 0.9440x |

Store-only OP speedup in the same comparator:

| Rows | OP production store | Accepted store | Speedup |
| ---: | ---: | ---: | ---: |
| 4096 | 0.032864001 ms | 0.024668799 ms | 1.3322x |
| 8192 | 0.063532400 ms | 0.047162801 ms | 1.3471x |

## Rejections

`store_warp_pack_mla_decode_vec4_split_balanced` was not promoted. In the
round20 repaired acceptance gate it showed a target-path signal, but integration
was mixed: r64 topk4096 decode improved 1.1624x, while r1/r4 topk2048 decode
regressed to 0.8404x and 0.8430x. Standalone r64 topk4096 was allclose but not
bit-exact (`max_abs=0.00048828125`). Non-bit-exact output is not by itself a
rejection for HIGGS; the blocker is the combination of small-shape regressions
and missing quality/tolerance validation for the new reduction order.

`store_warp_pack_mla_decode_vec4_split_large_batch` was also not promoted. The
final rejected probe measured 1.0339x at r64 topk4096 and was allclose with
cosine approximately 1.0 (`max_abs=0.00048828125`, mean absolute error around
1e-8). Small shapes were effectively neutral (`0.9976x` at r1 topk2048,
`1.0020x` at r4 topk2048). This remains useful evidence, but it is not a
production-ready improvement until the decode tolerance and quality tests are
defined and passed.

Round13 pair-lane scale-broadcast dequant candidates stayed slower than
production (`~0.89x` direct and `~0.867x` page-table). Round16 scale-broadcast
MLA decode was slower (`~0.947x` at r64 topk4096). Round17 constant-codebook
decode was rejected due large regressions. Round18 vec4 unpack was bit-exact
but sub-gate (`~1.021x` at r64 topk4096).

## Correctness Gate

The accepted store path is bit-exact because it only changes the CUDA scoring
and packing schedule, not the HIGGS codec. Future decode candidates do not need
to be bit-exact if they are functionally correct for the target model and the
reference papers. The minimum gate is: HIGGS codec invariants stay intact
(Hadamard/RHT rotation, EDEN2 lattice lookup, scale reconstruction, packed-slot
layout), SAW-INT4-inspired fused-write or BDR changes preserve the serving
contract, and quality/perplexity/generation smoke tests confirm that any
changed reduction order is harmless for the target deployment.

## CuTe / CZS

Final CZS capture:
`/root/b200-run-20260518/workers/higgs/artifacts/worker_loop_20260518/round24_final_closeout_verification/czs_higgs_dense_2bit_b200_optin.out`.

Result: 5 proved, 4 disproved, 0 unknown, exit code 1. The existing HIGGS CZS
artifact is layout/vectorization-only and has no MMA atoms, TMA descriptors, or
`ldmatrix` obligations. It is not a CuTe/tensor-op promotion proof.

CuTe/tensor-op successor status: blocked, not ignored. IKP/autoinfer showed the
remaining decode hot path is EDEN2 gather/argmax, packed-byte traffic, online
softmax, and FWHT/butterfly work rather than a clean SM100 MMA tile. A tensor
rewrite would need to materialize dense topk x 512 values or use very skinny
K=2, N=16 tiles before non-GEMM softmax/butterfly work, and would require a new
CZS-proved module before promotion.

## References

`togethercomputer/saw-int4` was inspected as a BDR/fused-KV-write reference. It
supports the no-global-scratch Hadamard and fused write design direction, but it
targets MHA affine INT4 KV, not dense MLA EDEN2-16 lattice slots with a 258-byte
HIGGS layout.

`Dao-AILab/fast-hadamard-transform` was inspected as the FWHT implementation
reference. It validates the register/warp/shared-memory butterfly family, but
as a standalone transform it would add launch/global traffic and does not cover
EDEN2 lookup, packed HIGGS slots, page-table selection, or fused MLA softmax.

## Verification

- Final focused correctness: `CUDA_VISIBLE_DEVICES=1
  python -m pytest python/sglang/test/test_higgs_dense_2bit_kv.py
  test/srt/test_higgs_dense_2bit_kv_integration.py -q` passed 21 tests.
- Final B200 verification: round24 store+selected+decode integration, rejected
  decode probe, flashtraining comparator, and store IKP all completed under
  idle gates.
- The accepted store path is the default production selector; rejected
  decode/dequant variants remain opt-in candidates only.
