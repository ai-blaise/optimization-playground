# Sparse-MLA Decode with NVFP4 KV — design notes

Source: FlashMLA `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/` (Apache-2.0).
Target: drop-in replacement for the trtllm-gen sparse-MLA decode cubin, reading NVFP4 KV directly. No materialize round-trip.

## KV format change

| | FP8 baseline (FlashMLA today) | NVFP4 variant (new) |
|---|---|---|
| Latent value bytes | 512 (fp8_e4m3, 1 byte/elem) | **256** (e2m1, 2 elems/byte) |
| Block size for scales | 64 | **16** |
| Number of scale bytes / token | 7 + 1 pad (E8M0, 1 byte each) | **32** (E4M3, 1 byte each) |
| Rope bytes (BF16, 64 dims) | 128 | 128 (unchanged) |
| **Total bytes / token** | **~648** | **~416** (~36% less) |

## Code hotspots in `phase1.cuh` (line numbers from current head128/phase1.cuh)

### 1. SMEM K_raw buffer size (config.h:88)

```cpp
// FP8 today:
array_aligned<fp8_e4m3, B_TOPK*(D_K/2)> K_raw[NUM_RAW_K_BUFS];

// NVFP4 (half the bytes):
array_aligned<uint8_t, B_TOPK*(D_K/4)> K_raw_fp4[NUM_RAW_K_BUFS];  // D_K/4 = 128 bytes per token K_nope
```

### 2. SMEM scale buffer (config.h:95)

```cpp
// FP8 today (E8M0, 7+1 scales per token):
CUTE_ALIGNAS(16) fp8_e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN/2];

// NVFP4 (E4M3, 32 scales per token):
CUTE_ALIGNAS(16) fp8_e4m3 scales_fp4[NUM_INDEX_BUFS][B_TOPK][32];
```

Per-token scale storage grows: 4 bytes (E8M0/2) → 32 bytes (E4M3). But total still small vs FP8 baseline's 4-byte scales × 4 buffers.

### 3. TMA descriptor stride (config.h)

```cpp
// FP8: K_nope_bytes + 2*K_rope_bytes = 448 + 128 = 576
static constexpr int TMA_K_STRIDE_FOR_DECODING = D_NOPE + 2*D_ROPE;

// NVFP4: K_nope_bytes(half) + 2*K_rope_bytes + scale_bytes = 224 + 128 + 32 = 384
static constexpr int TMA_K_STRIDE_FOR_DECODING_FP4 = (D_NOPE/2) + 2*D_ROPE + 32;
```

### 4. TMA load (phase1.cuh:339, K_nope tile load)

```cpp
// FP8 today loads B_TOPK × (D_K/2) bytes as fp8_e4m3
// NVFP4 loads B_TOPK × (D_K/4) bytes as uint8 — half the TMA payload
```

The TMA descriptor needs to be constructed with the NVFP4 element type and the halved stride. The `tma_params.tensor_map_kv_nope` (config.h:35) is built host-side in `run()` — separate descriptor with the FP4 layout.

### 5. Dequant inner loop (phase1.cuh:471-510)

This is the main algorithmic change. Today's loop:

```cpp
uint64_t cur_data_fp8x8 = get_raw_fp8(local_row_idx, 0);  // 8 FP8 bytes = 8 elems
for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
    ku::nve4m3x2 data_fp8[4];   // 4 pairs of FP8
    *(uint64_t*)data_fp8 = cur_data_fp8x8;
    bf16 scale = scales[local_col_idx];
    for (int i = 0; i < 4; ++i)
        data_bf16[i] = fp8x2_to_bf16x2_with_scale(data_fp8[i], scale);
    st_128b(...);  // store 8 BF16
}
```

New NVFP4 loop:

```cpp
// Load 4 bytes (8 NVFP4 elements packed: 4 bytes × 2 elems/byte) per inner iter
uint32_t cur_data_fp4x8 = get_raw_fp4(local_row_idx, 0);  // 4 bytes packed FP4 = 8 elems
for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
    // Unpack 4 bytes → 8 fp4 nibbles → 8 fp16/bf16 values
    bf16 data_bf16[8];
    // Scale: each block of 16 NVFP4 elements has its own E4M3 scale
    //   At this iter we cover 8 elements; depending on alignment we use 1 or 2 scales
    fp8_e4m3 scale_e4m3 = scales_fp4[index_buf_idx][row_idx][local_col_idx/2];
    bf16 scale_bf16 = e4m3_to_bf16(scale_e4m3);
    // Unpack 4 bytes into 8 e2m1 nibbles, dequant each
    nvfp4x2_to_bf16x2_with_scale(cur_data_fp4x8, scale_bf16, data_bf16);
    if (local_col_idx+1 < COLS_PER_GROUP)
        cur_data_fp4x8 = get_raw_fp4(local_row_idx, local_col_idx+1);
    st_128b(local_row_idx, local_col_idx, *(__int128_t*)data_bf16);
}
```

Need to add helper:

```cpp
__device__ __forceinline__ void
nvfp4x2_to_bf16x2_with_scale(uint32_t fp4_bytes, bf16 scale, bf16 out[8]) {
    // Each of the 4 bytes holds 2 e2m1 values
    // e2m1 encoding: 1 sign bit, 2 exp bits, 1 mantissa bit
    constexpr float kE2M1Table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        uint8_t byte = (fp4_bytes >> (b*8)) & 0xff;
        uint8_t lo = byte & 0xf;
        uint8_t hi = (byte >> 4) & 0xf;
        float lo_val = (lo & 0x8 ? -1.0f : 1.0f) * kE2M1Table[lo & 0x7];
        float hi_val = (hi & 0x8 ? -1.0f : 1.0f) * kE2M1Table[hi & 0x7];
        out[b*2 + 0] = __float2bfloat16(lo_val * __bfloat162float(scale));
        out[b*2 + 1] = __float2bfloat16(hi_val * __bfloat162float(scale));
    }
}
```

(In practice we'd use `cvt.rn.bf16x2.e2m1x2` PTX instruction on SM_100 for direct hardware dequant — much faster than scalar table lookup.)

### 6. K rope load (no change)

Rope dims stay BF16 — same `tma_params.tensor_map_kv_rope` and load logic.

### 7. UMMA pipeline (no change)

After dequant we still feed BF16 to the existing `TiledMMA_P` and `TiledMMA_O`. The dequant target is BF16 in SMEM (`smem.K[k_buf_idx]`), same shape as today.

## Host-side wiring

The kernel takes `params.kv_cache_ptr` + new `params.kv_scales_ptr` (NVFP4 has separate scale buffer). The Python wrapper constructs the TMA descriptors for both.

## Build path

Add as a new pybind module in `optimization-playground/python/sglang/jit_kernel/sparse_mla_nvfp4_kv/`. Use the existing `flashinfer-style` build pipeline (`tvm-ffi` JIT module) to compile on first call, similar to how the HIGGS kernels are wired.

## SGLang integration (`dsa_backend.py`)

In `_forward_trtllm` (currently at line 2637), add a branch:

```cpp
if (
    self.kv_cache_dtype == torch.uint8  // NVFP4 packed
    and getattr(self.token_to_kv_pool, "indexer_quantization", None) is not None
):
    # Route to new NVFP4 sparse-MLA decode kernel
    return self._forward_flashmla_nvfp4(q, q_rope, ..., topk_indices, ...)
```

Where `_forward_flashmla_nvfp4` builds the params struct + calls the new kernel.

Lift the DSA `kv_cache_dtype` assertion at server_args.py:1867 to allow `fp4_e2m1`.

## Wiring up the KV pool

The NVFP4 KV pool already exists (`NVFP4KVMethod` in `fp4_kv_cache_quant_method.py`). Hook it in:

1. Allow `enable_nvfp4_dense_kv_cache=True` (new server arg) OR auto-enable when `kv_cache_dtype=fp4_e2m1`.
2. `quantize_and_store` writes packed FP4 + E4M3 scales to the pool buffers (already implemented).
3. `_forward_flashmla_nvfp4` reads from the pool directly — no dequant materialize.

## Estimated effort

- Kernel modifications (config.h, phase1.cuh): ~300 lines diff
- Python wrapper + build glue: ~150 lines
- SGLang dispatch wiring + assertion lift: ~80 lines
- Testing harness + correctness check vs FP8 baseline: ~100 lines

Total: ~600-700 lines, multi-day work.

## Expected impact

Per the bandwidth math (NVFP4 KV is 56% of FP8 KV reads, sparse-MLA is HBM-bound):
- FP8 baseline TPOT: 30.95 ms
- NVFP4 KV target: **~22-25 ms TPOT** (beat baseline by ~6 ms)

Per the SAW-INT4 paper's "zero overhead" claim once fusion is structural (no materialize round-trip).
