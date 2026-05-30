DSA NVFP4 indexer iter6 — recon checkpoint
==========================================

State on entry: HEAD at 8987cb270, iter5 SECONDARY (bf9338a9e) landed.

iter5 SECONDARY hot loop bottleneck (per its own commit body):

```
for i in [0, pair_end) step 2:
  s0, s1   = scales_row[i..i+1]                  # 1 LDS.b64
  off0     = smem_value_offset[i]                # short
  off1     = smem_value_offset[i+1]              # short
  b0       = smem_pages[off0 + dim_byte]         # 1 LDS.b8 -- independent address
  b1       = smem_pages[off1 + dim_byte]         # 1 LDS.b8 -- independent address
  sum0    += decode_e2m1_nibble(c0, s0)
  sum1    += decode_e2m1_nibble(c1, s1)
```

The 2 LDS.b8 are serialized: off0 and off1 point to different page slots
in smem_pages, so the compiler cannot coalesce into a single LDS.b16.

iter6 PRIMARY vector: precompute a transposed value SMEM table.

```
__shared__ uint8_t smem_values_t[kNVFP4ValueBytes][kBlockSize];
                                = [64][128] = 8192 B = 8 KB.
```

(Brief said 32 KB or 64 KB — but kNVFP4ValueBytes = head_dim/2 = 64,
not 128, so the actual cost is 8 KB; no cudaFuncSetAttribute escalation
needed. The brief math used the wrong dimension.)

Cooperative staging: in the existing predecode pass each thread tid is
already an owner of token-local i = tid. After resolving
value_byte_base = smem_value_offset[i], also do a 64-byte gather from
smem_pages[value_byte_base..+64] and scatter into smem_values_t[d][i]
for d in [0, 64). Reads: 4 LDS.128 (4 uint4) from smem_pages. Writes:
64 STS.b8 to transposed slots (acceptable; happens once per CTA per
hisa_block, not in the hot per-dim loop).

Inner loop, thread tid runs with fixed dim, fixed dim_byte = dim greater than greater than 1:

```
const uint8_t* row = and smem_values_t[dim_byte][0];   // 128 contiguous bytes
for i in [0, pair_end) step 2:
  uint16_t bb = *reinterpret_cast(const uint16_t*)(row + i);  # LDS.b16 = 2 tokens
  uint8_t b0  = bb and 0xff;
  uint8_t b1  = bb greater than greater than 8;
  ...
```

The 2 LDS.b8 collapse into 1 LDS.b16. The LDS issue count for the hot
loop drops from 3 per pair (1 scales LDS.b64 + 2 value LDS.b8) to 2
per pair (1 scales LDS.b64 + 1 value LDS.b16). 33 percent LDS-issue
reduction in the hot loop.

SMEM budget (sm_100a default 48 KB unless overridden):

```
smem_pages              kStagedBytesPerPage * 2  = (64+4)*64 * 2 = 8704 B
smem_page_ids[2]                                 = 8 B
smem_scales_t[4][128]                            = 2048 B
smem_value_offset[128]                           = 256 B
smem_values_t[64][128]  (NEW)                    = 8192 B
TOTAL                                            ~ 19208 B  =  ~18.8 KB
```

Comfortably under the 48 KB default, no FuncSetAttribute escalation.

Projected mean_pool win at 64/32768: ~10-15 percent over iter5 SECONDARY's
240us -- target less than 225us (per brief).

iter6 SECONDARY (if time): block_score WMMA (separate from cand_score
WMMA which iter5 PRIMARY already did).
iter6 TERTIARY: cand_score Stage A WMMA.

Bit-identical: the inner loop processes the same set of bytes and scales
in the same partition (sum0 even-i, sum1 odd-i). Transpose only changes
the address mapping; same fp32 multiplications and additions.

Tests: tests/srt/test_dsa_indexer_iter5_fma2.py adapted to iter6
transpose variant; all 48 existing iter1-5 regression tests retained.

Risk register:
- Stage write bank conflicts: each thread writes 64 bytes to columns of
  smem_values_t. Consecutive threads write the same row at consecutive
  columns, so 4 threads share a bank, so 4-way bank conflict per warp.
  Mitigation: this is OFF the hot path (one-shot staging). Cost is
  ~32 cycles for 4-way conflict vs ~8 for conflict-free, but happens
  exactly once per CTA invocation. Hot loop savings dwarf this.
- LDS.b16 from byte-offset address: smem_values_t[d][:] is 128 bytes
  aligned to 128 B boundary (since 64 rows times 128-aligned column start),
  so any pair (i, i+1) where i is even is naturally 2-B aligned.
- Odd token_count tail: iter5 already handles via scalar tail; same
  pattern applies for iter6 (single LDS.b8 + scalar FFMA at pair_end).
