#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${OUT_DIR:-/tmp/sglang_flashsampling_ikp_$(date +%s)}
IKP_REPO=${IKP_REPO:-${HOME}/intra-kernel-profiler}
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}
VOCAB_SIZE=${VOCAB_SIZE:-151936}
HIDDEN_SIZE=${HIDDEN_SIZE:-2048}
BATCH_SIZES=${BATCH_SIZES:-"1 8 32 128 256"}
WARMUP_ITERS=${WARMUP_ITERS:-10}
ITERS=${ITERS:-50}
KERNEL_REGEX=${KERNEL_REGEX:-"fused_mm_sample|triton|gemm|matmul|argmax|rand|uniform|exponential"}
PROFILE_CAPTURE_RANGE=${PROFILE_CAPTURE_RANGE:-1}
FLASHSAMPLING_PROVIDER=${FLASHSAMPLING_PROVIDER:-triton}

mkdir -p "${OUT_DIR}/nsys" "${OUT_DIR}/ikp"

cat >"${OUT_DIR}/flashsampling_kernel_bench.py" <<'PY'
import json
import os
import pathlib

import torch

provider = os.environ.get("FLASHSAMPLING_PROVIDER", "triton")
if provider == "target":
    if torch.cuda.get_device_capability()[0] >= 10:
        from sglang.srt.layers.flashsampling.target_kernel_blackwell import (
            fused_mm_sample_blackwell as fused_mm_sample,
        )
    else:
        from sglang.srt.layers.flashsampling.target_kernel import (
            fused_mm_sample_target as fused_mm_sample,
        )
elif provider == "triton":
    from sglang.srt.layers.flashsampling.core import (
        fused_mm_sample_triton as fused_mm_sample,
    )
else:
    raise ValueError(f"unknown FLASHSAMPLING_PROVIDER={provider!r}")


device = torch.device(f"cuda:{os.environ.get('CUDA_DEVICE', '0')}")
torch.cuda.set_device(device)

vocab_size = int(os.environ["VOCAB_SIZE"])
hidden_size = int(os.environ["HIDDEN_SIZE"])
batch_sizes = [int(x) for x in os.environ["BATCH_SIZES"].split()]
warmup_iters = int(os.environ["WARMUP_ITERS"])
iters = int(os.environ["ITERS"])
out_dir = pathlib.Path(os.environ["OUT_DIR"])
profile_capture_range = os.environ.get("PROFILE_CAPTURE_RANGE", "1") == "1"

torch.manual_seed(0)
weights = (
    torch.randn(vocab_size, hidden_size, device=device, dtype=torch.bfloat16)
    / hidden_size**0.5
)
temperature = torch.tensor(0.6, device=device, dtype=torch.float32)


def run_warmup(fn):
    for _ in range(warmup_iters):
        fn()


def measure(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def dense_sample(hidden_states):
    logits = (hidden_states @ weights.T).float() / temperature
    noise = -torch.log(-torch.log(torch.rand_like(logits)))
    return (logits + noise).argmax(dim=-1)


cases = []
for batch_size in batch_sizes:
    hidden_states = torch.randn(
        batch_size, hidden_size, device=device, dtype=torch.bfloat16
    )
    seed = 0

    def fused():
        nonlocal_seed[0] += 1
        return fused_mm_sample(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temperature,
            seed=nonlocal_seed[0],
            valid_vocab_size=vocab_size,
        )

    nonlocal_seed = [seed]
    run_warmup(lambda: dense_sample(hidden_states))
    run_warmup(fused)
    cases.append((batch_size, hidden_states, fused))

torch.cuda.synchronize()
if profile_capture_range:
    torch.cuda.cudart().cudaProfilerStart()

rows = []
for batch_size, hidden_states, fused in cases:
    dense_ms = measure(lambda: dense_sample(hidden_states))
    fused_ms = measure(fused)
    rows.append(
        {
            "batch_size": batch_size,
            "dense_ms": dense_ms,
            "flashsampling_ms": fused_ms,
            "speedup_pct": (dense_ms - fused_ms) / dense_ms * 100.0,
            "provider": provider,
        }
    )

if profile_capture_range:
    torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()

with (out_dir / "kernel_timing.json").open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print("batch_size,dense_ms,flashsampling_ms,speedup_pct")
for row in rows:
    print(
        f"{row['batch_size']},{row['dense_ms']:.4f},"
        f"{row['flashsampling_ms']:.4f},{row['speedup_pct']:.2f}"
    )
PY

export CUDA_DEVICE VOCAB_SIZE HIDDEN_SIZE BATCH_SIZES WARMUP_ITERS ITERS OUT_DIR
export PROFILE_CAPTURE_RANGE FLASHSAMPLING_PROVIDER
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE}}

nsys_args=(
  profile
  --output="${OUT_DIR}/nsys/flashsampling_kernel"
  --force-overwrite=true
  --trace=cuda,nvtx
  --sample=none
  --cpuctxsw=none
)
if [[ "${PROFILE_CAPTURE_RANGE}" == "1" ]]; then
  nsys_args+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
fi

nsys "${nsys_args[@]}" \
  "${PYTHON_BIN}" "${OUT_DIR}/flashsampling_kernel_bench.py" \
  | tee "${OUT_DIR}/kernel_timing.txt"

if [[ ! -d "${IKP_REPO}" ]]; then
  echo "IKP repo not found at ${IKP_REPO}; skipping IKP import." >&2
  exit 0
fi

"${PYTHON_BIN}" "${IKP_REPO}/scripts/ikp_nsys_import.py" \
  --nsys-rep "${OUT_DIR}/nsys/flashsampling_kernel.nsys-rep" \
  --out-dir "${OUT_DIR}/ikp" \
  --kernel-regex "${KERNEL_REGEX}"

"${PYTHON_BIN}" - "${OUT_DIR}/ikp/nsys_kernels.json" <<'PY' | tee "${OUT_DIR}/ikp_kernel_summary.txt"
import collections
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
kernels = data.get("kernels", data) if isinstance(data, dict) else data

by_name = collections.defaultdict(lambda: {"count": 0, "duration_ns": 0})
for kernel in kernels:
    name = kernel.get("name") or kernel.get("demangled_name") or "<unknown>"
    item = by_name[name]
    item["count"] += 1
    item["duration_ns"] += int(kernel.get("duration_ns", 0))

print("kernel,count,total_ms,mean_us")
for name, item in sorted(
    by_name.items(), key=lambda kv: kv[1]["duration_ns"], reverse=True
)[:25]:
    total_ms = item["duration_ns"] / 1e6
    mean_us = item["duration_ns"] / max(item["count"], 1) / 1e3
    print(f"{name},{item['count']},{total_ms:.3f},{mean_us:.3f}")
PY

echo "IKP artifacts: ${OUT_DIR}"
