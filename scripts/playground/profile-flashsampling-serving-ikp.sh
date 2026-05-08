#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${OUT_DIR:-/tmp/sglang_flashsampling_serving_ikp_$(date +%s)}
IKP_REPO=${IKP_REPO:-${HOME}/intra-kernel-profiler}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
HOST=${HOST:-127.0.0.1}
CONCURRENCY=${CONCURRENCY:-32}
NUM_PROMPTS=${NUM_PROMPTS:-256}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
DATASET_PATH=${DATASET_PATH:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-900}
SAMPLING_BACKEND=${SAMPLING_BACKEND:-}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
FLASHSAMPLING_WARMUP_BATCH_SIZES=${FLASHSAMPLING_WARMUP_BATCH_SIZES:-}
FLASHSAMPLING_MIN_BATCH_SIZE=${FLASHSAMPLING_MIN_BATCH_SIZE:-}
FLASHSAMPLING_MAX_BATCH_SIZE=${FLASHSAMPLING_MAX_BATCH_SIZE:-}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0.6,"top_k":-1,"top_p":1.0}'}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}
DISABLE_PIECEWISE_CUDA_GRAPH=${DISABLE_PIECEWISE_CUDA_GRAPH:-1}
NSYS_DELAY_SECONDS=${NSYS_DELAY_SECONDS:-10}
NSYS_DURATION_SECONDS=${NSYS_DURATION_SECONDS:-45}
KERNEL_REGEX=${KERNEL_REGEX:-".*"}

mkdir -p "${OUT_DIR}"

free_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

for port in range(30000, 36000):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
PY
}

wait_for_server() {
  local port=$1
  local pid=$2
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SECONDS}"); do
    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

server_cmd() {
  local port=$1
  local variant=$2
  local args=(
    -m sglang.launch_server
    --model-path "${MODEL}"
    --trust-remote-code
    --context-length "${MAX_MODEL_LEN}"
    --disable-radix-cache
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --host "${HOST}"
    --port "${port}"
  )
  if [[ -n "${SAMPLING_BACKEND}" ]]; then
    args+=(--sampling-backend "${SAMPLING_BACKEND}")
  fi
  if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    args+=(--disable-cuda-graph)
  fi
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
    args+=(--disable-piecewise-cuda-graph)
  fi
  if [[ "${variant}" == "flashsampling" ]]; then
    args+=(--enable-flashsampling --flashsampling-fallback error)
    if [[ -n "${FLASHSAMPLING_MIN_BATCH_SIZE}" ]]; then
      args+=(--flashsampling-min-batch-size "${FLASHSAMPLING_MIN_BATCH_SIZE}")
    fi
    if [[ -n "${FLASHSAMPLING_MAX_BATCH_SIZE}" ]]; then
      args+=(--flashsampling-max-batch-size "${FLASHSAMPLING_MAX_BATCH_SIZE}")
    fi
    if [[ -n "${FLASHSAMPLING_WARMUP_BATCH_SIZES}" ]]; then
      args+=(--flashsampling-warmup-batch-sizes)
      read -r -a warmup_batch_sizes <<<"${FLASHSAMPLING_WARMUP_BATCH_SIZES}"
      args+=("${warmup_batch_sizes[@]}")
    fi
  fi
  printf '%q ' "${PYTHON_BIN}" "${args[@]}"
}

bench_cmd() {
  local port=$1
  local output_file=$2
  local warmup_requests=${WARMUP_REQUESTS}
  if [[ "${warmup_requests}" == "concurrency" ]]; then
    warmup_requests=${CONCURRENCY}
  fi
  printf '%q ' "${PYTHON_BIN}" -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --host "${HOST}" \
    --port "${port}" \
    --model "${MODEL}"
  if [[ -n "${DATASET_PATH}" ]]; then
    printf '%q ' \
      --dataset-name custom \
      --dataset-path "${DATASET_PATH}" \
      --sharegpt-output-len "${OUTPUT_LEN}" \
      --sharegpt-context-len "${MAX_MODEL_LEN}"
  else
    printf '%q ' \
      --dataset-name random \
      --random-input-len "${INPUT_LEN}" \
      --random-output-len "${OUTPUT_LEN}" \
      --random-range-ratio 1
  fi
  printf '%q ' \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${CONCURRENCY}" \
    --max-concurrency "${CONCURRENCY}" \
    --seed 1 \
    --flush-cache \
    --warmup-requests "${warmup_requests}" \
    --disable-tqdm \
    --extra-request-body "${EXTRA_REQUEST_BODY}" \
    --output-file "${output_file}"
}

import_ikp() {
  local variant_dir=$1
  if [[ ! -d "${IKP_REPO}" ]]; then
    echo "IKP repo not found at ${IKP_REPO}; skipping IKP import." >&2
    return
  fi
  rm -f "${variant_dir}/server.sqlite"
  "${PYTHON_BIN}" "${IKP_REPO}/scripts/ikp_nsys_import.py" \
    --nsys-rep "${variant_dir}/server.nsys-rep" \
    --out-dir "${variant_dir}/ikp" \
    --kernel-regex "${KERNEL_REGEX}"
  "${PYTHON_BIN}" - "${variant_dir}/ikp/nsys_kernels.json" <<'PY' \
    | tee "${variant_dir}/ikp_kernel_summary.txt"
import collections
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
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
)[:40]:
    total_ms = item["duration_ns"] / 1e6
    mean_us = item["duration_ns"] / max(item["count"], 1) / 1e3
    print(f"{name},{item['count']},{total_ms:.3f},{mean_us:.3f}")
PY
}

run_variant() {
  local variant=$1
  local variant_dir="${OUT_DIR}/${variant}"
  local port
  mkdir -p "${variant_dir}"
  port=$(free_port)

  local serve
  serve=$(server_cmd "${port}" "${variant}")
  echo "${serve}" >"${variant_dir}/serve.cmd"
  nsys profile \
    --output="${variant_dir}/server" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --delay="${NSYS_DELAY_SECONDS}" \
    --duration="${NSYS_DURATION_SECONDS}" \
    bash -lc "${serve}" >"${variant_dir}/server.log" 2>&1 &
  local server_pid=$!

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server failed for ${variant}; log=${variant_dir}/server.log" >&2
    tail -200 "${variant_dir}/server.log" >&2
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    return 1
  fi

  local bench
  bench=$(bench_cmd "${port}" "${variant_dir}/bench.jsonl")
  echo "${bench}" >"${variant_dir}/bench.cmd"
  local bench_status=0
  bash -lc "${bench}" | tee "${variant_dir}/bench.txt" || bench_status=$?

  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" >/dev/null 2>&1 || true
  if (( bench_status != 0 )); then
    return "${bench_status}"
  fi

  if [[ -f "${variant_dir}/server.nsys-rep" ]]; then
    import_ikp "${variant_dir}"
  else
    echo "NSys report was not produced for ${variant}." >&2
    return 1
  fi
}

run_variant baseline
run_variant flashsampling

echo "Serving IKP artifacts: ${OUT_DIR}"
