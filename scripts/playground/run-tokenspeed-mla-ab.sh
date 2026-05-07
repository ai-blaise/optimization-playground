#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-R1}
OUT_DIR=${OUT_DIR:-/tmp/sglang_tokenspeed_mla_ab_$(date +%s)}
PYTHON_BIN=${PYTHON_BIN:-python3}
HOST=${HOST:-127.0.0.1}
BASELINE_PORT=${BASELINE_PORT:-31000}
CANDIDATE_PORT=${CANDIDATE_PORT:-32000}
BASELINE_BACKEND=${BASELINE_BACKEND:-trtllm_mla}
CANDIDATE_BACKEND=${CANDIDATE_BACKEND:-tokenspeed_mla}
TP_SIZE=${TP_SIZE:-8}
DP_SIZE=${DP_SIZE:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
PAGE_SIZE=${PAGE_SIZE:-64}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MATRIX=${MATRIX:-"8192:1024 16384:1024 16384:4096 32768:4096 65536:4096"}
BENCH_CONCURRENCY=${BENCH_CONCURRENCY:-16}
BENCH_PROMPTS=${BENCH_PROMPTS:-64}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-900}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUT_DIR}"

common_args=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --page-size "${PAGE_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
)

if [[ -n "${LOAD_FORMAT:-}" ]]; then
  common_args+=(--load-format "${LOAD_FORMAT}")
fi

if [[ -n "${QUANTIZATION:-}" ]]; then
  common_args+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${ATTENTION_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${ATTENTION_EXTRA_ARGS})
  common_args+=("${extra_args[@]}")
fi

launch_server() {
  local name=$1
  local backend=$2
  local port=$3
  local log_file=$4
  local args=(
    "${common_args[@]}"
    --attention-backend "${backend}"
    --host "${HOST}"
    --port "${port}"
  )
  echo "launch name=${name} backend=${backend} log=${log_file}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q -m sglang.launch_server' "${PYTHON_BIN}"
    printf ' %q' "${args[@]}"
    printf '\n'
    return
  fi
  "${PYTHON_BIN}" -m sglang.launch_server "${args[@]}" >"${log_file}" 2>&1 &
  echo $!
}

wait_ready() {
  local port=$1
  local pid=$2
  local log_file=$3
  local attempts=$((SERVER_READY_TIMEOUT_SECONDS / 5))
  if (( attempts < 1 )); then
    attempts=1
  fi
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      tail -240 "${log_file}" || true
      return 1
    fi
    sleep 5
  done
  tail -240 "${log_file}" || true
  return 1
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_backend() {
  local name=$1
  local backend=$2
  local port=$3
  local backend_dir="${OUT_DIR}/${name}"
  mkdir -p "${backend_dir}"

  SERVER_PID=""
  if [[ "${DRY_RUN}" == "1" ]]; then
    launch_server "${name}" "${backend}" "${port}" "${backend_dir}/server.log"
  else
    SERVER_PID=$(launch_server "${name}" "${backend}" "${port}" "${backend_dir}/server.log")
    wait_ready "${port}" "${SERVER_PID}" "${backend_dir}/server.log"
  fi

  for spec in ${MATRIX}; do
    input_len=${spec%%:*}
    output_len=${spec##*:}
    bench_name="${input_len}_${output_len}"
    echo "bench name=${name} backend=${backend} input=${input_len} output=${output_len}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q -m sglang.bench_serving --backend sglang-oai --host %q --port %q' \
        "${PYTHON_BIN}" "${HOST}" "${port}"
      printf ' --model %q --dataset-name random --random-range-ratio 1' "${MODEL_PATH}"
      printf ' --random-input-len %q --random-output-len %q' "${input_len}" "${output_len}"
      printf ' --max-concurrency %q --num-prompts %q --flush-cache --disable-tqdm' \
        "${BENCH_CONCURRENCY}" "${BENCH_PROMPTS}"
      printf ' --output-file %q\n' "${backend_dir}/${bench_name}.json"
    else
      "${PYTHON_BIN}" -m sglang.bench_serving \
        --backend sglang-oai \
        --host "${HOST}" \
        --port "${port}" \
        --model "${MODEL_PATH}" \
        --dataset-name random \
        --random-range-ratio 1 \
        --random-input-len "${input_len}" \
        --random-output-len "${output_len}" \
        --max-concurrency "${BENCH_CONCURRENCY}" \
        --num-prompts "${BENCH_PROMPTS}" \
        --flush-cache \
        --disable-tqdm \
        --output-file "${backend_dir}/${bench_name}.json" \
        | tee "${backend_dir}/${bench_name}.txt"
    fi
  done

  cleanup
  SERVER_PID=""
}

run_backend baseline "${BASELINE_BACKEND}" "${BASELINE_PORT}"
run_backend candidate "${CANDIDATE_BACKEND}" "${CANDIDATE_PORT}"

echo "results=${OUT_DIR}"
