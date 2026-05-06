#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1}
SMC_DRAFT_MODEL_PATH=${SMC_DRAFT_MODEL_PATH:-/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}
OUT_DIR=${OUT_DIR:-/tmp/sglang_ncclx_pd_matrix_$(date +%s)}
PYTHON_BIN=${PYTHON_BIN:-python3}
HOST=${HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-31000}
DECODE_PORT=${DECODE_PORT:-32000}
PREFILL_DEVICES=${PREFILL_DEVICES:-0,1,2,3}
DECODE_DEVICES=${DECODE_DEVICES:-4,5,6,7}
PREFILL_CP_SIZE=${PREFILL_CP_SIZE:-4}
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-4}
EP_SIZE=${EP_SIZE:-4}
DISAGG_TRANSFER_BACKEND=${DISAGG_TRANSFER_BACKEND:-mooncake}
DISAGG_BOOTSTRAP_PORT=${DISAGG_BOOTSTRAP_PORT:-8998}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
BENCH_CONCURRENCY=${BENCH_CONCURRENCY:-16}
BENCH_PROMPTS=${BENCH_PROMPTS:-64}
MATRIX=${MATRIX:-"8192:1024 16384:1024 16384:4096 32768:4096 65536:4096"}
BACKENDS=${BACKENDS:-"default ncclx"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUT_DIR}"

common_args=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --enable-dp-attention
  --nsa-indexer-mode indexcache
  --enable-turboquant-dense-kv-cache
  --turboquant-dense-kv-preset latent_2p5bit_nc
  --turboquant-execution-mode fused_decode
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --disaggregation-transfer-backend "${DISAGG_TRANSFER_BACKEND}"
  --disaggregation-bootstrap-port "${DISAGG_BOOTSTRAP_PORT}"
)

if [[ -n "${DISAGG_IB_DEVICE:-}" ]]; then
  common_args+=(--disaggregation-ib-device "${DISAGG_IB_DEVICE}")
fi

prefill_args=(
  "${common_args[@]}"
  --disaggregation-mode prefill
  --enable-nsa-prefill-context-parallel
  --attn-cp-size "${PREFILL_CP_SIZE}"
  --nsa-prefill-cp-mode round-robin-split
  --nsa-prefill-cp-kv-storage-mode layersplit
  --nsa-prefill-cp-layersplit-layout interleaved
  --host "${HOST}"
  --port "${PREFILL_PORT}"
)

decode_args=(
  "${common_args[@]}"
  --disaggregation-mode decode
  --ep-size "${EP_SIZE}"
  --enable-hisparse
  --disable-radix-cache
  --speculative-algorithm SMC
  --speculative-draft-model-path "${SMC_DRAFT_MODEL_PATH}"
  --speculative-draft-model-quantization "${SMC_DRAFT_MODEL_QUANTIZATION:-fp8}"
  --speculative-draft-attention-backend "${SMC_DRAFT_ATTENTION_BACKEND:-triton}"
  --smc-n-particles "${SMC_N_PARTICLES:-4}"
  --smc-gamma "${SMC_GAMMA:-6}"
  --smc-draft-kv-cache-dtype "${SMC_DRAFT_KV_CACHE_DTYPE:-fp8_e4m3}"
  --speculative-attention-mode decode
  --host "${HOST}"
  --port "${DECODE_PORT}"
)

launch_server() {
  local role=$1
  local backend=$2
  local log_file=$3
  shift 3
  local backend_args=()
  if [[ "${backend}" == "ncclx" ]]; then
    backend_args+=(--device-collective-backend ncclx --torchcomms-ncclx-strict)
    backend_args+=(--enable-torchcomms-ncclx-rdma)
    if [[ -n "${TORCHCOMMS_NCCLX_HINTS:-}" ]]; then
      backend_args+=(--torchcomms-ncclx-hints "${TORCHCOMMS_NCCLX_HINTS}")
    fi
  fi
  echo "launch role=${role} backend=${backend} log=${log_file}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q %q -m sglang.launch_server' \
      "${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}"
    printf ' %q' "$@"
    if (( ${#backend_args[@]} > 0 )); then
      printf ' %q' "${backend_args[@]}"
    fi
    printf '\n'
    return
  fi
  if (( ${#backend_args[@]} > 0 )); then
    "${PYTHON_BIN}" -m sglang.launch_server "$@" "${backend_args[@]}" >"${log_file}" 2>&1 &
  else
    "${PYTHON_BIN}" -m sglang.launch_server "$@" >"${log_file}" 2>&1 &
  fi
  echo $!
}

wait_ready() {
  local port=$1
  local pid=$2
  local log_file=$3
  for _ in $(seq 1 180); do
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
  for pid in ${PIDS_TO_CLEANUP:-}; do
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for backend in ${BACKENDS}; do
  backend_dir="${OUT_DIR}/${backend}"
  mkdir -p "${backend_dir}"
  PIDS_TO_CLEANUP=""

  if [[ "${DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${PREFILL_DEVICES}" \
      launch_server prefill "${backend}" "${backend_dir}/prefill.log" "${prefill_args[@]}"
  else
    prefill_pid=$(
      CUDA_VISIBLE_DEVICES="${PREFILL_DEVICES}" \
        launch_server prefill "${backend}" "${backend_dir}/prefill.log" "${prefill_args[@]}"
    )
    PIDS_TO_CLEANUP="${PIDS_TO_CLEANUP} ${prefill_pid}"
    wait_ready "${PREFILL_PORT}" "${prefill_pid}" "${backend_dir}/prefill.log"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${DECODE_DEVICES}" \
      launch_server decode "${backend}" "${backend_dir}/decode.log" "${decode_args[@]}"
  else
    decode_pid=$(
      CUDA_VISIBLE_DEVICES="${DECODE_DEVICES}" \
        launch_server decode "${backend}" "${backend_dir}/decode.log" "${decode_args[@]}"
    )
    PIDS_TO_CLEANUP="${PIDS_TO_CLEANUP} ${decode_pid}"
    wait_ready "${DECODE_PORT}" "${decode_pid}" "${backend_dir}/decode.log"
  fi

  for spec in ${MATRIX}; do
    input_len=${spec%%:*}
    output_len=${spec##*:}
    name="${input_len}_${output_len}"
    echo "bench backend=${backend} input=${input_len} output=${output_len}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q -m sglang.bench_serving --backend sglang-oai --host %q --port %q' \
        "${PYTHON_BIN}" "${HOST}" "${DECODE_PORT}"
      printf ' --model %q --dataset-name random --random-range-ratio 1' "${MODEL_PATH}"
      printf ' --random-input-len %q --random-output-len %q' "${input_len}" "${output_len}"
      printf ' --max-concurrency %q --num-prompts %q --flush-cache' \
        "${BENCH_CONCURRENCY}" "${BENCH_PROMPTS}"
      printf ' --output-file %q\n' "${backend_dir}/${name}.json"
    else
      "${PYTHON_BIN}" -m sglang.bench_serving \
        --backend sglang-oai \
        --host "${HOST}" \
        --port "${DECODE_PORT}" \
        --model "${MODEL_PATH}" \
        --dataset-name random \
        --random-range-ratio 1 \
        --random-input-len "${input_len}" \
        --random-output-len "${output_len}" \
        --max-concurrency "${BENCH_CONCURRENCY}" \
        --num-prompts "${BENCH_PROMPTS}" \
        --flush-cache \
        --disable-tqdm \
        --output-file "${backend_dir}/${name}.json" \
        | tee "${backend_dir}/${name}.txt"
    fi
  done

  cleanup
  PIDS_TO_CLEANUP=""
done

echo "results=${OUT_DIR}"
