#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1}
SMC_DRAFT_MODEL_PATH=${SMC_DRAFT_MODEL_PATH:-BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}
OUT_DIR=${OUT_DIR:-/tmp/sglang_ncclx_pd_matrix_$(date +%s)}
PYTHON_BIN=${PYTHON_BIN:-python3}
HOST=${HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-31000}
DECODE_PORT=${DECODE_PORT:-32000}
ROUTER_PORT=${ROUTER_PORT:-33000}
PREFILL_DEVICES=${PREFILL_DEVICES:-0,1,2,3}
DECODE_DEVICES=${DECODE_DEVICES:-4,5,6,7}
PREFILL_CP_SIZE=${PREFILL_CP_SIZE:-4}
PREFILL_TP_SIZE=${PREFILL_TP_SIZE:-4}
PREFILL_DP_SIZE=${PREFILL_DP_SIZE:-1}
PREFILL_ENABLE_DP_ATTENTION=${PREFILL_ENABLE_DP_ATTENTION:-0}
DECODE_TP_SIZE=${DECODE_TP_SIZE:-4}
DECODE_DP_SIZE=${DECODE_DP_SIZE:-4}
DECODE_EP_SIZE=${DECODE_EP_SIZE:-4}
DECODE_MOE_A2A_BACKEND=${DECODE_MOE_A2A_BACKEND:-deepep}
DECODE_ENABLE_DP_LM_HEAD=${DECODE_ENABLE_DP_LM_HEAD:-1}
DECODE_MOE_DENSE_TP_SIZE=${DECODE_MOE_DENSE_TP_SIZE:-1}
DECODE_DP_ATTENTION_LOCAL_CONTROL_BROADCAST=${DECODE_DP_ATTENTION_LOCAL_CONTROL_BROADCAST:-1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
NSA_INDEXER_MODE=${NSA_INDEXER_MODE:-indexcache}
NSA_INDEXCACHE_FREQ=${NSA_INDEXCACHE_FREQ:-4}
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
ENABLE_TORCHCOMMS_NCCLX_RDMA=${ENABLE_TORCHCOMMS_NCCLX_RDMA:-0}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-900}

export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=${SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER:-1}

torch_lib_dir=$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import pathlib
import torch

path = pathlib.Path(torch.__file__).resolve().parent / "lib"
if path.is_dir():
    print(path)
PY
)
if [[ -n "${torch_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${torch_lib_dir}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${DISAGG_TRANSFER_BACKEND}" == "nixl" && -z "${UCX_MODULE_DIR:-}" ]]; then
  nixl_site=$("${PYTHON_BIN}" -c 'import pathlib, nixl_cu13; print(pathlib.Path(nixl_cu13.__file__).resolve().parents[1])' 2>/dev/null || true)
  if [[ -n "${nixl_site}" && -d "${nixl_site}/nixl_cu13.libs/ucx" ]]; then
    export UCX_MODULE_DIR="${nixl_site}/nixl_cu13.libs/ucx"
    export LD_LIBRARY_PATH="${nixl_site}/nixl_cu13.libs:${UCX_MODULE_DIR}:${LD_LIBRARY_PATH:-}"
  fi
fi

mkdir -p "${OUT_DIR}"

common_args=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --nsa-indexer-mode "${NSA_INDEXER_MODE}"
  --nsa-indexcache-freq "${NSA_INDEXCACHE_FREQ}"
  --enable-hisparse
  --disable-radix-cache
  --enable-turboquant-dense-kv-cache
  --turboquant-dense-kv-preset latent_2p5bit_nc
  --turboquant-execution-mode fused_decode
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --disaggregation-transfer-backend "${DISAGG_TRANSFER_BACKEND}"
  --disaggregation-bootstrap-port "${DISAGG_BOOTSTRAP_PORT}"
)

if [[ -n "${HISA_BLOCK_SIZE:-}" ]]; then
  common_args+=(--hisa-block-size "${HISA_BLOCK_SIZE}")
fi

if [[ -n "${HISA_BLOCK_TOPK:-}" ]]; then
  common_args+=(--hisa-block-topk "${HISA_BLOCK_TOPK}")
fi

if [[ -n "${HISA_MIN_SEQ_LEN:-}" ]]; then
  common_args+=(--hisa-min-seq-len "${HISA_MIN_SEQ_LEN}")
fi

if [[ -n "${HISA_EXECUTION_MODE:-}" ]]; then
  common_args+=(--hisa-execution-mode "${HISA_EXECUTION_MODE}")
fi

if [[ -n "${LOAD_FORMAT:-}" ]]; then
  common_args+=(--load-format "${LOAD_FORMAT}")
fi

if [[ -n "${QUANTIZATION:-}" ]]; then
  common_args+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${FP8_GEMM_BACKEND:-}" ]]; then
  common_args+=(--fp8-gemm-backend "${FP8_GEMM_BACKEND}")
fi

if [[ -n "${FP4_GEMM_BACKEND:-}" ]]; then
  common_args+=(--fp4-gemm-backend "${FP4_GEMM_BACKEND}")
fi

if [[ -n "${MOE_RUNNER_BACKEND:-}" ]]; then
  common_args+=(--moe-runner-backend "${MOE_RUNNER_BACKEND}")
fi

if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  common_args+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  common_args+=(--disable-cuda-graph)
fi

if [[ "${SKIP_SERVER_WARMUP:-0}" == "1" ]]; then
  common_args+=(--skip-server-warmup)
fi

if [[ -n "${DISAGG_IB_DEVICE:-}" ]]; then
  common_args+=(--disaggregation-ib-device "${DISAGG_IB_DEVICE}")
fi

prefill_args=(
  "${common_args[@]}"
  --tp-size "${PREFILL_TP_SIZE}"
  --dp-size "${PREFILL_DP_SIZE}"
  --disaggregation-mode prefill
  --enable-nsa-prefill-context-parallel
  --attn-cp-size "${PREFILL_CP_SIZE}"
  --nsa-prefill-cp-mode round-robin-split
  --nsa-prefill-cp-kv-storage-mode layersplit
  --nsa-prefill-cp-layersplit-layout interleaved
  --host "${HOST}"
  --port "${PREFILL_PORT}"
)

if [[ "${PREFILL_ENABLE_DP_ATTENTION}" == "1" ]]; then
  prefill_args+=(--enable-dp-attention)
  if [[ "${PREFILL_DP_ATTENTION_LOCAL_CONTROL_BROADCAST:-1}" == "1" ]]; then
    prefill_args+=(--enable-dp-attention-local-control-broadcast)
  fi
fi

decode_args=(
  "${common_args[@]}"
  --tp-size "${DECODE_TP_SIZE}"
  --dp-size "${DECODE_DP_SIZE}"
  --enable-dp-attention
  --disaggregation-mode decode
  --ep-size "${DECODE_EP_SIZE}"
  --moe-a2a-backend "${DECODE_MOE_A2A_BACKEND}"
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

if [[ "${DECODE_DP_ATTENTION_LOCAL_CONTROL_BROADCAST}" == "1" ]]; then
  decode_args+=(--enable-dp-attention-local-control-broadcast)
fi

if [[ "${DECODE_ENABLE_DP_LM_HEAD}" == "1" ]]; then
  decode_args+=(--enable-dp-lm-head)
fi

if [[ -n "${DECODE_MOE_DENSE_TP_SIZE}" ]]; then
  decode_args+=(--moe-dense-tp-size "${DECODE_MOE_DENSE_TP_SIZE}")
fi

launch_server() {
  local role=$1
  local backend=$2
  local log_file=$3
  shift 3
  local backend_args=()
  if [[ "${backend}" == "ncclx" ]]; then
    backend_args+=(--device-collective-backend ncclx --torchcomms-ncclx-strict)
    if [[ "${ENABLE_TORCHCOMMS_NCCLX_RDMA}" == "1" ]]; then
      backend_args+=(--enable-torchcomms-ncclx-rdma)
    fi
    if [[ "${TORCHCOMMS_NCCLX_ABORT_ON_TIMEOUT:-0}" == "1" ]]; then
      backend_args+=(--torchcomms-ncclx-abort-on-timeout)
    fi
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

launch_router() {
  local log_file=$1
  local router_args=(
    --pd-disaggregation
    --prefill "http://${HOST}:${PREFILL_PORT}"
    --decode "http://${HOST}:${DECODE_PORT}"
    --host "${HOST}"
    --port "${ROUTER_PORT}"
  )
  echo "launch role=router log=${log_file}" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q -m sglang_router.launch_router' "${PYTHON_BIN}"
    printf ' %q' "${router_args[@]}"
    printf '\n'
    return
  fi
  "${PYTHON_BIN}" -m sglang_router.launch_router "${router_args[@]}" >"${log_file}" 2>&1 &
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

  if [[ "${DRY_RUN}" == "1" ]]; then
    launch_router "${backend_dir}/router.log"
  else
    router_pid=$(launch_router "${backend_dir}/router.log")
    PIDS_TO_CLEANUP="${PIDS_TO_CLEANUP} ${router_pid}"
    wait_ready "${ROUTER_PORT}" "${router_pid}" "${backend_dir}/router.log"
  fi

  for spec in ${MATRIX}; do
    input_len=${spec%%:*}
    output_len=${spec##*:}
    name="${input_len}_${output_len}"
    echo "bench backend=${backend} input=${input_len} output=${output_len}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q -m sglang.bench_serving --backend sglang-oai --host %q --port %q' \
        "${PYTHON_BIN}" "${HOST}" "${ROUTER_PORT}"
      printf ' --model %q --dataset-name random --random-range-ratio 1' "${MODEL_PATH}"
      printf ' --random-input-len %q --random-output-len %q' "${input_len}" "${output_len}"
      printf ' --max-concurrency %q --num-prompts %q --flush-cache' \
        "${BENCH_CONCURRENCY}" "${BENCH_PROMPTS}"
      printf ' --output-file %q\n' "${backend_dir}/${name}.json"
    else
      "${PYTHON_BIN}" -m sglang.bench_serving \
        --backend sglang-oai \
        --host "${HOST}" \
        --port "${ROUTER_PORT}" \
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
