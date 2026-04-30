#!/usr/bin/env bash
set -euo pipefail

BASELINE_REPO=${BASELINE_REPO:-"${HOME}/optimization-playground-combo-baseline"}
REPO=${REPO:-"${HOME}/optimization-playground-combo-candidate"}
VENV=${VENV:-"${HOME}/optimization-playground-codex/.venv"}
MODEL_ID=${MODEL_ID:-cerebras/DeepSeek-V3.2-REAP-345B-A37B}
MODEL_REVISION=${MODEL_REVISION:-4fd8e8c3e08442c4a6dde6dd3fa3dac481a0205b}
MODEL_PATH=${MODEL_PATH:-/models/cerebras/DeepSeek-V3.2-REAP-345B-A37B}
MODEL_LOCAL_FILES_ONLY=${MODEL_LOCAL_FILES_ONLY:-1}
SPECULATOR_MODEL_ID=${SPECULATOR_MODEL_ID:-BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}
SPECULATOR_MODEL_PATH=${SPECULATOR_MODEL_PATH:-/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP}
SPECULATOR_LOCAL_FILES_ONLY=${SPECULATOR_LOCAL_FILES_ONLY:-1}
INDEXCACHE_PATTERN=${INDEXCACHE_PATTERN:-FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-32000}
RUN_ROOT=${RUN_ROOT:-"${HOME}/reap-smcsd-combo-ab-$(date +%Y%m%d-%H%M%S)"}
RUN_SET=${RUN_SET:-pair}
BENCH_PROFILE=${BENCH_PROFILE:-gate8k1k}
BENCH_BACKEND=${BENCH_BACKEND:-sglang-oai}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-random}
BENCH_RANDOM_RANGE_RATIO=${BENCH_RANDOM_RANGE_RATIO:-1.0}
BENCH_SEED=${BENCH_SEED:-20260425}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-7200}
BENCH_OUTPUT_DETAILS=${BENCH_OUTPUT_DETAILS:-0}
BENCH_FLUSH_CACHE_AFTER_WARMUP=${BENCH_FLUSH_CACHE_AFTER_WARMUP:-1}
DECODE_NUM_PROMPTS=${DECODE_NUM_PROMPTS:-8}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-8}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-${DECODE_CONCURRENCY}}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-73728}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-1048576}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
TP_SIZE=${TP_SIZE:-8}
DP_SIZE=${DP_SIZE:-8}
BENCH_WARMUP_REQUESTS=${BENCH_WARMUP_REQUESTS:-${DP_SIZE}}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}
TARGET_ATTENTION_BACKEND=${TARGET_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}
DRAFT_ATTENTION_BACKEND=${DRAFT_ATTENTION_BACKEND:-triton}
DTYPE=${DTYPE:-}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
NSA_INDEXER_MODE=${NSA_INDEXER_MODE:-indexcache}
NSA_PREFILL_BACKEND=${NSA_PREFILL_BACKEND:-flashmla_kv}
NSA_DECODE_BACKEND=${NSA_DECODE_BACKEND:-flashmla_kv}
ENABLE_TURBOQUANT_DENSE_KV_CACHE=${ENABLE_TURBOQUANT_DENSE_KV_CACHE:-1}
TURBOQUANT_DENSE_KV_PRESET=${TURBOQUANT_DENSE_KV_PRESET:-latent_2p5bit_nc}
TURBOQUANT_RESIDUAL_WINDOW_SIZE=${TURBOQUANT_RESIDUAL_WINDOW_SIZE:-128}
TURBOQUANT_EXECUTION_MODE=${TURBOQUANT_EXECUTION_MODE:-fused_decode}
TURBOQUANT_MLA_DECODE_NUM_SPLITS=${TURBOQUANT_MLA_DECODE_NUM_SPLITS:-16}
SPECULATIVE_DRAFT_MODEL_QUANTIZATION=${SPECULATIVE_DRAFT_MODEL_QUANTIZATION:-fp8}
SMC_DRAFT_KV_CACHE_DTYPE=${SMC_DRAFT_KV_CACHE_DTYPE:-fp8_e4m3}
FP8_GEMM_BACKEND=${FP8_GEMM_BACKEND:-auto}
SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND=${SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND:-triton}
SPECULATIVE_MOE_RUNNER_BACKEND=${SPECULATIVE_MOE_RUNNER_BACKEND:-}
SMC_N_PARTICLES=${SMC_N_PARTICLES:-4}
SMC_GAMMA=${SMC_GAMMA:-6}
SMC_DRAFT_TEMPERATURE=${SMC_DRAFT_TEMPERATURE:-0.7}
SMC_TARGET_TEMPERATURE=${SMC_TARGET_TEMPERATURE:-1.0}
SMC_RESAMPLE_THRESHOLD=${SMC_RESAMPLE_THRESHOLD:-0.5}
SMC_RESAMPLE_METHOD=${SMC_RESAMPLE_METHOD:-systematic}
SMC_PROBE=${SMC_PROBE:-0}
SGLANG_SMC_PROBE_RECORD_PATH=${SGLANG_SMC_PROBE_RECORD_PATH:-}
SGLANG_SMC_DIAG_PATH=${SGLANG_SMC_DIAG_PATH:-}
SGLANG_SMC_PREFILL_STREAM_YIELD_MS=${SGLANG_SMC_PREFILL_STREAM_YIELD_MS:-0}
SGLANG_SMC_TARGET_VERIFY_GRAPH=${SGLANG_SMC_TARGET_VERIFY_GRAPH:-0}
BENCH_DISABLE_CUDA_GRAPH=${BENCH_DISABLE_CUDA_GRAPH:-0}
BENCH_CUDA_GRAPH_MAX_BS=${BENCH_CUDA_GRAPH_MAX_BS:-8}
BENCH_SKIP_SERVER_WARMUP=${BENCH_SKIP_SERVER_WARMUP:-0}
BENCH_DISABLE_CUSTOM_ALL_REDUCE=${BENCH_DISABLE_CUSTOM_ALL_REDUCE:-1}
GATE_MIN_CANDIDATE_DELTA_PCT=${GATE_MIN_CANDIDATE_DELTA_PCT:-0.0}
GATE_MAX_TTFT_REGRESSION_PCT=${GATE_MAX_TTFT_REGRESSION_PCT:-1.0}
GATE_TTFT_METRIC=${GATE_TTFT_METRIC:-mean_ttft_ms}
GATE_MIN_RETOKENIZED_OUTPUT_RATIO=${GATE_MIN_RETOKENIZED_OUTPUT_RATIO:-0.95}
GATE_MIN_FALLBACK_RETOKENIZED_OUTPUT_RATIO=${GATE_MIN_FALLBACK_RETOKENIZED_OUTPUT_RATIO:-0.75}
GATE_MIN_SERVER_OUTPUT_RATIO=${GATE_MIN_SERVER_OUTPUT_RATIO:-0.95}
GATE_MIN_GENERATED_CHARS_PER_TOKEN=${GATE_MIN_GENERATED_CHARS_PER_TOKEN:-0.25}
GATE_MIN_ITL_SAMPLES_PER_REQUEST=${GATE_MIN_ITL_SAMPLES_PER_REQUEST:-1}

if [[ "${HOSTNAME:-}" != "a4-us-001-rl9" ]]; then
  echo "This benchmark must run on a4-us-001-rl9; current host is ${HOSTNAME:-unknown}." >&2
  exit 2
fi

if [[ "${BENCH_RANDOM_RANGE_RATIO}" != "1.0" && "${BENCH_RANDOM_RANGE_RATIO}" != "1" ]]; then
  echo "SMC-SD combo gates require BENCH_RANDOM_RANGE_RATIO=1.0 for bounded inputs." >&2
  exit 2
fi

if [[ "${NSA_INDEXER_MODE}" != "indexcache" ]]; then
  echo "SMC-SD combo gates require NSA_INDEXER_MODE=indexcache." >&2
  exit 2
fi

if [[ "${ENABLE_TURBOQUANT_DENSE_KV_CACHE}" != "1" ]]; then
  echo "SMC-SD combo gates require ENABLE_TURBOQUANT_DENSE_KV_CACHE=1." >&2
  exit 2
fi

if (( BENCH_WARMUP_REQUESTS > 0 && BENCH_WARMUP_REQUESTS < DP_SIZE )); then
  echo "SMC-SD combo gates require BENCH_WARMUP_REQUESTS=0 or >= DP_SIZE (${DP_SIZE}) so all DP ranks participate in warmup collectives." >&2
  exit 2
fi

if [[ ${#INDEXCACHE_PATTERN} -ne 61 ]]; then
  echo "INDEXCACHE_PATTERN must be the 61-layer DeepSeek V3.2 REAP searched pattern." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
  # shellcheck disable=SC1091
  source /opt/rh/gcc-toolset-13/enable
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
export CUDA_HOME=${CUDA_HOME:-"${VENV}/lib/python3.12/site-packages/nvidia/cu13"}
export CUDA_PATH=${CUDA_PATH:-"${CUDA_HOME}"}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${VENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${VENV}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
LD_LIBRARY_PATH=$(
  printf '%s' "${LD_LIBRARY_PATH}" \
    | tr ':' '\n' \
    | awk 'NF && $0 != "/usr/local/nccl-rdma-sharp-plugins/lib"' \
    | paste -sd: -
)
export LD_LIBRARY_PATH
export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
export SGLANG_DISABLE_FLASHINFER_FUSED_TOPK_DEEPSEEK=${SGLANG_DISABLE_FLASHINFER_FUSED_TOPK_DEEPSEEK:-0}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
export NCCL_IB_DISABLE=${COMBO_NCCL_IB_DISABLE:-1}
export NCCL_NVLS_ENABLE=${COMBO_NCCL_NVLS_ENABLE:-0}
export NCCL_P2P_LEVEL=${COMBO_NCCL_P2P_LEVEL:-NVL}
export NCCL_P2P_DISABLE=${COMBO_NCCL_P2P_DISABLE:-0}
export NCCL_CUMEM_ENABLE=${COMBO_NCCL_CUMEM_ENABLE:-0}
export NCCL_ALGO=${COMBO_NCCL_ALGO:-Ring}
unset NCCL_NET
unset NCCL_NET_PLUGIN
unset NCCL_PROTO
unset NCCL_IB_HCA
unset NCCL_IB_GID_INDEX
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_READ
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
unset NCCL_CROSS_NIC
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_SPLIT_DATA_ON_QPS
unset NCCL_BUFFSIZE
unset NCCL_IB_TIMEOUT
unset NCCL_IB_RETRY_CNT
unset NCCL_IB_PCI_RELAXED_ORDERING
unset NCCL_IB_AR_THRESHOLD
unset NCCL_NTHREADS
unset NCCL_GRAPH_MIXING_SUPPORT

if [[ ! -d "${MODEL_PATH}" ]]; then
  MODEL_PATH=$(
    python - "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_LOCAL_FILES_ONLY}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision, local_only = sys.argv[1:4]
print(snapshot_download(model_id, revision=revision, local_files_only=local_only == "1"))
PY
  )
fi

if [[ -z "${SPECULATOR_MODEL_PATH}" || ! -d "${SPECULATOR_MODEL_PATH}" ]]; then
  SPECULATOR_MODEL_PATH=$(
    python - "${SPECULATOR_MODEL_ID}" "${SPECULATOR_LOCAL_FILES_ONLY}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, local_only = sys.argv[1:3]
print(snapshot_download(model_id, local_files_only=local_only == "1"))
PY
  )
fi

CUDA_GRAPH_ARGS=()
if [[ "${BENCH_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  CUDA_GRAPH_ARGS=(--disable-cuda-graph)
else
  CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "${BENCH_CUDA_GRAPH_MAX_BS}")
fi

CUSTOM_ALL_REDUCE_ARGS=()
if [[ "${BENCH_DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  CUSTOM_ALL_REDUCE_ARGS=(--disable-custom-all-reduce)
fi

DTYPE_ARGS=()
if [[ -n "${DTYPE}" ]]; then
  DTYPE_ARGS=(--dtype "${DTYPE}")
fi

MAX_TOTAL_TOKENS_ARGS=()
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  MAX_TOTAL_TOKENS_ARGS=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

SERVER_WARMUP_ARGS=()
if [[ "${BENCH_SKIP_SERVER_WARMUP}" == "1" ]]; then
  SERVER_WARMUP_ARGS=(--skip-server-warmup)
fi

DP_ARGS=()
if (( DP_SIZE > 1 )); then
  DP_ARGS=(--dp "${DP_SIZE}")
fi
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
  DP_ARGS+=(--enable-dp-attention)
fi

INDEXER_ARGS=(
  --nsa-indexer-mode "${NSA_INDEXER_MODE}"
  --nsa-indexcache-pattern "${INDEXCACHE_PATTERN}"
)

TURBOQUANT_ARGS=(
  --enable-turboquant-dense-kv-cache
  --turboquant-dense-kv-preset "${TURBOQUANT_DENSE_KV_PRESET}"
  --turboquant-residual-window-size "${TURBOQUANT_RESIDUAL_WINDOW_SIZE}"
  --turboquant-execution-mode "${TURBOQUANT_EXECUTION_MODE}"
)

SMC_ARGS=(
  --speculative-algorithm SMC
  --speculative-draft-model-path "${SPECULATOR_MODEL_PATH}"
  --speculative-draft-model-quantization "${SPECULATIVE_DRAFT_MODEL_QUANTIZATION}"
  --fp8-gemm-backend "${FP8_GEMM_BACKEND}"
  --page-size "${PAGE_SIZE:-64}"
  --smc-n-particles "${SMC_N_PARTICLES}"
  --smc-gamma "${SMC_GAMMA}"
  --smc-draft-temperature "${SMC_DRAFT_TEMPERATURE}"
  --smc-target-temperature "${SMC_TARGET_TEMPERATURE}"
  --smc-resample-threshold "${SMC_RESAMPLE_THRESHOLD}"
  --smc-resample-method "${SMC_RESAMPLE_METHOD}"
)
if [[ -n "${TARGET_ATTENTION_BACKEND}" ]]; then
  SMC_ARGS+=(--attention-backend "${TARGET_ATTENTION_BACKEND}")
fi
if [[ -n "${DRAFT_ATTENTION_BACKEND}" ]]; then
  SMC_ARGS+=(--speculative-draft-attention-backend "${DRAFT_ATTENTION_BACKEND}")
fi
if [[ -n "${SPECULATIVE_MOE_RUNNER_BACKEND}" ]]; then
  SMC_ARGS+=(--speculative-moe-runner-backend "${SPECULATIVE_MOE_RUNNER_BACKEND}")
fi

selected() {
  local name=$1
  case "${RUN_SET}" in
    all)
      return 0
      ;;
    pair)
      [[ "${name}" != "none" ]]
      ;;
    baseline)
      [[ "${name}" == "baseline_combo" ]]
      ;;
    candidate)
      [[ "${name}" == "candidate_combo" ]]
      ;;
    *)
      [[ ",${RUN_SET}," == *,"${name}",* ]]
      ;;
  esac
}

repo_supports_server_arg() {
  local repo=$1
  local pattern=$2
  grep -q -- "${pattern}" "${repo}/python/sglang/srt/server_args.py" 2>/dev/null
}

stop_server() {
  local pid_file=$1
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid=$(cat "${pid_file}")
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      sleep 10
      kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  pkill -f "sglang::scheduler" 2>/dev/null || true
}

wait_for_health() {
  local log_file=$1
  local deadline=$((SECONDS + 3600))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if grep -Eiq "Traceback|RuntimeError|OutOfMemory|CUDA out of memory|NCCL error|Subprocess .* crashed" "${log_file}" 2>/dev/null; then
      tail -n 200 "${log_file}"
      return 1
    fi
    sleep 15
  done
  tail -n 200 "${log_file}"
  return 1
}

collect_state() {
  local run_dir=$1
  local label=$2
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "${run_dir}/${label}.nvidia-smi.csv" 2>&1 || true
  curl -fsS "http://${HOST}:${PORT}/server_info" > "${run_dir}/${label}.server_info.json" 2>/dev/null || true
}

run_bench() {
  local run_dir=$1
  local tag=$2
  local input_len=$3
  local output_len=$4
  local output_detail_args=()
  local flush_cache_args=()
  if [[ "${BENCH_OUTPUT_DETAILS}" == "1" ]]; then
    output_detail_args+=(--output-details)
  fi
  if [[ "${BENCH_FLUSH_CACHE_AFTER_WARMUP}" == "1" ]]; then
    flush_cache_args+=(--flush-cache)
  fi

  local bench_status=0
  set +e
  timeout "${BENCH_TIMEOUT}" python -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "http://${HOST}:${PORT}" \
    --model "${MODEL_ID}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name "${BENCH_DATASET_NAME}" \
    --num-prompts "${DECODE_NUM_PROMPTS}" \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --random-range-ratio "${BENCH_RANDOM_RANGE_RATIO}" \
    --max-concurrency "${DECODE_CONCURRENCY}" \
    --request-rate inf \
    --seed "${BENCH_SEED}" \
    --warmup-requests "${BENCH_WARMUP_REQUESTS}" \
    "${output_detail_args[@]}" \
    "${flush_cache_args[@]}" \
    --disable-tqdm \
    --output-file "${run_dir}/bench.jsonl" \
    --tag "${tag}" \
    --extra-request-body '{"temperature":0.0,"ignore_eos":true}' \
    > "${run_dir}/${tag}.bench.log" 2>&1
  bench_status=$?
  set -e
  echo "${bench_status}" > "${run_dir}/${tag}.bench.status"
  collect_state "${run_dir}" "after-${tag}"
  if (( bench_status != 0 )); then
    return "${bench_status}"
  fi
}

run_profile() {
  local run_dir=$1
  case "${BENCH_PROFILE}" in
    gate8k1k)
      run_bench "${run_dir}" decode_8k_out1k 8192 1024
      ;;
    gate16k1k)
      run_bench "${run_dir}" decode_16k_out1k 16384 1024
      ;;
    gate16k4k)
      run_bench "${run_dir}" decode_16k_out4k 16384 4096
      ;;
    gate32k4k)
      run_bench "${run_dir}" decode_32k_out4k 32768 4096
      ;;
    gate64k4k)
      run_bench "${run_dir}" decode_64k_out4k 65536 4096
      ;;
    smcsd_combo_matrix)
      run_bench "${run_dir}" decode_8k_out1k 8192 1024
      run_bench "${run_dir}" decode_16k_out1k 16384 1024
      run_bench "${run_dir}" decode_16k_out4k 16384 4096
      run_bench "${run_dir}" decode_32k_out4k 32768 4096
      run_bench "${run_dir}" decode_64k_out4k 65536 4096
      ;;
    *)
      echo "Unsupported BENCH_PROFILE=${BENCH_PROFILE}" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local name=$1
  local repo=$2
  selected "${name}" || return 0

  local run_dir="${RUN_ROOT}/${name}"
  local log_file="${run_dir}/server.log"
  local pid_file="${run_dir}/server.pid"
  local smc_probe_record_path="${SGLANG_SMC_PROBE_RECORD_PATH}"
  local smc_diag_path="${SGLANG_SMC_DIAG_PATH}"
  local turboquant_args=("${TURBOQUANT_ARGS[@]}")
  local smc_args=("${SMC_ARGS[@]}")
  local turboquant_mla_decode_num_splits_effective="repo-default"
  local smc_draft_kv_cache_dtype_effective="repo-default"
  mkdir -p "${run_dir}"

  if repo_supports_server_arg "${repo}" "turboquant_mla_decode_num_splits"; then
    turboquant_args+=(--turboquant-mla-decode-num-splits "${TURBOQUANT_MLA_DECODE_NUM_SPLITS}")
    turboquant_mla_decode_num_splits_effective="${TURBOQUANT_MLA_DECODE_NUM_SPLITS}"
  fi
  if [[ -n "${SMC_DRAFT_KV_CACHE_DTYPE}" ]]; then
    if repo_supports_server_arg "${repo}" "smc_draft_kv_cache_dtype"; then
      smc_args+=(--smc-draft-kv-cache-dtype "${SMC_DRAFT_KV_CACHE_DTYPE}")
      smc_draft_kv_cache_dtype_effective="${SMC_DRAFT_KV_CACHE_DTYPE}"
    else
      smc_draft_kv_cache_dtype_effective="unsupported"
    fi
  fi

  if [[ "${name}" == "candidate_combo" && "${SMC_PROBE}" == "1" && -z "${smc_probe_record_path}" ]]; then
    smc_probe_record_path="${run_dir}/smc_probe.jsonl"
  fi
  if [[ "${name}" == "candidate_combo" && -n "${smc_diag_path}" && "${smc_diag_path}" != /* ]]; then
    smc_diag_path="${run_dir}/${smc_diag_path}"
  fi

  {
    echo "run=${name}"
    echo "repo=${repo}"
    echo "model=${MODEL_ID}"
    echo "revision=${MODEL_REVISION}"
    echo "model_path=${MODEL_PATH}"
    echo "speculator=${SPECULATOR_MODEL_PATH}"
    echo "bench_profile=${BENCH_PROFILE}"
    echo "prompts=${DECODE_NUM_PROMPTS}"
    echo "concurrency=${DECODE_CONCURRENCY}"
    echo "max_total_tokens=${MAX_TOTAL_TOKENS}"
    echo "tp=${TP_SIZE}"
    echo "dp=${DP_SIZE}"
    echo "enable_dp_attention=${ENABLE_DP_ATTENTION}"
    echo "target_attention_backend=${TARGET_ATTENTION_BACKEND:-auto}"
    echo "draft_attention_backend=${DRAFT_ATTENTION_BACKEND:-auto}"
    echo "dtype=${DTYPE:-auto}"
    echo "kv_cache_dtype=${KV_CACHE_DTYPE}"
    echo "nsa_indexer_mode=${NSA_INDEXER_MODE}"
    echo "nsa_prefill_backend=${NSA_PREFILL_BACKEND}"
    echo "nsa_decode_backend=${NSA_DECODE_BACKEND}"
    echo "enable_turboquant_dense_kv_cache=${ENABLE_TURBOQUANT_DENSE_KV_CACHE}"
    echo "turboquant_dense_kv_preset=${TURBOQUANT_DENSE_KV_PRESET}"
    echo "turboquant_residual_window_size=${TURBOQUANT_RESIDUAL_WINDOW_SIZE}"
    echo "turboquant_execution_mode=${TURBOQUANT_EXECUTION_MODE}"
    echo "turboquant_mla_decode_num_splits_requested=${TURBOQUANT_MLA_DECODE_NUM_SPLITS}"
    echo "turboquant_mla_decode_num_splits_effective=${turboquant_mla_decode_num_splits_effective}"
    echo "smc_draft_kv_cache_dtype=${smc_draft_kv_cache_dtype_effective}"
    echo "fp8_gemm_backend=${FP8_GEMM_BACKEND}"
    echo "sglang_smc_draft_fp8_gemm_backend=${SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND}"
    echo "smc_gamma=${SMC_GAMMA}"
    echo "smc_probe=${SMC_PROBE}"
    echo "sglang_smc_probe_record_path=${smc_probe_record_path}"
    echo "sglang_smc_diag_path=${smc_diag_path}"
    echo "sglang_smc_prefill_stream_yield_ms=${SGLANG_SMC_PREFILL_STREAM_YIELD_MS}"
    echo "sglang_smc_target_verify_graph=${SGLANG_SMC_TARGET_VERIFY_GRAPH}"
    echo "bench_flush_cache_after_warmup=${BENCH_FLUSH_CACHE_AFTER_WARMUP}"
    echo "bench_disable_custom_all_reduce=${BENCH_DISABLE_CUSTOM_ALL_REDUCE}"
    echo "nccl_ib_disable=${NCCL_IB_DISABLE}"
    echo "nccl_nvls_enable=${NCCL_NVLS_ENABLE}"
    echo "nccl_p2p_level=${NCCL_P2P_LEVEL}"
    echo "nccl_p2p_disable=${NCCL_P2P_DISABLE}"
    echo "nccl_cumem_enable=${NCCL_CUMEM_ENABLE}"
    echo "nccl_algo=${NCCL_ALGO}"
  } | tee "${run_dir}/run.env"

  stop_server "${pid_file}"
  setsid env \
    -u NCCL_NET \
    -u NCCL_NET_PLUGIN \
    -u NCCL_PROTO \
    -u NCCL_IB_HCA \
    -u NCCL_IB_GID_INDEX \
    -u NCCL_NET_GDR_LEVEL \
    -u NCCL_NET_GDR_READ \
    -u NCCL_MIN_NCHANNELS \
    -u NCCL_MAX_NCHANNELS \
    -u NCCL_CROSS_NIC \
    -u NCCL_IB_QPS_PER_CONNECTION \
    -u NCCL_IB_SPLIT_DATA_ON_QPS \
    -u NCCL_BUFFSIZE \
    -u NCCL_IB_TIMEOUT \
    -u NCCL_IB_RETRY_CNT \
    -u NCCL_IB_PCI_RELAXED_ORDERING \
    -u NCCL_IB_AR_THRESHOLD \
    -u NCCL_NTHREADS \
    -u NCCL_GRAPH_MIXING_SUPPORT \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${repo}/python:${PYTHONPATH:-}" \
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    CUDA_HOME="${CUDA_HOME}" \
    CUDA_PATH="${CUDA_PATH}" \
    PATH="${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2}" \
    SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM}" \
    SGLANG_DISABLE_FLASHINFER_FUSED_TOPK_DEEPSEEK="${SGLANG_DISABLE_FLASHINFER_FUSED_TOPK_DEEPSEEK}" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN}" \
    SGLANG_SMC_PROBE_RECORD_PATH="${smc_probe_record_path}" \
    SGLANG_SMC_DIAG_PATH="${smc_diag_path}" \
    SGLANG_SMC_PREFILL_STREAM_YIELD_MS="${SGLANG_SMC_PREFILL_STREAM_YIELD_MS}" \
    SGLANG_SMC_TARGET_VERIFY_GRAPH="${SGLANG_SMC_TARGET_VERIFY_GRAPH}" \
    SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND="${SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND}" \
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
    NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE}" \
    NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}" \
    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
    NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE}" \
    NCCL_ALGO="${NCCL_ALGO}" \
    python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --revision "${MODEL_REVISION}" \
      --served-model-name "${MODEL_ID}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tp "${TP_SIZE}" \
      "${DP_ARGS[@]}" \
      --trust-remote-code \
      "${DTYPE_ARGS[@]}" \
      --kv-cache-dtype "${KV_CACHE_DTYPE}" \
      --nsa-prefill-backend "${NSA_PREFILL_BACKEND}" \
      --nsa-decode-backend "${NSA_DECODE_BACKEND}" \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --disable-flashinfer-autotune \
      --watchdog-timeout 1800 \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --context-length "${CONTEXT_LENGTH}" \
      "${MAX_TOTAL_TOKENS_ARGS[@]}" \
      --reasoning-parser deepseek-v3 \
      "${SERVER_WARMUP_ARGS[@]}" \
      "${CUDA_GRAPH_ARGS[@]}" \
      "${CUSTOM_ALL_REDUCE_ARGS[@]}" \
      "${INDEXER_ARGS[@]}" \
      "${turboquant_args[@]}" \
      "${smc_args[@]}" \
      > "${log_file}" 2>&1 &
  echo $! > "${pid_file}"

  local run_status=0
  wait_for_health "${log_file}" || run_status=$?
  if (( run_status == 0 )); then
    collect_state "${run_dir}" ready
    run_profile "${run_dir}" || run_status=$?
    collect_state "${run_dir}" done
  fi
  stop_server "${pid_file}"
  return "${run_status}"
}

echo "run_root=${RUN_ROOT}"
echo "baseline_repo=${BASELINE_REPO}"
echo "candidate_repo=${REPO}"
echo "model_path=${MODEL_PATH}"
echo "speculator_model_path=${SPECULATOR_MODEL_PATH}"

overall_status=0
run_checked() {
  local status=0
  run_one "$@" || status=$?
  if (( status != 0 && overall_status == 0 )); then
    overall_status=${status}
  fi
}

case "${RUN_SET}" in
  pair | all)
    run_checked baseline_combo "${BASELINE_REPO}"
    run_checked candidate_combo "${REPO}"
    ;;
  baseline | baseline_combo)
    run_checked baseline_combo "${BASELINE_REPO}"
    ;;
  candidate | candidate_combo)
    run_checked candidate_combo "${REPO}"
    ;;
  *)
    echo "RUN_SET must be one of: pair, all, baseline, baseline_combo, candidate, candidate_combo." >&2
    exit 2
    ;;
esac

export BENCH_RUN_STATUS="${overall_status}"
summary_status=0
set +e
python - "${RUN_ROOT}" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = {
    "run_root": str(root),
    "runs": {},
    "bench_run_status": int(os.environ.get("BENCH_RUN_STATUS", "0")),
}
for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    run = {"benchmarks": [], "bench_statuses": {}}
    ready = run_dir / "ready.server_info.json"
    if ready.exists():
        info = json.loads(ready.read_text())
        run["ready"] = {
            "max_total_num_tokens": info.get("max_total_num_tokens"),
            "kv_cache_dtype": info.get("kv_cache_dtype"),
            "speculative_algorithm": info.get("speculative_algorithm"),
            "nsa_indexer_mode": info.get("nsa_indexer_mode"),
            "enable_turboquant_dense_kv_cache": info.get(
                "enable_turboquant_dense_kv_cache"
            ),
        }
    bench_file = run_dir / "bench.jsonl"
    if bench_file.exists():
        for line in bench_file.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            output_lens = item.get("output_lens")
            if not isinstance(output_lens, list):
                output_lens = None
            errors = item.get("errors")
            error_count = None
            if isinstance(errors, list):
                error_count = sum(1 for error in errors if error)
            generated_texts = item.get("generated_texts")
            generated_text_char_lens = None
            if isinstance(generated_texts, list):
                generated_text_char_lens = [
                    len(text) if isinstance(text, str) else 0
                    for text in generated_texts
                ]
            itls = item.get("itls")
            itl_counts = None
            if isinstance(itls, list):
                itl_counts = [
                    len(samples) if isinstance(samples, list) else 0
                    for samples in itls
                ]
            run["benchmarks"].append({
                "tag": item.get("tag"),
                "completed": item.get("completed"),
                "input_throughput": item.get("input_throughput"),
                "output_throughput": item.get("output_throughput"),
                "total_throughput": item.get("total_throughput"),
                "mean_ttft_ms": item.get("mean_ttft_ms"),
                "median_ttft_ms": item.get("median_ttft_ms"),
                "p99_ttft_ms": item.get("p99_ttft_ms"),
                "mean_tpot_ms": item.get("mean_tpot_ms"),
                "median_tpot_ms": item.get("median_tpot_ms"),
                "p99_tpot_ms": item.get("p99_tpot_ms"),
                "duration": item.get("duration"),
                "accept_length": item.get("accept_length"),
                "successful_requests": item.get("successful_requests"),
                "total_input_tokens": item.get("total_input_tokens"),
                "total_output_tokens": item.get("total_output_tokens"),
                "total_output_tokens_retokenized": item.get(
                    "total_output_tokens_retokenized"
                ),
                "output_lens": output_lens,
                "error_count": error_count,
                "generated_text_char_lens": generated_text_char_lens,
                "itl_counts": itl_counts,
            })
    for status_file in sorted(run_dir.glob("*.bench.status")):
        tag = status_file.name.removesuffix(".bench.status")
        try:
            run["bench_statuses"][tag] = int(status_file.read_text().strip())
        except ValueError:
            run["bench_statuses"][tag] = status_file.read_text().strip()
    run["bench_failed_tags"] = [
        tag for tag, status in run["bench_statuses"].items() if status != 0
    ]
    summary["runs"][run_dir.name] = run

profile = os.environ.get("BENCH_PROFILE")
if profile == "gate8k1k":
    gate_tags = ["decode_8k_out1k"]
elif profile == "gate16k1k":
    gate_tags = ["decode_16k_out1k"]
elif profile == "gate16k4k":
    gate_tags = ["decode_16k_out4k"]
elif profile == "gate32k4k":
    gate_tags = ["decode_32k_out4k"]
elif profile == "gate64k4k":
    gate_tags = ["decode_64k_out4k"]
else:
    gate_tags = [
        "decode_8k_out1k",
        "decode_16k_out1k",
        "decode_16k_out4k",
        "decode_32k_out4k",
        "decode_64k_out4k",
    ]

expected_output_per_request = {
    "decode_8k_out1k": 1024,
    "decode_16k_out1k": 1024,
    "decode_16k_out4k": 4096,
    "decode_32k_out4k": 4096,
    "decode_64k_out4k": 4096,
}

num_prompts = int(os.environ.get("DECODE_NUM_PROMPTS", "8"))
min_retok_ratio = float(os.environ.get("GATE_MIN_RETOKENIZED_OUTPUT_RATIO", "0.95"))
min_fallback_retok_ratio = float(
    os.environ.get("GATE_MIN_FALLBACK_RETOKENIZED_OUTPUT_RATIO", "0.75")
)
min_server_ratio = float(os.environ.get("GATE_MIN_SERVER_OUTPUT_RATIO", "0.95"))
min_generated_chars_per_token = float(
    os.environ.get("GATE_MIN_GENERATED_CHARS_PER_TOKEN", "0.25")
)
min_itl_samples_per_request = int(
    os.environ.get("GATE_MIN_ITL_SAMPLES_PER_REQUEST", "1")
)
output_checks = {}
for run_name, run in summary["runs"].items():
    run_checks = {}
    for item in run.get("benchmarks", []):
        tag = item["tag"]
        expected_per_request = expected_output_per_request.get(tag, 0)
        expected = expected_per_request * num_prompts
        if not expected:
            expected = item.get("total_output_tokens")
        retok = item.get("total_output_tokens_retokenized")
        retok_ratio = None
        if expected and retok is not None:
            retok_ratio = retok / expected
        output_lens = item.get("output_lens")
        server_tokens = None
        min_output_len = None
        full_output_lens = False
        if isinstance(output_lens, list):
            server_tokens = sum(
                value for value in output_lens if isinstance(value, (int, float))
            )
            numeric_lens = [
                value for value in output_lens if isinstance(value, (int, float))
            ]
            if numeric_lens:
                min_output_len = min(numeric_lens)
            full_output_lens = (
                len(numeric_lens) == num_prompts
                and expected_per_request > 0
                and min_output_len is not None
                and min_output_len >= expected_per_request * min_server_ratio
            )
        if server_tokens is None:
            server_tokens = item.get("total_output_tokens")
        server_ratio = None
        if expected and server_tokens is not None:
            server_ratio = server_tokens / expected
        generated_text_char_lens = item.get("generated_text_char_lens")
        generated_chars = None
        generated_chars_per_token = None
        if isinstance(generated_text_char_lens, list):
            generated_chars = sum(
                value
                for value in generated_text_char_lens
                if isinstance(value, (int, float))
            )
            if expected:
                generated_chars_per_token = generated_chars / expected
        itl_counts = item.get("itl_counts")
        min_itl_count = None
        if isinstance(itl_counts, list) and itl_counts:
            numeric_itl_counts = [
                value for value in itl_counts if isinstance(value, (int, float))
            ]
            if numeric_itl_counts:
                min_itl_count = min(numeric_itl_counts)
        completed_ok = item.get("completed") == num_prompts
        errors_ok = item.get("error_count") in (None, 0)
        primary_valid = (
            completed_ok
            and errors_ok
            and retok_ratio is not None
            and retok_ratio >= min_retok_ratio
        )
        fallback_valid = (
            completed_ok
            and errors_ok
            and server_ratio is not None
            and server_ratio >= min_server_ratio
            and full_output_lens
            and retok_ratio is not None
            and retok_ratio >= min_fallback_retok_ratio
            and generated_chars_per_token is not None
            and generated_chars_per_token >= min_generated_chars_per_token
            and min_itl_count is not None
            and min_itl_count >= min_itl_samples_per_request
        )
        run_checks[tag] = {
            "valid": primary_valid or fallback_valid,
            "validation_method": (
                "retokenized"
                if primary_valid
                else "server_output_with_decode_evidence"
                if fallback_valid
                else "failed"
            ),
            "total_output_tokens": expected,
            "total_output_tokens_retokenized": retok,
            "retokenized_output_ratio": retok_ratio,
            "min_retokenized_output_ratio": min_retok_ratio,
            "min_fallback_retokenized_output_ratio": min_fallback_retok_ratio,
            "server_output_tokens": server_tokens,
            "server_output_ratio": server_ratio,
            "min_server_output_ratio": min_server_ratio,
            "min_output_len": min_output_len,
            "full_output_lens": full_output_lens,
            "completed": item.get("completed"),
            "expected_completed": num_prompts,
            "error_count": item.get("error_count"),
            "generated_chars": generated_chars,
            "generated_chars_per_token": generated_chars_per_token,
            "min_generated_chars_per_token": min_generated_chars_per_token,
            "min_itl_count": min_itl_count,
            "min_itl_samples_per_request": min_itl_samples_per_request,
        }
    output_checks[run_name] = run_checks
summary["output_validation"] = output_checks

def by_tag(run_name):
    return {
        item["tag"]: item
        for item in summary["runs"].get(run_name, {}).get("benchmarks", [])
    }

baseline = by_tag("baseline_combo")
candidate = by_tag("candidate_combo")
if baseline and candidate:
    comparisons = {}
    min_delta = float(os.environ.get("GATE_MIN_CANDIDATE_DELTA_PCT", "0.0"))
    max_ttft_regression = float(os.environ.get("GATE_MAX_TTFT_REGRESSION_PCT", "1.0"))
    ttft_metric = os.environ.get("GATE_TTFT_METRIC", "mean_ttft_ms")
    missing_tags = []
    for tag in gate_tags:
        if tag not in baseline or tag not in candidate:
            missing_tags.append(tag)
            continue
        base_metric = baseline[tag]["total_throughput"]
        cand_metric = candidate[tag]["total_throughput"]
        base_ttft = baseline[tag][ttft_metric]
        cand_ttft = candidate[tag][ttft_metric]
        delta_pct = (cand_metric - base_metric) / base_metric * 100.0
        ttft_delta_pct = (cand_ttft - base_ttft) / base_ttft * 100.0
        comparisons[tag] = {
            "candidate_over_baseline": delta_pct >= min_delta
            and ttft_delta_pct <= max_ttft_regression,
            "candidate_throughput_over_baseline": delta_pct >= min_delta,
            "ttft_gate_pass": ttft_delta_pct <= max_ttft_regression,
            "metric": "total_throughput",
            "baseline_metric": base_metric,
            "candidate_metric": cand_metric,
            "candidate_over_baseline_pct": delta_pct,
            "min_candidate_delta_pct": min_delta,
            "ttft_metric": ttft_metric,
            "baseline_ttft_ms": base_ttft,
            "candidate_ttft_ms": cand_ttft,
            "candidate_ttft_delta_pct": ttft_delta_pct,
            "max_ttft_regression_pct": max_ttft_regression,
        }
    if missing_tags:
        summary["gates"] = {
            "skipped": True,
            "reason": "missing benchmark tags",
            "missing_tags": missing_tags,
            "gate_tags": gate_tags,
            "comparisons": comparisons,
        }
    else:
        summary["gates"] = {
            "candidate_over_baseline": all(
                item["candidate_over_baseline"] for item in comparisons.values()
            ),
            "gate_tags": gate_tags,
            "comparisons": comparisons,
        }
else:
    summary["gates"] = {
        "skipped": True,
        "reason": "missing baseline_combo or candidate_combo run",
        "gate_tags": gate_tags,
    }

(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
invalid_outputs = [
    f"{run_name}:{tag}"
    for run_name, checks in output_checks.items()
    for tag, check in checks.items()
    if not check["valid"]
]
if invalid_outputs:
    raise SystemExit(
        "Output validation failed for "
        + ", ".join(invalid_outputs)
        + "; neither strict retokenized output nor server-count fallback decode evidence met the requested output-token guard."
    )
if not summary["gates"].get("skipped") and not summary["gates"]["candidate_over_baseline"]:
    raise SystemExit("Candidate combo did not improve over baseline combo within TTFT gate.")
PY
summary_status=$?
set -e

echo "run_root=${RUN_ROOT}"
if (( overall_status != 0 )); then
  exit "${overall_status}"
fi
exit "${summary_status}"
