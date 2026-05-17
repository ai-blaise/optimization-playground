#!/usr/bin/env bash
set -euo pipefail

MODELS=${MODELS:-"Qwen/Qwen3-1.7B Qwen/Qwen3-8B google/gemma-3-1b-it"}
CONCURRENCIES=${CONCURRENCIES:-"1 2 4 8 16 32 64 128 256"}
OUT_DIR=${OUT_DIR:-/tmp/sglang_flashsampling_paper_ab_$(date +%s)}
DATASET_REPO=${DATASET_REPO:-AI-MO/aimo-validation-aime}
DATASET_SPLIT=${DATASET_SPLIT:-train}
DATASET_PATH=${DATASET_PATH:-${OUT_DIR}/aimo-validation-aime.jsonl}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
PYTHON_BIN=${PYTHON_BIN:-python3}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-256}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-1200}
NUM_RUNS=${NUM_RUNS:-3}
SEED=${SEED:-1}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-1}
DRY_RUN=${DRY_RUN:-0}
AUTO_INSTALL_DATASETS=${AUTO_INSTALL_DATASETS:-0}
ENABLE_XET=${ENABLE_XET:-1}
PRE_DOWNLOAD=${PRE_DOWNLOAD:-0}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}
DISABLE_PIECEWISE_CUDA_GRAPH=${DISABLE_PIECEWISE_CUDA_GRAPH:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
SAMPLING_BACKEND=${SAMPLING_BACKEND:-}
RESTART_SERVER_PER_CASE=${RESTART_SERVER_PER_CASE:-0}
FLASHSAMPLING_WARMUP_BATCH_SIZES=${FLASHSAMPLING_WARMUP_BATCH_SIZES:-}
FLASHSAMPLING_PROVIDER=${FLASHSAMPLING_PROVIDER:-}
FLASHSAMPLING_MIN_BATCH_SIZE=${FLASHSAMPLING_MIN_BATCH_SIZE:-}
FLASHSAMPLING_MAX_BATCH_SIZE=${FLASHSAMPLING_MAX_BATCH_SIZE:-}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0.6,"top_k":-1,"top_p":1.0}'}

mkdir -p "${OUT_DIR}"

if [[ "${ENABLE_XET}" == "1" ]]; then
  export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
fi

prompt_count_for_concurrency() {
  case "$1" in
    1) echo 10 ;;
    2) echo 20 ;;
    4) echo 40 ;;
    8) echo 80 ;;
    16) echo 160 ;;
    32) echo 256 ;;
    64) echo 384 ;;
    128) echo 640 ;;
    256) echo 1024 ;;
    *) echo $((10 * $1)) ;;
  esac
}

max_prompts() {
  local max=0
  for concurrency in ${CONCURRENCIES}; do
    local prompts
    prompts=$(prompt_count_for_concurrency "${concurrency}")
    if (( prompts > max )); then
      max=${prompts}
    fi
  done
  echo "${max}"
}

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

prepare_dataset() {
  if [[ -s "${DATASET_PATH}" ]]; then
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN would materialize ${DATASET_REPO}:${DATASET_SPLIT} at ${DATASET_PATH}"
    return
  fi

  if [[ "${AUTO_INSTALL_DATASETS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install -q datasets
  fi

  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import datasets  # noqa: F401
PY
  then
    echo "The Python package 'datasets' is required to materialize ${DATASET_REPO}." >&2
    echo "Rerun with AUTO_INSTALL_DATASETS=1 or install it in the active environment." >&2
    exit 1
  fi

  local count
  count=$(max_prompts)
  "${PYTHON_BIN}" - "${DATASET_REPO}" "${DATASET_SPLIT}" "${DATASET_PATH}" "${count}" <<'PY'
import itertools
import json
import sys

from datasets import load_dataset


repo, split, out_path, count_s = sys.argv[1:]
count = int(count_s)
try:
    rows = list(load_dataset(repo, split=split))
except Exception:
    dataset = load_dataset(repo)
    first_split = next(iter(dataset.keys()))
    rows = list(dataset[first_split])

if not rows:
    raise RuntimeError(f"{repo} produced no rows")

def prompt_from_row(row):
    for key in ("problem", "question", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError(f"cannot identify prompt field in row keys={list(row.keys())}")

with open(out_path, "w", encoding="utf-8") as f:
    for row in itertools.islice(itertools.cycle(rows), count):
        prompt = prompt_from_row(row)
        f.write(
            json.dumps(
                {
                    "conversations": [
                        {"content": prompt},
                        {"content": "placeholder"},
                    ]
                },
                ensure_ascii=False,
            )
            + "\n"
        )

print(f"wrote {count} prompts to {out_path}")
PY
}

server_args() {
  local model=$1
  local variant=$2
  local args=(
    -m sglang.launch_server
    --model-path "${model}"
    --context-length "${MAX_MODEL_LEN}"
    --disable-radix-cache
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --host "${HOST}"
    --port "${PORT}"
  )
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    args+=(--trust-remote-code)
  fi
  if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    args+=(--disable-cuda-graph)
  fi
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
    args+=(--disable-piecewise-cuda-graph)
  fi
  if [[ -n "${SAMPLING_BACKEND}" ]]; then
    args+=(--sampling-backend "${SAMPLING_BACKEND}")
  fi
  if [[ "${variant}" == "flashsampling" ]]; then
    args+=(--enable-flashsampling --flashsampling-fallback error)
    if [[ -n "${FLASHSAMPLING_PROVIDER}" ]]; then
      args+=(--flashsampling-provider "${FLASHSAMPLING_PROVIDER}")
    fi
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

bench_args() {
  local model=$1
  local concurrency=$2
  local prompts=$3
  local output_file=$4
  local warmup_requests=${WARMUP_REQUESTS}
  if [[ "${warmup_requests}" == "concurrency" ]]; then
    warmup_requests=${concurrency}
  fi
  printf '%q ' "${PYTHON_BIN}" -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${model}" \
    --dataset-name custom \
    --dataset-path "${DATASET_PATH}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --sharegpt-context-len "${MAX_MODEL_LEN}" \
    --num-prompts "${prompts}" \
    --request-rate "${concurrency}" \
    --max-concurrency "${concurrency}" \
    --seed "${SEED}" \
    --flush-cache \
    --warmup-requests "${warmup_requests}" \
    --disable-tqdm \
    --extra-request-body "${EXTRA_REQUEST_BODY}" \
    --output-file "${output_file}"
}

wait_for_server() {
  local pid=$1
  for _ in $(seq 1 "${SERVER_READY_TIMEOUT_SECONDS}"); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

run_variant() {
  local model=$1
  local variant=$2
  local model_slug=${model//\//__}
  local variant_dir="${OUT_DIR}/${model_slug}/${variant}"
  mkdir -p "${variant_dir}"
  PORT=$(free_port)

  local serve_cmd
  serve_cmd=$(server_args "${model}" "${variant}")
  echo "SERVE ${model} ${variant}: ${serve_cmd}" | tee "${variant_dir}/serve.cmd"
  if [[ "${DRY_RUN}" == "1" ]]; then
    for run in $(seq 1 "${NUM_RUNS}"); do
      for concurrency in ${CONCURRENCIES}; do
        local prompts
        prompts=$(prompt_count_for_concurrency "${concurrency}")
        local output_file="${variant_dir}/run${run}_c${concurrency}.jsonl"
        local bench_cmd
        bench_cmd=$(bench_args "${model}" "${concurrency}" "${prompts}" "${output_file}")
        echo "BENCH ${model} ${variant} run=${run} c=${concurrency}: ${bench_cmd}" \
          | tee "${variant_dir}/run${run}_c${concurrency}.cmd"
      done
    done
    return
  fi

  local server_pid=""
  start_server() {
    local server_log=$1
    bash -lc "${serve_cmd}" >"${server_log}" 2>&1 &
    server_pid=$!
    if ! wait_for_server "${server_pid}"; then
      echo "server failed for ${model} ${variant}; log=${server_log}"
      tail -200 "${server_log}" || true
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" >/dev/null 2>&1 || true
      server_pid=""
      return 1
    fi
  }

  stop_server() {
    if [[ -n "${server_pid}" ]]; then
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" >/dev/null 2>&1 || true
      server_pid=""
    fi
  }

  run_case() {
    local run=$1
    local concurrency=$2
    local prompts
    prompts=$(prompt_count_for_concurrency "${concurrency}")
    local output_file="${variant_dir}/run${run}_c${concurrency}.jsonl"
    local bench_cmd
    bench_cmd=$(bench_args "${model}" "${concurrency}" "${prompts}" "${output_file}")
    echo "BENCH ${model} ${variant} run=${run} c=${concurrency}: ${bench_cmd}" \
      | tee "${variant_dir}/run${run}_c${concurrency}.cmd"
    local status=0
    bash -lc "${bench_cmd}" | tee "${variant_dir}/run${run}_c${concurrency}.txt" || status=$?
    return "${status}"
  }

  if [[ "${RESTART_SERVER_PER_CASE}" != "1" ]]; then
    start_server "${variant_dir}/server.log" || return 1
  fi

  local status=0
  for run in $(seq 1 "${NUM_RUNS}"); do
    for concurrency in ${CONCURRENCIES}; do
      if [[ "${RESTART_SERVER_PER_CASE}" == "1" ]]; then
        start_server "${variant_dir}/server_run${run}_c${concurrency}.log" || return 1
      elif ! kill -0 "${server_pid}" >/dev/null 2>&1; then
        echo "server exited before ${model} ${variant} run=${run} c=${concurrency}" >&2
        status=1
        break 2
      fi
      run_case "${run}" "${concurrency}" || status=$?
      if [[ "${RESTART_SERVER_PER_CASE}" == "1" ]]; then
        stop_server
      fi
      if (( status != 0 )); then
        break 2
      fi
    done
  done

  stop_server
  if (( status != 0 )); then
    return "${status}"
  fi
}

summarize() {
  "${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json
import pathlib
import statistics
import sys


root = pathlib.Path(sys.argv[1])
rows = {}
for path in root.glob("*/*/run*_c*.jsonl"):
    model = path.parents[1].name
    variant = path.parent.name
    run = int(path.stem.split("_c")[0].removeprefix("run"))
    concurrency = int(path.stem.split("_c")[-1])
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        continue
    result = json.loads(lines[-1])
    rows[(model, concurrency, variant, run)] = result

print("model,concurrency,run_count,median_speedup_pct,mean_speedup_pct,last_speedup_pct,baseline_median_tpot_ms,flashsampling_median_tpot_ms,baseline_output_tps,flashsampling_output_tps")
for model in sorted({key[0] for key in rows}):
    concurrencies = sorted({key[1] for key in rows if key[0] == model})
    for concurrency in concurrencies:
        runs = sorted(
            {
                key[3]
                for key in rows
                if key[0] == model and key[1] == concurrency
            }
        )
        pairs = []
        for run in runs:
            base = rows.get((model, concurrency, "baseline", run))
            cand = rows.get((model, concurrency, "flashsampling", run))
            if base and cand:
                speedup = (
                    (base["median_tpot_ms"] - cand["median_tpot_ms"])
                    / base["median_tpot_ms"]
                    * 100
                    if base["median_tpot_ms"]
                    else 0.0
                )
                pairs.append((speedup, base, cand))
        if not pairs:
            continue
        speedups = [item[0] for item in pairs]
        base_tpots = [item[1]["median_tpot_ms"] for item in pairs]
        cand_tpots = [item[2]["median_tpot_ms"] for item in pairs]
        base_tps = [item[1]["output_throughput"] for item in pairs]
        cand_tps = [item[2]["output_throughput"] for item in pairs]
        print(
            f"{model},{concurrency},{len(pairs)},"
            f"{statistics.median(speedups):.2f},{statistics.mean(speedups):.2f},"
            f"{speedups[-1]:.2f},{statistics.median(base_tpots):.4f},"
            f"{statistics.median(cand_tpots):.4f},{statistics.median(base_tps):.4f},"
            f"{statistics.median(cand_tps):.4f}"
        )
PY
}

prepare_dataset

for model in ${MODELS}; do
  if [[ "${PRE_DOWNLOAD}" == "1" && "${DRY_RUN}" != "1" ]]; then
    hf download "${model}" >/dev/null
  fi
  run_variant "${model}" baseline
  run_variant "${model}" flashsampling
done

if [[ "${DRY_RUN}" != "1" ]]; then
  summarize | tee "${OUT_DIR}/summary.csv"
else
  echo "DRY_RUN complete. Results would be written to ${OUT_DIR}."
fi
