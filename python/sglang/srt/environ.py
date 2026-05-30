import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any, Optional


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str):
        super().__init__(default)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    SGLANG_SORT_WEIGHT_FILES = EnvBool(False)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    SGLANG_CUDA_COREDUMP_DIR = EnvStr("/tmp/sglang_cuda_coredumps")
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    SGLANG_SMC_PREFILL_STREAM_YIELD_MS = EnvFloat(0.0)
    SGLANG_SMC_TARGET_VERIFY_GRAPH = EnvBool(False)
    SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND = EnvStr("triton")
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)
    # Extra slots in req_to_token_pool for decode workers (only effective when
    # max_num_reqs > 32). Increases pool capacity so more KV cache transfers
    # can overlap with decode execution without raising max_running_requests.
    SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS = EnvInt(0)

    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER = EnvFloat(1.0)
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB = EnvInt(64)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)

    # MPS (Apple Silicon)
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Quantize x to int8 in the dispatch operator
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False) # This argument is deprecated
    SGLANG_NPU_FUSED_MOE_MODE = EnvInt(1)

    # MTHREADS & MUSA
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_ENABLE_WARP_DECODE = EnvBool(False)
    SGLANG_WARP_DECODE_CUTE = EnvStr("auto")
    SGLANG_WARP_DECODE_MAX_BATCH = EnvInt(64)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    # Iter3 #15 NVFP4 MoE: deploy-path opt-in for the fused
    # (residual-add + RMSNorm + linear NVFP4 quantize) kernel. When True,
    # the post-attention layernorm step in `prepare_mlp` is replaced by
    # `sglang.jit_kernel.nvfp4.fused_rmsnorm_scaled_fp4_quant_linear`
    # which writes the FP4-packed activation + linear SF tensors directly
    # into a per-layer scratch the downstream MoE consumes — eliminating
    # the BF16 hidden-states roundtrip between RMSNorm and `fp4_quantize`.
    # Effective only when:
    #   * MoE scheme is `compressed_tensors_w4a4_nvfp4_moe` and the
    #     flashinfer trtllm runner is selected (`flashinfer_trtllm`).
    #   * `should_allreduce_fusion=False` (no closed flashinfer
    #     allreduce-fusion engaged for the next layer).
    #   * RMSNorm is the non-`cast_x_before_out_mul` variant
    #     (Llama-style; DeepSeek-V3.2-REAP qualifies).
    # On all other paths the env var is a no-op and the existing
    # fused_add_rmsnorm + fp4_quantize pair runs unchanged.
    SGLANG_USE_SGL_NVFP4_FUSED_RMSNORM = EnvBool(False)

    # ai-blaise #15 iter7 PRIMARY: opt-in for the patched flashinfer
    # trtllm_fp4_block_scale_moe BF16-act path that unlocks the in-cubin
    # Bmm_Bfloat16_E2m1E2m1_*.cubin SM_100 variants and eliminates the
    # host-side scaled_fp4_quant_linear call. Requires the patches at
    # python/sglang/srt/external_kernels/flashinfer/patches/ to be
    # applied AND the vendored launcher to be built (Stage B). When
    # False or the vendored symbol unavailable, the iter1-3 + iter4
    # PRIMARY production path is taken unchanged. See
    # notes/nvfp4_moe_iter7_recon.md.
    SGLANG_USE_TRTLLM_BF16_ACT_FP4_MOE = EnvBool(False)

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Enable per-token NVFP4 activation scaling path for FlashInfer TRT-LLM MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # HIGGS dense 2-bit MoE expert-weight scheme (task #15, B200 DSv3.2).
    # When True, the HIGGS scheme keeps its 2-bit packed expert weights on
    # device and runs trtllm_bf16_moe at runtime by re-dequanting per
    # call. Trades GEMM throughput for ~8x weight-memory savings. When
    # False (default), HIGGS-packed weights are dequanted once at load
    # time and re-quantized to NVFP4 so the trtllm fp4 kernel runs
    # unchanged — no runtime perf delta vs the W4A4 NVFP4 path but no
    # GPU memory savings either. The follow-on flashinfer fork that
    # consumes HIGGS-packed weights inline will subsume this knob.
    SGLANG_OPT_USE_HIGGS_MOE_2BIT_BF16_RUNTIME = EnvBool(False)
    # Optional sub-row FWHT block size for HIGGS MoE expert weights.
    # 0 / unset = use the full row (one FWHT block of size in_dim per
    # row, matching the dense MLA latent convention). Power-of-two
    # values let larger hidden_size models (e.g. 7168) trade codec
    # accuracy against per-row CTA shared-memory budget when the
    # follow-on CUDA kernel lands.
    SGLANG_OPT_HIGGS_MOE_2BIT_BLOCK_SIZE = EnvInt(0)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    # TODO(mmangkad): Remove this once the FlashInfer unified allreduce-fusion
    # transport issue on GB200/GB300 platforms is fixed and verified resolved.
    SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT = EnvBool(None)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)

    # DeepSeek MHA Optimization
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False) # This argument is deprecated
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    # Force dynamic DeepEP Waterfill with runtime EP all-reduce instead of the
    # default static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)

    # NIXL-EP
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)

    # DSA Backend (canonical names; fall back to SGLANG_NSA_* with deprecation warning)
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA"
    )
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_USE_FUSED_METADATA_COPY = EnvBool(True)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)

    # HIGGS dense 2-bit + trtllm DSA decode FP8 sparse-materialize path
    # (ai-blaise #19 iter3 vector A). When set, the HIGGS sparse-materialize
    # adapter emits FP8 (e4m3) instead of BF16 (576 vs 1152 B/row, -50%
    # HBM traffic), and the query is FP8-quantized so the trtllm-gen
    # ``QkvE4m3OBfloat16`` sparse-MLA cubin set is selected. Saves ~6 ms
    # TPOT on the 12.3 ms BF16-bound materialization round-trip.
    SGLANG_HIGGS_DSA_TRTLLM_FP8 = EnvBool(False)
    # ai-blaise #19 iter9 PRIMARY vector: swap HIGGS sparse-MLA FP8
    # materialization for the iter8 inline producer kernel
    # (higgs_inline_sparse_mla_produce_fp8). Same FFI as
    # dequantize_higgs_dense_2bit_page_table_fp8 (iter3) but uses
    # cp.async slot prefetch + depth-2 SMEM ping-pong staging, which
    # the iter8 microbench measured +20.2%% per-kernel (~9.56 ms TPOT
    # across 61 layers) at B=128 K=2048. Same SAW-INT4 design pattern:
    # fused rotation+quantization into a single pass that the trtllm
    # cubin reads. Requires SGLANG_HIGGS_DSA_TRTLLM_FP8=1 to be
    # set as well; default off pending production-shape A/B.
    SGLANG_HIGGS_DSA_INLINE_PRODUCER = EnvBool(False)
    # Structural HIGGS fix (2026-05-30): route HIGGS dense KV decode
    # to the fused HIGGS+MLA decode kernel
    # (``forward_higgs_dense_2bit_mla_decode``) that does inline
    # FWHT_512 + EDEN2-16 dequant in the same kernel as the topk
    # attention, eliminating the materialize-then-attend FP8 HBM
    # round-trip the trtllm-gen sparse-MLA cubin path imposes.
    # SAW-INT4 (arxiv 2604.19157) calls this the "fused
    # rotation-quantization kernel ... zero measurable end-to-end
    # overhead" pattern. The kernel already exists (was only
    # reachable from target_verify); this env enables routing from
    # the regular decode path. Default off pending production-shape
    # A/B vs the iter3/iter9 dequant + cubin path.
    SGLANG_HIGGS_DSA_FUSED_MLA_DECODE = EnvBool(False)
    # Per-tensor scale applied before the FP8 cast in the HIGGS dequant
    # kernel. The downstream attention's ``bmm1_scale`` then multiplies
    # by ``1/SGLANG_HIGGS_DSA_TRTLLM_FP8_INV_KV_SCALE`` to recover the
    # original BMM1 magnitudes. Default 1.0 → saturating cast (HIGGS
    # decompress output is post-norm so typically fits in [-448, 448]).
    SGLANG_HIGGS_DSA_TRTLLM_FP8_INV_KV_SCALE = EnvFloat(1.0)
    # ai-blaise #19 iter4 vector B: side-stream HIGGS dequant inside
    # ``_forward_trtllm``. The dequant kernel runs on a dedicated CUDA
    # stream so the GPU scheduler can overlap it with the on-main-stream
    # work that already happens before the trtllm-gen sparse-MLA launch
    # (``set_mla_kv_buffer`` of the current token's K, the q_all FP8 cast,
    # ``transform_index_page_table_decode``, and the trtllm-gen kernel's
    # own warmup ramp). Cross-stream sync is event-based and
    # cuda-graph-capture safe (mirrors the ``Indexer.alt_stream`` pattern
    # in ``dsa/dsa_indexer.py``). Default off — flip to ``1`` on B200 with
    # ``SGLANG_HIGGS_DSA_TRTLLM_FP8=1`` to measure.
    SGLANG_HIGGS_DSA_TRTLLM_DEQUANT_STREAM = EnvBool(False)
    # ai-blaise #19 iter5 (primary vector): ping-pong compact dequant
    # buffers in the HIGGS DSA trtllm decode path. With a single
    # ``_higgs_selected_buffer_fp8`` slot, layer N+1's side-stream
    # dequant cannot overlap with layer N's trtllm-gen sparse-MLA read
    # because both touch the same physical scratch buffer. Allocating
    # two slots (parity = ``layer_id & 1``) breaks that write-after-read
    # hazard: layer N reads slot 0 while layer N+1 writes slot 1. Paired
    # with a tighter side-stream wait (event-based on the same-layer
    # ``set_mla_kv_buffer`` + page_table_1 transform only, *not* on the
    # prior trtllm-gen tail) this turns the iter4 sub-ms side-stream
    # scheduling win into a real cross-layer overlap of the dequant
    # kernel and the prior-layer trtllm-gen kernel. Memory cost: 2× the
    # compact FP8 + BF16 + compact_page_table buffers (~600 MiB total
    # vs 300 MiB) — acceptable on B200's 192 GiB HBM. Default off —
    # flip together with ``SGLANG_HIGGS_DSA_TRTLLM_DEQUANT_STREAM=1`` and
    # ``SGLANG_HIGGS_DSA_TRTLLM_FP8=1`` on B200 to measure.
    SGLANG_HIGGS_DSA_TRTLLM_PINGPONG = EnvBool(False)
    # ai-blaise #19 iter6 (primary vector): dedicated CUDA stream for
    # the trtllm-gen sparse-MLA kernel itself. With iter4's side-stream
    # dequant + iter5's ping-pong scratch in place, the remaining
    # serialization that prevents cross-layer overlap is that
    # trtllm-gen N runs on the main stream — so layer N+1's
    # side-stream dequant ``wait_stream(main)`` transitively waits on
    # prior trtllm-gen N via main-stream FIFO. Putting trtllm-gen on
    # its OWN stream leaves only the short ``set_mla_kv_buffer`` +
    # ``page_table_1`` transform + Q FP8 cast on main, so layer N+1's
    # dequant ``wait_stream(main)`` only blocks on those small kernels
    # and can run concurrently with the prior trtllm-gen. Causal
    # ordering preserved: trtllm stream waits on the dequant event
    # before launching; main stream waits on the trtllm completion
    # event before any downstream consumer of the attn output runs.
    # cuda-graph-capture safe (stream construction happens at backend
    # init outside any capture context; wait_event / wait_stream /
    # record_event are recorded into the captured graph). Requires
    # SGLANG_HIGGS_DSA_TRTLLM_DEQUANT_STREAM=1 and
    # SGLANG_HIGGS_DSA_TRTLLM_PINGPONG=1 to be set as well — otherwise
    # the dependent event chain is incomplete (dedicated trtllm stream
    # with single-slot scratch reintroduces the buffer aliasing
    # hazard). Default off — flip together with the iter5 stack on
    # B200 to measure.
    SGLANG_HIGGS_DSA_TRTLLM_DEDICATED_STREAM = EnvBool(False)
    # ai-blaise #19 iter7 (tertiary vector): depth-4 ping-pong rotation
    # for the HIGGS sparse-dequant compact buffers. iter5 introduced
    # depth-2 (``layer_id & 1`` parity) which broke the same-buffer
    # aliasing between layer N and N+1. iter6 added a dedicated
    # trtllm-gen stream so layer N+1 dequant's wait_stream(main) no
    # longer transitively pulls in the prior trtllm-gen tail. The
    # remaining bottleneck at iter7 is that the *single* dedicated
    # trtllm-gen stream serializes back-to-back trtllm-gen kernels
    # (layer N+1's trtllm-gen can't launch until layer N's trtllm-gen
    # drains on the same stream). Depth-4 adds 4 ping-pong scratch
    # slots indexed by ``layer_id & 3``, and the companion
    # ``SGLANG_HIGGS_DSA_TRTLLM_DEDICATED_STREAM_DUAL`` flag below adds
    # a second dedicated trtllm-gen stream so the parity-A and parity-B
    # streams run concurrently — layer N (stream A) overlaps with
    # layer N+1 (stream B) overlaps with layer N+2 (stream A again
    # after N's trtllm-gen drains). Memory cost: 4x compact dequant
    # buffer (~1.2 GiB total at production shape vs ~600 MiB depth-2),
    # comfortably within B200's 192 GiB HBM budget. Requires PINGPONG
    # to be on; falls back to depth-2 silently when off. Default off
    # — flip together with the iter6 stack to measure.
    SGLANG_HIGGS_DSA_TRTLLM_PINGPONG_DEPTH4 = EnvBool(False)
    # ai-blaise #19 iter7 (tertiary vector): a second dedicated CUDA
    # stream for the trtllm-gen sparse-MLA kernel itself. Paired with
    # depth-4 ping-pong above. Even layer ids run trtllm-gen on the
    # iter6 ``_higgs_trtllm_stream``; odd layer ids run on a new
    # ``_higgs_trtllm_stream_b``. The two streams have disjoint
    # ping-pong slots (parity 0/2 on stream A, parity 1/3 on stream B),
    # so back-to-back trtllm-gens overlap. Together with the iter4
    # dequant stream + iter5 ping-pong + iter6 dedicated_stream + this
    # iter7 depth4 + dual_stream, the per-layer pipeline becomes:
    #   layer N    : main→{kv_proj, set_kv, page_table_1}
    #              ; side_stream→dequant_{N}
    #              ; trtllm_stream_A→trtllm_{N}   (slot N&3)
    #   layer N+1  : main→{kv_proj_{N+1}, set_kv_{N+1}, page_table_1_{N+1}}
    #              ; side_stream→dequant_{N+1}
    #              ; trtllm_stream_B→trtllm_{N+1} (slot (N+1)&3)
    # Requires PINGPONG_DEPTH4=1 + DEDICATED_STREAM=1 (which itself
    # requires DEQUANT_STREAM=1 + PINGPONG=1). Default off.
    SGLANG_HIGGS_DSA_TRTLLM_DEDICATED_STREAM_DUAL = EnvBool(False)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # Flash Attention
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)

    # Kernels
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_DISABLE_PYNCCL = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_SPEC_V2 = EnvBool(True)
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)


    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Unified Radix Tree
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)

    # Breakable CUDA Graph
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # HTTP Server
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)

    # HTTP/2 Server
    SGLANG_GRANIAN_PARENT_PID = EnvInt(None)

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # Encoder gRPC
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")

    # Native gRPC server (internal, not yet user-facing)
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_ENABLE_GRPC = EnvBool(False)

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(False)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer (Kimi tiktoken: cache all_special_tokens / all_special_ids; the ITL can differ by +10x under high batch size).
    SGLANG_PATCH_TOKENIZER = EnvBool(True)

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    # ZBAL, zero buffer accelerate library, currently worked only in npu
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ====================================================================
    # DeepSeek V4
    SGLANG_OPT_DPSK_V4_RADIX = EnvBool(True)
    SGLANG_OPT_USE_OLD_COMPRESSOR = EnvBool(False)
    SGLANG_OPT_USE_TRITON_SWA_PREPARE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS_TRITON = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL = EnvBool(True)
    SGLANG_FIX_MTP_HC_HIDDEN = EnvBool(False)
    # ====================================================================

    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Default reasoning_effort for dsv4 chat encoder when request doesn't set it.
    # Accepts "", "max", "high" (empty string means unset); other values filtered to None.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")

    # CUDA kernels
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_TRITON_FUSED_MHC = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_OPT_USE_COMPRESSOR_V2 = EnvBool(True)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_TOPK_TRANSFORM_512_TORCH = EnvBool(False)

    # SWA radix cache
    SGLANG_OPT_CACHE_SWA_TRANSLATION = EnvBool(True)
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)
    SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = EnvBool(False)

    # DeepGemm Mega MoE
    SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)

    # When set, the mega-MoE x slot is packed E2M1 (FP4) instead of FP8 E4M3.
    # Halves symm-buffer footprint and unlocks the MXF4 mainloop downstream.
    # Setting this also exports DG_USE_FP4_ACTS=1 so DeepGEMM's symm-buffer
    # sizing + fp8_fp4_mega_moe pick up the FP4 layout.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS = EnvBool(False)
    # Switches the L1+L2 mainloops from kind::mxf8f6f4 (K=32 with-padding) to
    # kind::mxf4 (K=64 dense) inside fp8_fp4_mega_moe. No effect unless
    # SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is also set; DeepGEMM asserts
    # this combination on the host side.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND = EnvBool(False)
    SGLANG_OPT_FIX_MEGA_MOE_MEMORY = EnvBool(False)

    # TopK
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK = EnvBool(True)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # GEMM / kernel fusion
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_USE_JIT_EP_ACTIVATION = EnvBool(True)
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_SWIGLU_CLAMP_FUSION = EnvBool(True)

    # Cache / overlap
    SGLANG_OPT_USE_FUSED_STORE_CACHE = EnvBool(True)
    SGLANG_OPT_USE_JIT_NORM = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # CUDA graph
    SGLANG_PREP_IN_CUDA_GRAPH = EnvBool(True)

    # Distributed
    SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER = EnvBool(True)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # fmt: on

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)

    # Elastic EP Backup Port
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # Sglang Cache Dir
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)

    # Plugin system
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")


envs = Envs()
EnvField._allow_set_name = False


def _print_deprecated_env(old_name: str, new_name: Optional[str] = None):
    if old_name in os.environ:
        if new_name is None:
            warnings.warn(f"Environment variable {old_name} has been deprecated.")
        else:
            warnings.warn(
                f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
            )
            os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_GC_LOG", "SGLANG_LOG_GC")
    _print_deprecated_env(
        "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH", "SGLANG_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
    )
    _print_deprecated_env("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")
    _print_deprecated_env("SGLANG_ENABLE_THINKING", "SGLANG_DEFAULT_THINKING")
    _print_deprecated_env("SGLANG_REASONING_EFFORT", "SGLANG_DSV4_REASONING_EFFORT")
    _print_deprecated_env(
        "SGLANG_USE_JIT_ALL_REDUCE", "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError(f"assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
