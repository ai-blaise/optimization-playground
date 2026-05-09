# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Warp Decode: warp-level MoE decode kernels for small-batch inference.

Inspired by Cursor's warp decode approach. Instead of the traditional
expert-centric pipeline (pad -> scatter -> compute -> combine), each
Triton program instance owns exactly ONE output scalar. This eliminates
five of the eight stages in the traditional MoE path and removes all
intermediate buffers.

Two kernels:
  - Gate/Up kernel: each instance computes one intermediate neuron for
    one (token, expert) pair. Streams over the hidden dimension,
    accumulating gate and up projections in FP32. Applies SiLU(gate)*up.
  - Down kernel: each instance computes one output dimension for one
    token. Loops over top-k experts, folding routing weights into a
    single FP32 accumulator.

Target: SM100 (B200/GB200 Blackwell), small-batch decode (B <= 64).
For prefill and large batches, fall back to expert-centric execution.
"""

from sglang.srt.layers.moe.warp_decode.kernels import (
    _CUTE_AVAILABLE,
    _should_use_cute,
    warp_decode_moe,
    warp_decode_moe_packed,
)
from sglang.srt.layers.moe.warp_decode.runner import (
    WarpDecodeRunnerCore,
    is_warp_decode_enabled,
    should_use_warp_decode,
)

__all__ = [
    "warp_decode_moe",
    "warp_decode_moe_packed",
    "WarpDecodeRunnerCore",
    "is_warp_decode_enabled",
    "should_use_warp_decode",
    "_CUTE_AVAILABLE",
    "_should_use_cute",
]
