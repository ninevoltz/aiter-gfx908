#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
aiter_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${aiter_root}${PYTHONPATH:+:${PYTHONPATH}}"
export GPU_ARCHS=gfx908 PYTORCH_ROCM_ARCH=gfx908
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}" HIP_PATH="${HIP_PATH:-/opt/rocm}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-${aiter_root}/.jit}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${aiter_root}/.triton-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${aiter_root}/.vllm-cache}"
export MAX_JOBS="${MAX_JOBS:-4}"
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
export VLLM_ROCM_USE_AITER_TRITON_GEMM="${VLLM_ROCM_USE_AITER_TRITON_GEMM:-1}"
export VLLM_ROCM_USE_AITER_TRITON_ROPE="${VLLM_ROCM_USE_AITER_TRITON_ROPE:-1}"
export VLLM_ROCM_USE_AITER_LINEAR="${VLLM_ROCM_USE_AITER_LINEAR:-0}"
export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-0}"
export VLLM_ROCM_USE_AITER_MHA="${VLLM_ROCM_USE_AITER_MHA:-0}"
export VLLM_ROCM_USE_AITER_MLA="${VLLM_ROCM_USE_AITER_MLA:-0}"
export VLLM_ROCM_USE_AITER_CUSTOM_AR="${VLLM_ROCM_USE_AITER_CUSTOM_AR:-0}"
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION="${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-0}"
rocr_runtime="${MI100_ROCR_RUNTIME:-${aiter_root}/../vllm-mi100_support/rocr-fixed/lib/libhsa-runtime64.so.1}"
if [[ -r "${rocr_runtime}" ]]; then
    export LD_PRELOAD="${rocr_runtime}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARGS...]" >&2
    exit 2
fi
exec "$@"
