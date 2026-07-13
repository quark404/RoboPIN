#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-QwQ2/RoboPIN-4B}"
MODEL_NAME="${MODEL_NAME:-RoboPIN-4B}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-eval_erqa.py}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

python "${BENCHMARK_SCRIPT}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --backbone qwen3 \
  --max_model_len "${MAX_MODEL_LEN}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
