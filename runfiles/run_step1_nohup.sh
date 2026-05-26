#!/usr/bin/env bash
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

STEP1_SCRIPT="${PROJECT_ROOT}/runfiles/01_calibrate_ptq_standard.sh"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S_%N)}"
LOG_FILE="${LOG_DIR}/step1_calibrate_${RUN_TAG}.log"
GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
RESUME="${RESUME:-1}"
CHECKPOINT_EVERY_PROMPT="${CHECKPOINT_EVERY_PROMPT:-1}"
CHECKPOINT_INTERVAL_PROMPTS="${CHECKPOINT_INTERVAL_PROMPTS:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data/prompts/OpenS2V-5M_to_mm_calib_30.json}"

nohup env \
RUN_TAG="${RUN_TAG}" \
GPU_COUNT="${GPU_COUNT}" \
CUDA_DEVICES="${CUDA_DEVICES}" \
RESUME="${RESUME}" \
CHECKPOINT_EVERY_PROMPT="${CHECKPOINT_EVERY_PROMPT}" \
CHECKPOINT_INTERVAL_PROMPTS="${CHECKPOINT_INTERVAL_PROMPTS}" \
SAMPLE_STEPS="${SAMPLE_STEPS}" \
ACT_QUANT_MODE="${ACT_QUANT_MODE}" \
PROMPT_FILE="${PROMPT_FILE}" \
bash "${STEP1_SCRIPT}" \
  > "${LOG_FILE}" 2>&1 &

echo "[RUNNING] step1_calibrate pid=$!"
echo "[LOG] ${LOG_FILE}"
echo "[TIP] device mode: GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES}"
echo "[TIP] resume mode: RESUME=${RESUME} (1=continue from shard checkpoints)"
echo "[TIP] checkpoint config: CHECKPOINT_EVERY_PROMPT=${CHECKPOINT_EVERY_PROMPT} CHECKPOINT_INTERVAL_PROMPTS=${CHECKPOINT_INTERVAL_PROMPTS}"
echo "[TIP] calibration config: prompt_file=${PROMPT_FILE} sample_steps=${SAMPLE_STEPS}"
echo "[TIP] prompt-level behavior: text uses prompt->cap fallback; frame_num is fixed to 61; fps uses Wan2.2 task default."
echo "[TIP] quant config: ACT_QUANT_MODE=${ACT_QUANT_MODE} (stage1 keep_blocks ignored)"
echo "[TIP] stage1 saves per-channel + timestep-branch min/max for SmoothQuant/lookup (rerun with RESUME=0 if artifacts are old)"
echo "[TIP] shard logs: ${PROJECT_ROOT}/logs/calib30_shard0_latest.log, ${PROJECT_ROOT}/logs/calib30_shard1_latest.log"
echo "[TIP] merge log: ${PROJECT_ROOT}/logs/calib30_merge_latest.log"
