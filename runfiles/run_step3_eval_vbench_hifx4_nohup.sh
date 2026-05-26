#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

VIDEOS_INPUT_DIR="${VIDEOS_INPUT_DIR:-${PROJECT_ROOT}/video_output/hifx4/OpenS2V-5M_to_mm_vbench_30}"
RUN_TAG="${RUN_TAG:-vbench_hifx4_OpenS2V-5M_to_mm_vbench_30}"
RESUME="${RESUME:-1}"
EXPECTED_VIDEO_CNT="${EXPECTED_VIDEO_CNT:-0}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
VBENCH_DIR="${VBENCH_DIR:-${PROJECT_ROOT}/vbench}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/eval_output}"
OUT_PATH="${OUT_PATH:-${OUT_ROOT}/${RUN_TAG}}"
STATE_ROOT="${STATE_ROOT:-${PROJECT_ROOT}/state_eval}"
STATE_DIR="${STATE_DIR:-${STATE_ROOT}/${RUN_TAG}}"
DONE_DIMS_FILE="${DONE_DIMS_FILE:-${STATE_DIR}/vbench_custom5_done_dims.txt}"
EVAL_VIDEOS_PATH="${EVAL_VIDEOS_PATH:-${STATE_DIR}/eval_videos_${RUN_TAG}}"

if [[ "${RESUME}" == "0" ]]; then
  rm -rf "${STATE_DIR}" "${OUT_PATH}"
fi

RUN_TIME_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/step3_eval_${RUN_TAG}_${RUN_TIME_TAG}.log"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/step3_eval_${RUN_TAG}_latest.log"

nohup env \
  CONDA_SH="${CONDA_SH}" \
  CONDA_ENV="${CONDA_ENV}" \
  VBENCH_DIR="${VBENCH_DIR}" \
  VIDEOS_INPUT_DIR="${VIDEOS_INPUT_DIR}" \
  RUN_TAG="${RUN_TAG}" \
  RESUME="${RESUME}" \
  OUT_ROOT="${OUT_ROOT}" \
  OUT_PATH="${OUT_PATH}" \
  STATE_ROOT="${STATE_ROOT}" \
  STATE_DIR="${STATE_DIR}" \
  DONE_DIMS_FILE="${DONE_DIMS_FILE}" \
  EXPECTED_VIDEO_CNT="${EXPECTED_VIDEO_CNT}" \
  EVAL_VIDEOS_PATH="${EVAL_VIDEOS_PATH}" \
  bash "${PROJECT_ROOT}/runfiles/03_eval_vbench_video_dir_custom5.sh" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[RUNNING] eval_${RUN_TAG} pid=${PID}"
echo "[LOG] ${LOG_FILE}"
echo "[LATEST] tail -f ${LOG_DIR}/step3_eval_${RUN_TAG}_latest.log"
echo "[VIDEOS] ${VIDEOS_INPUT_DIR}"
echo "[OUTPUT] ${OUT_PATH}"
