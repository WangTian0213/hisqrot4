#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

QTYPE="${QTYPE:-hifx4}"
FIRST_N="${FIRST_N:-}"
PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data/prompts/OpenS2V-5M_to_mm_vbench_30.json}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_FOLDER="${OUT_FOLDER:-}"
if [[ -n "${OUT_FOLDER}" ]]; then
  if [[ "${OUT_FOLDER}" = /* ]]; then
    VIDEOS_INPUT_DIR="${OUT_FOLDER}"
  else
    VIDEOS_INPUT_DIR="${PROJECT_ROOT}/${OUT_FOLDER}"
  fi
  OUT_SUBDIR="$(basename "${VIDEOS_INPUT_DIR}")"
elif [[ -n "${OUT_SUBDIR:-}" ]]; then
  OUT_SUBDIR="${OUT_SUBDIR}"
  VIDEOS_INPUT_DIR="${PROJECT_ROOT}/video_output/${QTYPE}/${OUT_SUBDIR}"
else
  PROMPT_BASENAME="$(basename "${PROMPT_FILE}")"
  PROMPT_STEM="${PROMPT_BASENAME%.*}"
  SAFE_PROMPT_STEM="$(printf '%s' "${PROMPT_STEM}" | tr '/ ' '__')"
  OUT_SUBDIR="${SAFE_PROMPT_STEM}_${RUN_TS}"
  VIDEOS_INPUT_DIR="${PROJECT_ROOT}/video_output/${QTYPE}/${OUT_SUBDIR}"
fi
AUTO_VBENCH="${AUTO_VBENCH:-0}"
EVAL_RESUME="${EVAL_RESUME:-1}"

SAFE_SUBDIR="$(printf '%s' "${OUT_SUBDIR}" | tr '/ ' '__')"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-vbench_${QTYPE}_${SAFE_SUBDIR}}"
EVAL_EXPECTED_VIDEO_CNT="${EVAL_EXPECTED_VIDEO_CNT:-${FIRST_N:-0}}"

echo "[PIPELINE] step1/2: batch video generation"
QTYPE="${QTYPE}" \
FIRST_N="${FIRST_N}" \
PROMPT_FILE="${PROMPT_FILE}" \
RUN_TS="${RUN_TS}" \
OUT_FOLDER="${OUT_FOLDER}" \
OUT_SUBDIR="${OUT_SUBDIR}" \
bash "${PROJECT_ROOT}/runfiles/03_batch_custom_prompt_file_infer.sh"

if [[ "${AUTO_VBENCH}" != "1" ]]; then
  echo "[PIPELINE] AUTO_VBENCH=${AUTO_VBENCH}, skip VBench evaluation."
  exit 0
fi

echo "[PIPELINE] step2/2: VBench 5-dim evaluation"
VIDEOS_INPUT_DIR="${VIDEOS_INPUT_DIR}" \
RUN_TAG="${EVAL_RUN_TAG}" \
RESUME="${EVAL_RESUME}" \
EXPECTED_VIDEO_CNT="${EVAL_EXPECTED_VIDEO_CNT}" \
bash "${PROJECT_ROOT}/runfiles/03_eval_vbench_video_dir_custom5.sh"

echo "[PIPELINE] done."
