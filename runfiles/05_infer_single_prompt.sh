#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export ART_ROOT="${HISQROT4_RELEASE_ART_ROOT:-${PROJECT_ROOT}/artifacts/hisqrot4_alpha_0p9}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/models/Wan2.2-T2V-A14B}"
export PROMPT_FILE="${PROMPT_FILE:-}"
export PROMPT="${PROMPT:-A cinematic shot of a cat surfing on the sea.}"
export OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_ROOT}/video_output/hifx4/single_prompt_alpha0p9.mp4}"
export SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
export FRAME_NUM="${FRAME_NUM:-61}"
export ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
export OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"

bash "${PROJECT_ROOT}/runfiles/03_infer_ptq_standard.sh"
