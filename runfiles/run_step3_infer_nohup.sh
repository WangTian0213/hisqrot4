#!/usr/bin/env bash
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

STEP3_SCRIPT="${PROJECT_ROOT}/runfiles/03_infer_ptq_standard.sh"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S_%N)}"
LOG_FILE="${LOG_DIR}/step3_infer_ptq_${RUN_TAG}.log"
ART_ROOT="${ART_ROOT:-${PROJECT_ROOT}/state_quant/hif4_ptq}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/models/Wan2.2-T2V-A14B}"
GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
RESUME="${RESUME:-1}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
INFER_ENGINE="${INFER_ENGINE:-python}"
PROMPT="${PROMPT:-a cat surfing on the sea}"
OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_ROOT}/video_output/ptq_hifx4_standard.mp4}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
KEEP_BLOCKS="${KEEP_BLOCKS:-}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
if [[ "${PROMPT_FILE+x}" == "x" ]]; then
  PROMPT_FILE="${PROMPT_FILE}"
else
  PROMPT_FILE="${PROJECT_ROOT}/data/prompts/OpenS2V-5M_to_mm_vbench_30.json"
fi
PROMPT_INDEX="${PROMPT_INDEX:-0}"
FRAME_NUM="${FRAME_NUM:-61}"
EXPECT_SMOOTHQUANT="${EXPECT_SMOOTHQUANT:-0}"

nohup env \
RUN_TAG="${RUN_TAG}" \
ART_ROOT="${ART_ROOT}" \
CKPT_DIR="${CKPT_DIR}" \
GPU_COUNT="${GPU_COUNT}" \
CUDA_DEVICES="${CUDA_DEVICES}" \
RESUME="${RESUME}" \
OFFLOAD_MODEL="${OFFLOAD_MODEL}" \
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE}" \
INFER_ENGINE="${INFER_ENGINE}" \
PROMPT="${PROMPT}" \
OUTPUT_FILE="${OUTPUT_FILE}" \
SAMPLE_STEPS="${SAMPLE_STEPS}" \
KEEP_BLOCKS="${KEEP_BLOCKS}" \
ACT_QUANT_MODE="${ACT_QUANT_MODE}" \
PROMPT_FILE="${PROMPT_FILE}" \
PROMPT_INDEX="${PROMPT_INDEX}" \
FRAME_NUM="${FRAME_NUM}" \
EXPECT_SMOOTHQUANT="${EXPECT_SMOOTHQUANT}" \
bash "${STEP3_SCRIPT}" \
  > "${LOG_FILE}" 2>&1 &

echo "[RUNNING] step3_infer_ptq pid=$!"
echo "[LOG] ${LOG_FILE}"
echo "[TIP] device mode: GPU_COUNT=${GPU_COUNT:-<auto>} CUDA_DEVICES=${CUDA_DEVICES:-<auto>}"
echo "[TIP] artifact root: ART_ROOT=${ART_ROOT}"
echo "[TIP] ckpt dir: CKPT_DIR=${CKPT_DIR}"
echo "[TIP] resume mode: RESUME=${RESUME} (1=skip if output already exists)"
echo "[TIP] offload mode: OFFLOAD_MODEL=${OFFLOAD_MODEL} (Wan official path)"
echo "[TIP] alloc conf : PYTORCH_CUDA_ALLOC_CONF_VALUE=${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
echo "[TIP] quant config: SAMPLE_STEPS=${SAMPLE_STEPS} ACT_QUANT_MODE=${ACT_QUANT_MODE} KEEP_BLOCKS=${KEEP_BLOCKS:-<none>}"
echo "[TIP] infer engine: INFER_ENGINE=${INFER_ENGINE} (PTQ route requires python)"
echo "[TIP] prompt source: PROMPT_FILE=${PROMPT_FILE:-<none>} PROMPT_INDEX=${PROMPT_INDEX}"
echo "[TIP] fixed frame config: FRAME_NUM=${FRAME_NUM} (fps uses Wan2.2 task default)"
echo "[TIP] artifact assert: EXPECT_SMOOTHQUANT=${EXPECT_SMOOTHQUANT} (1=require SQ artifacts)"
echo "[TIP] output file: ${OUTPUT_FILE}"
echo "[TIP] stage3 log: ${PROJECT_ROOT}/logs/infer_ptq_latest.log"
