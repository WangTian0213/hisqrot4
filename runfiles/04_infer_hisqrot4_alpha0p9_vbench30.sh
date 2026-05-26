#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export QTYPE="${QTYPE:-hifx4}"
export ART_ROOT="${HISQROT4_RELEASE_ART_ROOT:-${PROJECT_ROOT}/artifacts/hisqrot4_alpha_0p9}"
export CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/models/Wan2.2-T2V-A14B}"
export PROMPT_FILE="${PROMPT_FILE:-${PROJECT_ROOT}/data/prompts/OpenS2V-5M_to_mm_vbench_30.json}"
export FIRST_N="${FIRST_N:-}"
export RUN_TS="${RUN_TS:-alpha0p9}"
export OUT_FOLDER="${OUT_FOLDER:-}"
if [[ -z "${OUT_FOLDER}" ]]; then
  export OUT_SUBDIR="${OUT_SUBDIR:-OpenS2V-5M_to_mm_vbench_30_alpha0p9}"
else
  export OUT_SUBDIR="${OUT_SUBDIR:-}"
fi
export ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
export SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
export FRAME_NUM="${FRAME_NUM:-61}"
export OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
export INFER_ENGINE="${INFER_ENGINE:-python}"
export RESUME="${RESUME:-1}"

_make_default_cuda_devices() {
  local count="$1"
  local joined=""
  local i
  for ((i=0; i<count; i++)); do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${i}"
  done
  printf '%s' "${joined}"
}

_detect_visible_gpu_count() {
  if [[ -n "${CUDA_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  else
    python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  fi
}

if [[ -z "${CUDA_DEVICES:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
CUDA_DEVICES="$(printf '%s' "${CUDA_DEVICES:-}" | tr -d '[:space:]')"

VISIBLE_GPU_COUNT="$(_detect_visible_gpu_count)"
if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${VISIBLE_GPU_COUNT}" -le 0 ]]; then
  echo "[HiSQRot4][ERROR] failed to auto-detect visible GPUs." >&2
  exit 1
fi

if [[ -z "${GPU_COUNT:-}" ]]; then
  export GPU_COUNT="${VISIBLE_GPU_COUNT}"
elif ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "[HiSQRot4][ERROR] GPU_COUNT must be a positive integer, got ${GPU_COUNT}" >&2
  exit 1
elif [[ "${GPU_COUNT}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "[HiSQRot4][WARN] requested GPU_COUNT=${GPU_COUNT}, but visible only ${VISIBLE_GPU_COUNT}; using ${VISIBLE_GPU_COUNT}." >&2
  export GPU_COUNT="${VISIBLE_GPU_COUNT}"
fi

if [[ -z "${CUDA_DEVICES}" ]]; then
  export CUDA_DEVICES="$(_make_default_cuda_devices "${GPU_COUNT}")"
else
  export CUDA_DEVICES
fi

_require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[HiSQRot4][ERROR] missing file: ${path}" >&2
    exit 1
  fi
}

_require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[HiSQRot4][ERROR] missing directory: ${path}" >&2
    exit 1
  fi
}

_require_dir "${CKPT_DIR}"
_require_file "${CKPT_DIR}/configuration.json"
_require_file "${CKPT_DIR}/Wan2.1_VAE.pth"
_require_file "${CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
_require_file "${CKPT_DIR}/low_noise_model/diffusion_pytorch_model.safetensors.index.json"
_require_file "${CKPT_DIR}/high_noise_model/diffusion_pytorch_model.safetensors.index.json"

_require_file "${PROMPT_FILE}"
_require_file "${ART_ROOT}/low_noise_model/${QTYPE}/prepared.pt"
_require_file "${ART_ROOT}/low_noise_model/${QTYPE}/manifest.json"
_require_file "${ART_ROOT}/high_noise_model/${QTYPE}/prepared.pt"
_require_file "${ART_ROOT}/high_noise_model/${QTYPE}/manifest.json"

echo "============================================================"
echo "[HiSQRot4] alpha=0.9 Stage3 batch inference"
echo "[HiSQRot4] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[HiSQRot4] CKPT_DIR=${CKPT_DIR}"
echo "[HiSQRot4] ART_ROOT=${ART_ROOT}"
echo "[HiSQRot4] PROMPT_FILE=${PROMPT_FILE}"
echo "[HiSQRot4] FIRST_N=${FIRST_N:-<all prompts>}"
echo "[HiSQRot4] GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES}"
echo "[HiSQRot4] OUT_FOLDER=${OUT_FOLDER:-<auto>}"
echo "[HiSQRot4] OUT_SUBDIR=${OUT_SUBDIR:-<unused>}"
echo "============================================================"

bash "${PROJECT_ROOT}/runfiles/03_batch_custom_prompt_file_infer.sh"
