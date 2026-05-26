#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ART_ROOT="${ART_ROOT:-${ROOT_DIR}/state_quant/hif4_ptq}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/models/Wan2.2-T2V-A14B}"
LOG_DIR="${ROOT_DIR}/logs"
GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
RESUME="${RESUME:-1}"
KEEP_BLOCKS="${KEEP_BLOCKS:-}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
WEIGHT_GROUP_SIZE="${WEIGHT_GROUP_SIZE:-64}"
TIMESTEP_COUNT="${TIMESTEP_COUNT:-40}"
SMOOTHQUANT_ENABLE="${SMOOTHQUANT_ENABLE:-1}"
SMOOTHQUANT_ALPHA="${SMOOTHQUANT_ALPHA:-0.9}"
SMOOTHQUANT_EPS="${SMOOTHQUANT_EPS:-1e-5}"
ROTATION_ENABLE="${ROTATION_ENABLE:-1}"
ROTATION_PATH="${ROTATION_PATH:-}"
ROTATION_SEED="${ROTATION_SEED:-17}"
HIGH_NOISE_SPLIT_RANGES="${HIGH_NOISE_SPLIT_RANGES:-}"

mkdir -p "${LOG_DIR}" "${ART_ROOT}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
PREP_LOG="${LOG_DIR}/prepare_ptq_${RUN_TAG}.log"

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

if [[ -z "${CUDA_DEVICES}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
CUDA_DEVICES="$(printf '%s' "${CUDA_DEVICES}" | tr -d '[:space:]')"
if [[ -n "${CUDA_DEVICES}" ]]; then
  VISIBLE_GPU_COUNT="$(CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
else
  VISIBLE_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
fi
if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${VISIBLE_GPU_COUNT}" -le 0 ]]; then
  echo "[PREPARE] Failed to auto-detect visible GPUs. Set GPU_COUNT and CUDA_DEVICES explicitly."
  exit 1
fi
if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="${VISIBLE_GPU_COUNT}"
elif ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "[PREPARE] GPU_COUNT must be a positive integer, got ${GPU_COUNT}."
  exit 1
elif [[ "${GPU_COUNT}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "[PREPARE] WARN: requested GPU_COUNT=${GPU_COUNT}, but visible only ${VISIBLE_GPU_COUNT}; using ${VISIBLE_GPU_COUNT}."
  GPU_COUNT="${VISIBLE_GPU_COUNT}"
fi
if [[ -z "${CUDA_DEVICES}" ]]; then
  CUDA_DEVICES="$(_make_default_cuda_devices "${GPU_COUNT}")"
fi
FIRST_GPU="${CUDA_DEVICES%%,*}"

if pgrep -f "${ROOT_DIR}/generate.py .*--hifx4_stage prepare" >/dev/null 2>&1; then
  echo "[PREPARE] Found stale prepare processes. Killing them first..."
  pkill -9 -f "${ROOT_DIR}/generate.py .*--hifx4_stage prepare" || true
  sleep 1
fi
LOW_CALIB="${ART_ROOT}/low_noise_model/hifx4/calibration.pt"
HIGH_CALIB="${ART_ROOT}/high_noise_model/hifx4/calibration.pt"
LOW_PREP="${ART_ROOT}/low_noise_model/hifx4/prepared.pt"
HIGH_PREP="${ART_ROOT}/high_noise_model/hifx4/prepared.pt"

if [[ ! -f "${LOW_CALIB}" || ! -f "${HIGH_CALIB}" ]]; then
  echo "[PREPARE] Missing Stage1 calibration artifacts."
  echo "[PREPARE] Need both: ${LOW_CALIB} and ${HIGH_CALIB}"
  exit 1
fi

echo "${RUN_TAG}" > "${LOG_DIR}/prepare_ptq_latest.tag"
ln -sfn "${PREP_LOG}" "${LOG_DIR}/prepare_ptq_latest.log"
echo "[PREPARE] log : ${PREP_LOG}"
echo "[PREPARE] tail: tail -f ${LOG_DIR}/prepare_ptq_latest.log"
echo "[PREPARE] GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES} (stage2 uses single process on first device)"
echo "[PREPARE] CKPT_DIR=${CKPT_DIR}"
echo "[PREPARE] KEEP_BLOCKS=${KEEP_BLOCKS:-<none>} ACT_QUANT_MODE=${ACT_QUANT_MODE} WEIGHT_GROUP_SIZE=${WEIGHT_GROUP_SIZE}(compat-only) TIMESTEP_COUNT=${TIMESTEP_COUNT}"
echo "[PREPARE] smoothquant: ENABLE=${SMOOTHQUANT_ENABLE} ALPHA=${SMOOTHQUANT_ALPHA} EPS=${SMOOTHQUANT_EPS}"
echo "[PREPARE] rotation: ENABLE=${ROTATION_ENABLE} SEED=${ROTATION_SEED} PATH=${ROTATION_PATH:-<internal deterministic>}"
echo "[PREPARE] high_noise_split_ranges: ${HIGH_NOISE_SPLIT_RANGES:-<disabled (grouped lookup still uses default 13+13 split)>}"
touch "${PREP_LOG}"

if [[ "${RESUME}" == "1" && -f "${LOW_PREP}" && -f "${HIGH_PREP}" ]]; then
  echo "[PREPARE] prepared.pt already exists for low/high, skip rerun."
  if [[ "${SMOOTHQUANT_ENABLE}" == "1" || "${ROTATION_ENABLE}" == "1" || -n "${ROTATION_PATH}" ]]; then
    echo "[PREPARE][WARN] RESUME=1 means requested SmoothQuant/rotation settings are not applied this run."
    echo "[PREPARE][WARN] Use RESUME=0 to rebuild prepared artifacts with new settings."
  fi
  echo "[$(date '+%F %T')] prepare skipped (prepared.pt exists for both low/high)." >> "${PREP_LOG}"
  exit 0
fi

if [[ "${RESUME}" != "1" ]]; then
  echo "[PREPARE] Fresh mode: remove previous prepared artifacts..."
  rm -f "${LOW_PREP}" "${HIGH_PREP}"
fi

if [[ "${SMOOTHQUANT_ENABLE}" == "1" ]]; then
  echo "[PREPARE] validating Stage1 min-max channel stats for SmoothQuant..."
  if ! python3 - <<PY
import sys
import torch

paths = [
    ("low", "${LOW_CALIB}"),
    ("high", "${HIGH_CALIB}"),
]
for tag, path in paths:
    payload = torch.load(path, map_location="cpu")
    layers = payload.get("layers", {}) if isinstance(payload, dict) else {}
    if not isinstance(layers, dict) or len(layers) == 0:
        print(f"[PREPARE][ERROR] {tag} calibration has no layers: {path}")
        sys.exit(1)
    missing = []
    for name, layer in layers.items():
        if not isinstance(layer, dict):
            missing.append(name)
            continue
        ch_min = layer.get("act_min_per_channel_global", None)
        ch_max = layer.get("act_max_per_channel_global", None)
        tb_min = layer.get("act_min_group_branch", layer.get("act_min_timestep_branch", None))
        tb_max = layer.get("act_max_group_branch", layer.get("act_max_timestep_branch", None))
        if not isinstance(ch_min, (list, tuple)) or len(ch_min) == 0:
            missing.append(name)
            continue
        if not isinstance(ch_max, (list, tuple)) or len(ch_max) == 0:
            missing.append(name)
            continue
        if not isinstance(tb_min, dict):
            missing.append(name)
            continue
        if not isinstance(tb_max, dict):
            missing.append(name)
    if missing:
        print(f"[PREPARE][ERROR] {tag} calibration missing min-max channel stats in {len(missing)} layers.")
        print(f"[PREPARE][ERROR] example layer: {missing[0]}")
        print("[PREPARE][ERROR] rerun Stage1 with RESUME=0 under current code before enabling SmoothQuant.")
        sys.exit(1)
print("[PREPARE] Stage1 grouped min-max stats check passed for SmoothQuant.")
sys.exit(0)
PY
  then
    exit 1
  fi
fi

echo "[PREPARE] Stage2 starts (offline_model=all)..."
EXTRA_ARGS=()
if [[ "${RESUME}" != "1" ]]; then
  EXTRA_ARGS+=(--ptq_force_rebuild)
fi
if [[ -n "${KEEP_BLOCKS}" ]]; then
  EXTRA_ARGS+=(--ptq_keep_blocks "${KEEP_BLOCKS}")
fi
if [[ "${SMOOTHQUANT_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=(--ptq_enable_smoothquant)
fi
EXTRA_ARGS+=(--ptq_smoothquant_alpha "${SMOOTHQUANT_ALPHA}")
EXTRA_ARGS+=(--ptq_smoothquant_eps "${SMOOTHQUANT_EPS}")
if [[ "${ROTATION_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=(--ptq_enable_rotation)
fi
if [[ -n "${ROTATION_PATH}" ]]; then
  EXTRA_ARGS+=(--ptq_rotation_path "${ROTATION_PATH}")
fi
EXTRA_ARGS+=(--ptq_rotation_seed "${ROTATION_SEED}")
if [[ -n "${HIGH_NOISE_SPLIT_RANGES}" ]]; then
  EXTRA_ARGS+=(--ptq_high_noise_split_ranges "${HIGH_NOISE_SPLIT_RANGES}")
fi

CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONUNBUFFERED=1 python "${ROOT_DIR}/generate.py" \
  --hifx4_qtype hifx4 \
  --hifx4_quant_method ptq \
  --hifx4_stage prepare \
  --ptq_offline_model all \
  --ptq_act_quant_mode "${ACT_QUANT_MODE}" \
  --ptq_weight_group_size "${WEIGHT_GROUP_SIZE}" \
  --ptq_timestep_count "${TIMESTEP_COUNT}" \
  --ptq_artifact_root "${ART_ROOT}" \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir "${CKPT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  > "${PREP_LOG}" 2>&1

if [[ ! -f "${LOW_PREP}" || ! -f "${HIGH_PREP}" ]]; then
  echo "[PREPARE] Stage2 finished but prepared artifacts are incomplete."
  echo "[PREPARE] Need both: ${LOW_PREP} and ${HIGH_PREP}"
  exit 1
fi

echo "[PREPARE] Stage2 finished."
echo "[PREPARE] prepared:"
echo "  ${LOW_PREP}"
echo "  ${HIGH_PREP}"
