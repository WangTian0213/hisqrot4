#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ART_ROOT="${ART_ROOT:-${ROOT_DIR}/state_quant/hif4_ptq}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/data/prompts/OpenS2V-5M_to_mm_calib_30.json}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/models/Wan2.2-T2V-A14B}"
LOG_DIR="${ROOT_DIR}/logs"
DETACH="${DETACH:-0}"
DETACH_LOG="${DETACH_LOG:-}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
CALIB_MODEL_PARALLEL_GPUS="${CALIB_MODEL_PARALLEL_GPUS:-0}"
CALIB_MODEL_PARALLEL_DEVICES="${CALIB_MODEL_PARALLEL_DEVICES:-}"
RESUME="${RESUME:-0}"
CHECKPOINT_EVERY_PROMPT="${CHECKPOINT_EVERY_PROMPT:-1}"
CHECKPOINT_INTERVAL_PROMPTS="${CHECKPOINT_INTERVAL_PROMPTS:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
CALIB_DISTRIBUTED_NODE="${CALIB_DISTRIBUTED_NODE:-0}"
NODE_SHARDS_RAW="${NODE_SHARDS:-}"
NODE_SHARD_IDX_RAW="${NODE_SHARD_IDX:-}"
NODE_SHARDS="${NODE_SHARDS:-2}"
NODE_SHARD_IDX="${NODE_SHARD_IDX:-0}"
CLEAN_ARTIFACTS_ON_START="${CLEAN_ARTIFACTS_ON_START:-0}"
TORCHRUN_MASTER_ADDR="${TORCHRUN_MASTER_ADDR:-127.0.0.1}"
TORCHRUN_MASTER_PORT="${TORCHRUN_MASTER_PORT:-29500}"

mkdir -p "${LOG_DIR}" "${ART_ROOT}"

_count_cuda_devices() {
  local raw="${1// /}"
  local count=0
  local item
  IFS=',' read -r -a items <<< "${raw}"
  for item in "${items[@]}"; do
    if [[ -n "${item}" ]]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

_make_default_cuda_devices() {
  local n="${1}"
  local out=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ -z "${out}" ]]; then
      out="${i}"
    else
      out="${out},${i}"
    fi
  done
  echo "${out}"
}

AUTO_GPU_CONFIG_SOURCE=""
if [[ "${CALIB_MODEL_PARALLEL_GPUS}" != "0" ]]; then
  if ! [[ "${CALIB_MODEL_PARALLEL_GPUS}" =~ ^[0-9]+$ ]] || (( CALIB_MODEL_PARALLEL_GPUS < 2 )); then
    echo "[CALIB] CALIB_MODEL_PARALLEL_GPUS must be an integer >=2, got ${CALIB_MODEL_PARALLEL_GPUS}"
    exit 1
  fi
  CALIB_DISTRIBUTED_NODE="1"
  if [[ -z "${GPU_COUNT}" ]]; then
    GPU_COUNT="${CALIB_MODEL_PARALLEL_GPUS}"
    AUTO_GPU_CONFIG_SOURCE="CALIB_MODEL_PARALLEL_GPUS"
  elif [[ "${GPU_COUNT}" != "${CALIB_MODEL_PARALLEL_GPUS}" ]]; then
    echo "[CALIB] GPU_COUNT=${GPU_COUNT} conflicts with CALIB_MODEL_PARALLEL_GPUS=${CALIB_MODEL_PARALLEL_GPUS}"
    exit 1
  fi
  if [[ -z "${CUDA_DEVICES}" ]]; then
    if [[ -n "${CALIB_MODEL_PARALLEL_DEVICES}" ]]; then
      CUDA_DEVICES="${CALIB_MODEL_PARALLEL_DEVICES}"
      AUTO_GPU_CONFIG_SOURCE="${AUTO_GPU_CONFIG_SOURCE:-CALIB_MODEL_PARALLEL_DEVICES}"
    else
      CUDA_DEVICES="$(_make_default_cuda_devices "${CALIB_MODEL_PARALLEL_GPUS}")"
      AUTO_GPU_CONFIG_SOURCE="${AUTO_GPU_CONFIG_SOURCE:-CALIB_MODEL_PARALLEL_GPUS default devices}"
    fi
  fi
  if [[ -z "${NODE_SHARDS_RAW}" ]]; then
    NODE_SHARDS="1"
  fi
  if [[ -z "${NODE_SHARD_IDX_RAW}" ]]; then
    NODE_SHARD_IDX="0"
  fi
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
  if [[ -z "${GPU_COUNT}" ]]; then
    GPU_COUNT="$(_count_cuda_devices "${CUDA_DEVICES}")"
    AUTO_GPU_CONFIG_SOURCE="CUDA_DEVICES"
  fi
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
  if [[ -z "${GPU_COUNT}" ]]; then
    GPU_COUNT="$(_count_cuda_devices "${CUDA_DEVICES}")"
  fi
  AUTO_GPU_CONFIG_SOURCE="CUDA_VISIBLE_DEVICES"
elif [[ -z "${GPU_COUNT}" ]]; then
  DETECTED_GPU_COUNT="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
  if [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( DETECTED_GPU_COUNT > 0 )); then
    GPU_COUNT="${DETECTED_GPU_COUNT}"
    CUDA_DEVICES="$(_make_default_cuda_devices "${GPU_COUNT}")"
    AUTO_GPU_CONFIG_SOURCE="torch.cuda.device_count()"
  fi
fi

if [[ -z "${GPU_COUNT}" ]]; then
  echo "[CALIB] Failed to auto-detect visible GPUs. Set GPU_COUNT and CUDA_DEVICES explicitly."
  exit 1
fi
if [[ -z "${CUDA_DEVICES}" ]]; then
  if [[ "${GPU_COUNT}" == "1" ]]; then
    CUDA_DEVICES="0"
  else
    CUDA_DEVICES="$(_make_default_cuda_devices "${GPU_COUNT}")"
  fi
fi

if [[ "${DETACH}" == "1" && "${_CALIB_DETACH_CHILD:-0}" != "1" ]]; then
  DETACH_TAG="$(date +%Y%m%d_%H%M%S)"
  DETACH_LOG_PATH="${DETACH_LOG:-${LOG_DIR}/calib30_detach_${DETACH_TAG}.log}"
  echo "[CALIB] DETACH=1 detected. Relaunch in background with nohup+setsid+disown."
  echo "[CALIB] detach driver log: ${DETACH_LOG_PATH}"
  ln -sfn "${DETACH_LOG_PATH}" "${LOG_DIR}/calib30_detach_latest.log"
  nohup setsid env _CALIB_DETACH_CHILD=1 DETACH=0 bash "$0" "$@" \
    > "${DETACH_LOG_PATH}" 2>&1 < /dev/null &
  DETACH_PID=$!
  disown "${DETACH_PID}" 2>/dev/null || true
  echo "[CALIB] detached pid=${DETACH_PID}"
  echo "[CALIB] tail detach log: tail -f ${LOG_DIR}/calib30_detach_latest.log"
  exit 0
fi

if [[ "${_CALIB_DETACH_CHILD:-0}" == "1" ]]; then
  trap '' HUP
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF_VALUE}}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
FIRST_GPU="${CUDA_DEVICES%%,*}"
SECOND_GPU="${CUDA_DEVICES##*,}"

if [[ "${CALIB_DISTRIBUTED_NODE}" == "1" ]]; then
  SHARDS="${NODE_SHARDS}"
  SHARD_IDX="${NODE_SHARD_IDX}"
  SHARD0_LOG="${LOG_DIR}/calib30_shard${SHARD_IDX}_of${SHARDS}_${RUN_TAG}.log"
  SHARD1_LOG="${LOG_DIR}/calib30_shard${SHARD_IDX}_of${SHARDS}_${RUN_TAG}.log"
else
  SHARDS="${GPU_COUNT}"
  SHARD_IDX="0"
  SHARD0_LOG="${LOG_DIR}/calib30_shard0_of${SHARDS}_${RUN_TAG}.log"
  SHARD1_LOG="${LOG_DIR}/calib30_shard1_of${SHARDS}_${RUN_TAG}.log"
fi
MERGE_LOG="${LOG_DIR}/calib30_merge_${RUN_TAG}.log"
LOW_CALIB="${ART_ROOT}/low_noise_model/hifx4/calibration.pt"
HIGH_CALIB="${ART_ROOT}/high_noise_model/hifx4/calibration.pt"

if pgrep -f "${ROOT_DIR}/generate.py .*--hifx4_stage calibrate" >/dev/null 2>&1; then
  echo "[CALIB] Found stale calibrate processes. Killing them first..."
  pkill -9 -f "${ROOT_DIR}/generate.py .*--hifx4_stage calibrate" || true
  sleep 1
fi

if [[ "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "2" ]]; then
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
    echo "[CALIB] GPU_COUNT must be a positive integer, got ${GPU_COUNT}"
    exit 1
  fi
fi
if [[ "${CALIB_DISTRIBUTED_NODE}" != "1" && "${GPU_COUNT}" != "1" && "${GPU_COUNT}" != "2" ]]; then
  echo "[CALIB] GPU_COUNT=${GPU_COUNT} requires distributed-node/model-parallel mode."
  echo "[CALIB] Set CALIB_DISTRIBUTED_NODE=1 (or CALIB_MODEL_PARALLEL_GPUS=${GPU_COUNT})."
  exit 1
fi
if [[ "${CALIB_DISTRIBUTED_NODE}" == "1" && "${GPU_COUNT}" -lt 2 ]]; then
  echo "[CALIB] CALIB_DISTRIBUTED_NODE=1 expects GPU_COUNT>=2."
  exit 1
fi
if (( SHARD_IDX < 0 || SHARD_IDX >= SHARDS )); then
  echo "[CALIB] Invalid shard index: NODE_SHARD_IDX=${SHARD_IDX}, NODE_SHARDS=${SHARDS}"
  exit 1
fi
SHOW_SHARD1_LOG="0"
SHOW_MERGE_LOG="0"
if [[ "${CALIB_DISTRIBUTED_NODE}" == "1" ]]; then
  SHOW_MERGE_LOG="1"
elif [[ "${GPU_COUNT}" == "2" ]]; then
  SHOW_SHARD1_LOG="1"
  SHOW_MERGE_LOG="1"
fi

if [[ "${RESUME}" != "1" ]]; then
  if [[ "${CALIB_DISTRIBUTED_NODE}" == "1" && "${CLEAN_ARTIFACTS_ON_START}" != "1" ]]; then
    echo "[CALIB] Fresh run mode + distributed node: skip auto-clean."
    echo "[CALIB] If needed, set CLEAN_ARTIFACTS_ON_START=1 on one node to clean artifacts."
  else
    echo "[CALIB] Fresh run mode: cleaning previous stage1 artifacts..."
    rm -f "${ART_ROOT}/low_noise_model/hifx4/calibration.pt" "${ART_ROOT}/low_noise_model/hifx4/manifest.json"
    rm -f "${ART_ROOT}/high_noise_model/hifx4/calibration.pt" "${ART_ROOT}/high_noise_model/hifx4/manifest.json"
    rm -f "${ART_ROOT}/low_noise_model/hifx4/calibration_shards/"*.pt 2>/dev/null || true
    rm -f "${ART_ROOT}/high_noise_model/hifx4/calibration_shards/"*.pt 2>/dev/null || true
  fi
else
  echo "[CALIB] Resume mode: keep existing shard stats and continue."
  if [[ -f "${LOW_CALIB}" && -f "${HIGH_CALIB}" ]]; then
    if python3 - <<PY
import sys
import torch

paths = [
    "${LOW_CALIB}",
    "${HIGH_CALIB}",
]
for path in paths:
    payload = torch.load(path, map_location="cpu")
    layers = payload.get("layers", {}) if isinstance(payload, dict) else {}
    if not isinstance(layers, dict) or len(layers) == 0:
        sys.exit(1)
    for _, layer in layers.items():
        if not isinstance(layer, dict):
            sys.exit(1)
        min_ch = layer.get("act_min_per_channel_global", None)
        max_ch = layer.get("act_max_per_channel_global", None)
        tb_min = layer.get("act_min_timestep_branch", None)
        tb_max = layer.get("act_max_timestep_branch", None)
        if not isinstance(min_ch, (list, tuple)) or len(min_ch) == 0:
            sys.exit(1)
        if not isinstance(max_ch, (list, tuple)) or len(max_ch) == 0:
            sys.exit(1)
        if not isinstance(tb_min, dict) or len(tb_min) == 0:
            sys.exit(1)
        if not isinstance(tb_max, dict) or len(tb_max) == 0:
            sys.exit(1)
sys.exit(0)
PY
    then
      echo "[CALIB] Existing calibration already contains min/max stats required by current Stage2."
    else
      echo "[CALIB][WARN] Existing calibration artifacts do not contain the current min/max schema."
      echo "[CALIB][WARN] If Stage2 uses current SQ/rotation/PTQ route, rerun Stage1 with RESUME=0."
    fi
  fi
fi

echo "[CALIB] Standard stage1 calibration starts."
echo "[CALIB] GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES} distributed_node=${CALIB_DISTRIBUTED_NODE}"
if [[ -n "${AUTO_GPU_CONFIG_SOURCE}" ]]; then
  echo "[CALIB] GPU config source: ${AUTO_GPU_CONFIG_SOURCE}"
fi
echo "[CALIB] PROMPT_FILE=${PROMPT_FILE}"
echo "[CALIB] CKPT_DIR=${CKPT_DIR}"
echo "[CALIB] SAMPLE_STEPS=${SAMPLE_STEPS} ACT_QUANT_MODE=${ACT_QUANT_MODE} SHARDS=${SHARDS}"
echo "[CALIB] CHECKPOINT_EVERY_PROMPT=${CHECKPOINT_EVERY_PROMPT} CHECKPOINT_INTERVAL_PROMPTS=${CHECKPOINT_INTERVAL_PROMPTS}"
echo "[CALIB] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[CALIB] keep_blocks is ignored in calibration (all layers participate)."
echo "[CALIB] calibration records include per-channel and timestep-branch min/max for SmoothQuant + lookup."
echo "[CALIB] official low/high timestep switching stays inside generate."
echo "${RUN_TAG}" > "${LOG_DIR}/calib30_latest.tag"
ln -sfn "${SHARD0_LOG}" "${LOG_DIR}/calib30_shard0_latest.log"
if [[ "${SHOW_SHARD1_LOG}" == "1" ]]; then
  ln -sfn "${SHARD1_LOG}" "${LOG_DIR}/calib30_shard1_latest.log"
else
  rm -f "${LOG_DIR}/calib30_shard1_latest.log"
fi
if [[ "${SHOW_MERGE_LOG}" == "1" ]]; then
  ln -sfn "${MERGE_LOG}" "${LOG_DIR}/calib30_merge_latest.log"
else
  rm -f "${LOG_DIR}/calib30_merge_latest.log"
fi
echo "[CALIB] shard0 log : ${SHARD0_LOG}"
echo "[CALIB] tail shard0: tail -f ${LOG_DIR}/calib30_shard0_latest.log"
touch "${SHARD0_LOG}"
if [[ "${SHOW_SHARD1_LOG}" == "1" ]]; then
  echo "[CALIB] shard1 log : ${SHARD1_LOG}"
  echo "[CALIB] tail shard1: tail -f ${LOG_DIR}/calib30_shard1_latest.log"
  touch "${SHARD1_LOG}"
fi
if [[ "${SHOW_MERGE_LOG}" == "1" ]]; then
  echo "[CALIB] merge log  : ${MERGE_LOG}"
  echo "[CALIB] tail merge : tail -f ${LOG_DIR}/calib30_merge_latest.log"
  touch "${MERGE_LOG}"
fi

COMMON_ARGS=(
  --hifx4_qtype hifx4
  --hifx4_quant_method ptq
  --hifx4_stage calibrate
  --ptq_offline_model all
  --ptq_calib_prompts_file "${PROMPT_FILE}"
  --ptq_calib_prompt_shards "${SHARDS}"
  --ptq_timestep_count "${SAMPLE_STEPS}"
  --ptq_act_quant_mode "${ACT_QUANT_MODE}"
  --ptq_artifact_root "${ART_ROOT}"
  --task t2v-A14B
  --size 1280*720
  --sample_steps "${SAMPLE_STEPS}"
  --ckpt_dir "${CKPT_DIR}"
)
if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--ptq_calib_resume_shard)
fi
if [[ "${CHECKPOINT_INTERVAL_PROMPTS}" =~ ^[0-9]+$ ]]; then
  if (( CHECKPOINT_INTERVAL_PROMPTS > 0 )); then
    COMMON_ARGS+=(--ptq_calib_checkpoint_interval "${CHECKPOINT_INTERVAL_PROMPTS}")
  elif [[ "${CHECKPOINT_EVERY_PROMPT}" == "1" ]]; then
    COMMON_ARGS+=(--ptq_calib_checkpoint_every_prompt)
  fi
else
  echo "[CALIB] CHECKPOINT_INTERVAL_PROMPTS must be an integer >= 0, got ${CHECKPOINT_INTERVAL_PROMPTS}"
  exit 1
fi

if [[ "${CALIB_DISTRIBUTED_NODE}" == "1" ]]; then
  echo "[CALIB] Node-distributed mode: shard ${SHARD_IDX}/${SHARDS}, GPUs=${GPU_COUNT} (single prompt uses both GPUs)."
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONUNBUFFERED=1 torchrun \
    --nnodes 1 \
    --nproc_per_node "${GPU_COUNT}" \
    --master_addr "${TORCHRUN_MASTER_ADDR}" \
    --master_port "${TORCHRUN_MASTER_PORT}" \
    "${ROOT_DIR}/generate.py" \
    "${COMMON_ARGS[@]}" \
    --ptq_calib_prompt_shard_idx "${SHARD_IDX}" \
    --t5_fsdp \
    --dit_fsdp \
    --ulysses_size "${GPU_COUNT}" \
    > "${SHARD0_LOG}" 2>&1
  echo "[CALIB] Node shard collection finished: ${SHARD_IDX}/${SHARDS}"
  echo "[CALIB] After all shards finish, run merge-only once:"
  echo "CUDA_VISIBLE_DEVICES=${FIRST_GPU} PYTHONUNBUFFERED=1 python ${ROOT_DIR}/generate.py \\"
  echo "  ${COMMON_ARGS[*]} --ptq_calib_prompt_shard_idx 0 --ptq_calib_merge_only"
elif [[ "${GPU_COUNT}" == "1" ]]; then
  if [[ -f "${ART_ROOT}/low_noise_model/hifx4/calibration.pt" && -f "${ART_ROOT}/high_noise_model/hifx4/calibration.pt" && "${RESUME}" == "1" ]]; then
    echo "[CALIB] single-gpu calibration already finished, skip rerun."
    echo "[$(date '+%F %T')] single-gpu calibration skipped (final artifacts exist)." >> "${SHARD0_LOG}"
  else
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONUNBUFFERED=1 python "${ROOT_DIR}/generate.py" \
      "${COMMON_ARGS[@]}" \
      --ptq_calib_prompt_shard_idx 0 \
      > "${SHARD0_LOG}" 2>&1
    echo "[$(date '+%F %T')] single-gpu calibration finished." >> "${SHARD0_LOG}"
  fi
else
  SHARD0_LOW_STAT="${ART_ROOT}/low_noise_model/hifx4/calibration_shards/shard_00_of_02.pt"
  SHARD0_HIGH_STAT="${ART_ROOT}/high_noise_model/hifx4/calibration_shards/shard_00_of_02.pt"
  SHARD1_LOW_STAT="${ART_ROOT}/low_noise_model/hifx4/calibration_shards/shard_01_of_02.pt"
  SHARD1_HIGH_STAT="${ART_ROOT}/high_noise_model/hifx4/calibration_shards/shard_01_of_02.pt"

  PID_S0=""
  PID_S1=""
  if [[ -f "${SHARD0_LOW_STAT}" && -f "${SHARD0_HIGH_STAT}" ]]; then
    echo "[CALIB] shard 0/2 already finished, skip rerun."
    echo "[$(date '+%F %T')] shard 0/2 skipped (checkpoint exists)." >> "${SHARD0_LOG}"
  else
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONUNBUFFERED=1 python "${ROOT_DIR}/generate.py" \
      "${COMMON_ARGS[@]}" \
      --ptq_calib_prompt_shard_idx 0 \
      > "${SHARD0_LOG}" 2>&1 &
    PID_S0=$!
  fi

  if [[ -f "${SHARD1_LOW_STAT}" && -f "${SHARD1_HIGH_STAT}" ]]; then
    echo "[CALIB] shard 1/2 already finished, skip rerun."
    echo "[$(date '+%F %T')] shard 1/2 skipped (checkpoint exists)." >> "${SHARD1_LOG}"
  else
    CUDA_VISIBLE_DEVICES="${SECOND_GPU}" PYTHONUNBUFFERED=1 python "${ROOT_DIR}/generate.py" \
      "${COMMON_ARGS[@]}" \
      --ptq_calib_prompt_shard_idx 1 \
      > "${SHARD1_LOG}" 2>&1 &
    PID_S1=$!
  fi

  if [[ -n "${PID_S0}" ]]; then
    wait "${PID_S0}"
  fi
  if [[ -n "${PID_S1}" ]]; then
    wait "${PID_S1}"
  fi

  echo "[CALIB] Prompt-shard collection finished. Start merge/finalize..."
  if [[ -f "${ART_ROOT}/low_noise_model/hifx4/calibration.pt" && -f "${ART_ROOT}/high_noise_model/hifx4/calibration.pt" && "${RESUME}" == "1" ]]; then
    echo "[CALIB] merge/finalize already finished, skip."
    echo "[$(date '+%F %T')] merge skipped (final calibration artifacts exist)." >> "${MERGE_LOG}"
  else
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" PYTHONUNBUFFERED=1 python "${ROOT_DIR}/generate.py" \
      "${COMMON_ARGS[@]}" \
      --ptq_calib_prompt_shard_idx 0 \
      --ptq_calib_merge_only \
      > "${MERGE_LOG}" 2>&1
  fi
fi

echo "[CALIB] Stage1 calibration finished."
echo "[CALIB] Logs:"
echo "  ${SHARD0_LOG}"
if [[ "${SHOW_SHARD1_LOG}" == "1" ]]; then
  echo "  ${SHARD1_LOG}"
fi
if [[ "${SHOW_MERGE_LOG}" == "1" ]]; then
  echo "  ${MERGE_LOG}"
fi
