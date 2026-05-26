#!/usr/bin/env bash
set -euo pipefail

# Batch inference from a custom prompt JSON list.
# JSON format: [{"prompt":"...", "path":"videos/name.mp4", ...}, ...]
# Outputs videos to: video_output/<qtype>/<out_subdir>/<basename(path)>.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ART_ROOT="${ART_ROOT:-${ROOT_DIR}/state_quant/hif4_ptq}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/models/Wan2.2-T2V-A14B}"
VIDEOS_ROOT="${ROOT_DIR}/video_output"
QTYPE="${QTYPE:-hifx4}"

PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/data/prompts/OpenS2V-5M_to_mm_vbench_30.json}"
FIRST_N="${FIRST_N:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_FOLDER="${OUT_FOLDER:-}"
if [[ -n "${OUT_FOLDER}" ]]; then
  if [[ "${OUT_FOLDER}" = /* ]]; then
    OUT_DIR="${OUT_FOLDER}"
  else
    OUT_DIR="${ROOT_DIR}/${OUT_FOLDER}"
  fi
elif [[ -n "${OUT_SUBDIR:-}" ]]; then
  OUT_SUBDIR="${OUT_SUBDIR}"
  OUT_DIR="${VIDEOS_ROOT}/${QTYPE}/${OUT_SUBDIR}"
else
  PROMPT_BASENAME="$(basename "${PROMPT_FILE}")"
  PROMPT_STEM="${PROMPT_BASENAME%.*}"
  SAFE_PROMPT_STEM="$(printf '%s' "${PROMPT_STEM}" | tr '/ ' '__')"
  OUT_SUBDIR="${SAFE_PROMPT_STEM}_${RUN_TS}"
  OUT_DIR="${VIDEOS_ROOT}/${QTYPE}/${OUT_SUBDIR}"
fi

GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
RESUME="${RESUME:-1}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
NODE_SHARDS="${NODE_SHARDS:-1}"
NODE_SHARD_IDX="${NODE_SHARD_IDX:-0}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
INFER_ENGINE="${INFER_ENGINE:-python}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
KEEP_BLOCKS="${KEEP_BLOCKS:-}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
FRAME_NUM="${FRAME_NUM:-61}"

_make_default_cuda_devices() {
  local count="${1}"
  local idxs=()
  local i
  for ((i=0; i<count; i++)); do
    idxs+=("${i}")
  done
  local joined=""
  local item
  for item in "${idxs[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${item}"
  done
  printf '%s' "${joined}"
}

_detect_gpu_count_with_visible_devices() {
  local visible="${1}"
  CUDA_VISIBLE_DEVICES="${visible}" python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "[CUSTOM-BATCH] prompt file not found: ${PROMPT_FILE}"
  exit 1
fi
if [[ -n "${FIRST_N}" ]] && (! [[ "${FIRST_N}" =~ ^[0-9]+$ ]] || [[ "${FIRST_N}" -le 0 ]]); then
  echo "[CUSTOM-BATCH] FIRST_N must be empty or a positive integer, got ${FIRST_N}"
  exit 1
fi
if [[ -z "${CUDA_DEVICES}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
CUDA_DEVICES_CLEAN="$(printf '%s' "${CUDA_DEVICES}" | tr -d '[:space:]')"
if [[ -n "${CUDA_DEVICES_CLEAN}" ]]; then
  VISIBLE_GPU_COUNT="$(_detect_gpu_count_with_visible_devices "${CUDA_DEVICES_CLEAN}")"
  if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${VISIBLE_GPU_COUNT}" -le 0 ]]; then
    echo "[CUSTOM-BATCH] No visible GPU from CUDA_DEVICES=${CUDA_DEVICES_CLEAN}"
    exit 1
  fi
else
  DETECTED_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
  if ! [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${DETECTED_GPU_COUNT}" -le 0 ]]; then
    echo "[CUSTOM-BATCH] Failed to auto-detect visible GPUs."
    exit 1
  fi
  VISIBLE_GPU_COUNT="${DETECTED_GPU_COUNT}"
  CUDA_DEVICES_CLEAN="$(_make_default_cuda_devices "${VISIBLE_GPU_COUNT}")"
fi
if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="${VISIBLE_GPU_COUNT}"
else
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "[CUSTOM-BATCH] GPU_COUNT must be a positive integer, got ${GPU_COUNT}"
    exit 1
  fi
  if [[ "${GPU_COUNT}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
    echo "[CUSTOM-BATCH] WARN: requested GPU_COUNT=${GPU_COUNT}, but visible only ${VISIBLE_GPU_COUNT} (CUDA_DEVICES=${CUDA_DEVICES_CLEAN})."
    GPU_COUNT="${VISIBLE_GPU_COUNT}"
  fi
fi
CUDA_DEVICES="${CUDA_DEVICES_CLEAN}"
FIRST_GPU="${CUDA_DEVICES%%,*}"
if ! [[ "${NODE_SHARDS}" =~ ^[0-9]+$ ]] || [[ "${NODE_SHARDS}" -le 0 ]]; then
  echo "[CUSTOM-BATCH] NODE_SHARDS must be a positive integer, got ${NODE_SHARDS}"
  exit 1
fi
if ! [[ "${NODE_SHARD_IDX}" =~ ^[0-9]+$ ]] || (( NODE_SHARD_IDX < 0 || NODE_SHARD_IDX >= NODE_SHARDS )); then
  echo "[CUSTOM-BATCH] NODE_SHARD_IDX must be in [0, NODE_SHARDS), got idx=${NODE_SHARD_IDX}, shards=${NODE_SHARDS}"
  exit 1
fi
if [[ "${INFER_ENGINE}" != "python" ]]; then
  echo "[CUSTOM-BATCH] PTQ route only supports INFER_ENGINE=python. got=${INFER_ENGINE}"
  exit 1
fi

LOW_PREP="${ART_ROOT}/low_noise_model/hifx4/prepared.pt"
HIGH_PREP="${ART_ROOT}/high_noise_model/hifx4/prepared.pt"
if [[ ! -f "${LOW_PREP}" || ! -f "${HIGH_PREP}" ]]; then
  echo "[CUSTOM-BATCH] Missing Stage2 prepared artifacts."
  echo "[CUSTOM-BATCH] Need both: ${LOW_PREP} and ${HIGH_PREP}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

CURRENT_LAUNCH_PGID=""
BATCH_INTERRUPTED=0

cleanup_launch_group() {
  local pgid="${1:-}"
  if [[ -z "${pgid}" ]]; then
    return 0
  fi
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-${pgid}" 2>/dev/null || true
}

cleanup_current_launch_group() {
  cleanup_launch_group "${CURRENT_LAUNCH_PGID}"
  CURRENT_LAUNCH_PGID=""
}

handle_interrupt() {
  BATCH_INTERRUPTED=1
  echo "[CUSTOM-BATCH] Interrupted; stopping batch inference." >&2
  cleanup_current_launch_group
  exit 130
}

trap 'handle_interrupt' INT TERM
trap 'cleanup_current_launch_group' EXIT

PROMPT_COUNT="$(python3 - <<PY
import json
with open("${PROMPT_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise RuntimeError("Prompt file must be a JSON list.")
print(len(data))
PY
)"
TARGET_N="${FIRST_N:-${PROMPT_COUNT}}"
if [[ "${TARGET_N}" -gt "${PROMPT_COUNT}" ]]; then
  TARGET_N="${PROMPT_COUNT}"
fi
SELECTED_N="$(python3 - <<PY
target_n = int("${TARGET_N}")
node_shards = int("${NODE_SHARDS}")
node_shard_idx = int("${NODE_SHARD_IDX}")
cnt = 0
for i in range(target_n):
    if i % node_shards == node_shard_idx:
        cnt += 1
print(cnt)
PY
)"

echo "============================================================"
echo "[CUSTOM-BATCH] Prompt file : ${PROMPT_FILE}"
echo "[CUSTOM-BATCH] RUN_TS      : ${RUN_TS}"
echo "[CUSTOM-BATCH] Total prompts: ${PROMPT_COUNT}"
echo "[CUSTOM-BATCH] Selected    : ${TARGET_N}${FIRST_N:+ (FIRST_N=${FIRST_N})}"
echo "[CUSTOM-BATCH] Shard split : NODE_SHARDS=${NODE_SHARDS} NODE_SHARD_IDX=${NODE_SHARD_IDX} selected=${SELECTED_N}"
echo "[CUSTOM-BATCH] Output dir  : ${OUT_DIR}"
echo "[CUSTOM-BATCH] GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES}"
echo "[CUSTOM-BATCH] SAMPLE_STEPS=${SAMPLE_STEPS} ACT_QUANT_MODE=${ACT_QUANT_MODE} KEEP_BLOCKS=${KEEP_BLOCKS:-<none>}"
echo "[CUSTOM-BATCH] INFER_ENGINE=${INFER_ENGINE} RESUME=${RESUME}"
echo "============================================================"

TOTAL=0
SKIPPED=0
DONE=0
FAILED=0

EXTRA_ARGS=(
  --sample_steps "${SAMPLE_STEPS}"
  --ptq_timestep_count "${SAMPLE_STEPS}"
  --ptq_act_quant_mode "${ACT_QUANT_MODE}"
)
if [[ -n "${KEEP_BLOCKS}" ]]; then
  EXTRA_ARGS+=(--ptq_keep_blocks "${KEEP_BLOCKS}")
fi

for IDX in $(seq 0 $((TARGET_N - 1))); do
  if (( IDX % NODE_SHARDS != NODE_SHARD_IDX )); then
    continue
  fi
  TOTAL=$((TOTAL + 1))
  OUTPUT_NAME="$(python3 - <<PY
import json
import os
with open("${PROMPT_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)
item = data[${IDX}]
path = item.get("path")
if isinstance(path, str) and path.strip():
    name = os.path.basename(path.strip())
else:
    name = f"item_${IDX+1:04d}"
if not os.path.splitext(name)[1]:
    name = f"{name}.mp4"
print(name)
PY
)"
  PROMPT_TEXT="$(python3 - <<PY
import json
with open("${PROMPT_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)
item = data[${IDX}]
prompt = item.get("prompt")
if not isinstance(prompt, str) or not prompt.strip():
    prompt = item.get("cap")
print(prompt.strip() if isinstance(prompt, str) else "")
PY
)"
  OUTPUT_NAME="$(printf '%s' "${OUTPUT_NAME}" | tr '/' '_')"
  OUTPUT_FILE="${OUT_DIR}/${OUTPUT_NAME}"

  if [[ "${RESUME}" == "1" && -f "${OUTPUT_FILE}" ]]; then
    SKIPPED=$((SKIPPED + 1))
    echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] SKIP (exists): ${OUTPUT_FILE}"
    continue
  fi

  RUN_OFFLOAD_MODEL="${OFFLOAD_MODEL}"

  INFER_OK=0
  if [[ "${GPU_COUNT}" == "1" ]]; then
    echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] generating: ${OUTPUT_NAME} (frames=${FRAME_NUM}, offload_model=${RUN_OFFLOAD_MODEL})"
    set +e
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}" \
      python "${ROOT_DIR}/generate.py" \
        --hifx4_qtype hifx4 \
        --hifx4_quant_method ptq \
        --hifx4_stage infer \
        --ptq_infer_engine "${INFER_ENGINE}" \
        --ptq_validate_artifact \
        --ptq_artifact_root "${ART_ROOT}" \
        --task t2v-A14B \
        --size "1280*720" \
        --ckpt_dir "${CKPT_DIR}" \
        --offload_model "${RUN_OFFLOAD_MODEL}" \
        --frame_num "${FRAME_NUM}" \
        --prompt "${PROMPT_TEXT}" \
        --save_file "${OUTPUT_FILE}" \
        "${EXTRA_ARGS[@]}"
    launch_status=$?
    set -e
    if [[ "${BATCH_INTERRUPTED}" == "1" || "${launch_status}" == "130" || "${launch_status}" == "143" ]]; then
      echo "[CUSTOM-BATCH] Interrupted during ${OUTPUT_NAME}; stopping batch inference." >&2
      exit 130
    fi
    if [[ "${launch_status}" == "0" ]]; then
      INFER_OK=1
    fi
  else
    echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] generating: ${OUTPUT_NAME} (frames=${FRAME_NUM}, offload_model=${RUN_OFFLOAD_MODEL})"
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}" \
      setsid torchrun --nproc_per_node="${GPU_COUNT}" "${ROOT_DIR}/generate.py" \
        --hifx4_qtype hifx4 \
        --hifx4_quant_method ptq \
        --hifx4_stage infer \
        --ptq_infer_engine "${INFER_ENGINE}" \
        --ptq_validate_artifact \
        --ptq_artifact_root "${ART_ROOT}" \
        --task t2v-A14B \
        --size "1280*720" \
        --ckpt_dir "${CKPT_DIR}" \
        --offload_model "${RUN_OFFLOAD_MODEL}" \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size "${GPU_COUNT}" \
        --frame_num "${FRAME_NUM}" \
        --prompt "${PROMPT_TEXT}" \
        --save_file "${OUTPUT_FILE}" \
        "${EXTRA_ARGS[@]}" &
    CURRENT_LAUNCH_PGID="$!"
    wait "${CURRENT_LAUNCH_PGID}"
    launch_status=$?
    set -e
    if [[ "${BATCH_INTERRUPTED}" == "1" || "${launch_status}" == "130" || "${launch_status}" == "143" ]]; then
      cleanup_current_launch_group
      echo "[CUSTOM-BATCH] Interrupted during ${OUTPUT_NAME}; stopping batch inference." >&2
      exit 130
    fi
    if [[ "${launch_status}" == "0" ]]; then
      INFER_OK=1
      CURRENT_LAUNCH_PGID=""
    else
      echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] cleanup stale children for failed launch pgid=${CURRENT_LAUNCH_PGID}"
      cleanup_current_launch_group
    fi
  fi

  if [[ "${INFER_OK}" == "1" && -f "${OUTPUT_FILE}" ]]; then
    DONE=$((DONE + 1))
    echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] OK: ${OUTPUT_FILE}"
  else
    FAILED=$((FAILED + 1))
    echo "[CUSTOM-BATCH] [${IDX}/${TARGET_N}] FAILED: ${OUTPUT_NAME}"
  fi
done

echo "============================================================"
echo "[CUSTOM-BATCH] Finished."
echo "[CUSTOM-BATCH] selected=${SELECTED_N} attempted=${TOTAL} done=${DONE} skipped=${SKIPPED} failed=${FAILED}"
echo "[CUSTOM-BATCH] output dir: ${OUT_DIR}"
echo "============================================================"

if [[ "${FAILED}" -gt 0 ]]; then
  echo "[CUSTOM-BATCH] WARNING: ${FAILED} video(s) failed. Re-run with RESUME=1 to retry."
  exit 1
fi
