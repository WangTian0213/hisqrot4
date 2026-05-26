#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ART_ROOT="${ART_ROOT:-${ROOT_DIR}/state_quant/hif4_ptq}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/models/Wan2.2-T2V-A14B}"
LOG_DIR="${ROOT_DIR}/logs"
OUT_DIR="${ROOT_DIR}/video_output"
GPU_COUNT="${GPU_COUNT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-}"
RESUME="${RESUME:-1}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-true}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"
INFER_ENGINE="${INFER_ENGINE:-python}"
PROMPT="${PROMPT:-a cat surfing on the sea}"
if [[ "${PROMPT_FILE+x}" == "x" ]]; then
  PROMPT_FILE="${PROMPT_FILE}"
else
  PROMPT_FILE="${ROOT_DIR}/data/prompts/OpenS2V-5M_to_mm_vbench_30.json"
fi
PROMPT_INDEX="${PROMPT_INDEX:-0}"
OUTPUT_FILE_DEFAULT="${OUT_DIR}/ptq_hifx4_standard.mp4"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_FILE_DEFAULT}}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
KEEP_BLOCKS="${KEEP_BLOCKS:-}"
ACT_QUANT_MODE="${ACT_QUANT_MODE:-lookup}"
FRAME_NUM="${FRAME_NUM:-61}"
EXPECT_SMOOTHQUANT="${EXPECT_SMOOTHQUANT:-0}"

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

if [[ -z "${CUDA_DEVICES}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
fi
CUDA_DEVICES="$(printf '%s' "${CUDA_DEVICES}" | tr -d '[:space:]')"

if [[ -n "${CUDA_DEVICES}" ]]; then
  VISIBLE_GPU_COUNT="$(_detect_gpu_count_with_visible_devices "${CUDA_DEVICES}")"
  if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${VISIBLE_GPU_COUNT}" -le 0 ]]; then
    echo "[INFER] No visible GPU from CUDA_DEVICES=${CUDA_DEVICES}"
    exit 1
  fi
else
  DETECTED_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
  if ! [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${DETECTED_GPU_COUNT}" -le 0 ]]; then
    echo "[INFER] Failed to auto-detect visible GPUs."
    exit 1
  fi
  VISIBLE_GPU_COUNT="${DETECTED_GPU_COUNT}"
  CUDA_DEVICES="$(_make_default_cuda_devices "${VISIBLE_GPU_COUNT}")"
fi

if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="${VISIBLE_GPU_COUNT}"
else
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "[INFER] GPU_COUNT must be a positive integer, got ${GPU_COUNT}"
    exit 1
  fi
  if [[ "${GPU_COUNT}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
    echo "[INFER] WARN: requested GPU_COUNT=${GPU_COUNT}, but visible only ${VISIBLE_GPU_COUNT} (CUDA_DEVICES=${CUDA_DEVICES})."
    GPU_COUNT="${VISIBLE_GPU_COUNT}"
  fi
fi
FIRST_GPU="${CUDA_DEVICES%%,*}"

if [[ "${INFER_ENGINE}" != "python" ]]; then
  echo "[INFER] PTQ route only supports INFER_ENGINE=python. got=${INFER_ENGINE}"
  exit 1
fi
if [[ -n "${PROMPT_FILE}" && ! -f "${PROMPT_FILE}" ]]; then
  echo "[INFER] PROMPT_FILE not found: ${PROMPT_FILE}"
  exit 1
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  PROMPT_JSON="$(python3 - <<PY
import json
import os
idx = int("${PROMPT_INDEX}")
with open("${PROMPT_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise RuntimeError("PROMPT_FILE must be json list")
if idx < 0 or idx >= len(data):
    raise RuntimeError(f"PROMPT_INDEX out of range: {idx}, len={len(data)}")
item = data[idx]
if not isinstance(item, dict):
    raise RuntimeError("Prompt item must be dict")
prompt = item.get("prompt")
if not isinstance(prompt, str) or not prompt.strip():
    prompt = item.get("cap")
prompt = str(prompt).strip() if prompt is not None else ""
if not prompt:
    raise RuntimeError("Prompt item missing valid prompt/cap")
path = item.get("path")
if isinstance(path, str) and path.strip():
    output_name = os.path.basename(path.strip())
else:
    output_name = f"item_{idx:04d}"
if not os.path.splitext(output_name)[1]:
    output_name = f"{output_name}.mp4"
print(json.dumps({"prompt": prompt, "output_name": output_name}))
PY
)"
  PROMPT="$(python3 - <<PY
import json
obj = json.loads('''${PROMPT_JSON}''')
print(obj["prompt"])
PY
)"
  if [[ "${OUTPUT_FILE}" == "${OUTPUT_FILE_DEFAULT}" ]]; then
    OUTPUT_NAME="$(python3 - <<PY
import json
obj = json.loads('''${PROMPT_JSON}''')
print(obj["output_name"])
PY
)"
    OUTPUT_NAME="$(printf '%s' "${OUTPUT_NAME}" | tr '/' '_')"
    OUTPUT_FILE="${OUT_DIR}/${OUTPUT_NAME}"
  fi
fi

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
INFER_LOG="${LOG_DIR}/infer_ptq_${RUN_TAG}.log"

if pgrep -f "torchrun --nproc_per_node=[0-9][0-9]* .*generate.py .*--hifx4_stage infer" >/dev/null 2>&1; then
  echo "[INFER] Found stale infer processes. Killing them first..."
  pkill -9 -f "torchrun --nproc_per_node=[0-9][0-9]* .*generate.py .*--hifx4_stage infer" || true
  sleep 1
fi
if pgrep -f "${ROOT_DIR}/generate.py .*--hifx4_stage infer" >/dev/null 2>&1; then
  echo "[INFER] Found stale single-process infer. Killing it first..."
  pkill -9 -f "${ROOT_DIR}/generate.py .*--hifx4_stage infer" || true
  sleep 1
fi
LOW_PREP="${ART_ROOT}/low_noise_model/hifx4/prepared.pt"
HIGH_PREP="${ART_ROOT}/high_noise_model/hifx4/prepared.pt"
if [[ ! -f "${LOW_PREP}" || ! -f "${HIGH_PREP}" ]]; then
  echo "[INFER] Missing Stage2 prepared artifacts."
  echo "[INFER] Need both: ${LOW_PREP} and ${HIGH_PREP}"
  exit 1
fi
LOW_MANIFEST="${ART_ROOT}/low_noise_model/hifx4/manifest.json"
HIGH_MANIFEST="${ART_ROOT}/high_noise_model/hifx4/manifest.json"

echo "[INFER] Artifact summary (manifest):"
if ! python3 - <<PY
import json
import os
import sys

pairs = [
    ("low", "${LOW_MANIFEST}"),
    ("high", "${HIGH_MANIFEST}"),
]
expect_sq = str("${EXPECT_SMOOTHQUANT}").strip().lower() in ("1", "true", "yes")
sq_any = False

for tag, path in pairs:
    if not os.path.isfile(path):
        print(f"[INFER][WARN] {tag} manifest not found: {path}")
        continue
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[INFER][WARN] failed to read {tag} manifest: {exc}")
        continue

    sq_layers = data.get("smoothquant_applied_layers", -1)
    en_sq = data.get("enable_smoothquant", "n/a")
    alpha = data.get("smoothquant_alpha", "n/a")
    eps = data.get("smoothquant_eps", "n/a")
    rot_layers = data.get("rotation_merged_layers", -1)
    weight_format = data.get("weight_format", "n/a")
    lookup_stat = data.get("lookup_stat", "n/a")
    activation_grouping = data.get("activation_grouping", "n/a")
    print(
        f"[INFER][MANIFEST] {tag}: smoothquant_applied_layers={sq_layers}, "
        f"enable_smoothquant={en_sq}, smoothquant_alpha={alpha}, smoothquant_eps={eps}, "
        f"rotation_merged_layers={rot_layers}, weight_format={weight_format}, "
        f"lookup_stat={lookup_stat}, activation_grouping={activation_grouping}"
    )
    if isinstance(sq_layers, int) and sq_layers > 0:
        sq_any = True

if expect_sq and not sq_any:
    print("[INFER][ERROR] EXPECT_SMOOTHQUANT=1 but manifest does not show smoothquant_applied_layers>0.")
    sys.exit(2)
sys.exit(0)
PY
then
  exit 1
fi

echo "${RUN_TAG}" > "${LOG_DIR}/infer_ptq_latest.tag"
ln -sfn "${INFER_LOG}" "${LOG_DIR}/infer_ptq_latest.log"
echo "[INFER] log   : ${INFER_LOG}"
echo "[INFER] video : ${OUTPUT_FILE}"
echo "[INFER] tail  : tail -f ${LOG_DIR}/infer_ptq_latest.log"
echo "[INFER] GPU_COUNT=${GPU_COUNT} CUDA_DEVICES=${CUDA_DEVICES}"
echo "[INFER] CKPT_DIR=${CKPT_DIR}"
echo "[INFER] OFFLOAD_MODEL=${OFFLOAD_MODEL}"
echo "[INFER] SAMPLE_STEPS=${SAMPLE_STEPS} ACT_QUANT_MODE=${ACT_QUANT_MODE} KEEP_BLOCKS=${KEEP_BLOCKS:-<none>}"
if [[ "${ACT_QUANT_MODE}" == "lookup" ]]; then
  echo "[INFER] lookup mode consumes Stage2 grouped activation min-max tables from ${ART_ROOT}"
fi
echo "[INFER] FRAME_NUM=${FRAME_NUM} (fixed for all prompts)"
echo "[INFER] EXPECT_SMOOTHQUANT=${EXPECT_SMOOTHQUANT} (1 means require SQ-prepared artifacts)"
echo "[INFER] PROMPT_FILE=${PROMPT_FILE:-<none>} PROMPT_INDEX=${PROMPT_INDEX}"
echo "[INFER] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
echo "[INFER] INFER_ENGINE=${INFER_ENGINE}"
touch "${INFER_LOG}"

if [[ "${RESUME}" == "1" && -f "${OUTPUT_FILE}" ]]; then
  echo "[INFER] output already exists, skip rerun: ${OUTPUT_FILE}"
  echo "[$(date '+%F %T')] infer skipped (output exists)." >> "${INFER_LOG}"
  exit 0
fi

EXTRA_ARGS=(
  --sample_steps "${SAMPLE_STEPS}"
  --ptq_timestep_count "${SAMPLE_STEPS}"
  --ptq_act_quant_mode "${ACT_QUANT_MODE}"
)
if [[ -n "${KEEP_BLOCKS}" ]]; then
  EXTRA_ARGS+=(--ptq_keep_blocks "${KEEP_BLOCKS}")
fi

if [[ "${GPU_COUNT}" == "1" ]]; then
  echo "[INFER] Stage3 starts (single GPU)..."
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
    --size 1280*720 \
    --ckpt_dir "${CKPT_DIR}" \
    --offload_model "${OFFLOAD_MODEL}" \
    --frame_num "${FRAME_NUM}" \
    --prompt "${PROMPT}" \
    --save_file "${OUTPUT_FILE}" \
    "${EXTRA_ARGS[@]}" \
    > "${INFER_LOG}" 2>&1
else
  echo "[INFER] Stage3 starts (multi-GPU torchrun, nproc=${GPU_COUNT})..."
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}" \
  torchrun --nproc_per_node="${GPU_COUNT}" "${ROOT_DIR}/generate.py" \
    --hifx4_qtype hifx4 \
    --hifx4_quant_method ptq \
    --hifx4_stage infer \
    --ptq_infer_engine "${INFER_ENGINE}" \
    --ptq_validate_artifact \
    --ptq_artifact_root "${ART_ROOT}" \
    --task t2v-A14B \
    --size 1280*720 \
    --ckpt_dir "${CKPT_DIR}" \
    --offload_model "${OFFLOAD_MODEL}" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size "${GPU_COUNT}" \
    --frame_num "${FRAME_NUM}" \
    --prompt "${PROMPT}" \
    --save_file "${OUTPUT_FILE}" \
    "${EXTRA_ARGS[@]}" \
    > "${INFER_LOG}" 2>&1
fi

if [[ ! -f "${OUTPUT_FILE}" ]]; then
  echo "[INFER] Stage3 finished but output file not found: ${OUTPUT_FILE}"
  exit 1
fi

echo "[INFER] Stage3 finished."
echo "[INFER] output: ${OUTPUT_FILE}"
