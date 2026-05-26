#!/usr/bin/env bash
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

# Evaluate videos from an arbitrary directory in VBench custom_input mode (5 dimensions).
# This script supports resume and validates that each dimension produces *_eval_results.json.

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
if [ -n "${CONDA_SH}" ] && [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  if [ -n "${CONDA_ENV}" ]; then
    conda activate "${CONDA_ENV}"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VBENCH_DIR="${VBENCH_DIR:-${PROJECT_ROOT}/vbench}"
VIDEOS_INPUT_DIR="${VIDEOS_INPUT_DIR:-${PROJECT_ROOT}/video_output/hifx4/OpenS2V-5M_to_mm_vbench_30}"
RUN_TAG="${RUN_TAG:-vbench_hifx4_OpenS2V-5M_to_mm_vbench_30}"
RESUME="${RESUME:-1}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/eval_output}"
OUT_PATH="${OUT_PATH:-${OUT_ROOT}/${RUN_TAG}}"
STATE_ROOT="${STATE_ROOT:-${PROJECT_ROOT}/state_eval}"
STATE_DIR="${STATE_DIR:-${STATE_ROOT}/${RUN_TAG}}"
DONE_DIMS_FILE="${DONE_DIMS_FILE:-${STATE_DIR}/vbench_custom5_done_dims.txt}"
EXPECTED_VIDEO_CNT="${EXPECTED_VIDEO_CNT:-0}"
EVAL_VIDEOS_PATH="${EVAL_VIDEOS_PATH:-${STATE_DIR}/eval_videos_${RUN_TAG}}"

if [ "${RESUME}" = "0" ]; then
  echo "[INFO] RESUME=0 -> clear ${STATE_DIR} and ${OUT_PATH}"
  rm -rf "${STATE_DIR}" "${OUT_PATH}"
fi
mkdir -p "${STATE_DIR}"
touch "${DONE_DIMS_FILE}"
export VBENCH_DIR VIDEOS_INPUT_DIR RUN_TAG OUT_PATH DONE_DIMS_FILE EXPECTED_VIDEO_CNT EVAL_VIDEOS_PATH

echo "[INFO] VBENCH_DIR=${VBENCH_DIR}"
echo "[INFO] VIDEOS_INPUT_DIR=${VIDEOS_INPUT_DIR}"
echo "[INFO] RUN_TAG=${RUN_TAG}"
echo "[INFO] OUT_PATH=${OUT_PATH}"
echo "[INFO] DONE_DIMS_FILE=${DONE_DIMS_FILE}"
echo "[INFO] EXPECTED_VIDEO_CNT=${EXPECTED_VIDEO_CNT}"
echo "[INFO] EVAL_VIDEOS_PATH=${EVAL_VIDEOS_PATH}"

# NumPy 2.0 compatibility shim for imgaug/pyiqa in imaging_quality.
PYCOMPAT_DIR="${PROJECT_ROOT}/runfiles/pycompat"
if [ -d "${PYCOMPAT_DIR}" ]; then
  if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PYCOMPAT_DIR}:${PYTHONPATH}"
  else
    export PYTHONPATH="${PYCOMPAT_DIR}"
  fi
  echo "[INFO] PYTHONPATH prepend for compat: ${PYCOMPAT_DIR}"
fi

find_eval_results_for_dim() {
  local dim_name="$1"
  local start_ts="${2:-0}"
  DIM_NAME="${dim_name}" START_TS="${start_ts}" OUT_PATH="${OUT_PATH}" python - <<'PY'
import glob
import json
import os

dim = os.environ["DIM_NAME"]
out_path = os.environ["OUT_PATH"]
start_ts = float(os.environ.get("START_TS", "0"))

hits = []
for p in glob.glob(os.path.join(out_path, "results_*_eval_results.json")):
    try:
        mtime = os.path.getmtime(p)
        if mtime < start_ts:
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and dim in data:
            hits.append((mtime, p))
    except Exception:
        continue

if not hits:
    raise SystemExit(1)
hits.sort(key=lambda x: x[0])
print(hits[-1][1])
PY
}

remove_done_mark_for_dim() {
  local dim_name="$1"
  DIM_NAME="${dim_name}" DONE_DIMS_FILE="${DONE_DIMS_FILE}" python - <<'PY'
import os

dim = os.environ["DIM_NAME"]
done_file = os.environ["DONE_DIMS_FILE"]
kept = []
with open(done_file, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s or s == dim:
            continue
        kept.append(s)
with open(done_file, "w", encoding="utf-8") as f:
    for s in kept:
        f.write(s + "\n")
PY
}

python - <<'PY'
import os
import shutil

src = os.environ["VIDEOS_INPUT_DIR"]
dst = os.environ["EVAL_VIDEOS_PATH"]
exts = (".mp4", ".gif", ".webm", ".avi", ".mov", ".mkv")

if not os.path.isdir(src):
    raise RuntimeError(f"Missing videos input directory: {src}")

os.makedirs(dst, exist_ok=True)
for name in os.listdir(dst):
    p = os.path.join(dst, name)
    if os.path.islink(p) or os.path.isfile(p):
        os.remove(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)

collected = 0
for dp, _, fs in os.walk(src):
    for fname in sorted(fs):
        if not fname.lower().endswith(exts):
            continue
        src_path = os.path.join(dp, fname)
        rel = os.path.relpath(src_path, src).replace("/", "__")
        link_name = os.path.join(dst, rel)
        if os.path.exists(link_name):
            base, ext = os.path.splitext(rel)
            i = 1
            while os.path.exists(os.path.join(dst, f"{base}__{i}{ext}")):
                i += 1
            link_name = os.path.join(dst, f"{base}__{i}{ext}")
        os.symlink(src_path, link_name)
        collected += 1
print(f"[INFO] collected_videos={collected}")
PY

VIDEO_CNT="$(python - <<'PY'
import os
root = os.environ["EVAL_VIDEOS_PATH"]
exts = (".mp4", ".gif", ".webm", ".avi", ".mov", ".mkv")
c = 0
for _, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(exts):
            c += 1
print(c)
PY
)"

echo "[INFO] detected videos: ${VIDEO_CNT}"
if [ "${VIDEO_CNT}" -eq 0 ]; then
  echo "[ERROR] No video files found under ${EVAL_VIDEOS_PATH}"
  exit 2
fi
if [ "${EXPECTED_VIDEO_CNT}" -gt 0 ] && [ "${VIDEO_CNT}" -ne "${EXPECTED_VIDEO_CNT}" ]; then
  echo "[ERROR] Expected ${EXPECTED_VIDEO_CNT} videos but found ${VIDEO_CNT} under ${EVAL_VIDEOS_PATH}"
  exit 3
fi

# For motion_smoothness with torch>=2.6 behavior change.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "${OUT_PATH}"
cd "${VBENCH_DIR}"

for DIM in \
  imaging_quality \
  aesthetic_quality \
  overall_consistency \
  subject_consistency \
  motion_smoothness
do
  if grep -Fxq "${DIM}" "${DONE_DIMS_FILE}"; then
    if RESULT_PATH="$(find_eval_results_for_dim "${DIM}" "0")"; then
      echo "[$(date)] skip ${DIM} (already done, result=${RESULT_PATH})"
      continue
    fi
    echo "[$(date)] warn ${DIM} marked done but no eval_results found; remove stale mark and rerun"
    remove_done_mark_for_dim "${DIM}"
  fi

  DIM_START_TS="$(python - <<'PY'
import time
print(time.time())
PY
)"

  echo "[$(date)] start ${DIM}"
  if ! vbench evaluate \
    --videos_path "${EVAL_VIDEOS_PATH}" \
    --dimension "${DIM}" \
    --mode custom_input \
    --output_path "${OUT_PATH}"
  then
    echo "[$(date)] error ${DIM} vbench evaluate failed"
    exit 10
  fi

  if ! RESULT_PATH="$(find_eval_results_for_dim "${DIM}" "${DIM_START_TS}")"; then
    echo "[$(date)] error ${DIM} produced no matching *_eval_results.json"
    echo "[$(date)] hint  full_info may exist but eval_results missing; keeping ${DIM} pending"
    exit 11
  fi

  echo "[$(date)] result ${DIM}: ${RESULT_PATH}"
  echo "${DIM}" >> "${DONE_DIMS_FILE}"
  echo "[$(date)] done  ${DIM}"
done

echo "[DONE] VBench custom_input 5-dim evaluation completed."
