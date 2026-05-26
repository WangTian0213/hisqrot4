---
license: apache-2.0
pipeline_tag: text-to-video
tags:
  - quantization
  - post-training-quantization
  - text-to-video
  - wan2.2
  - fp4
  - w4a4
---

# HiSQRot4

**HiSQRot4: SmoothQuant-Rotation HiFloat4 PTQ for W4A4 Text-to-Video Diffusion Models**

HiSQRot4 is a post-training quantization project for Wan2.2 text-to-video inference. It keeps the original Wan2.2 denoising path intact and replaces target `Linear` layers with a W4A4 HiFloat4 inference path.

## Table of Contents

1. [Method Overview](#1-method-overview)
2. [Environment Setup](#2-environment-setup)
3. [Download Model Weights and Quantized Artifacts](#3-download-model-weights-and-quantized-artifacts)
4. [Inference with Released Quantized Weights](#4-inference-with-released-quantized-weights)
5. [Reproducing the Quantization Pipeline](#5-reproducing-the-quantization-pipeline)
6. [VBench Evaluation](#6-vbench-evaluation)
7. [Repository Layout](#7-repository-layout)
8. [Acknowledgements](#8-acknowledgements)

## 1. Method Overview

HiSQRot4 uses a three-stage post-training quantization pipeline for Wan2.2 text-to-video inference.

- **Stage 1: Calibration** collects per-layer activation min/max statistics at group, branch, and channel granularity.
- **Stage 2: Quantized artifact preparation** builds SmoothQuant channel masks, applies a Hadamard-style rotation matrix to the input channel space, quantizes target weights to HiFloat4, and builds MinMax lookup ranges for runtime activation quantization.
- **Stage 3: Inference** loads the prepared artifacts and runs Wan2.2 generation with **all `Linear` layers in every transformer block** replaced by the HiFloat4 W4A4 path.

| Component         | Role                                                                  |
| ----------------- | --------------------------------------------------------------------- |
| HiFloat4          | 4-bit floating-point W4A4 inference path                              |
| MinMax lookup     | Offline activation range lookup for Stage 3 activation quantization   |
| SmoothQuant       | Channel scaling derived from activation min/max and weight magnitudes |
| Hadamard rotation | Input channel rotation with a deterministic Hadamard-style matrix folded into the prepared weight path |
| `alpha=0.9`       | SmoothQuant alpha used for the released Stage 2 artifacts             |

This GitHub repository contains the code, scripts, prompts, configuration files,
and lightweight manifests needed to run HiSQRot4. Large model files are hosted on
Hugging Face and should be downloaded after cloning:

- Original Wan2.2-T2V-A14B weights: `Wan-AI/Wan2.2-T2V-A14B`.
- Prebuilt HiSQRot4 alpha=0.9 Stage 2 artifacts: `BitsPlease/HiSQRot4`.
- The 30-prompt VBench input file is included at `data/prompts/OpenS2V-5M_to_mm_vbench_30.json`.

VBench evaluation results:

| Model                 | Image quality | Aesthetic quality | Overall consistency | Subject consistency | Motion smoothness |
| --------------------- | ------------- | ----------------- | ------------------- | ------------------- | ----------------- |
| wan2.2 original       | 71.53%        | 59.03%            | 8.45%               | 95.40%              | 98.92%            |
| wan2.2 W4A4 quantized | 73.06%        | 58.98%            | 8.55%               | 96.12%              | 98.83%            |


## 2. Environment Setup

The released artifacts were validated with the following runtime:

```text
python 3.10.20
torch 2.10.0+cu128
torchvision 0.25.0+cu128
torchaudio 2.10.0+cu128
triton 3.6.0
flash-attn 2.8.3
```

Create the conda environment and install the pinned PyTorch stack first:

```bash
conda create -n hisqrot4 python=3.10 -y
conda activate hisqrot4

pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining runtime dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes a prebuilt `flash-attn` wheel for Linux x86_64,
Python 3.10, and PyTorch 2.10.0+cu128. This avoids multi-hour local source
builds. If you use a different Python, PyTorch, CUDA, or platform combination,
install a matching `flash-attn` wheel or build it from source.

Build the HiFloat4 CUDA extension from the repository root:

```bash
cd hifloat4/hifx4_gpu
bash build.sh
cd ../..
```

## 3. Download Model Weights and Quantized Artifacts

This GitHub repository does not track the original Wan2.2 weights or the
prepared HiSQRot4 `prepared.pt` files. Download them from Hugging Face after
installing the dependencies.

Download the original Wan2.2-T2V-A14B checkpoint:

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B \
  --local-dir models/Wan2.2-T2V-A14B
```

Download the released HiSQRot4 alpha=0.9 artifacts:

```bash
huggingface-cli download BitsPlease/HiSQRot4 \
  --include "artifacts/hisqrot4_alpha_0p9/**" \
  --local-dir .
```

After both downloads, the following files should exist:

```text
models/Wan2.2-T2V-A14B/
  Wan2.1_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
  low_noise_model/diffusion_pytorch_model.safetensors.index.json
  high_noise_model/diffusion_pytorch_model.safetensors.index.json

artifacts/hisqrot4_alpha_0p9/
  low_noise_model/hifx4/prepared.pt
  low_noise_model/hifx4/manifest.json
  high_noise_model/hifx4/prepared.pt
  high_noise_model/hifx4/manifest.json
```

The `.gitignore` file intentionally ignores these large downloaded weight files
so they are not accidentally committed to GitHub.

## 4. Inference with Released Quantized Weights

Use this section after downloading the original Wan2.2 weights and released
HiSQRot4 artifacts. You do **not** need to run Stage 1 or Stage 2 when using the
released alpha=0.9 artifacts.


### 4.1 Single-Prompt Inference

Run Stage 3 W4A4 inference with the released alpha=0.9 artifacts:

```bash
GPU_COUNT=1 \
PROMPT="A cinematic shot of a cat surfing on the sea." \
OUTPUT_FILE="video_output/hifx4/single_prompt_alpha0p9.mp4" \
bash runfiles/05_infer_single_prompt.sh
```

Replace the `PROMPT` line with your own text prompt for interactive testing.

### 4.2 Batch Inference from a Prompt File

Override the defaults when needed:

```bash
GPU_COUNT=1 \
PROMPT_FILE="data/prompts/OpenS2V-5M_to_mm_vbench_30.json" \
OUT_FOLDER="video_output/hifx4/OpenS2V-5M_to_mm_vbench_30_alpha0p9" \
bash runfiles/04_infer_hisqrot4_alpha0p9_vbench30.sh
```

For prompt files with a `path` field, each generated video uses the basename of
that path as its output filename, for example `videos/example.mp4` becomes
`example.mp4`.

## 5. Reproducing the Quantization Pipeline

Use this section if you want to rebuild the calibration statistics and quantized artifacts yourself. The released alpha=0.9 quick start in Section 4 does not need these steps.

### 5.1 Stage 1: Calibration

Stage 1 runs the original model and records activation min/max statistics for target `Linear` layers.

```bash
PROMPT_FILE="data/prompts/OpenS2V-5M_to_mm_calib_30.json" \
ART_ROOT="state_quant/hisqrot4_ptq" \
bash runfiles/01_calibrate_ptq_standard.sh
```

Expected outputs:

```text
${ART_ROOT}/
  low_noise_model/hifx4/calibration.pt
  high_noise_model/hifx4/calibration.pt
```

### 5.2 Stage 2: Quantized Artifact Preparation

Stage 2 consumes Stage 1 calibration artifacts and creates the prepared HiFloat4 weight path plus runtime MinMax lookup ranges:

```bash
ART_ROOT="state_quant/hisqrot4_ptq" \
SMOOTHQUANT_ALPHA=0.9 \
bash runfiles/02_prepare_ptq_standard.sh
```

Expected outputs:

```text
${ART_ROOT}/
  low_noise_model/hifx4/prepared.pt
  low_noise_model/hifx4/manifest.json
  high_noise_model/hifx4/prepared.pt
  high_noise_model/hifx4/manifest.json
```

Set `ROTATION_PATH` only if you want to override the internal deterministic Hadamard-style rotation with a local rotation checkpoint.

### 5.3 Stage 3: Inference with Rebuilt Artifacts

Point `ART_ROOT` at your rebuilt artifact root:

```bash
ART_ROOT="state_quant/hisqrot4_ptq" \
PROMPT="A cinematic shot of a cat surfing on the sea." \
OUTPUT_FILE="video_output/hifx4/custom_stage123_sample.mp4" \
bash runfiles/03_infer_ptq_standard.sh
```

For batch inference with rebuilt artifacts:

```bash
ART_ROOT="state_quant/hisqrot4_ptq" \
PROMPT_FILE="data/prompts/OpenS2V-5M_to_mm_vbench_30.json" \
OUT_FOLDER="video_output/hifx4/custom_stage123_vbench30" \
bash runfiles/03_batch_custom_prompt_file_infer.sh
```

## 6. VBench Evaluation

Install VBench and its evaluation dependencies following the official
[VBench](https://github.com/Vchitect/VBench) instructions. If you keep the
VBench checkout inside this repository, place it at `vbench/`; that directory is
ignored by Git.

```bash
git clone https://github.com/Vchitect/VBench.git vbench
pip install -e vbench
pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab"
```

After batch generation, evaluate the output directory:

```bash
VIDEOS_INPUT_DIR="video_output/hifx4/OpenS2V-5M_to_mm_vbench_30_alpha0p9" \
RUN_TAG="vbench_hisqrot4_alpha0p9_vbench30" \
bash runfiles/03_eval_vbench_video_dir_custom5.sh
```

Evaluation results are written to:

```text
eval_output/vbench_hisqrot4_alpha0p9_vbench30/
```

Set `EXPECTED_VIDEO_CNT` only when you want the evaluation script to validate an exact number of generated videos.

## 7. Repository Layout

```text
hisqrot4/
  generate.py
  hifx4_linear_quant.py
  hifx4_ptq_backend.py
  hifloat4/
  wan2.2/
  runfiles/
  data/prompts/OpenS2V-5M_to_mm_vbench_30.json
  models/Wan2.2-T2V-A14B/              # download from Wan-AI/Wan2.2-T2V-A14B
  artifacts/hisqrot4_alpha_0p9/        # download from BitsPlease/HiSQRot4
  requirements.txt
```

The GitHub repository tracks source code, scripts, prompts, configs, tokenizer
metadata, and artifact manifests. It does not track original-precision model
weights, quantized `prepared.pt` artifacts, generated videos, evaluation outputs,
logs, or Hugging Face caches.


## 8. Acknowledgements

HiSQRot4 builds on:

- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [HiFloat4](https://github.com/global-computing-consortium/HiFloat4)
- [SmoothQuant](https://github.com/mit-han-lab/smoothquant)
- [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q)
- [VBench](https://github.com/Vchitect/VBench)

Please cite the upstream Wan2.2 and relevant quantization work when using this project in research.