#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from hifx4_linear_quant import (
    Hifx4Config,
    WRAPPER_TIMER,
    finalize_calibration_observers,
    load_calibration_shard_stats,
    load_merged_calibration_shard_stats,
    mark_calibration_prompt_run,
    patch_want2v_configure_model,
    save_calibration_shard_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _default_wan_root() -> str:
    return os.environ.get("WAN_ROOT", str((PROJECT_ROOT / "wan2.2").resolve()))


def _default_hifloat4_root() -> str:
    return os.environ.get("HIFLOAT4_ROOT", str((PROJECT_ROOT / "hifloat4").resolve()))


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_int(value, default: int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_prompt_records(
    prompt_files: list[Path],
    default_frame_num: int,
    default_fps: float,
) -> list[dict]:
    records: list[dict] = []
    for fp in prompt_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = item.get("cap")
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if not prompt:
                continue
            orig_frame_num = int(default_frame_num)
            effective_frame_num = int(default_frame_num)
            fps = float(default_fps)
            source_path = item.get("path")
            if isinstance(source_path, str) and source_path.strip():
                source_name = Path(source_path.strip()).name
            else:
                source_name = f"{fp.stem}_{idx:06d}"
            records.append(
                {
                    "source_name": source_name,
                    "prompt": prompt,
                    "orig_frame_num": int(orig_frame_num),
                    "frame_num": int(effective_frame_num),
                    "fps": float(fps),
                }
            )
    return records


def _dist_setup_for_launcher(args) -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel is not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert (
            args.ulysses_size == world_size
        ), f"The number of ulysses_size should be equal to world_size, got {args.ulysses_size} != {world_size}."
        from wan.distributed.util import init_distributed_group  # pylint: disable=import-error

        init_distributed_group()

    return rank, world_size, local_rank


def _dist_cleanup_for_launcher() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


# Legacy svdq/svdquant CLI alias mapping is intentionally disabled.
# The old implementation is kept below as comments for audit/history only;
# current PTQ mainline should fail fast on legacy flags instead of auto-mapping.
#
# def _normalize_legacy_cli_aliases(argv: list[str]) -> tuple[list[str], bool]:
#     """
#     Backward compatibility:
#     - --svdq_*  -> --ptq_*
#     - --hifx4_quant_method svdquant -> ptq
#     """
#     mapped: list[str] = []
#     changed = False
#     i = 0
#     while i < len(argv):
#         token = argv[i]
#
#         if token == "--hifx4_quant_method" and i + 1 < len(argv):
#             mapped.append(token)
#             value = argv[i + 1]
#             if value == "svdquant":
#                 value = "ptq"
#                 changed = True
#             mapped.append(value)
#             i += 2
#             continue
#
#         if token.startswith("--hifx4_quant_method="):
#             key, value = token.split("=", 1)
#             if value == "svdquant":
#                 token = f"{key}=ptq"
#                 changed = True
#
#         if token.startswith("--svdq_"):
#             token = "--ptq_" + token[len("--svdq_") :]
#             changed = True
#
#         mapped.append(token)
#         i += 1
#     return mapped, changed


def _parse_launcher_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 HIFX4 inference launcher (isolated from official repo)."
    )
    parser.add_argument(
        "--wan_root",
        type=str,
        default=_default_wan_root(),
        help="Official Wan2.2 repo path (read-only usage).",
    )
    parser.add_argument(
        "--hifloat4_root",
        type=str,
        default=_default_hifloat4_root(),
        help="HiFloat4 repo root path.",
    )
    parser.add_argument(
        "--hifx4_qtype",
        type=str,
        default="hifx4",
        help="HiFloat4 quant type string, e.g. hifx4.",
    )
    parser.add_argument(
        "--hifx4_quant_method",
        type=str,
        default="hifx4",
        choices=["hifx4", "ptq"],
        help="Quantization method. `ptq` enables the Wan2.2 PTQ pipeline.",
    )
    parser.add_argument(
        "--hifx4_stage",
        type=str,
        default="infer",
        choices=["calibrate", "prepare", "infer", "all"],
        help=(
            "PTQ stage control: calibrate (stage1), prepare (stage2), "
            "infer (stage3), all (stage1+2+3). Non-ptq methods ignore this."
        ),
    )
    parser.add_argument(
        "--ptq_rank",
        type=int,
        default=32,
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_alpha",
        type=float,
        default=0.5,
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_eps",
        type=float,
        default=1e-5,
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_energy_threshold",
        type=float,
        default=0.98,
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_artifact_root",
        type=str,
        default="./state_quant/hif4_ptq",
        help="Root directory to save/load PTQ offline artifacts.",
    )
    parser.add_argument(
        "--ptq_pipeline_mode",
        type=str,
        default=None,
        choices=["infer", "all", "prepare"],
        help=(
            "Deprecated alias for stage control. Maps to: infer->infer, "
            "prepare->prepare, all->all."
        ),
    )
    parser.add_argument(
        "--ptq_force_rebuild",
        action="store_true",
        help="Force rebuilding calibration/prepared artifacts in PTQ modes.",
    )
    parser.add_argument(
        "--ptq_validate_artifact",
        action="store_true",
        help="Validate artifact metadata/model signature before runtime replacement.",
    )
    parser.add_argument(
        "--ptq_offline_model",
        type=str,
        default="all",
        choices=["all", "low", "high"],
        help="For calibrate/prepare: select which DiT model to process (all/low/high).",
    )
    parser.add_argument(
        "--ptq_calib_prompts_dir",
        type=str,
        default="",
        help="Directory containing *.json prompts for real stage1 calibration.",
    )
    parser.add_argument(
        "--ptq_calib_prompts_file",
        type=str,
        default="",
        help="Single prompt json file for real stage1 calibration.",
    )
    parser.add_argument(
        "--ptq_calib_prompt_limit",
        type=int,
        default=0,
        help="Optional max number of calibration prompts to use (0 means all).",
    )
    parser.add_argument(
        "--ptq_calib_prompt_shards",
        type=int,
        default=1,
        help="Shard count for calibration prompts (>=1).",
    )
    parser.add_argument(
        "--ptq_calib_prompt_shard_idx",
        type=int,
        default=0,
        help="Current shard index for calibration prompts [0, shards).",
    )
    parser.add_argument(
        "--ptq_calib_merge_only",
        action="store_true",
        help="Merge shard stats and finalize stage1 artifacts without running prompts.",
    )
    parser.add_argument(
        "--ptq_calib_resume_shard",
        action="store_true",
        help="Resume stage1 shard from saved shard stats if available.",
    )
    parser.add_argument(
        "--ptq_calib_checkpoint_every_prompt",
        action="store_true",
        help="Save shard stats after every prompt for crash recovery.",
    )
    parser.add_argument(
        "--ptq_calib_checkpoint_interval",
        type=int,
        default=0,
        help="Save shard stats every N prompts during stage1 real calibration (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--ptq_alpha_candidates",
        type=str,
        default="0.3,0.5,0.7",
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_finalize_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Deprecated compatibility option (ignored in PTQ-only route).",
    )
    parser.add_argument(
        "--ptq_calib_keep_vae_decode",
        action="store_true",
        help="Keep VAE decode during stage1 real calibration (default skips decode for speed).",
    )
    parser.add_argument(
        "--ptq_calib_store_act_absmax_seq",
        action="store_true",
        help="Store per-forward act_absmax trajectories for visualization/debugging (default off to save RAM).",
    )
    parser.add_argument(
        "--ptq_infer_engine",
        type=str,
        default="python",
        choices=["python"],
        help="Stage3 infer engine selector. PTQ route currently supports python only.",
    )
    parser.add_argument(
        "--nunchaku_precision",
        type=str,
        default="auto",
        choices=["auto", "int4", "fp4"],
        help="Deprecated compatibility option. Ignored in PTQ-only route.",
    )
    parser.add_argument(
        "--hifx4_force_py",
        action="store_true",
        help="Force Python fallback quantization (slower).",
    )
    parser.add_argument(
        "--hifx4_no_force_fp32",
        action="store_true",
        help="Disable force_fp32 in quant_dequant_float.",
    )
    parser.add_argument(
        "--hifx4_quant_output",
        action="store_true",
        help="Quantize Linear output as well (default: off).",
    )
    parser.add_argument(
        "--hifx4_include_non_block_linears",
        action="store_true",
        help="Also quantize non-block linears (default only blocks.*).",
    )
    parser.add_argument(
        "--ptq_keep_blocks",
        type=str,
        default="",
        help="Comma-separated block ids kept in FP (at most 2), e.g. 3,27.",
    )
    parser.add_argument(
        "--ptq_timestep_count",
        type=int,
        default=40,
        help="Expected sampling timestep count for timestep-aware activation quantization.",
    )
    parser.add_argument(
        "--ptq_act_quant_mode",
        type=str,
        default="online",
        choices=["online", "lookup"],
        help="Activation quant mode: online (runtime min-max) or lookup (stage1/2 min-max table).",
    )
    parser.add_argument(
        "--ptq_weight_group_size",
        type=int,
        default=64,
        help="Deprecated compatibility option. Ignored in the current FP4 prepare route.",
    )
    parser.add_argument(
        "--ptq_enable_smoothquant",
        action="store_true",
        help="Enable SmoothQuant-style channel scaling during stage2 prepare (derived from Stage1 min-max stats).",
    )
    parser.add_argument(
        "--ptq_smoothquant_alpha",
        type=float,
        default=0.8,
        help="SmoothQuant alpha in [0,1].",
    )
    parser.add_argument(
        "--ptq_smoothquant_eps",
        type=float,
        default=1e-5,
        help="Numerical epsilon for SmoothQuant channel mask.",
    )
    parser.add_argument(
        "--ptq_enable_rotation",
        action="store_true",
        help="Enable input-side rotation during stage2 prepare/infer. Defaults to internal deterministic hadamard-style rotation.",
    )
    parser.add_argument(
        "--ptq_rotation_path",
        type=str,
        default="",
        help="Optional external input-rotation checkpoint. Leave empty to use internal deterministic rotation.",
    )
    parser.add_argument(
        "--ptq_rotation_seed",
        type=int,
        default=17,
        help="Base seed for internal deterministic rotation generation.",
    )
    parser.add_argument(
        "--ptq_high_noise_split_ranges",
        type=str,
        default="",
        help=(
            "Stage2 only. Optional timestep range split for high_noise_model, "
            "format: '0-18,19-38'. Empty means disabled."
        ),
    )

    # Legacy svdq/svdquant auto-mapping is intentionally disabled.
    # Historical compatibility code is kept commented near the helper above.
    # normalized_argv, used_legacy_alias = _normalize_legacy_cli_aliases(sys.argv[1:])
    # if used_legacy_alias:
    #     print(
    #         "[PTQ][Compat] detected legacy svdq/svdquant flags, auto-mapped to ptq equivalents.",
    #         flush=True,
    #     )
    launcher_args, remaining = parser.parse_known_args(sys.argv[1:])
    return launcher_args, remaining


def main():
    launcher_args, wan_cli_args = _parse_launcher_args()
    wan_root = launcher_args.wan_root
    if not os.path.isdir(wan_root):
        raise FileNotFoundError(f"wan_root not found: {wan_root}")

    # Make official wan package importable.
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)

    import wan.text2video as wan_text2video  # pylint: disable=import-error

    stage = launcher_args.hifx4_stage
    if launcher_args.ptq_pipeline_mode is not None:
        stage = launcher_args.ptq_pipeline_mode
        if stage == "prepare":
            stage = "prepare"
        elif stage == "all":
            stage = "all"
        else:
            stage = "infer"

    alpha_candidates = [
        float(x.strip()) for x in launcher_args.ptq_alpha_candidates.split(",") if x.strip()
    ]
    if len(alpha_candidates) == 0:
        alpha_candidates = [launcher_args.ptq_alpha]

    keep_blocks: list[int] = []
    if launcher_args.ptq_keep_blocks.strip():
        keep_blocks = [int(x.strip()) for x in launcher_args.ptq_keep_blocks.split(",") if x.strip()]
    if stage == "calibrate" and keep_blocks:
        print(
            f"[HIF4][Stage1] keep_blocks is ignored during calibration (all layers must participate): {keep_blocks}",
            flush=True,
        )
        keep_blocks = []

    cfg = Hifx4Config(
        qtype=launcher_args.hifx4_qtype,
        quant_method=launcher_args.hifx4_quant_method,
        stage=stage,
        force_py=launcher_args.hifx4_force_py,
        force_fp32=not launcher_args.hifx4_no_force_fp32,
        quant_output=launcher_args.hifx4_quant_output,
        only_blocks=not launcher_args.hifx4_include_non_block_linears,
        ptq_rank=launcher_args.ptq_rank,
        ptq_alpha=launcher_args.ptq_alpha,
        ptq_alpha_candidates=alpha_candidates,
        ptq_eps=launcher_args.ptq_eps,
        ptq_energy_threshold=launcher_args.ptq_energy_threshold,
        ptq_artifact_root=launcher_args.ptq_artifact_root,
        ptq_force_rebuild=launcher_args.ptq_force_rebuild,
        ptq_validate_artifact=launcher_args.ptq_validate_artifact,
        ptq_offline_model=launcher_args.ptq_offline_model,
        ptq_calib_skip_vae_decode=not launcher_args.ptq_calib_keep_vae_decode,
        ptq_finalize_device=launcher_args.ptq_finalize_device,
        ptq_infer_engine=launcher_args.ptq_infer_engine,
        nunchaku_precision=launcher_args.nunchaku_precision,
        keep_blocks=keep_blocks,
        timestep_count=launcher_args.ptq_timestep_count,
        act_quant_mode=launcher_args.ptq_act_quant_mode,
        weight_group_size=launcher_args.ptq_weight_group_size,
        ptq_enable_smoothquant=launcher_args.ptq_enable_smoothquant,
        ptq_smoothquant_alpha=launcher_args.ptq_smoothquant_alpha,
        ptq_smoothquant_eps=launcher_args.ptq_smoothquant_eps,
        ptq_enable_rotation=launcher_args.ptq_enable_rotation,
        ptq_rotation_path=launcher_args.ptq_rotation_path,
        ptq_rotation_seed=launcher_args.ptq_rotation_seed,
        ptq_high_noise_split_ranges=launcher_args.ptq_high_noise_split_ranges,
        ptq_store_act_absmax_seq=launcher_args.ptq_calib_store_act_absmax_seq,
    )
    if cfg.ptq_infer_engine != "python":
        raise ValueError(
            f"Unsupported --ptq_infer_engine={cfg.ptq_infer_engine}. "
            "PTQ route only supports python."
        )
    patch_want2v_configure_model(
        wan_text2video_module=wan_text2video,
        hifloat4_root=launcher_args.hifloat4_root,
        cfg=cfg,
    )

    original_generate_py = os.path.join(wan_root, "generate.py")
    wan_generate = _load_module_from_file("wan22_generate_original", original_generate_py)

    # Delegate Wan CLI parsing/execution to official script to keep compatibility.
    sys.argv = [original_generate_py] + wan_cli_args
    args = wan_generate._parse_args()

    is_real_calib = (
        cfg.quant_method == "ptq"
        and cfg.stage == "calibrate"
        and bool(
            launcher_args.ptq_calib_prompts_dir.strip()
            or launcher_args.ptq_calib_prompts_file.strip()
        )
    )
    is_offline_prepare = cfg.quant_method == "ptq" and cfg.stage == "prepare"
    if is_offline_prepare:
        if "t2v" not in args.task:
            raise NotImplementedError("Offline prepare currently supports t2v tasks only.")
        if args.t5_fsdp or args.dit_fsdp or args.ulysses_size > 1:
            print(
                "[PTQ][Stage2] Offline prepare runs in single-process mode. "
                "Ignoring distributed flags for artifact build.",
                flush=True,
            )
            args.t5_fsdp = False
            args.dit_fsdp = False
            args.ulysses_size = 1
        if args.offload_model is None:
            args.offload_model = True
        args.save_file = None
        print(
            f"[PTQ][Stage2] Offline prepare starts: task={args.task} size={args.size} "
            f"artifact_root={cfg.ptq_artifact_root} offline_model={cfg.ptq_offline_model}",
            flush=True,
        )
        model_cfg = wan_generate.WAN_CONFIGS[args.task]
        _ = wan_text2video.WanT2V(
            config=model_cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        print("[PTQ][Stage2] Offline prepare finished.", flush=True)
        return
    if not is_real_calib:
        is_stage3 = cfg.quant_method == "ptq" and cfg.stage == "infer"
        if is_stage3:
            WRAPPER_TIMER.enable()
            print(
                f"[TIMER] Stage3 infer starting (engine={cfg.ptq_infer_engine})...",
                flush=True,
            )
            wall_t0 = time.perf_counter()
        wan_generate.generate(args)

        if is_stage3:
            wall_elapsed = time.perf_counter() - wall_t0
            summary = WRAPPER_TIMER.summary()
            print(f"[TIMER] ======== Stage3 Timing Summary ========", flush=True)
            print(f"[TIMER] engine           : {cfg.ptq_infer_engine}", flush=True)
            print(f"[TIMER] wall_time_sec    : {wall_elapsed:.2f}", flush=True)
            print(f"[TIMER] linear_total_sec : {summary['total_sec']}", flush=True)
            print(f"[TIMER] linear_calls     : {summary['call_count']}", flush=True)
            print(f"[TIMER] linear_avg_ms    : {summary['avg_ms_per_call']}", flush=True)
            print(f"[TIMER] ========================================", flush=True)
        return

    prompt_files: list[Path] = []
    source_desc = ""
    if launcher_args.ptq_calib_prompts_file.strip():
        prompt_file = Path(launcher_args.ptq_calib_prompts_file).expanduser().resolve()
        if not prompt_file.is_file():
            raise FileNotFoundError(f"ptq_calib_prompts_file not found: {prompt_file}")
        prompt_files = [prompt_file]
        source_desc = str(prompt_file)
    else:
        prompts_dir = Path(launcher_args.ptq_calib_prompts_dir).expanduser().resolve()
        if not prompts_dir.is_dir():
            raise FileNotFoundError(f"ptq_calib_prompts_dir not found: {prompts_dir}")
        prompt_files = sorted(prompts_dir.glob("*.json"))
        if len(prompt_files) == 0:
            raise FileNotFoundError(f"No prompt json found in: {prompts_dir}")
        source_desc = str(prompts_dir)

    prompt_records = _load_prompt_records(
        prompt_files=prompt_files,
        default_frame_num=61,
        default_fps=float(wan_generate.WAN_CONFIGS[args.task].sample_fps),
    )
    if launcher_args.ptq_calib_prompt_limit > 0:
        prompt_records = prompt_records[: launcher_args.ptq_calib_prompt_limit]
    shards = max(1, int(launcher_args.ptq_calib_prompt_shards))
    shard_idx = int(launcher_args.ptq_calib_prompt_shard_idx)
    if not (0 <= shard_idx < shards):
        raise ValueError(f"ptq_calib_prompt_shard_idx must be in [0, {shards}), got {shard_idx}")
    if shards > 1:
        prompt_records = [p for i, p in enumerate(prompt_records) if (i % shards) == shard_idx]
    if len(prompt_records) == 0 and not launcher_args.ptq_calib_merge_only:
        raise RuntimeError(f"No valid prompts loaded from {source_desc}")

    if "t2v" not in args.task:
        raise NotImplementedError("Real calibration loop currently supports t2v tasks only.")

    rank, world_size, local_rank = _dist_setup_for_launcher(args)
    is_rank0 = rank == 0
    try:
        if args.offload_model is None:
            # Align with official Wan2.2 defaults.
            args.offload_model = False if world_size > 1 else True

        if dist.is_initialized():
            base_seed = [args.base_seed] if is_rank0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = int(base_seed[0])

        args.save_file = None
        if is_rank0:
            print(
                f"[PTQ][Stage1] Real calibration starts: prompts={len(prompt_records)} "
                f"source={source_desc} offline_model={cfg.ptq_offline_model} "
                f"shard={shard_idx}/{shards} rank={rank}/{world_size}",
                flush=True,
            )
        model_cfg = wan_generate.WAN_CONFIGS[args.task]
        t2v = wan_text2video.WanT2V(
            config=model_cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=local_rank,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        if cfg.ptq_calib_skip_vae_decode:
            # Stage1 only needs DiT activations; skipping VAE decode does not affect observer stats.
            t2v.vae.decode = lambda latents: latents
            if is_rank0:
                print("[PTQ][Stage1] VAE decode skipped for calibration speed.", flush=True)
        if cfg.ptq_offline_model in ("low", "high") and is_rank0:
            print(
                f"[PTQ][Stage1] Collecting calibration stats for selector={cfg.ptq_offline_model} "
                "(Wan2.2 forward path remains unchanged).",
                flush=True,
            )
        if launcher_args.ptq_calib_merge_only:
            if is_rank0:
                print(
                    f"[PTQ][Stage1] Merge-only mode: loading shard stats (num_shards={shards})",
                    flush=True,
                )
            load_merged_calibration_shard_stats(num_shards=shards)
            finalize_calibration_observers()
            if is_rank0:
                print("[PTQ][Stage1] Merge-only finalize finished.", flush=True)
            return
        if cfg.ptq_offline_model == "all" and launcher_args.ptq_calib_resume_shard:
            loaded_prompt_runs = load_calibration_shard_stats(
                shard_idx=shard_idx,
                num_shards=shards,
                allow_missing=True,
            )
            if loaded_prompt_runs > 0:
                if loaded_prompt_runs >= len(prompt_records):
                    if is_rank0:
                        print(
                            f"[PTQ][Stage1] Resume: shard {shard_idx}/{shards} already complete "
                            f"(prompt_runs={loaded_prompt_runs}, prompts={len(prompt_records)}). "
                            "Proceed to finalize.",
                            flush=True,
                        )
                    prompt_records = []
                else:
                    prompt_records = prompt_records[loaded_prompt_runs:]
                if is_rank0:
                    print(
                        f"[PTQ][Stage1] Resume: skip first {loaded_prompt_runs} prompts for shard "
                        f"{shard_idx}/{shards}, remaining={len(prompt_records)}.",
                        flush=True,
                    )
        size = wan_generate.SIZE_CONFIGS[args.size]
        checkpoint_interval = int(max(0, launcher_args.ptq_calib_checkpoint_interval))
        if checkpoint_interval <= 0 and launcher_args.ptq_calib_checkpoint_every_prompt:
            checkpoint_interval = 1
        for idx, rec in enumerate(prompt_records, start=1):
            prompt = rec["prompt"]
            orig_frame_num = int(rec.get("orig_frame_num", rec["frame_num"]))
            frame_num = int(rec["frame_num"])
            sample_fps = float(rec["fps"])
            if sample_fps > 0:
                t2v.config.sample_fps = sample_fps
            run_offload_model = bool(args.offload_model)
            if is_rank0:
                print(
                    f"[PTQ][Stage1] Prompt {idx}/{len(prompt_records)} source={rec['source_name']} "
                    f"orig_frame_num={orig_frame_num} effective_frame_num={frame_num} "
                    f"fps={sample_fps} "
                    f"offload_model={run_offload_model} "
                    f"rank={rank}/{world_size}",
                    flush=True,
                )
                print(
                    f"[PTQ][Stage1] Prompt text: {json.dumps(prompt, ensure_ascii=False)}",
                    flush=True,
                )
            _ = t2v.generate(
                prompt,
                size=size,
                frame_num=frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=run_offload_model,
            )
            completed_prompt_runs = int(mark_calibration_prompt_run())
            if cfg.ptq_offline_model == "all" and checkpoint_interval > 0:
                should_checkpoint = (
                    completed_prompt_runs > 0
                    and completed_prompt_runs % checkpoint_interval == 0
                )
                if should_checkpoint:
                    save_calibration_shard_stats(
                        shard_idx=shard_idx,
                        num_shards=shards,
                        clear_after_save=bool(cfg.ptq_incremental_checkpoint),
                    )
                    if is_rank0:
                        print(
                            f"[PTQ][Stage1] Periodic checkpoint saved at prompt_runs="
                            f"{completed_prompt_runs} (interval={checkpoint_interval}).",
                            flush=True,
                        )
            torch.cuda.empty_cache()
        if cfg.ptq_offline_model == "all":
            save_calibration_shard_stats(
                shard_idx=shard_idx,
                num_shards=shards,
                clear_after_save=bool(cfg.ptq_incremental_checkpoint),
            )
            if cfg.ptq_incremental_checkpoint:
                load_calibration_shard_stats(
                    shard_idx=shard_idx,
                    num_shards=shards,
                    allow_missing=False,
                )
        if shards > 1 and cfg.ptq_offline_model == "all":
            if is_rank0:
                print(
                    "[PTQ][Stage1] Shard calibration collection finished. "
                    "Run merge-only command to finalize calibration artifacts.",
                    flush=True,
                )
        else:
            finalize_calibration_observers()
            if is_rank0:
                print("[PTQ][Stage1] Real calibration finished.", flush=True)
    finally:
        _dist_cleanup_for_launcher()


if __name__ == "__main__":
    main()
