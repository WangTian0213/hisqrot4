import hashlib
import importlib
import json
import math
import os
import sys
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from hifx4_ptq_backend import run_hifx4_ptq_fused_linear


ARTIFACT_SCHEMA_VERSION = "wan22-hif4-fp4-minmax-sqrot.v4"
_TIMESTEP_BRANCH_SEP = "::"
_KNOWN_BRANCHES = ("cond", "uncond", "single")
_DEFAULT_ACTIVATION_GROUPING = "low_all_high_split2"


class _WrapperTimer:
    """Aggregate runtime of quantized linear wrappers."""

    def __init__(self):
        self.total_sec: float = 0.0
        self.call_count: int = 0
        self._enabled: bool = False

    def enable(self):
        self._enabled = True
        self.total_sec = 0.0
        self.call_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, elapsed_sec: float):
        if self._enabled:
            self.total_sec += elapsed_sec
            self.call_count += 1

    def summary(self) -> Dict[str, Any]:
        avg_ms = (self.total_sec / self.call_count * 1000.0) if self.call_count else 0.0
        return {
            "total_sec": round(self.total_sec, 4),
            "call_count": self.call_count,
            "avg_ms_per_call": round(avg_ms, 4),
        }


WRAPPER_TIMER = _WrapperTimer()
_ROTATION_PAYLOAD_CACHE: Dict[str, Dict[str, Any]] = {}
_HADAMARD_MATRIX_CACHE: Dict[int, torch.Tensor] = {}
_PREPARED_LAYER_KEYS = (
    "weight_fp4",
    "shape",
    "bias",
    "act_min_global",
    "act_max_global",
    "act_min_group_branch",
    "act_max_group_branch",
    "smoothquant_applied",
    "smoothquant_channel_mask",
    "rotation_applied",
    "rotation_kind",
    "rotation_mode",
    "rotation_source_path",
    "rotation_seed",
    "rotation_block_size",
    "weight_format",
)


@dataclass
class Hifx4Config:
    qtype: str = "hifx4"
    quant_method: str = "ptq"
    stage: str = "infer"  # calibrate | prepare | infer | all
    force_py: bool = False
    force_fp32: bool = True
    quant_output: bool = False
    only_blocks: bool = True
    ptq_rank: int = 0
    ptq_alpha: float = 0.5
    ptq_alpha_candidates: List[float] = field(default_factory=lambda: [0.5])
    ptq_eps: float = 1e-5
    ptq_energy_threshold: float = 0.98
    ptq_artifact_root: str = "./state_quant/hif4_ptq"
    ptq_force_rebuild: bool = False
    ptq_validate_artifact: bool = False
    ptq_offline_model: str = "all"  # all | low | high
    ptq_calib_skip_vae_decode: bool = True
    ptq_finalize_device: str = "auto"  # auto | cpu | cuda
    ptq_infer_engine: str = "python"  # python
    nunchaku_precision: str = "auto"  # auto | int4 | fp4
    keep_blocks: List[int] = field(default_factory=list)
    timestep_count: int = 40
    act_quant_mode: str = "online"  # online | lookup (min-max)
    act_clip_percentile: float = 1.0
    weight_group_size: int = 64  # deprecated compatibility option in FP4 PTQ route
    ptq_enable_smoothquant: bool = False
    ptq_smoothquant_alpha: float = 0.8
    ptq_smoothquant_eps: float = 1e-5
    ptq_rotation_path: str = ""
    ptq_enable_rotation: bool = False
    ptq_rotation_seed: int = 17
    ptq_high_noise_split_ranges: str = ""
    ptq_store_act_absmax_seq: bool = False
    ptq_incremental_checkpoint: bool = True


def _split_count_evenly(total: int, bucket_count: int) -> List[int]:
    total = max(0, int(total))
    bucket_count = max(1, int(bucket_count))
    base = total // bucket_count
    rem = total % bucket_count
    return [base + (1 if i < rem else 0) for i in range(bucket_count)]


def _build_group_ranges_from_step_tags(step_model_tags: List[str]) -> Dict[str, List[Dict[str, int]]]:
    ranges: Dict[str, List[Dict[str, int]]] = {
        "low_noise_model": [],
        "high_noise_model": [],
    }
    if len(step_model_tags) == 0:
        return ranges

    low_step_ids = [idx for idx, tag in enumerate(step_model_tags) if tag == "low_noise_model"]
    high_step_ids = [idx for idx, tag in enumerate(step_model_tags) if tag == "high_noise_model"]

    if low_step_ids:
        ranges["low_noise_model"].append(
            {
                "group_id": 0,
                "timestep_start": int(low_step_ids[0]),
                "timestep_end": int(low_step_ids[-1]),
                "step_count": int(len(low_step_ids)),
            }
        )

    if high_step_ids:
        split_sizes = _split_count_evenly(len(high_step_ids), 2)
        cursor = 0
        for group_id, step_count in enumerate(split_sizes):
            if step_count <= 0:
                continue
            selected = high_step_ids[cursor : cursor + step_count]
            cursor += step_count
            ranges["high_noise_model"].append(
                {
                    "group_id": int(group_id),
                    "timestep_start": int(selected[0]),
                    "timestep_end": int(selected[-1]),
                    "step_count": int(step_count),
                }
            )
    return ranges


def _default_group_ranges_for_model(model_tag: str, expected_steps: int) -> List[Dict[str, int]]:
    expected_steps = max(0, int(expected_steps))
    if model_tag == "high_noise_model":
        split_sizes = _split_count_evenly(expected_steps, 2)
        out: List[Dict[str, int]] = []
        cursor = 0
        for group_id, step_count in enumerate(split_sizes):
            if step_count <= 0:
                continue
            start = cursor
            end = cursor + step_count - 1
            out.append(
                {
                    "group_id": int(group_id),
                    "timestep_start": int(start),
                    "timestep_end": int(end),
                    "step_count": int(step_count),
                }
            )
            cursor += step_count
        return out
    if expected_steps <= 0:
        return []
    return [
        {
            "group_id": 0,
            "timestep_start": 0,
            "timestep_end": max(0, expected_steps - 1),
            "step_count": int(expected_steps),
        }
    ]


def _get_group_ranges_for_model(
    model_tag: str,
    expected_steps: int,
    grouping_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, int]]:
    grouping_meta = grouping_meta if isinstance(grouping_meta, dict) else {}
    model_ranges = grouping_meta.get("model_group_ranges", {})
    ranges = model_ranges.get(model_tag, [])
    out: List[Dict[str, int]] = []
    if isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "group_id": int(item.get("group_id", 0)),
                    "timestep_start": int(item.get("timestep_start", -1)),
                    "timestep_end": int(item.get("timestep_end", -1)),
                    "step_count": int(item.get("step_count", 0)),
                }
            )
    if out:
        return out
    return _default_group_ranges_for_model(model_tag=model_tag, expected_steps=expected_steps)


def _resolve_activation_group(
    model_tag: str,
    timestep_id: int,
    expected_steps: int,
    grouping_meta: Optional[Dict[str, Any]] = None,
) -> int:
    timestep_id = int(timestep_id)
    ranges = _get_group_ranges_for_model(
        model_tag=model_tag,
        expected_steps=expected_steps,
        grouping_meta=grouping_meta,
    )
    for item in ranges:
        if int(item["timestep_start"]) <= timestep_id <= int(item["timestep_end"]):
            return int(item["group_id"])
    if model_tag == "high_noise_model":
        split = max(1, int(math.ceil(max(1, expected_steps) / 2.0)))
        return 0 if timestep_id < split else 1
    return 0


def _group_ids_for_timestep_range(
    model_tag: str,
    expected_steps: int,
    timestep_range: Optional[Tuple[int, int]],
    grouping_meta: Optional[Dict[str, Any]] = None,
) -> Optional[set[int]]:
    if timestep_range is None:
        return None
    start, end = int(timestep_range[0]), int(timestep_range[1])
    out: set[int] = set()
    for item in _get_group_ranges_for_model(
        model_tag=model_tag,
        expected_steps=expected_steps,
        grouping_meta=grouping_meta,
    ):
        if int(item["timestep_end"]) < start or int(item["timestep_start"]) > end:
            continue
        out.add(int(item["group_id"]))
    return out


class QuantContext:
    """
    Runtime context used by activation quantization.

    It tracks:
    - timestep_id (step index, not absolute diffusion t value)
    - branch (cond/uncond/single)
    - model_tag (low/high)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.expected_steps = 0
        self.timestep_id = -1
        self.group_id = -1
        self.branch = "single"
        self.model_tag = ""
        self._last_t_value: Optional[int] = None
        self._step_counter = -1
        self._branch_call_counter = 0
        self.step_model_tags: List[str] = []
        self.step_group_ids: List[int] = []
        self.grouping_meta: Dict[str, Any] = {}

    def begin_generation(
        self,
        expected_steps: int,
        step_model_tags: Optional[List[str]] = None,
        step_group_ids: Optional[List[int]] = None,
        grouping_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reset()
        self.active = True
        self.expected_steps = max(0, int(expected_steps))
        self.step_model_tags = [str(x) for x in (step_model_tags or [])]
        self.step_group_ids = [int(x) for x in (step_group_ids or [])]
        self.grouping_meta = dict(grouping_meta or {})

    def end_generation(self) -> None:
        self.active = False

    def update_from_forward(self, t: Optional[torch.Tensor], model_tag: str) -> None:
        if t is None:
            return
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return
        t_value = int(round(float(t.reshape(-1)[0].detach().cpu().item())))

        if self._last_t_value is None or t_value != self._last_t_value:
            self._step_counter += 1
            self._branch_call_counter = 0
            self._last_t_value = t_value
            self.branch = "cond"
        else:
            self._branch_call_counter += 1
            if self._branch_call_counter == 1:
                self.branch = "uncond"
            else:
                self.branch = "single"

        if self.expected_steps > 0:
            self.timestep_id = max(0, min(self._step_counter, self.expected_steps - 1))
        else:
            self.timestep_id = max(0, self._step_counter)
        self.model_tag = model_tag
        if 0 <= self.timestep_id < len(self.step_group_ids):
            self.group_id = int(self.step_group_ids[self.timestep_id])
        else:
            self.group_id = _resolve_activation_group(
                model_tag=model_tag,
                timestep_id=self.timestep_id,
                expected_steps=self.expected_steps,
                grouping_meta=self.grouping_meta,
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "timestep_id": int(self.timestep_id),
            "group_id": int(self.group_id),
            "branch": str(self.branch),
            "model_tag": str(self.model_tag),
        }

    def export_grouping_metadata(self, model_tag: Optional[str] = None) -> Dict[str, Any]:
        meta = dict(self.grouping_meta or {})
        if model_tag is None:
            return meta
        model_ranges = dict(meta.get("model_group_ranges", {}))
        return {
            "mode": str(meta.get("mode", _DEFAULT_ACTIVATION_GROUPING)),
            "expected_steps": int(meta.get("expected_steps", self.expected_steps)),
            "model_group_ranges": {
                str(model_tag): list(model_ranges.get(model_tag, [])),
            },
        }


GLOBAL_QUANT_CONTEXT = QuantContext()


def _tb_key(timestep_id: int, branch: str) -> str:
    return f"{int(timestep_id)}{_TIMESTEP_BRANCH_SEP}{branch}"


def _parse_tb_key(key: str) -> Tuple[int, str]:
    if _TIMESTEP_BRANCH_SEP in key:
        t, b = key.split(_TIMESTEP_BRANCH_SEP, 1)
        return int(t), b
    return -1, "single"


def _sanitize_branch(branch: str) -> str:
    if branch in ("cond", "uncond", "single"):
        return branch
    return "single"


def _parse_timestep_range_spec(spec: str) -> List[Tuple[int, int]]:
    text = str(spec or "").strip()
    if not text:
        return []
    out: List[Tuple[int, int]] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(
                f"[HIF4][Stage2] invalid timestep range token '{token}', expected START-END."
            )
        lhs, rhs = token.split("-", 1)
        try:
            start = int(lhs.strip())
            end = int(rhs.strip())
        except ValueError as exc:
            raise ValueError(
                f"[HIF4][Stage2] invalid timestep range token '{token}', expected START-END."
            ) from exc
        if start < 0 or end < 0 or start > end:
            raise ValueError(
                f"[HIF4][Stage2] invalid timestep range token '{token}', require 0<=start<=end."
            )
        out.append((start, end))
    for i in range(len(out)):
        s_i, e_i = out[i]
        for j in range(i + 1, len(out)):
            s_j, e_j = out[j]
            if not (e_i < s_j or e_j < s_i):
                raise ValueError(
                    f"[HIF4][Stage2] timestep ranges overlap: {out[i]} vs {out[j]} in '{text}'."
                )
    return out


def _filter_group_branch_dict(
    tb_dict: Dict[str, torch.Tensor],
    group_ids: Optional[set[int]],
) -> Dict[str, torch.Tensor]:
    if group_ids is None:
        return tb_dict
    out: Dict[str, torch.Tensor] = {}
    for key, value in tb_dict.items():
        try:
            group_id, _ = _parse_tb_key(str(key))
        except Exception:
            continue
        if group_id in group_ids:
            out[str(key)] = value
    return out


def _validate_keep_blocks(keep_blocks: List[int]) -> None:
    if len(keep_blocks) > 2:
        raise ValueError(f"keep_blocks allows at most 2 blocks, got {keep_blocks}")
    if len(set(keep_blocks)) != len(keep_blocks):
        raise ValueError(f"keep_blocks has duplicates: {keep_blocks}")
    for idx in keep_blocks:
        if idx < 0:
            raise ValueError(f"keep_blocks must be >= 0, got {keep_blocks}")


def _extract_block_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "blocks" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.isdigit():
                return int(nxt)
    return None


def _normalize_linear_name(name: str) -> str:
    parts = [p for p in name.split(".") if p not in ("_fsdp_wrapped_module", "_checkpoint_wrapped_module")]
    while parts and parts[0] in ("module", "_orig_mod"):
        parts = parts[1:]
    return ".".join(parts)


def _ensure_hifloat4_importable(hifloat4_root: str) -> None:
    hifx4_gpu_path = f"{hifloat4_root.rstrip('/')}/hifx4_gpu"
    if hifx4_gpu_path not in sys.path:
        sys.path.insert(0, hifx4_gpu_path)


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


def _dist_is_main_process() -> bool:
    return _dist_rank() == 0


def _dist_barrier() -> None:
    if _dist_is_initialized():
        dist.barrier()


def _dist_all_reduce_max_scalar(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def _dist_all_reduce_max_tensor(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def _dist_all_reduce_min_scalar(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.MIN)
    return y


def _dist_all_reduce_min_tensor(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.MIN)
    return y


def _dist_broadcast_object_from_main(obj: Any) -> Any:
    if not _dist_is_initialized():
        return obj
    payload = [obj] if _dist_is_main_process() else [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _is_transformer_block_linear(name: str, cfg: Hifx4Config) -> bool:
    parts = name.split(".")
    if cfg.only_blocks and "blocks" not in parts:
        return False
    block_idx = _extract_block_index(name)
    if block_idx is not None and block_idx in set(cfg.keep_blocks):
        return False
    return True


def _iter_target_linears(model: nn.Module, cfg: Hifx4Config) -> Iterable[Tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_transformer_block_linear(name, cfg):
            yield name, module


def _model_signature_with_options(
    model: nn.Module,
    cfg: Hifx4Config,
    normalize_name: bool,
    include_dtype: bool,
) -> str:
    hasher = hashlib.sha256()
    for name, linear in _iter_target_linears(model, cfg):
        key_name = _normalize_linear_name(name) if normalize_name else name
        hasher.update(key_name.encode("utf-8"))
        hasher.update(str(linear.in_features).encode("utf-8"))
        hasher.update(str(linear.out_features).encode("utf-8"))
        if include_dtype:
            hasher.update(str(linear.weight.dtype).encode("utf-8"))
    return hasher.hexdigest()[:16]


def _model_signature(model: nn.Module, cfg: Hifx4Config) -> str:
    return _model_signature_with_options(model, cfg, normalize_name=True, include_dtype=True)


def _compatible_model_signatures(model: nn.Module, cfg: Hifx4Config) -> List[str]:
    candidates = [
        _model_signature_with_options(model, cfg, normalize_name=True, include_dtype=True),
        _model_signature_with_options(model, cfg, normalize_name=True, include_dtype=False),
        _model_signature_with_options(model, cfg, normalize_name=False, include_dtype=True),
        _model_signature_with_options(model, cfg, normalize_name=False, include_dtype=False),
    ]
    dedup: List[str] = []
    for sig in candidates:
        if sig not in dedup:
            dedup.append(sig)
    return dedup


def _artifact_paths(cfg: Hifx4Config, model_tag: str) -> Dict[str, str]:
    root = Path(cfg.ptq_artifact_root).expanduser().resolve() / model_tag / cfg.qtype
    root.mkdir(parents=True, exist_ok=True)
    return {
        "dir": str(root),
        "calibration": str(root / "calibration.pt"),
        "prepared": str(root / "prepared.pt"),
        "prepared_nunchaku": str(root / "prepared_nunchaku.pt"),  # compatibility placeholder
        "manifest": str(root / "manifest.json"),
    }


def _prepared_layer_view(layer_payload: Dict[str, Any], layer_key: str) -> Dict[str, Any]:
    prepared = {k: v for k, v in layer_payload.items() if k in _PREPARED_LAYER_KEYS}
    prepared["rotation_layer_key"] = layer_key
    return prepared


def _load_high_noise_prepared_banks(
    paths: Dict[str, str],
    model_tag: str,
) -> List[Dict[str, Any]]:
    if model_tag != "high_noise_model":
        return []
    index_path = Path(paths["prepared"]).with_name("prepared_banks_index.json")
    if not index_path.is_file():
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        index_payload = json.load(f)
    if index_payload.get("schema") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"[HIF4][Stage3] prepared bank index schema mismatch: "
            f"expected {ARTIFACT_SCHEMA_VERSION}, got {index_payload.get('schema')}"
        )
    if index_payload.get("model_tag") != model_tag:
        raise ValueError(
            f"[HIF4][Stage3] prepared bank index model mismatch: "
            f"expected {model_tag}, got {index_payload.get('model_tag')}"
        )
    banks = index_payload.get("banks", [])
    if not isinstance(banks, list) or len(banks) == 0:
        return []

    out: List[Dict[str, Any]] = []
    seen_ranges: List[Tuple[int, int]] = []
    for record in banks:
        if not isinstance(record, dict):
            raise ValueError(f"[HIF4][Stage3] invalid bank record in {index_path}: {record!r}")
        timestep_range = record.get("timestep_range", None)
        if not isinstance(timestep_range, (list, tuple)) or len(timestep_range) != 2:
            raise ValueError(f"[HIF4][Stage3] invalid bank timestep_range: {timestep_range!r}")
        start, end = int(timestep_range[0]), int(timestep_range[1])
        if start < 0 or end < start:
            raise ValueError(f"[HIF4][Stage3] invalid bank timestep_range: {timestep_range!r}")
        for prev_start, prev_end in seen_ranges:
            if not (end < prev_start or prev_end < start):
                raise ValueError(
                    f"[HIF4][Stage3] overlapping prepared bank ranges: "
                    f"{(start, end)} vs {(prev_start, prev_end)}"
                )
        seen_ranges.append((start, end))

        prepared_name = str(record.get("prepared", "") or "").strip()
        if not prepared_name:
            raise ValueError(f"[HIF4][Stage3] bank record missing prepared path: {record!r}")
        prepared_path = (index_path.parent / prepared_name).resolve()
        if not prepared_path.is_file():
            raise FileNotFoundError(f"[HIF4][Stage3] missing prepared bank artifact: {prepared_path}")
        payload = torch.load(str(prepared_path), map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"[HIF4][Stage3] invalid prepared bank payload: {prepared_path}")
        if payload.get("meta", {}).get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"[HIF4][Stage3] prepared bank artifact schema mismatch for {prepared_path}: "
                f"expected {ARTIFACT_SCHEMA_VERSION}, got {payload.get('meta', {}).get('artifact_schema_version')}"
            )
        out.append(
            {
                "bank_idx": int(record.get("bank_idx", len(out))),
                "timestep_range": (start, end),
                "prepared_path": str(prepared_path),
                "payload": payload,
            }
        )
    out.sort(key=lambda item: (int(item["timestep_range"][0]), int(item["bank_idx"])))
    return out


def _save_manifest(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_torch_save(payload: Any, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_existing_calibration_payload(
    path: str,
    *,
    expected_model_signature: str,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta", {})
    if meta.get("model_signature") != expected_model_signature:
        return None
    return payload


def _pack_signed_int4_to_int8(q_int4: torch.Tensor) -> torch.Tensor:
    if q_int4.dtype != torch.int8:
        raise TypeError(f"q_int4 must be int8, got {q_int4.dtype}")
    if q_int4.ndim != 2:
        raise ValueError(f"q_int4 must be 2D, got shape={tuple(q_int4.shape)}")
    if q_int4.shape[1] % 2 != 0:
        raise ValueError("in_features must be even before packing int4.")

    q_u = q_int4.to(torch.int16).bitwise_and(0x0F).to(torch.uint8)
    low = q_u[:, 0::2].to(torch.int16)
    high = q_u[:, 1::2].to(torch.int16)
    packed_u16 = low.bitwise_or(high.bitwise_left_shift(4))
    packed_i16 = torch.where(packed_u16 > 127, packed_u16 - 256, packed_u16)
    return packed_i16.to(torch.int8)


def _unpack_signed_int4_from_int8(packed: torch.Tensor, in_features_padded: int) -> torch.Tensor:
    if packed.dtype != torch.int8:
        raise TypeError(f"packed must be int8, got {packed.dtype}")
    p16 = packed.to(torch.int16)
    low = p16.bitwise_and(0x0F)
    high = p16.bitwise_right_shift(4).bitwise_and(0x0F)
    low = torch.where(low > 7, low - 16, low).to(torch.int8)
    high = torch.where(high > 7, high - 16, high).to(torch.int8)
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), device=packed.device, dtype=torch.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :in_features_padded]


def _quantize_weight_to_packed_int4(weight_fp32: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if weight_fp32.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight_fp32.shape)}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")

    out_features, in_features = int(weight_fp32.shape[0]), int(weight_fp32.shape[1])
    in_features_padded = in_features if in_features % 2 == 0 else in_features + 1
    weight_pad = torch.zeros((out_features, in_features_padded), device=weight_fp32.device, dtype=torch.float32)
    weight_pad[:, :in_features] = weight_fp32

    n_groups = (in_features_padded + group_size - 1) // group_size
    q_int4 = torch.zeros((out_features, in_features_padded), device=weight_fp32.device, dtype=torch.int8)
    scales = torch.ones((n_groups, out_features), device=weight_fp32.device, dtype=torch.float32)
    eps = 1e-8
    for gi in range(n_groups):
        s = gi * group_size
        e = min(in_features_padded, s + group_size)
        block = weight_pad[:, s:e]
        if block.numel() == 0:
            continue
        scale = block.abs().amax(dim=1).clamp_min(eps) / 7.0
        q_block = torch.round(block / scale.unsqueeze(1)).clamp(-8, 7).to(torch.int8)
        q_int4[:, s:e] = q_block
        scales[gi, :] = scale

    packed = _pack_signed_int4_to_int8(q_int4)
    return packed.cpu(), scales.cpu(), int(in_features_padded)


def _dequantize_weight_from_packed_int4(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    in_features: int,
    in_features_padded: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    q = _unpack_signed_int4_from_int8(qweight.to(device=device, non_blocking=True), in_features_padded).to(torch.float32)
    scales = wscales.to(device=device, dtype=torch.float32, non_blocking=True)
    out_features = q.shape[0]
    weight = torch.empty((out_features, in_features_padded), dtype=torch.float32, device=device)
    n_groups = scales.shape[0]
    for gi in range(n_groups):
        s = gi * group_size
        e = min(in_features_padded, s + group_size)
        scale = scales[gi, :].unsqueeze(1)
        weight[:, s:e] = q[:, s:e].to(torch.float32) * scale
    return weight[:, :in_features].to(dtype=dtype)


def _load_rotation_payload(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = str(Path(path).expanduser().resolve())
    if p in _ROTATION_PAYLOAD_CACHE:
        return _ROTATION_PAYLOAD_CACHE[p]
    if not os.path.exists(p):
        raise FileNotFoundError(f"rotation payload not found: {p}")
    payload = torch.load(p, map_location="cpu")
    if isinstance(payload, dict):
        if "layers" in payload and isinstance(payload["layers"], dict):
            _ROTATION_PAYLOAD_CACHE[p] = payload
            return payload
        if "rotations" in payload and isinstance(payload["rotations"], dict):
            out = dict(payload)
            out["layers"] = payload["rotations"]
            _ROTATION_PAYLOAD_CACHE[p] = out
            return out
        if all(isinstance(k, str) for k in payload.keys()):
            out = {"layers": payload}
            _ROTATION_PAYLOAD_CACHE[p] = out
            return out
    raise ValueError(f"Unsupported rotation payload format: {type(payload)}")


def _is_pow2(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1) == 0)


def _largest_pow2_divisor(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 1
    out = 1
    while n % 2 == 0:
        out *= 2
        n //= 2
    return max(1, out)


def _select_rotation_block_size(n: int, max_block_size: int = 1024) -> int:
    block = _largest_pow2_divisor(int(n))
    while block > int(max_block_size):
        block //= 2
    return max(1, block)


def _build_hadamard_matrix(order: int) -> torch.Tensor:
    order = int(order)
    if order <= 1:
        return torch.ones((1, 1), dtype=torch.float32)
    if order in _HADAMARD_MATRIX_CACHE:
        return _HADAMARD_MATRIX_CACHE[order]
    if not _is_pow2(order):
        raise ValueError(f"Hadamard order must be power-of-two, got {order}")
    h = torch.tensor([[1.0]], dtype=torch.float32)
    size = 1
    while size < order:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0)
        size *= 2
    h = h / math.sqrt(float(order))
    _HADAMARD_MATRIX_CACHE[order] = h
    return h


def _derive_layer_rotation_seed(base_seed: int, layer_key: str) -> int:
    payload = f"{int(base_seed)}::{layer_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


def _build_internal_rotation_info(layer_key: str, in_features: int, cfg: Hifx4Config) -> Dict[str, Any]:
    return {
        "kind": "internal_hadamard",
        "seed": int(_derive_layer_rotation_seed(int(cfg.ptq_rotation_seed), layer_key)),
        "block_size": int(_select_rotation_block_size(int(in_features))),
        "in_features": int(in_features),
    }


def _build_internal_rotation_runtime(rotation_info: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    n = int(rotation_info["in_features"])
    seed = int(rotation_info["seed"])
    block_size = int(rotation_info["block_size"])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int64)
    signs = (signs * 2 - 1).to(device=device, dtype=dtype)
    perm = torch.randperm(n, generator=gen, dtype=torch.int64).to(device=device)
    had = None
    if block_size > 1:
        had = _build_hadamard_matrix(block_size).to(device=device, dtype=dtype)
    return {
        "signs": signs,
        "perm": perm,
        "hadamard": had,
        "block_size": block_size,
        "in_features": n,
    }


def _apply_internal_rotation_tensor(x: torch.Tensor, rotation_info: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    if runtime is None:
        runtime = _build_internal_rotation_runtime(rotation_info, x.device, x.dtype)
    y = x * runtime["signs"]
    y = y.index_select(-1, runtime["perm"])
    had = runtime.get("hadamard")
    if had is None:
        return y
    block_size = int(runtime["block_size"])
    n = int(runtime["in_features"])
    orig_shape = y.shape
    y = y.reshape(-1, n // block_size, block_size)
    y = torch.matmul(y, had)
    return y.reshape(orig_shape)


def _apply_internal_rotation_minmax(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    rotation_info: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if runtime is None:
        runtime = _build_internal_rotation_runtime(rotation_info, x_min.device, x_min.dtype)
    signs = runtime["signs"]
    out_min = torch.where(signs > 0, x_min, -x_max)
    out_max = torch.where(signs > 0, x_max, -x_min)
    perm = runtime["perm"]
    out_min = out_min.index_select(-1, perm)
    out_max = out_max.index_select(-1, perm)
    had = runtime.get("hadamard")
    if had is None:
        return out_min, out_max
    block_size = int(runtime["block_size"])
    n = int(runtime["in_features"])
    orig_shape = out_min.shape
    out_min = out_min.reshape(-1, n // block_size, block_size)
    out_max = out_max.reshape(-1, n // block_size, block_size)
    pos = torch.clamp(had, min=0.0)
    neg = torch.clamp(had, max=0.0)
    rot_min = torch.matmul(out_min, pos) + torch.matmul(out_max, neg)
    rot_max = torch.matmul(out_max, pos) + torch.matmul(out_min, neg)
    return rot_min.reshape(orig_shape), rot_max.reshape(orig_shape)


def _resolve_input_rotation_entry(
    layer_key: str,
    in_features: int,
    rotation_payload: Dict[str, Any],
) -> Optional[torch.Tensor]:
    layers = rotation_payload.get("layers", {})
    entry = layers.get(layer_key)
    if entry is None:
        return None

    mode = "auto"
    rot: Optional[torch.Tensor] = None
    if isinstance(entry, torch.Tensor):
        rot = entry
    elif isinstance(entry, dict):
        mode = str(entry.get("mode", "auto")).lower()
        matrix = entry.get("matrix", entry.get("rotation", None))
        if isinstance(matrix, torch.Tensor):
            rot = matrix
    if rot is None:
        print(f"[ROT][WARN] skip {layer_key}: invalid payload entry", flush=True)
        return None
    if rot.ndim != 2:
        print(f"[ROT][WARN] skip {layer_key}: rotation must be 2D, got {tuple(rot.shape)}", flush=True)
        return None

    expect_shape = (int(in_features), int(in_features))
    if mode in ("out", "left"):
        print(
            f"[ROT][WARN] skip {layer_key}: PTQ runtime only supports input/right rotations, got mode={mode}",
            flush=True,
        )
        return None
    if rot.shape != expect_shape:
        print(
            f"[ROT][WARN] skip {layer_key}: expect input rotation shape {expect_shape}, got {tuple(rot.shape)}",
            flush=True,
        )
        return None
    return rot.detach().to(torch.float32).cpu()


def _parse_channel_vector(
    data: Any,
    expected_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        vec = data.detach().to(torch.float32).reshape(-1).cpu()
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return None
        vec = torch.tensor([float(x) for x in data], dtype=torch.float32)
    else:
        return None
    if expected_dim is not None and vec.numel() != int(expected_dim):
        return None
    return vec


def _parse_channel_absmax_vector(
    data: Any,
    expected_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    return _parse_channel_vector(data, expected_dim=expected_dim)


def _merge_channel_vectors_max(
    old_data: Any,
    new_data: Any,
    expected_dim: Optional[int] = None,
) -> List[float]:
    old_vec = _parse_channel_vector(old_data, expected_dim=expected_dim)
    new_vec = _parse_channel_vector(new_data, expected_dim=expected_dim)
    if old_vec is None and new_vec is None:
        return []
    if old_vec is None:
        return new_vec.tolist()
    if new_vec is None:
        return old_vec.tolist()
    if old_vec.numel() != new_vec.numel():
        return new_vec.tolist()
    return torch.maximum(old_vec, new_vec).tolist()


def _merge_channel_vectors_min(
    old_data: Any,
    new_data: Any,
    expected_dim: Optional[int] = None,
) -> List[float]:
    old_vec = _parse_channel_vector(old_data, expected_dim=expected_dim)
    new_vec = _parse_channel_vector(new_data, expected_dim=expected_dim)
    if old_vec is None and new_vec is None:
        return []
    if old_vec is None:
        return new_vec.tolist()
    if new_vec is None:
        return old_vec.tolist()
    if old_vec.numel() != new_vec.numel():
        return new_vec.tolist()
    return torch.minimum(old_vec, new_vec).tolist()


def _parse_timestep_branch_channel_dict(
    data: Any,
    expected_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if not isinstance(data, dict):
        return out
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        vec = _parse_channel_vector(value, expected_dim=expected_dim)
        if vec is not None:
            out[key] = vec
    return out


def _serialize_timestep_branch_channel_dict(data: Dict[str, torch.Tensor]) -> Dict[str, List[float]]:
    return {str(k): v.detach().to(torch.float32).cpu().tolist() for k, v in data.items()}


def _merge_timestep_branch_channel_dict(
    old_data: Any,
    new_data: Any,
    reduce: str,
    expected_dim: Optional[int] = None,
) -> Dict[str, List[float]]:
    old_dict = _parse_timestep_branch_channel_dict(old_data, expected_dim=expected_dim)
    new_dict = _parse_timestep_branch_channel_dict(new_data, expected_dim=expected_dim)
    keys = set(old_dict.keys()) | set(new_dict.keys())
    merged: Dict[str, List[float]] = {}
    for key in sorted(keys):
        old_vec = old_dict.get(key)
        new_vec = new_dict.get(key)
        if old_vec is None and new_vec is None:
            continue
        if old_vec is None:
            merged[key] = new_vec.tolist()
            continue
        if new_vec is None:
            merged[key] = old_vec.tolist()
            continue
        if old_vec.numel() != new_vec.numel():
            merged[key] = new_vec.tolist()
            continue
        if reduce == "min":
            merged[key] = torch.minimum(old_vec, new_vec).tolist()
        else:
            merged[key] = torch.maximum(old_vec, new_vec).tolist()
    return merged


def _parse_timestep_channel_matrix(
    data: Any,
    expected_dim: Optional[int] = None,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if not isinstance(data, (list, tuple)):
        return out
    for row in data:
        vec = _parse_channel_vector(row, expected_dim=expected_dim)
        if vec is not None:
            out.append(vec)
    return out


def _serialize_timestep_channel_matrix(data: List[torch.Tensor]) -> List[List[float]]:
    return [v.detach().to(torch.float32).cpu().tolist() for v in data]


def _concat_timestep_channel_matrix(
    old_data: Any,
    new_data: Any,
    expected_dim: Optional[int] = None,
) -> List[List[float]]:
    out: List[List[float]] = []
    for vec in _parse_timestep_channel_matrix(old_data, expected_dim=expected_dim):
        out.append(vec.tolist())
    for vec in _parse_timestep_channel_matrix(new_data, expected_dim=expected_dim):
        out.append(vec.tolist())
    return out


def _concat_timestep_key_sequence(old_data: Any, new_data: Any, target_len: Optional[int] = None) -> List[str]:
    out: List[str] = []
    if isinstance(old_data, (list, tuple)):
        out.extend([str(x) for x in old_data])
    if isinstance(new_data, (list, tuple)):
        out.extend([str(x) for x in new_data])
    if target_len is not None:
        if len(out) > target_len:
            out = out[:target_len]
        elif len(out) < target_len:
            out.extend(["-1::single"] * (target_len - len(out)))
    return out


def _extract_layer_global_minmax_vectors(
    layer_cfg: Dict[str, Any],
    expected_dim: int,
    group_ids: Optional[set[int]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    global_min = _parse_channel_vector(layer_cfg.get("act_min_per_channel_global", None), expected_dim=expected_dim)
    global_max = _parse_channel_vector(layer_cfg.get("act_max_per_channel_global", None), expected_dim=expected_dim)
    if group_ids is None and global_min is not None and global_max is not None:
        return global_min, global_max

    min_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_min_group_branch", layer_cfg.get("act_min_timestep_branch", {})),
        expected_dim=expected_dim,
    )
    max_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_max_group_branch", layer_cfg.get("act_max_timestep_branch", {})),
        expected_dim=expected_dim,
    )
    min_dict = _filter_group_branch_dict(min_dict, group_ids)
    max_dict = _filter_group_branch_dict(max_dict, group_ids)
    if len(min_dict) == 0 and len(max_dict) == 0:
        return global_min, global_max

    keys = sorted(set(min_dict.keys()) | set(max_dict.keys()))
    agg_min: Optional[torch.Tensor] = None
    agg_max: Optional[torch.Tensor] = None
    for key in keys:
        cur_min = min_dict.get(key)
        cur_max = max_dict.get(key)
        if cur_min is None or cur_max is None:
            continue
        agg_min = cur_min.clone() if agg_min is None else torch.minimum(agg_min, cur_min)
        agg_max = cur_max.clone() if agg_max is None else torch.maximum(agg_max, cur_max)
    return agg_min, agg_max


def _build_smoothquant_act_mask_from_minmax(
    layer_key: str,
    layer_cfg: Dict[str, Any],
    expected_dim: int,
    eps: float,
    group_ids: Optional[set[int]] = None,
) -> torch.Tensor:
    act_mask = torch.zeros((expected_dim,), dtype=torch.float32)
    min_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_min_group_branch", layer_cfg.get("act_min_timestep_branch", {})),
        expected_dim=expected_dim,
    )
    max_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_max_group_branch", layer_cfg.get("act_max_timestep_branch", {})),
        expected_dim=expected_dim,
    )
    min_dict = _filter_group_branch_dict(min_dict, group_ids)
    max_dict = _filter_group_branch_dict(max_dict, group_ids)
    keys = sorted(set(min_dict.keys()) | set(max_dict.keys()))
    for key in keys:
        cur_min = min_dict.get(key)
        cur_max = max_dict.get(key)
        if cur_min is None or cur_max is None:
            continue
        cur_amp = torch.maximum(cur_min.abs(), cur_max.abs())
        act_mask = torch.maximum(act_mask, cur_amp)

    global_min, global_max = _extract_layer_global_minmax_vectors(
        layer_cfg,
        expected_dim=expected_dim,
        group_ids=group_ids,
    )
    if global_min is not None and global_max is not None:
        global_amp = torch.maximum(global_min.abs(), global_max.abs())
        act_mask = torch.maximum(act_mask, global_amp)

    act_mask = act_mask.clamp_min(eps)
    if act_mask.numel() != expected_dim or torch.isnan(act_mask).any():
        raise ValueError(
            f"[SQ] Invalid min-max activation stats for layer={layer_key}. "
            "Re-run Stage1 calibration under the current code."
        )
    return act_mask


def _apply_minmax_preprocess(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    channel_mask: Optional[torch.Tensor],
    rotation_info: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_min = x_min.to(torch.float32)
    out_max = x_max.to(torch.float32)
    if channel_mask is not None:
        scale = channel_mask.to(torch.float32)
        out_min = out_min * scale
        out_max = out_max * scale
    if rotation_info is not None:
        kind = str(rotation_info.get("kind", "none"))
        if kind == "external_input_matrix":
            rot = rotation_info["matrix"].to(torch.float32)
            pos = torch.clamp(rot, min=0.0)
            neg = torch.clamp(rot, max=0.0)
            rot_min = torch.matmul(out_min, pos) + torch.matmul(out_max, neg)
            rot_max = torch.matmul(out_max, pos) + torch.matmul(out_min, neg)
            out_min, out_max = rot_min, rot_max
        elif kind == "internal_hadamard":
            out_min, out_max = _apply_internal_rotation_minmax(out_min, out_max, rotation_info)
    return out_min, out_max


def _build_runtime_act_minmax_lookup(
    layer_key: str,
    layer_cfg: Dict[str, Any],
    in_features: int,
    channel_mask: Optional[torch.Tensor],
    rotation_info: Optional[Dict[str, Any]],
    group_ids: Optional[set[int]] = None,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    global_min_vec, global_max_vec = _extract_layer_global_minmax_vectors(
        layer_cfg,
        expected_dim=in_features,
        group_ids=group_ids,
    )
    if global_min_vec is None or global_max_vec is None:
        raise ValueError(
            f"[HIF4] Missing global min-max calibration stats for layer={layer_key}. "
            "Re-run Stage1 calibration under the current code."
        )

    global_min_vec, global_max_vec = _apply_minmax_preprocess(
        global_min_vec,
        global_max_vec,
        channel_mask=channel_mask,
        rotation_info=rotation_info,
    )
    act_min_global = float(global_min_vec.min().item())
    act_max_global = float(global_max_vec.max().item())

    min_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_min_group_branch", layer_cfg.get("act_min_timestep_branch", {})),
        expected_dim=in_features,
    )
    max_dict = _parse_timestep_branch_channel_dict(
        layer_cfg.get("act_max_group_branch", layer_cfg.get("act_max_timestep_branch", {})),
        expected_dim=in_features,
    )
    min_dict = _filter_group_branch_dict(min_dict, group_ids)
    max_dict = _filter_group_branch_dict(max_dict, group_ids)
    act_min_lookup: Dict[str, float] = {}
    act_max_lookup: Dict[str, float] = {}
    for key in sorted(set(min_dict.keys()) | set(max_dict.keys())):
        cur_min = min_dict.get(key)
        cur_max = max_dict.get(key)
        if cur_min is None or cur_max is None:
            continue
        cur_min, cur_max = _apply_minmax_preprocess(
            cur_min,
            cur_max,
            channel_mask=channel_mask,
            rotation_info=rotation_info,
        )
        act_min_lookup[key] = float(cur_min.min().item())
        act_max_lookup[key] = float(cur_max.max().item())
    return act_min_global, act_max_global, act_min_lookup, act_max_lookup


def _apply_rotation_to_tensor(
    x: torch.Tensor,
    rotation_info: Optional[Dict[str, Any]],
) -> torch.Tensor:
    if rotation_info is None:
        return x
    kind = str(rotation_info.get("kind", "none"))
    if kind == "external_input_matrix":
        rot = rotation_info["matrix"].to(device=x.device, dtype=x.dtype)
        return torch.matmul(x, rot)
    if kind == "internal_hadamard":
        runtime = _build_internal_rotation_runtime(rotation_info, x.device, x.dtype)
        return _apply_internal_rotation_tensor(x, rotation_info, runtime=runtime)
    return x


def _build_smoothquant_channel_mask(
    layer_key: str,
    weight_fp32: torch.Tensor,
    layer_cfg: Dict[str, Any],
    cfg: Hifx4Config,
    group_ids: Optional[set[int]] = None,
) -> torch.Tensor:
    in_features = int(weight_fp32.shape[1])
    alpha = float(cfg.ptq_smoothquant_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"[SQ] ptq_smoothquant_alpha must be in [0,1], got {alpha}")
    eps = max(float(cfg.ptq_smoothquant_eps), 1e-8)

    act_mask = _build_smoothquant_act_mask_from_minmax(
        layer_key=layer_key,
        layer_cfg=layer_cfg,
        expected_dim=in_features,
        eps=eps,
        group_ids=group_ids,
    )
    weight_mask = weight_fp32.detach().abs().amax(dim=0).to(torch.float32).cpu().clamp_min(eps)
    channel_mask = (weight_mask.pow(alpha) / act_mask.pow(1.0 - alpha)).clamp_min(eps)
    return channel_mask.to(device=weight_fp32.device, dtype=torch.float32)


def _apply_smoothquant_to_weight(
    layer_key: str,
    weight_fp32: torch.Tensor,
    layer_cfg: Dict[str, Any],
    cfg: Hifx4Config,
    group_ids: Optional[set[int]] = None,
    sq_group_ids: Optional[set[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    if not cfg.ptq_enable_smoothquant:
        return weight_fp32, None, False

    channel_mask = _build_smoothquant_channel_mask(
        layer_key=layer_key,
        weight_fp32=weight_fp32,
        layer_cfg=layer_cfg,
        cfg=cfg,
        group_ids=sq_group_ids,
    )
    scaled = weight_fp32 / channel_mask.unsqueeze(0)
    return scaled, channel_mask.detach().cpu(), True


class PTQCalibrationObserverLinear(nn.Module):
    def __init__(self, name: str, linear: nn.Linear, cfg: Hifx4Config):
        super().__init__()
        self.name = name
        self.linear = linear
        self.cfg = cfg
        self.register_buffer("act_min_global", torch.tensor(float("inf"), dtype=torch.float32), persistent=False)
        self.register_buffer("act_max_global", torch.tensor(float("-inf"), dtype=torch.float32), persistent=False)
        self.register_buffer("_act_min_per_channel", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_act_max_per_channel", torch.empty(0, dtype=torch.float32), persistent=False)
        self.observe_steps = 0
        self._group_branch_min: Dict[str, torch.Tensor] = {}
        self._group_branch_max: Dict[str, torch.Tensor] = {}
        # ViDiT-Q style per-call activation trajectory: [T, C], where T follows forward call order.
        self._act_absmax_timestep_seq: List[torch.Tensor] = []
        self._act_absmax_timestep_keys: List[str] = []

    def reset_running_stats(self) -> None:
        self.act_min_global.fill_(float("inf"))
        self.act_max_global.fill_(float("-inf"))
        self._act_min_per_channel = torch.empty(0, dtype=torch.float32, device=self._act_min_per_channel.device)
        self._act_max_per_channel = torch.empty(0, dtype=torch.float32, device=self._act_max_per_channel.device)
        self.observe_steps = 0
        self._group_branch_min = {}
        self._group_branch_max = {}
        self._act_absmax_timestep_seq = []
        self._act_absmax_timestep_keys = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cur_channel_min = torch.empty(0, dtype=torch.float32)
            cur_channel_max = torch.empty(0, dtype=torch.float32)
            cur_min_t = x.detach().amin().to(torch.float32)
            cur_max_t = x.detach().amax().to(torch.float32)
            if _dist_is_initialized():
                backend = dist.get_backend()
                if backend == "nccl" and cur_min_t.device.type != "cuda":
                    cur_min_t = cur_min_t.to(
                        torch.device("cuda", torch.cuda.current_device())
                    )
                    cur_max_t = cur_max_t.to(
                        torch.device("cuda", torch.cuda.current_device())
                    )
                cur_min_t = _dist_all_reduce_min_scalar(cur_min_t)
                cur_max_t = _dist_all_reduce_max_scalar(cur_max_t)
            cur_min = float(cur_min_t.detach().cpu().item())
            cur_max = float(cur_max_t.detach().cpu().item())
            if cur_min < float(self.act_min_global.item()):
                self.act_min_global.fill_(cur_min)
            if cur_max > float(self.act_max_global.item()):
                self.act_max_global.fill_(cur_max)

            if x.ndim >= 1 and x.shape[-1] > 0:
                x_flat = x.detach().reshape(-1, x.shape[-1])
                cur_channel_min_t = x_flat.amin(dim=0).to(torch.float32)
                cur_channel_max_t = x_flat.amax(dim=0).to(torch.float32)
                if _dist_is_initialized():
                    backend = dist.get_backend()
                    if backend == "nccl" and cur_channel_min_t.device.type != "cuda":
                        cur_channel_min_t = cur_channel_min_t.to(
                            torch.device("cuda", torch.cuda.current_device())
                        )
                        cur_channel_max_t = cur_channel_max_t.to(
                            torch.device("cuda", torch.cuda.current_device())
                        )
                    cur_channel_min_t = _dist_all_reduce_min_tensor(cur_channel_min_t)
                    cur_channel_max_t = _dist_all_reduce_max_tensor(cur_channel_max_t)
                cur_channel_min = cur_channel_min_t.detach().to(torch.float32)
                cur_channel_max = cur_channel_max_t.detach().to(torch.float32)
                # Keep runtime merge on the same device as observer buffers.
                # Export path still converts to CPU explicitly.
                min_target_device = self._act_min_per_channel.device
                max_target_device = self._act_max_per_channel.device
                if cur_channel_min.device != min_target_device:
                    cur_channel_min = cur_channel_min.to(min_target_device)
                if cur_channel_max.device != max_target_device:
                    cur_channel_max = cur_channel_max.to(max_target_device)
                if self._act_min_per_channel.numel() == 0:
                    self._act_min_per_channel = cur_channel_min
                elif self._act_min_per_channel.numel() == cur_channel_min.numel():
                    self._act_min_per_channel = torch.minimum(
                        self._act_min_per_channel,
                        cur_channel_min,
                    )
                else:
                    self._act_min_per_channel = cur_channel_min
                if self._act_max_per_channel.numel() == 0:
                    self._act_max_per_channel = cur_channel_max
                elif self._act_max_per_channel.numel() == cur_channel_max.numel():
                    self._act_max_per_channel = torch.maximum(
                        self._act_max_per_channel,
                        cur_channel_max,
                    )
                else:
                    self._act_max_per_channel = cur_channel_max

            snap = GLOBAL_QUANT_CONTEXT.snapshot()
            branch = _sanitize_branch(str(snap.get("branch", "single")))
            group_id = int(snap.get("group_id", -1))
            key = _tb_key(group_id, branch)
            prev_min = self._group_branch_min.get(key)
            prev_max = self._group_branch_max.get(key)
            cur_channel_min_cpu = cur_channel_min.detach().to(torch.float32).cpu()
            cur_channel_max_cpu = cur_channel_max.detach().to(torch.float32).cpu()
            cur_channel_absmax_cpu = torch.maximum(cur_channel_min_cpu.abs(), cur_channel_max_cpu.abs())
            self._group_branch_min[key] = (
                cur_channel_min_cpu.clone()
                if prev_min is None or prev_min.numel() != cur_channel_min_cpu.numel()
                else torch.minimum(prev_min, cur_channel_min_cpu)
            )
            self._group_branch_max[key] = (
                cur_channel_max_cpu.clone()
                if prev_max is None or prev_max.numel() != cur_channel_max_cpu.numel()
                else torch.maximum(prev_max, cur_channel_max_cpu)
            )
            if self.cfg.ptq_store_act_absmax_seq:
                self._act_absmax_timestep_seq.append(cur_channel_absmax_cpu.clone())
                self._act_absmax_timestep_keys.append(str(key))
            self.observe_steps += 1
        return self.linear(x)

    def export_group_branch_min(self) -> Dict[str, List[float]]:
        return _serialize_timestep_branch_channel_dict(self._group_branch_min)

    def export_group_branch_max(self) -> Dict[str, List[float]]:
        return _serialize_timestep_branch_channel_dict(self._group_branch_max)

    def import_group_branch_min(self, data: Dict[str, Any]) -> None:
        parsed = _parse_timestep_branch_channel_dict(data)
        merged = _parse_timestep_branch_channel_dict(self._group_branch_min)
        for key, value in parsed.items():
            old_value = merged.get(key)
            merged[key] = value if old_value is None or old_value.numel() != value.numel() else torch.minimum(old_value, value)
        self._group_branch_min = merged

    def import_group_branch_max(self, data: Dict[str, Any]) -> None:
        parsed = _parse_timestep_branch_channel_dict(data)
        merged = _parse_timestep_branch_channel_dict(self._group_branch_max)
        for key, value in parsed.items():
            old_value = merged.get(key)
            merged[key] = value if old_value is None or old_value.numel() != value.numel() else torch.maximum(old_value, value)
        self._group_branch_max = merged

    def export_act_absmax_timestep_seq(self) -> List[List[float]]:
        return _serialize_timestep_channel_matrix(self._act_absmax_timestep_seq)

    def export_act_absmax_timestep_keys(self) -> List[str]:
        return [str(k) for k in self._act_absmax_timestep_keys]

    def import_act_absmax_timestep_seq(self, data: Any, keys: Optional[Any] = None) -> None:
        parsed = _parse_timestep_channel_matrix(data)
        if len(parsed) == 0:
            return
        self._act_absmax_timestep_seq.extend([v.detach().to(torch.float32).cpu() for v in parsed])
        parsed_keys: List[str] = [str(x) for x in keys] if isinstance(keys, (list, tuple)) else []
        if len(parsed_keys) < len(parsed):
            parsed_keys.extend(["-1::single"] * (len(parsed) - len(parsed_keys)))
        self._act_absmax_timestep_keys.extend(parsed_keys[: len(parsed)])

    def export_act_min_per_channel(self) -> List[float]:
        if self._act_min_per_channel.numel() == 0:
            return []
        return self._act_min_per_channel.detach().to(torch.float32).cpu().tolist()

    def export_act_max_per_channel(self) -> List[float]:
        if self._act_max_per_channel.numel() == 0:
            return []
        return self._act_max_per_channel.detach().to(torch.float32).cpu().tolist()

    def import_act_min_per_channel(self, data: Any) -> None:
        vec = _parse_channel_vector(data)
        if vec is None:
            return
        vec = vec.detach().to(torch.float32)
        target_device = self._act_min_per_channel.device
        if vec.device != target_device:
            vec = vec.to(target_device)
        if self._act_min_per_channel.numel() == 0:
            self._act_min_per_channel = vec
            return
        if self._act_min_per_channel.numel() != vec.numel():
            self._act_min_per_channel = vec
            return
        self._act_min_per_channel = torch.minimum(self._act_min_per_channel, vec)

    def import_act_max_per_channel(self, data: Any) -> None:
        vec = _parse_channel_vector(data)
        if vec is None:
            return
        vec = vec.detach().to(torch.float32)
        target_device = self._act_max_per_channel.device
        if vec.device != target_device:
            vec = vec.to(target_device)
        if self._act_max_per_channel.numel() == 0:
            self._act_max_per_channel = vec
            return
        if self._act_max_per_channel.numel() != vec.numel():
            self._act_max_per_channel = vec
            return
        self._act_max_per_channel = torch.maximum(self._act_max_per_channel, vec)


_CALIB_OBSERVERS: Dict[str, Dict[str, Any]] = {}


def _register_calibration_observers(
    model: nn.Module,
    cfg: Hifx4Config,
    model_tag: str,
    model_signature: str,
) -> Tuple[int, int, Dict[str, Any]]:
    name_to_module = dict(model.named_modules())
    observers: Dict[str, PTQCalibrationObserverLinear] = {}
    total_linear = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    replaced = 0
    for name, module in _iter_target_linears(model, cfg):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = name_to_module[parent_name] if parent_name else model
        obs = PTQCalibrationObserverLinear(name=name, linear=module, cfg=cfg)
        setattr(parent, child_name, obs)
        observers[name] = obs
        replaced += 1

    paths = _artifact_paths(cfg, model_tag)
    _CALIB_OBSERVERS[model_tag] = {
        "cfg": cfg,
        "model_signature": model_signature,
        "paths": paths,
        "observers": observers,
        "prompt_runs": 0,
        "grouping": {},
    }

    report = {
        "model_tag": model_tag,
        "target_layers": replaced,
        "hit_layers": replaced,
        "miss_layers": 0,
        "artifact_dir": paths["dir"],
    }
    return replaced, total_linear, report


def mark_calibration_prompt_run() -> int:
    latest_prompt_runs = 0
    for _, state in _CALIB_OBSERVERS.items():
        state["prompt_runs"] = int(state.get("prompt_runs", 0)) + 1
        latest_prompt_runs = max(latest_prompt_runs, int(state["prompt_runs"]))
    return latest_prompt_runs


def _serialize_observer_layer_state(obs: PTQCalibrationObserverLinear) -> Dict[str, Any]:
    return {
        "act_min_global": float(obs.act_min_global.item()),
        "act_max_global": float(obs.act_max_global.item()),
        "act_min_group_branch": obs.export_group_branch_min(),
        "act_max_group_branch": obs.export_group_branch_max(),
        "act_absmax_timestep_seq": obs.export_act_absmax_timestep_seq(),
        "act_absmax_timestep_keys": obs.export_act_absmax_timestep_keys(),
        "act_min_per_channel_global": obs.export_act_min_per_channel(),
        "act_max_per_channel_global": obs.export_act_max_per_channel(),
        "observe_steps": int(obs.observe_steps),
    }


def _merge_calibration_layer_state(
    old_layer: Dict[str, Any],
    new_layer: Dict[str, Any],
    expected_dim: Optional[int] = None,
    store_act_absmax_seq: bool = False,
) -> Dict[str, Any]:
    merged_act_absmax_seq = []
    merged_act_absmax_keys: List[str] = []
    if store_act_absmax_seq:
        merged_act_absmax_seq = _concat_timestep_channel_matrix(
            old_layer.get("act_absmax_timestep_seq", []),
            new_layer.get("act_absmax_timestep_seq", []),
            expected_dim=expected_dim,
        )
        merged_act_absmax_keys = _concat_timestep_key_sequence(
            old_layer.get("act_absmax_timestep_keys", []),
            new_layer.get("act_absmax_timestep_keys", []),
            target_len=len(merged_act_absmax_seq),
        )
    return {
        "act_min_global": min(
            float(old_layer.get("act_min_global", float("inf"))),
            float(new_layer.get("act_min_global", float("inf"))),
        ),
        "act_max_global": max(
            float(old_layer.get("act_max_global", float("-inf"))),
            float(new_layer.get("act_max_global", float("-inf"))),
        ),
        "act_min_group_branch": _merge_timestep_branch_channel_dict(
            old_layer.get("act_min_group_branch", old_layer.get("act_min_timestep_branch", {})),
            new_layer.get("act_min_group_branch", new_layer.get("act_min_timestep_branch", {})),
            reduce="min",
            expected_dim=expected_dim,
        ),
        "act_max_group_branch": _merge_timestep_branch_channel_dict(
            old_layer.get("act_max_group_branch", old_layer.get("act_max_timestep_branch", {})),
            new_layer.get("act_max_group_branch", new_layer.get("act_max_timestep_branch", {})),
            reduce="max",
            expected_dim=expected_dim,
        ),
        "act_absmax_timestep_seq": merged_act_absmax_seq,
        "act_absmax_timestep_keys": merged_act_absmax_keys,
        "act_min_per_channel_global": _merge_channel_vectors_min(
            old_layer.get("act_min_per_channel_global", None),
            new_layer.get("act_min_per_channel_global", None),
            expected_dim=expected_dim,
        ),
        "act_max_per_channel_global": _merge_channel_vectors_max(
            old_layer.get("act_max_per_channel_global", None),
            new_layer.get("act_max_per_channel_global", None),
            expected_dim=expected_dim,
        ),
        "observe_steps": int(old_layer.get("observe_steps", 0)) + int(new_layer.get("observe_steps", 0)),
    }


def _reset_calibration_observers_running_stats() -> None:
    for state in _CALIB_OBSERVERS.values():
        observers: Dict[str, PTQCalibrationObserverLinear] = state["observers"]
        for obs in observers.values():
            obs.reset_running_stats()


def save_calibration_shard_stats(shard_idx: int, num_shards: int, clear_after_save: bool = False) -> None:
    for model_tag, state in _CALIB_OBSERVERS.items():
        cfg: Hifx4Config = state["cfg"]
        observers: Dict[str, PTQCalibrationObserverLinear] = state["observers"]
        paths: Dict[str, str] = state["paths"]
        model_signature: str = state["model_signature"]
        shard_dir = Path(paths["dir"]) / "calibration_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.pt"

        existing_payload: Optional[Dict[str, Any]] = None
        existing_layers: Dict[str, Dict[str, Any]] = {}
        existing_grouping: Dict[str, Any] = {}
        if _dist_is_main_process() and shard_path.exists():
            existing_payload = torch.load(str(shard_path), map_location="cpu")
            if existing_payload.get("meta", {}).get("model_signature") == model_signature:
                existing_layers = dict(existing_payload.get("layers", {}))
                existing_grouping = dict(existing_payload.get("meta", {}).get("grouping", {}))

        layer_stats: Dict[str, Dict[str, Any]] = {}
        for name, obs in observers.items():
            layer_key = _normalize_linear_name(name)
            expected_dim = int(obs.linear.in_features)
            layer_stats[layer_key] = _merge_calibration_layer_state(
                existing_layers.get(layer_key, {}),
                _serialize_observer_layer_state(obs),
                expected_dim=expected_dim,
                store_act_absmax_seq=bool(cfg.ptq_store_act_absmax_seq),
            )

        grouping_meta = GLOBAL_QUANT_CONTEXT.export_grouping_metadata(model_tag=model_tag)
        if not grouping_meta:
            grouping_meta = existing_grouping
        state["grouping"] = grouping_meta
        payload = {
            "meta": {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "model_tag": model_tag,
                "model_signature": model_signature,
                "shard_idx": int(shard_idx),
                "num_shards": int(num_shards),
                "prompt_runs": int(state.get("prompt_runs", 0)),
                "timestamp_utc": _utcnow(),
                "grouping": grouping_meta,
            },
            "layers": layer_stats,
        }
        _dist_barrier()
        if _dist_is_main_process():
            _atomic_torch_save(payload, str(shard_path))
            print(f"[HIF4][Stage1] Saved shard for {model_tag} -> {shard_path}", flush=True)
    _dist_barrier()
    if clear_after_save:
        _reset_calibration_observers_running_stats()


def load_calibration_shard_stats(shard_idx: int, num_shards: int, allow_missing: bool = False) -> int:
    loaded_prompt_runs = 0
    for model_tag, state in _CALIB_OBSERVERS.items():
        cfg: Hifx4Config = state["cfg"]
        observers: Dict[str, PTQCalibrationObserverLinear] = state["observers"]
        paths: Dict[str, str] = state["paths"]
        model_signature: str = state["model_signature"]
        shard_dir = Path(paths["dir"]) / "calibration_shards"
        shard_path = shard_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.pt"
        shard_payload: Optional[Dict[str, Any]] = None
        shard_prompt_runs = 0
        error_message = ""
        if _dist_is_main_process():
            if not shard_path.exists():
                if not allow_missing:
                    error_message = f"Missing shard stats file: {shard_path}"
            else:
                shard_payload = torch.load(str(shard_path), map_location="cpu")
                shard_prompt_runs = int(shard_payload.get("meta", {}).get("prompt_runs", 0))
        if _dist_is_initialized():
            payload = _dist_broadcast_object_from_main(
                {
                    "shard_payload": shard_payload,
                    "shard_prompt_runs": shard_prompt_runs,
                    "error_message": error_message,
                }
            )
            shard_payload = payload.get("shard_payload", None)
            shard_prompt_runs = int(payload.get("shard_prompt_runs", 0))
            error_message = str(payload.get("error_message", ""))
        if error_message:
            raise FileNotFoundError(error_message)
        if shard_payload is None:
            continue

        shard_meta = shard_payload.get("meta", {})
        if shard_meta.get("model_signature") != model_signature:
            raise RuntimeError(
                f"Shard signature mismatch for {model_tag}: "
                f"{shard_meta.get('model_signature')} != {model_signature}"
            )
        state["grouping"] = dict(shard_meta.get("grouping", {}))

        shard_layers = shard_payload.get("layers", {})
        for name, obs in observers.items():
            layer_key = _normalize_linear_name(name)
            layer_state = shard_layers.get(layer_key, shard_layers.get(name))
            if layer_state is None:
                continue
            obs.act_min_global.fill_(float(layer_state.get("act_min_global", float("inf"))))
            obs.act_max_global.fill_(float(layer_state.get("act_max_global", float("-inf"))))
            obs.import_group_branch_min(
                layer_state.get("act_min_group_branch", layer_state.get("act_min_timestep_branch", {}))
            )
            obs.import_group_branch_max(
                layer_state.get("act_max_group_branch", layer_state.get("act_max_timestep_branch", {}))
            )
            if cfg.ptq_store_act_absmax_seq:
                obs.import_act_absmax_timestep_seq(
                    layer_state.get("act_absmax_timestep_seq", []),
                    keys=layer_state.get("act_absmax_timestep_keys", []),
                )
            obs.import_act_min_per_channel(layer_state.get("act_min_per_channel_global", None))
            obs.import_act_max_per_channel(layer_state.get("act_max_per_channel_global", None))
            obs.observe_steps = int(layer_state.get("observe_steps", 0))

        state["prompt_runs"] = int(shard_prompt_runs)
        loaded_prompt_runs = max(loaded_prompt_runs, int(state["prompt_runs"]))
        if _dist_is_main_process():
            print(
                f"[HIF4][Stage1] Loaded shard stats for {model_tag} from {shard_path} "
                f"(prompt_runs={state['prompt_runs']})",
                flush=True,
            )
    return loaded_prompt_runs


def load_merged_calibration_shard_stats(num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")

    for model_tag, state in _CALIB_OBSERVERS.items():
        cfg: Hifx4Config = state["cfg"]
        observers: Dict[str, PTQCalibrationObserverLinear] = state["observers"]
        paths: Dict[str, str] = state["paths"]
        model_signature: str = state["model_signature"]
        shard_dir = Path(paths["dir"]) / "calibration_shards"

        merged_prompt_runs = 0
        merged_grouping: Dict[str, Any] = {}
        merged_global_min: Dict[str, float] = {}
        merged_global_max: Dict[str, float] = {}
        merged_tb_min: Dict[str, Dict[str, List[float]]] = {}
        merged_tb_max: Dict[str, Dict[str, List[float]]] = {}
        merged_act_absmax_seq: Dict[str, List[List[float]]] = {}
        merged_act_absmax_keys: Dict[str, List[str]] = {}
        merged_channel_min: Dict[str, List[float]] = {}
        merged_channel_max: Dict[str, List[float]] = {}
        merged_steps: Dict[str, int] = {}
        error_message = ""

        if _dist_is_main_process():
            try:
                for i in range(num_shards):
                    shard_path = shard_dir / f"shard_{i:02d}_of_{num_shards:02d}.pt"
                    if not shard_path.exists():
                        raise FileNotFoundError(f"Missing shard stats file: {shard_path}")
                    shard_payload = torch.load(str(shard_path), map_location="cpu")
                    shard_meta = shard_payload.get("meta", {})
                    if shard_meta.get("model_signature") != model_signature:
                        raise RuntimeError(
                            f"Shard signature mismatch for {model_tag}: "
                            f"{shard_meta.get('model_signature')} != {model_signature}"
                        )
                    if not merged_grouping:
                        merged_grouping = dict(shard_meta.get("grouping", {}))
                    merged_prompt_runs += int(shard_meta.get("prompt_runs", 0))

                    for name, layer_state in shard_payload.get("layers", {}).items():
                        layer_key = _normalize_linear_name(name)
                        g_min = float(layer_state.get("act_min_global", float("inf")))
                        g_max = float(layer_state.get("act_max_global", float("-inf")))
                        merged_global_min[layer_key] = min(
                            float(merged_global_min.get(layer_key, float("inf"))),
                            g_min,
                        )
                        merged_global_max[layer_key] = max(
                            float(merged_global_max.get(layer_key, float("-inf"))),
                            g_max,
                        )

                        merged_tb_min[layer_key] = _merge_timestep_branch_channel_dict(
                            merged_tb_min.get(layer_key, {}),
                            layer_state.get("act_min_group_branch", layer_state.get("act_min_timestep_branch", {})),
                            reduce="min",
                        )
                        merged_tb_max[layer_key] = _merge_timestep_branch_channel_dict(
                            merged_tb_max.get(layer_key, {}),
                            layer_state.get("act_max_group_branch", layer_state.get("act_max_timestep_branch", {})),
                            reduce="max",
                        )
                        if cfg.ptq_store_act_absmax_seq:
                            merged_act_absmax_seq[layer_key] = _concat_timestep_channel_matrix(
                                merged_act_absmax_seq.get(layer_key, []),
                                layer_state.get("act_absmax_timestep_seq", []),
                            )
                            merged_act_absmax_keys[layer_key] = _concat_timestep_key_sequence(
                                merged_act_absmax_keys.get(layer_key, []),
                                layer_state.get("act_absmax_timestep_keys", []),
                                target_len=len(merged_act_absmax_seq[layer_key]),
                            )

                        ch_min = _parse_channel_vector(layer_state.get("act_min_per_channel_global", None))
                        ch_max = _parse_channel_vector(layer_state.get("act_max_per_channel_global", None))
                        if ch_min is not None:
                            merged_channel_min[layer_key] = _merge_channel_vectors_min(
                                merged_channel_min.get(layer_key, []),
                                ch_min,
                            )
                        if ch_max is not None:
                            merged_channel_max[layer_key] = _merge_channel_vectors_max(
                                merged_channel_max.get(layer_key, []),
                                ch_max,
                            )

                        merged_steps[layer_key] = int(merged_steps.get(layer_key, 0)) + int(
                            layer_state.get("observe_steps", 0)
                        )
            except Exception as exc:
                error_message = str(exc)
        if _dist_is_initialized():
            payload = _dist_broadcast_object_from_main(
                {
                    "merged_prompt_runs": merged_prompt_runs,
                    "merged_grouping": merged_grouping,
                    "merged_global_min": merged_global_min,
                    "merged_global_max": merged_global_max,
                    "merged_tb_min": merged_tb_min,
                    "merged_tb_max": merged_tb_max,
                    "merged_act_absmax_seq": merged_act_absmax_seq,
                    "merged_act_absmax_keys": merged_act_absmax_keys,
                    "merged_channel_min": merged_channel_min,
                    "merged_channel_max": merged_channel_max,
                    "merged_steps": merged_steps,
                    "error_message": error_message,
                }
            )
            merged_prompt_runs = int(payload.get("merged_prompt_runs", 0))
            merged_grouping = dict(payload.get("merged_grouping", {}))
            merged_global_min = dict(payload.get("merged_global_min", {}))
            merged_global_max = dict(payload.get("merged_global_max", {}))
            merged_tb_min = dict(payload.get("merged_tb_min", {}))
            merged_tb_max = dict(payload.get("merged_tb_max", {}))
            merged_act_absmax_seq = dict(payload.get("merged_act_absmax_seq", {}))
            merged_act_absmax_keys = dict(payload.get("merged_act_absmax_keys", {}))
            merged_channel_min = dict(payload.get("merged_channel_min", {}))
            merged_channel_max = dict(payload.get("merged_channel_max", {}))
            merged_steps = dict(payload.get("merged_steps", {}))
            error_message = str(payload.get("error_message", ""))
        if error_message:
            raise RuntimeError(error_message)
        state["grouping"] = merged_grouping

        for name, obs in observers.items():
            layer_key = _normalize_linear_name(name)
            if layer_key in merged_global_min:
                obs.act_min_global.fill_(float(merged_global_min[layer_key]))
            if layer_key in merged_global_max:
                obs.act_max_global.fill_(float(merged_global_max[layer_key]))
            if layer_key in merged_tb_min:
                obs.import_group_branch_min(merged_tb_min[layer_key])
            if layer_key in merged_tb_max:
                obs.import_group_branch_max(merged_tb_max[layer_key])
            if cfg.ptq_store_act_absmax_seq and layer_key in merged_act_absmax_seq:
                obs.import_act_absmax_timestep_seq(
                    merged_act_absmax_seq[layer_key],
                    keys=merged_act_absmax_keys.get(layer_key, []),
                )
            if layer_key in merged_channel_min:
                obs.import_act_min_per_channel(merged_channel_min[layer_key])
            if layer_key in merged_channel_max:
                obs.import_act_max_per_channel(merged_channel_max[layer_key])
            obs.observe_steps = int(merged_steps.get(layer_key, 0))

        state["prompt_runs"] = merged_prompt_runs
        if _dist_is_main_process():
            print(
                f"[HIF4][Stage1] Loaded and merged {num_shards} shards for {model_tag} "
                f"(prompt_runs={merged_prompt_runs})",
                flush=True,
            )


def _merge_layer_timestep_branch_stats(
    old_tb: Dict[str, Any],
    new_tb: Dict[str, Any],
    reduce: str,
) -> Dict[str, List[float]]:
    return _merge_timestep_branch_channel_dict(old_tb, new_tb, reduce=reduce)


def finalize_calibration_observers() -> None:
    _dist_barrier()
    is_main_process = _dist_is_main_process()
    finished_tags: List[str] = []
    for model_tag, state in _CALIB_OBSERVERS.items():
        if not is_main_process:
            finished_tags.append(model_tag)
            continue
        cfg: Hifx4Config = state["cfg"]
        observers: Dict[str, PTQCalibrationObserverLinear] = state["observers"]
        paths: Dict[str, str] = state["paths"]
        model_signature: str = state["model_signature"]
        current_prompt_runs = int(state.get("prompt_runs", 0))
        grouping_meta = dict(state.get("grouping", {}))

        old_payload: Optional[Dict[str, Any]] = None
        if os.path.exists(paths["calibration"]):
            try:
                old_payload = _load_existing_calibration_payload(
                    paths["calibration"],
                    expected_model_signature=model_signature,
                )
            except Exception as exc:
                print(
                    f"[HIF4][Stage1][WARN] Ignore unreadable calibration for {model_tag}: "
                    f"{paths['calibration']} ({exc})",
                    flush=True,
                )
                old_payload = None

        old_prompt_runs = 0
        if isinstance(old_payload, dict):
            old_prompt_runs = int(old_payload.get("meta", {}).get("summary", {}).get("prompt_runs", 0))
            if current_prompt_runs > 0 and old_prompt_runs >= current_prompt_runs:
                print(
                    f"[HIF4][Stage1] Reuse finalized calibration for {model_tag} <- "
                    f"{paths['calibration']} (prompt_runs={old_prompt_runs})",
                    flush=True,
                )
                finished_tags.append(model_tag)
                continue

        layers: Dict[str, Dict[str, Any]] = {}
        total_layers = len(observers)
        for idx, (name, obs) in enumerate(observers.items(), start=1):
            layer_key = _normalize_linear_name(name)
            linear = obs.linear

            old_layer = {}
            if isinstance(old_payload, dict):
                old_layer = old_payload.get("layers", {}).get(layer_key, {})

            old_global_min = float(old_layer.get("act_min_global", float("inf")))
            old_global_max = float(old_layer.get("act_max_global", float("-inf")))
            new_global_min = float(obs.act_min_global.item())
            new_global_max = float(obs.act_max_global.item())
            global_min = min(old_global_min, new_global_min)
            global_max = max(old_global_max, new_global_max)

            old_tb_min = old_layer.get("act_min_group_branch", old_layer.get("act_min_timestep_branch", {}))
            new_tb_min = obs.export_group_branch_min()
            merged_tb_min = _merge_layer_timestep_branch_stats(old_tb_min, new_tb_min, reduce="min")
            old_tb_max = old_layer.get("act_max_group_branch", old_layer.get("act_max_timestep_branch", {}))
            new_tb_max = obs.export_group_branch_max()
            merged_tb_max = _merge_layer_timestep_branch_stats(old_tb_max, new_tb_max, reduce="max")
            merged_act_absmax_seq: List[List[float]] = []
            merged_act_absmax_keys: List[str] = []
            if cfg.ptq_store_act_absmax_seq:
                old_act_absmax_seq = old_layer.get("act_absmax_timestep_seq", [])
                old_act_absmax_keys = old_layer.get("act_absmax_timestep_keys", [])
                new_act_absmax_seq = obs.export_act_absmax_timestep_seq()
                new_act_absmax_keys = obs.export_act_absmax_timestep_keys()
                merged_act_absmax_seq = _concat_timestep_channel_matrix(
                    old_act_absmax_seq,
                    new_act_absmax_seq,
                    expected_dim=int(linear.in_features),
                )
                merged_act_absmax_keys = _concat_timestep_key_sequence(
                    old_act_absmax_keys,
                    new_act_absmax_keys,
                    target_len=len(merged_act_absmax_seq),
                )
            merged_channel_min = _merge_channel_vectors_min(
                old_layer.get("act_min_per_channel_global", None),
                obs.export_act_min_per_channel(),
                expected_dim=int(linear.in_features),
            )
            merged_channel_max = _merge_channel_vectors_max(
                old_layer.get("act_max_per_channel_global", None),
                obs.export_act_max_per_channel(),
                expected_dim=int(linear.in_features),
            )

            layers[layer_key] = {
                "shape": [int(linear.out_features), int(linear.in_features)],
                "act_min_global": float(global_min),
                "act_max_global": float(global_max),
                "act_min_group_branch": merged_tb_min,
                "act_max_group_branch": merged_tb_max,
                "act_absmax_timestep_seq": merged_act_absmax_seq,
                "act_absmax_timestep_keys": merged_act_absmax_keys,
                "act_min_per_channel_global": merged_channel_min,
                "act_max_per_channel_global": merged_channel_max,
                "observe_steps": int(obs.observe_steps),
            }

            if idx % 50 == 0 or idx == total_layers:
                print(f"[HIF4][Stage1] finalize {model_tag}: {idx}/{total_layers}", flush=True)

        meta = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "stage": "calibrate",
            "method": "hif4_fp4_minmax_sqrot",
            "qtype": cfg.qtype,
            "timestamp_utc": _utcnow(),
            "model_tag": model_tag,
            "model_signature": model_signature,
            "grouping": grouping_meta,
            "summary": {
                "target_layers": len(layers),
                "prompt_runs": old_prompt_runs + int(state.get("prompt_runs", 0)),
                "timestep_count": int(cfg.timestep_count),
            },
            "defaults": {
                "act_quant_mode": cfg.act_quant_mode,
                "activation_grouping": str(grouping_meta.get("mode", _DEFAULT_ACTIVATION_GROUPING)),
                "act_clip_percentile": cfg.act_clip_percentile,
                "legacy_weight_group_size_compat": cfg.weight_group_size,
                "keep_blocks": list(cfg.keep_blocks),
                "lookup_stat": "minmax",
            },
        }
        payload = {"meta": meta, "layers": layers}
        _atomic_torch_save(payload, paths["calibration"])
        _save_manifest(
            paths["manifest"],
            {
                "schema": ARTIFACT_SCHEMA_VERSION,
                "method": "hif4_fp4_minmax_sqrot",
                "qtype": cfg.qtype,
                "model_tag": model_tag,
                "model_signature": model_signature,
                "calibration": os.path.basename(paths["calibration"]),
                "timestamp_utc": meta["timestamp_utc"],
                "prompt_runs": meta["summary"]["prompt_runs"],
                "timestep_count": meta["summary"]["timestep_count"],
            },
        )
        print(f"[HIF4][Stage1] Saved calibration for {model_tag} -> {paths['calibration']}", flush=True)
        del payload
        del layers
        del old_payload
        finished_tags.append(model_tag)

    _dist_barrier()
    for tag in finished_tags:
        _CALIB_OBSERVERS.pop(tag, None)


def _run_ptq_calibration(
    model: nn.Module,
    cfg: Hifx4Config,
    model_tag: str,
    model_signature: str,
    calib_path: str,
    manifest_path: str,
) -> Dict[str, Any]:
    # Fallback mode for stage=all (no real forward stats): infer conservative stats from weights.
    layers: Dict[str, Dict[str, Any]] = {}
    for name, module in _iter_target_linears(model, cfg):
        layer_key = _normalize_linear_name(name)
        weight_absmax = float(module.weight.detach().to(torch.float32).abs().amax().item())
        weight_absmax_per_channel = (
            module.weight.detach().to(torch.float32).abs().amax(dim=0).cpu().tolist()
        )
        layers[layer_key] = {
            "shape": [int(module.out_features), int(module.in_features)],
            "act_min_global": -weight_absmax,
            "act_max_global": weight_absmax,
            "act_min_group_branch": {},
            "act_max_group_branch": {},
            "act_absmax_timestep_seq": [],
            "act_absmax_timestep_keys": [],
            "act_min_per_channel_global": [-float(x) for x in weight_absmax_per_channel],
            "act_max_per_channel_global": [float(x) for x in weight_absmax_per_channel],
            "observe_steps": 0,
        }

    meta = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "calibrate",
        "method": "hif4_fp4_minmax_sqrot",
        "qtype": cfg.qtype,
        "timestamp_utc": _utcnow(),
        "model_tag": model_tag,
        "model_signature": model_signature,
        "grouping": {
            "mode": _DEFAULT_ACTIVATION_GROUPING,
            "expected_steps": int(cfg.timestep_count),
            "model_group_ranges": {
                model_tag: _default_group_ranges_for_model(model_tag, int(cfg.timestep_count)),
            },
        },
        "summary": {
            "target_layers": len(layers),
            "prompt_runs": 0,
            "timestep_count": int(cfg.timestep_count),
        },
        "defaults": {
            "act_quant_mode": cfg.act_quant_mode,
            "activation_grouping": _DEFAULT_ACTIVATION_GROUPING,
            "act_clip_percentile": cfg.act_clip_percentile,
            "legacy_weight_group_size_compat": cfg.weight_group_size,
            "keep_blocks": list(cfg.keep_blocks),
            "lookup_stat": "minmax",
        },
    }
    payload = {"meta": meta, "layers": layers}
    torch.save(payload, calib_path)
    _save_manifest(
        manifest_path,
        {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "method": "hif4_fp4_minmax_sqrot",
            "qtype": cfg.qtype,
            "model_tag": model_tag,
            "model_signature": model_signature,
            "calibration": os.path.basename(calib_path),
            "timestamp_utc": meta["timestamp_utc"],
            "timestep_count": meta["summary"]["timestep_count"],
        },
    )
    return payload


def _run_ptq_prepare(
    model: nn.Module,
    cfg: Hifx4Config,
    model_tag: str,
    model_signature: str,
    calib_payload: Dict[str, Any],
    prepared_path: str,
    prepared_nunchaku_path: str,
    manifest_path: str,
    qtype_cls,
    quant_fn,
) -> Dict[str, Any]:
    if qtype_cls is None or quant_fn is None:
        raise RuntimeError("[HIF4][Stage2] missing HiFloat4 quantization operators for FP4 prepare.")
    calib_layers = calib_payload.get("layers", {})

    rotation_payload: Dict[str, Any] = {}
    rotation_source_path = ""
    rotation_impl = "disabled"
    if cfg.ptq_enable_rotation:
        rotation_source_path = str(Path(cfg.ptq_rotation_path.strip()).expanduser().resolve()) if cfg.ptq_rotation_path.strip() else ""
        if rotation_source_path:
            rotation_payload = _load_rotation_payload(rotation_source_path)
            rotation_impl = "external_input_matrix"
            print(
                f"[ROT] input rotation enabled for {model_tag}: external source={rotation_source_path}",
                flush=True,
            )
        else:
            rotation_impl = "internal_hadamard"
            print(
                f"[ROT] input rotation enabled for {model_tag}: internal deterministic hadamard seed={cfg.ptq_rotation_seed}",
                flush=True,
            )
    if cfg.ptq_enable_smoothquant:
        print(
            f"[SQ] enabled for {model_tag}: alpha={cfg.ptq_smoothquant_alpha} eps={cfg.ptq_smoothquant_eps}",
            flush=True,
        )
    qtype_w = qtype_cls(cfg.qtype).dim(-1)
    grouping_meta = dict(calib_payload.get("meta", {}).get("grouping", {}))

    split_ranges: List[Tuple[int, int]] = []
    if model_tag == "high_noise_model":
        split_ranges = _parse_timestep_range_spec(cfg.ptq_high_noise_split_ranges)
        if split_ranges:
            print(
                f"[HIF4][Stage2] high_noise timestep split enabled: {split_ranges}",
                flush=True,
            )
            if cfg.ptq_enable_smoothquant:
                print(
                    f"[HIF4][Stage2] SmoothQuant using unified mask across all banks (sq_group_ids=None)",
                    flush=True,
                )

    def _prepare_layers_for_bank(
        timestep_range: Optional[Tuple[int, int]],
    ) -> Tuple[Dict[str, Dict[str, Any]], int, int, int, int]:
        group_ids = _group_ids_for_timestep_range(
            model_tag=model_tag,
            expected_steps=int(cfg.timestep_count),
            timestep_range=timestep_range,
            grouping_meta=grouping_meta,
        )
        # 当启用high_noise split时，SmoothQuant使用全局统计（统一mask）
        # 但min-max lookup仍使用各自的group_ids
        sq_group_ids = None if split_ranges else group_ids
        prepared_local: Dict[str, Dict[str, Any]] = {}
        hit_local = 0
        miss_local = 0
        rotation_merged_local = 0
        smoothquant_applied_local = 0
        for name, module in _iter_target_linears(model, cfg):
            layer_key = _normalize_linear_name(name)
            if layer_key not in calib_layers:
                miss_local += 1
                continue
            layer_cfg = calib_layers[layer_key]

            weight_fp32 = module.weight.detach().to(torch.float32)
            weight_fp32, sq_channel_mask, sq_applied = _apply_smoothquant_to_weight(
                layer_key=layer_key,
                weight_fp32=weight_fp32,
                layer_cfg=layer_cfg,
                cfg=cfg,
                group_ids=group_ids,
                sq_group_ids=sq_group_ids,
            )
            if sq_applied:
                smoothquant_applied_local += 1
            rotation_info: Optional[Dict[str, Any]] = None
            if cfg.ptq_enable_rotation:
                if rotation_impl == "external_input_matrix":
                    rotation_matrix_cpu = _resolve_input_rotation_entry(
                        layer_key=layer_key,
                        in_features=int(module.in_features),
                        rotation_payload=rotation_payload,
                    )
                    if rotation_matrix_cpu is not None:
                        rotation_info = {"kind": "external_input_matrix", "matrix": rotation_matrix_cpu}
                else:
                    rotation_info = _build_internal_rotation_info(
                        layer_key=layer_key,
                        in_features=int(module.in_features),
                        cfg=cfg,
                    )
            if rotation_info is not None:
                weight_fp32 = _apply_rotation_to_tensor(weight_fp32, rotation_info)
                rotation_merged_local += 1

            act_min_global, act_max_global, act_min_group_branch, act_max_group_branch = _build_runtime_act_minmax_lookup(
                layer_key=layer_key,
                layer_cfg=layer_cfg,
                in_features=int(module.in_features),
                channel_mask=sq_channel_mask,
                rotation_info=rotation_info,
                group_ids=group_ids,
            )
            if not torch.cuda.is_available():
                raise RuntimeError("[HIF4][Stage2] FP4 prepare requires CUDA, but no CUDA device is available.")
            quant_device = torch.device("cuda", torch.cuda.current_device())
            weight_fp32_cuda = weight_fp32.to(device=quant_device, dtype=torch.float32, non_blocking=True)
            weight_fp4 = quant_fn(
                weight_fp32_cuda,
                qtype_w,
                force_py=cfg.force_py,
                force_fp32=cfg.force_fp32,
            ).detach()
            weight_fp4_cpu = weight_fp4.to(torch.float16).cpu()
            del weight_fp32_cuda, weight_fp4

            prepared_local[layer_key] = {
                "weight_fp4": weight_fp4_cpu,
                "shape": [int(module.out_features), int(module.in_features)],
                "bias": module.bias.detach().to(torch.float32).cpu() if module.bias is not None else None,
                "act_min_global": act_min_global,
                "act_max_global": act_max_global,
                "act_min_group_branch": act_min_group_branch,
                "act_max_group_branch": act_max_group_branch,
                "smoothquant_applied": bool(sq_applied),
                "smoothquant_channel_mask": (
                    sq_channel_mask.to(torch.float16)
                    if isinstance(sq_channel_mask, torch.Tensor)
                    else None
                ),
                "rotation_applied": bool(rotation_info is not None),
                "rotation_kind": str(rotation_info.get("kind", "none")) if rotation_info is not None else "none",
                "rotation_mode": "input_right" if rotation_info is not None else "none",
                "rotation_source_path": rotation_source_path if (rotation_info is not None and rotation_impl == "external_input_matrix") else "",
                "rotation_seed": int(rotation_info.get("seed", cfg.ptq_rotation_seed)) if rotation_info is not None else int(cfg.ptq_rotation_seed),
                "rotation_block_size": int(rotation_info.get("block_size", 1)) if rotation_info is not None else 1,
                "weight_format": "fp4_sim",
            }
            hit_local += 1
        return prepared_local, hit_local, miss_local, smoothquant_applied_local, rotation_merged_local

    bank_ranges: List[Optional[Tuple[int, int]]] = split_ranges if split_ranges else [None]
    bank_results: List[Dict[str, Any]] = []
    for bank_idx, timestep_range in enumerate(bank_ranges):
        if timestep_range is None:
            print(f"[HIF4][Stage2] preparing single bank for {model_tag}", flush=True)
        else:
            print(
                f"[HIF4][Stage2] preparing bank={bank_idx} range={timestep_range} for {model_tag}",
                flush=True,
            )
        prepared_bank, hit_bank, miss_bank, sq_bank, rot_bank = _prepare_layers_for_bank(timestep_range)
        bank_results.append(
            {
                "bank_idx": int(bank_idx),
                "timestep_range": timestep_range,
                "prepared": prepared_bank,
                "hit": int(hit_bank),
                "miss": int(miss_bank),
                "smoothquant_applied_layers": int(sq_bank),
                "rotation_merged_layers": int(rot_bank),
            }
        )

    primary_bank = bank_results[0]
    prepared = primary_bank["prepared"]
    hit = int(primary_bank["hit"])
    miss = int(primary_bank["miss"])
    smoothquant_applied_layers = int(primary_bank["smoothquant_applied_layers"])
    rotation_merged_layers = int(primary_bank["rotation_merged_layers"])

    prepared_meta = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": "prepare",
        "method": "hif4_fp4_minmax_sqrot",
        "qtype": cfg.qtype,
        "timestamp_utc": _utcnow(),
        "model_tag": model_tag,
        "model_signature": model_signature,
        "grouping": grouping_meta,
        "from_calibration_timestamp": calib_payload.get("meta", {}).get("timestamp_utc", ""),
        "summary": {
            "target_layers": hit + miss,
            "prepared_layers": hit,
            "miss_layers": miss,
            "smoothquant_applied_layers": smoothquant_applied_layers,
            "rotation_merged_layers": rotation_merged_layers,
        },
        "defaults": {
            "act_quant_mode": cfg.act_quant_mode,
            "timestep_count": int(cfg.timestep_count),
            "keep_blocks": list(cfg.keep_blocks),
            "enable_smoothquant": bool(cfg.ptq_enable_smoothquant),
            "smoothquant_alpha": float(cfg.ptq_smoothquant_alpha),
            "smoothquant_eps": float(cfg.ptq_smoothquant_eps),
            "enable_rotation": bool(cfg.ptq_enable_rotation),
            "rotation_impl": rotation_impl,
            "rotation_source_path": rotation_source_path,
            "rotation_seed": int(cfg.ptq_rotation_seed),
            "weight_format": "fp4_sim",
            "lookup_stat": "minmax",
            "activation_grouping": str(grouping_meta.get("mode", _DEFAULT_ACTIVATION_GROUPING)),
            "legacy_weight_group_size_compat": int(cfg.weight_group_size),
            "high_noise_split_ranges": str(cfg.ptq_high_noise_split_ranges or ""),
        },
    }

    payload = {"meta": prepared_meta, "layers": prepared}
    torch.save(payload, prepared_path)
    # Keep compatibility with old scripts that may check this file.
    torch.save(payload, prepared_nunchaku_path)

    high_noise_banks_manifest: List[Dict[str, Any]] = []
    if split_ranges:
        prepared_path_obj = Path(prepared_path)
        nunchaku_path_obj = Path(prepared_nunchaku_path)
        index_path = prepared_path_obj.with_name("prepared_banks_index.json")
        index_payload: Dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "model_tag": model_tag,
            "high_noise_split_enabled": True,
            "ranges": [[int(s), int(e)] for s, e in split_ranges],
            "banks": [],
        }
        for bank in bank_results:
            bank_idx = int(bank["bank_idx"])
            timestep_range = bank["timestep_range"]
            assert timestep_range is not None
            start, end = int(timestep_range[0]), int(timestep_range[1])
            bank_payload = {"meta": prepared_meta, "layers": bank["prepared"]}
            bank_prepared_path = prepared_path_obj.with_name(
                f"{prepared_path_obj.stem}_bank{bank_idx}_t{start}_{end}{prepared_path_obj.suffix}"
            )
            bank_nunchaku_path = nunchaku_path_obj.with_name(
                f"{nunchaku_path_obj.stem}_bank{bank_idx}_t{start}_{end}{nunchaku_path_obj.suffix}"
            )
            torch.save(bank_payload, str(bank_prepared_path))
            torch.save(bank_payload, str(bank_nunchaku_path))
            bank_record = {
                "bank_idx": bank_idx,
                "timestep_range": [start, end],
                "prepared": bank_prepared_path.name,
                "prepared_nunchaku": bank_nunchaku_path.name,
                "prepared_layers": int(bank["hit"]),
                "miss_layers": int(bank["miss"]),
                "smoothquant_applied_layers": int(bank["smoothquant_applied_layers"]),
                "rotation_merged_layers": int(bank["rotation_merged_layers"]),
            }
            index_payload["banks"].append(bank_record)
            high_noise_banks_manifest.append(bank_record)
        _save_manifest(str(index_path), index_payload)

    _save_manifest(
        manifest_path,
        {
            "schema": ARTIFACT_SCHEMA_VERSION,
            "method": "hif4_fp4_minmax_sqrot",
            "qtype": cfg.qtype,
            "model_tag": model_tag,
            "model_signature": model_signature,
            "calibration_timestamp": calib_payload.get("meta", {}).get("timestamp_utc", ""),
            "prepared_timestamp": prepared_meta["timestamp_utc"],
            "prepared": os.path.basename(prepared_path),
            "prepared_nunchaku": os.path.basename(prepared_nunchaku_path),
            "prepared_layers": hit,
            "miss_layers": miss,
            "smoothquant_applied_layers": smoothquant_applied_layers,
            "enable_smoothquant": bool(cfg.ptq_enable_smoothquant),
            "smoothquant_alpha": float(cfg.ptq_smoothquant_alpha),
            "smoothquant_eps": float(cfg.ptq_smoothquant_eps),
            "rotation_merged_layers": rotation_merged_layers,
            "rotation_impl": rotation_impl,
            "timestep_count": int(cfg.timestep_count),
            "keep_blocks": list(cfg.keep_blocks),
            "act_quant_mode": cfg.act_quant_mode,
            "weight_format": "fp4_sim",
            "lookup_stat": "minmax",
            "rotation_source_path": rotation_source_path,
            "rotation_seed": int(cfg.ptq_rotation_seed),
            "legacy_weight_group_size_compat": int(cfg.weight_group_size),
            "high_noise_split_enabled": bool(split_ranges),
            "high_noise_banks": high_noise_banks_manifest,
        },
    )
    return payload


def _validate_prepared_or_raise(
    prepared_payload: Dict[str, Any],
    cfg: Hifx4Config,
    model_signature: str,
    compatible_signatures: Optional[List[str]] = None,
    runtime_layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    meta = prepared_payload.get("meta", {})
    if meta.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"[HIF4] artifact schema mismatch: expected {ARTIFACT_SCHEMA_VERSION}, "
            f"got {meta.get('artifact_schema_version')}"
        )
    if meta.get("qtype") != cfg.qtype:
        raise ValueError(f"[HIF4] qtype mismatch: artifact={meta.get('qtype')} runtime={cfg.qtype}")

    artifact_sig = meta.get("model_signature")
    if artifact_sig != model_signature:
        compat_set = set(compatible_signatures or [])
        if artifact_sig not in compat_set:
            raise ValueError(
                "[HIF4] artifact model signature mismatch. "
                "Use --ptq_force_rebuild with stage prepare/all to regenerate artifacts."
            )

    if runtime_layer_shapes is None:
        return
    prepared_layers = prepared_payload.get("layers", {})
    for layer_name, rt_shape in runtime_layer_shapes.items():
        lp = prepared_layers.get(layer_name)
        if lp is None:
            raise ValueError(f"[HIF4] missing layer in artifact: {layer_name}")
        shape = lp.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"[HIF4] invalid shape for layer {layer_name}: {shape}")
        prep_shape = (int(shape[0]), int(shape[1]))
        if prep_shape != rt_shape:
            raise ValueError(f"[HIF4] shape mismatch for {layer_name}: artifact={prep_shape}, runtime={rt_shape}")
        weight_fp4 = lp.get("weight_fp4", None)
        if not isinstance(weight_fp4, torch.Tensor):
            raise ValueError(f"[HIF4] missing weight_fp4 tensor for {layer_name}")
        if tuple(weight_fp4.shape) != prep_shape:
            raise ValueError(
                f"[HIF4] weight_fp4 shape mismatch for {layer_name}: "
                f"artifact={tuple(weight_fp4.shape)} runtime={prep_shape}"
            )
        if bool(lp.get("smoothquant_applied", False)):
            sq_mask = _parse_channel_vector(lp.get("smoothquant_channel_mask", None), expected_dim=int(rt_shape[1]))
            if sq_mask is None:
                raise ValueError(
                    f"[HIF4] invalid smoothquant_channel_mask for {layer_name}: expect dim={int(rt_shape[1])}"
                )
        if bool(lp.get("rotation_applied", False)):
            rotation_kind = str(lp.get("rotation_kind", lp.get("rotation_mode", "none")))
            if rotation_kind == "internal_hadamard":
                block_size = int(lp.get("rotation_block_size", 0))
                if block_size <= 0 or int(rt_shape[1]) % block_size != 0:
                    raise ValueError(
                        f"[HIF4] invalid internal rotation_block_size for {layer_name}: "
                        f"block_size={block_size} in_features={int(rt_shape[1])}"
                    )
            elif rotation_kind == "external_input_matrix":
                if not str(lp.get("rotation_source_path", "")).strip():
                    raise ValueError(f"[HIF4] missing rotation_source_path for {layer_name}")
        act_min_tb = lp.get("act_min_group_branch", lp.get("act_min_timestep_branch", {}))
        act_max_tb = lp.get("act_max_group_branch", lp.get("act_max_timestep_branch", {}))
        if not isinstance(act_min_tb, dict) or not isinstance(act_max_tb, dict):
            raise ValueError(f"[HIF4] invalid min-max lookup table for {layer_name}")


class Hifx4LinearWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, cfg: Hifx4Config, qtype_cls, quant_fn):
        super().__init__()
        self.linear = linear
        self.cfg = cfg
        self.quant_fn = quant_fn
        self.qp_w = qtype_cls(cfg.qtype).dim(-1)
        self.qp_in = qtype_cls(cfg.qtype).dim(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.quant_fn(x, self.qp_in, force_py=self.cfg.force_py, force_fp32=self.cfg.force_fp32)
        w_q = self.quant_fn(
            self.linear.weight, self.qp_w, force_py=self.cfg.force_py, force_fp32=self.cfg.force_fp32
        )
        out = F.linear(x_q, w_q, self.linear.bias)
        if self.cfg.quant_output:
            out = self.quant_fn(out, self.qp_in, force_py=self.cfg.force_py, force_fp32=self.cfg.force_fp32)
        return out


class PTQPreparedLinearWrapper(nn.Module):
    _BANKED_WRAPPERS = weakref.WeakSet()
    _GLOBAL_ACTIVE_BANK_IDX: Optional[int] = None

    def __init__(
        self,
        prepared: Dict[str, Any],
        cfg: Hifx4Config,
        qtype_cls,
        quant_fn,
        prepared_banks: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.quant_fn = quant_fn
        self.qp_in = qtype_cls(cfg.qtype).dim(-1)
        self.default_state = self._make_state(prepared, bank_idx=0, timestep_range=None)
        self.bank_states = [
            self._make_state(
                dict(bank["prepared"]),
                bank_idx=int(bank["bank_idx"]),
                timestep_range=tuple(bank["timestep_range"]),
            )
            for bank in (prepared_banks or [])
        ]
        self.shape = self.default_state["shape"]
        self._warned_missing_bank = False
        self._active_bank_idx: Optional[int] = None
        if self.bank_states:
            PTQPreparedLinearWrapper._BANKED_WRAPPERS.add(self)

    def _make_state(
        self,
        prepared: Dict[str, Any],
        *,
        bank_idx: int,
        timestep_range: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        shape = tuple(int(x) for x in prepared["shape"])
        sq_mask = _parse_channel_vector(
            prepared.get("smoothquant_channel_mask", None),
            expected_dim=int(shape[1]),
        )
        return {
            "bank_idx": int(bank_idx),
            "timestep_range": timestep_range,
            "weight_fp4": prepared["weight_fp4"],
            "shape": shape,
            "bias": prepared.get("bias"),
            "act_min_global": float(prepared.get("act_min_global", float("-inf"))),
            "act_max_global": float(prepared.get("act_max_global", float("inf"))),
            "act_min_group_branch": {
                str(k): float(v)
                for k, v in dict(prepared.get("act_min_group_branch", prepared.get("act_min_timestep_branch", {}))).items()
            },
            "act_max_group_branch": {
                str(k): float(v)
                for k, v in dict(prepared.get("act_max_group_branch", prepared.get("act_max_timestep_branch", {}))).items()
            },
            "smoothquant_applied": bool(prepared.get("smoothquant_applied", False)),
            "smoothquant_channel_mask": sq_mask if sq_mask is not None else None,
            "rotation_applied": bool(prepared.get("rotation_applied", False)),
            "rotation_kind": str(prepared.get("rotation_kind", prepared.get("rotation_mode", "none"))),
            "rotation_mode": str(prepared.get("rotation_mode", "none")),
            "rotation_source_path": str(prepared.get("rotation_source_path", "") or ""),
            "rotation_seed": int(prepared.get("rotation_seed", 17)),
            "rotation_block_size": int(prepared.get("rotation_block_size", 1)),
            "rotation_layer_key": str(prepared.get("rotation_layer_key", "")),
            "_weight_cache": {},
            "_rotation_cache": {},
        }

    def _apply(self, fn):
        super()._apply(fn)
        for state in [self.default_state] + self.bank_states:
            state["_weight_cache"] = {}
            state["_rotation_cache"] = {}
        return self

    def _select_state(self) -> Dict[str, Any]:
        if not self.bank_states:
            return self.default_state
        snap = GLOBAL_QUANT_CONTEXT.snapshot()
        timestep_id = int(snap.get("timestep_id", -1))
        for state in self.bank_states:
            timestep_range = state.get("timestep_range")
            if timestep_range is None:
                continue
            start, end = int(timestep_range[0]), int(timestep_range[1])
            if start <= timestep_id <= end:
                self._activate_bank(int(state["bank_idx"]))
                return state
        if not self._warned_missing_bank:
            ranges = ",".join(
                f"{int(s['timestep_range'][0])}-{int(s['timestep_range'][1])}"
                for s in self.bank_states
                if s.get("timestep_range") is not None
            )
            print(
                f"[HIF4][Stage3][WARN] no prepared bank matched timestep_id={timestep_id}; "
                f"fallback to bank={self.bank_states[0]['bank_idx']} ranges={ranges}",
                flush=True,
            )
            self._warned_missing_bank = True
        self._activate_bank(int(self.bank_states[0]["bank_idx"]))
        return self.bank_states[0]

    @staticmethod
    def _offload_state_to_cpu(state: Dict[str, Any]) -> bool:
        moved = False
        for key in ("weight_fp4", "bias", "smoothquant_channel_mask"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                state[key] = value.detach().cpu()
                moved = True
        state["_weight_cache"] = {}
        state["_rotation_cache"] = {}
        return moved

    @classmethod
    def _offload_inactive_banks_global(cls, active_bank_idx: int) -> None:
        moved = False
        for wrapper in list(cls._BANKED_WRAPPERS):
            for state in wrapper.bank_states:
                if int(state["bank_idx"]) == int(active_bank_idx):
                    continue
                moved = cls._offload_state_to_cpu(state) or moved
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _activate_bank(self, bank_idx: int) -> None:
        bank_idx = int(bank_idx)
        if PTQPreparedLinearWrapper._GLOBAL_ACTIVE_BANK_IDX != bank_idx:
            PTQPreparedLinearWrapper._GLOBAL_ACTIVE_BANK_IDX = bank_idx
            self._offload_inactive_banks_global(bank_idx)
        self._active_bank_idx = bank_idx

    def _ensure_state_device(self, state: Dict[str, Any], device: torch.device) -> None:
        moved = False
        if state["weight_fp4"].device != device:
            state["weight_fp4"] = state["weight_fp4"].to(device=device, non_blocking=True)
            moved = True
        bias = state.get("bias")
        if isinstance(bias, torch.Tensor) and bias.device != device:
            state["bias"] = bias.to(device=device, non_blocking=True)
            moved = True
        sq_mask = state.get("smoothquant_channel_mask")
        if (
            bool(state.get("smoothquant_applied", False))
            and isinstance(sq_mask, torch.Tensor)
            and sq_mask.device != device
        ):
            state["smoothquant_channel_mask"] = sq_mask.to(device=device, non_blocking=True)
            moved = True
        if moved:
            state["_weight_cache"] = {}
            state["_rotation_cache"] = {}

    def _cache_key(self, device: torch.device, dtype: torch.dtype) -> str:
        return f"{device.type}:{device.index}:{str(dtype)}"

    def _get_weight(self, state: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = self._cache_key(device, dtype)
        weight_cache = state["_weight_cache"]
        if key in weight_cache:
            return weight_cache[key]
        weight = state["weight_fp4"].to(device=device, dtype=dtype, non_blocking=True)
        weight_cache[key] = weight
        return weight

    def _get_rotation_runtime(self, state: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Optional[Any]:
        if not bool(state.get("rotation_applied", False)):
            return None
        key = self._cache_key(device, dtype)
        rotation_cache = state["_rotation_cache"]
        if key in rotation_cache:
            return rotation_cache[key]
        rotation_kind = str(state.get("rotation_kind", "none"))
        if rotation_kind == "internal_hadamard":
            runtime = _build_internal_rotation_runtime(
                {
                    "kind": "internal_hadamard",
                    "seed": int(state["rotation_seed"]),
                    "block_size": int(state["rotation_block_size"]),
                    "in_features": int(state["shape"][1]),
                },
                device=device,
                dtype=dtype,
            )
            rotation_cache[key] = runtime
            return runtime
        rotation_source_path = str(state.get("rotation_source_path", "") or "")
        rotation_layer_key = str(state.get("rotation_layer_key", "") or "")
        if not rotation_source_path or not rotation_layer_key:
            raise RuntimeError(
                f"[ROT] layer={rotation_layer_key or '<unknown>'} requires rotation_source_path in prepared artifact"
            )
        rotation_payload = _load_rotation_payload(rotation_source_path)
        rot_cpu = _resolve_input_rotation_entry(
            layer_key=rotation_layer_key,
            in_features=int(state["shape"][1]),
            rotation_payload=rotation_payload,
        )
        if rot_cpu is None:
            raise RuntimeError(f"[ROT] missing input rotation matrix for layer={rotation_layer_key} from {rotation_source_path}")
        rot = rot_cpu.to(device=device, dtype=dtype, non_blocking=True)
        rotation_cache[key] = rot
        return rot

    def _resolve_act_minmax(self, state: Dict[str, Any], x: torch.Tensor) -> Tuple[float, float]:
        if self.cfg.act_quant_mode == "online":
            x_cpu = x.detach().to(torch.float32)
            return float(x_cpu.amin().item()), float(x_cpu.amax().item())

        snap = GLOBAL_QUANT_CONTEXT.snapshot()
        group_id = int(snap.get("group_id", -1))
        branch = _sanitize_branch(str(snap.get("branch", "single")))
        candidates = [
            _tb_key(group_id, branch),
            _tb_key(group_id, "single"),
        ]
        act_min_group_branch = state["act_min_group_branch"]
        act_max_group_branch = state["act_max_group_branch"]
        for key in candidates:
            if key in act_min_group_branch and key in act_max_group_branch:
                return float(act_min_group_branch[key]), float(act_max_group_branch[key])
        if float(state["act_min_global"]) < float(state["act_max_global"]):
            return float(state["act_min_global"]), float(state["act_max_global"])
        x_cpu = x.detach().to(torch.float32)
        return float(x_cpu.amin().item()), float(x_cpu.amax().item())

    def _apply_smoothquant_input(self, state: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        if not bool(state.get("smoothquant_applied", False)):
            return x
        sq_mask = state.get("smoothquant_channel_mask")
        if not isinstance(sq_mask, torch.Tensor):
            return x
        if sq_mask.numel() != int(x.shape[-1]):
            raise RuntimeError(f"[SQ] mask dim mismatch: mask={sq_mask.numel()} input={int(x.shape[-1])}")
        view_shape = [1] * (x.ndim - 1) + [int(x.shape[-1])]
        return x * sq_mask.to(device=x.device, dtype=x.dtype).view(*view_shape)

    def _apply_rotation_input(self, state: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        if not bool(state.get("rotation_applied", False)):
            return x
        runtime = self._get_rotation_runtime(state, x.device, torch.float32)
        if runtime is None:
            return x
        dtype_out = x.dtype
        rotation_kind = str(state.get("rotation_kind", "none"))
        if rotation_kind == "internal_hadamard":
            x_rot = _apply_internal_rotation_tensor(
                x.to(torch.float32),
                {
                    "kind": "internal_hadamard",
                    "seed": int(state["rotation_seed"]),
                    "block_size": int(state["rotation_block_size"]),
                    "in_features": int(state["shape"][1]),
                },
                runtime=runtime,
            )
        else:
            x_rot = torch.matmul(x.to(torch.float32), runtime)
        return x_rot.to(dtype=dtype_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._select_state()
        self._ensure_state_device(state, x.device)
        x_for_linear = self._apply_smoothquant_input(state, x)
        x_for_linear = self._apply_rotation_input(state, x_for_linear)
        weight = self._get_weight(
            state,
            x.device,
            x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float32,
        )
        act_min, act_max = self._resolve_act_minmax(state, x_for_linear)

        if WRAPPER_TIMER.enabled:
            torch.cuda.synchronize(x_for_linear.device)
            t0 = time.perf_counter()

        out = run_hifx4_ptq_fused_linear(
            x=x_for_linear,
            weight=weight,
            bias=state.get("bias"),
            qp_in=self.qp_in,
            quant_fn=self.quant_fn,
            force_py=self.cfg.force_py,
            force_fp32=self.cfg.force_fp32,
            quant_output=self.cfg.quant_output,
            act_min=act_min,
            act_max=act_max,
        )

        if WRAPPER_TIMER.enabled:
            torch.cuda.synchronize(x_for_linear.device)
            WRAPPER_TIMER.record(time.perf_counter() - t0)
        return out


def assert_quant_coverage(model: nn.Module, keep_blocks: List[int]) -> None:
    keep_set = set(keep_blocks)
    name_to_module = dict(model.named_modules())
    misses: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parent_name = ".".join(name.split(".")[:-1])
        parent = name_to_module.get(parent_name)
        if isinstance(parent, (PTQCalibrationObserverLinear, Hifx4LinearWrapper, PTQPreparedLinearWrapper)):
            continue
        block_idx = _extract_block_index(name)
        if block_idx is None:
            continue
        if block_idx in keep_set:
            continue
        misses.append(name)
    if misses:
        joined = "\n  - ".join(misses[:64])
        raise RuntimeError(
            "[HIF4][Coverage] Found unquantized Linear in non-keep blocks:\n"
            f"  - {joined}"
        )
def enable_training_ste_for_hifx4_wrappers(model: nn.Module) -> int:
    """Switch all HiFX4 runtime wrappers in a model to STE quantization."""
    try:
        from quant_cy import quant_func
    except Exception as exc:
        raise RuntimeError(
            "Failed to import quant_cy.quant_func for HiFX4 STE training."
        ) from exc

    switched = 0
    for module in model.modules():
        if isinstance(module, (Hifx4LinearWrapper, PTQPreparedLinearWrapper)):
            module.quant_fn = quant_func
            switched += 1
    return switched


def replace_wan_linear_with_hifx4(
    model: nn.Module,
    hifloat4_root: str,
    cfg: Hifx4Config,
    model_tag: str,
) -> Tuple[int, int, Dict[str, Any]]:
    _validate_keep_blocks(cfg.keep_blocks)

    name_to_module = dict(model.named_modules())
    total_linear = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
    replaced = 0
    report: Dict[str, Any] = {"model_tag": model_tag, "stage": cfg.stage}
    target_names = [name for name, _ in _iter_target_linears(model, cfg)]

    if cfg.quant_method != "ptq":
        _ensure_hifloat4_importable(hifloat4_root)
        try:
            from quant_cy import QType, quant_dequant_float
        except Exception as exc:
            raise RuntimeError(
                "Failed to import quant_cy. Build HiFloat4 CUDA extension in hifloat4/hifx4_gpu first."
            ) from exc
        for name, module in _iter_target_linears(model, cfg):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = name_to_module[parent_name] if parent_name else model
            setattr(parent, child_name, Hifx4LinearWrapper(module, cfg, QType, quant_dequant_float))
            replaced += 1
        report.update({"target_layers": len(target_names), "hit_layers": replaced, "miss_layers": 0})
        return replaced, total_linear, report

    stage = cfg.stage
    if stage not in ("calibrate", "prepare", "infer", "all"):
        raise ValueError(f"Invalid stage: {stage}")

    need_quant_ext = stage in ("prepare", "infer", "all")
    QType = None
    quant_dequant_float = None
    if need_quant_ext:
        _ensure_hifloat4_importable(hifloat4_root)
        try:
            from quant_cy import QType as _QType, quant_dequant_float as _quant_dequant_float
        except Exception as exc:
            raise RuntimeError(
                "Failed to import quant_cy. Build HiFloat4 CUDA extension in hifloat4/hifx4_gpu first."
            ) from exc
        QType = _QType
        quant_dequant_float = _quant_dequant_float

    model_signature = _model_signature(model, cfg)
    compatible_signatures = _compatible_model_signatures(model, cfg)
    paths = _artifact_paths(cfg, model_tag)
    report["artifact_dir"] = paths["dir"]
    report["model_signature"] = model_signature

    calib_payload = None
    if stage in ("calibrate", "all"):
        if stage == "calibrate":
            replaced, total_linear, report = _register_calibration_observers(
                model=model,
                cfg=cfg,
                model_tag=model_tag,
                model_signature=model_signature,
            )
            print(
                f"[HIF4][Stage1] Observing model={model_tag} layers={report['target_layers']} "
                f"artifact_dir={report['artifact_dir']}",
                flush=True,
            )
            return replaced, total_linear, report

        if cfg.ptq_force_rebuild or (not os.path.exists(paths["calibration"])):
            print(f"[HIF4][Stage1] Building calibration fallback for {model_tag} -> {paths['calibration']}", flush=True)
            calib_payload = _run_ptq_calibration(
                model=model,
                cfg=cfg,
                model_tag=model_tag,
                model_signature=model_signature,
                calib_path=paths["calibration"],
                manifest_path=paths["manifest"],
            )
        else:
            calib_payload = torch.load(paths["calibration"], map_location="cpu")
            if calib_payload.get("meta", {}).get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(
                    f"[HIF4][Stage1] calibration schema mismatch for {model_tag}. "
                    "Re-run stage calibrate with --ptq_force_rebuild (or RESUME=0)."
                )
            print(f"[HIF4][Stage1] Reusing calibration for {model_tag} <- {paths['calibration']}", flush=True)

    if stage in ("prepare", "all"):
        if calib_payload is None:
            if not os.path.exists(paths["calibration"]):
                raise FileNotFoundError(
                    f"[HIF4] Missing calibration for '{model_tag}': {paths['calibration']} "
                    "(run stage calibrate first)"
                )
            calib_payload = torch.load(paths["calibration"], map_location="cpu")
        if calib_payload.get("meta", {}).get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"[HIF4][Stage2] calibration schema mismatch for {model_tag}. "
                "Re-run stage calibrate under the current code."
            )

        need_prepare = cfg.ptq_force_rebuild or (not os.path.exists(paths["prepared"]))
        if (not need_prepare) and os.path.exists(paths["prepared"]):
            prepared_probe = torch.load(paths["prepared"], map_location="cpu")
            if prepared_probe.get("meta", {}).get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
                print(
                    f"[HIF4][Stage2] Existing prepared artifact schema is stale for {model_tag}; rebuilding.",
                    flush=True,
                )
                need_prepare = True
        if need_prepare:
            print(f"[HIF4][Stage2] Preparing '{model_tag}' -> {paths['prepared']}", flush=True)
            prepared_payload = _run_ptq_prepare(
                model=model,
                cfg=cfg,
                model_tag=model_tag,
                model_signature=model_signature,
                calib_payload=calib_payload,
                prepared_path=paths["prepared"],
                prepared_nunchaku_path=paths["prepared_nunchaku"],
                manifest_path=paths["manifest"],
                qtype_cls=QType,
                quant_fn=quant_dequant_float,
            )
        else:
            prepared_payload = torch.load(paths["prepared"], map_location="cpu")
            print(f"[HIF4][Stage2] Reusing prepared '{model_tag}' <- {paths['prepared']}", flush=True)

        summary = prepared_payload.get("meta", {}).get("summary", {})
        report.update(
            {
                "target_layers": int(summary.get("target_layers", len(target_names))),
                "hit_layers": int(summary.get("prepared_layers", 0)),
                "miss_layers": int(summary.get("miss_layers", 0)),
            }
        )

    if stage in ("infer", "all"):
        infer_engine = (cfg.ptq_infer_engine or "python").lower()
        if infer_engine != "python":
            raise NotImplementedError(
                "[HIF4] nunchaku runtime path is disabled in this branch. "
                "Use --ptq_infer_engine python."
            )
        if not os.path.exists(paths["prepared"]):
            raise FileNotFoundError(
                f"[HIF4] Missing prepared artifact for '{model_tag}': {paths['prepared']} "
                "(run stage prepare first)"
            )
        prepared_payload = torch.load(paths["prepared"], map_location="cpu")
        if prepared_payload.get("meta", {}).get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"[HIF4][Stage3] prepared artifact schema mismatch for {model_tag}. "
                "Re-run stage prepare under the current code."
            )
        runtime_layer_shapes = {
            _normalize_linear_name(name): (int(module.out_features), int(module.in_features))
            for name, module in _iter_target_linears(model, cfg)
        }
        prepared_bank_entries = _load_high_noise_prepared_banks(paths, model_tag=model_tag)
        if cfg.ptq_validate_artifact:
            _validate_prepared_or_raise(
                prepared_payload=prepared_payload,
                cfg=cfg,
                model_signature=model_signature,
                compatible_signatures=compatible_signatures,
                runtime_layer_shapes=runtime_layer_shapes,
            )
            for bank in prepared_bank_entries:
                _validate_prepared_or_raise(
                    prepared_payload=bank["payload"],
                    cfg=cfg,
                    model_signature=model_signature,
                    compatible_signatures=compatible_signatures,
                    runtime_layer_shapes=runtime_layer_shapes,
                )

        prepared_layers = prepared_payload.get("layers", {})
        misses: List[str] = []
        replaced = 0
        for name, module in _iter_target_linears(model, cfg):
            layer_key = _normalize_linear_name(name)
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = name_to_module[parent_name] if parent_name else model
            if layer_key not in prepared_layers:
                misses.append(layer_key)
                continue
            prepared = _prepared_layer_view(prepared_layers[layer_key], layer_key)
            prepared_banks: List[Dict[str, Any]] = []
            for bank in prepared_bank_entries:
                bank_layers = bank["payload"].get("layers", {})
                if layer_key not in bank_layers:
                    raise ValueError(
                        f"[HIF4][Stage3] missing layer={layer_key} in prepared bank "
                        f"{bank['bank_idx']} ({bank['prepared_path']})"
                    )
                prepared_banks.append(
                    {
                        "bank_idx": int(bank["bank_idx"]),
                        "timestep_range": tuple(bank["timestep_range"]),
                        "prepared": _prepared_layer_view(bank_layers[layer_key], layer_key),
                    }
                )
            wrapper = PTQPreparedLinearWrapper(
                prepared,
                cfg,
                QType,
                quant_dequant_float,
                prepared_banks=prepared_banks or None,
            )
            setattr(parent, child_name, wrapper)
            replaced += 1

        bank_ranges_text = ",".join(
            f"{int(bank['timestep_range'][0])}-{int(bank['timestep_range'][1])}"
            for bank in prepared_bank_entries
        )
        report.update(
            {
                "target_layers": len(target_names),
                "hit_layers": replaced,
                "miss_layers": len(misses),
                "miss_examples": misses[:8],
                "artifact_version": prepared_payload.get("meta", {}).get("artifact_schema_version"),
                "infer_engine": infer_engine,
                "prepared_banks": len(prepared_bank_entries),
                "prepared_bank_ranges": bank_ranges_text,
            }
        )
        print(
            f"[HIF4][Stage3] model={model_tag} engine={infer_engine} hit={replaced}/{len(target_names)} "
            f"miss={len(misses)} artifact={report['artifact_version']} "
            f"banks={len(prepared_bank_entries)} ranges={bank_ranges_text or '<none>'}",
            flush=True,
        )
        if misses:
            print(f"[HIF4][Stage3] miss examples: {report['miss_examples']}", flush=True)

        assert_quant_coverage(model, cfg.keep_blocks)
    else:
        if "hit_layers" not in report:
            report.update({"target_layers": len(target_names), "hit_layers": 0, "miss_layers": len(target_names)})

    return replaced, total_linear, report


def _patch_model_forward_for_quant_context(model: nn.Module, model_tag: str) -> None:
    if getattr(model, "_hif4_quantctx_patched", False):
        return
    original_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        t = kwargs.get("t")
        if t is None and len(args) >= 2:
            t = args[1]
        GLOBAL_QUANT_CONTEXT.update_from_forward(t=t, model_tag=model_tag)
        return original_forward(*args, **kwargs)

    model.forward = _wrapped_forward
    model._hif4_quantctx_patched = True


def _patch_generate_for_quant_context(WanT2V, wan_text2video_module, cfg: Hifx4Config) -> None:
    if getattr(WanT2V, "_hif4_quantctx_generate_patched", False):
        return
    original_generate = WanT2V.generate

    def _wrapped_generate(self, *args, **kwargs):
        sampling_steps = kwargs.get("sampling_steps")
        if sampling_steps is None and len(args) >= 6:
            # Signature: input_prompt, size, frame_num, shift, sample_solver, sampling_steps, ...
            sampling_steps = args[5]
        sample_solver = kwargs.get("sample_solver")
        if sample_solver is None and len(args) >= 5:
            sample_solver = args[4]
        if sample_solver is None:
            sample_solver = "unipc"
        shift = kwargs.get("shift")
        if shift is None and len(args) >= 4:
            shift = args[3]
        if shift is None:
            shift = 5.0
        if sampling_steps is None:
            sampling_steps = cfg.timestep_count
        expected_steps = int(sampling_steps)
        step_model_tags: List[str] = []
        step_group_ids: List[int] = []
        grouping_meta: Dict[str, Any] = {}
        try:
            if str(sample_solver) == "unipc":
                sample_scheduler = wan_text2video_module.FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    expected_steps,
                    device=self.device,
                    shift=shift,
                )
                timesteps = sample_scheduler.timesteps
            elif str(sample_solver) == "dpm++":
                sample_scheduler = wan_text2video_module.FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = wan_text2video_module.get_sampling_sigmas(expected_steps, shift)
                timesteps, _ = wan_text2video_module.retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas,
                )
            else:
                timesteps = []
            boundary = float(self.boundary) * float(self.num_train_timesteps)
            step_model_tags = [
                "high_noise_model" if int(t.item()) >= boundary else "low_noise_model"
                for t in timesteps
            ]
            ranges = _build_group_ranges_from_step_tags(step_model_tags)
            step_group_ids = [-1] * len(step_model_tags)
            for model_tag, items in ranges.items():
                for item in items:
                    start = int(item["timestep_start"])
                    end = int(item["timestep_end"])
                    for step_idx in range(start, end + 1):
                        if 0 <= step_idx < len(step_group_ids):
                            step_group_ids[step_idx] = int(item["group_id"])
            for step_idx, group_id in enumerate(step_group_ids):
                if group_id < 0:
                    step_group_ids[step_idx] = _resolve_activation_group(
                        model_tag=step_model_tags[step_idx],
                        timestep_id=step_idx,
                        expected_steps=len(step_model_tags),
                    )
            grouping_meta = {
                "mode": _DEFAULT_ACTIVATION_GROUPING,
                "expected_steps": int(len(step_model_tags)),
                "model_group_ranges": ranges,
            }
        except Exception:
            step_model_tags = []
            step_group_ids = []
            grouping_meta = {}
        GLOBAL_QUANT_CONTEXT.begin_generation(
            expected_steps,
            step_model_tags=step_model_tags,
            step_group_ids=step_group_ids,
            grouping_meta=grouping_meta,
        )
        try:
            return original_generate(self, *args, **kwargs)
        finally:
            GLOBAL_QUANT_CONTEXT.end_generation()

    WanT2V.generate = _wrapped_generate
    WanT2V._hif4_quantctx_generate_patched = True


def patch_want2v_configure_model(
    wan_text2video_module,
    hifloat4_root: str,
    cfg: Hifx4Config,
) -> None:
    _validate_keep_blocks(cfg.keep_blocks)

    WanT2V = wan_text2video_module.WanT2V
    if getattr(WanT2V, "_hifx4_patched", False):
        return

    _patch_generate_for_quant_context(WanT2V, wan_text2video_module, cfg)

    original_configure_model = WanT2V._configure_model
    state = {"model_index": 0, "offline_done": 0}

    def _wrapped_configure_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype):
        model_tag = "low_noise_model" if state["model_index"] == 0 else "high_noise_model"
        state["model_index"] += 1

        offline_stage = cfg.quant_method == "ptq" and cfg.stage in ("calibrate", "prepare")
        if offline_stage:
            want_low = cfg.ptq_offline_model in ("all", "low")
            want_high = cfg.ptq_offline_model in ("all", "high")
            should_process = (model_tag == "low_noise_model" and want_low) or (
                model_tag == "high_noise_model" and want_high
            )

            # Keep official Wan2.2 configure path unchanged.
            model = original_configure_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype)
            # Must patch the *returned* module (e.g. FSDP). Patching the pre-configure `model` is a no-op
            # because configure replaces the object; otherwise GLOBAL_QUANT_CONTEXT never sees `t` and
            # calibration keys stay at "-1::single".
            _patch_model_forward_for_quant_context(model, model_tag=model_tag)

            if cfg.stage == "calibrate":
                if not should_process:
                    print(
                        f"[HIF4] Skip calibration observers for {model_tag} "
                        f"(selector={cfg.ptq_offline_model})",
                        flush=True,
                    )
                    return model

                replaced, total_linear, report = replace_wan_linear_with_hifx4(
                    model=model,
                    hifloat4_root=hifloat4_root,
                    cfg=cfg,
                    model_tag=model_tag,
                )
                print(
                    f"[HIF4] stage={cfg.stage} method={cfg.quant_method} model={model_tag} "
                    f"replaced={replaced}/{total_linear} target={report.get('target_layers', 0)} "
                    f"hit={report.get('hit_layers', 0)} miss={report.get('miss_layers', 0)} "
                    f"artifact_dir={report.get('artifact_dir', '-')}",
                    flush=True,
                )
                return model

            if not should_process:
                print(
                    f"[HIF4] Skip offline stage '{cfg.stage}' for {model_tag} "
                    f"(selector={cfg.ptq_offline_model})",
                    flush=True,
                )
                return model

            replaced, total_linear, report = replace_wan_linear_with_hifx4(
                model=model,
                hifloat4_root=hifloat4_root,
                cfg=cfg,
                model_tag=model_tag,
            )
            print(
                f"[HIF4] stage={cfg.stage} method={cfg.quant_method} model={model_tag} "
                f"replaced={replaced}/{total_linear} target={report.get('target_layers', 0)} "
                f"hit={report.get('hit_layers', 0)} miss={report.get('miss_layers', 0)} "
                f"artifact_dir={report.get('artifact_dir', '-')}",
                flush=True,
            )
            state["offline_done"] += 1
            target_count = 2 if cfg.ptq_offline_model == "all" else 1
            if state["offline_done"] >= target_count:
                print(f"[HIF4] Offline stage '{cfg.stage}' finished for selected models. Exit by design.", flush=True)
                raise SystemExit(0)
            return model

        pre_replace_before_config = cfg.quant_method == "ptq" and cfg.stage == "infer" and bool(dit_fsdp)
        if pre_replace_before_config:
            replaced, total_linear, report = replace_wan_linear_with_hifx4(
                model=model,
                hifloat4_root=hifloat4_root,
                cfg=cfg,
                model_tag=model_tag,
            )
            model = original_configure_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype)
            _patch_model_forward_for_quant_context(model, model_tag=model_tag)
        else:
            model = original_configure_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype)
            _patch_model_forward_for_quant_context(model, model_tag=model_tag)
            replaced, total_linear, report = replace_wan_linear_with_hifx4(
                model=model,
                hifloat4_root=hifloat4_root,
                cfg=cfg,
                model_tag=model_tag,
            )

        print(
            f"[HIF4] stage={cfg.stage} method={cfg.quant_method} model={model_tag} "
            f"replaced={replaced}/{total_linear} target={report.get('target_layers', 0)} "
            f"hit={report.get('hit_layers', 0)} miss={report.get('miss_layers', 0)} "
            f"artifact_dir={report.get('artifact_dir', '-')}",
            flush=True,
        )
        return model

    WanT2V._configure_model = _wrapped_configure_model
    WanT2V._hifx4_patched = True
