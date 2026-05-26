from typing import Optional

import torch
import torch.nn.functional as F


def _infer_chunk_rows(x2d: torch.Tensor, target_chunk_bytes: int = 256 * 1024 * 1024) -> int:
    if x2d.ndim != 2 or x2d.shape[1] <= 0:
        return max(1, x2d.shape[0])
    bytes_per_elem = max(1, x2d.element_size())
    row_bytes = int(x2d.shape[1]) * bytes_per_elem
    if row_bytes <= 0:
        return max(1, x2d.shape[0])
    return max(1, min(int(x2d.shape[0]), target_chunk_bytes // row_bytes))


def run_hifx4_ptq_fused_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    qp_in,
    quant_fn,
    force_py: bool,
    force_fp32: bool,
    quant_output: bool,
    act_min: Optional[float] = None,
    act_max: Optional[float] = None,
) -> torch.Tensor:
    """
    HiF4 W4A4 Python runtime path (low-rank removed).

    Note:
    - Low-rank branches are intentionally removed.
    - Runtime keeps only activation quant + linear GEMM (+ optional output quant).
    """
    if act_min is not None or act_max is not None:
        clamp_min = float(act_min) if act_min is not None else None
        clamp_max = float(act_max) if act_max is not None else None
        if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
            clamp_min, clamp_max = clamp_max, clamp_min
        if clamp_min is not None or clamp_max is not None:
            x = x.clamp(min=clamp_min, max=clamp_max)

    in_dim = int(x.shape[-1])
    x2d = x.reshape(-1, in_dim)
    rows = int(x2d.shape[0])
    chunk_rows = _infer_chunk_rows(x2d)

    if chunk_rows >= rows:
        x_q = quant_fn(x, qp_in, force_py=force_py, force_fp32=force_fp32)
        out = F.linear(x_q, weight, bias)
    else:
        out2d = None
        out_dim = int(weight.shape[0])
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            x_chunk = x2d[start:end].reshape(-1, in_dim)
            x_q_chunk = quant_fn(x_chunk, qp_in, force_py=force_py, force_fp32=force_fp32)
            out_chunk = F.linear(x_q_chunk, weight, bias)
            if out2d is None:
                out2d = torch.empty((rows, out_dim), device=out_chunk.device, dtype=out_chunk.dtype)
            out2d[start:end] = out_chunk
        assert out2d is not None
        out = out2d.reshape(*x.shape[:-1], out_dim)

    if quant_output:
        out = quant_fn(out, qp_in, force_py=force_py, force_fp32=force_fp32)
    return out
