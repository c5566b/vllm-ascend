# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.layernorm_gated_dispatch import (
    DispatchConfigError,
    _select_layernorm_launch,
)
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, get_vectorcore_num

_C2_SIGNED_I32_MAX = 2**31 - 1


def _check_c2_launch_bounds(M, N_total, group_size, ngroups, block_m, block_n_chunk):
    """Prove the public C2 launch stays inside the lowering's i32 domains.

    The C2 kernel intentionally retains i32 loop/count values.  This guard is
    host-only and uses the selected launch constants, so every rejection is
    completed before the grid is constructed or the kernel is indexed.
    """
    values = {
        "M": M,
        "N_total": N_total,
        "group_size": group_size,
        "ngroups": ngroups,
        "block_m": block_m,
        "block_n_chunk": block_n_chunk,
    }
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise RuntimeError("layer_norm_fwd_npu: C2 launch bounds require positive Python ints")

    if M > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 M exceeds signed i32: {M}")
    if N_total > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 N_total exceeds signed i32: {N_total}")
    if ngroups > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 ngroups exceeds signed i32: {ngroups}")
    if block_m > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 BLOCK_M exceeds signed i32: {block_m}")
    if block_n_chunk > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 BLOCK_N_CHUNK exceeds signed i32: {block_n_chunk}")

    feature_offset_domain = group_size * ngroups
    if feature_offset_domain > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 feature offset exceeds signed i32: {feature_offset_domain}")

    stats_offset = ngroups * M
    if stats_offset > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 stats offset exceeds signed i32: {stats_offset}")

    num_m_tiles = (M + block_m - 1) // block_m
    max_row_index = (num_m_tiles - 1) * block_m + (block_m - 1)
    if max_row_index > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 row/tile index exceeds signed i32: {max_row_index}")

    max_group_offset = (ngroups - 1) * group_size + (group_size - 1)
    if max_group_offset > _C2_SIGNED_I32_MAX:
        raise RuntimeError(f"layer_norm_fwd_npu: C2 group feature offset exceeds signed i32: {max_group_offset}")

    if group_size > _C2_SIGNED_I32_MAX - (block_n_chunk - 1):
        raise RuntimeError("layer_norm_fwd_npu: C2 group_size exceeds the i32 chunk/count bound")
    last_chunk_start = ((group_size - 1) // block_n_chunk) * block_n_chunk
    final_chunk_update = last_chunk_start + block_n_chunk
    if final_chunk_update > _C2_SIGNED_I32_MAX:
        raise RuntimeError("layer_norm_fwd_npu: C2 chunk-loop update exceeds signed i32")
    count_upper_bound = group_size
    if count_upper_bound > _C2_SIGNED_I32_MAX:
        raise RuntimeError("layer_norm_fwd_npu: C2 count exceeds signed i32")


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(do_not_specialize=["stride_x_row", "stride_y_row", "stride_z_row", "M", "N", "eps"])
def _layer_norm_fwd_1pass_kernel_npu(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N

    # Compute row indices for this program
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # Mask for valid rows and cols
    row_mask = rows < M
    col_mask = cols < N

    # Load weight once (broadcasted over rows)
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    # Load X: shape [BLOCK_M, BLOCK_N]
    x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :] + group * N
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)

    # Load Z if needed
    if HAS_Z:
        z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :] + group * N
        z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if not NORM_BEFORE_GATE:
            x *= z * tl.sigmoid(z)

    # Compute statistics per row
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # [BLOCK_M]
        xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
        tl.store(Mean + rows, mean, mask=row_mask)
    else:
        xbar = tl.where(col_mask[None, :], x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N

    rstd = 1.0 / tl.sqrt(var + eps)  # [BLOCK_M]
    tl.store(Rstd + rows, rstd, mask=row_mask)

    # Normalize
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]
    if HAS_BIAS:
        y += b[None, :]

    # Post-gate
    if HAS_Z and NORM_BEFORE_GATE:
        y *= z * tl.sigmoid(z)

    # Store output
    y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :] + group * N
    tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(
    do_not_specialize=[
        "stride_x_row",
        "stride_y_row",
        "stride_z_row",
        "M",
        "N",
        "eps",
        "num_m_blocks",
    ]
)
def _layer_norm_fwd_persistent_hoist_kernel_npu(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    num_m_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask).to(tl.float32)

    for tile_id in range(pid, num_m_blocks, num_programs):
        rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        x_ptrs = X + rows[:, None] * stride_x_row + cols[None, :]
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + cols[None, :]
            z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :]).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x *= z * tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            xbar = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            tl.store(Mean + rows, mean, mask=row_mask)
        else:
            xbar = tl.where(col_mask[None, :], x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + rows, rstd, mask=row_mask)
        if not IS_RMS_NORM:
            x_hat = (x - mean[:, None]) * rstd[:, None]
        else:
            x_hat = x * rstd[:, None]
        y = x_hat * w[None, :]
        if HAS_BIAS:
            y += b[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y *= z * tl.sigmoid(z)
        y_ptrs = Y + rows[:, None] * stride_y_row + cols[None, :]
        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit(
    do_not_specialize=[
        "stride_x_row",
        "stride_y_row",
        "stride_z_row",
        "M",
        "N",
        "eps",
    ]
)
def _layer_norm_fwd_c2_nchunk_kernel_npu(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N_CHUNK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Non-persistent two-pass N-chunk kernel.  Each program owns one (M tile,
    # normalization group): it scans the complete group in finite N chunks to
    # form statistics, then reloads X chunks to normalize, affine, gate, and
    # store.  N is a runtime loop bound; only BLOCK_M / BLOCK_N_CHUNK are
    # constexpr.  Stats are group-major; RMS has no Mean allocation.
    pid_m = tl.program_id(0)
    group = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lane_cols = tl.arange(0, BLOCK_N_CHUNK)
    row_mask = rows < M
    group_offset = group * N

    mean_ptr = Mean
    rstd_ptr = Rstd + group * M
    if not IS_RMS_NORM:
        mean_ptr = Mean + group * M

    # Pass 1: scan all chunks in ascending order.  RMS retains the FP32 sumsq
    # path; LN performs the frozen left-to-right Welford merge.
    sumsq = tl.zeros((BLOCK_M,), tl.float32)
    acc_count = tl.zeros((), dtype=tl.int32)
    acc_mean = tl.zeros((BLOCK_M,), tl.float32)
    acc_m2 = tl.zeros((BLOCK_M,), tl.float32)
    for chunk_start in tl.range(0, N, BLOCK_N_CHUNK):
        cols = chunk_start + lane_cols
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = X + rows[:, None] * stride_x_row + group_offset + cols[None, :]
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z_ptrs = Z + rows[:, None] * stride_z_row + group_offset + cols[None, :]
            z_chunk = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
            x_chunk *= z_chunk * tl.sigmoid(z_chunk)

        valid_count = tl.minimum(N - chunk_start, BLOCK_N_CHUNK)
        if IS_RMS_NORM:
            sumsq += tl.sum(tl.where(col_mask[None, :], x_chunk * x_chunk, 0.0), axis=1)
        else:
            valid_count_f = valid_count.to(tl.float32)
            chunk_mean = tl.sum(tl.where(col_mask[None, :], x_chunk, 0.0), axis=1) / valid_count_f
            centered = tl.where(col_mask[None, :], x_chunk - chunk_mean[:, None], 0.0)
            chunk_m2 = tl.sum(centered * centered, axis=1)
            total_count = acc_count + valid_count
            total_count_f = total_count.to(tl.float32)
            delta = chunk_mean - acc_mean
            acc_m2 += chunk_m2 + delta * delta * (acc_count.to(tl.float32) * valid_count_f / total_count_f)
            acc_mean += delta * (valid_count_f / total_count_f)
            acc_count = total_count

    if IS_RMS_NORM:
        rstd = 1.0 / tl.sqrt(sumsq / N + eps)
    else:
        mean = acc_mean
        rstd = 1.0 / tl.sqrt(acc_m2 / acc_count.to(tl.float32) + eps)
        tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(rstd_ptr + rows, rstd, mask=row_mask)

    # Pass 2: reload X; W/B stream per N chunk.  Ordering is after the complete
    # Pass-1 scan so an independent out is safe.
    for chunk_start in tl.range(0, N, BLOCK_N_CHUNK):
        cols = chunk_start + lane_cols
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        x_ptrs = X + rows[:, None] * stride_x_row + group_offset + cols[None, :]
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        z_chunk = None
        if HAS_Z:
            z_ptrs = Z + rows[:, None] * stride_z_row + group_offset + cols[None, :]
            z_chunk = tl.load(z_ptrs, mask=mask, other=0.0).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x_chunk *= z_chunk * tl.sigmoid(z_chunk)

        w_chunk = tl.load(W + group_offset + cols, mask=col_mask, other=0.0).to(tl.float32)
        if IS_RMS_NORM:
            y = x_chunk * rstd[:, None]
        else:
            y = (x_chunk - mean[:, None]) * rstd[:, None]
        y *= w_chunk[None, :]
        if HAS_BIAS:
            b_chunk = tl.load(B + group_offset + cols, mask=col_mask, other=0.0).to(tl.float32)
            y += b_chunk[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y *= z_chunk * tl.sigmoid(z_chunk)
        y_ptrs = Y + rows[:, None] * stride_y_row + group_offset + cols[None, :]
        tl.store(y_ptrs, y, mask=mask)


def layer_norm_fwd_npu(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size

    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    runtime_p = None
    ub_bytes = None
    if getattr(getattr(x, "device", None), "type", None) == "npu":
        runtime_p = get_vectorcore_num()
        if group_size > 128:
            ub_bytes = get_ub_size_bytes()
    spec = _select_layernorm_launch(
        M,
        group_size,
        ngroups,
        runtime_p,
        ub_bytes=ub_bytes,
    )

    # BASE selections reuse the upstream kernel and feature-dimension guard.
    # Non-NPU inputs retain BASE64. N>512 NPU inputs use C2 only when the
    # existing initialized UB getter reports its qualified minimum.
    if spec.impl == "FT_BASE":
        max_fused_size = 65536 // x.element_size()
        block_n = min(max_fused_size, triton.next_power_of_2(group_size))
        if group_size > block_n:
            raise RuntimeError(
                f"layer_norm_fwd_npu: Feature dim too large, got {group_size}, max supported is {block_n}."
            )
        grid = (triton.cdiv(M, spec.block_m), ngroups)
        _layer_norm_fwd_1pass_kernel_npu[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_M=spec.block_m,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    if spec.impl == "C2_BASE":
        if spec.block_n_chunk is None:
            raise DispatchConfigError("C2_BASE spec missing block_n_chunk")
        _check_c2_launch_bounds(M, N, group_size, ngroups, spec.block_m, spec.block_n_chunk)
        grid = (triton.cdiv(M, spec.block_m), ngroups)
        _layer_norm_fwd_c2_nchunk_kernel_npu[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_M=spec.block_m,
            BLOCK_N_CHUNK=spec.block_n_chunk,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    if runtime_p is None:
        raise DispatchConfigError("persistent selection requires an initialized vector-core count")
    block_n = min(65536 // x.element_size(), triton.next_power_of_2(group_size))

    if spec.impl == "FT_PERSIST_HOIST":
        if ngroups != 1:
            raise DispatchConfigError("FT_PERSIST_HOIST requires ngroups == 1")
        num_m_blocks = triton.cdiv(M, spec.block_m)
        grid = (min(runtime_p, num_m_blocks),)
        _layer_norm_fwd_persistent_hoist_kernel_npu[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            num_m_blocks,
            BLOCK_M=spec.block_m,
            BLOCK_N=block_n,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
        )
        return out, mean, rstd

    raise DispatchConfigError(f"impl {spec.impl} is unsupported or unmaterialized")
