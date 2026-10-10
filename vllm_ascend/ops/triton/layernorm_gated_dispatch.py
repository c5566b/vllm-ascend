"""Pure-scalar launch selection for the gated LayerNorm kernels.

The selector deliberately knows nothing about torch, devices, or resource
queries. The wrapper supplies initialized vector-core properties for
single-group NPU inputs and wide C2 inputs. Other calls retain BASE64.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST_HOIST", "C2_BASE"]
    block_m: int
    block_n_chunk: int | None = None


class DispatchConfigError(ValueError):
    """Raised when a selector input is invalid or no qualified route exists."""


BM_BASE16 = 16
BM_HOIST = 32
HOIST_QUARTER_WAVE_DIVISOR = 4
BASE16_MAX_N_GROUP = 512
BASE16_MIN_UB_BYTES = 196_608
BM_C2 = 64
BN_C2_CHUNK = 64
C2_MIN_UB_BYTES = 196_608


def _is_positive_exact_int(value) -> bool:
    return type(value) is int and value > 0


def _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes) -> None:
    for name, value in (("M", M), ("N_group", N_group), ("ngroups", ngroups)):
        if not _is_positive_exact_int(value):
            raise DispatchConfigError(f"{name} must be a positive exact int")
    if runtime_p is not None and not _is_positive_exact_int(runtime_p):
        raise DispatchConfigError("runtime_p must be None or a positive exact int")
    if ub_bytes is not None and not _is_positive_exact_int(ub_bytes):
        raise DispatchConfigError("ub_bytes must be None or a positive exact int")


def _select_layernorm_launch(
    M,
    N_group,
    ngroups,
    runtime_p,
    *,
    ub_bytes: int | None = None,
) -> LaunchSpec:
    """Select the fixed PR1 policy, adding C2 only for qualified wide N.

    Missing vector-core properties retain the upstream BASE64 launch. For
    ``N_group > 512``, C2 requires a known 192 KiB UB budget and otherwise
    fails closed before launch.
    """
    _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes)
    if runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    if N_group > BASE16_MAX_N_GROUP:
        if ub_bytes is None or ub_bytes < C2_MIN_UB_BYTES:
            raise DispatchConfigError("no resource-qualified LayerNorm-Gated path")
        return LaunchSpec("C2_BASE", BM_C2, BN_C2_CHUNK)

    # Grouped full-tile inputs retain the upstream PR1 launch. Wide C2
    # selection above applies to both single-group and grouped inputs.
    if ngroups > 1:
        return LaunchSpec("FT_BASE", 64)

    # This branch is exactly the qualified PR1 BASE16 envelope.
    if N_group > 128:
        if ub_bytes is not None and ub_bytes >= BASE16_MIN_UB_BYTES:
            return LaunchSpec("FT_BASE", BM_BASE16)
        return LaunchSpec("FT_BASE", 64)

    if N_group < 128:
        return LaunchSpec("FT_BASE", BM_BASE16)

    hoist_tiles = (M + BM_HOIST - 1) // BM_HOIST
    if HOIST_QUARTER_WAVE_DIVISOR * hoist_tiles >= runtime_p:
        return LaunchSpec("FT_PERSIST_HOIST", BM_HOIST)
    return LaunchSpec("FT_BASE", BM_BASE16)
