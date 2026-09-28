"""Pure-scalar launch selection for the gated LayerNorm kernels.

The selector deliberately knows nothing about torch, devices, or resource
queries.  The wrapper supplies the initialized vector-core count on NPU and
``None`` for non-NPU tensors. Wide-N NPU choices use the initialized UB size
and the measured resource envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST", "FT_PERSIST_HOIST", "C2_BASE"]
    block_m: int
    block_n_chunk: int | None = None


class C2Config(NamedTuple):
    """N-chunk launch shape and minimum tested UB budget in bytes."""

    block_m: int
    block_n_chunk: int
    minimum_qualified_ub_bytes: int


BM_PERSIST_SINGLE = 32
# A tile wave is one BM32 tile per initialized vector core.
HOIST_MIN_TILE_WAVES = 16
BM_LARGE_N_BASE = 16


@dataclass(frozen=True)
class DispatchParams:
    """Per-group launch policy; ``bm_*`` are row-tile heights.

    ``k_persist_num/den`` and ``k_c2_num/den`` are workload ratios relative
    to the initialized vector-core count, not UB limits. ``None`` marks an
    unconfigured route, which is rejected if reached.
    """

    bm_small: int | None = None
    bm_multi: int | None = None
    k_persist_num: int | None = None
    k_persist_den: int | None = None
    n_persist_min: int = 1
    hoist_qualified: bool = False
    persist_single_qualified: bool = False
    k_c2_num: int | None = None
    k_c2_den: int | None = None
    c2_config: C2Config | None = None
    # (power-of-two BLOCK_N, minimum tested UB budget in bytes).
    full_tile_ub_envelope: tuple[tuple[int, int], ...] = ()


DEFAULT_PARAMS = DispatchParams()


class DispatchConfigError(ValueError):
    """Raised when a selector input or policy is not materialized."""


def _is_positive_exact_int(value) -> bool:
    return type(value) is int and value > 0


def _ceil_div(value: int, divisor: int) -> int:
    if not _is_positive_exact_int(value) or not _is_positive_exact_int(divisor):
        raise DispatchConfigError("ceil-div inputs must be positive exact ints")
    return (value + divisor - 1) // divisor


def _need(value, name: str):
    if value is None:
        raise DispatchConfigError(f"{name} not materialized")
    return value


def validate_params(params) -> None:
    if type(params) is not DispatchParams:
        raise DispatchConfigError("params must be a DispatchParams instance")
    for name in ("hoist_qualified", "persist_single_qualified"):
        if type(getattr(params, name)) is not bool:
            raise DispatchConfigError(f"{name} must be an exact bool")
    if not _is_positive_exact_int(params.n_persist_min):
        raise DispatchConfigError("n_persist_min must be a positive exact int")
    for name in ("bm_small", "bm_multi", "k_persist_num", "k_persist_den", "k_c2_num", "k_c2_den"):
        value = getattr(params, name)
        if value is not None and not _is_positive_exact_int(value):
            raise DispatchConfigError(f"{name} must be None or a positive exact int")
    if params.hoist_qualified and not params.persist_single_qualified:
        raise DispatchConfigError("hoist_qualified implies persist_single_qualified")
    if params.c2_config is not None:
        if type(params.c2_config) is not C2Config:
            raise DispatchConfigError("c2_config must be C2Config or None")
        for value in params.c2_config:
            if not _is_positive_exact_int(value):
                raise DispatchConfigError("C2 configuration values must be positive exact ints")
    if type(params.full_tile_ub_envelope) is not tuple:
        raise DispatchConfigError("full_tile_ub_envelope must be a tuple")
    seen = set()
    for entry in params.full_tile_ub_envelope:
        if type(entry) is not tuple or len(entry) != 2:
            raise DispatchConfigError("full-tile envelope entries must be (BLOCK_N, min_ub) pairs")
        block_n, min_ub = entry
        if not _is_positive_exact_int(block_n) or not _is_positive_exact_int(min_ub):
            raise DispatchConfigError("full-tile envelope values must be positive exact ints")
        if block_n & (block_n - 1) or block_n in seen:
            raise DispatchConfigError("full-tile envelope BLOCK_N must be unique powers of two")
        seen.add(block_n)


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
    params: DispatchParams = DEFAULT_PARAMS,
    *,
    ub_bytes: int | None = None,
) -> LaunchSpec:
    """Select a qualified full-tile or N-chunk path.

    Non-NPU calls retain the upstream BASE64 launch. NPU calls supply the
    worker-initialized vector-core count and UB budget.
    """
    validate_params(params)
    _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes)
    if runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    if N_group > 128:
        block_n = 1 << (N_group - 1).bit_length()
        required_ft_ub = dict(params.full_tile_ub_envelope).get(block_n)
        ft_safe = ub_bytes is not None and required_ft_ub is not None and ub_bytes >= required_ft_ub
        c2 = params.c2_config
        c2_safe = ub_bytes is not None and c2 is not None and ub_bytes >= c2.minimum_qualified_ub_bytes
        if ft_safe and not c2_safe:
            return LaunchSpec("FT_BASE", BM_LARGE_N_BASE)
        if ft_safe:
            base_tiles = _ceil_div(M, BM_LARGE_N_BASE) * ngroups
            if base_tiles * _need(params.k_c2_den, "k_c2_den") < _need(params.k_c2_num, "k_c2_num") * runtime_p:
                return LaunchSpec("FT_BASE", BM_LARGE_N_BASE)
        if c2_safe and c2 is not None:
            return LaunchSpec("C2_BASE", c2.block_m, c2.block_n_chunk)
        raise DispatchConfigError("no resource-qualified LayerNorm-Gated path")

    if N_group < _need(params.n_persist_min, "n_persist_min"):
        return LaunchSpec("FT_BASE", _need(params.bm_small, "bm_small"))

    bm_persist = BM_PERSIST_SINGLE if ngroups == 1 else _need(params.bm_multi, "bm_multi")

    # The only qualified multi-group PR1 route is BASE32 at N_group=128.
    if N_group == 128 and ngroups > 1:
        return LaunchSpec("FT_BASE", bm_persist)

    # A persistent launch is considered once its tile count reaches a
    # calibrated fraction of the initialized vector-core count.
    persist_tiles = _ceil_div(M, bm_persist) * ngroups
    if (
        persist_tiles * _need(params.k_persist_den, "k_persist_den")
        < _need(params.k_persist_num, "k_persist_num") * runtime_p
    ):
        return LaunchSpec("FT_BASE", _need(params.bm_small, "bm_small"))

    if ngroups == 1:
        if params.hoist_qualified and persist_tiles >= HOIST_MIN_TILE_WAVES * runtime_p:
            return LaunchSpec("FT_PERSIST_HOIST", BM_PERSIST_SINGLE)
        if params.persist_single_qualified:
            return LaunchSpec("FT_PERSIST", BM_PERSIST_SINGLE)
        return LaunchSpec("FT_BASE", BM_PERSIST_SINGLE)

    return LaunchSpec("FT_BASE", _need(params.bm_multi, "bm_multi"))
