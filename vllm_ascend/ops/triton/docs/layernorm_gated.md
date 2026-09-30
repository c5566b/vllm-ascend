# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` before or after normalization.
- **Formula**: For a group of width `N_group`, LayerNorm uses `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)` and `y = (x - mean) * rstd * weight + bias`. RMSNorm omits the mean and uses `rstd = rsqrt(sum(x^2) / N_group + eps)`. If `z` is present, `x *= silu(z)` before normalization when `norm_before_gate=False`; otherwise `y *= silu(z)` after the affine transform.
- **Algorithm flow** (rows and groups are independent):
  1. The wrapper validates the input layout, allocates `out`, `mean` (LayerNorm only), and `rstd`, and obtains the initialized vector-core count on NPU.
  2. A scalar selector chooses a BASE row tile or, for a qualified single group, a persistent M-axis launch. BASE uses the existing two-dimensional `(row tiles, groups)` grid with `BLOCK_M=16`, `32`, or `64`.
  3. PERSIST32 caps its one-dimensional grid at the vector-core count and walks M-axis tiles with a grid-stride loop. HOIST32 uses the same scheduling but loads the single group's weight and optional bias before that loop.
  4. For wide groups, an FT16 tile or C2 N-chunk path is selected using the initialized UB budget and the tested resource envelope. C2 scans feature chunks once for statistics and again for normalization, affine, and optional gate. A host launch guard bounds its integer indexing before the kernel launch.
  5. Each tile computes statistics in fp32 and stores them group-major as `[group, row]`.
- **Supported modes**: Inference-time LayerNorm and RMSNorm, with optional bias and pre/post gate. The selector is not device-name or dtype allowlisted. The optimized routes have single-operator A2/Ascend910B3 BF16 evidence; performance on other devices and dtypes is not established. Model-level graph-capture validation is outside this operator document.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Activations `[M, N]` | Floating point | 2D ND; contiguous last dimension |
| `weight` | Input | Per-channel scale `[N]` | Floating point | 1D contiguous |
| `bias` | Optional input | Per-channel bias `[N]` | Floating point | 1D contiguous, or `None` |
| `eps` | Attribute | Stability constant added before reciprocal square root | Float | Scalar |
| `z` | Optional input | Gate activations `[M, N]` | Floating point | 2D ND with contiguous last dimension, or `None` |
| `out` | Optional input/output | Caller-provided output, or allocated by the wrapper | Same as `x` | `[M, N]` with contiguous last dimension |
| `group_size` | Optional attribute | Per-group width `N_group`; defaults to `N` | Integer | Scalar dividing `N` |
| `norm_before_gate` | Attribute | Apply the gate after normalization when true, before when false | Boolean | Scalar |
| `is_rms_norm` | Attribute | Select RMSNorm rather than mean-subtracting LayerNorm | Boolean | Scalar |
| return value | Output | `(out, mean, rstd)`; `mean=None` for RMSNorm | Output dtype / fp32 statistics | `[M, N]`, `[ngroups * M]` |

## Constraints

- `N` must be divisible by `group_size`; `weight` and optional `bias` have shape `[N]`; optional `z` and `out` have shape `[M, N]`. The wrapper checks these conditions and the last-dimension strides. Resource tiling and the routes below use the **per-group** width `N_group=group_size`, not total `N` (for example, total `N=384` with `group_size=128` has three groups of width 128).
- The current policy uses BASE16 below `N_group=128`, BASE32 for `ngroups>1` at `N_group=128`, and considers PERSIST32 for a single group at `N_group=128` when `ceil(M / 32) >= P / 4`, where `P` is the initialized vector-core count. It chooses HOIST32 when `ceil(M / 32) >= 16P`. Multi-group persistent execution is not enabled.
- For `N_group>128`, the measured A2/Ascend910B3 192 KiB UB budget admits FT16 at `BLOCK_N=256,512` and C2 at `BLOCK_M=64, BLOCK_N_CHUNK=64`. If both paths are eligible, FT16 is selected while `ceil(M/16) * ngroups < 4P` and C2 otherwise. When only C2 is eligible, it is selected regardless of M. Other resource buckets are not inferred by interpolation; if neither path is eligible, the NPU call raises before launch. The existing `get_ub_size_bytes()` supplies this budget and may use its compatibility fallback or debugging override; these are not independent compiler-resource measurements.
- NPU calls require the existing worker-initialized vector-core count; an uninitialized count raises. Non-NPU calls retain BASE64. The BASE feature-width guard remains in its launch branch. The C2 path uses a separate host guard for the signed-i32 row, group, and chunk-loop domains.
- The BASE feature-width guard alone does not guarantee UB feasibility. On A2/Ascend910B3 with 192 KiB UB, the tested PR1 parent BASE64 specialization for BF16, post-gated `M=65, N_group=256` failed compilation (289 KiB required); the PR2 FT16 route passed the corresponding single-operator numerical and route checks. This is a configuration-specific result, not a universal width threshold or a performance claim for other devices and dtypes.
- Route selection depends on shape, group count, initialized vector-core count, and the initialized UB budget, not on tensor values. No claim is made that every route is faster on every device or dtype.

## Origin and Differences

- **Origin**: The existing `layernorm_gated.py` implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial. PR1 reuses the original BASE kernel's normalization and gating math.
- **Differences**:
    - NPU execution can use a smaller BASE row tile or a capped persistent M-axis grid instead of always launching one BASE64 program per row tile.
    - HOIST32 moves single-group weight and optional bias loads outside each program's M-tile loop. This describes source-level work placement, not an isolated measured speedup claim.
    - Wide `N_group` can use FT16 or the two-pass C2 kernel under an explicit resource envelope. Paired FT16/C2 measurements selected `K_c2=4` as the smallest tested workload ratio with gains at both measured widths. This threshold affects performance selection only when both paths are feasible; it cannot override a resource rejection.

## Test Cases

- Host selector and wrapper-route tests check the calibrated boundaries, multi-group and wide-N fallbacks, and launch arguments without an NPU:

  ```bash
  python -m unittest discover -s tests/ut/ops -p 'test_layernorm_gated_*.py'
  ```

- The existing single-card operator test checks LayerNorm/RMSNorm, optional bias/gate, grouping, dtype tolerances, and `out` behavior against a CPU reference:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
  ```

- Three BF16 cases from PR1 derive M from the initialized vector-core count to exercise PERSIST32 and HOIST32. Two further BF16 cases exercise the public C2 route at the first `ceil(M/16)=4P` tile for `N_group=192,384`, compare `out`, `mean`, and `rstd` with the CPU reference, and verify the actual JIT grid. An A2/B3 offline single-operator run checked representative numerics and routes at an earlier PR2 runtime checkpoint, before the switch to the existing UB getter. The current source and these in-tree cases still require execution on a matching-main NPU environment.
