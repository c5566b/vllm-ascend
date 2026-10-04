# Gated LayerNorm / RMSNorm

## Description

- **Function**: Normalize each row and group, apply per-channel weight and optional bias, and optionally gate with `silu(z)` before or after normalization.
- **Formula**: For a group of width `N_group`, LayerNorm uses `mean = sum(x) / N_group`, `rstd = rsqrt(sum((x - mean)^2) / N_group + eps)` and `y = (x - mean) * rstd * weight + bias`. RMSNorm omits the mean and uses `rstd = rsqrt(sum(x^2) / N_group + eps)`. If `z` is present, `x *= silu(z)` before normalization when `norm_before_gate=False`; otherwise `y *= silu(z)` after the affine transform.
- **Algorithm flow** (rows and groups are independent):
  1. The wrapper validates the input layout, allocates `out`, `mean` (LayerNorm only), and `rstd`, and obtains the initialized vector-core count on NPU. For the qualified wide per-group domain, it also reads UB through the existing initialized-properties getter.
  2. A scalar selector chooses a BASE row tile or, for a qualified single group, a HOIST32 M-axis launch. BASE uses the existing two-dimensional `(row tiles, groups)` grid with `BLOCK_M=16`, `32`, or `64`.
  3. HOIST32 caps its one-dimensional grid at the vector-core count, walks M-axis tiles with a grid-stride loop, and loads the single group's weight and optional bias before that loop.
  4. For `N_group>512`, the selector uses C2 with `BLOCK_M=64` and `BLOCK_N_CHUNK=64` only when the initialized UB getter reports at least 196608 bytes. C2 scans feature chunks for statistics, then again for normalization, affine, and optional gate. A host guard bounds its integer indexing before launch.
  5. Each tile computes statistics in fp32 and stores them group-major as `[group, row]`.
- **Supported modes**: Inference-time LayerNorm and RMSNorm, with optional bias and pre/post gate. The selector is not device-name or dtype allowlisted. Evidence includes bounded Ascend910B4/P40 BF16/FP16 public C2 numerical/resource checks and separately identified historical Ascend910B3 results. These do not establish all-width, all-device or model-level graph-capture qualification. No performance measurement qualifies this resource-only C2 selector.

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
- For `N_group<128`, the NPU selector uses BASE16. At `N_group=128`, grouped inputs use BASE32; a single group uses HOIST32 when `4 * ceil(M / 32) >= P`, where `P` is the initialized vector-core count. The BM32 tiles need to cover at least one quarter of the initialized runtime vector cores: the boundary is M=289 for P=40 and M=353 for P=48. This is an integer tile-count rule, not a measured optimal crossover. Multi-group persistent execution is not enabled.
- For `128 < N_group <= 512`, NPU calls use FT_BASE with `BLOCK_M=16` when `get_ub_size_bytes()` returns at least 196608 bytes. `BLOCK_N` is 256 for `N_group` 129–256 and 512 for 257–512. The getter may return its compatibility default or debugging override; its value is a routing input, not an independent compiler-resource measurement. Missing or lower UB and non-NPU calls retain BASE64. The width is the per-group `N_group`, not total `N`.
- For `N_group>512`, C2 is selected for any M and group count only when the initialized vector-core count is available and the UB getter reports at least 196608 bytes; unknown or lower UB raises before launch. C2 uses the existing host guard for signed-i32 row, group, and chunk-loop domains. A direct FT16 BM16/BN1024 control was compiler-rejected for UB overflow in the tested B4/P40 FP16 RMS+Z case with `norm_before_gate=False`; matching public C2 calls passed. This proves resource value for that case, not necessity for every N>512 mode.
- The BASE feature-width guard alone does not qualify every resource bucket. The 192 KiB UB evidence covers only its recorded B3 history and B4 regression cases; other resource buckets, devices, and dtypes are not inferred by interpolation. Getter defaults/debugging overrides are routing inputs, not independent measurements. Route selection depends on shape, group count, and initialized resource getters, not tensor values. Non-NPU calls retain BASE64; uninitialized NPU vector-core properties raise before selection.
- Former q=4 C2 selection measurements are historical and do not quantify performance lost by this resource-only simplification. No performance claim is made for the new selector.

## Origin and Differences

- **Origin**: The existing `layernorm_gated.py` implementation is adapted from Flash Linear Attention's gated LayerNorm and the Triton LayerNorm tutorial. PR1 reuses the original BASE kernel's normalization and gating math.
- **Differences**:
    - NPU execution can use a smaller BASE row tile or a capped persistent M-axis grid instead of always launching one BASE64 program per row tile.
    - HOIST32 moves single-group weight and optional bias loads outside each program's M-tile loop. This describes source-level work placement, not an isolated measured speedup claim.
    - `N_group>512` can use the two-pass C2 kernel under the existing 192 KiB UB floor. All qualified N_group<=512 routes belong to PR1. The earlier q=4 measurements are historical; the public selector no longer chooses C2 by a performance ratio, and this simplification gives up the former FT-safe C2 performance policy.

## Test Cases

- Host selector and wrapper-route tests check the integer quarter-wave tile boundary and nearby rows, grouped BASE32, the FT16 `N_group`/UB envelope through N=512, C2 selection and rejection above N=512, non-NPU fallbacks, and launch arguments without an NPU:

  ```bash
  python -m unittest discover -s tests/ut/ops -p 'test_layernorm_gated_*.py'
  ```

- The existing single-card operator test checks LayerNorm/RMSNorm, optional bias/gate, grouping, dtype tolerances, and `out` behavior against a CPU reference:

  ```bash
  pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_layernorm_gated.py
  ```

- In-tree public C2 cases cover N=513 BF16 RMSNorm with post-gate, N=1024 FP16 RMSNorm with pre-gate, and N=513 grouped LayerNorm with bias and gate. They use fixed seeds and compare outputs/statistics with the CPU reference while recording the actual grid. Matching-main execution of this Nightly file remains pending; the separate native regression below does not replace it.

## Bounded offline validation

Code checkpoint `49b190c043bbaee7837eb9e96ee002d7375d3247` is stacked on PR1 `bdbdc07a9f268751bed01dff3a0b300b0005cb0f`. Run `pr1_20261004T135627Z_0ada2e84` used one Ascend910B4/P40 device, measured UB 196608 bytes and preinstalled CANN 8.5. Six unforced public C2 calls (two seeds per case) passed output and applicable mean/rstd comparisons against the CPU reference:

| M | N_group | Groups | Dtype and mode | norm_before_gate | C2 result |
| ---: | ---: | ---: | --- | --- | --- |
| 65 | 513 | 1 | BF16 RMS+Z, no bias | True | 2/2 PASS |
| 65 | 1024 | 1 | FP16 RMS+Z, no bias | False | 2/2 PASS |
| 65 | 513 | 2 | BF16 LN+bias+Z | False | 2/2 PASS |

Actual launch was BM64/BNc64, with grid `[2,1]` or `[2,2]`. Same-seed N1024 FT16 controls used identical inputs/references and attempted BM16/BN1024 compilation. Both were rejected with required UB 2392064 and/or 2654208 bits versus available 1572864 bits; their numeric status is NOT_RUN, timing and speedup N/A. A separate health probe passed after each negative control. Maximum C2 output absolute error was 0.00390625; all saved tensors were finite and passed the configured absolute/relative tolerances.

This is native single-operator evidence for the frozen three-case/two-seed matrix. There is no new C2 timing, N5120/model validation, full-width qualification or matching-main CI/Nightly result. Wider model configurations motivate the API-level work but are not claimed supported by this test. Historical q=4 speedups do not measure this code's resource-only policy.
