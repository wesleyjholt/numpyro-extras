# Agent Task 03: Test Specification for Task 08 (Quantile Interpolator + Sigmoid Tails)

## Objective
Create detailed tests for `instruction_08_quantile_interpolator.md` before writing interpolator code.

This task writes tests only.

## Target Under Test
Expected module: `src/numpyro_extras/quantile_interpolator.py`  
Expected class/object: `QuantileInterpolator1D`.

## Inputs (Contract for This Agent)
1. `instruction_08_quantile_interpolator.md`.
2. Synthetic `QuantileKnotSet` fixtures representing known monotone functions.
3. Optional reference `icdf/cdf` pairs for oracle comparisons.

## Outputs (Contract for This Agent)
1. Test file: `tests/test_quantile_interpolator.py`.
2. Reusable fixtures for knot sets:
   - smooth interior case,
   - tail-stress case,
   - near-degenerate slope case.

## Test Design Requirements

### 1) API contract tests
Verify object exposes:
1. `icdf(u)`, `cdf(x)`.
2. `dudx(x)`, `dxdu(u)`.
3. `log_abs_dxdu(u)`.
4. `stitch_points()`.

### 2) Roundtrip and inverse-consistency tests
1. Dense `u` grid: `cdf(icdf(u)) ~ u`.
2. Dense `x` grid in interior: `icdf(cdf(x)) ~ x`.
3. Batched inputs preserve shape semantics.

### 3) Monotonicity and bijection tests
1. `icdf` strictly/non-strictly increasing as specified.
2. `cdf` increasing over full tested domain.
3. `dxdu > 0` and `dudx > 0` within tolerance floors.

### 4) Stitch continuity tests (critical)
At both boundaries (`u0/x0`, `uN/xN`):
1. Value continuity across piecewise branch switch.
2. First-derivative continuity across switch (`C1` requirement), where boundary slope is taken from the interior interpolator gradient.
3. No discontinuity spikes under finite differencing.

Additional requirement:
- Include a regression test that would fail if C1 slope is computed from adjacent-knot finite differences instead of interior interpolator derivative.

### 5) Tail behavior tests
1. Extreme `u` values remain finite in `icdf`.
2. Extreme `x` values map to valid `u in (0,1)` without NaN.
3. `arctanh` safety clipping prevents domain errors.

### 6) Numerical robustness tests
1. Very small `u0` and very large `uN`.
2. Near-zero endpoint slopes (flooring behavior).
3. Densely packed knots with tiny spacing.

### 7) JAX behavior tests
1. `jit` compile for `cdf`, `icdf`, and derivative calls.
2. `vmap` compatibility over input vectors.
3. No NumPy fallback in interpolation runtime path (all traced ops remain JAX-native).

## Required Tolerances
1. Interior roundtrip: `< 1e-4`.
2. Tail roundtrip: `< 5e-4`.
3. Stitch continuity (value): `< 1e-6` relative/absolute where feasible.
4. Stitch continuity (slope): `< 1e-4` finite-difference proxy.

## Structure and Naming
Recommended names:
1. `test_quantile_interpolator_api_contract`
2. `test_roundtrip_cdf_icdf_dense_grid`
3. `test_reverse_roundtrip_icdf_cdf_interior`
4. `test_monotonicity_of_cdf_and_icdf`
5. `test_stitch_value_continuity_left_and_right`
6. `test_stitch_slope_continuity_left_and_right`
7. `test_tail_inputs_remain_finite_and_valid`
8. `test_jit_and_vmap_compatibility`

## Acceptance Criteria
1. Tests fully define expected behavior for Task 08.
2. Stitch/tail behavior is explicitly pinned by assertions.
3. Failures localize to formula, continuity, or stability class.

## Handoff to Task 08 Agent
Provide:
1. Finalized test file and fixtures.
2. Explicit expected formulas at stitch points in test comments.
3. Tolerance rationale so implementation can tune safely.

