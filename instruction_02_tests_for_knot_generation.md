# Agent Task 02: Test Specification for Task 07 (Knot Generation)

## Objective
Write detailed tests for `instruction_07_knot_generation.md` before implementation starts.

This task produces tests and fixtures only.

## Target Under Test
Expected module: `src/numpyro_extras/mixture_knots.py`  
Expected primary type: `QuantileKnotSet` with validated knot arrays and endpoint slopes.

## Inputs (Contract for This Agent)
1. `instruction_07_knot_generation.md`.
2. A quantile backend fixture (mock or real) exposing deterministic `icdf(u)` for controlled testing.
3. Project testing stack.

## Outputs (Contract for This Agent)
1. Test file: `tests/test_mixture_knots.py`.
2. Optional fixture helpers in `tests/conftest.py` for reusable mock quantile backends.
3. Documented expected fields in `QuantileKnotSet`.

## Test Design Requirements

### 1) API and schema tests
1. Constructor/entrypoint accepts config object and backend.
2. Returned object has required fields:
   - `u_knots`, `x_knots`, `du_dx_left`, `du_dx_right`, `meta`.
3. Output arrays have expected rank and matching lengths.

### 2) Grid strategy coverage
For each grid mode (`uniform_u`, `logit_u`, `hybrid`):
1. Verify generation succeeds.
2. Validate tails receive expected point density pattern.
3. Confirm `u_knots` start/end close to configured bounds.

### 3) Ordering and monotonicity tests
1. Strictly increasing `u_knots`.
2. Nondecreasing `x_knots`.
3. Handle and repair small monotonicity violations from noisy backend.

### 4) Slope estimation tests
1. Endpoint slopes finite and positive.
2. Slope floor behavior when `dx` is tiny.
3. Regression test for duplicate or near-duplicate `x` values.

### 5) Metadata quality tests
1. `meta` includes required keys:
   - grid strategy, point count, cleanup counts, min/max spacing, non-finite count.
2. Values are consistent with generated outputs.

### 6) Robustness tests
1. Backend returns non-finite values for some `u` -> function filters/reports as designed.
2. Small `num_knots` warning path (`N < 16`) reflected in metadata.
3. Invalid config options rejected with clear message.

### 7) JAX behavior tests
1. JIT compatibility for core knot generation function (if supported by API).
2. Vectorized backend interactions behave deterministically.

## Required Assertions
1. `jnp.all(jnp.diff(u_knots) > 0)` must hold.
2. `jnp.all(jnp.diff(x_knots) >= 0)` must hold after cleanup.
3. `du_dx_left > 0` and `du_dx_right > 0`.
4. All knots and slopes are finite.

## Structure and Naming
Recommended test names:
1. `test_quantile_knotset_schema_contract`
2. `test_grid_modes_generate_valid_knots`
3. `test_knots_are_strictly_ordered_and_monotone`
4. `test_endpoint_slope_estimation_is_positive_and_finite`
5. `test_metadata_contains_quality_metrics`
6. `test_nonfinite_quantiles_are_reported_and_handled`
7. `test_invalid_configs_raise_useful_errors`

## Acceptance Criteria
1. Tests define exact acceptance behavior for Task 07.
2. Tests isolate failures to one contract at a time.
3. Deterministic output under fixed seeds and fixed mock backend.

## Handoff to Task 07 Agent
Provide:
1. Ready-to-run test suite with clear expected semantics.
2. Mock backend fixture used for edge-case injection.
3. Explicit checklist of contract conditions implementation must satisfy.

