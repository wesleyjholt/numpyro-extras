# Agent Task 01: Test Specification for Task 06 (Mixture -> Inverse-CDF Backend)

## Objective
Author detailed, test-first coverage for `instruction_06_mixture_inverse_cdf.md` before implementation work starts.

This task writes tests and fixtures only. Do not implement backend logic here.

## Target Under Test
Expected module: `src/numpyro_extras/mixture_quantile.py`  
Expected public surface:
1. Backend constructor/factory.
2. `cdf(x)`.
3. `log_prob(x)`.
4. `icdf(u)`.
5. Optional diagnostics method(s).

## Inputs (Contract for This Agent)
1. `instruction_06_mixture_inverse_cdf.md` (source spec).
2. Existing project test stack (`pytest`, `jax`, `numpyro`).
3. Known reference distributions with analytic CDF/ICDF (for oracle checks).

## Outputs (Contract for This Agent)
1. Test file: `tests/test_mixture_quantile.py`.
2. Optional shared fixture helper: `tests/conftest.py` additions (only if required).
3. A short test manifest at top-of-file listing required API names and expected behavior.

## Test Design Requirements

### 1) API contract tests
Write tests that fail clearly if the backend API shape is wrong:
1. Constructor accepts required configs.
2. Object exposes `cdf`, `icdf`, and `log_prob`.
3. Methods accept scalar and batched JAX arrays.

### 2) Correctness tests with oracle baselines
1. **Single-component reduction**
   - Mixture with one component should match that component’s `cdf/icdf/log_prob`.
2. **Roundtrip identity**
   - Dense `u` grid: `cdf(icdf(u))` matches `u` within tolerance.
3. **Monotonic quantile**
   - `icdf` outputs nondecreasing values for sorted `u`.

### 3) Numerical stability tests
1. Tail probabilities: `u = [1e-8, 1e-6, 1-1e-6, 1-1e-8]`.
2. Highly imbalanced weights.
3. Very small and very large component scales.
4. Ensure no NaN/Inf outputs.

### 4) Root-finding robustness tests
1. Bracket expansion path executes for difficult mixtures.
2. Failure mode is explicit and informative when bracketing cannot be established.
3. If diagnostics API exists, assert convergence flags and iteration counts are sane.

### 5) JAX behavior tests
1. `jit` compile of `icdf` and `cdf`.
2. `vmap` over batch inputs.
3. Stable dtypes and output shapes.

## Required Tolerances
Use defaults unless repo already has tighter standards:
1. Interior roundtrip: `abs(cdf(icdf(u)) - u) < 5e-5`.
2. Extreme tails: `< 2e-4`.
3. Single-component match: `< 1e-5` interior, `< 1e-4` tails.

## Test Data Matrix
Cover at least these mixture setups:
1. Balanced bimodal normal mixture.
2. Imbalanced bimodal normal mixture (`w=[0.99, 0.01]`).
3. Nearly overlapping components.
4. Wide-separation components.

## Structure and Naming
Use explicit names:
1. `test_api_contract_mixture_quantile_backend`
2. `test_single_component_reduces_to_base_distribution`
3. `test_roundtrip_cdf_icdf_dense_grid`
4. `test_tail_inputs_remain_finite`
5. `test_icdf_is_monotone`
6. `test_jit_and_vmap_compatibility`
7. `test_bracket_failure_reports_useful_error`

## Negative/Failure Assertions
Include tests that confirm:
1. Invalid weights (negative or non-normalized) are rejected.
2. Invalid `u` values outside `[0,1]` produce deterministic handling or explicit error, matching spec.

## Acceptance Criteria
1. Tests are deterministic (fixed PRNG keys).
2. No fragile timing-based checks.
3. Failures identify exact violated contract (not generic assert messages).
4. Tests can run before implementation and clearly indicate missing behavior.

## Handoff to Task 06 Agent
Provide:
1. Complete test file with TODO comments only where unavoidable.
2. Clear expected API signatures in docstring/comments.
3. List of strict pass gates the implementation must satisfy.

