# Agent Task 04: Test Specification for Task 09 (NumPyro Transform Wrappers)

## Objective
Define comprehensive tests for `instruction_09_numpyro_transforms.md` before transform implementation.

This task writes tests and scaffolding only.

## Target Under Test
Expected module: `src/numpyro_extras/transforms.py`  
Expected classes:
1. `UniformToMixtureTransform`
2. `NormalToMixtureTransform`

## Inputs (Contract for This Agent)
1. `instruction_09_numpyro_transforms.md`.
2. A deterministic mock/stub interpolator implementing:
   - `icdf`, `cdf`, `dxdu`, `log_abs_dxdu`.
3. NumPyro transform conventions from existing code style.
4. Agent 08 contract assumptions:
   - interior interpolation is `interpax`-based and JAX-native,
   - C1 stitch slopes are sourced from interior interpolator boundary gradients when enabled.

## Outputs (Contract for This Agent)
1. Test file: `tests/test_mixture_transforms.py`.
2. Minimal reusable mock interpolator fixture in test file or `conftest.py`.

## Test Design Requirements

### 1) Transform API contract tests
For each class:
1. `__call__`, `_inverse`, `log_abs_det_jacobian` exist and run.
2. `domain` and `codomain` are set.
3. `tree_flatten/tree_unflatten` roundtrip preserves behavior.

### 2) Forward/inverse consistency tests
1. Uniform transform:
   - `_inverse(__call__(u)) ~ u` across random and grid `u`.
2. Normal transform:
   - `_inverse(__call__(z)) ~ z` across representative `z`.

### 3) Jacobian correctness tests
1. Uniform:
   - Compare reported `log_abs_det_jacobian` against finite-difference estimate of `dy/du`.
2. Normal:
   - Validate chain-rule form:
     - `log|dy/dz| = log|dx/du| + log phi(z)`.
3. Ensure finite outputs near boundaries and tail inputs.

### 4) Boundary and clipping tests
1. Uniform inputs at exact 0 and 1 are handled according to clipping policy.
2. Near-boundary values do not produce NaN/Inf.
3. Extreme normal inputs (`|z| > 8`) stay numerically stable.
4. Wrapper behavior remains consistent with interpolator stitch semantics (no extra wrapper-level tail switching that changes boundary behavior).

### 5) Batch/JAX compatibility tests
1. Batched forward and inverse calls preserve shape.
2. `jit` compile both transforms.
3. `vmap` over batches works.
4. Runtime path stays JAX-native (no NumPy host fallback such as `np.asarray` coercion in forward/inverse/Jacobian code paths).

### 6) Serialization and determinism tests
1. Pytree flatten/unflatten reproduces outputs exactly or within tolerance.
2. Optional: pickle/serialization behavior if project supports it.

## Required Tolerances
1. Forward/inverse roundtrip:
   - uniform `< 1e-5` interior, `< 1e-4` tails.
   - normal `< 2e-5` interior, `< 2e-4` tails.
2. Jacobian finite-difference agreement:
   - relative error `< 5e-3` interior.

## Structure and Naming
Recommended names:
1. `test_uniform_transform_roundtrip`
2. `test_normal_transform_roundtrip`
3. `test_uniform_log_det_matches_finite_difference`
4. `test_normal_log_det_matches_chain_rule`
5. `test_boundary_inputs_are_stable`
6. `test_jit_and_vmap_compatibility`
7. `test_tree_flatten_unflatten_roundtrip`

## Acceptance Criteria
1. Tests lock down transform math and API conventions.
2. Failure messages identify whether issue is mapping, Jacobian, or serialization.
3. Tests are deterministic and independent of random global state.

## Handoff to Task 09 Agent
Provide:
1. Completed test suite with mock interpolator fixture.
2. Explicit Jacobian formulas encoded in assertions.
3. Clear pass/fail thresholds for production implementation.

