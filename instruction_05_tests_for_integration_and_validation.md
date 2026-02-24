# Agent Task 05: Test Specification for Task 10 (Integration + Factory Validation)

## Objective
Write end-to-end and integration test instructions for `instruction_10_integration_and_validation.md` prior to implementation.

This task is test-first and produces integration/acceptance test definitions only.

## Target Under Test
Expected module: `src/numpyro_extras/mixture_transform_builder.py`  
Expected entrypoints:
1. `build_mixture_transform(...)`
2. Optional convenience builders for uniform/normal base.

## Inputs (Contract for This Agent)
1. `instruction_10_integration_and_validation.md`.
2. Availability of component modules from Tasks 06-09 (or mocks until complete).
3. Representative mixture configurations for smoke and stress tests.
4. Cross-task contract from Agent 08:
   - interpolator interior uses `interpax` in a JAX-native runtime path,
   - C1 stitching uses interior boundary gradients when enabled.

## Outputs (Contract for This Agent)
1. Test file: `tests/test_mixture_transform_builder.py`.
2. Optional benchmark-smoke test file:
   - `tests/test_mixture_transform_performance_smoke.py`.
3. Shared data/fixture helpers for common mixture setups.

## Test Design Requirements

### 1) Factory API contract tests
1. Builder accepts supported input modes:
   - mixture distribution object.
   - explicit weights + component params.
2. Builder returns correct transform type for `base="uniform"` and `base="normal"`.
3. Returned object includes expected metadata/diagnostics handles.

### 2) End-to-end functional tests
1. Build transform using default config and run forward/inverse roundtrip.
2. Confirm no NaN/Inf in:
   - forward outputs,
   - inverse outputs,
   - log-det Jacobians.
3. Validate outputs for multiple mixture shapes.

### 3) Distributional sanity tests
1. Compare quantiles from transform-generated samples vs direct mixture samples.
2. Compare selected summary stats:
   - median and central quantiles (required),
   - mean/variance where stable (optional).
3. Assert approximation errors below defined tolerances.

### 4) Config propagation tests
1. Custom nested configs are honored by downstream steps.
2. Invalid config values fail fast with useful error text.
3. Default config object is deterministic and reproducible.
4. Tail/interpolation diagnostics encode stitch provenance (for example, whether C1 stitching is enabled and boundary slope source).

### 5) Failure-path tests
1. Invalid weights and bad parameter shapes are rejected.
2. Bracket failure or non-monotone knot errors propagate meaningful exceptions.
3. Diagnostic metadata includes failure reason/context when available.

### 6) Compatibility tests
1. Builder outputs are JIT-compatible where expected.
2. Batch inputs through built transforms preserve shape.
3. Smoke test both `uniform` and `normal` bases on same mixture.
4. E2E runtime path remains JAX-native without NumPy host-fallback in interpolation/transform execution.

### 7) Performance smoke tests (non-gating)
1. Track build time for default config.
2. Track forward pass throughput for batched inputs.
3. Assert only coarse upper bounds to catch regressions, not micro-optimizations.

## Required Tolerances
1. End-to-end roundtrip:
   - default target `< 2e-4` interior.
2. Quantile agreement against direct sampling:
   - central quantiles absolute error `< 2e-2` (sampling-noise aware).
3. No NaN/Inf anywhere in tested pipelines.

## Structure and Naming
Recommended names:
1. `test_builder_returns_expected_transform_type`
2. `test_builder_default_roundtrip_uniform_base`
3. `test_builder_default_roundtrip_normal_base`
4. `test_distributional_sanity_against_direct_mixture_sampling`
5. `test_custom_config_propagates_through_pipeline`
6. `test_invalid_inputs_fail_fast_with_context`
7. `test_builder_performance_smoke`

## Acceptance Criteria
1. Tests pin all integration contracts for Task 10.
2. Test failures clearly indicate stage of pipeline failure.
3. Smoke performance checks are stable across local runs.

## Handoff to Task 10 Agent
Provide:
1. Final integration test suite and fixtures.
2. Config and diagnostics expectations encoded in assertions.
3. Explicit acceptance gates for release readiness.

