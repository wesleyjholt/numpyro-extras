# Agent Task 10: Integration, Factory API, and End-to-End Validation

## Objective
Integrate all prior components into a single user-facing pipeline that builds mixture transforms from mixture parameters/distributions, and validate behavior end-to-end with accuracy and stability tests.

This agent owns assembly quality, configuration ergonomics, and regression protection.

## Precondition
Task 05 test specification must be completed first, and Task 10 implementation must satisfy those tests.

## Scope
- In scope:
  - Factory API to build transforms.
  - Config dataclasses and default presets.
  - End-to-end tests and smoke benchmarks.
  - Documentation of assumptions and limitations.
- Out of scope:
  - Rewriting core algorithms from earlier agents (except minor bugfixes).

## Required Upstream Dependencies
From Agent 06:
1. Root-based mixture quantile backend.

From Agent 07:
2. Knot generation (`QuantileKnotSet`).

From Agent 08:
3. `QuantileInterpolator1D`.
   - must satisfy the Agent 08 contract: `interpax`-based interior interpolation, JAX-native runtime path, and C1 stitch slope from interior boundary gradient when enabled.

From Agent 09:
4. `UniformToMixtureTransform`, `NormalToMixtureTransform`.

## Inputs (Contract)
Factory inputs should support either:
1. A fully-formed mixture distribution object, or
2. Explicit mixture params (`weights`, component params, component family class).

Factory config:
1. `base: Literal["uniform", "normal"]`
2. `solver_cfg` (forwarded to Agent 06)
3. `knot_cfg` (forwarded to Agent 07)
4. `interp_cfg` and `tail_cfg` (forwarded to Agent 08)
5. `transform_cfg` (forwarded to Agent 09)

## Outputs (Contract)
Expose public entrypoints:
1. `build_mixture_transform(...) -> Transform`
2. Optional convenience:
   - `build_uniform_to_mixture_transform(...)`
   - `build_normal_to_mixture_transform(...)`

Return object should include accessible references (or metadata) for:
1. Chosen configs.
2. Knot diagnostics.
3. Approximation summary metrics.

## File Targets
- Add integration module:
  - `src/numpyro_extras/mixture_transform_builder.py`
- Update exports:
  - `src/numpyro_extras/__init__.py` as needed.
- Add integration tests:
  - `tests/test_mixture_transform_builder.py`
- Add docs section:
  - `README.md` usage snippet (or dedicated docs file if repo has docs structure).

## Implementation Requirements

### 1) Factory assembly pipeline
Implement exact sequence:
1. Build quantile backend (Agent 06).
2. Generate knots and optional endpoint slope seeds (Agent 07).
3. Construct interpolator with sigmoid tails (Agent 08).
4. Wrap in requested transform class (Agent 09).
5. Preserve Agent 08 stitch semantics end-to-end; do not override C1 slope source in builder/wrapper layers.

### 2) Config model
- Define explicit dataclasses:
  - `MixtureQuantileConfig`
  - `KnotGenerationConfig`
  - `InterpolatorConfig`
  - `TailConfig`
  - `TransformConfig`
  - `MixtureTransformBuildConfig` (composed top-level)
- Provide robust defaults with documented rationale.

### 3) Error handling and diagnostics
- Validate user inputs early:
  - weight normalization, parameter shapes, finite values.
- Emit informative errors on:
  - failed bracket search,
  - invalid knot set,
  - non-monotone interpolator.
- Surface diagnostics structure in returned object or attached metadata.
- Include interpolation diagnostics indicating stitch configuration/provenance (for example whether C1 stitching is enabled and boundary slope source).

### 4) End-to-end quality checks (automated tests)
Must include:
1. **Distributional sanity**
   - Compare quantiles from transform-induced samples vs direct mixture samples.
2. **Roundtrip**
   - Base-space -> target-space -> base-space small error.
3. **Tail behavior**
   - Extreme percentiles remain finite and monotone.
4. **Jacobian consistency**
   - Numerical finite-difference spot checks.
5. **JAX execution path**
   - `jit`/`vmap` e2e checks for build and forward/inverse evaluations, with no NumPy host-fallback requirement in the runtime interpolation/transform path.

### 5) Performance smoke checks
- Add lightweight benchmark-like tests (not strict perf gates) recording:
  - build time estimate,
  - forward throughput for batched calls.
- Ensure no pathological slowdown with default `num_knots`.

## Edge Cases to Handle
1. Highly separated component means.
2. Very small component scales.
3. Dominant single component with tiny secondary weights.
4. Non-default mixed batch shapes.

## Acceptance Criteria
All must pass:
1. Integration tests green.
2. No NaN/Inf in forward/inverse/Jacobian for tested ranges.
3. Default factory builds both uniform-based and normal-based transforms.
4. README example runs with minimal setup.

## Deliverables Checklist
- [ ] `mixture_transform_builder.py` with public factory functions.
- [ ] Config dataclasses and defaults.
- [ ] End-to-end tests plus smoke performance checks.
- [ ] User-facing usage documentation.

## Final Handoff Artifact
Provide a concise integration report containing:
1. Final public API signatures.
2. Default config values.
3. Measured approximation error summary.
4. Known limitations and next-step recommendations.
