# Canonical Contracts

This document is the single source of truth for cross-task behavior contracts in the interpolation/transform pipeline.

## Scope

These contracts apply to:

- `instruction_07_knot_generation.md`
- `instruction_08_quantile_interpolator.md`
- `instruction_09_numpyro_transforms.md`
- `instruction_10_integration_and_validation.md`
- Related test-spec files (`instruction_02` through `instruction_05`)

## Interpolator Runtime Contract

1. Interior interpolation must use `interpax`.
2. Runtime interpolation and transform paths must be JAX-native:
   - must be compatible with `jit` and `vmap`,
   - must avoid NumPy host-fallback runtime coercions (for example `np.asarray` in execution paths).
3. Bidirectional maps are required:
   - `x(u)` (ICDF approximation),
   - `u(x)` (CDF approximation),
   - plus derivative helpers used for Jacobians.

## C1 Stitching Contract

When `enforce_c1_stitch=True`:

1. Left and right tail stitch slopes (`m0`, `mN`) must be computed from the interior interpolator boundary gradients.
2. Adjacent-knot finite differences are not a valid source for canonical C1 stitch slopes.

When `enforce_c1_stitch=False`:

1. External endpoint slopes (for example from knot metadata) may be used as fallback slope seeds.
2. Fallback slopes must be finite, positive, and guarded by configured floors.

## Knot Generation Contract

1. Knot generation provides `u_knots`, `x_knots`, and optional endpoint slope seeds.
2. Endpoint slope values from knot generation are not canonical C1 stitch slopes.
3. Any slope-estimation language in downstream docs/tests must preserve this distinction.

## Wrapper/Builder Contract

1. Transform wrappers must not reimplement or override interpolator stitch semantics.
2. Builder/integration layers must preserve the interpolator's C1 behavior end-to-end.
3. Diagnostics should expose stitch provenance where possible:
   - whether C1 stitching is enabled,
   - what boundary slope source is used.

## Test Enforcement Contract

Test specs should include checks for:

1. C1 continuity with boundary slopes sourced from interior interpolator gradients.
2. Regression protection against finite-difference-knot-slope substitution for C1 mode.
3. JAX execution-path compatibility (`jit`/`vmap`) without NumPy host fallback.
4. Stability and finiteness in tails and near boundaries.
