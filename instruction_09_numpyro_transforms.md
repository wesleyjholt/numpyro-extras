# Agent Task 09: Quantile Interpolator -> NumPyro Transform Classes

## Objective
Implement NumPyro-compatible transform classes that expose:
1. `Uniform(0,1) -> mixture` via interpolated inverse-CDF.
2. `Normal(0,1) -> mixture` via composition with Normal CDF/ICDF.

This agent turns the interpolation engine into production-facing bijectors.

## Precondition
Task 04 test specification must be completed first, and Task 09 implementation must satisfy those tests.

## Scope
- In scope:
  - New `Transform` subclasses.
  - Forward/inverse map logic.
  - Log-abs-det-Jacobian computation.
  - Domain/codomain constraints.
  - JAX pytree flatten/unflatten support.
- Out of scope:
  - Recomputing knots from raw mixture parameters.
  - Interpolation algorithm changes.

## Required Upstream Dependency
From Agent 08:
1. `QuantileInterpolator1D` exposing `icdf(u)`, `cdf(x)`, `dxdu(u)` or `log_abs_dxdu(u)`.
2. Interpolator contract assumptions to preserve (must not be reimplemented in transforms):
   - interior interpolation is `interpax`-based and JAX-native,
   - C1 stitch slopes come from interior interpolator boundary gradients when `enforce_c1_stitch=True`.

## Inputs (Contract)
1. `interpolator: QuantileInterpolator1D`
2. `transform_cfg`
   - `clip_u_eps: float` (default `1e-10`)
   - `validate_args: bool` (default `False`)
3. For Normal-based transform:
   - standard normal distribution object (use NumPyro `Normal(0,1)`).

## Outputs (Contract)
Implement and export:
1. `UniformToMixtureTransform(Transform)`
2. `NormalToMixtureTransform(Transform)`

Each class must provide:
1. `__call__(x)` forward map.
2. `_inverse(y)` inverse map.
3. `log_abs_det_jacobian(x, y, intermediates=None)`.
4. `domain` and `codomain`.
5. `tree_flatten` / `tree_unflatten`.

## File Targets
- Primary edit:
  - `src/numpyro_extras/transforms.py`
- Optional exports:
  - package `__init__` file if needed.
- Tests:
  - `tests/test_mixture_transforms.py`

## Implementation Requirements

### 1) Uniform base transform
Define:
- Forward: `y = icdf(u)` where `u = clip(x, eps, 1-eps)`
- Inverse: `u = cdf(y)`

Jacobian:
- `log|dy/du| = log_abs_dxdu(u)` from interpolator
- Use `u` corresponding to forward input `x` (clipped consistently).

### 2) Normal base transform
Define:
- `u = Phi(z)` where `Phi` is standard normal CDF.
- Forward: `y = icdf(u)`.
- Inverse: `z = Phi^{-1}(cdf(y))`.

Jacobian (forward `z -> y`):
- `dy/dz = (dx/du) * (du/dz)`
- `log|dy/dz| = log|dx/du| + log phi(z)`
  - where `phi` is standard normal PDF.
- Compute using interpolator + `Normal(0,1).log_prob(z)`.

### 3) Constraints and typing
- Uniform transform:
  - `domain = unit_interval`
  - `codomain = real` (or mixture support if representable; for first cut use real if tails cover full real line).
- Normal transform:
  - `domain = real`
  - `codomain = real` (or support where appropriate).

### 4) Pytree serialization
- Preserve parameters needed to reconstruct transform.
- Keep tree metadata minimal and deterministic.

### 5) Vectorization and compilation
- Support scalar and batched inputs.
- JIT-safe with no Python branching on traced values.
- Do not introduce NumPy host fallbacks (`np.asarray`-style runtime coercion) in transform forward/inverse/Jacobian paths.

## Edge Cases to Handle
1. Input values at/near boundaries (`u=0`, `u=1`).
2. Very extreme normal inputs (`|z| > 8`).
3. Approximation-induced tiny negative derivatives (clip/floor before `log`).
4. Shape broadcasting for batch/event dims.
5. Preserve boundary behavior implied by Agent 08 C1 stitching; transform wrappers must not add alternate tail-switch logic that conflicts with interpolator stitch semantics.

## Validation and Acceptance Tests
In `tests/test_mixture_transforms.py`:

1. **Uniform roundtrip**
   - `_inverse(__call__(u)) ~ u` for random `u`.
2. **Normal roundtrip**
   - `_inverse(__call__(z)) ~ z` for random `z`.
3. **Jacobian finiteness**
   - no NaN/Inf over representative range.
4. **Numerical consistency**
   - finite-difference check of Jacobian vs reported `log_abs_det_jacobian`.
5. **JIT/vmap compatibility**
   - compile and batch-call both transforms.

## Deliverables Checklist
- [ ] Two new transform classes implemented.
- [ ] Correct Jacobian formulas for both bases.
- [ ] Pytree methods and constraints set.
- [ ] Unit tests passing.

## Handoff to Next Agent
Provide:
1. Ready-to-use transform classes.
2. Any caveats around codomain/support assumptions.
3. Error tolerance bounds from tests.

Integration agent should only compose these classes with builder APIs and end-to-end tests.
