# Agent Task 07: Inverse-CDF -> Interpolation Knot Generator

## Objective
Generate high-quality interpolation knots `(u_i, x_i)` from the root-based inverse-CDF backend, optionally including endpoint slope seeds for non-C1 fallback tail stitching.

This agent converts expensive root solves into reusable tabulated geometry.

## Precondition
Task 02 test specification must be completed first, and Task 07 implementation must satisfy those tests.

## Scope
- In scope:
  - Grid design in probability space `u`.
  - Quantile evaluation `x_i = Q(u_i)`.
  - Knot post-processing (strict ordering, finite-value filtering).
  - Optional endpoint slope seeds for non-C1 fallback modes.
- Out of scope:
  - Interpolator implementation.
  - NumPyro transform classes.
  - New root-finding methods.

## Required Upstream Dependency
From Agent 06:
1. `icdf(u)` callable.
2. `cdf(x)` callable for optional consistency checks.

## Inputs (Contract)
1. `quantile_backend`
   - Object exposing `icdf(u)` and ideally diagnostics.
2. `knot_cfg`
   - `num_knots: int` (default `256`)
   - `u_min: float` (default `1e-6`)
   - `u_max: float` (default `1 - 1e-6`)
   - `grid_type: Literal["uniform_u", "logit_u", "hybrid"]` (default `"logit_u"`)
   - `tail_density: float` (default `0.35`, used by hybrid strategy)
   - `min_delta_u: float` (default `1e-10`)
3. `slope_cfg`
   - `method: Literal["finite_diff_u", "autodiff_on_interp_seed"]` (default `"finite_diff_u"`)
   - `delta_u: float` (default `5e-5`)
   - `min_positive_slope: float` (default `1e-8`)

## Outputs (Contract)
Return an immutable dataclass `QuantileKnotSet` with:

1. `u_knots: Array[N]` strictly increasing.
2. `x_knots: Array[N]` nondecreasing and finite.
3. `du_dx_left: float` and `du_dx_right: float`
   - optional endpoint slope seeds for CDF-space tails when `enforce_c1_stitch=False`.
   - C1 stitching in Agent 08 must not depend on finite-difference knot slopes.
4. `meta: dict`
   - grid strategy, `N`, clip epsilon, and quality metrics.

## File Targets
- Add module:
  - `src/numpyro_extras/mixture_knots.py`
- Add tests:
  - `tests/test_mixture_knots.py`

## Implementation Requirements

### 1) Grid construction
Implement three modes:

1. `uniform_u`
   - linearly spaced `u` on `[u_min, u_max]`.
2. `logit_u`
   - uniform spacing in logit-space then inverse-logit to `u`.
3. `hybrid`
   - blend interior uniform points with tail-enriched logit points.

Defaults should favor better tail fidelity (`logit_u`).

### 2) Quantile sampling
- Evaluate `x_i = icdf(u_i)` with vectorized calls.
- Remove any non-finite points; if removals are non-trivial, report in `meta`.
- Enforce strict `u` increase using `min_delta_u`.
- Keep the workflow JAX-native and jit/vmap compatible; avoid mandatory NumPy host conversions in the core path.

### 3) Monotonic cleanup
- If numerical noise causes non-monotone `x_knots`, repair minimally:
  - either isotonic projection (preferred) or monotone mask filtering.
- Record cleanup strategy and count in `meta`.

### 4) Endpoint slope estimation
Need optional `du/dx` seeds at left and right knots for non-C1 tail fallback modes.

- Important:
  - Agent 08 C1 stitching must compute endpoint slope from the interior interpolator gradient at the stitch boundary.
  - Finite-difference knot slopes must not be treated as the canonical C1 slope.

- Acceptable estimation for optional seeds:
  - robust local finite-difference proxy in `u(x)` space using nearest knot pairs.
  - left seed: `(u1-u0)/(x1-x0)`
  - right seed: `(uN-uN-1)/(xN-xN-1)`
- Stabilize:
  - floor slopes at `min_positive_slope`.
  - guard against division by tiny `dx`.

### 5) Quality metrics
Include in `meta`:
1. monotonicity violations before cleanup.
2. non-finite count.
3. min/max `dx` and `du`.
4. optional roundtrip sample errors using `cdf(x_i)`.

## Edge Cases to Handle
1. Flat regions from extreme weight imbalance.
2. Very tight components causing large local gradients.
3. Nearly duplicate `x_i` at tail points.
4. Small `N` values (`N < 16`) with warning in metadata.

## Validation and Acceptance Tests
In `tests/test_mixture_knots.py`:

1. **Strict ordering**
   - `jnp.all(jnp.diff(u_knots) > 0)`
2. **Monotonic quantile**
   - `jnp.all(jnp.diff(x_knots) >= 0)`
3. **Finite slopes**
   - slopes finite and strictly positive.
4. **Tail coverage**
   - knots include near-`u_min` and near-`u_max`.
5. **Quality metadata present**
   - expected keys exist and values sane.

## Deliverables Checklist
- [ ] `QuantileKnotSet` dataclass and constructor utility.
- [ ] Deterministic knot-generation pipeline.
- [ ] Optional endpoint slope-seed estimator for non-C1 fallback.
- [ ] Tests covering all grid modes.

## Handoff to Next Agent
Provide:
1. `QuantileKnotSet` with validated knots and slopes.
2. Recommended default `num_knots` and grid mode from experiments.
3. Any metadata flags that should trigger warnings downstream.

The next agent should consume only this struct and not call root solving directly.
